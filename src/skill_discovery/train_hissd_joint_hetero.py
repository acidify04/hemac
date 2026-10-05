#!/usr/bin/env python3
"""Train ONE joint heterogeneous HiSSD baseline with NO common-skill adapter.

This is the ablation matched to train_hissd_joint_hetero_adapter.py:
- same synchronized D3O1 / D4O2 joint windows
- same role-specific observation encoders and action decoders
- same shared CommonSkillEncoder / task-skill / value / planner stack
- same equal source balancing
- same 0.5 Observer + 0.5 Drone action loss
- NO A([c_i,h_i]) module; the raw HiSSD common skill c_i is used directly

For strict comparison with the already-trained separate BC checkpoints, outcome
balanced sampling defaults to True here. Run the adapter trainer with
`--balanced-sampling` as well if these models are compared in one ablation table.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import torch

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from skill_discovery import train_hissd_hetero_baseline as base
from skill_discovery import train_hissd_multisource as multi
from skill_discovery import train_hissd_joint_hetero_adapter as joint_utils
from skill_discovery.hissd_joint_hetero_models import JointHeterogeneousHiSSD
from skill_discovery.multi_source_hissd import SourceSpec, balanced_source_batches

ROLE_ORDER = ("observer", "drone")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest-31", type=Path, required=True)
    p.add_argument("--data-root-31", type=Path, required=True)
    p.add_argument("--manifest-42", type=Path, required=True)
    p.add_argument("--data-root-42", type=Path, required=True)
    p.add_argument("--drone-bc-checkpoint", type=Path, required=True)
    p.add_argument("--observer-bc-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)

    p.add_argument("--epochs", type=int, default=70)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--sequence-length", type=int, default=128)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--dataset-cache-size", type=int, default=16)
    p.add_argument("--batches-per-source", type=int, default=0)
    p.add_argument(
        "--balanced-sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Outcome-balanced source sampling. Default=True to match current separate BC.",
    )

    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--task-learning-rate-multiplier", type=float)
    p.add_argument("--weight-decay", type=float)
    p.add_argument("--task-weight-decay", type=float)
    p.add_argument("--task-warmup-epochs", type=int)
    p.add_argument("--task-contrastive-weight", type=float)
    p.add_argument("--task-action-contrastive-weight", type=float)
    p.add_argument("--early-stopping-patience", type=int, default=0)
    p.add_argument("--max-train-batches", type=int)
    p.add_argument("--max-val-batches", type=int)
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def configure_training_args(cli: argparse.Namespace) -> argparse.Namespace:
    args = multi.base_default_args("drone")
    args.role = "joint"
    args.output_dir = cli.output_dir
    args.epochs = cli.epochs
    args.batch_size = cli.batch_size
    args.sequence_length = cli.sequence_length
    args.stride = cli.stride
    args.num_workers = cli.num_workers
    args.dataset_cache_size = cli.dataset_cache_size
    args.seed = cli.seed
    args.device = cli.device
    args.no_progress = cli.no_progress
    args.max_train_batches = cli.max_train_batches
    args.max_val_batches = cli.max_val_batches

    for name in (
        "learning_rate",
        "task_learning_rate_multiplier",
        "weight_decay",
        "task_weight_decay",
        "task_warmup_epochs",
        "task_contrastive_weight",
        "task_action_contrastive_weight",
    ):
        value = getattr(cli, name)
        if value is not None:
            setattr(args, name, value)

    args.early_stopping_patience = int(cli.early_stopping_patience)
    if hasattr(base, "configure_ablation"):
        base.configure_ablation(args)
    if args.skill_structure != "split":
        raise ValueError(
            "Joint heterogeneous HiSSD baseline requires skill_structure='split', "
            f"got {args.skill_structure!r}."
        )
    return args


def joint_controller_objective(
    model: JointHeterogeneousHiSSD,
    batch: dict[str, Any],
    args: argparse.Namespace,
    *,
    task_only: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    features, valid_mask, role_slices = model.encode_joint_observations(
        batch["observations_by_role"],
        batch["valid_agents_by_role"],
    )

    skill_outputs = model.infer_joint_skills(
        features,
        valid_mask,
        task_observation_features=features.detach(),
    )
    common = skill_outputs["common_skills"]
    task_skill = skill_outputs["task_skills"]
    query = skill_outputs["contrastive_skills"]

    decoder_task_skill = (
        task_skill.detach() if args.detach_task_skill_for_action else task_skill
    )
    predictions = model.decode_joint_actions(
        features,
        common,
        decoder_task_skill,
        role_slices,
    )

    role_action_losses = {
        role: base.masked_action_mse(
            predictions[role],
            batch["actions_by_role"][role],
            batch["valid_agents_by_role"][role],
        )
        for role in ROLE_ORDER
    }
    action_loss = 0.5 * (
        role_action_losses["observer"] + role_action_losses["drone"]
    )

    descriptor_loss, metric_loss, variance_loss, descriptor_metrics = (
        base.continuous_task_objective(model, query, batch, args)
    )
    contrastive_loss, contrastive_metrics = base.task_contrastive_objective(
        model, query, batch, args
    )
    (
        task_action_descriptor_loss,
        task_action_variance_loss,
        task_action_metrics,
    ) = base.task_action_skill_objective(model, task_skill, batch, args)
    (
        task_action_contrastive_loss,
        raw_task_action_contrastive_metrics,
    ) = base.task_contrastive_objective(model, task_skill, batch, args)
    task_action_contrastive_metrics = {
        name.replace("task_contrastive", "task_action_contrastive", 1): value
        for name, value in raw_task_action_contrastive_metrics.items()
    }

    task_objective = (
        args.descriptor_weight * descriptor_loss
        + args.descriptor_metric_weight * metric_loss
        + args.task_variance_weight * variance_loss
        + args.task_contrastive_weight * contrastive_loss
        + args.task_action_descriptor_weight * task_action_descriptor_loss
        + args.task_action_variance_weight * task_action_variance_loss
        + args.task_action_contrastive_weight * task_action_contrastive_loss
    )
    total = task_objective if task_only else action_loss + task_objective

    metrics = {
        "controller_loss": float(total.detach()),
        "action_mse": float(action_loss.detach()),
        "observer_action_mse": float(role_action_losses["observer"].detach()),
        "drone_action_mse": float(role_action_losses["drone"].detach()),
        "weighted_descriptor_loss": float(
            (args.descriptor_weight * descriptor_loss).detach()
        ),
        "weighted_descriptor_metric_loss": float(
            (args.descriptor_metric_weight * metric_loss).detach()
        ),
        "weighted_task_variance_loss": float(
            (args.task_variance_weight * variance_loss).detach()
        ),
        "weighted_task_contrastive_loss": float(
            (args.task_contrastive_weight * contrastive_loss).detach()
        ),
        "weighted_task_action_descriptor_loss": float(
            (args.task_action_descriptor_weight * task_action_descriptor_loss).detach()
        ),
        "weighted_task_action_variance_loss": float(
            (args.task_action_variance_weight * task_action_variance_loss).detach()
        ),
        "weighted_task_action_contrastive_loss": float(
            (
                args.task_action_contrastive_weight
                * task_action_contrastive_loss
            ).detach()
        ),
        "common_skill_std": float(
            base.skill_standard_deviation(common, valid_mask).detach()
        ),
        "task_skill_std": float(
            base.skill_standard_deviation(task_skill, valid_mask).detach()
        ),
        **descriptor_metrics,
        **contrastive_metrics,
        **task_action_metrics,
        **task_action_contrastive_metrics,
    }
    return total, metrics


def joint_planner_objective(
    model: JointHeterogeneousHiSSD,
    batch: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Same planner objective as adapter model, but using raw common skill c_i."""
    with torch.no_grad():
        features, valid_mask, role_slices = model.encode_joint_observations(
            batch["observations_by_role"],
            batch["valid_agents_by_role"],
        )
        features = features.detach()
        target_next_features, _, _ = model.encode_joint_observations(
            batch["next_observations_by_role"],
            batch["valid_agents_by_role"],
            target=True,
        )

    planning_skill = model.common_skill_encoder(features, valid_mask)
    predicted_central, predicted_local = model.forward_predictor(
        planning_skill,
        valid_mask.to(features.dtype),
    )

    with torch.no_grad():
        current_value = model.total_value(
            features,
            batch["central_map"],
            valid_mask.to(features.dtype),
        ).squeeze(-1)
        predicted_next_value = model.total_value(
            predicted_local.detach(),
            predicted_central.detach(),
            valid_mask.to(features.dtype),
            target=True,
        ).squeeze(-1)
        scaled_reward = batch["team_reward"] / args.reward_scale
        residual = (
            scaled_reward
            + args.gamma * (~batch["done"]).float() * predicted_next_value
            - current_value
        )
        advantage_weight = torch.exp(residual / args.alpha).clamp(max=100.0)

    central_error = (
        predicted_central - batch["next_central_map"]
    ).square().mean(dim=(2, 3, 4))

    local_error, local_error_by_role = (
        joint_utils._role_balanced_local_prediction_error(
            predicted_local,
            target_next_features.detach(),
            valid_mask,
            role_slices,
        )
    )

    prediction_error = central_error + local_error
    step_mask = batch["valid_steps"].to(prediction_error.dtype)
    denominator = step_mask.sum().clamp_min(1.0)
    loss = (
        prediction_error
        * advantage_weight.detach()
        * step_mask
    ).sum() / denominator

    return loss, {
        "planner_loss": float(loss.detach()),
        "central_prediction_mse": float(
            (central_error.detach() * step_mask).sum() / denominator
        ),
        "local_prediction_mse": float(
            (local_error.detach() * step_mask).sum() / denominator
        ),
        "observer_local_prediction_mse": float(
            (local_error_by_role["observer"].detach() * step_mask).sum()
            / denominator
        ),
        "drone_local_prediction_mse": float(
            (local_error_by_role["drone"].detach() * step_mask).sum()
            / denominator
        ),
        "advantage_weight_mean": float(
            (advantage_weight.detach() * step_mask).sum() / denominator
        ),
        "advantage_weight_max": float(advantage_weight.detach().max()),
        "planner_predicted_value_mean": float(
            (predicted_next_value * step_mask).sum() / denominator
        ),
        "planner_td_residual_mean": float(
            (residual * step_mask).sum() / denominator
        ),
    }


def initialization_diagnostics(model, raw_batch, device):
    batch = joint_utils.prepare_joint_batch(raw_batch, device)
    outputs = model.forward_joint(
        batch["observations_by_role"],
        batch["valid_agents_by_role"],
    )
    features = outputs["observation_features"]
    role_slices = outputs["role_slices"]
    result = {}
    for role in ROLE_ORDER:
        role_features = features[..., role_slices[role], :]
        bc_action = torch.tanh(
            model.action_decoder[role].base_action_head(role_features)
        )
        result[f"{role}_bc_action_max_difference"] = float(
            (outputs["actions"][role] - bc_action).abs().max().item()
        )
    return result


def add_metrics(accumulator, metrics):
    for key, value in metrics.items():
        accumulator[key] += float(value)


def average_metrics(accumulator, count):
    if count <= 0:
        raise RuntimeError("No batches were processed.")
    return {key: float(value) / count for key, value in accumulator.items()}


def equal_source_average(by_source):
    keys = set.intersection(*(set(m) for m in by_source.values()))
    return {
        key: sum(m[key] for m in by_source.values()) / len(by_source)
        for key in keys
    }


def compact(metrics):
    def g(name, default=float("nan")):
        return float(metrics.get(name, default))
    return (
        f"action={g('action_mse'):.5f} "
        f"obs={g('observer_action_mse'):.5f} "
        f"drone={g('drone_action_mse'):.5f} "
        f"task={g('task_contrastive_loss'):.4f} "
        f"value={g('value_loss'):.4f} "
        f"planner={g('planner_loss'):.4f}"
    )


def train_epoch(
    model,
    loaders,
    optimizer,
    device,
    args,
    *,
    epoch,
    batches_per_source,
    task_only,
):
    model.train()
    total = defaultdict(float)
    per_source = {name: defaultdict(float) for name in loaders}
    counts = defaultdict(int)
    processed = 0

    iterator = balanced_source_batches(
        dict(loaders),
        seed=args.seed + epoch,
        batches_per_source=batches_per_source,
    )

    for source_name, raw_batch in iterator:
        if args.max_train_batches is not None and processed >= args.max_train_batches:
            break
        batch = joint_utils.prepare_joint_batch(raw_batch, device)

        controller_loss, controller_metrics = joint_controller_objective(
            model, batch, args, task_only=task_only
        )
        controller_metrics["controller_grad_norm"] = base.optimize(
            controller_loss, model, optimizer, args.grad_clip
        )

        if task_only:
            with torch.no_grad():
                _, value_metrics = joint_utils.joint_value_objective(
                    model, batch, args
                )
                _, planner_metrics = joint_planner_objective(model, batch, args)
            value_metrics["value_grad_norm"] = 0.0
            planner_metrics["planner_grad_norm"] = 0.0
            planner_metrics["planner_update_skipped"] = 0.0
        else:
            value_loss, value_metrics = joint_utils.joint_value_objective(
                model, batch, args
            )
            value_metrics["value_grad_norm"] = base.optimize(
                value_loss, model, optimizer, args.grad_clip
            )
            with base.strict_planner_math(device):
                planner_loss, planner_metrics = joint_planner_objective(
                    model, batch, args
                )
                planner_grad = base.optimize(
                    planner_loss,
                    model,
                    optimizer,
                    args.grad_clip,
                    skip_nonfinite=True,
                )
            planner_metrics["planner_update_skipped"] = float(
                not math.isfinite(planner_grad)
            )
            planner_metrics["planner_grad_norm"] = (
                planner_grad if math.isfinite(planner_grad) else 0.0
            )
            model.update_targets(args.target_tau)

        merged = {**controller_metrics, **value_metrics, **planner_metrics}
        add_metrics(total, merged)
        add_metrics(per_source[source_name], merged)
        counts[source_name] += 1
        processed += 1

    return (
        average_metrics(total, processed),
        {
            name: average_metrics(per_source[name], counts[name])
            for name in loaders
            if counts[name] > 0
        },
    )


@torch.inference_mode()
def validate_one_source(model, loader, device, args):
    model.eval()
    accumulator = defaultdict(float)
    count = 0
    for raw_batch in loader:
        if args.max_val_batches is not None and count >= args.max_val_batches:
            break
        batch = joint_utils.prepare_joint_batch(raw_batch, device)
        _, controller = joint_controller_objective(model, batch, args)
        _, value = joint_utils.joint_value_objective(model, batch, args)
        _, planner = joint_planner_objective(model, batch, args)
        add_metrics(accumulator, {**controller, **value, **planner})
        count += 1
    return average_metrics(accumulator, count)


@torch.inference_mode()
def validate_all_sources(model, loaders, device, args):
    by_source = {
        name: validate_one_source(model, loader, device, args)
        for name, loader in loaders.items()
    }
    return equal_source_average(by_source), by_source


def save_checkpoint(
    path,
    model,
    optimizer,
    *,
    epoch,
    args,
    cli,
    train_metrics,
    train_by_source,
    val_metrics,
    val_by_source,
    bc_reports,
    population_counts,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_type": "hemac_joint_heterogeneous_hissd",
            "model_config": model.config(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch),
            "skill_structure": model.skill_structure,
            "architecture": {
                "roles": list(ROLE_ORDER),
                "role_specific_observation_encoders": True,
                "role_specific_action_decoders": True,
                "shared_common_skill_encoder": True,
                "shared_task_skill_encoder": True,
                "adapter_count": 0,
                "role_embedding": False,
                "agent_id_embedding": False,
                "population_embedding": False,
                "planner_uses_raw_common_skill": True,
                "controller_uses_raw_common_skill": True,
                "role_balanced_action_loss": True,
                "role_balanced_planner_local_loss": True,
            },
            "bc_initialization": {
                role: {
                    "checkpoint": str(
                        cli.drone_bc_checkpoint
                        if role == "drone"
                        else cli.observer_bc_checkpoint
                    ),
                    "epoch": bc_reports[role].get("epoch"),
                    "metrics": bc_reports[role].get("metrics", {}),
                }
                for role in ROLE_ORDER
            },
            "training_metrics": dict(train_metrics),
            "training_metrics_by_source": {
                name: dict(metrics) for name, metrics in train_by_source.items()
            },
            "validation_metrics": dict(val_metrics),
            "validation_metrics_by_source": {
                name: dict(metrics) for name, metrics in val_by_source.items()
            },
            "multisource": {
                "configuration_balancing": "equal_batches_per_source",
                "outcome_balanced_sampling": bool(cli.balanced_sampling),
                "population_counts": {
                    name: dict(counts) for name, counts in population_counts.items()
                },
                "sources": {
                    "3-1": {
                        "manifest": str(cli.manifest_31),
                        "data_root": str(cli.data_root_31),
                    },
                    "4-2": {
                        "manifest": str(cli.manifest_42),
                        "data_root": str(cli.data_root_42),
                    },
                },
            },
            "training_args": vars(args),
        },
        path,
    )


def main() -> None:
    cli = parse_args()
    if cli.epochs <= 0 or cli.batch_size <= 0 or cli.sequence_length <= 0:
        raise ValueError("epochs/batch-size/sequence-length must be positive")
    if cli.batches_per_source < 0:
        raise ValueError("--batches-per-source cannot be negative")
    if cli.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience cannot be negative")

    args = configure_training_args(cli)
    multi.seed_everything(cli.seed)
    device = multi.resolve_device(cli.device)
    if hasattr(base, "configure_gpu_backend"):
        base.configure_gpu_backend(device)

    source_specs = {
        "3-1": SourceSpec("3-1", cli.manifest_31, cli.data_root_31),
        "4-2": SourceSpec("4-2", cli.manifest_42, cli.data_root_42),
    }
    tasks31, names31 = multi.manifest_info(cli.manifest_31)
    tasks42, names42 = multi.manifest_info(cli.manifest_42)
    if tasks31 != tasks42 or names31 != names42:
        raise ValueError("D3O1/D4O2 manifests use different task schemas")
    if not tasks31:
        raise ValueError("No source tasks found")

    pin_memory = device.type == "cuda"
    train_datasets, train_loaders = {}, {}
    val_datasets, val_loaders = {}, {}
    for index, (name, spec) in enumerate(source_specs.items()):
        train_datasets[name], train_loaders[name] = joint_utils.create_joint_dataloader(
            spec,
            "source_train",
            cli,
            seed=cli.seed + index * 1000,
            pin_memory=pin_memory,
        )
        val_datasets[name], val_loaders[name] = joint_utils.create_joint_dataloader(
            spec,
            "source_val",
            cli,
            seed=cli.seed + 10_000 + index * 1000,
            pin_memory=pin_memory,
        )

    sample31 = train_datasets["3-1"][0]
    central_shape = sample31["global_state"]["central_map"].shape

    args.task_descriptor_names = names31
    args.training_tasks = tasks31
    args.held_out_tasks = []
    descriptor_mean, descriptor_scale = multi.combined_descriptor_statistics(
        train_datasets, args.descriptor_scale_floor
    )
    args.task_descriptor_mean = descriptor_mean.tolist()
    args.task_descriptor_scale = descriptor_scale.tolist()

    model, bc_reports = JointHeterogeneousHiSSD.from_bc_checkpoints(
        drone_checkpoint=cli.drone_bc_checkpoint,
        observer_checkpoint=cli.observer_bc_checkpoint,
        central_map_channels=int(central_shape[-3]),
        central_map_size=tuple(int(x) for x in central_shape[-2:]),
        reference_observer_count=int(sample31["observer"]["actions"].shape[-2]),
        reference_drone_count=int(sample31["drone"]["actions"].shape[-2]),
        hidden_dim=args.hidden_dim,
        skill_dim=args.skill_dim,
        transformer_heads=args.transformer_heads,
        contrastive_from_action_skill=False,
        task_context_pooling=False,
        task_descriptor_dim=(
            int(sample31["task_descriptor"].numel()) if args.descriptor_enabled else 0
        ),
        task_prior_count=(len(tasks31) if args.task_contrastive_enabled else 0),
        task_dropout=args.task_dropout,
        task_feature_deltas=True,
        learned_task_classifier=args.task_contrastive_enabled,
        normalize_task_context=False,
        task_running_statistics=True,
        direct_task_summary=False,
        task_action_residual=False,
        skill_structure="split",
    )
    model = model.to(device)

    population_counts = joint_utils.validate_joint_source_compatibility(
        train_datasets, model
    )
    optimizer, base_params, task_params = multi.optimizer_for_model(model, args)

    initial_raw_batch = next(iter(train_loaders["3-1"]))
    init_diag = initialization_diagnostics(model, initial_raw_batch, device)
    for role in ROLE_ORDER:
        error = init_diag[f"{role}_bc_action_max_difference"]
        if error > 1e-6:
            raise RuntimeError(
                f"{role} policy does not preserve BC action at initialization: {error:.3e}"
            )

    cli.output_dir.mkdir(parents=True, exist_ok=True)
    batches_per_source = None if cli.batches_per_source <= 0 else cli.batches_per_source
    effective_bps = (
        max(len(loader) for loader in train_loaders.values())
        if batches_per_source is None
        else batches_per_source
    )

    print(f"device={device} source_populations={population_counts}", flush=True)
    print(
        f"outcome_balanced_sampling={cli.balanced_sampling} "
        f"balanced_batches_per_source={effective_bps}",
        flush=True,
    )
    print(
        f"base_params={sum(p.numel() for p in base_params):,} "
        f"task_params={sum(p.numel() for p in task_params):,} adapter_params=0",
        flush=True,
    )
    print(
        "BC initialization "
        f"drone_epoch={bc_reports['drone'].get('epoch')} "
        f"observer_epoch={bc_reports['observer'].get('epoch')} "
        f"drone_action_diff={init_diag['drone_bc_action_max_difference']:.3e} "
        f"observer_action_diff={init_diag['observer_bc_action_max_difference']:.3e}",
        flush=True,
    )
    print(
        "architecture=role_specific_obs_encoder + shared_common_skill_encoder + "
        "role_specific_action_decoder (NO adapter)",
        flush=True,
    )

    best_combined = math.inf
    best_task = math.inf
    best_task_epoch = 0
    no_improve = 0
    aux_enabled = bool(args.descriptor_enabled or args.task_contrastive_enabled)

    epoch_iterator = range(1, cli.epochs + 1)
    if tqdm is not None and not cli.no_progress:
        epoch_iterator = tqdm(
            epoch_iterator,
            total=cli.epochs,
            desc="joint heterogeneous HiSSD (no adapter)",
            unit="epoch",
            dynamic_ncols=True,
        )

    for epoch in epoch_iterator:
        task_only = epoch <= args.task_warmup_epochs
        train_metrics, train_by_source = train_epoch(
            model,
            train_loaders,
            optimizer,
            device,
            args,
            epoch=epoch,
            batches_per_source=batches_per_source,
            task_only=task_only,
        )
        val_metrics, val_by_source = validate_all_sources(
            model, val_loaders, device, args
        )
        combined_loss, task_loss = base.validation_losses(val_metrics, args)

        line = (
            f"epoch={epoch:03d} phase={'task_warmup' if task_only else 'joint'} "
            f"train[{compact(train_metrics)}] val[{compact(val_metrics)}] "
            + " ".join(
                f"{name}[{compact(metrics)}]"
                for name, metrics in val_by_source.items()
            )
        )
        if tqdm is not None and not cli.no_progress:
            tqdm.write(line)
            epoch_iterator.set_postfix(
                train=f"{train_metrics['action_mse']:.4f}",
                val=f"{val_metrics['action_mse']:.4f}",
            )
        else:
            print(line, flush=True)

        save_checkpoint(
            cli.output_dir / "hissd_joint_last.pt",
            model,
            optimizer,
            epoch=epoch,
            args=args,
            cli=cli,
            train_metrics=train_metrics,
            train_by_source=train_by_source,
            val_metrics=val_metrics,
            val_by_source=val_by_source,
            bc_reports=bc_reports,
            population_counts=population_counts,
        )

        if combined_loss < best_combined:
            best_combined = combined_loss
            save_checkpoint(
                cli.output_dir / "hissd_joint_best.pt",
                model,
                optimizer,
                epoch=epoch,
                args=args,
                cli=cli,
                train_metrics=train_metrics,
                train_by_source=train_by_source,
                val_metrics=val_metrics,
                val_by_source=val_by_source,
                bc_reports=bc_reports,
                population_counts=population_counts,
            )

        selection = task_loss if aux_enabled else combined_loss
        if selection < best_task:
            best_task = selection
            best_task_epoch = epoch
            no_improve = 0
            save_checkpoint(
                cli.output_dir / "hissd_joint_best_task.pt",
                model,
                optimizer,
                epoch=epoch,
                args=args,
                cli=cli,
                train_metrics=train_metrics,
                train_by_source=train_by_source,
                val_metrics=val_metrics,
                val_by_source=val_by_source,
                bc_reports=bc_reports,
                population_counts=population_counts,
            )
        else:
            no_improve += 1

        patience = int(args.early_stopping_patience)
        if patience > 0 and no_improve >= patience:
            print(
                f"Early stopping after {patience} non-improving epochs "
                f"(best_task_epoch={best_task_epoch}).",
                flush=True,
            )
            break

    print(
        f"done best_combined={best_combined:.6f} "
        f"best_task={best_task:.6f}@{best_task_epoch}",
        flush=True,
    )


if __name__ == "__main__":
    main()
