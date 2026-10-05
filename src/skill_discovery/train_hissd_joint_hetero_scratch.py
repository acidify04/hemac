#!/usr/bin/env python3
"""Train ONE joint heterogeneous HiSSD FROM SCRATCH, without an adapter.

This is the initialization ablation for train_hissd_joint_hetero.py.  It keeps
the same synchronized D3O1/D4O2 data, role-balanced controller loss, shared
CommonSkillEncoder/task-skill/value/planner stack, hyperparameters, and model
architecture, but it does NOT load any Behavior Cloning checkpoint.

Role-specific observation encoders and action decoders are randomly initialized.
The raw HiSSD common skill c_i is used directly (no common-skill adapter).
Outcome-balanced sampling defaults to True to match the existing BC-initialized
Joint HiSSD run.
"""

from __future__ import annotations

import argparse
import json
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
from skill_discovery import evaluate_hissd_joint_hetero_zero_shot as rollout_eval
from skill_discovery.hissd_joint_hetero_models import JointHeterogeneousHiSSD
from skill_discovery.multi_source_hissd import SourceSpec, balanced_source_batches

ROLE_ORDER = ("observer", "drone")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest-31", type=Path, required=True)
    p.add_argument("--data-root-31", type=Path, required=True)
    p.add_argument("--manifest-42", type=Path, required=True)
    p.add_argument("--data-root-42", type=Path, required=True)
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
    p.add_argument(
        "--rollout-eval-every",
        type=int,
        default=5,
        help=(
            "Run source-only D3O1/D4O2 mission rollouts every N epochs. "
            "0 disables rollout diagnostics. These rollouts are diagnostic only "
            "and never affect checkpoint selection."
        ),
    )
    p.add_argument(
        "--rollout-eval-episodes",
        type=int,
        default=20,
        help="Episodes per source population for each rollout diagnostic.",
    )
    p.add_argument(
        "--rollout-eval-seed-base",
        type=int,
        default=200_000_000,
        help=(
            "Fixed source-rollout diagnostic seed base. Default differs from the "
            "final zero-shot evaluation seed base (100000000)."
        ),
    )
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
            "initialization": "scratch",
            "bc_initialization": None,
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



def role_config_from_sample(sample: Mapping[str, Any], role: str) -> dict[str, Any]:
    """Build the same role I/O architecture used by the existing BC-initialized run.

    The current multisource BC policies use two 96-unit hidden layers with ReLU.
    All observation/action dimensions are inferred from the synchronized dataset,
    so no BC checkpoint is consulted.
    """
    payload = sample[role]
    observations = payload["observations"]
    actions = payload["actions"]
    return {
        "global_map_channels": int(observations["global_map"].shape[-3]),
        "local_map_channels": int(observations["local_map"].shape[-3]),
        "global_map_size": tuple(int(x) for x in observations["global_map"].shape[-2:]),
        "local_map_size": tuple(int(x) for x in observations["local_map"].shape[-2:]),
        "action_history_shape": tuple(int(x) for x in observations["action_history"].shape[-2:]),
        "hidden_sizes": (96, 96),
        "activation": "relu",
        "action_dim": int(actions.shape[-1]),
    }


def validate_role_schemas_from_data(
    train_datasets: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Require D3O1 and D4O2 to expose the same per-role observation/action schema."""
    configs_by_source: dict[str, dict[str, dict[str, Any]]] = {}
    for source_name, dataset in train_datasets.items():
        sample = dataset[0]
        configs_by_source[source_name] = {
            role: role_config_from_sample(sample, role)
            for role in ROLE_ORDER
        }
    reference_name = next(iter(configs_by_source))
    reference = configs_by_source[reference_name]
    for source_name, config in configs_by_source.items():
        if config != reference:
            raise ValueError(
                "D3O1/D4O2 role observation/action schemas differ: "
                f"{reference_name}={reference}, {source_name}={config}"
            )
    return reference


def build_source_rollout_configs(cli: argparse.Namespace) -> dict[str, dict[str, Any]]:
    """Build canonical D1 source environments for training diagnostics only."""
    _, template31 = rollout_eval.load_env_template(cli.data_root_31)
    _, template42 = rollout_eval.load_env_template(cli.data_root_42)
    return {
        "D3O1": rollout_eval.build_target_config(
            template31, difficulty=1, n_drones=3, n_observers=1
        ),
        "D4O2": rollout_eval.build_target_config(
            template42, difficulty=1, n_drones=4, n_observers=2
        ),
    }


@torch.inference_mode()
def evaluate_source_rollouts(
    model: JointHeterogeneousHiSSD,
    rollout_configs: Mapping[str, Mapping[str, Any]],
    *,
    device: torch.device,
    episodes: int,
    seed_base: int,
) -> dict[str, dict[str, float]]:
    """Evaluate true environment mission success on source populations.

    Fixed seeds are reused at every diagnostic epoch so curves reflect policy
    changes rather than changing evaluation episodes.  This function is not
    used for checkpoint selection.
    """
    was_training = model.training
    model.eval()
    try:
        results: dict[str, dict[str, float]] = {}
        for source_name, config in rollout_configs.items():
            records = [
                rollout_eval.run_episode(
                    config,
                    model,
                    seed=int(seed_base) + 100_000 + episode_index,
                    difficulty=1,
                    device=device,
                )
                for episode_index in range(int(episodes))
            ]
            results[source_name] = rollout_eval.aggregate(records)
        return results
    finally:
        model.train(was_training)


def append_rollout_history(
    output_dir: Path,
    *,
    epoch: int,
    results: Mapping[str, Mapping[str, float]],
    episodes: int,
    seed_base: int,
) -> None:
    """Persist source-rollout diagnostics for later plotting."""
    payload = {
        "epoch": int(epoch),
        "episodes_per_source": int(episodes),
        "seed_base": int(seed_base),
        "seed_formula": "seed_base + 100000 + episode_index",
        "sources": {
            name: {key: float(value) for key, value in metrics.items()}
            for name, metrics in results.items()
        },
    }
    path = output_dir / "source_rollout_history.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def main() -> None:
    cli = parse_args()
    if cli.epochs <= 0 or cli.batch_size <= 0 or cli.sequence_length <= 0:
        raise ValueError("epochs/batch-size/sequence-length must be positive")
    if cli.batches_per_source < 0:
        raise ValueError("--batches-per-source cannot be negative")
    if cli.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience cannot be negative")
    if cli.rollout_eval_every < 0:
        raise ValueError("--rollout-eval-every cannot be negative")
    if cli.rollout_eval_episodes <= 0:
        raise ValueError("--rollout-eval-episodes must be positive")

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

    role_configs = validate_role_schemas_from_data(train_datasets)
    model = JointHeterogeneousHiSSD(
        role_configs=role_configs,
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
    ).to(device)

    population_counts = joint_utils.validate_joint_source_compatibility(
        train_datasets, model
    )
    optimizer, base_params, task_params = multi.optimizer_for_model(model, args)

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
        "initialization=scratch (NO BC checkpoint; all trainable model parameters random-init)",
        flush=True,
    )
    print(
        "architecture=role_specific_obs_encoder + shared_common_skill_encoder + "
        "role_specific_action_decoder (NO adapter)",
        flush=True,
    )

    rollout_configs = None
    if cli.rollout_eval_every > 0:
        rollout_configs = build_source_rollout_configs(cli)
        print(
            "source_rollout_diagnostics="
            f"every_{cli.rollout_eval_every}_epochs "
            f"episodes={cli.rollout_eval_episodes}/source "
            f"seed_base={cli.rollout_eval_seed_base} "
            "sources=D3O1,D4O2 checkpoint_selection=OFF",
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
            desc="joint heterogeneous HiSSD scratch (no adapter)",
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

        rollout_results = None
        should_rollout = (
            rollout_configs is not None
            and (epoch % cli.rollout_eval_every == 0 or epoch == cli.epochs)
        )
        if should_rollout:
            rollout_results = evaluate_source_rollouts(
                model,
                rollout_configs,
                device=device,
                episodes=cli.rollout_eval_episodes,
                seed_base=cli.rollout_eval_seed_base,
            )
            append_rollout_history(
                cli.output_dir,
                epoch=epoch,
                results=rollout_results,
                episodes=cli.rollout_eval_episodes,
                seed_base=cli.rollout_eval_seed_base,
            )

        line = (
            f"epoch={epoch:03d} phase={'task_warmup' if task_only else 'joint'} "
            f"train[{compact(train_metrics)}] val[{compact(val_metrics)}] "
            + " ".join(
                f"{name}[{compact(metrics)}]"
                for name, metrics in val_by_source.items()
            )
        )
        if rollout_results is not None:
            s31 = rollout_results["D3O1"]["success"]
            s42 = rollout_results["D4O2"]["success"]
            source_success = 0.5 * (s31 + s42)
            line += (
                f" rollout[D3O1_succ={s31:.3f} "
                f"D4O2_succ={s42:.3f} mean_succ={source_success:.3f} "
                f"D3O1_crash={rollout_results['D3O1']['fatal_crash']:.3f} "
                f"D4O2_crash={rollout_results['D4O2']['fatal_crash']:.3f}]"
            )

        if tqdm is not None and not cli.no_progress:
            tqdm.write(line)
            postfix = {
                "train": f"{train_metrics['action_mse']:.4f}",
                "val": f"{val_metrics['action_mse']:.4f}",
            }
            if rollout_results is not None:
                postfix.update(
                    s31=f"{rollout_results['D3O1']['success']:.2f}",
                    s42=f"{rollout_results['D4O2']['success']:.2f}",
                    succ=f"{source_success:.2f}",
                )
            epoch_iterator.set_postfix(**postfix)
        else:
            print(line, flush=True)

        save_checkpoint(
            cli.output_dir / "hissd_joint_scratch_last.pt",
            model,
            optimizer,
            epoch=epoch,
            args=args,
            cli=cli,
            train_metrics=train_metrics,
            train_by_source=train_by_source,
            val_metrics=val_metrics,
            val_by_source=val_by_source,
            population_counts=population_counts,
        )

        if combined_loss < best_combined:
            best_combined = combined_loss
            save_checkpoint(
                cli.output_dir / "hissd_joint_scratch_best.pt",
                model,
                optimizer,
                epoch=epoch,
                args=args,
                cli=cli,
                train_metrics=train_metrics,
                train_by_source=train_by_source,
                val_metrics=val_metrics,
                val_by_source=val_by_source,
                population_counts=population_counts,
            )

        selection = task_loss if aux_enabled else combined_loss
        if selection < best_task:
            best_task = selection
            best_task_epoch = epoch
            no_improve = 0
            save_checkpoint(
                cli.output_dir / "hissd_joint_scratch_best_task.pt",
                model,
                optimizer,
                epoch=epoch,
                args=args,
                cli=cli,
                train_metrics=train_metrics,
                train_by_source=train_by_source,
                val_metrics=val_metrics,
                val_by_source=val_by_source,
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
