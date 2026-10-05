#!/usr/bin/env python3
"""Jointly train HiSSD + common-skill adapter on D3O1 and D4O2.

The model is initialized directly from the role-matched multisource BC
checkpoint. No pretrained HiSSD checkpoint is required.

Architecture:
    CommonSkillEncoder -> residual adapter -> unchanged downstream HiSSD

The adapter is part of the normal HiSSD optimizer, so gradients from the
existing HiSSD objectives update both the adapter and the CommonSkillEncoder.
No adapter-specific loss is added.

D3O1 and D4O2 are kept in separate minibatches because their agent counts
differ. Equal numbers of minibatches are sampled from each population.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

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
from skill_discovery.hissd_adapter_variable_models import (
    VariableAgentAdaptedHeMACHISSD,
)
from skill_discovery.multi_source_hissd import (
    SourceSpec,
    create_role_dataloader,
    prepare_role_batch,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--role", choices=("drone", "observer"), required=True)

    p.add_argument("--manifest-31", type=Path, required=True)
    p.add_argument("--data-root-31", type=Path, required=True)
    p.add_argument("--manifest-42", type=Path, required=True)
    p.add_argument("--data-root-42", type=Path, required=True)

    p.add_argument("--bc-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)

    p.add_argument("--epochs", type=int, default=70)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--sequence-length", type=int, default=128)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--dataset-cache-size", type=int, default=16)
    p.add_argument(
        "--batches-per-source",
        type=int,
        default=0,
        help=(
            "Equal train batches from EACH population per epoch. "
            "0 uses max(len(loader31), len(loader42))."
        ),
    )

    p.add_argument(
        "--adapter-hidden-dim",
        type=int,
        default=128,
        help="Hidden width of the residual common-skill adapter.",
    )

    p.add_argument("--seed", type=int, default=2026)
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )

    # Optional overrides; otherwise preserve the installed HiSSD defaults.
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--task-learning-rate-multiplier", type=float)
    p.add_argument("--weight-decay", type=float)
    p.add_argument("--task-weight-decay", type=float)
    p.add_argument("--task-warmup-epochs", type=int)
    p.add_argument("--task-contrastive-weight", type=float)
    p.add_argument("--task-action-contrastive-weight", type=float)
    p.add_argument("--early-stopping-patience", type=int)
    p.add_argument("--max-train-batches", type=int)
    p.add_argument("--max-val-batches", type=int)
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def configure_training_args(cli: argparse.Namespace) -> argparse.Namespace:
    # Reuse the exact vanilla multisource HiSSD defaults.
    args = multi.base_default_args(cli.role)
    args.role = cli.role
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
    args.bc_checkpoint = cli.bc_checkpoint

    for name in (
        "learning_rate",
        "task_learning_rate_multiplier",
        "weight_decay",
        "task_weight_decay",
        "task_warmup_epochs",
        "task_contrastive_weight",
        "task_action_contrastive_weight",
        "early_stopping_patience",
    ):
        value = getattr(cli, name)
        if value is not None:
            setattr(args, name, value)

    if hasattr(base, "configure_ablation"):
        base.configure_ablation(args)

    if args.skill_structure != "split":
        raise ValueError(
            "Adapter experiment requires the normal split HiSSD structure; "
            f"got skill_structure={args.skill_structure!r}."
        )
    return args


def expected_bc_model_type(role: str) -> str:
    return (
        "drone_behavior_cloning"
        if role == "drone"
        else "observer_behavior_cloning"
    )


def validate_bc_checkpoint(
    path: Path,
    role: str,
) -> dict[str, Any]:
    payload = torch.load(
        path.expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    expected = expected_bc_model_type(role)
    actual = payload.get("model_type")
    if actual != expected:
        raise ValueError(
            f"Role {role!r} requires BC type {expected!r}, got "
            f"{actual!r}: {path}"
        )
    return payload


def build_model(
    sample: dict[str, Any],
    args: argparse.Namespace,
    role: str,
    source_task_count: int,
    adapter_hidden_dim: int,
) -> VariableAgentAdaptedHeMACHISSD:
    observations = sample[role]["observations"]
    global_shape = observations["global_map"].shape
    local_shape = observations["local_map"].shape
    history_shape = observations["action_history"].shape
    central_shape = sample["global_state"]["central_map"].shape
    action_shape = sample[role]["actions"].shape

    return VariableAgentAdaptedHeMACHISSD(
        global_map_channels=int(global_shape[-3]),
        local_map_channels=int(local_shape[-3]),
        central_map_channels=int(central_shape[-3]),
        agent_count=int(action_shape[-2]),
        action_dim=int(action_shape[-1]),
        global_map_size=tuple(global_shape[-2:]),
        local_map_size=tuple(local_shape[-2:]),
        central_map_size=tuple(central_shape[-2:]),
        action_history_shape=tuple(history_shape[-2:]),
        hidden_dim=args.hidden_dim,
        skill_dim=args.skill_dim,
        transformer_heads=args.transformer_heads,
        contrastive_from_action_skill=False,
        task_context_pooling=False,
        task_descriptor_dim=(
            int(sample["task_descriptor"].numel())
            if args.descriptor_enabled
            else 0
        ),
        task_prior_count=(
            source_task_count
            if args.task_contrastive_enabled
            else 0
        ),
        task_dropout=args.task_dropout,
        task_feature_deltas=True,
        separate_task_observation_encoder=True,
        learned_task_classifier=args.task_contrastive_enabled,
        task_spatial_statistics=True,
        normalize_task_context=False,
        task_running_statistics=True,
        direct_task_summary=False,
        skill_structure=args.skill_structure,
        common_skill_adapter_hidden_dim=adapter_hidden_dim,
    )


@torch.inference_mode()
def initialization_diagnostics(
    model: VariableAgentAdaptedHeMACHISSD,
    raw_batch: dict[str, Any],
    device: torch.device,
    role: str,
) -> dict[str, float]:
    """Verify both BC preservation and exact adapter identity at initialization."""
    batch = prepare_role_batch(
        raw_batch,
        device,
        role,
        base_trainer=base,
    )

    features = model.encode_observations(batch["observations"])
    task_features = model.encode_task_observations(batch["observations"])

    identity_error = model.adapter_identity_error(
        features,
        batch["valid_agents"],
    )

    common, task_skill, _ = model.infer_skills(
        features,
        batch["valid_agents"],
        task_features,
    )
    hissd_action = model.decode_actions(
        features,
        common,
        task_skill,
    )
    bc_action = torch.tanh(
        model.action_decoder.base_action_head(features)
    )
    bc_error = float(
        (hissd_action - bc_action).abs().max().item()
    )

    return {
        "adapter_identity_error": identity_error,
        "bc_action_max_difference": bc_error,
    }


@torch.inference_mode()
def adapter_diagnostics(
    model: VariableAgentAdaptedHeMACHISSD,
    raw_batch: dict[str, Any],
    device: torch.device,
    role: str,
) -> dict[str, float]:
    batch = prepare_role_batch(
        raw_batch,
        device,
        role,
        base_trainer=base,
    )
    features = model.encode_observations(batch["observations"])
    return model.adapter_delta_statistics(
        features,
        batch["valid_agents"],
    )


def save_checkpoint(
    path: Path,
    model: VariableAgentAdaptedHeMACHISSD,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    role: str,
    args: argparse.Namespace,
    cli: argparse.Namespace,
    train_metrics: dict[str, float],
    train_by_source: dict[str, dict[str, float]],
    val_metrics: dict[str, float],
    val_by_source: dict[str, dict[str, float]],
    bc_info: dict[str, Any],
    adapter_metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "format_version": 3,
            "model_type": "hemac_variable_agent_hissd_common_adapter",
            "role": role,
            "model_config": model.config(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "skill_structure": model.skill_structure,
            "adapter": {
                "placement": "immediately_after_common_skill_encoder",
                "input": "common_skill_only",
                "form": "residual_mlp",
                "hidden_dim": cli.adapter_hidden_dim,
                "downstream_scope": "all_existing_common_skill_consumers",
                "metrics": adapter_metrics,
            },
            "bc_initialization": {
                "checkpoint": str(cli.bc_checkpoint),
                "epoch": bc_info.get("epoch"),
                "model_type": bc_info.get("model_type"),
                "metrics": bc_info.get("metrics", {}),
            },
            "training_metrics": train_metrics,
            "training_metrics_by_source": train_by_source,
            "validation_metrics": val_metrics,
            "validation_metrics_by_source": val_by_source,
            "multisource": {
                "task_identity": "difficulty_only",
                "configuration_balancing": "equal_batches_per_source",
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


def progress_write(message: str) -> None:
    """Print a persistent log line without overwriting active progress bars."""
    if tqdm is not None:
        tqdm.write(message)
    else:
        print(message, flush=True)


def main() -> None:
    cli = parse_args()

    if cli.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if cli.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if cli.sequence_length <= 0:
        raise ValueError("--sequence-length must be positive.")
    if cli.adapter_hidden_dim <= 0:
        raise ValueError("--adapter-hidden-dim must be positive.")
    if cli.batches_per_source < 0:
        raise ValueError("--batches-per-source cannot be negative.")

    args = configure_training_args(cli)
    multi.seed_everything(cli.seed)
    device = multi.resolve_device(cli.device)

    if hasattr(base, "configure_gpu_backend"):
        base.configure_gpu_backend(device)

    bc_info = validate_bc_checkpoint(
        cli.bc_checkpoint,
        cli.role,
    )

    source_specs = {
        "3-1": SourceSpec(
            "3-1",
            cli.manifest_31,
            cli.data_root_31,
        ),
        "4-2": SourceSpec(
            "4-2",
            cli.manifest_42,
            cli.data_root_42,
        ),
    }

    task31, names31 = multi.manifest_info(cli.manifest_31)
    task42, names42 = multi.manifest_info(cli.manifest_42)

    if task31 != task42:
        raise ValueError(
            "Both population configurations must use the same source "
            f"difficulty IDs. 3-1={task31}, 4-2={task42}"
        )
    if not task31:
        raise ValueError("No source difficulties found in manifests.")
    if task31 != list(range(1, len(task31) + 1)):
        raise ValueError(
            f"Source difficulties must be contiguous from 1: {task31}"
        )
    if names31 != names42:
        raise ValueError(
            "3-1 and 4-2 task descriptor schemas differ."
        )

    pin = device.type == "cuda"
    train_datasets = {}
    train_loaders = {}
    val_datasets = {}
    val_loaders = {}

    for index, (name, spec) in enumerate(source_specs.items()):
        train_datasets[name], train_loaders[name] = create_role_dataloader(
            spec,
            "source_train",
            role=cli.role,
            sequence_length=cli.sequence_length,
            stride=cli.stride,
            batch_size=cli.batch_size,
            num_workers=cli.num_workers,
            seed=cli.seed + index * 1000,
            pin_memory=pin,
            cache_size=cli.dataset_cache_size,
        )
        val_datasets[name], val_loaders[name] = create_role_dataloader(
            spec,
            "source_val",
            role=cli.role,
            sequence_length=cli.sequence_length,
            stride=cli.stride,
            batch_size=cli.batch_size,
            num_workers=cli.num_workers,
            seed=cli.seed + 10_000 + index * 1000,
            pin_memory=pin,
            cache_size=cli.dataset_cache_size,
        )

    args.task_descriptor_names = names31
    args.training_tasks = task31
    args.held_out_tasks = []

    descriptor_mean, descriptor_scale = (
        multi.combined_descriptor_statistics(
            train_datasets,
            args.descriptor_scale_floor,
        )
    )
    args.task_descriptor_mean = descriptor_mean.tolist()
    args.task_descriptor_scale = descriptor_scale.tolist()

    model = build_model(
        train_datasets["3-1"][0],
        args,
        cli.role,
        source_task_count=len(task31),
        adapter_hidden_dim=cli.adapter_hidden_dim,
    ).to(device)

    # Exact same BC initialization used by vanilla HiSSD.
    loaded_bc_info = model.initialize_from_bc(cli.bc_checkpoint)
    # Keep richer checkpoint fields when available.
    bc_info.update(loaded_bc_info)

    multi.validate_source_compatibility(
        train_datasets,
        cli.role,
        model,
    )

    expected_prior_count = (
        len(task31)
        if args.task_contrastive_enabled
        else 0
    )
    if model.task_prior_count != expected_prior_count:
        raise RuntimeError(
            "Unexpected task prior count: "
            f"model={model.task_prior_count}, "
            f"expected={expected_prior_count}"
        )

    # Preserve vanilla ablation semantics.
    if model.skill_structure in {"shared", "common_only"}:
        raise ValueError(
            "This adapter experiment is defined for skill_structure='split'."
        )

    optimizer, base_params, task_params = multi.optimizer_for_model(
        model,
        args,
    )

    adapter_params = sum(
        p.numel()
        for p in model.common_skill_encoder.adapter.parameters()
    )
    common_base_params = sum(
        p.numel()
        for p in model.common_skill_encoder.base_encoder.parameters()
    )

    # Verify initialization on a real D3O1 minibatch.
    initial_raw_batch = next(iter(train_loaders["3-1"]))
    init_diag = initialization_diagnostics(
        model,
        initial_raw_batch,
        device,
        cli.role,
    )
    if init_diag["adapter_identity_error"] > 1e-8:
        raise RuntimeError(
            "Adapter is not identity at initialization: "
            f"{init_diag['adapter_identity_error']:.3e}"
        )
    if init_diag["bc_action_max_difference"] > 1e-6:
        raise RuntimeError(
            "HiSSD+adapter does not preserve BC actions at initialization: "
            f"{init_diag['bc_action_max_difference']:.3e}"
        )

    cli.output_dir.mkdir(parents=True, exist_ok=True)

    batches_per_source = (
        None
        if cli.batches_per_source <= 0
        else cli.batches_per_source
    )
    effective_bps = (
        max(len(loader) for loader in train_loaders.values())
        if batches_per_source is None
        else batches_per_source
    )

    print(
        f"device={device} role={cli.role} "
        f"source_population_sizes="
        f"{{'3-1': {train_datasets['3-1'][0][cli.role]['actions'].shape[-2]}, "
        f"'4-2': {train_datasets['4-2'][0][cli.role]['actions'].shape[-2]}}}",
        flush=True,
    )
    print(
        "train_windows="
        + str({name: len(ds) for name, ds in train_datasets.items()})
        + " val_windows="
        + str({name: len(ds) for name, ds in val_datasets.items()}),
        flush=True,
    )
    print(
        f"balanced_batches_per_source={effective_bps} "
        f"updates_per_epoch={2 * effective_bps}",
        flush=True,
    )
    print(
        f"base_params={sum(p.numel() for p in base_params):,} "
        f"task_params={sum(p.numel() for p in task_params):,} "
        f"common_encoder_params={common_base_params:,} "
        f"adapter_params={adapter_params:,}",
        flush=True,
    )
    print(
        f"BC initialization epoch={bc_info.get('epoch')} "
        f"max_action_difference="
        f"{init_diag['bc_action_max_difference']:.3e} "
        f"adapter_identity_error="
        f"{init_diag['adapter_identity_error']:.3e}",
        flush=True,
    )
    print(
        f"adapter=common_skill_only residual_mlp "
        f"hidden_dim={cli.adapter_hidden_dim} "
        f"placement=after_common_skill_encoder "
        f"downstream=all",
        flush=True,
    )
    print(
        f"descriptor_scale="
        f"{[round(x, 4) for x in args.task_descriptor_scale]} "
        f"task_ids=difficulty_only:{task31}",
        flush=True,
    )

    best_combined = math.inf
    best_task = math.inf
    best_task_epoch = 0
    no_improve = 0
    aux_enabled = bool(
        args.descriptor_enabled
        or args.task_contrastive_enabled
    )

    # Fixed diagnostic minibatch so adapter magnitude is comparable by epoch.
    diagnostic_raw_batch = next(iter(val_loaders["3-1"]))

    show_progress = tqdm is not None and not cli.no_progress
    epoch_progress = range(1, cli.epochs + 1)
    if show_progress:
        epoch_progress = tqdm(
            epoch_progress,
            total=cli.epochs,
            desc=f"{cli.role} HiSSD adapter",
            unit="epoch",
            dynamic_ncols=True,
            position=0,
        )

    try:
        for epoch in epoch_progress:
            task_only = epoch <= args.task_warmup_epochs

            train_metrics, train_by_source = multi.train_epoch(
                model,
                train_loaders,
                optimizer,
                device,
                args,
                role=cli.role,
                epoch=epoch,
                batches_per_source=batches_per_source,
                task_only=task_only,
                show_progress=show_progress,
            )
            val_metrics, val_by_source = multi.validate_all_sources(
                model,
                val_loaders,
                device,
                args,
                cli.role,
                epoch=epoch,
                show_progress=show_progress,
            )
            combined_loss, task_loss = base.validation_losses(
                val_metrics,
                args,
            )
            adapter_metrics = adapter_diagnostics(
                model,
                diagnostic_raw_batch,
                device,
                cli.role,
            )

            line = (
                f"epoch={epoch:03d} "
                f"phase={'task_warmup' if task_only else 'joint'} "
                f"train[{multi.compact(train_metrics)}] "
                f"val[{multi.compact(val_metrics)}] "
                f"adapter_delta={adapter_metrics['adapter_delta_rms']:.5f} "
                f"adapter_ratio={adapter_metrics['adapter_to_common_ratio']:.4f} "
                + " ".join(
                    f"{name}[{multi.compact(metrics)}]"
                    for name, metrics in val_by_source.items()
                )
            )
            progress_write(line)
            if show_progress and hasattr(epoch_progress, "set_postfix"):
                epoch_progress.set_postfix(
                    train=f"{train_metrics['action_mse']:.4f}",
                    val=f"{val_metrics['action_mse']:.4f}",
                    refresh=False,
                )

            save_checkpoint(
                cli.output_dir / "hissd_adapter_last.pt",
                model,
                optimizer,
                epoch=epoch,
                role=cli.role,
                args=args,
                cli=cli,
                train_metrics=train_metrics,
                train_by_source=train_by_source,
                val_metrics=val_metrics,
                val_by_source=val_by_source,
                bc_info=bc_info,
                adapter_metrics=adapter_metrics,
            )

            if combined_loss < best_combined:
                best_combined = combined_loss
                save_checkpoint(
                    cli.output_dir / "hissd_adapter_best.pt",
                    model,
                    optimizer,
                    epoch=epoch,
                    role=cli.role,
                    args=args,
                    cli=cli,
                    train_metrics=train_metrics,
                    train_by_source=train_by_source,
                    val_metrics=val_metrics,
                    val_by_source=val_by_source,
                    bc_info=bc_info,
                    adapter_metrics=adapter_metrics,
                )

            selection = task_loss if aux_enabled else combined_loss
            if selection < best_task:
                best_task = selection
                best_task_epoch = epoch
                no_improve = 0
                save_checkpoint(
                    cli.output_dir / "hissd_adapter_best_task.pt",
                    model,
                    optimizer,
                    epoch=epoch,
                    role=cli.role,
                    args=args,
                    cli=cli,
                    train_metrics=train_metrics,
                    train_by_source=train_by_source,
                    val_metrics=val_metrics,
                    val_by_source=val_by_source,
                    bc_info=bc_info,
                    adapter_metrics=adapter_metrics,
                )
            else:
                no_improve += 1

            patience = int(args.early_stopping_patience)
            if patience > 0 and no_improve >= patience:
                progress_write(
                    "Early stopping: no task-validation improvement for "
                    f"{patience} epochs "
                    f"(best epoch={best_task_epoch})."
                )
                break
    finally:
        if show_progress and hasattr(epoch_progress, "close"):
            epoch_progress.close()

    progress_write(
        f"done best_combined={best_combined:.6f} "
        f"best_task={best_task:.6f}@{best_task_epoch}"
    )


if __name__ == "__main__":
    main()
