#!/usr/bin/env python3
"""Jointly train one variable-agent HiSSD checkpoint on 3-1 and 4-2 sources.

The two population configurations are never padded into one tensor. Instead,
whole minibatches are alternated with equal configuration probability:

  drone:    [B,T,3,...] <-> [B,T,4,...]
  observer: [B,T,1,...] <-> [B,T,2,...]

Task IDs remain the source *difficulty* IDs shared by both configurations.
Thus population size is not turned into a task label.

This trainer reuses the objective functions in train_hissd_hetero_baseline.py
and only replaces data scheduling, role-batch preparation, and the model class.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
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
from skill_discovery.hissd_variable_models import load_variable_hissd_checkpoint
from skill_discovery.multi_source_hissd import (
    SourceSpec,
    balanced_source_batches,
    create_role_dataloader,
    prepare_role_batch,
)
from skill_discovery.task_descriptor import REALIZED_TASK_DESCRIPTOR_NAMES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--role", choices=("drone", "observer"), required=True)
    p.add_argument("--manifest-31", type=Path, required=True)
    p.add_argument("--data-root-31", type=Path, required=True)
    p.add_argument("--manifest-42", type=Path, required=True)
    p.add_argument("--data-root-42", type=Path, required=True)
    p.add_argument("--warmstart-checkpoint", type=Path, required=True)
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
            "Equal train batches drawn from each population per epoch. "
            "0 uses max(len(loader31), len(loader42))."
        ),
    )
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")

    # Common overrides. Values omitted here retain the installed single-source
    # trainer's defaults, keeping this wrapper aligned with repo revisions.
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


def base_default_args(role: str) -> argparse.Namespace:
    """Obtain parser defaults without satisfying required CLI arguments.

    Some repo revisions mark --role or other arguments as required. Calling
    base.parse_args() normally would raise SystemExit. Temporarily replace
    ArgumentParser.parse_args with a defaults-only implementation, let the
    base trainer construct its parser normally, then restore argparse.
    """
    original_parse_args = argparse.ArgumentParser.parse_args

    def defaults_only_parse_args(
        parser: argparse.ArgumentParser,
        args=None,
        namespace=None,
    ) -> argparse.Namespace:
        result = argparse.Namespace() if namespace is None else namespace
        for action in parser._actions:
            if action.dest == "help":
                continue
            if action.default is argparse.SUPPRESS:
                continue
            setattr(result, action.dest, action.default)
        return result

    argparse.ArgumentParser.parse_args = defaults_only_parse_args
    try:
        args = base.parse_args()
    finally:
        argparse.ArgumentParser.parse_args = original_parse_args

    args.role = role
    return args


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_training_args(cli: argparse.Namespace) -> argparse.Namespace:
    args = base_default_args(cli.role)
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

    # Recompute derived ablation switches after overrides if the base exposes it.
    if hasattr(base, "configure_ablation"):
        base.configure_ablation(args)
    return args


def manifest_info(path: Path) -> tuple[list[int], tuple[str, ...]]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    tasks = sorted(int(x) for x in payload.get("source_difficulties", ()))
    schema = payload.get("task_descriptor", {})
    names = tuple(schema.get("names", REALIZED_TASK_DESCRIPTOR_NAMES))
    return tasks, names


def sample_signature(sample: dict[str, Any], role: str) -> dict[str, Any]:
    obs = sample[role]["observations"]
    return {
        "global_map_channels": int(obs["global_map"].shape[-3]),
        "local_map_channels": int(obs["local_map"].shape[-3]),
        "global_map_size": tuple(obs["global_map"].shape[-2:]),
        "local_map_size": tuple(obs["local_map"].shape[-2:]),
        "action_history_shape": tuple(obs["action_history"].shape[-2:]),
        "action_dim": int(sample[role]["actions"].shape[-1]),
        "agent_count": int(sample[role]["actions"].shape[-2]),
        "central_map_channels": int(sample["global_state"]["central_map"].shape[-3]),
        "central_map_size": tuple(sample["global_state"]["central_map"].shape[-2:]),
    }


def validate_source_compatibility(
    datasets: dict[str, Any],
    role: str,
    model,
) -> None:
    signatures = {name: sample_signature(ds[0], role) for name, ds in datasets.items()}
    reference = next(iter(signatures.values()))
    ignore = {"agent_count"}
    for name, sig in signatures.items():
        for key, value in sig.items():
            if key in ignore:
                continue
            if value != reference[key]:
                raise ValueError(
                    f"Source observation/action schema mismatch for {key}: "
                    f"reference={reference[key]}, {name}={value}"
                )

    cfg = model.config()
    for key in (
        "global_map_channels",
        "local_map_channels",
        "global_map_size",
        "local_map_size",
        "action_history_shape",
        "action_dim",
        "central_map_channels",
        "central_map_size",
    ):
        expected = reference[key]
        actual = cfg[key]
        if isinstance(actual, list):
            actual = tuple(actual)
        if isinstance(expected, tuple):
            actual = tuple(actual)
        if actual != expected:
            raise ValueError(
                f"Warmstart model incompatible with source data for {key}: "
                f"model={actual}, data={expected}"
            )

    populations = {name: sig["agent_count"] for name, sig in signatures.items()}
    print(f"role={role} source_population_sizes={populations}", flush=True)


def combined_descriptor_statistics(datasets: dict[str, Any], floor: float):
    """Equal-configuration average of per-source descriptor statistics."""
    means = []
    scales = []
    for dataset in datasets.values():
        mean, scale = dataset.task_descriptor_statistics(floor)
        means.append(mean.float())
        scales.append(scale.float())
    return torch.stack(means).mean(0), torch.stack(scales).mean(0)


def optimizer_for_model(model, args: argparse.Namespace):
    task_parameters = [
        p for p in model.task_skill_encoder.parameters() if p.requires_grad
    ]
    if model.task_descriptor_head is not None:
        task_parameters.extend(p for p in model.task_descriptor_head.parameters() if p.requires_grad)
    if model.task_observation_encoder is not None:
        task_parameters.extend(p for p in model.task_observation_encoder.parameters() if p.requires_grad)
    if model.task_classifier_head is not None:
        task_parameters.extend(p for p in model.task_classifier_head.parameters() if p.requires_grad)

    task_ids = {id(p) for p in task_parameters}
    base_parameters = [
        p for p in model.parameters() if p.requires_grad and id(p) not in task_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": base_parameters,
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
            },
            {
                "params": task_parameters,
                "lr": args.learning_rate * args.task_learning_rate_multiplier,
                "weight_decay": args.task_weight_decay,
            },
        ]
    )
    return optimizer, base_parameters, task_parameters


def add_metrics(accumulator: dict[str, float], metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        accumulator[key] += float(value)


def average_metrics(accumulator: dict[str, float], n: int) -> dict[str, float]:
    if n <= 0:
        raise RuntimeError("No batches processed")
    return {k: v / n for k, v in accumulator.items()}


def train_epoch(
    model,
    loaders: dict[str, Any],
    optimizer,
    device: torch.device,
    args: argparse.Namespace,
    *,
    role: str,
    epoch: int,
    batches_per_source: int | None,
    task_only: bool,
):
    model.train()
    total = defaultdict(float)
    per_source = {name: defaultdict(float) for name in loaders}
    counts = defaultdict(int)

    iterator = balanced_source_batches(
        loaders,
        seed=args.seed + epoch,
        batches_per_source=batches_per_source,
    )
    max_total = args.max_train_batches
    processed = 0
    for source_name, raw_batch in iterator:
        if max_total is not None and processed >= max_total:
            break
        batch = prepare_role_batch(
            raw_batch, device, role, base_trainer=base
        )

        controller_loss, controller_metrics = base.controller_objective(
            model, batch, args, task_only=task_only
        )
        controller_metrics["controller_grad_norm"] = base.optimize(
            controller_loss, model, optimizer, args.grad_clip
        )

        if task_only:
            with torch.no_grad():
                _, value_metrics = base.value_objective(model, batch, args)
                _, planner_metrics = base.planner_objective(model, batch, args)
            value_metrics["value_grad_norm"] = 0.0
            planner_metrics["planner_grad_norm"] = 0.0
            planner_metrics["planner_update_skipped"] = 0.0
        else:
            value_loss, value_metrics = base.value_objective(model, batch, args)
            value_metrics["value_grad_norm"] = base.optimize(
                value_loss, model, optimizer, args.grad_clip
            )
            with base.strict_planner_math(device):
                planner_loss, planner_metrics = base.planner_objective(
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
def validate_one_source(model, loader, device, args, role: str):
    model.eval()
    acc = defaultdict(float)
    count = 0
    for raw_batch in loader:
        if args.max_val_batches is not None and count >= args.max_val_batches:
            break
        batch = prepare_role_batch(raw_batch, device, role, base_trainer=base)
        _, c = base.controller_objective(model, batch, args)
        _, v = base.value_objective(model, batch, args)
        _, p = base.planner_objective(model, batch, args)
        add_metrics(acc, {**c, **v, **p})
        count += 1
    return average_metrics(acc, count)


@torch.inference_mode()
def validate_all_sources(model, loaders, device, args, role: str):
    by_source = {
        name: validate_one_source(model, loader, device, args, role)
        for name, loader in loaders.items()
    }
    keys = set.intersection(*(set(m) for m in by_source.values()))
    equal_config_average = {
        key: sum(metrics[key] for metrics in by_source.values()) / len(by_source)
        for key in keys
    }
    return equal_config_average, by_source


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    *,
    epoch: int,
    role: str,
    args,
    cli,
    train_metrics,
    train_by_source,
    val_metrics,
    val_by_source,
    warmstart_payload,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 2,
            "model_type": "hemac_variable_agent_hissd",
            "role": role,
            "model_config": model.config(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "skill_structure": model.skill_structure,
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
                "warmstart_checkpoint": str(cli.warmstart_checkpoint),
                "warmstart_epoch": warmstart_payload.get("epoch"),
                "target_cardinality_intent": (
                    "drone<=5, observer<=2"
                ),
            },
            "training_args": vars(args),
        },
        path,
    )


def compact(metrics: dict[str, float]) -> str:
    def g(name, default=float("nan")):
        return metrics.get(name, default)
    return (
        f"action={g('action_mse'):.5f} "
        f"task={g('task_contrastive_loss'):.4f} "
        f"task_acc={g('task_contrastive_accuracy'):.3f} "
        f"z_task_acc={g('task_action_contrastive_accuracy'):.3f} "
        f"value={g('value_loss'):.4f} "
        f"planner={g('planner_loss'):.4f}"
    )


def main() -> None:
    cli = parse_args()
    args = configure_training_args(cli)
    seed_everything(cli.seed)
    device = resolve_device(cli.device)
    if hasattr(base, "configure_gpu_backend"):
        base.configure_gpu_backend(device)

    source_specs = {
        "3-1": SourceSpec("3-1", cli.manifest_31, cli.data_root_31),
        "4-2": SourceSpec("4-2", cli.manifest_42, cli.data_root_42),
    }
    task31, names31 = manifest_info(cli.manifest_31)
    task42, names42 = manifest_info(cli.manifest_42)
    if task31 != task42:
        raise ValueError(
            "Both population configurations must use the same source difficulty "
            f"IDs. 3-1={task31}, 4-2={task42}"
        )
    if task31 != list(range(1, len(task31) + 1)):
        raise ValueError(f"Source difficulties must be contiguous from 1: {task31}")
    if names31 != names42:
        raise ValueError(
            "3-1 and 4-2 task descriptor schemas differ; do not silently mix them"
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
            seed=cli.seed + 10000 + index * 1000,
            pin_memory=pin,
            cache_size=cli.dataset_cache_size,
        )

    model, warmstart_payload = load_variable_hissd_checkpoint(
        cli.warmstart_checkpoint,
        device,
    )
    validate_source_compatibility(train_datasets, cli.role, model)

    args.task_descriptor_names = names31
    args.training_tasks = task31
    args.held_out_tasks = []
    mean, scale = combined_descriptor_statistics(
        train_datasets,
        args.descriptor_scale_floor,
    )
    args.task_descriptor_mean = mean.tolist()
    args.task_descriptor_scale = scale.tolist()

    # The warmstart already has the correct two difficulty heads. Refuse to
    # silently reinterpret task IDs if it was trained on another task count.
    expected_prior_count = len(task31) if args.task_contrastive_enabled else 0
    if model.task_prior_count != expected_prior_count:
        raise ValueError(
            "Warmstart task prior count does not match combined source tasks: "
            f"model={model.task_prior_count}, expected={expected_prior_count}"
        )

    # Keep existing ablation freezes consistent with the original trainer.
    if model.skill_structure in {"shared", "common_only"}:
        for module in (
            model.task_observation_encoder,
            model.task_skill_encoder,
            model.target_task_skill_encoder,
        ):
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
    if model.skill_structure == "task_only":
        for parameter in model.common_skill_encoder.parameters():
            parameter.requires_grad_(False)

    optimizer, base_params, task_params = optimizer_for_model(model, args)
    cli.output_dir.mkdir(parents=True, exist_ok=True)

    batches_per_source = (
        None if cli.batches_per_source <= 0 else cli.batches_per_source
    )
    if batches_per_source is None:
        effective_bps = max(len(x) for x in train_loaders.values())
    else:
        effective_bps = batches_per_source

    print(f"device={device} role={cli.role}", flush=True)
    print(
        "train_windows="
        + str({name: len(ds) for name, ds in train_datasets.items()})
        + " val_windows="
        + str({name: len(ds) for name, ds in val_datasets.items()}),
        flush=True,
    )
    print(
        f"balanced_batches_per_source={effective_bps} "
        f"updates_per_epoch={2 * effective_bps} "
        f"base_params={sum(p.numel() for p in base_params):,} "
        f"task_params={sum(p.numel() for p in task_params):,}",
        flush=True,
    )
    print(
        f"descriptor_scale={[round(x,4) for x in args.task_descriptor_scale]} "
        f"task_ids=difficulty_only:{task31}",
        flush=True,
    )

    best_combined = math.inf
    best_task = math.inf
    best_task_epoch = 0
    no_improve = 0
    aux_enabled = bool(args.descriptor_enabled or args.task_contrastive_enabled)

    for epoch in range(1, cli.epochs + 1):
        task_only = epoch <= args.task_warmup_epochs
        train_metrics, train_by_source = train_epoch(
            model,
            train_loaders,
            optimizer,
            device,
            args,
            role=cli.role,
            epoch=epoch,
            batches_per_source=batches_per_source,
            task_only=task_only,
        )
        val_metrics, val_by_source = validate_all_sources(
            model, val_loaders, device, args, cli.role
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
        print(line, flush=True)

        save_checkpoint(
            cli.output_dir / "hissd_last.pt",
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
            warmstart_payload=warmstart_payload,
        )
        if combined_loss < best_combined:
            best_combined = combined_loss
            save_checkpoint(
                cli.output_dir / "hissd_best.pt",
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
                warmstart_payload=warmstart_payload,
            )

        selection = task_loss if aux_enabled else combined_loss
        if selection < best_task:
            best_task = selection
            best_task_epoch = epoch
            no_improve = 0
            save_checkpoint(
                cli.output_dir / "hissd_best_task.pt",
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
                warmstart_payload=warmstart_payload,
            )
        else:
            no_improve += 1

        patience = int(args.early_stopping_patience)
        if patience > 0 and no_improve >= patience:
            print(
                f"Early stopping: no task-validation improvement for "
                f"{patience} epochs (best epoch={best_task_epoch}).",
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
