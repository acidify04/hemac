"""Train one role-specific BC policy from D3O1 and D4O2 source datasets.

The role remains fixed (drone or observer), while population configurations are
treated as separate source domains. Each epoch draws an equal number of minibatches
from D3O1 and D4O2 and updates one shared role policy.

Examples:
  drone:
    python -u src/skill_discovery/train_bc_multisource.py \
      --role drone \
      --manifest-31 src/skill_discovery/offline_data-3-1/dataset_splits_d1.json \
      --data-root-31 src/skill_discovery/offline_data-3-1 \
      --manifest-42 src/skill_discovery/offline_data-4-2/dataset_splits_d1.json \
      --data-root-42 src/skill_discovery/offline_data-4-2 \
      --output-dir src/skill_discovery/checkpoints/bc-source-combined/drone \
      --device cuda

  observer:
    python -u src/skill_discovery/train_bc_multisource.py \
      --role observer \
      --manifest-31 src/skill_discovery/offline_data-3-1/dataset_splits_d1.json \
      --data-root-31 src/skill_discovery/offline_data-3-1 \
      --manifest-42 src/skill_discovery/offline_data-4-2/dataset_splits_d1.json \
      --data-root-42 src/skill_discovery/offline_data-4-2 \
      --output-dir src/skill_discovery/checkpoints/bc-source-combined/observer \
      --device cuda
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from skill_discovery.dataset import create_dataloader
from skill_discovery.models import DroneBehaviorCloningPolicy


@dataclass(frozen=True)
class SourceSpec:
    name: str
    manifest: Path
    data_root: Path


@dataclass
class MetricAccumulator:
    """Accumulate masked BC errors without batch-size or agent-count bias."""

    squared_error: float = 0.0
    absolute_error: float = 0.0
    zero_squared_error: float = 0.0
    valid_values: int = 0
    valid_actions: int = 0
    axis_squared_error: torch.Tensor | None = None

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> None:
        mask = action_mask.to(dtype=prediction.dtype).unsqueeze(-1)
        difference = prediction - target
        action_dim = int(target.shape[-1])

        self.squared_error += float((difference.square() * mask).sum().item())
        self.absolute_error += float((difference.abs() * mask).sum().item())
        self.zero_squared_error += float((target.square() * mask).sum().item())

        action_count = int(action_mask.sum().item())
        self.valid_actions += action_count
        self.valid_values += action_count * action_dim

        reduce_dims = tuple(range(difference.ndim - 1))
        axis_error = (
            (difference.square() * mask)
            .sum(dim=reduce_dims)
            .detach()
            .cpu()
        )
        if self.axis_squared_error is None:
            self.axis_squared_error = axis_error
        else:
            self.axis_squared_error += axis_error

    def compute(self) -> dict[str, Any]:
        if self.valid_values <= 0 or self.valid_actions <= 0:
            raise RuntimeError("No valid role actions were present.")

        assert self.axis_squared_error is not None
        axis_mse = self.axis_squared_error / self.valid_actions
        return {
            "mse": self.squared_error / self.valid_values,
            "mae": self.absolute_error / self.valid_values,
            "zero_action_mse": self.zero_squared_error / self.valid_values,
            "axis_mse": axis_mse.tolist(),
            "valid_actions": self.valid_actions,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("drone", "observer"), required=True)

    parser.add_argument("--manifest-31", type=Path, required=True)
    parser.add_argument("--data-root-31", type=Path, required=True)
    parser.add_argument("--manifest-42", type=Path, required=True)
    parser.add_argument("--data-root-42", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument(
        "--balanced-sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the repository's outcome-balanced sampling inside each source.",
    )
    parser.add_argument(
        "--batches-per-source",
        type=int,
        default=0,
        help=(
            "Training minibatches drawn from EACH population per epoch. "
            "0 uses max(len(loader31), len(loader42)), oversampling the smaller "
            "source as needed. Thus D3O1 and D4O2 receive equal update counts."
        ),
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        help="Optional validation-batch cap per population; useful for smoke tests.",
    )

    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm epoch, training, and validation progress bars.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def create_role_dataloader(
    source: SourceSpec,
    split: str,
    args: argparse.Namespace,
    *,
    seed: int,
    pin_memory: bool,
):
    training = split == "source_train"
    return create_dataloader(
        manifest_path=source.manifest,
        split=split,
        data_root=source.data_root,
        sequence_length=args.sequence_length,
        stride=args.stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize_actions=True,
        include_observer=(args.role == "observer"),
        include_labels=False,
        balanced_sampling=(args.balanced_sampling if training else False),
        seed=seed,
        pin_memory=pin_memory,
        drop_last_batch=training,
    )


def role_tensors(
    batch: dict[str, Any],
    role: str,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """Extract decentralized observations/actions for one role.

    HeMAC joint ordering is observer(s) first, followed by drone(s).
    """
    role_payload = batch[role]
    observations = role_payload["observations"]

    global_map = observations["global_map"].to(device, non_blocking=True)
    local_map = observations["local_map"].to(device, non_blocking=True)
    action_history = observations["action_history"].to(device, non_blocking=True)
    actions = role_payload["actions"].to(device, non_blocking=True)

    role_count = int(actions.shape[-2])
    full_agent_mask = batch["agent_mask"]

    if int(full_agent_mask.shape[-1]) == role_count:
        role_mask = full_agent_mask
    elif role == "observer":
        role_mask = full_agent_mask[..., :role_count]
    else:
        role_mask = full_agent_mask[..., -role_count:]

    role_mask = role_mask.to(device, non_blocking=True).bool()
    filled = batch["filled"].to(device, non_blocking=True).bool()

    # Dataset convention is normally [B,T,1] for filled. Make the broadcasting
    # requirement explicit rather than silently accepting a malformed shape.
    if filled.ndim == role_mask.ndim - 1:
        filled = filled.unsqueeze(-1)
    if filled.shape[-1] not in (1, role_count):
        raise RuntimeError(
            f"Unexpected filled shape {tuple(filled.shape)} for "
            f"{role_count} {role} agents."
        )

    action_mask = role_mask & filled
    return global_map, local_map, action_history, actions, action_mask


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    mask = action_mask.to(dtype=prediction.dtype).unsqueeze(-1)
    denominator = mask.sum() * target.shape[-1]
    if denominator.item() == 0:
        raise RuntimeError("Batch contains no valid role actions.")
    return ((prediction - target).square() * mask).sum() / denominator


def build_model(
    sample: dict[str, Any],
    role: str,
) -> DroneBehaviorCloningPolicy:
    """Infer role observation/action dimensions; population size is not a parameter."""
    observations = sample[role]["observations"]
    global_shape = observations["global_map"].shape
    local_shape = observations["local_map"].shape
    history_shape = observations["action_history"].shape
    action_dim = int(sample[role]["actions"].shape[-1])

    return DroneBehaviorCloningPolicy(
        global_map_channels=global_shape[-3],
        local_map_channels=local_shape[-3],
        global_map_size=tuple(global_shape[-2:]),
        local_map_size=tuple(local_shape[-2:]),
        action_history_shape=tuple(history_shape[-2:]),
        action_dim=action_dim,
    )


def assert_source_compatibility(
    dataset_31,
    dataset_42,
    role: str,
) -> dict[str, Any]:
    model_31 = build_model(dataset_31[0], role)
    model_42 = build_model(dataset_42[0], role)
    config_31 = model_31.config()
    config_42 = model_42.config()

    if config_31 != config_42:
        raise ValueError(
            "D3O1 and D4O2 role model configs differ; refusing to mix them.\n"
            f"D3O1: {json.dumps(config_31, sort_keys=True)}\n"
            f"D4O2: {json.dumps(config_42, sort_keys=True)}"
        )
    return config_31


def resettable_next(
    source_name: str,
    loaders: dict[str, Any],
    iterators: dict[str, Any],
):
    try:
        return next(iterators[source_name])
    except StopIteration:
        iterators[source_name] = iter(loaders[source_name])
        try:
            return next(iterators[source_name])
        except StopIteration as exc:
            raise RuntimeError(
                f"Training loader {source_name!r} is empty."
            ) from exc


def run_train_epoch(
    model: DroneBehaviorCloningPolicy,
    loaders: dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
    *,
    epoch: int,
    show_progress: bool = False,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Update one shared role model with equal minibatch counts from both sources."""
    model.train()

    per_source_acc = {name: MetricAccumulator() for name in loaders}
    iterators = {name: iter(loader) for name, loader in loaders.items()}

    if args.batches_per_source > 0:
        batches_per_source = args.batches_per_source
    else:
        batches_per_source = max(len(loader) for loader in loaders.values())

    source_names = list(loaders)
    rng = random.Random(args.seed + epoch * 100_003)
    progress = None
    if tqdm is not None and show_progress:
        progress = tqdm(
            total=batches_per_source * len(source_names),
            desc=f"epoch {epoch} train",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
            position=1,
        )

    try:
        for _ in range(batches_per_source):
            # Keep equal counts but remove fixed source-order optimizer bias.
            order = list(source_names)
            rng.shuffle(order)

            for source_name in order:
                batch = resettable_next(source_name, loaders, iterators)
                (
                    global_map,
                    local_map,
                    action_history,
                    target,
                    action_mask,
                ) = role_tensors(batch, args.role, device)

                prediction = model(global_map, local_map, action_history)
                loss = masked_mse(prediction, target, action_mask)

                optimizer = args._optimizer
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

                per_source_acc[source_name].update(
                    prediction.detach(),
                    target,
                    action_mask,
                )
                if progress is not None:
                    progress.update(1)
                    progress.set_postfix(
                        source=source_name,
                        mse=f"{loss.item():.4f}",
                        refresh=False,
                    )
    finally:
        if progress is not None:
            progress.close()

    by_source = {
        name: accumulator.compute()
        for name, accumulator in per_source_acc.items()
    }
    return equal_source_average(by_source), by_source


@torch.inference_mode()
def run_validation(
    model: DroneBehaviorCloningPolicy,
    loaders: dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
    *,
    epoch: int | None = None,
    show_progress: bool = False,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    model.eval()
    by_source: dict[str, dict[str, Any]] = {}
    validation_total = sum(
        min(len(loader), args.max_val_batches)
        if args.max_val_batches is not None
        else len(loader)
        for loader in loaders.values()
    )
    progress = None
    if tqdm is not None and show_progress:
        progress = tqdm(
            total=validation_total,
            desc=f"epoch {epoch or '?'} validation",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
            position=1,
        )

    try:
        for source_name, loader in loaders.items():
            accumulator = MetricAccumulator()
            batch_count = 0

            for batch in loader:
                if (
                    args.max_val_batches is not None
                    and batch_count >= args.max_val_batches
                ):
                    break

                (
                    global_map,
                    local_map,
                    action_history,
                    target,
                    action_mask,
                ) = role_tensors(batch, args.role, device)

                prediction = model(global_map, local_map, action_history)
                accumulator.update(prediction, target, action_mask)
                batch_count += 1
                if progress is not None:
                    progress.update(1)
                    progress.set_postfix(source=source_name, refresh=False)

            if batch_count == 0:
                raise RuntimeError(
                    f"Validation loader {source_name!r} produced no batches."
                )
            by_source[source_name] = accumulator.compute()
    finally:
        if progress is not None:
            progress.close()

    return equal_source_average(by_source), by_source


def equal_source_average(
    by_source: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Average each population equally, independent of its agent/sample count."""
    if not by_source:
        raise RuntimeError("No source metrics to average.")

    metrics = list(by_source.values())
    axis_count = len(metrics[0]["axis_mse"])
    if any(len(item["axis_mse"]) != axis_count for item in metrics):
        raise ValueError("Action dimensions differ across source configurations.")

    n = float(len(metrics))
    return {
        "mse": sum(item["mse"] for item in metrics) / n,
        "mae": sum(item["mae"] for item in metrics) / n,
        "zero_action_mse": sum(item["zero_action_mse"] for item in metrics) / n,
        "axis_mse": [
            sum(item["axis_mse"][axis] for item in metrics) / n
            for axis in range(axis_count)
        ],
        "valid_actions": sum(int(item["valid_actions"]) for item in metrics),
    }


def save_checkpoint(
    path: Path,
    model: DroneBehaviorCloningPolicy,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    train_metrics: dict[str, Any],
    train_by_source: dict[str, dict[str, Any]],
    val_metrics: dict[str, Any],
    val_by_source: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    model_type = (
        "drone_behavior_cloning"
        if args.role == "drone"
        else "observer_behavior_cloning"
    )
    action_normalization = (
        "divide by per-episode drone_max_speed"
        if args.role == "drone"
        else "divide by per-episode observer_speed"
    )

    # Do not serialize the transient optimizer alias stored on args.
    serialized_args = {
        key: value
        for key, value in vars(args).items()
        if key != "_optimizer"
    }

    torch.save(
        {
            "format_version": 2,
            "model_type": model_type,
            "role": args.role,
            "model_config": model.config(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": {
                "train": train_metrics,
                "validation": val_metrics,
                "train_by_source": train_by_source,
                "validation_by_source": val_by_source,
            },
            "action_normalization": action_normalization,
            "multisource": {
                "configuration_balancing": "equal_batches_per_source",
                "sources": {
                    "3-1": {
                        "manifest": str(args.manifest_31),
                        "data_root": str(args.data_root_31),
                    },
                    "4-2": {
                        "manifest": str(args.manifest_42),
                        "data_root": str(args.data_root_42),
                    },
                },
            },
            "args": serialized_args,
        },
        path,
    )


def create_writer(args: argparse.Namespace):
    if args.no_tensorboard:
        return None
    from torch.utils.tensorboard import SummaryWriter
    return SummaryWriter(log_dir=args.output_dir / "tensorboard")


def progress_write(message: str) -> None:
    """Print a persistent line without corrupting active progress bars."""
    if tqdm is not None:
        tqdm.write(message)
    else:
        print(message, flush=True)


def format_source_metrics(
    prefix: str,
    metrics: dict[str, dict[str, Any]],
) -> str:
    parts = []
    for source_name in ("3-1", "4-2"):
        if source_name not in metrics:
            continue
        item = metrics[source_name]
        parts.append(
            f"{prefix}_{source_name}_mse={item['mse']:.6f}"
        )
    return " ".join(parts)


def main() -> None:
    args = parse_args()

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.sequence_length <= 0:
        raise ValueError("--sequence-length must be positive.")
    if args.batches_per_source < 0:
        raise ValueError("--batches-per-source cannot be negative.")
    if args.max_val_batches is not None and args.max_val_batches <= 0:
        raise ValueError("--max-val-batches must be positive when supplied.")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    pin_memory = device.type == "cuda"

    sources = {
        "3-1": SourceSpec(
            "3-1",
            args.manifest_31.expanduser().resolve(),
            args.data_root_31.expanduser().resolve(),
        ),
        "4-2": SourceSpec(
            "4-2",
            args.manifest_42.expanduser().resolve(),
            args.data_root_42.expanduser().resolve(),
        ),
    }

    train_datasets = {}
    train_loaders = {}
    val_datasets = {}
    val_loaders = {}

    for index, (name, source) in enumerate(sources.items()):
        train_datasets[name], train_loaders[name] = create_role_dataloader(
            source,
            "source_train",
            args,
            seed=args.seed + index * 1000,
            pin_memory=pin_memory,
        )
        val_datasets[name], val_loaders[name] = create_role_dataloader(
            source,
            "source_val",
            args,
            seed=args.seed + 10_000 + index * 1000,
            pin_memory=pin_memory,
        )

    model_config = assert_source_compatibility(
        train_datasets["3-1"],
        train_datasets["4-2"],
        args.role,
    )

    # Also verify validation tensors follow the same role schema.
    val_config_31 = build_model(val_datasets["3-1"][0], args.role).config()
    val_config_42 = build_model(val_datasets["4-2"][0], args.role).config()
    if val_config_31 != model_config or val_config_42 != model_config:
        raise ValueError(
            "Training and validation role schemas do not match."
        )

    model = build_model(train_datasets["3-1"][0], args.role).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    # Internal alias avoids threading optimizer through every call while keeping
    # checkpoint args clean.
    args._optimizer = optimizer

    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    population_counts = {
        name: int(dataset[0][args.role]["actions"].shape[-2])
        for name, dataset in train_datasets.items()
    }

    if args.batches_per_source > 0:
        effective_batches_per_source = args.batches_per_source
    else:
        effective_batches_per_source = max(
            len(loader) for loader in train_loaders.values()
        )

    print(
        f"device={device}, role={args.role}, parameters={parameter_count:,}, "
        f"population_counts={population_counts}"
    )
    print(f"model_config={json.dumps(model_config)}")
    print(
        "train_windows="
        f"3-1:{len(train_datasets['3-1'])}, "
        f"4-2:{len(train_datasets['4-2'])}; "
        "val_windows="
        f"3-1:{len(val_datasets['3-1'])}, "
        f"4-2:{len(val_datasets['4-2'])}"
    )
    print(
        f"configuration_balancing=equal_batches_per_source, "
        f"batches_per_source={effective_batches_per_source}, "
        f"updates_per_epoch={2 * effective_batches_per_source}"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    writer = create_writer(args)
    best_val_mse = float("inf")
    best_epoch = 0

    best_filename = f"{args.role}_bc_best.pt"
    last_filename = f"{args.role}_bc_last.pt"

    show_progress = tqdm is not None and not args.no_progress
    epoch_progress = range(1, args.epochs + 1)
    if show_progress:
        epoch_progress = tqdm(
            epoch_progress,
            total=args.epochs,
            desc=f"{args.role} multisource BC",
            unit="epoch",
            dynamic_ncols=True,
            position=0,
        )

    try:
        for epoch in epoch_progress:
            train_metrics, train_by_source = run_train_epoch(
                model,
                train_loaders,
                device,
                args,
                epoch=epoch,
                show_progress=show_progress,
            )
            val_metrics, val_by_source = run_validation(
                model,
                val_loaders,
                device,
                args,
                epoch=epoch,
                show_progress=show_progress,
            )

            progress_write(
                f"epoch={epoch:03d} "
                f"train_mse={train_metrics['mse']:.6f} "
                f"val_mse={val_metrics['mse']:.6f} "
                f"val_mae={val_metrics['mae']:.6f} "
                f"zero_mse={val_metrics['zero_action_mse']:.6f} "
                f"{format_source_metrics('train', train_by_source)} "
                f"{format_source_metrics('val', val_by_source)} "
                f"axis_mse="
                f"{[round(value, 6) for value in val_metrics['axis_mse']]}"
            )
            if show_progress and hasattr(epoch_progress, "set_postfix"):
                epoch_progress.set_postfix(
                    train=f"{train_metrics['mse']:.4f}",
                    val=f"{val_metrics['mse']:.4f}",
                    refresh=False,
                )

            if writer is not None:
                for name, value in train_metrics.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f"train_equal/{name}", value, epoch)
                for name, value in val_metrics.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f"val_equal/{name}", value, epoch)
                for source_name, metrics in train_by_source.items():
                    for name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            writer.add_scalar(
                                f"train_{source_name}/{name}", value, epoch
                            )
                for source_name, metrics in val_by_source.items():
                    for name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            writer.add_scalar(
                                f"val_{source_name}/{name}", value, epoch
                            )

            save_checkpoint(
                args.output_dir / last_filename,
                model,
                optimizer,
                epoch=epoch,
                train_metrics=train_metrics,
                train_by_source=train_by_source,
                val_metrics=val_metrics,
                val_by_source=val_by_source,
                args=args,
            )

            if val_metrics["mse"] < best_val_mse:
                best_val_mse = float(val_metrics["mse"])
                best_epoch = epoch
                save_checkpoint(
                    args.output_dir / best_filename,
                    model,
                    optimizer,
                    epoch=epoch,
                    train_metrics=train_metrics,
                    train_by_source=train_by_source,
                    val_metrics=val_metrics,
                    val_by_source=val_by_source,
                    args=args,
                )
    finally:
        if show_progress and hasattr(epoch_progress, "close"):
            epoch_progress.close()
        if writer is not None:
            writer.close()

    progress_write(
        f"Best equal-source validation MSE: {best_val_mse:.6f} "
        f"at epoch {best_epoch} "
        f"({args.output_dir / best_filename})"
    )


if __name__ == "__main__":
    main()
