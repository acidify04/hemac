"""Train and validate a shared drone behavior-cloning policy.

This is a data-pipeline baseline for drone-only skill discovery. The actor uses
only decentralized drone observations; central-map and central-vector inputs
are deliberately excluded.
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from skill_discovery.dataset import DEFAULT_MANIFEST_PATH, create_dataloader
from skill_discovery.models import DroneBehaviorCloningPolicy


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "src/skill_discovery/bc_checkpoints"


@dataclass
class MetricAccumulator:
    """Accumulate masked action errors without batch-size bias."""

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
        """Add one batch of masked prediction statistics."""
        mask = action_mask.to(dtype=prediction.dtype).unsqueeze(-1)
        difference = prediction - target
        action_dim = target.shape[-1]
        self.squared_error += float((difference.square() * mask).sum().item())
        self.absolute_error += float((difference.abs() * mask).sum().item())
        self.zero_squared_error += float((target.square() * mask).sum().item())
        action_count = int(action_mask.sum().item())
        self.valid_actions += action_count
        self.valid_values += action_count * action_dim
        axis_error = (difference.square() * mask).sum(dim=(0, 1, 2)).detach().cpu()
        if self.axis_squared_error is None:
            self.axis_squared_error = axis_error
        else:
            self.axis_squared_error += axis_error

    def compute(self) -> dict[str, Any]:
        """Return scalar metrics normalized over valid action elements."""
        if self.valid_values == 0 or self.valid_actions == 0:
            raise RuntimeError("No valid drone actions were present in this epoch.")
        axis_mse = self.axis_squared_error / self.valid_actions
        return {
            "mse": self.squared_error / self.valid_values,
            "mae": self.absolute_error / self.valid_values,
            "zero_action_mse": self.zero_squared_error / self.valid_values,
            "axis_mse": axis_mse.tolist(),
            "valid_actions": self.valid_actions,
        }


def parse_args() -> argparse.Namespace:
    """Parse training and fast data-pipeline diagnostic settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--balanced-sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--overfit-steps",
        type=int,
        default=0,
        help="Repeatedly train one batch this many times, then exit.",
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--no-tensorboard", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    """Seed model initialization and DataLoader sampling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """Choose a requested accelerator or a safe automatic default."""
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def drone_tensors(
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """Move only decentralized actor inputs, targets, and masks to the device."""
    observations = batch["drone"]["observations"]
    global_map = observations["global_map"].to(device, non_blocking=True)
    local_map = observations["local_map"].to(device, non_blocking=True)
    action_history = observations["action_history"].to(device, non_blocking=True)
    actions = batch["drone"]["actions"].to(device, non_blocking=True)

    drone_count = actions.shape[-2]
    agent_mask = batch["agent_mask"][..., -drone_count:].to(
        device, non_blocking=True
    )
    filled = batch["filled"].to(device, non_blocking=True)
    action_mask = agent_mask.bool() & filled.bool()
    return global_map, local_map, action_history, actions, action_mask


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    """Average squared error over valid time/drone/action elements only."""
    mask = action_mask.to(dtype=prediction.dtype).unsqueeze(-1)
    denominator = mask.sum() * target.shape[-1]
    if denominator.item() == 0:
        raise RuntimeError("Batch contains no valid drone actions.")
    return ((prediction - target).square() * mask).sum() / denominator


def run_epoch(
    model: DroneBehaviorCloningPolicy,
    loader,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    grad_clip: float,
    max_batches: int | None,
) -> dict[str, Any]:
    """Run one masked BC training or validation epoch."""
    training = optimizer is not None
    model.train(training)
    metrics = MetricAccumulator()
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        tensors = drone_tensors(batch, device)
        global_map, local_map, action_history, target, action_mask = tensors
        with torch.set_grad_enabled(training):
            prediction = model(global_map, local_map, action_history)
            loss = masked_mse(prediction, target, action_mask)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        metrics.update(prediction.detach(), target, action_mask)
    return metrics.compute()


def build_model(sample: dict[str, Any]) -> DroneBehaviorCloningPolicy:
    """Infer current observation dimensions from persisted data."""
    observations = sample["drone"]["observations"]
    global_shape = observations["global_map"].shape
    local_shape = observations["local_map"].shape
    history_shape = observations["action_history"].shape
    action_dim = sample["drone"]["actions"].shape[-1]
    return DroneBehaviorCloningPolicy(
        global_map_channels=global_shape[-3],
        local_map_channels=local_shape[-3],
        global_map_size=tuple(global_shape[-2:]),
        local_map_size=tuple(local_shape[-2:]),
        action_history_shape=tuple(history_shape[-2:]),
        action_dim=action_dim,
    )


def save_checkpoint(
    path: Path,
    model: DroneBehaviorCloningPolicy,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    metrics: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """Persist enough metadata to reconstruct the offline BC policy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_type": "drone_behavior_cloning",
            "model_config": model.config(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "action_normalization": "divide by per-episode drone_max_speed",
            "args": vars(args),
        },
        path,
    )


def run_overfit_diagnostic(
    model: DroneBehaviorCloningPolicy,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    """Overfit one fixed batch to expose data, mask, or model wiring errors."""
    batch = next(iter(loader))
    global_map, local_map, action_history, target, action_mask = drone_tensors(
        batch, device
    )
    model.train()
    with torch.no_grad():
        initial = float(
            masked_mse(
                model(global_map, local_map, action_history), target, action_mask
            ).item()
        )
    current = initial
    for step in range(1, args.overfit_steps + 1):
        prediction = model(global_map, local_map, action_history)
        loss = masked_mse(prediction, target, action_mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        current = float(loss.item())
        if step == 1 or step % args.log_every == 0 or step == args.overfit_steps:
            print(f"overfit step={step:5d} mse={current:.6f}")

    ratio = current / max(initial, 1e-12)
    print(
        f"Overfit diagnostic: initial_mse={initial:.6f}, "
        f"final_mse={current:.6f}, ratio={ratio:.4f}"
    )
    if current >= initial:
        raise RuntimeError("Overfit MSE did not decrease; inspect the data pipeline.")
    save_checkpoint(
        args.output_dir / "drone_bc_overfit.pt",
        model,
        optimizer,
        epoch=0,
        metrics={"initial_mse": initial, "final_mse": current, "ratio": ratio},
        args=args,
    )


def create_writer(args: argparse.Namespace):
    """Create TensorBoard lazily so diagnostics can run without it."""
    if args.no_tensorboard:
        return None
    from torch.utils.tensorboard import SummaryWriter

    return SummaryWriter(log_dir=args.output_dir / "tensorboard")


def main() -> None:
    """Train the BC baseline or run a one-batch overfit diagnostic."""
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.sequence_length <= 0:
        raise ValueError("epochs, batch-size, and sequence-length must be positive.")
    if args.overfit_steps < 0:
        raise ValueError("overfit-steps cannot be negative.")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    pin_memory = device.type == "cuda"

    train_dataset, train_loader = create_dataloader(
        manifest_path=args.manifest,
        split="source_train",
        data_root=args.data_root,
        sequence_length=args.sequence_length,
        stride=args.stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize_actions=True,
        include_observer=False,
        include_labels=False,
        balanced_sampling=args.balanced_sampling,
        seed=args.seed,
        pin_memory=pin_memory,
    )
    model = build_model(train_dataset[0]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"device={device}, train_windows={len(train_dataset)}, "
        f"parameters={parameter_count:,}"
    )
    print(f"model_config={json.dumps(model.config())}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.overfit_steps:
        run_overfit_diagnostic(model, train_loader, optimizer, device, args)
        return

    val_dataset, val_loader = create_dataloader(
        manifest_path=args.manifest,
        split="source_val",
        data_root=args.data_root,
        sequence_length=args.sequence_length,
        stride=args.stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize_actions=True,
        include_observer=False,
        include_labels=False,
        balanced_sampling=False,
        seed=args.seed,
        pin_memory=pin_memory,
        drop_last_batch=False,
    )
    print(f"val_windows={len(val_dataset)}")
    writer = create_writer(args)
    best_val_mse = float("inf")
    try:
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(
                model,
                train_loader,
                device,
                optimizer=optimizer,
                grad_clip=args.grad_clip,
                max_batches=args.max_train_batches,
            )
            with torch.inference_mode():
                val_metrics = run_epoch(
                    model,
                    val_loader,
                    device,
                    optimizer=None,
                    grad_clip=args.grad_clip,
                    max_batches=args.max_val_batches,
                )
            print(
                f"epoch={epoch:03d} "
                f"train_mse={train_metrics['mse']:.6f} "
                f"val_mse={val_metrics['mse']:.6f} "
                f"val_mae={val_metrics['mae']:.6f} "
                f"zero_mse={val_metrics['zero_action_mse']:.6f} "
                f"axis_mse={[round(value, 6) for value in val_metrics['axis_mse']]}"
            )
            if writer is not None:
                for name, value in train_metrics.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f"train/{name}", value, epoch)
                for name, value in val_metrics.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f"val/{name}", value, epoch)

            combined_metrics = {"train": train_metrics, "validation": val_metrics}
            save_checkpoint(
                args.output_dir / "drone_bc_last.pt",
                model,
                optimizer,
                epoch=epoch,
                metrics=combined_metrics,
                args=args,
            )
            if val_metrics["mse"] < best_val_mse:
                best_val_mse = val_metrics["mse"]
                save_checkpoint(
                    args.output_dir / "drone_bc_best.pt",
                    model,
                    optimizer,
                    epoch=epoch,
                    metrics=combined_metrics,
                    args=args,
                )
    finally:
        if writer is not None:
            writer.close()
    print(
        f"Best validation MSE: {best_val_mse:.6f} "
        f"({args.output_dir / 'drone_bc_best.pt'})"
    )


if __name__ == "__main__":
    main()

