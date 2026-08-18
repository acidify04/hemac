"""Train drone-only HiSSD on HeMAC source difficulties 1-3.

The optimization order follows the official HiSSD learner: low-level
controller plus MoCo task discrimination, expectile value learning, then an
advantage-weighted high-level forward-prediction planner update.
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
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from skill_discovery.dataset import DEFAULT_MANIFEST_PATH, create_dataloader
from skill_discovery.hissd_models import HeMACHISSD


DEFAULT_BC_CHECKPOINT = (
    PROJECT_ROOT / "src/skill_discovery/bc_checkpoints/drone_bc_best.pt"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "src/skill_discovery/hissd_checkpoints"


def parse_args() -> argparse.Namespace:
    """Parse offline training, HiSSD, and diagnostic settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--bc-checkpoint", type=Path, default=DEFAULT_BC_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--skill-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--transformer-heads", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--expectile", type=float, default=0.9)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--target-tau", type=float, default=0.005)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--max-contrastive-samples", type=int, default=256)
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=100.0,
        help="Divide team rewards by this value before IQL updates.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--no-tensorboard", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    """Seed all stochastic sampling used by the offline learner."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """Choose an available training device."""
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def move_observations(
    observations: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Move decentralized actor inputs while excluding central_vector."""
    return {
        name: observations[name].to(device, non_blocking=True)
        for name in ("global_map", "local_map", "action_history")
    }


def prepare_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Extract the tensors needed by the drone-only HiSSD objectives."""
    actions = batch["drone"]["actions"].to(device, non_blocking=True)
    drone_count = actions.shape[-2]
    agent_mask = batch["agent_mask"][..., -drone_count:].to(
        device, non_blocking=True
    )
    filled = batch["filled"].squeeze(-1).to(device, non_blocking=True).bool()
    valid_agents = agent_mask.bool() & filled.unsqueeze(-1)
    terminated = batch["terminated"].squeeze(-1).to(device, non_blocking=True)
    truncated = batch["truncated"].squeeze(-1).to(device, non_blocking=True)
    return {
        "observations": move_observations(
            batch["drone"]["observations"], device
        ),
        "next_observations": move_observations(
            batch["drone"]["next_observations"], device
        ),
        "actions": actions,
        "central_map": batch["global_state"]["central_map"].to(
            device, non_blocking=True
        ),
        "next_central_map": batch["next_global_state"]["central_map"].to(
            device, non_blocking=True
        ),
        "team_reward": batch["team_reward"].squeeze(-1).to(
            device, non_blocking=True
        ),
        "task_id": batch["task_id"].to(device, non_blocking=True),
        "valid_agents": valid_agents,
        "valid_steps": filled,
        "done": (terminated.bool() | truncated.bool()),
    }


def masked_action_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_agents: torch.Tensor,
) -> torch.Tensor:
    """Average continuous reconstruction loss over valid drone actions."""
    mask = valid_agents.unsqueeze(-1).to(prediction.dtype)
    denominator = mask.sum() * target.shape[-1]
    return ((prediction - target).square() * mask).sum() / denominator.clamp_min(1.0)


def moco_contrastive_loss(
    query: torch.Tensor,
    momentum_key: torch.Tensor,
    task_id: torch.Tensor,
    valid_agents: torch.Tensor,
    *,
    temperature: float,
    max_samples: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Use another same-task drone as positive and other tasks as negatives."""
    positive_key = torch.roll(momentum_key, shifts=1, dims=2)
    expanded_tasks = task_id[:, None, None].expand(valid_agents.shape)
    flat_valid = valid_agents.reshape(-1)
    queries = query.reshape(-1, query.shape[-1])[flat_valid]
    positives = positive_key.reshape(-1, positive_key.shape[-1])[flat_valid]
    keys = momentum_key.reshape(-1, momentum_key.shape[-1])[flat_valid]
    tasks = expanded_tasks.reshape(-1)[flat_valid]
    if queries.shape[0] > max_samples:
        indices = torch.randperm(queries.shape[0], device=queries.device)[:max_samples]
        queries = queries[indices]
        positives = positives[indices]
        keys = keys[indices]
        tasks = tasks[indices]
    if queries.shape[0] == 0 or torch.unique(tasks).numel() < 2:
        zero = query.sum() * 0.0
        return zero, zero.detach(), 0

    positive_logits = (queries * positives).sum(dim=-1, keepdim=True) / temperature
    negative_logits = queries @ keys.transpose(0, 1) / temperature
    different_task = tasks[:, None] != tasks[None, :]
    negative_logits = negative_logits.masked_fill(~different_task, float("-inf"))
    logits = torch.cat((positive_logits, negative_logits), dim=1)
    loss = (-positive_logits.squeeze(1) + torch.logsumexp(logits, dim=1)).mean()
    max_negative = negative_logits.max(dim=1).values
    accuracy = (positive_logits.squeeze(1) > max_negative).float().mean()
    return loss, accuracy, queries.shape[0]


def skill_standard_deviation(
    skills: torch.Tensor,
    valid_agents: torch.Tensor,
) -> torch.Tensor:
    """Measure latent spread to expose representation collapse."""
    valid_skills = skills[valid_agents]
    if valid_skills.shape[0] < 2:
        return skills.new_zeros(())
    return valid_skills.std(dim=0, unbiased=False).mean()


def controller_objective(
    model: HeMACHISSD,
    batch: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Optimize official Eq. 11 with continuous action reconstruction."""
    features = model.encode_observations(batch["observations"])
    common, task_skill, query = model.infer_skills(
        features, batch["valid_agents"]
    )
    prediction = model.decode_actions(features, common, task_skill)
    action_loss = masked_action_mse(
        prediction, batch["actions"], batch["valid_agents"]
    )
    with torch.no_grad():
        target_features = model.encode_observations(
            batch["observations"], target=True
        )
        _, momentum_key = model.target_task_skill_encoder(
            target_features, batch["valid_agents"]
        )
    contrastive_loss, contrastive_accuracy, sample_count = moco_contrastive_loss(
        query,
        momentum_key,
        batch["task_id"],
        batch["valid_agents"],
        temperature=args.contrastive_temperature,
        max_samples=args.max_contrastive_samples,
    )
    total = action_loss + args.beta * contrastive_loss
    metrics = {
        "controller_loss": float(total.detach()),
        "action_mse": float(action_loss.detach()),
        "contrastive_loss": float(contrastive_loss.detach()),
        "contrastive_accuracy": float(contrastive_accuracy.detach()),
        "contrastive_samples": float(sample_count),
        "common_skill_std": float(
            skill_standard_deviation(common, batch["valid_agents"]).detach()
        ),
        "task_skill_std": float(
            skill_standard_deviation(task_skill, batch["valid_agents"]).detach()
        ),
    }
    return total, metrics


def value_predictions(
    model: HeMACHISSD,
    batch: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute current value, target, and TD residual for Eq. 5."""
    features = model.encode_observations(batch["observations"])
    value = model.total_value(
        features,
        batch["central_map"],
        batch["valid_agents"].to(features.dtype),
    ).squeeze(-1)
    with torch.no_grad():
        next_features = model.encode_observations(
            batch["next_observations"], target=True
        )
        target_next_value = model.total_value(
            next_features,
            batch["next_central_map"],
            batch["valid_agents"].to(next_features.dtype),
            target=True,
        ).squeeze(-1)
        scaled_reward = batch["team_reward"] / args.reward_scale
        target = scaled_reward + args.gamma * (~batch["done"]).float() * target_next_value
    return value, target, target - value


def value_objective(
    model: HeMACHISSD,
    batch: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train the centralized value with the official expectile objective."""
    value, target, residual = value_predictions(model, batch, args)
    expectile_weight = torch.abs(
        args.expectile - (residual.detach() < 0).to(residual.dtype)
    )
    mask = batch["valid_steps"].to(residual.dtype)
    loss = (expectile_weight * residual.square() * mask).sum() / mask.sum().clamp_min(1.0)
    metrics = {
        "value_loss": float(loss.detach()),
        "value_mean": float((value.detach() * mask).sum() / mask.sum().clamp_min(1.0)),
        "target_value_mean": float(
            (target.detach() * mask).sum() / mask.sum().clamp_min(1.0)
        ),
        "td_residual_mean": float(
            (residual.detach() * mask).sum() / mask.sum().clamp_min(1.0)
        ),
    }
    return loss, metrics


def planner_objective(
    model: HeMACHISSD,
    batch: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Optimize Eq. 7 using global and local one-step prediction errors."""
    with torch.no_grad():
        features = model.encode_observations(batch["observations"]).detach()
        target_next_features = model.encode_observations(
            batch["next_observations"], target=True
        )

    common = model.common_skill_encoder(features, batch["valid_agents"])
    predicted_central, predicted_local = model.forward_predictor(
        common, batch["valid_agents"].to(features.dtype)
    )
    with torch.no_grad():
        current_value = model.total_value(
            features,
            batch["central_map"],
            batch["valid_agents"].to(features.dtype),
        ).squeeze(-1)
        predicted_next_value = model.total_value(
            predicted_local.detach(),
            predicted_central.detach(),
            batch["valid_agents"].to(features.dtype),
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
    local_error_by_agent = (
        predicted_local - target_next_features.detach()
    ).square().mean(dim=-1)
    agent_mask = batch["valid_agents"].to(local_error_by_agent.dtype)
    local_error = (local_error_by_agent * agent_mask).sum(dim=2) / agent_mask.sum(
        dim=2
    ).clamp_min(1.0)
    prediction_error = central_error + local_error
    step_mask = batch["valid_steps"].to(prediction_error.dtype)
    weighted_error = prediction_error * advantage_weight.detach() * step_mask
    loss = weighted_error.sum() / step_mask.sum().clamp_min(1.0)
    metrics = {
        "planner_loss": float(loss.detach()),
        "central_prediction_mse": float(
            (central_error.detach() * step_mask).sum()
            / step_mask.sum().clamp_min(1.0)
        ),
        "local_prediction_mse": float(
            (local_error.detach() * step_mask).sum()
            / step_mask.sum().clamp_min(1.0)
        ),
        "advantage_weight_mean": float(
            (advantage_weight.detach() * step_mask).sum()
            / step_mask.sum().clamp_min(1.0)
        ),
        "advantage_weight_max": float(advantage_weight.detach().max()),
        "planner_predicted_value_mean": float(
            (predicted_next_value * step_mask).sum()
            / step_mask.sum().clamp_min(1.0)
        ),
        "planner_td_residual_mean": float(
            (residual * step_mask).sum() / step_mask.sum().clamp_min(1.0)
        ),
    }
    return loss, metrics


def optimize(
    loss: torch.Tensor,
    model: HeMACHISSD,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
) -> float:
    """Apply one of the three sequential official HiSSD updates."""
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite HiSSD loss: {float(loss.detach())}")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        grad_clip,
    )
    optimizer.step()
    return float(grad_norm)


def accumulate_metrics(
    accumulator: dict[str, float],
    metrics: dict[str, float],
) -> None:
    """Accumulate batch-level diagnostics for compact epoch reporting."""
    for name, value in metrics.items():
        accumulator[name] += float(value)


def average_metrics(
    accumulator: dict[str, float],
    batch_count: int,
) -> dict[str, float]:
    """Average accumulated diagnostics over processed batches."""
    if batch_count == 0:
        raise RuntimeError("No batches were processed.")
    return {name: value / batch_count for name, value in accumulator.items()}


def run_train_epoch(
    model: HeMACHISSD,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    """Run controller, value, and planner updates in official order."""
    model.train()
    accumulator: dict[str, float] = defaultdict(float)
    batch_count = 0
    for batch_index, raw_batch in enumerate(loader):
        if args.max_train_batches is not None and batch_index >= args.max_train_batches:
            break
        batch = prepare_batch(raw_batch, device)

        controller_loss, controller_metrics = controller_objective(model, batch, args)
        controller_metrics["controller_grad_norm"] = optimize(
            controller_loss, model, optimizer, args.grad_clip
        )

        value_loss, value_metrics = value_objective(model, batch, args)
        value_metrics["value_grad_norm"] = optimize(
            value_loss, model, optimizer, args.grad_clip
        )

        planner_loss, planner_metrics = planner_objective(model, batch, args)
        planner_metrics["planner_grad_norm"] = optimize(
            planner_loss, model, optimizer, args.grad_clip
        )
        model.update_targets(args.target_tau)

        accumulate_metrics(accumulator, controller_metrics)
        accumulate_metrics(accumulator, value_metrics)
        accumulate_metrics(accumulator, planner_metrics)
        batch_count += 1
    return average_metrics(accumulator, batch_count)


@torch.inference_mode()
def run_validation(
    model: HeMACHISSD,
    loader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    """Evaluate all three objectives without updating source models."""
    model.eval()
    accumulator: dict[str, float] = defaultdict(float)
    batch_count = 0
    for batch_index, raw_batch in enumerate(loader):
        if args.max_val_batches is not None and batch_index >= args.max_val_batches:
            break
        batch = prepare_batch(raw_batch, device)
        _, controller_metrics = controller_objective(model, batch, args)
        _, value_metrics = value_objective(model, batch, args)
        _, planner_metrics = planner_objective(model, batch, args)
        accumulate_metrics(accumulator, controller_metrics)
        accumulate_metrics(accumulator, value_metrics)
        accumulate_metrics(accumulator, planner_metrics)
        batch_count += 1
    return average_metrics(accumulator, batch_count)


def build_model(sample: dict[str, Any], args: argparse.Namespace) -> HeMACHISSD:
    """Infer current HeMAC observation dimensions from one source window."""
    observations = sample["drone"]["observations"]
    global_shape = observations["global_map"].shape
    local_shape = observations["local_map"].shape
    history_shape = observations["action_history"].shape
    central_shape = sample["global_state"]["central_map"].shape
    action_shape = sample["drone"]["actions"].shape
    return HeMACHISSD(
        global_map_channels=global_shape[-3],
        local_map_channels=local_shape[-3],
        central_map_channels=central_shape[-3],
        agent_count=action_shape[-2],
        action_dim=action_shape[-1],
        global_map_size=tuple(global_shape[-2:]),
        local_map_size=tuple(local_shape[-2:]),
        central_map_size=tuple(central_shape[-2:]),
        action_history_shape=tuple(history_shape[-2:]),
        hidden_dim=args.hidden_dim,
        skill_dim=args.skill_dim,
        transformer_heads=args.transformer_heads,
    )


@torch.inference_mode()
def verify_bc_initialization(
    model: HeMACHISSD,
    sample: dict[str, Any],
    device: torch.device,
) -> float:
    """Verify zero skill residual leaves the validated BC action unchanged."""
    batch = prepare_batch(sample, device)
    features = model.encode_observations(batch["observations"])
    common, task_skill, _ = model.infer_skills(features, batch["valid_agents"])
    hissd_action = model.decode_actions(features, common, task_skill)
    bc_action = torch.tanh(model.action_decoder.base_action_head(features))
    return float((hissd_action - bc_action).abs().max())


def save_checkpoint(
    path: Path,
    model: HeMACHISSD,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    train_metrics: dict[str, float],
    validation_metrics: dict[str, float],
    args: argparse.Namespace,
    bc_info: dict[str, Any],
) -> None:
    """Save the complete offline model and reproducibility metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_type": "hemac_drone_hissd",
            "official_structure": (
                "controller+MoCo, expectile value, advantage-weighted planner"
            ),
            "model_config": model.config(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "bc_initialization": bc_info,
            "hyperparameters": vars(args),
            "training_tasks": [1, 2, 3],
            "held_out_tasks": [4, 5, 6],
        },
        path,
    )


def create_writer(args: argparse.Namespace):
    """Create TensorBoard logging only when requested dependencies are present."""
    if args.no_tensorboard:
        return None
    from torch.utils.tensorboard import SummaryWriter

    return SummaryWriter(log_dir=args.output_dir / "tensorboard")


def validate_args(args: argparse.Namespace) -> None:
    """Reject unstable or nonsensical HiSSD settings early."""
    positive_names = (
        "epochs",
        "batch_size",
        "sequence_length",
        "learning_rate",
        "grad_clip",
        "skill_dim",
        "hidden_dim",
        "transformer_heads",
        "gamma",
        "alpha",
        "target_tau",
        "contrastive_temperature",
        "max_contrastive_samples",
        "reward_scale",
    )
    for name in positive_names:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if not 0 < args.expectile < 1:
        raise ValueError("--expectile must be between 0 and 1.")
    if args.beta < 0:
        raise ValueError("--beta cannot be negative.")
    if args.hidden_dim % args.transformer_heads != 0:
        raise ValueError("hidden-dim must be divisible by transformer-heads.")


def main() -> None:
    """Train HiSSD from the validated source-task BC initialization."""
    args = parse_args()
    validate_args(args)
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
        balanced_sampling=True,
        seed=args.seed,
        pin_memory=pin_memory,
    )
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
        balanced_sampling=True,
        seed=args.seed,
        pin_memory=pin_memory,
        drop_last_batch=False,
    )
    source_tasks = {int(entry["difficulty"]) for entry in train_dataset.entries}
    if not source_tasks.issubset({1, 2, 3}):
        raise ValueError(f"source_train contains held-out task data: {source_tasks}")

    model = build_model(train_dataset[0], args)
    bc_info = model.initialize_from_bc(args.bc_checkpoint)
    model.to(device)
    sample_loader = create_dataloader(
        manifest_path=args.manifest,
        split="source_train",
        data_root=args.data_root,
        sequence_length=min(args.sequence_length, 4),
        batch_size=1,
        num_workers=0,
        normalize_actions=True,
        include_observer=False,
        include_labels=False,
        balanced_sampling=False,
        seed=args.seed,
        pin_memory=False,
        drop_last_batch=False,
    )[1]
    initialization_error = verify_bc_initialization(
        model, next(iter(sample_loader)), device
    )
    if initialization_error > 1e-6:
        raise RuntimeError(
            f"HiSSD decoder does not preserve BC actions: {initialization_error}"
        )

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"device={device}, train_windows={len(train_dataset)}, "
        f"val_windows={len(val_dataset)}, "
        f"trainable_parameters={sum(p.numel() for p in trainable_parameters):,}"
    )
    print(f"model_config={json.dumps(model.config())}")
    print(
        f"BC initialization epoch={bc_info.get('epoch')}, "
        f"max_action_difference={initialization_error:.3e}"
    )

    writer = create_writer(args)
    best_validation_loss = math.inf
    try:
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_train_epoch(
                model, train_loader, optimizer, device, args
            )
            validation_metrics = run_validation(model, val_loader, device, args)
            validation_loss = (
                validation_metrics["action_mse"]
                + args.beta * validation_metrics["contrastive_loss"]
                + validation_metrics["value_loss"]
                + validation_metrics["planner_loss"]
            )
            print(
                f"epoch={epoch:03d} "
                f"action={train_metrics['action_mse']:.5f}/"
                f"{validation_metrics['action_mse']:.5f} "
                f"contrast={train_metrics['contrastive_loss']:.5f}/"
                f"{validation_metrics['contrastive_loss']:.5f} "
                f"contrast_acc={validation_metrics['contrastive_accuracy']:.3f} "
                f"value={train_metrics['value_loss']:.5f}/"
                f"{validation_metrics['value_loss']:.5f} "
                f"planner={train_metrics['planner_loss']:.5f}/"
                f"{validation_metrics['planner_loss']:.5f} "
                f"c_std={validation_metrics['common_skill_std']:.4f} "
                f"z_std={validation_metrics['task_skill_std']:.4f}"
            )
            if writer is not None:
                for name, value in train_metrics.items():
                    writer.add_scalar(f"train/{name}", value, epoch)
                for name, value in validation_metrics.items():
                    writer.add_scalar(f"validation/{name}", value, epoch)
                writer.add_scalar("validation/combined_loss", validation_loss, epoch)

            save_checkpoint(
                args.output_dir / "hissd_last.pt",
                model,
                optimizer,
                epoch=epoch,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                args=args,
                bc_info=bc_info,
            )
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                save_checkpoint(
                    args.output_dir / "hissd_best.pt",
                    model,
                    optimizer,
                    epoch=epoch,
                    train_metrics=train_metrics,
                    validation_metrics=validation_metrics,
                    args=args,
                    bc_info=bc_info,
                )
    finally:
        if writer is not None:
            writer.close()
    print(
        f"Best combined validation loss: {best_validation_loss:.6f} "
        f"({args.output_dir / 'hissd_best.pt'})"
    )


if __name__ == "__main__":
    main()
