"""Train drone-only HiSSD on the source difficulties in a dataset manifest.

The optimization order follows the official HiSSD learner: low-level
controller plus continuous task-context learning, expectile value learning,
then an advantage-weighted high-level forward-prediction planner update.
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
from skill_discovery.task_descriptor import REALIZED_TASK_DESCRIPTOR_NAMES


DEFAULT_BC_CHECKPOINT = (
    PROJECT_ROOT
    / "src/skill_discovery/checkpoints/bc_checkpoints/drone_bc_best.pt"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "src/skill_discovery/checkpoints/hissd_checkpoints"
)
ABLATION_CHOICES = (
    "full",
    "no_descriptor",
    "no_task_contrast",
    "no_task_auxiliary",
)


def parse_args() -> argparse.Namespace:
    """Parse offline training, HiSSD, and diagnostic settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--bc-checkpoint", type=Path, default=DEFAULT_BC_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--ablation",
        choices=ABLATION_CHOICES,
        default="full",
        help=(
            "Task-skill auxiliary objective to remove. Ablation runs use a "
            "separate default output directory."
        ),
    )
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=128,
        help="Causal task-context length; 128 covers nearly all HeMAC episodes.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=128,
        help="Window stride; use the full context by default without overlap.",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--task-learning-rate-multiplier",
        type=float,
        default=5.0,
        help="Learning-rate multiplier for the task encoder and descriptor head.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--task-weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--skill-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--transformer-heads", type=int, default=1)
    parser.add_argument(
        "--descriptor-weight",
        type=float,
        default=1.0,
        help="Weight for variance-standardized task descriptor regression.",
    )
    parser.add_argument(
        "--descriptor-metric-weight",
        type=float,
        default=0.25,
        help="Weight for matching latent distances to descriptor distances.",
    )
    parser.add_argument(
        "--task-contrastive-weight",
        type=float,
        default=0.1,
        help="Secondary weight for source difficulty discrimination.",
    )
    parser.add_argument(
        "--descriptor-scale-floor",
        type=float,
        default=0.03,
        help="Minimum source standard deviation used to scale descriptor errors.",
    )
    parser.add_argument(
        "--task-variance-weight",
        type=float,
        default=2.0,
        help="Weight preventing the episode-level task context from collapsing.",
    )
    parser.add_argument(
        "--task-variance-target",
        type=float,
        default=0.05,
        help="Minimum per-dimension standard deviation for task contexts.",
    )
    parser.add_argument(
        "--task-action-descriptor-weight",
        type=float,
        default=0.5,
        help=(
            "Weight for decoding the realized descriptor from the task skill "
            "that is passed to the action decoder."
        ),
    )
    parser.add_argument(
        "--task-action-variance-weight",
        type=float,
        default=2.0,
        help="Weight preventing the action-facing task skill from collapsing.",
    )
    parser.add_argument(
        "--task-action-contrastive-weight",
        type=float,
        default=0.1,
        help="Weight for source-task discrimination directly from action task skills.",
    )
    parser.add_argument(
        "--task-warmup-epochs",
        type=int,
        default=10,
        help="Train only task representation objectives before joint HiSSD updates.",
    )
    parser.add_argument(
        "--detach-task-skill-for-action",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep action reconstruction gradients from overriding the task "
            "representation objectives."
        ),
    )
    parser.add_argument(
        "--task-contrastive-temperature",
        type=float,
        default=0.15,
    )
    parser.add_argument("--task-label-smoothing", type=float, default=0.05)
    parser.add_argument("--task-dropout", type=float, default=0.0)
    parser.add_argument(
        "--task-contrastive-tail-steps",
        type=int,
        default=8,
        help=(
            "Supervise only the last N valid contexts, after enough of the "
            "episode has been observed to infer its task dynamics."
        ),
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--expectile", type=float, default=0.9)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--target-tau", type=float, default=0.005)
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
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=20,
        help="Stop after this many epochs without task-validation improvement; 0 disables.",
    )
    parser.add_argument("--no-tensorboard", action="store_true")
    args = parser.parse_args()
    configure_ablation(args)
    return args


def configure_ablation(args: argparse.Namespace) -> None:
    """Resolve one named ablation into model and loss settings."""
    args.descriptor_enabled = args.ablation not in {
        "no_descriptor",
        "no_task_auxiliary",
    }
    args.task_contrastive_enabled = args.ablation not in {
        "no_task_contrast",
        "no_task_auxiliary",
    }
    if not args.descriptor_enabled:
        args.descriptor_weight = 0.0
        args.descriptor_metric_weight = 0.0
        args.task_action_descriptor_weight = 0.0
    if not args.task_contrastive_enabled:
        args.task_contrastive_weight = 0.0
        args.task_action_contrastive_weight = 0.0
    if not args.descriptor_enabled and not args.task_contrastive_enabled:
        args.task_variance_weight = 0.0
        args.task_action_variance_weight = 0.0
    if args.ablation != "full" and args.output_dir == DEFAULT_OUTPUT_DIR:
        args.output_dir = DEFAULT_OUTPUT_DIR.parent / (
            f"{DEFAULT_OUTPUT_DIR.name}_{args.ablation}"
        )


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
        "team_reward": batch["drone_task_reward"].squeeze(-1).to(
            device, non_blocking=True
        ),
        "task_descriptor": batch["task_descriptor"].to(
            device, non_blocking=True
        ),
        "task_descriptor_available": batch["task_descriptor_available"].to(
            device, non_blocking=True
        ).bool(),
        "task_id": batch["task_id"].to(device, non_blocking=True),
        "task_supervision": (
            batch["window_start"].to(device, non_blocking=True) == 0
        ),
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


def skill_standard_deviation(
    skills: torch.Tensor,
    valid_agents: torch.Tensor,
) -> torch.Tensor:
    """Measure latent spread to expose representation collapse."""
    valid_skills = skills[valid_agents]
    if valid_skills.shape[0] < 2:
        return skills.new_zeros(())
    return valid_skills.std(dim=0, unbiased=False).mean()


def task_tail_mask(
    valid_agents: torch.Tensor,
    task_supervision: torch.Tensor,
    tail_steps: int,
) -> torch.Tensor:
    """Select late causal steps after sufficient task dynamics were observed."""
    valid_steps = valid_agents.any(dim=2)
    step_number = valid_steps.long().cumsum(dim=1)
    valid_count = valid_steps.long().sum(dim=1, keepdim=True)
    tail_start = (valid_count - tail_steps).clamp_min(0)
    return (
        valid_steps
        & (step_number > tail_start)
        & task_supervision.unsqueeze(1)
    )


def standardized_descriptor_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    """Apply equal relative pressure to low- and high-variance components."""
    scale = torch.as_tensor(
        getattr(args, "task_descriptor_scale", [1.0] * target.shape[-1]),
        device=target.device,
        dtype=target.dtype,
    )
    if scale.numel() != target.shape[-1]:
        raise ValueError(
            f"Descriptor scale has {scale.numel()} values, expected {target.shape[-1]}."
        )
    normalized_error = (prediction - target) / scale
    return nn.functional.smooth_l1_loss(
        normalized_error,
        torch.zeros_like(normalized_error),
    )


def continuous_task_objective(
    model: HeMACHISSD,
    contrastive_skills: torch.Tensor,
    batch: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Regress realized episode dynamics and preserve latent geometry."""
    if model.task_descriptor_head is None:
        zero = contrastive_skills.sum() * 0.0
        return zero, zero, zero, empty_descriptor_metrics(args, 0)

    prediction, valid_windows = model.predict_task_descriptor(
        contrastive_skills, batch["valid_agents"]
    )
    available = (
        batch["task_descriptor_available"]
        & valid_windows
        & batch["task_supervision"]
    )
    if not available.any():
        zero = contrastive_skills.sum() * 0.0
        return zero, zero, zero, empty_descriptor_metrics(
            args, model.task_descriptor_dim
        )

    sequence_prediction, sequence_valid = model.predict_task_descriptor_sequence(
        contrastive_skills, batch["valid_agents"]
    )
    sequence_supervision = (
        task_tail_mask(
            batch["valid_agents"],
            batch["task_supervision"],
            args.task_contrastive_tail_steps,
        )
        & sequence_valid
        & batch["task_descriptor_available"].unsqueeze(1)
    )
    sequence_target = batch["task_descriptor"].unsqueeze(1).expand_as(
        sequence_prediction
    )
    descriptor_loss = standardized_descriptor_loss(
        sequence_prediction[sequence_supervision],
        sequence_target[sequence_supervision],
        args,
    )
    target = batch["task_descriptor"][available]
    predicted = prediction[available]
    descriptor_mae = (predicted - target).abs().mean()
    median_baseline = target.median(dim=0).values
    median_baseline_mae = (target - median_baseline).abs().mean()

    context, _ = model.pool_task_context(
        contrastive_skills, batch["valid_agents"]
    )
    context = nn.functional.normalize(context[available], dim=-1)
    if context.shape[0] < 2:
        metric_loss = descriptor_loss.new_zeros(())
        distance_correlation = descriptor_loss.new_zeros(())
    else:
        descriptor_scale = torch.as_tensor(
            getattr(args, "task_descriptor_scale", [1.0] * target.shape[-1]),
            device=target.device,
            dtype=target.dtype,
        )
        standardized_target = target / descriptor_scale
        descriptor_distance = torch.cdist(
            standardized_target, standardized_target
        ) / math.sqrt(
            target.shape[-1]
        )
        latent_distance = 1.0 - context @ context.transpose(0, 1)
        pair_mask = torch.triu(
            torch.ones(
                context.shape[0],
                context.shape[0],
                dtype=torch.bool,
                device=context.device,
            ),
            diagonal=1,
        )
        latent_distance = latent_distance[pair_mask]
        target_distance = descriptor_distance[pair_mask]
        centered_latent = latent_distance - latent_distance.mean()
        centered_target = target_distance - target_distance.mean()
        latent_variance = centered_latent.square().sum()
        target_variance = centered_target.square().sum()
        if float(target_variance.detach()) <= 1e-8:
            # A single-task batch has no descriptor-distance ordering to learn.
            # Skipping it also avoids sqrt(0) producing NaN gradients.
            distance_correlation = descriptor_loss.new_zeros(())
            metric_loss = contrastive_skills.sum() * 0.0
        else:
            denominator = (
                latent_variance.clamp_min(1e-8).sqrt()
                * target_variance.clamp_min(1e-8).sqrt()
            )
            distance_correlation = (
                centered_latent * centered_target
            ).sum() / denominator
            # Directly optimize geometry. The previous raw-similarity MSE admitted
            # a low-loss constant-distance solution with near-zero correlation.
            metric_loss = 1.0 - distance_correlation

    if context.shape[0] < 2:
        variance_loss = context.sum() * 0.0
    else:
        context_std_by_dimension = context.std(dim=0, unbiased=False)
        variance_loss = torch.relu(
            args.task_variance_target - context_std_by_dimension
        ).mean()

    prediction_std = predicted.std(dim=0, unbiased=False).mean()
    target_std = target.std(dim=0, unbiased=False).mean()
    metrics = {
        "descriptor_loss": float(descriptor_loss.detach()),
        "descriptor_mae": float(descriptor_mae.detach()),
        "descriptor_metric_loss": float(metric_loss.detach()),
        "task_variance_loss": float(variance_loss.detach()),
        "descriptor_distance_correlation": float(distance_correlation.detach()),
        "descriptor_samples": float(available.sum()),
        "descriptor_training_samples": float(sequence_supervision.sum()),
        "descriptor_prediction_std": float(prediction_std.detach()),
        "descriptor_target_std": float(target_std.detach()),
        "descriptor_std_ratio": float(
            (prediction_std / target_std.clamp_min(1e-8)).detach()
        ),
        "descriptor_median_baseline_mae": float(
            median_baseline_mae.detach()
        ),
        "descriptor_mae_gain": float(
            (median_baseline_mae - descriptor_mae).detach()
        ),
        "task_context_std": float(
            context.std(dim=0, unbiased=False).mean().detach()
        ),
    }
    descriptor_names = resolved_descriptor_names(args, predicted.shape[-1])
    for index, name in enumerate(descriptor_names):
        metrics[f"descriptor_mae/{name}"] = float(
            (predicted[:, index] - target[:, index]).abs().mean().detach()
        )
    return descriptor_loss, metric_loss, variance_loss, metrics


def task_action_skill_objective(
    model: HeMACHISSD,
    task_skills: torch.Tensor,
    batch: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Keep the action-facing task skill informative and non-collapsed."""
    if model.task_descriptor_head is None and model.task_prior_count <= 0:
        zero = task_skills.sum() * 0.0
        return zero, zero, {
            "task_action_descriptor_loss": 0.0,
            "task_action_descriptor_mae": 0.0,
            "task_action_variance_loss": 0.0,
            "task_action_context_std": 0.0,
        }

    numeric_mask = batch["valid_agents"].unsqueeze(-1).to(task_skills.dtype)
    sequence_context = (task_skills * numeric_mask).sum(dim=2) / numeric_mask.sum(
        dim=2
    ).clamp_min(1.0)
    sequence_valid = batch["valid_agents"].any(dim=2)
    sequence_supervision = (
        task_tail_mask(
            batch["valid_agents"],
            batch["task_supervision"],
            args.task_contrastive_tail_steps,
        )
        & sequence_valid
        & batch["task_descriptor_available"].unsqueeze(1)
    )

    if model.task_descriptor_head is not None and sequence_supervision.any():
        prediction = model.task_descriptor_head(
            model.task_descriptor_dropout(sequence_context)
        )
        target = batch["task_descriptor"].unsqueeze(1).expand_as(prediction)
        descriptor_loss = standardized_descriptor_loss(
            prediction[sequence_supervision],
            target[sequence_supervision],
            args,
        )
        descriptor_mae = (
            prediction[sequence_supervision] - target[sequence_supervision]
        ).abs().mean()
    else:
        descriptor_loss = task_skills.sum() * 0.0
        descriptor_mae = descriptor_loss.detach()

    context, valid_windows = model.pool_task_context(
        task_skills, batch["valid_agents"]
    )
    available = valid_windows & batch["task_supervision"]
    available_context = context[available]
    if available_context.shape[0] < 2:
        variance_loss = task_skills.sum() * 0.0
        context_std = variance_loss.detach()
    else:
        context_std_by_dimension = available_context.std(dim=0, unbiased=False)
        variance_loss = torch.relu(
            args.task_variance_target - context_std_by_dimension
        ).mean()
        context_std = context_std_by_dimension.mean()

    return descriptor_loss, variance_loss, {
        "task_action_descriptor_loss": float(descriptor_loss.detach()),
        "task_action_descriptor_mae": float(descriptor_mae.detach()),
        "task_action_variance_loss": float(variance_loss.detach()),
        "task_action_context_std": float(context_std.detach()),
    }


def resolved_descriptor_names(
    args: argparse.Namespace,
    descriptor_dim: int,
) -> tuple[str, ...]:
    """Return the manifest descriptor schema used by all agent roles."""
    names = tuple(
        getattr(args, "task_descriptor_names", REALIZED_TASK_DESCRIPTOR_NAMES)
    )
    if descriptor_dim == 0:
        return names
    if len(names) != descriptor_dim:
        raise ValueError(
            "Task descriptor schema does not match the training tensor: "
            f"names={len(names)}, tensor={descriptor_dim}."
        )
    return names


def empty_descriptor_metrics(
    args: argparse.Namespace,
    descriptor_dim: int,
) -> dict[str, float]:
    """Return stable zero-valued logging fields for descriptor ablations."""
    metrics = {
        "descriptor_loss": 0.0,
        "descriptor_mae": 0.0,
        "descriptor_metric_loss": 0.0,
        "task_variance_loss": 0.0,
        "descriptor_distance_correlation": 0.0,
        "descriptor_samples": 0.0,
        "descriptor_training_samples": 0.0,
        "descriptor_prediction_std": 0.0,
        "descriptor_target_std": 0.0,
        "descriptor_std_ratio": 0.0,
        "descriptor_median_baseline_mae": 0.0,
        "descriptor_mae_gain": 0.0,
        "task_context_std": 0.0,
    }
    metrics.update(
        {
            f"descriptor_mae/{name}": 0.0
            for name in resolved_descriptor_names(args, descriptor_dim)
        }
    )
    return metrics


def task_contrastive_objective(
    model: HeMACHISSD,
    contrastive_skills: torch.Tensor,
    batch: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Match every valid causal context to its fixed source-task prior."""
    if model.task_prior_count <= 0:
        zero = contrastive_skills.sum() * 0.0
        return zero, {
            "task_contrastive_loss": 0.0,
            "task_contrastive_accuracy": 0.0,
            "task_contrastive_chance": 0.0,
            "task_contrastive_samples": 0.0,
            "task_contrastive_confidence": 0.0,
            "task_contrastive_entropy": 0.0,
        }

    logits = model.task_prior_sequence_logits(
        contrastive_skills,
        args.task_contrastive_temperature,
    )
    in_task_tail = task_tail_mask(
        batch["valid_agents"],
        batch["task_supervision"],
        args.task_contrastive_tail_steps,
    )
    supervised = (
        batch["valid_agents"]
        & in_task_tail.unsqueeze(-1)
    )
    task_ids = batch["task_id"][:, None, None].expand_as(supervised)[supervised]
    logits = logits[supervised]
    if logits.shape[0] == 0:
        zero = contrastive_skills.sum() * 0.0
        return zero, {
            "task_contrastive_loss": 0.0,
            "task_contrastive_accuracy": 0.0,
            "task_contrastive_chance": 1.0 / model.task_prior_count,
            "task_contrastive_samples": 0.0,
            "task_contrastive_confidence": 0.0,
            "task_contrastive_entropy": 0.0,
        }
    loss = nn.functional.cross_entropy(
        logits,
        task_ids,
        label_smoothing=(args.task_label_smoothing if model.training else 0.0),
    )
    probabilities = logits.softmax(dim=-1)
    accuracy = (logits.argmax(dim=-1) == task_ids).float().mean()
    confidence = probabilities.max(dim=-1).values.mean()
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(
        dim=-1
    ).mean()
    return loss, {
        "task_contrastive_loss": float(loss.detach()),
        "task_contrastive_accuracy": float(accuracy.detach()),
        "task_contrastive_chance": 1.0 / model.task_prior_count,
        "task_contrastive_samples": float(supervised.sum()),
        "task_contrastive_confidence": float(confidence.detach()),
        "task_contrastive_entropy": float(entropy.detach()),
    }


def controller_objective(
    model: HeMACHISSD,
    batch: dict[str, Any],
    args: argparse.Namespace,
    *,
    task_only: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Optimize official Eq. 11 with continuous action reconstruction."""
    features = model.encode_observations(batch["observations"])
    task_features = model.encode_task_observations(batch["observations"])
    common, task_skill, query = model.infer_skills(
        features, batch["valid_agents"], task_features
    )
    decoder_task_skill = (
        task_skill.detach()
        if args.detach_task_skill_for_action
        else task_skill
    )
    prediction = model.decode_actions(features, common, decoder_task_skill)
    action_loss = masked_action_mse(
        prediction, batch["actions"], batch["valid_agents"]
    )
    (
        descriptor_loss,
        metric_loss,
        variance_loss,
        descriptor_metrics,
    ) = continuous_task_objective(model, query, batch, args)
    contrastive_loss, contrastive_metrics = task_contrastive_objective(
        model,
        query,
        batch,
        args,
    )
    (
        task_action_descriptor_loss,
        task_action_variance_loss,
        task_action_metrics,
    ) = task_action_skill_objective(model, task_skill, batch, args)
    task_action_contrastive_loss, raw_task_action_contrastive_metrics = (
        task_contrastive_objective(model, task_skill, batch, args)
    )
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
            (
                args.task_action_descriptor_weight * task_action_descriptor_loss
            ).detach()
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
            skill_standard_deviation(common, batch["valid_agents"]).detach()
        ),
        "task_skill_std": float(
            skill_standard_deviation(task_skill, batch["valid_agents"]).detach()
        ),
        **descriptor_metrics,
        **contrastive_metrics,
        **task_action_metrics,
        **task_action_contrastive_metrics,
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
        error_if_nonfinite=True,
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
    *,
    task_only: bool = False,
) -> dict[str, float]:
    """Warm up task context or run all HiSSD updates in official order."""
    model.train()
    accumulator: dict[str, float] = defaultdict(float)
    batch_count = 0
    for batch_index, raw_batch in enumerate(loader):
        if args.max_train_batches is not None and batch_index >= args.max_train_batches:
            break
        batch = prepare_batch(raw_batch, device)

        controller_loss, controller_metrics = controller_objective(
            model, batch, args, task_only=task_only
        )
        controller_metrics["controller_grad_norm"] = optimize(
            controller_loss, model, optimizer, args.grad_clip
        )

        if task_only:
            with torch.no_grad():
                value_loss, value_metrics = value_objective(model, batch, args)
                planner_loss, planner_metrics = planner_objective(model, batch, args)
            value_metrics["value_grad_norm"] = 0.0
            planner_metrics["planner_grad_norm"] = 0.0
        else:
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
    sampler_generator = getattr(getattr(loader, "sampler", None), "generator", None)
    if sampler_generator is None:
        sampler_generator = getattr(
            getattr(loader, "batch_sampler", None), "generator", None
        )
    if sampler_generator is not None:
        # Keep validation batches identical across epochs so metric changes
        # reflect the model rather than a new replacement-sampled validation set.
        sampler_generator.manual_seed(args.seed + 10_000)
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


def build_model(
    sample: dict[str, Any],
    args: argparse.Namespace,
    source_task_count: int,
) -> HeMACHISSD:
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
        contrastive_from_action_skill=True,
        task_context_pooling=True,
        task_descriptor_dim=(
            int(sample["task_descriptor"].numel())
            if args.descriptor_enabled
            else 0
        ),
        task_prior_count=(source_task_count if args.task_contrastive_enabled else 0),
        task_dropout=args.task_dropout,
        task_feature_deltas=True,
        separate_task_observation_encoder=True,
        learned_task_classifier=args.task_contrastive_enabled,
        task_spatial_statistics=True,
        normalize_task_context=False,
        task_running_statistics=True,
        direct_task_summary=False,
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
    task_features = model.encode_task_observations(batch["observations"])
    common, task_skill, _ = model.infer_skills(
        features, batch["valid_agents"], task_features
    )
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
                "controller+continuous task context, expectile value, "
                "advantage-weighted planner"
            ),
            "task_context_objective": "dual_path_realized_task_context_v18",
            "ablation": args.ablation,
            "descriptor_enabled": args.descriptor_enabled,
            "task_contrastive_enabled": args.task_contrastive_enabled,
            "task_descriptor_names": tuple(args.task_descriptor_names),
            "model_config": model.config(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "bc_initialization": bc_info,
            "hyperparameters": vars(args),
            "training_tasks": list(args.training_tasks),
            "held_out_tasks": list(args.held_out_tasks),
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
        "task_learning_rate_multiplier",
        "grad_clip",
        "skill_dim",
        "hidden_dim",
        "transformer_heads",
        "gamma",
        "alpha",
        "target_tau",
        "task_contrastive_temperature",
        "task_contrastive_tail_steps",
        "reward_scale",
        "descriptor_scale_floor",
        "task_variance_target",
    )
    for name in positive_names:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if not 0 < args.expectile < 1:
        raise ValueError("--expectile must be between 0 and 1.")
    if (
        args.descriptor_weight < 0
        or args.descriptor_metric_weight < 0
        or args.task_contrastive_weight < 0
        or args.task_variance_weight < 0
        or args.task_action_descriptor_weight < 0
        or args.task_action_variance_weight < 0
        or args.task_action_contrastive_weight < 0
    ):
        raise ValueError("Task loss weights cannot be negative.")
    if args.task_weight_decay < 0:
        raise ValueError("--task-weight-decay cannot be negative.")
    if not 0 <= args.task_label_smoothing < 1:
        raise ValueError("--task-label-smoothing must be in [0, 1).")
    if not 0 <= args.task_dropout < 1:
        raise ValueError("--task-dropout must be in [0, 1).")
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience cannot be negative.")
    if args.task_warmup_epochs < 0:
        raise ValueError("--task-warmup-epochs cannot be negative.")
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
        task_balanced_batches=True,
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
        task_balanced_batches=True,
        seed=args.seed,
        pin_memory=pin_memory,
        drop_last_batch=False,
    )
    source_tasks = sorted(
        {int(entry["difficulty"]) for entry in train_dataset.entries}
    )
    expected_source_tasks = list(range(1, len(source_tasks) + 1))
    if source_tasks != expected_source_tasks:
        raise ValueError(
            "Source difficulties must be contiguous and start at 1 so persisted "
            f"task IDs remain valid; got {source_tasks}."
        )
    manifest_payload = json.loads(
        args.manifest.expanduser().resolve().read_text(encoding="utf-8")
    )
    descriptor_schema = manifest_payload.get("task_descriptor", {})
    args.task_descriptor_names = tuple(
        descriptor_schema.get("names", REALIZED_TASK_DESCRIPTOR_NAMES)
    )
    args.training_tasks = source_tasks
    args.held_out_tasks = sorted(
        int(value) for value in manifest_payload.get("target_difficulties", ())
    )

    first_train_sample = train_dataset[0]
    first_val_sample = val_dataset[0]
    descriptor_dim = int(first_train_sample["task_descriptor"].numel())
    if len(args.task_descriptor_names) != descriptor_dim:
        raise ValueError(
            "Manifest and dataset task descriptor schemas differ: "
            f"manifest={len(args.task_descriptor_names)}, dataset={descriptor_dim}."
        )
    descriptor_mean, descriptor_scale = train_dataset.task_descriptor_statistics(
        args.descriptor_scale_floor
    )
    args.task_descriptor_mean = descriptor_mean.tolist()
    args.task_descriptor_scale = descriptor_scale.tolist()
    if args.descriptor_enabled and not bool(
        first_train_sample["task_descriptor_available"]
    ):
        raise ValueError(
            "The source dataset does not contain a task descriptor."
        )
    if args.descriptor_enabled and not bool(
        first_val_sample["task_descriptor_available"]
    ):
        raise ValueError("The validation dataset has no task descriptor.")

    model = build_model(first_train_sample, args, len(source_tasks))
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

    task_parameters = [
        parameter
        for parameter in model.task_skill_encoder.parameters()
        if parameter.requires_grad
    ]
    if model.task_descriptor_head is not None:
        task_parameters.extend(
            parameter
            for parameter in model.task_descriptor_head.parameters()
            if parameter.requires_grad
        )
    if model.task_observation_encoder is not None:
        task_parameters.extend(
            parameter
            for parameter in model.task_observation_encoder.parameters()
            if parameter.requires_grad
        )
    if model.task_classifier_head is not None:
        task_parameters.extend(
            parameter
            for parameter in model.task_classifier_head.parameters()
            if parameter.requires_grad
        )
    task_parameter_ids = {id(parameter) for parameter in task_parameters}
    base_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in task_parameter_ids
    ]
    trainable_parameters = base_parameters + task_parameters
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
        ],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"device={device}, train_windows={len(train_dataset)}, "
        f"val_windows={len(val_dataset)}, "
        f"trainable_parameters={sum(p.numel() for p in trainable_parameters):,}"
    )
    print(
        f"ablation={args.ablation}, descriptor={args.descriptor_enabled}, "
        f"task_contrastive={args.task_contrastive_enabled}, "
        f"output_dir={args.output_dir}"
    )
    print(
        f"learning_rate={args.learning_rate:.2e}, task_learning_rate="
        f"{args.learning_rate * args.task_learning_rate_multiplier:.2e}, "
        f"task_weight_decay={args.task_weight_decay:.2e}"
    )
    print(
        "descriptor_scale="
        f"{[round(value, 4) for value in args.task_descriptor_scale]}, "
        f"task_action_gradient={not args.detach_task_skill_for_action}"
    )
    print(f"model_config={json.dumps(model.config())}")
    print(
        f"BC initialization epoch={bc_info.get('epoch')}, "
        f"max_action_difference={initialization_error:.3e}"
    )

    writer = create_writer(args)
    best_validation_loss = math.inf
    best_task_validation_loss = math.inf
    best_validation_epoch = 0
    best_task_validation_epoch = 0
    epochs_without_task_improvement = 0
    task_auxiliary_enabled = (
        args.descriptor_enabled or args.task_contrastive_enabled
    )
    selection_metric_name = (
        "task auxiliary validation loss"
        if task_auxiliary_enabled
        else "combined validation loss"
    )
    try:
        for epoch in range(1, args.epochs + 1):
            task_only = task_auxiliary_enabled and epoch <= args.task_warmup_epochs
            train_metrics = run_train_epoch(
                model,
                train_loader,
                optimizer,
                device,
                args,
                task_only=task_only,
            )
            validation_metrics = run_validation(model, val_loader, device, args)
            validation_loss = (
                validation_metrics["action_mse"]
                + args.descriptor_weight * validation_metrics["descriptor_loss"]
                + args.descriptor_metric_weight
                * validation_metrics["descriptor_metric_loss"]
                + args.task_variance_weight
                * validation_metrics["task_variance_loss"]
                + args.task_contrastive_weight
                * validation_metrics["task_contrastive_loss"]
                + args.task_action_descriptor_weight
                * validation_metrics["task_action_descriptor_loss"]
                + args.task_action_variance_weight
                * validation_metrics["task_action_variance_loss"]
                + args.task_action_contrastive_weight
                * validation_metrics["task_action_contrastive_loss"]
                + validation_metrics["value_loss"]
                + validation_metrics["planner_loss"]
            )
            task_validation_loss = (
                args.descriptor_weight * validation_metrics["descriptor_loss"]
                + args.descriptor_metric_weight
                * validation_metrics["descriptor_metric_loss"]
                + args.task_variance_weight
                * validation_metrics["task_variance_loss"]
                + args.task_contrastive_weight
                * validation_metrics["task_contrastive_loss"]
                + args.task_action_descriptor_weight
                * validation_metrics["task_action_descriptor_loss"]
                + args.task_action_variance_weight
                * validation_metrics["task_action_variance_loss"]
                + args.task_action_contrastive_weight
                * validation_metrics["task_action_contrastive_loss"]
            )
            early_stopping_loss = (
                task_validation_loss
                if task_auxiliary_enabled
                else validation_loss
            )
            print(
                f"epoch={epoch:03d} "
                f"phase={'task_warmup' if task_only else 'joint'} "
                f"action={train_metrics['action_mse']:.5f}/"
                f"{validation_metrics['action_mse']:.5f} "
                f"descriptor={train_metrics['descriptor_loss']:.5f}/"
                f"{validation_metrics['descriptor_loss']:.5f} "
                f"descriptor_mae={validation_metrics['descriptor_mae']:.4f} "
                f"baseline="
                f"{validation_metrics['descriptor_median_baseline_mae']:.4f} "
                f"gain={validation_metrics['descriptor_mae_gain']:+.4f} "
                f"descriptor_std="
                f"{validation_metrics['descriptor_prediction_std']:.4f}/"
                f"{validation_metrics['descriptor_target_std']:.4f} "
                f"std_ratio="
                f"{validation_metrics['descriptor_std_ratio']:.3f} "
                f"metric={train_metrics['descriptor_metric_loss']:.5f}/"
                f"{validation_metrics['descriptor_metric_loss']:.5f} "
                f"distance_corr="
                f"{validation_metrics['descriptor_distance_correlation']:.3f} "
                f"variance={train_metrics['task_variance_loss']:.4f}/"
                f"{validation_metrics['task_variance_loss']:.4f} "
                f"task_contrast="
                f"{train_metrics['task_contrastive_loss']:.4f}/"
                f"{validation_metrics['task_contrastive_loss']:.4f} "
                f"task_acc="
                f"{train_metrics['task_contrastive_accuracy']:.3f}/"
                f"{validation_metrics['task_contrastive_accuracy']:.3f}/"
                f"{validation_metrics['task_contrastive_chance']:.3f} "
                f"task_conf="
                f"{train_metrics['task_contrastive_confidence']:.3f}/"
                f"{validation_metrics['task_contrastive_confidence']:.3f} "
                f"z_descriptor="
                f"{train_metrics['task_action_descriptor_loss']:.4f}/"
                f"{validation_metrics['task_action_descriptor_loss']:.4f} "
                f"z_variance="
                f"{train_metrics['task_action_variance_loss']:.4f}/"
                f"{validation_metrics['task_action_variance_loss']:.4f} "
                f"z_task_acc="
                f"{train_metrics['task_action_contrastive_accuracy']:.3f}/"
                f"{validation_metrics['task_action_contrastive_accuracy']:.3f} "
                f"z_context_std="
                f"{validation_metrics['task_action_context_std']:.4f} "
                f"task_val={task_validation_loss:.4f} "
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
                writer.add_scalar("validation/task_loss", task_validation_loss, epoch)

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
            # A warm-up checkpoint has untouched value/planner modules and is not
            # suitable for rollout or target adaptation. Select the deployable
            # checkpoint only after joint HiSSD optimization has started.
            if not task_only and validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_validation_epoch = epoch
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
            if early_stopping_loss < best_task_validation_loss:
                best_task_validation_loss = early_stopping_loss
                best_task_validation_epoch = epoch
                epochs_without_task_improvement = 0
                save_checkpoint(
                    args.output_dir / "hissd_best_task.pt",
                    model,
                    optimizer,
                    epoch=epoch,
                    train_metrics=train_metrics,
                    validation_metrics=validation_metrics,
                    args=args,
                    bc_info=bc_info,
                )
            elif not task_only:
                # Warm-up must not consume the patience intended for joint
                # controller/value/planner training.
                epochs_without_task_improvement += 1
            if (
                not task_only
                and args.early_stopping_patience > 0
                and epochs_without_task_improvement
                >= args.early_stopping_patience
            ):
                print(
                    f"Early stopping: {selection_metric_name} did not improve for "
                    f"{args.early_stopping_patience} epochs."
                )
                break
    finally:
        if writer is not None:
            writer.close()
    print(
        f"Best combined validation loss: {best_validation_loss:.6f} "
        f"at epoch {best_validation_epoch} "
        f"({args.output_dir / 'hissd_best.pt'})"
    )
    print(
        f"Best {selection_metric_name}: {best_task_validation_loss:.6f} "
        f"at epoch {best_task_validation_epoch} "
        f"({args.output_dir / 'hissd_best_task.pt'})"
    )


if __name__ == "__main__":
    main()
