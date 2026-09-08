"""Adapt HiSSD skills and drone actions to manifest-defined target tasks.

The common representation and BC-compatible base policy stay frozen. Target
actions are learned from successful, collision-safe trajectory segments while
source distillation limits forgetting.
"""

from __future__ import annotations

import argparse
import copy
import math
import random
import sys
from collections import defaultdict
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

from skill_discovery.dataset import (
    DEFAULT_MANIFEST_PATH,
    create_dataloader,
    load_manifest,
)
from skill_discovery.hissd_models import HeMACHISSD
from skill_discovery.train_hissd import (
    move_observations,
    resolve_device,
    standardized_descriptor_loss,
    task_tail_mask,
    value_objective,
)
from skill_discovery.visualize_hissd_skills import load_hissd_model


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "src/skill_discovery/checkpoints/hissd_checkpoints_v19/hissd_adapted_best.pt"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "src/skill_discovery/checkpoints/hissd_checkpoints_v22"


@dataclass(frozen=True)
class FrozenTaskTeacher:
    """Frozen source modules used to prevent representation forgetting."""

    task_observation_encoder: nn.Module | None
    task_skill_encoder: nn.Module
    action_decoder: nn.Module
    skill_structure: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--task-learning-rate", type=float, default=5e-5)
    parser.add_argument(
        "--classifier-learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for the newly expanded source+target task classifier.",
    )
    parser.add_argument("--task-action-learning-rate", type=float, default=2e-4)
    parser.add_argument("--value-learning-rate", type=float, default=1e-4)
    parser.add_argument("--value-loss-weight", type=float, default=0.5)
    parser.add_argument("--value-selection-weight", type=float, default=0.1)
    parser.add_argument("--expectile", type=float, default=0.7)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--reward-scale", type=float, default=100.0)
    parser.add_argument("--value-target-tau", type=float, default=0.005)
    parser.add_argument(
        "--adapt-shared-decoder",
        action="store_true",
        help=(
            "Also update the shared observation/common/task fusion decoder. By "
            "default adaptation changes only task-specific modules so target "
            "imitation cannot overwrite the source controller."
        ),
    )
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--source-distillation-weight", type=float, default=1.0)
    parser.add_argument("--target-anchor-weight", type=float, default=0.15)
    parser.add_argument("--descriptor-weight", type=float, default=2.0)
    parser.add_argument("--task-classification-weight", type=float, default=0.1)
    parser.add_argument("--query-contrastive-weight", type=float, default=0.25)
    parser.add_argument("--decoder-skill-contrastive-weight", type=float, default=0.02)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--joint-representation-scale", type=float, default=0.05)
    parser.add_argument("--action-loss-weight", type=float, default=2.0)
    parser.add_argument("--task-usage-weight", type=float, default=0.0)
    parser.add_argument("--task-usage-margin", type=float, default=0.0)
    parser.add_argument("--source-skill-distillation-weight", type=float, default=1.0)
    parser.add_argument("--task-tail-steps", type=int, default=8)
    parser.add_argument(
        "--skill-warmup-epochs",
        type=int,
        default=0,
        help=(
            "Optional representation-only warmup. The default is zero because v22 "
            "starts from the already separated v19 representation."
        ),
    )
    parser.add_argument("--success-weight", type=float, default=1.0)
    parser.add_argument("--goal-found-failure-weight", type=float, default=0.1)
    parser.add_argument("--goal-not-found-weight", type=float, default=0.01)
    parser.add_argument(
        "--crash-reward-threshold",
        type=float,
        default=-100.0,
        help="Do not imitate a drone action whose immediate local reward is below this.",
    )
    parser.add_argument(
        "--crash-lookback-steps",
        type=int,
        default=12,
        help=(
            "Also exclude this many actions before a drone collision. This avoids "
            "teaching the approach trajectory that led into the warning zone."
        ),
    )
    parser.add_argument(
        "--warning-focus-weight",
        type=float,
        default=0.5,
        help=(
            "Additional action-loss weight at successful steps where the local "
            "warning-zone channel is visible; 0.5 gives those steps up to 1.5x weight."
        ),
    )
    parser.add_argument(
        "--observer-warning-focus-weight",
        type=float,
        default=2.0,
        help=(
            "Additional successful-action weight when the observed warning zone "
            "overlaps the observer position in a drone's global map."
        ),
    )
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--dataset-cache-size",
        type=int,
        default=16,
        help="Number of mmap-backed episode files cached per DataLoader worker.",
    )
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_gpu_backend(device: torch.device) -> None:
    """Enable fast kernels for fixed-shape target adaptation workloads."""
    if device.type != "cuda":
        return
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "epochs",
        "batch_size",
        "sequence_length",
        "stride",
        "learning_rate",
        "task_learning_rate",
        "classifier_learning_rate",
        "task_action_learning_rate",
        "value_learning_rate",
        "reward_scale",
        "grad_clip",
        "task_tail_steps",
        "contrastive_temperature",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    for name in (
        "weight_decay",
        "source_distillation_weight",
        "target_anchor_weight",
        "descriptor_weight",
        "task_classification_weight",
        "query_contrastive_weight",
        "decoder_skill_contrastive_weight",
        "joint_representation_scale",
        "action_loss_weight",
        "task_usage_weight",
        "task_usage_margin",
        "source_skill_distillation_weight",
        "value_loss_weight",
        "value_selection_weight",
        "warning_focus_weight",
        "observer_warning_focus_weight",
        "success_weight",
        "goal_found_failure_weight",
        "goal_not_found_weight",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative.")
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience cannot be negative.")
    if args.skill_warmup_epochs < 0:
        raise ValueError("--skill-warmup-epochs cannot be negative.")
    if args.crash_lookback_steps < 0:
        raise ValueError("--crash-lookback-steps cannot be negative.")
    if args.dataset_cache_size <= 0:
        raise ValueError("--dataset-cache-size must be positive.")
    if args.epochs <= args.skill_warmup_epochs:
        raise ValueError("--epochs must exceed --skill-warmup-epochs.")
    if not 0.0 < args.expectile < 1.0:
        raise ValueError("--expectile must be in (0, 1).")
    if not 0.0 < args.gamma <= 1.0:
        raise ValueError("--gamma must be in (0, 1].")
    if not 0.0 < args.value_target_tau <= 1.0:
        raise ValueError("--value-target-tau must be in (0, 1].")


def prepare_batch(raw_batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move actor tensors and build category/reward masks for target BC."""
    actions = raw_batch["drone"]["actions"].to(device, non_blocking=True)
    drone_count = actions.shape[-2]
    filled = raw_batch["filled"].squeeze(-1).to(device, non_blocking=True).bool()
    agent_mask = raw_batch["agent_mask"][..., -drone_count:].to(
        device, non_blocking=True
    ).bool()
    all_individual_rewards = raw_batch["individual_rewards"].to(
        device, non_blocking=True
    )
    individual_rewards = all_individual_rewards[..., -drone_count:]
    observer_rewards = all_individual_rewards[..., :-drone_count]
    terminated = raw_batch["terminated"].squeeze(-1).to(
        device, non_blocking=True
    )
    truncated = raw_batch["truncated"].squeeze(-1).to(
        device, non_blocking=True
    )
    batch = {
        "observations": move_observations(raw_batch["drone"]["observations"], device),
        "next_observations": move_observations(
            raw_batch["drone"]["next_observations"], device
        ),
        "actions": actions,
        "valid_agents": agent_mask & filled.unsqueeze(-1),
        "valid_steps": filled,
        "central_map": raw_batch["global_state"]["central_map"].to(
            device, non_blocking=True
        ),
        "next_central_map": raw_batch["next_global_state"]["central_map"].to(
            device, non_blocking=True
        ),
        "team_reward": raw_batch["drone_task_reward"].squeeze(-1).to(
            device, non_blocking=True
        ),
        "done": terminated.bool() | truncated.bool(),
        "individual_rewards": individual_rewards,
        "observer_rewards": observer_rewards,
        "difficulty": raw_batch["difficulty"].to(device, non_blocking=True),
        "task_id": raw_batch["task_id"].to(device, non_blocking=True),
        "task_descriptor": raw_batch["task_descriptor"].to(
            device, non_blocking=True
        ),
        "task_descriptor_available": raw_batch["task_descriptor_available"].to(
            device, non_blocking=True
        ).bool(),
        "task_supervision": (
            raw_batch["window_start"].to(device, non_blocking=True) == 0
        ),
    }
    if "labels" in raw_batch:
        batch["outcome_category"] = raw_batch["labels"]["outcome_category"].to(
            device, non_blocking=True
        )
    return batch


def weighted_action_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_agents: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    """Average action MSE with independent episode and transition weights."""
    weights = valid_agents.to(prediction.dtype) * sample_weights
    denominator = weights.sum() * prediction.shape[-1]
    return (
        (prediction - target).square() * weights.unsqueeze(-1)
    ).sum() / denominator.clamp_min(1.0)


def masked_action_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_agents: torch.Tensor,
) -> torch.Tensor:
    return weighted_action_mse(
        prediction,
        target,
        valid_agents,
        torch.ones_like(valid_agents, dtype=prediction.dtype),
    )


def current_actor_outputs(
    model: HeMACHISSD,
    batch: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run frozen common features and trainable task-specific features."""
    with torch.no_grad():
        features = model.encode_observations(batch["observations"])
        if model.skill_structure == "task_only":
            common = features.new_zeros(
                *features.shape[:-1], model.skill_dim
            )
        else:
            common = model.common_skill_encoder(features, batch["valid_agents"])
    if model.skill_structure == "shared":
        task = common
        query = common
    elif model.skill_structure == "common_only":
        task = torch.zeros_like(common)
        query = torch.zeros_like(common)
    else:
        task_features = model.encode_task_observations(batch["observations"])
        task, query = model.task_skill_encoder(
            task_features, batch["valid_agents"]
        )
    prediction = model.decode_actions(features, common, task)
    return features, common, task, query, prediction


@torch.no_grad()
def teacher_actor_outputs(
    teacher: FrozenTaskTeacher,
    batch: dict[str, Any],
    features: torch.Tensor,
    common: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the original v15 task pathway for action and skill anchors."""
    if teacher.skill_structure == "shared":
        task = common
        query = common
    elif teacher.skill_structure == "common_only":
        task = torch.zeros_like(common)
        query = torch.zeros_like(common)
    else:
        if teacher.task_observation_encoder is None:
            task_features = features
        else:
            observations = batch["observations"]
            task_features = teacher.task_observation_encoder(
                observations["global_map"],
                observations["local_map"],
                observations["action_history"],
            )
        task, query = teacher.task_skill_encoder(
            task_features, batch["valid_agents"]
        )
    teacher_common = torch.zeros_like(common) if (
        teacher.skill_structure == "task_only"
    ) else common
    direct_skill = (
        teacher_common
        if teacher.skill_structure == "common_only"
        else task
    )
    action = torch.tanh(
        teacher.action_decoder.forward_logits(
            features,
            teacher_common,
            task,
            direct_residual_skills=direct_skill,
        )
    )
    return task, query, action


def target_sample_weights(
    batch: dict[str, Any],
    args: argparse.Namespace,
) -> torch.Tensor:
    """Select successful actions that did not lead directly into a collision."""
    category_weights = torch.tensor(
        [
            args.success_weight,
            args.goal_found_failure_weight,
            args.goal_not_found_weight,
        ],
        device=batch["actions"].device,
        dtype=batch["actions"].dtype,
    )
    episode_weights = category_weights[batch["outcome_category"]]
    collision_action = batch["individual_rewards"] <= args.crash_reward_threshold
    observer_rewards = batch.get("observer_rewards")
    if observer_rewards is None:
        observer_collision = torch.zeros_like(collision_action[..., :1])
    else:
        observer_collision = (
            observer_rewards <= args.crash_reward_threshold
        ).any(dim=-1, keepdim=True)
    # Observer death is a team failure: every drone action in its lead-up is unsafe.
    collision_action = collision_action | observer_collision.expand_as(collision_action)
    unsafe_action = collision_action.clone()
    # A collision is the result of an approach trajectory, not only its final action.
    # Marking a short lookback horizon avoids behavior-cloning those precursors.
    for offset in range(1, getattr(args, "crash_lookback_steps", 0) + 1):
        unsafe_action[:, :-offset] |= collision_action[:, offset:]
    safe_action = ~unsafe_action
    focus = torch.ones_like(batch["individual_rewards"])
    successful_episode = (batch["outcome_category"] == 0)[:, None, None]
    local_map = batch.get("observations", {}).get("local_map")
    warning_focus_weight = getattr(args, "warning_focus_weight", 0.0)
    if (
        warning_focus_weight > 0.0
        and local_map is not None
        and local_map.shape[-3] > 3
    ):
        warning_confidence = (
            local_map[..., 3, :, :].amax(dim=(-1, -2)).clamp(0.0, 1.0)
        )
        focus = focus + (
            warning_focus_weight * warning_confidence * successful_episode
        )
    global_map = batch.get("observations", {}).get("global_map")
    observer_warning_focus_weight = getattr(
        args, "observer_warning_focus_weight", 0.0
    )
    if (
        observer_warning_focus_weight > 0.0
        and global_map is not None
        and global_map.shape[-3] > 5
    ):
        observer_warning_overlap = (
            global_map[..., 3, :, :] * global_map[..., 5, :, :]
        ).amax(dim=(-1, -2)).clamp(0.0, 1.0)
        focus = focus + (
            observer_warning_focus_weight
            * observer_warning_overlap
            * successful_episode
        )
    return (
        episode_weights[:, None, None]
        * safe_action.to(batch["actions"].dtype)
        * focus
    )


def actor_predictions(
    model: HeMACHISSD,
    teacher: FrozenTaskTeacher,
    batch: dict[str, Any],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    features, common, task, query, prediction = current_actor_outputs(model, batch)
    teacher_task, teacher_query, teacher_prediction = teacher_actor_outputs(
        teacher, batch, features, common
    )
    with torch.no_grad():
        no_task_prediction = model.decode_actions(
            features, common, torch.zeros_like(task)
        )
        no_common_prediction = model.decode_actions(
            features, torch.zeros_like(common), task
        )
    return (
        prediction,
        teacher_prediction,
        no_task_prediction,
        no_common_prediction,
        task,
        query,
        model.conditioning_skill(common, task),
        teacher_task,
        teacher_query,
    )


def masked_vector_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_agents: torch.Tensor,
) -> torch.Tensor:
    """Average latent MSE over valid agent/time positions."""
    mask = valid_agents.unsqueeze(-1).to(prediction.dtype)
    denominator = mask.sum() * prediction.shape[-1]
    return ((prediction - target).square() * mask).sum() / denominator.clamp_min(1.0)


def task_representation_objective(
    model: HeMACHISSD,
    task_classifier: nn.Module,
    query: torch.Tensor,
    batch: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Learn continuous task dynamics and manifest-wide difficulty separation."""
    sequence_valid = batch["valid_agents"].bool().any(dim=2)
    supervision = (
        task_tail_mask(
            batch["valid_agents"],
            batch["task_supervision"],
            args.task_tail_steps,
        )
        & sequence_valid
        & batch["task_descriptor_available"].unsqueeze(1)
    )
    if model.task_descriptor_head is not None:
        sequence_prediction, _ = model.predict_task_descriptor_sequence(
            query, batch["valid_agents"]
        )
        target = batch["task_descriptor"].unsqueeze(1).expand_as(
            sequence_prediction
        )
        if supervision.any():
            descriptor_loss = standardized_descriptor_loss(
                sequence_prediction[supervision],
                target[supervision],
                args,
            )
            descriptor_mae = (
                sequence_prediction[supervision] - target[supervision]
            ).abs().mean()
        else:
            descriptor_loss = query.sum() * 0.0
            descriptor_mae = descriptor_loss.detach()
    else:
        descriptor_loss = query.sum() * 0.0
        descriptor_mae = descriptor_loss.detach()

    numeric_mask = batch["valid_agents"].unsqueeze(-1).to(query.dtype)
    step_context = (query * numeric_mask).sum(dim=2) / numeric_mask.sum(
        dim=2
    ).clamp_min(1.0)
    logits = task_classifier(step_context[supervision])
    labels = batch["task_id"].unsqueeze(1).expand_as(supervision)[supervision]
    if logits.shape[0] > 0:
        classification_loss = nn.functional.cross_entropy(logits, labels)
        accuracy = (logits.argmax(dim=-1) == labels).float().mean()
    else:
        classification_loss = query.sum() * 0.0
        accuracy = classification_loss.detach()
    return descriptor_loss, classification_loss, {
        "descriptor_mae": float(descriptor_mae.detach()),
        "task_accuracy": float(accuracy.detach()),
        "task_samples": float(supervision.sum()),
    }


def supervised_task_contexts(
    representation: torch.Tensor,
    batch: dict[str, Any],
    tail_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool agents and select late causal contexts with reliable task evidence."""
    supervision = task_tail_mask(
        batch["valid_agents"], batch["task_supervision"], tail_steps
    )
    numeric_mask = batch["valid_agents"].unsqueeze(-1).to(representation.dtype)
    context = (representation * numeric_mask).sum(dim=2) / numeric_mask.sum(
        dim=2
    ).clamp_min(1.0)
    labels = batch["task_id"].unsqueeze(1).expand_as(supervision)
    return context[supervision], labels[supervision]


def supervised_contrastive_loss(
    contexts: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pull same-task contexts together and repel every different task directly."""
    if contexts.shape[0] < 2:
        zero = contexts.sum() * 0.0
        return zero, zero.detach()
    contexts = nn.functional.normalize(contexts, dim=-1)
    logits = contexts @ contexts.transpose(0, 1) / temperature
    identity = torch.eye(
        contexts.shape[0], dtype=torch.bool, device=contexts.device
    )
    positive = labels[:, None].eq(labels[None, :]) & ~identity
    valid_anchor = positive.any(dim=1)
    if not valid_anchor.any():
        zero = contexts.sum() * 0.0
        return zero, zero.detach()
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = logits.exp().masked_fill(identity, 0.0)
    log_probability = logits - exp_logits.sum(dim=1, keepdim=True).clamp_min(
        1e-8
    ).log()
    positive_log_probability = (
        log_probability.masked_fill(~positive, 0.0).sum(dim=1)
        / positive.sum(dim=1).clamp_min(1)
    )
    loss = -positive_log_probability[valid_anchor].mean()
    with torch.no_grad():
        similarity = contexts @ contexts.transpose(0, 1)
        positive_similarity = similarity[positive].mean()
        negative = labels[:, None].ne(labels[None, :])
        negative_similarity = similarity[negative].mean()
        margin = positive_similarity - negative_similarity
    return loss, margin


def adaptation_objective(
    model: HeMACHISSD,
    teacher: FrozenTaskTeacher,
    task_classifier: nn.Module,
    target_batch: dict[str, Any],
    source_batch: dict[str, Any] | None,
    args: argparse.Namespace,
    *,
    action_enabled: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    (
        target_prediction,
        target_teacher,
        target_no_task,
        target_no_common,
        target_task,
        target_query,
        target_conditioning_skill,
        _,
        _,
    ) = actor_predictions(model, teacher, target_batch)
    sample_weights = target_sample_weights(target_batch, args)
    target_loss = weighted_action_mse(
        target_prediction,
        target_batch["actions"],
        target_batch["valid_agents"],
        sample_weights,
    )
    no_task_target_loss = weighted_action_mse(
        target_no_task,
        target_batch["actions"],
        target_batch["valid_agents"],
        sample_weights,
    )
    task_reconstruction_gain = no_task_target_loss - target_loss
    task_usage_loss = nn.functional.relu(
        target_loss.new_tensor(args.task_usage_margin)
        - task_reconstruction_gain
    )
    target_anchor = masked_action_mse(
        target_prediction,
        target_teacher,
        target_batch["valid_agents"],
    )
    target_descriptor, target_classification, target_skill_metrics = (
        task_representation_objective(
            model, task_classifier, target_query, target_batch, args
        )
    )
    target_value_loss, target_value_metrics = value_objective(
        model, target_batch, args
    )

    source_distillation = target_loss.new_zeros(())
    source_skill_distillation = target_loss.new_zeros(())
    source_descriptor = target_loss.new_zeros(())
    source_classification = target_loss.new_zeros(())
    source_skill_metrics = {"descriptor_mae": 0.0, "task_accuracy": 0.0}
    source_task = None
    source_query = None
    if source_batch is not None:
        (
            source_prediction,
            source_teacher,
            _,
            _,
            source_task,
            source_query,
            _,
            teacher_source_task,
            teacher_source_query,
        ) = actor_predictions(model, teacher, source_batch)
        source_distillation = masked_action_mse(
            source_prediction,
            source_teacher,
            source_batch["valid_agents"],
        )
        source_skill_distillation = 0.5 * (
            masked_vector_mse(
                source_task,
                teacher_source_task,
                source_batch["valid_agents"],
            )
            + masked_vector_mse(
                source_query,
                teacher_source_query,
                source_batch["valid_agents"],
            )
        )
        source_descriptor, source_classification, source_skill_metrics = (
            task_representation_objective(
                model, task_classifier, source_query, source_batch, args
            )
        )

    target_query_context, target_context_labels = supervised_task_contexts(
        target_query, target_batch, args.task_tail_steps
    )
    target_decoder_context, _ = supervised_task_contexts(
        target_task, target_batch, args.task_tail_steps
    )
    query_contexts = [target_query_context]
    decoder_contexts = [target_decoder_context]
    context_labels = [target_context_labels]
    if source_batch is not None and source_task is not None and source_query is not None:
        source_query_context, source_context_labels = supervised_task_contexts(
            source_query, source_batch, args.task_tail_steps
        )
        source_decoder_context, _ = supervised_task_contexts(
            source_task, source_batch, args.task_tail_steps
        )
        query_contexts.append(source_query_context)
        decoder_contexts.append(source_decoder_context)
        context_labels.append(source_context_labels)
    all_labels = torch.cat(context_labels, dim=0)
    query_contrastive, query_margin = supervised_contrastive_loss(
        torch.cat(query_contexts, dim=0),
        all_labels,
        args.contrastive_temperature,
    )
    decoder_contrastive, decoder_margin = supervised_contrastive_loss(
        torch.cat(decoder_contexts, dim=0),
        all_labels,
        args.contrastive_temperature,
    )

    conservative_anchor = (
        args.source_distillation_weight * source_distillation
        + args.target_anchor_weight * target_anchor
    )
    descriptor_total = 0.5 * (target_descriptor + source_descriptor)
    classification_total = 0.5 * (
        target_classification + source_classification
    )
    discovery_total = (
        args.descriptor_weight * descriptor_total
        + args.task_classification_weight * classification_total
        + args.query_contrastive_weight * query_contrastive
        + args.decoder_skill_contrastive_weight * decoder_contrastive
    )
    representation_scale = 1.0 if not action_enabled else args.joint_representation_scale
    weighted_action_loss = (
        args.action_loss_weight * target_loss
        if action_enabled
        else target_loss.detach() * 0.0
    )
    total = (
        representation_scale * discovery_total
        + args.source_skill_distillation_weight * source_skill_distillation
        + conservative_anchor
        + weighted_action_loss
        + (args.task_usage_weight * task_usage_loss if action_enabled else 0.0)
        + args.value_loss_weight * target_value_loss
    )
    valid = target_batch["valid_agents"].unsqueeze(-1)
    task_effect = (target_prediction - target_no_task).abs()[valid.expand_as(
        target_prediction
    )].mean()
    common_effect = (target_prediction - target_no_common).abs()[valid.expand_as(
        target_prediction
    )].mean()
    task_action_head = model.action_decoder.task_action_residual_head
    if task_action_head is None:
        direct_task_effect = task_effect.new_zeros(())
    else:
        direct_task_logits = task_action_head(target_conditioning_skill)
        direct_task_effect = direct_task_logits.abs()[valid.expand_as(
            direct_task_logits
        )].mean()
    metrics = {
        "loss": float(total.detach()),
        "target_weighted_mse": float(target_loss.detach()),
        "no_task_target_mse": float(no_task_target_loss.detach()),
        "task_reconstruction_gain": float(task_reconstruction_gain.detach()),
        "task_usage_loss": float(task_usage_loss.detach()),
        "representation_scale": float(representation_scale),
        "weighted_action_loss": float(weighted_action_loss.detach()),
        "target_anchor_mse": float(target_anchor.detach()),
        "source_distillation_mse": float(source_distillation.detach()),
        "source_skill_distillation_mse": float(
            source_skill_distillation.detach()
        ),
        "target_descriptor_loss": float(target_descriptor.detach()),
        "source_descriptor_loss": float(source_descriptor.detach()),
        "target_descriptor_mae": target_skill_metrics["descriptor_mae"],
        "source_descriptor_mae": source_skill_metrics["descriptor_mae"],
        "target_task_accuracy": target_skill_metrics["task_accuracy"],
        "source_task_accuracy": source_skill_metrics["task_accuracy"],
        "query_contrastive_loss": float(query_contrastive.detach()),
        "decoder_contrastive_loss": float(decoder_contrastive.detach()),
        "query_similarity_margin": float(query_margin),
        "decoder_similarity_margin": float(decoder_margin),
        "action_enabled": float(action_enabled),
        "task_action_effect": float(task_effect.detach()),
        "direct_task_logit_effect": float(direct_task_effect.detach()),
        "common_action_effect": float(common_effect.detach()),
        "effective_target_fraction": float(
            ((sample_weights > 0) & target_batch["valid_agents"]).float().sum()
            / target_batch["valid_agents"].float().sum().clamp_min(1.0)
        ),
        **target_value_metrics,
    }
    return total, metrics


def average_metrics(accumulator: dict[str, float], count: int) -> dict[str, float]:
    if count <= 0:
        raise RuntimeError("No adaptation batches were processed.")
    return {name: value / count for name, value in accumulator.items()}


def run_train_epoch(
    model: HeMACHISSD,
    teacher: FrozenTaskTeacher,
    task_classifier: nn.Module,
    target_loader,
    source_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
) -> dict[str, float]:
    model.train()
    task_classifier.train()
    # Frozen encoders must stay deterministic even while the decoder trains.
    model.observation_encoder.eval()
    if model.task_observation_encoder is not None:
        model.task_observation_encoder.eval()
    model.common_skill_encoder.eval()
    model.task_skill_encoder.eval()
    source_iterator = iter(source_loader)
    accumulator: dict[str, float] = defaultdict(float)
    count = 0
    for batch_index, raw_target in enumerate(target_loader):
        if args.max_train_batches is not None and batch_index >= args.max_train_batches:
            break
        target_batch = prepare_batch(raw_target, device)
        try:
            raw_source = next(source_iterator)
        except StopIteration:
            source_iterator = iter(source_loader)
            raw_source = next(source_iterator)
        source_batch = prepare_batch(raw_source, device)
        loss, metrics = adaptation_objective(
            model,
            teacher,
            task_classifier,
            target_batch,
            source_batch,
            args,
            action_enabled=epoch > args.skill_warmup_epochs,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite adaptation loss: {float(loss)}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            (
                parameter
                for parameter in (*model.parameters(), *task_classifier.parameters())
                if parameter.requires_grad
            ),
            args.grad_clip,
            error_if_nonfinite=True,
        )
        optimizer.step()
        model.update_targets(args.value_target_tau)
        metrics["grad_norm"] = float(grad_norm)
        for name, value in metrics.items():
            accumulator[name] += value
        count += 1
    return average_metrics(accumulator, count)


@torch.inference_mode()
def run_validation(
    model: HeMACHISSD,
    teacher: FrozenTaskTeacher,
    task_classifier: nn.Module,
    target_loader,
    source_loader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    task_classifier.eval()
    accumulator: dict[str, float] = defaultdict(float)
    category_error = torch.zeros(3, device=device)
    category_weight = torch.zeros(3, device=device)
    target_difficulties = tuple(args.target_difficulties)
    difficulty_category_error = torch.zeros(
        len(target_difficulties), 3, device=device
    )
    difficulty_category_weight = torch.zeros_like(difficulty_category_error)
    safe_success_error = torch.zeros(len(target_difficulties), device=device)
    safe_success_weight = torch.zeros_like(safe_success_error)
    count = 0
    source_iterator = iter(source_loader)
    for batch_index, raw_batch in enumerate(target_loader):
        if args.max_val_batches is not None and batch_index >= args.max_val_batches:
            break
        batch = prepare_batch(raw_batch, device)
        try:
            raw_source = next(source_iterator)
        except StopIteration:
            source_iterator = iter(source_loader)
            raw_source = next(source_iterator)
        source_batch = prepare_batch(raw_source, device)
        loss, metrics = adaptation_objective(
            model,
            teacher,
            task_classifier,
            batch,
            source_batch,
            args,
            action_enabled=True,
        )
        metrics["loss"] = float(loss)
        prediction = actor_predictions(model, teacher, batch)[0]
        squared_error = (prediction - batch["actions"]).square().mean(dim=-1)
        action_weights = target_sample_weights(batch, args)
        for category in range(3):
            category_mask = (
                batch["valid_agents"]
                & (batch["outcome_category"] == category)[:, None, None]
            )
            category_error[category] += squared_error[category_mask].sum()
            category_weight[category] += category_mask.sum()
            for difficulty_index, difficulty in enumerate(target_difficulties):
                difficulty_mask = (
                    category_mask
                    & (batch["difficulty"] == difficulty)[:, None, None]
                )
                difficulty_category_error[difficulty_index, category] += (
                    squared_error[difficulty_mask].sum()
                )
                difficulty_category_weight[difficulty_index, category] += (
                    difficulty_mask.sum()
                )
        for difficulty_index, difficulty in enumerate(target_difficulties):
            safe_mask = (
                batch["valid_agents"]
                & (batch["outcome_category"] == 0)[:, None, None]
                & (batch["difficulty"] == difficulty)[:, None, None]
            )
            safe_weights = action_weights * safe_mask.to(action_weights.dtype)
            safe_success_error[difficulty_index] += (
                squared_error * safe_weights
            ).sum()
            safe_success_weight[difficulty_index] += safe_weights.sum()
        for name, value in metrics.items():
            accumulator[name] += value
        count += 1
    result = average_metrics(accumulator, count)
    result["safe_success_mse"] = float(
        safe_success_error.sum() / safe_success_weight.sum().clamp_min(1.0)
    )
    names = ("success_mse", "goal_found_failure_mse", "goal_not_found_mse")
    for index, name in enumerate(names):
        result[name] = float(
            category_error[index] / category_weight[index].clamp_min(1.0)
        )
    for difficulty_index, difficulty in enumerate(target_difficulties):
        for category_index, category_name in enumerate(
            ("success", "goal_found_failure", "goal_not_found")
        ):
            result[f"difficulty_{difficulty}/{category_name}_mse"] = float(
                difficulty_category_error[difficulty_index, category_index]
                / difficulty_category_weight[
                    difficulty_index, category_index
                ].clamp_min(1.0)
            )
        result[f"difficulty_{difficulty}/safe_success_mse"] = float(
            safe_success_error[difficulty_index]
            / safe_success_weight[difficulty_index].clamp_min(1.0)
        )
    return result


@torch.inference_mode()
def run_source_validation(
    model: HeMACHISSD,
    teacher: FrozenTaskTeacher,
    loader,
    device: torch.device,
    args: argparse.Namespace,
) -> float:
    """Measure final policy drift on held-out source trajectories."""
    model.eval()
    squared_error = torch.zeros((), device=device)
    valid_count = torch.zeros((), device=device)
    for batch_index, raw_batch in enumerate(loader):
        if args.max_val_batches is not None and batch_index >= args.max_val_batches:
            break
        batch = prepare_batch(raw_batch, device)
        prediction, teacher_prediction = actor_predictions(
            model, teacher, batch
        )[:2]
        valid = batch["valid_agents"].unsqueeze(-1).expand_as(prediction)
        squared_error += (prediction - teacher_prediction).square()[valid].sum()
        valid_count += valid.sum()
    return float(squared_error / valid_count.clamp_min(1.0))


def build_loader(
    args: argparse.Namespace,
    split: str,
    *,
    include_labels: bool,
    train: bool,
    device: torch.device,
):
    return create_dataloader(
        manifest_path=args.manifest,
        split=split,
        data_root=args.data_root,
        sequence_length=args.sequence_length,
        stride=args.stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize_actions=True,
        include_observer=False,
        include_labels=include_labels,
        balanced_sampling=train,
        task_balanced_batches=train,
        seed=args.seed + (0 if split == "target_train" else 1),
        pin_memory=device.type == "cuda",
        drop_last_batch=train,
        cache_size=args.dataset_cache_size,
    )


def save_checkpoint(
    path: Path,
    model: HeMACHISSD,
    source_payload: dict[str, Any],
    task_classifier: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    epoch: int,
    train_metrics: dict[str, float],
    validation_metrics: dict[str, float],
) -> None:
    payload = dict(source_payload)
    payload.update(
        {
            "model_config": model.config(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "target_task_classifier_state_dict": task_classifier.state_dict(),
            "adaptation": {
                "method": (
                    "task_specific_adapter_with_shared_decoder_and_value"
                    if args.adapt_shared_decoder
                    else "conservative_task_specific_adapter_with_value"
                ),
                "source_checkpoint": str(args.checkpoint.expanduser().resolve()),
                "training_tasks": list(args.target_difficulties),
                "held_out_split": "target_test",
                "hyperparameters": vars(args),
            },
        }
    )
    payload["training_tasks"] = list(args.all_difficulties)
    payload["held_out_tasks"] = []
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)



def main() -> None:
    args = parse_args()
    validate_args(args)
    manifest = load_manifest(args.manifest)
    success_definition = manifest.get("success_definition", {})
    task_name = success_definition.get("name")
    supported_tasks = {"observer_goal_arrival", "drone_exploration"}
    if task_name not in supported_tasks:
        raise ValueError(
            "Target adaptation requires a mission or drone-exploration manifest, "
            f"but {args.manifest} defines {task_name or 'an unknown task'!r}."
        )
    args.source_difficulties = tuple(
        sorted(int(value) for value in manifest.get("source_difficulties", ()))
    )
    args.target_difficulties = tuple(
        sorted(int(value) for value in manifest.get("target_difficulties", ()))
    )
    if not args.source_difficulties or not args.target_difficulties:
        raise ValueError("Manifest must define non-empty source and target difficulties.")
    if set(args.source_difficulties).intersection(args.target_difficulties):
        raise ValueError("Source and target difficulties must not overlap.")
    args.all_difficulties = tuple(
        sorted((*args.source_difficulties, *args.target_difficulties))
    )
    expected_difficulties = tuple(range(1, max(args.all_difficulties) + 1))
    if args.all_difficulties != expected_difficulties:
        raise ValueError(
            "Difficulty IDs must be contiguous and start at 1; got "
            f"{args.all_difficulties}."
        )
    args.classifier_tasks = max(args.all_difficulties)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    configure_gpu_backend(device)
    model, source_payload = load_hissd_model(args.checkpoint, device)
    if model.skill_structure == "common_only":
        # This control intentionally has no task representation to adapt.
        args.descriptor_weight = 0.0
        args.task_classification_weight = 0.0
        args.query_contrastive_weight = 0.0
        args.decoder_skill_contrastive_weight = 0.0
        args.source_skill_distillation_weight = 0.0
        args.task_usage_weight = 0.0
        args.skill_warmup_epochs = 0
    source_hyperparameters = source_payload.get("hyperparameters", {})
    args.task_descriptor_scale = source_hyperparameters.get(
        "task_descriptor_scale",
        [1.0] * model.task_descriptor_dim,
    )
    checkpoint_source_tasks = tuple(
        sorted(int(value) for value in source_payload.get("training_tasks", ()))
    )
    if checkpoint_source_tasks != args.source_difficulties:
        raise ValueError(
            "Source checkpoint tasks do not match the manifest: "
            f"checkpoint={checkpoint_source_tasks}, "
            f"manifest={args.source_difficulties}."
        )
    teacher = FrozenTaskTeacher(
        task_observation_encoder=(
            copy.deepcopy(model.task_observation_encoder).to(device).eval()
            if model.task_observation_encoder is not None
            else None
        ),
        task_skill_encoder=copy.deepcopy(model.task_skill_encoder).to(device).eval(),
        action_decoder=copy.deepcopy(model.action_decoder).to(device).eval(),
        skill_structure=model.skill_structure,
    )
    model.enable_task_action_residual()
    for module in (
        teacher.task_observation_encoder,
        teacher.task_skill_encoder,
        teacher.action_decoder,
    ):
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.action_decoder.parameters():
        parameter.requires_grad_(args.adapt_shared_decoder)
    task_action_head = model.action_decoder.task_action_residual_head
    if task_action_head is not None:
        for parameter in task_action_head.parameters():
            parameter.requires_grad_(True)
    for parameter in model.action_decoder.base_action_head.parameters():
        parameter.requires_grad_(False)
    for module in (
        model.task_observation_encoder,
        model.task_skill_encoder,
        model.task_descriptor_head,
        model.value_network,
        model.central_state_encoder,
        model.value_mixer,
    ):
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    if model.skill_structure in {"shared", "common_only"}:
        for module in (
            model.task_observation_encoder,
            model.task_skill_encoder,
        ):
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
    task_classifier = nn.Linear(model.skill_dim, args.classifier_tasks).to(device)
    classifier_state = source_payload.get("target_task_classifier_state_dict")
    if classifier_state is not None:
        try:
            task_classifier.load_state_dict(classifier_state)
            print("Restored task classifier from the source adaptation checkpoint.")
        except RuntimeError:
            print(
                "Skipped incompatible task classifier from the source checkpoint; "
                f"new classifier has {args.classifier_tasks} outputs."
            )
            classifier_state = None
    if classifier_state is None and model.task_classifier_head is not None:
        source_classifier = model.task_classifier_head
        source_rows = min(
            source_classifier.out_features,
            len(args.source_difficulties),
            task_classifier.out_features,
        )
        with torch.no_grad():
            task_classifier.weight[:source_rows].copy_(
                source_classifier.weight[:source_rows]
            )
            task_classifier.bias[:source_rows].copy_(
                source_classifier.bias[:source_rows]
            )
        print(
            "Initialized the expanded task classifier with "
            f"{source_rows} source-task rows; target rows remain trainable."
        )

    target_dataset, target_loader = build_loader(
        args, "target_train", include_labels=True, train=True, device=device
    )
    source_dataset, source_loader = build_loader(
        args, "source_train", include_labels=False, train=True, device=device
    )
    val_dataset, val_loader = build_loader(
        args, "target_val", include_labels=True, train=False, device=device
    )
    _, source_val_loader = build_loader(
        args, "source_val", include_labels=False, train=False, device=device
    )
    target_descriptor_dim = int(target_dataset[0]["task_descriptor"].numel())
    if (
        model.skill_structure != "common_only"
        and model.task_descriptor_dim != target_descriptor_dim
    ):
        raise ValueError(
            "Source checkpoint and target dataset use different task descriptor "
            f"dimensions: checkpoint={model.task_descriptor_dim}, "
            f"dataset={target_descriptor_dim}. Retrain the source HiSSD checkpoint "
            "with the same realized-task descriptor schema before adaptation."
        )
    target_tasks = {int(entry["difficulty"]) for entry in target_dataset.entries}
    if target_tasks != set(args.target_difficulties):
        raise ValueError(
            "target_train difficulties do not match the manifest: "
            f"expected {args.target_difficulties}, got {sorted(target_tasks)}"
        )
    target_success_counts = {
        difficulty: sum(
            int(entry["difficulty"]) == difficulty
            and entry["category"] == "success"
            for entry in target_dataset.entries
        )
        for difficulty in args.target_difficulties
    }

    context_adapter = (
        getattr(model.task_skill_encoder, "context_adapter", None)
        if model.skill_structure in {"split", "task_only"}
        else None
    )
    task_action_parameters = list(
        context_adapter.parameters() if context_adapter is not None else ()
    )
    if task_action_head is not None:
        task_action_parameters.extend(task_action_head.parameters())
    task_action_parameter_ids = {
        id(parameter) for parameter in task_action_parameters
    }
    action_parameters = [
        parameter
        for parameter in model.action_decoder.parameters()
        if parameter.requires_grad
        and id(parameter) not in task_action_parameter_ids
    ]
    action_parameter_ids = {id(parameter) for parameter in action_parameters}
    task_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
        and id(parameter) not in action_parameter_ids
        and id(parameter) not in task_action_parameter_ids
    ]
    classifier_parameters = list(task_classifier.parameters())
    value_parameter_ids = {
        id(parameter)
        for module in (
            model.value_network,
            model.central_state_encoder,
            model.value_mixer,
        )
        for parameter in module.parameters()
    }
    value_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) in value_parameter_ids
    ]
    task_parameters = [
        parameter
        for parameter in task_parameters
        if id(parameter) not in value_parameter_ids
    ]
    parameters = (
        action_parameters
        + task_action_parameters
        + task_parameters
        + classifier_parameters
        + value_parameters
    )
    optimizer_groups = [
        {
            "params": action_parameters,
            "lr": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        {
            "params": task_parameters,
            "lr": args.task_learning_rate,
            "weight_decay": args.weight_decay,
        },
        {
            "params": classifier_parameters,
            "lr": args.classifier_learning_rate,
            "weight_decay": args.weight_decay,
        },
        {
            "params": value_parameters,
            "lr": args.value_learning_rate,
            "weight_decay": args.weight_decay,
        },
    ]
    if task_action_parameters:
        optimizer_groups.append(
            {
                "params": task_action_parameters,
                "lr": args.task_action_learning_rate,
                "weight_decay": args.weight_decay,
            }
        )
    optimizer = torch.optim.AdamW(
        optimizer_groups
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Adaptation source={args.checkpoint}, output={args.output_dir}, "
        f"manifest={args.manifest}, task={task_name}"
    )
    print(f"Target success episodes={target_success_counts}")
    missing_success_tasks = [
        difficulty
        for difficulty, count in target_success_counts.items()
        if count == 0
    ]
    if missing_success_tasks:
        print(
            "WARNING: no successful trajectories for target difficulties "
            f"{missing_success_tasks}; adaptation cannot learn successful behavior "
            "for those tasks until the dataset is recollected."
        )
    print(
        f"device={device}, target_train_windows={len(target_dataset)}, "
        f"source_replay_windows={len(source_dataset)}, val_windows={len(val_dataset)}, "
        f"trainable_parameters={sum(p.numel() for p in parameters):,}"
    )
    print(
        "Target labels are training weights only; target_test remains held out. "
        f"outcome_weights={args.success_weight}/"
        f"{args.goal_found_failure_weight}/{args.goal_not_found_weight}"
    )
    print(
        f"target_skill_learning=enabled, warmup_epochs={args.skill_warmup_epochs}, "
        f"shared_decoder={'trainable' if args.adapt_shared_decoder else 'frozen'}, "
        f"decoder_lr={args.learning_rate:.2e}, "
        f"task_action_lr={args.task_action_learning_rate:.2e}, "
        f"representation_lr={args.task_learning_rate:.2e}, "
        f"classifier_lr={args.classifier_learning_rate:.2e}, "
        f"value_lr={args.value_learning_rate:.2e}, "
        f"classifier_tasks={args.classifier_tasks}"
    )

    best_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        action_enabled = epoch > args.skill_warmup_epochs
        if args.adapt_shared_decoder:
            for parameter in model.action_decoder.parameters():
                parameter.requires_grad_(action_enabled)
        elif task_action_head is not None:
            for parameter in task_action_head.parameters():
                parameter.requires_grad_(action_enabled)
        for parameter in model.action_decoder.base_action_head.parameters():
            parameter.requires_grad_(False)
        train_metrics = run_train_epoch(
            model,
            teacher,
            task_classifier,
            target_loader,
            source_loader,
            optimizer,
            device,
            args,
            epoch,
        )
        val_metrics = run_validation(
            model,
            teacher,
            task_classifier,
            val_loader,
            source_val_loader,
            device,
            args,
        )
        val_metrics["source_validation_mse"] = run_source_validation(
            model, teacher, source_val_loader, device, args
        )
        difficulty_success_mse = sum(
            val_metrics[f"difficulty_{difficulty}/safe_success_mse"]
            for difficulty in args.target_difficulties
        ) / len(args.target_difficulties)
        selection_loss = (
            difficulty_success_mse
            + args.target_anchor_weight * val_metrics["target_anchor_mse"]
            + args.source_distillation_weight
            * val_metrics["source_validation_mse"]
            + 0.02 * val_metrics["target_descriptor_mae"]
            + 0.005
            * (
                2.0
                - val_metrics["target_task_accuracy"]
                - val_metrics["source_task_accuracy"]
            )
            + args.task_usage_weight * val_metrics["task_usage_loss"]
            + args.value_selection_weight * val_metrics["value_loss"]
            + 0.0001
            * (
                val_metrics["query_contrastive_loss"]
                + val_metrics["decoder_contrastive_loss"]
            )
        )
        success_by_task = "/".join(
            f"{val_metrics[f'difficulty_{difficulty}/success_mse']:.5f}"
            for difficulty in args.target_difficulties
        )
        safe_success_by_task = "/".join(
            f"{val_metrics[f'difficulty_{difficulty}/safe_success_mse']:.5f}"
            for difficulty in args.target_difficulties
        )
        print(
            f"epoch={epoch:03d} "
            f"target={train_metrics['target_weighted_mse']:.5f}/"
            f"{val_metrics['target_weighted_mse']:.5f} "
            f"safe_success={val_metrics['safe_success_mse']:.5f} "
            f"success={val_metrics['success_mse']:.5f} "
            f"goal_fail={val_metrics['goal_found_failure_mse']:.5f} "
            f"not_found={val_metrics['goal_not_found_mse']:.5f} "
            f"success_by_task={success_by_task} "
            f"safe_success_by_task={safe_success_by_task} "
            f"source_drift={val_metrics['source_validation_mse']:.6f} "
            f"target_drift={val_metrics['target_anchor_mse']:.6f} "
            f"descriptor={val_metrics['target_descriptor_mae']:.4f}/"
            f"{val_metrics['source_descriptor_mae']:.4f} "
            f"task_acc={val_metrics['target_task_accuracy']:.3f}/"
            f"{val_metrics['source_task_accuracy']:.3f} "
            f"contrast={val_metrics['query_contrastive_loss']:.3f}/"
            f"{val_metrics['decoder_contrastive_loss']:.3f} "
            f"margin={val_metrics['query_similarity_margin']:.3f}/"
            f"{val_metrics['decoder_similarity_margin']:.3f} "
            f"skill_drift={val_metrics['source_skill_distillation_mse']:.6f} "
            f"skill_effect={val_metrics['task_action_effect']:.4f}/"
            f"{val_metrics['common_action_effect']:.4f} "
            f"direct_task={val_metrics['direct_task_logit_effect']:.4f} "
            f"value={val_metrics['value_loss']:.5f} "
            f"task_gain={val_metrics['task_reconstruction_gain']:+.5f} "
            f"phase={'joint' if action_enabled else 'skill_warmup'}"
        )
        save_checkpoint(
            args.output_dir / "hissd_adapted_last.pt",
            model,
            source_payload,
            task_classifier,
            optimizer,
            args,
            epoch,
            train_metrics,
            val_metrics,
        )
        if action_enabled and selection_loss < best_loss:
            best_loss = selection_loss
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(
                args.output_dir / "hissd_adapted_best.pt",
                model,
                source_payload,
                task_classifier,
                optimizer,
                args,
                epoch,
                train_metrics,
                val_metrics,
            )
        elif action_enabled:
            stale_epochs += 1
        if args.early_stopping_patience and stale_epochs >= args.early_stopping_patience:
            print(
                "Early stopping: target success validation did not improve for "
                f"{args.early_stopping_patience} epochs."
            )
            break
    print(
        f"Best adapted target validation loss: {best_loss:.6f} at epoch "
        f"{best_epoch} ({args.output_dir / 'hissd_adapted_best.pt'})"
    )


if __name__ == "__main__":
    main()
