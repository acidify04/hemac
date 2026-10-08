#!/usr/bin/env python3
"""Train ONE heterogeneous HiSSD + cross-agent contextual adapter FROM SCRATCH.

Source configurations:
    D3O1 and D4O2

This is the initialization ablation for the BC-initialized joint HiSSD+Adapter
trainer. It keeps the same synchronized data, objectives, role-specific I/O,
shared HiSSD stack, and adapter architecture, but loads NO BC checkpoint.

Architecture:
    role-specific observation encoder E_r(o_i) -> h_i
    shared CommonSkillEncoder C(h_i)          -> raw common skill c_i
    joint context g_i = SelfAttention([h_1,c_1],...,[h_N,c_N])_i
    ONE shared adapter A([c_i, h_i, g_i])     -> delta_i
    adapted common skill c'_i = c_i + delta_i
    role-specific action decoder              -> a_i

Drone and Observer are processed in the SAME episode/minibatch. Their action
losses are averaged with equal role weight, so the larger Drone population does
not dominate imitation learning. D3O1 and D4O2 contribute equal minibatch counts.

No adapter-specific loss is introduced. The adapter learns only through the
existing HiSSD controller/planner objectives.

Important:
- Outcome-balanced sampling is OFF by default. This preserves the natural
  trajectory distribution in the newly collected source datasets.
- Early stopping is OFF by default. Checkpoints are still selected by validation.
- During task warmup, encoded role features are detached before the task-skill
  branch exactly as in the matched BC-initialized trainer.
- Role-specific observation encoders and action decoders are randomly initialized.
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
from skill_discovery.dataset import create_dataloader
from skill_discovery import evaluate_hissd_joint_hetero_cross_agent_adapter_zero_shot as rollout_eval
from skill_discovery.hissd_joint_hetero_cross_agent_adapter_models import (
    JointHeterogeneousCrossAgentAdapterHiSSD,
)
from skill_discovery.multi_source_hissd import (
    SourceSpec,
    balanced_source_batches,
)

ROLE_ORDER = ("observer", "drone")


# ---------------------------------------------------------------------------
# CLI / shared HiSSD defaults
# ---------------------------------------------------------------------------

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
    p.add_argument(
        "--batches-per-source",
        type=int,
        default=0,
        help=(
            "Equal train batches from EACH source per epoch. "
            "0 uses max(len(D3O1 loader), len(D4O2 loader))."
        ),
    )

    p.add_argument(
        "--balanced-sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Outcome-balanced sampling within each source. Default=True to match "
            "the current Joint HiSSD scratch / BC-initialized comparison runs."
        ),
    )

    p.add_argument("--adapter-hidden-dim", type=int, default=128)
    p.add_argument(
        "--cross-agent-context-dim",
        type=int,
        default=64,
        help="Width of the projected [h_i,c_i] token used for agent self-attention.",
    )
    p.add_argument(
        "--cross-agent-attention-heads",
        type=int,
        default=4,
        help="Number of heads in cross-agent self-attention.",
    )
    p.add_argument(
        "--cross-agent-attention-dropout",
        type=float,
        default=0.0,
        help="Dropout inside cross-agent MultiheadAttention.",
    )
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )

    # Optional overrides; otherwise inherit the installed HiSSD defaults.
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--task-learning-rate-multiplier", type=float)
    p.add_argument("--weight-decay", type=float)
    p.add_argument("--task-weight-decay", type=float)
    p.add_argument("--task-warmup-epochs", type=int)
    p.add_argument("--task-contrastive-weight", type=float)
    p.add_argument("--task-action-contrastive-weight", type=float)

    # Explicitly disabled by default for this one-task D1 experiment.
    p.add_argument("--early-stopping-patience", type=int, default=0)

    p.add_argument("--max-train-batches", type=int)
    p.add_argument("--max-val-batches", type=int)
    p.add_argument(
        "--rollout-eval-every",
        type=int,
        default=5,
        help=(
            "Run source-only D3O1/D4O2 mission rollouts every N epochs. "
            "0 disables rollout diagnostics. Rollouts never affect checkpoint selection."
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
            "Fixed diagnostic seed base, separate from final zero-shot evaluation "
            "(which uses 100000000)."
        ),
    )
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def configure_training_args(cli: argparse.Namespace) -> argparse.Namespace:
    """Reuse the installed HiSSD hyperparameters, then apply joint overrides."""
    # Shared skill/value/planner stack: use the repository's Drone defaults as
    # the canonical base. Role-specific BC policies still initialize both I/O
    # branches independently.
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

    # Do not inherit the old task-validation early stopping behavior.
    args.early_stopping_patience = int(cli.early_stopping_patience)

    if hasattr(base, "configure_ablation"):
        base.configure_ablation(args)

    if args.skill_structure != "split":
        raise ValueError(
            "Joint heterogeneous adapter experiment requires "
            f"skill_structure='split', got {args.skill_structure!r}."
        )
    return args


# ---------------------------------------------------------------------------
# Joint data loading
# ---------------------------------------------------------------------------

def create_joint_dataloader(
    source: SourceSpec,
    split: str,
    cli: argparse.Namespace,
    *,
    seed: int,
    pin_memory: bool,
):
    """Load complete joint episodes, including BOTH Observer and Drone."""
    training = split == "source_train"
    return create_dataloader(
        manifest_path=source.manifest,
        split=split,
        data_root=source.data_root,
        sequence_length=cli.sequence_length,
        stride=cli.stride,
        batch_size=cli.batch_size,
        num_workers=cli.num_workers,
        normalize_actions=True,
        include_observer=True,
        include_labels=False,
        balanced_sampling=(
            bool(cli.balanced_sampling) if training else False
        ),
        task_balanced_batches=True,
        seed=seed,
        pin_memory=pin_memory,
        drop_last_batch=training,
        cache_size=cli.dataset_cache_size,
    )


def _move_observations(
    observations: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        name: observations[name].to(device, non_blocking=True)
        for name in ("global_map", "local_map", "action_history")
    }


def prepare_joint_batch(
    raw_batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Convert one collated DnOm batch into joint heterogeneous tensors.

    HeMAC ordering is Observer(s) first, then Drone(s), which matches the joint
    model's canonical ROLE_ORDER.
    """
    for role in ROLE_ORDER:
        if role not in raw_batch:
            raise KeyError(
                f"Joint batch is missing role {role!r}. "
                "Use include_observer=True when creating the loader."
            )

    observer_actions = raw_batch["observer"]["actions"].to(
        device, non_blocking=True
    )
    drone_actions = raw_batch["drone"]["actions"].to(
        device, non_blocking=True
    )
    observer_count = int(observer_actions.shape[-2])
    drone_count = int(drone_actions.shape[-2])

    full_mask = raw_batch["agent_mask"].to(
        device, non_blocking=True
    ).bool()
    if int(full_mask.shape[-1]) != observer_count + drone_count:
        raise RuntimeError(
            "Joint agent_mask does not match Observer+Drone counts: "
            f"mask={full_mask.shape[-1]}, O={observer_count}, D={drone_count}."
        )

    filled = raw_batch["filled"].to(
        device, non_blocking=True
    ).bool()
    if filled.ndim == full_mask.ndim:
        if filled.shape[-1] != 1:
            raise ValueError(
                f"Expected filled[...,1], got {tuple(filled.shape)}"
            )
        filled = filled.squeeze(-1)
    if filled.shape != full_mask.shape[:-1]:
        raise ValueError(
            f"filled shape {tuple(filled.shape)} does not match "
            f"agent_mask prefix {tuple(full_mask.shape[:-1])}."
        )

    observer_mask = (
        full_mask[..., :observer_count] & filled.unsqueeze(-1)
    )
    drone_mask = (
        full_mask[
            ..., observer_count : observer_count + drone_count
        ]
        & filled.unsqueeze(-1)
    )
    joint_mask = torch.cat(
        (observer_mask, drone_mask),
        dim=-1,
    )

    terminated = raw_batch["terminated"].to(
        device, non_blocking=True
    ).bool()
    truncated = raw_batch["truncated"].to(
        device, non_blocking=True
    ).bool()
    if terminated.ndim == filled.ndim + 1:
        terminated = terminated.squeeze(-1)
    if truncated.ndim == filled.ndim + 1:
        truncated = truncated.squeeze(-1)

    if "team_reward" in raw_batch:
        team_reward = raw_batch["team_reward"].to(
            device, non_blocking=True
        )
    elif "drone_task_reward" in raw_batch:
        team_reward = raw_batch["drone_task_reward"].to(
            device, non_blocking=True
        )
    else:
        raise KeyError(
            "Dataset contains neither team_reward nor drone_task_reward."
        )
    if team_reward.ndim == filled.ndim + 1:
        team_reward = team_reward.squeeze(-1)

    batch = {
        "observations_by_role": {
            role: _move_observations(
                raw_batch[role]["observations"], device
            )
            for role in ROLE_ORDER
        },
        "next_observations_by_role": {
            role: _move_observations(
                raw_batch[role]["next_observations"], device
            )
            for role in ROLE_ORDER
        },
        "actions_by_role": {
            "observer": observer_actions,
            "drone": drone_actions,
        },
        "valid_agents_by_role": {
            "observer": observer_mask,
            "drone": drone_mask,
        },
        "valid_agents": joint_mask,
        "valid_steps": filled,
        "central_map": raw_batch["global_state"]["central_map"].to(
            device, non_blocking=True
        ),
        "next_central_map": raw_batch["next_global_state"][
            "central_map"
        ].to(device, non_blocking=True),
        "team_reward": team_reward,
        "task_descriptor": raw_batch["task_descriptor"].to(
            device, non_blocking=True
        ),
        "task_descriptor_available": raw_batch[
            "task_descriptor_available"
        ].to(device, non_blocking=True).bool(),
        "task_id": raw_batch["task_id"].to(
            device, non_blocking=True
        ),
        "task_supervision": (
            raw_batch["window_start"].to(
                device, non_blocking=True
            )
            == 0
        ),
        "done": terminated | truncated,
        "role_counts": {
            "observer": observer_count,
            "drone": drone_count,
        },
    }
    return batch


# ---------------------------------------------------------------------------
# Model/data compatibility
# ---------------------------------------------------------------------------

def _sample_role_schema(
    sample: dict[str, Any],
    role: str,
) -> dict[str, Any]:
    observations = sample[role]["observations"]
    actions = sample[role]["actions"]
    return {
        "global_map_channels": int(
            observations["global_map"].shape[-3]
        ),
        "local_map_channels": int(
            observations["local_map"].shape[-3]
        ),
        "global_map_size": tuple(
            int(x) for x in observations["global_map"].shape[-2:]
        ),
        "local_map_size": tuple(
            int(x) for x in observations["local_map"].shape[-2:]
        ),
        "action_history_shape": tuple(
            int(x)
            for x in observations["action_history"].shape[-2:]
        ),
        "action_dim": int(actions.shape[-1]),
        "agent_count": int(actions.shape[-2]),
    }


def validate_joint_source_compatibility(
    datasets: Mapping[str, Any],
    model: JointHeterogeneousCrossAgentAdapterHiSSD,
) -> dict[str, dict[str, int]]:
    """Validate role schemas and allow only population count to vary."""
    population_counts: dict[str, dict[str, int]] = {}

    reference_central = None
    for source_name, dataset in datasets.items():
        sample = dataset[0]
        population_counts[source_name] = {}

        central_shape = tuple(
            int(x)
            for x in sample["global_state"]["central_map"].shape[-3:]
        )
        if reference_central is None:
            reference_central = central_shape
        elif central_shape != reference_central:
            raise ValueError(
                "Central-map schema differs across source populations: "
                f"{reference_central} vs {central_shape}."
            )

        for role in ROLE_ORDER:
            schema = _sample_role_schema(sample, role)
            population_counts[source_name][role] = schema.pop(
                "agent_count"
            )
            expected = dict(model.role_configs[role])
            for key, actual in schema.items():
                target = expected[key]
                if isinstance(target, list):
                    target = tuple(target)
                if isinstance(actual, tuple):
                    target = tuple(target)
                if actual != target:
                    raise ValueError(
                        f"{source_name}/{role} schema mismatch for {key}: "
                        f"dataset={actual}, model={target}"
                    )

    assert reference_central is not None
    if int(reference_central[0]) != model.central_map_channels:
        raise ValueError(
            "Central-map channels differ from model: "
            f"data={reference_central[0]}, "
            f"model={model.central_map_channels}"
        )
    if tuple(reference_central[-2:]) != tuple(
        model.central_map_size
    ):
        raise ValueError(
            "Central-map size differs from model: "
            f"data={reference_central[-2:]}, "
            f"model={model.central_map_size}"
        )
    return population_counts


# ---------------------------------------------------------------------------
# Joint HiSSD objectives
# ---------------------------------------------------------------------------

def joint_controller_objective(
    model: JointHeterogeneousCrossAgentAdapterHiSSD,
    batch: dict[str, Any],
    args: argparse.Namespace,
    *,
    task_only: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """HiSSD controller/task objective with equal Observer/Drone action weight."""
    features, valid_mask, role_slices = (
        model.encode_joint_observations(
            batch["observations_by_role"],
            batch["valid_agents_by_role"],
        )
    )

    # Existing separate-role HiSSD used a separate task observation pathway.
    # The joint model deliberately has only role-specific actor encoders, so
    # detach their features before task auxiliary losses. This prevents task
    # warmup from drifting the BC-initialized actor encoders by itself.
    skill_outputs = model.infer_skills_with_adapter(
        features,
        valid_mask,
        task_observation_features=features.detach(),
    )
    common = skill_outputs["common_skills"]
    task_skill = skill_outputs["task_skills"]
    query = skill_outputs["contrastive_skills"]

    decoder_task_skill = (
        task_skill.detach()
        if args.detach_task_skill_for_action
        else task_skill
    )
    predictions = model.decode_joint_actions(
        features,
        common,
        decoder_task_skill,
        role_slices,
    )

    role_action_losses: dict[str, torch.Tensor] = {}
    for role in ROLE_ORDER:
        role_action_losses[role] = base.masked_action_mse(
            predictions[role],
            batch["actions_by_role"][role],
            batch["valid_agents_by_role"][role],
        )

    # Explicit role balancing. D4O2 has 4 drones and 2 observers, but each role
    # contributes exactly 50% of the action reconstruction objective.
    action_loss = 0.5 * (
        role_action_losses["observer"]
        + role_action_losses["drone"]
    )

    (
        descriptor_loss,
        metric_loss,
        variance_loss,
        descriptor_metrics,
    ) = base.continuous_task_objective(
        model, query, batch, args
    )
    contrastive_loss, contrastive_metrics = (
        base.task_contrastive_objective(
            model, query, batch, args
        )
    )
    (
        task_action_descriptor_loss,
        task_action_variance_loss,
        task_action_metrics,
    ) = base.task_action_skill_objective(
        model, task_skill, batch, args
    )
    (
        task_action_contrastive_loss,
        raw_task_action_contrastive_metrics,
    ) = base.task_contrastive_objective(
        model, task_skill, batch, args
    )
    task_action_contrastive_metrics = {
        name.replace(
            "task_contrastive",
            "task_action_contrastive",
            1,
        ): value
        for name, value
        in raw_task_action_contrastive_metrics.items()
    }

    task_objective = (
        args.descriptor_weight * descriptor_loss
        + args.descriptor_metric_weight * metric_loss
        + args.task_variance_weight * variance_loss
        + args.task_contrastive_weight * contrastive_loss
        + args.task_action_descriptor_weight
        * task_action_descriptor_loss
        + args.task_action_variance_weight
        * task_action_variance_loss
        + args.task_action_contrastive_weight
        * task_action_contrastive_loss
    )

    total = (
        task_objective
        if task_only
        else action_loss + task_objective
    )

    metrics = {
        "controller_loss": float(total.detach()),
        "action_mse": float(action_loss.detach()),
        "observer_action_mse": float(
            role_action_losses["observer"].detach()
        ),
        "drone_action_mse": float(
            role_action_losses["drone"].detach()
        ),
        "weighted_descriptor_loss": float(
            (args.descriptor_weight * descriptor_loss).detach()
        ),
        "weighted_descriptor_metric_loss": float(
            (
                args.descriptor_metric_weight * metric_loss
            ).detach()
        ),
        "weighted_task_variance_loss": float(
            (args.task_variance_weight * variance_loss).detach()
        ),
        "weighted_task_contrastive_loss": float(
            (
                args.task_contrastive_weight
                * contrastive_loss
            ).detach()
        ),
        "weighted_task_action_descriptor_loss": float(
            (
                args.task_action_descriptor_weight
                * task_action_descriptor_loss
            ).detach()
        ),
        "weighted_task_action_variance_loss": float(
            (
                args.task_action_variance_weight
                * task_action_variance_loss
            ).detach()
        ),
        "weighted_task_action_contrastive_loss": float(
            (
                args.task_action_contrastive_weight
                * task_action_contrastive_loss
            ).detach()
        ),
        "common_skill_std": float(
            base.skill_standard_deviation(
                common, valid_mask
            ).detach()
        ),
        "task_skill_std": float(
            base.skill_standard_deviation(
                task_skill, valid_mask
            ).detach()
        ),
        **descriptor_metrics,
        **contrastive_metrics,
        **task_action_metrics,
        **task_action_contrastive_metrics,
    }
    return total, metrics


def joint_value_predictions(
    model: JointHeterogeneousCrossAgentAdapterHiSSD,
    batch: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features, valid_mask, _ = model.encode_joint_observations(
        batch["observations_by_role"],
        batch["valid_agents_by_role"],
    )
    value = model.total_value(
        features,
        batch["central_map"],
        valid_mask.to(features.dtype),
    ).squeeze(-1)

    with torch.no_grad():
        next_features, next_valid_mask, _ = (
            model.encode_joint_observations(
                batch["next_observations_by_role"],
                batch["valid_agents_by_role"],
                target=True,
            )
        )
        target_next_value = model.total_value(
            next_features,
            batch["next_central_map"],
            next_valid_mask.to(next_features.dtype),
            target=True,
        ).squeeze(-1)

        scaled_reward = (
            batch["team_reward"] / args.reward_scale
        )
        target = (
            scaled_reward
            + args.gamma
            * (~batch["done"]).float()
            * target_next_value
        )

    return value, target, target - value


def joint_value_objective(
    model: JointHeterogeneousCrossAgentAdapterHiSSD,
    batch: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    value, target, residual = joint_value_predictions(
        model, batch, args
    )
    expectile_weight = torch.abs(
        args.expectile
        - (residual.detach() < 0).to(residual.dtype)
    )
    mask = batch["valid_steps"].to(residual.dtype)
    denominator = mask.sum().clamp_min(1.0)
    loss = (
        expectile_weight
        * residual.square()
        * mask
    ).sum() / denominator

    return loss, {
        "value_loss": float(loss.detach()),
        "value_mean": float(
            (value.detach() * mask).sum() / denominator
        ),
        "target_value_mean": float(
            (target.detach() * mask).sum() / denominator
        ),
        "td_residual_mean": float(
            (residual.detach() * mask).sum() / denominator
        ),
    }


def _role_balanced_local_prediction_error(
    predicted_local: torch.Tensor,
    target_local: torch.Tensor,
    valid_mask: torch.Tensor,
    role_slices: Mapping[str, slice],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Average local-prediction error within role, then equally across roles."""
    squared = (
        predicted_local - target_local
    ).square().mean(dim=-1)

    by_role: dict[str, torch.Tensor] = {}
    for role in ROLE_ORDER:
        role_slice = role_slices[role]
        error = squared[..., role_slice]
        mask = valid_mask[..., role_slice].to(error.dtype)
        by_role[role] = (
            (error * mask).sum(dim=2)
            / mask.sum(dim=2).clamp_min(1.0)
        )

    role_balanced = 0.5 * (
        by_role["observer"] + by_role["drone"]
    )
    return role_balanced, by_role


def joint_planner_objective(
    model: JointHeterogeneousCrossAgentAdapterHiSSD,
    batch: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    """HiSSD Eq.7-style planner using the ADAPTED common skill."""
    with torch.no_grad():
        features, valid_mask, role_slices = (
            model.encode_joint_observations(
                batch["observations_by_role"],
                batch["valid_agents_by_role"],
            )
        )
        features = features.detach()
        target_next_features, _, _ = (
            model.encode_joint_observations(
                batch["next_observations_by_role"],
                batch["valid_agents_by_role"],
                target=True,
            )
        )

    # The adapter is deliberately in the planner path as well as the controller
    # path: every downstream consumer sees c'_i, while raw c_i is diagnostic only.
    raw_common = model.common_skill_encoder(
        features, valid_mask
    )
    planning_skill, _ = model.common_skill_adapter(
        raw_common,
        features,
        valid_mask,
    )

    predicted_central, predicted_local = (
        model.forward_predictor(
            planning_skill,
            valid_mask.to(features.dtype),
        )
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

        scaled_reward = (
            batch["team_reward"] / args.reward_scale
        )
        residual = (
            scaled_reward
            + args.gamma
            * (~batch["done"]).float()
            * predicted_next_value
            - current_value
        )
        advantage_weight = torch.exp(
            residual / args.alpha
        ).clamp(max=100.0)

    central_error = (
        predicted_central
        - batch["next_central_map"]
    ).square().mean(dim=(2, 3, 4))

    (
        local_error,
        local_error_by_role,
    ) = _role_balanced_local_prediction_error(
        predicted_local,
        target_next_features.detach(),
        valid_mask,
        role_slices,
    )

    prediction_error = central_error + local_error
    step_mask = batch["valid_steps"].to(
        prediction_error.dtype
    )
    denominator = step_mask.sum().clamp_min(1.0)
    weighted_error = (
        prediction_error
        * advantage_weight.detach()
        * step_mask
    )
    loss = weighted_error.sum() / denominator

    metrics = {
        "planner_loss": float(loss.detach()),
        "central_prediction_mse": float(
            (
                central_error.detach()
                * step_mask
            ).sum()
            / denominator
        ),
        "local_prediction_mse": float(
            (
                local_error.detach()
                * step_mask
            ).sum()
            / denominator
        ),
        "observer_local_prediction_mse": float(
            (
                local_error_by_role["observer"].detach()
                * step_mask
            ).sum()
            / denominator
        ),
        "drone_local_prediction_mse": float(
            (
                local_error_by_role["drone"].detach()
                * step_mask
            ).sum()
            / denominator
        ),
        "advantage_weight_mean": float(
            (
                advantage_weight.detach()
                * step_mask
            ).sum()
            / denominator
        ),
        "advantage_weight_max": float(
            advantage_weight.detach().max()
        ),
        "planner_predicted_value_mean": float(
            (
                predicted_next_value
                * step_mask
            ).sum()
            / denominator
        ),
        "planner_td_residual_mean": float(
            (residual * step_mask).sum()
            / denominator
        ),
    }
    return loss, metrics


# ---------------------------------------------------------------------------
# Metrics / diagnostics
# ---------------------------------------------------------------------------

def add_metrics(
    accumulator: dict[str, float],
    metrics: Mapping[str, float],
) -> None:
    for key, value in metrics.items():
        accumulator[key] += float(value)


def average_metrics(
    accumulator: Mapping[str, float],
    count: int,
) -> dict[str, float]:
    if count <= 0:
        raise RuntimeError("No batches were processed.")
    return {
        key: float(value) / count
        for key, value in accumulator.items()
    }


def equal_source_average(
    by_source: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    if not by_source:
        raise RuntimeError("No source metrics were produced.")
    keys = set.intersection(
        *(set(metrics) for metrics in by_source.values())
    )
    return {
        key: sum(
            metrics[key] for metrics in by_source.values()
        )
        / len(by_source)
        for key in keys
    }


@torch.inference_mode()
def initialization_diagnostics(
    model: JointHeterogeneousCrossAgentAdapterHiSSD,
    raw_batch: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    batch = prepare_joint_batch(raw_batch, device)
    features, valid_mask, role_slices = (
        model.encode_joint_observations(
            batch["observations_by_role"],
            batch["valid_agents_by_role"],
        )
    )
    raw_common = model.common_skill_encoder(
        features, valid_mask
    )
    adapted_common, _ = model.common_skill_adapter(
        raw_common, features, valid_mask
    )
    task_skill, _ = model.task_skill_encoder(
        features.detach(), valid_mask
    )
    predictions = model.decode_joint_actions(
        features,
        adapted_common,
        task_skill,
        role_slices,
    )

    result = {
        "adapter_identity_error": float(
            (adapted_common - raw_common)
            .abs()
            .max()
            .item()
        )
    }
    for role in ROLE_ORDER:
        role_features = features[..., role_slices[role], :]
        bc_action = torch.tanh(
            model.action_decoder[role].base_action_head(
                role_features
            )
        )
        result[f"{role}_bc_action_max_difference"] = float(
            (
                predictions[role] - bc_action
            ).abs().max().item()
        )
    return result


def _masked_rms(
    tensor: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    numeric = mask.bool().unsqueeze(-1).to(tensor.dtype)
    denominator = (
        numeric.sum() * tensor.shape[-1]
    ).clamp_min(1.0)
    value = torch.sqrt(
        (tensor.square() * numeric).sum()
        / denominator
    )
    return float(value.item())


@torch.inference_mode()
def adapter_diagnostics(
    model: JointHeterogeneousCrossAgentAdapterHiSSD,
    raw_batch: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    batch = prepare_joint_batch(raw_batch, device)
    features, valid_mask, role_slices = (
        model.encode_joint_observations(
            batch["observations_by_role"],
            batch["valid_agents_by_role"],
        )
    )
    raw = model.common_skill_encoder(
        features, valid_mask
    )
    adapted, delta = model.common_skill_adapter(
        raw, features, valid_mask
    )

    overall_delta = _masked_rms(delta, valid_mask)
    overall_raw = _masked_rms(raw, valid_mask)

    result = {
        "adapter_identity_or_delta_max": float(
            (adapted - raw).abs().max().item()
        ),
        "adapter_delta_rms": overall_delta,
        "raw_common_rms": overall_raw,
        "adapter_to_common_ratio": (
            overall_delta / max(overall_raw, 1e-8)
        ),
    }

    for role in ROLE_ORDER:
        sl = role_slices[role]
        role_mask = valid_mask[..., sl]
        role_delta = delta[..., sl, :]
        role_raw = raw[..., sl, :]
        delta_rms = _masked_rms(
            role_delta, role_mask
        )
        raw_rms = _masked_rms(
            role_raw, role_mask
        )
        result[f"{role}_adapter_delta_rms"] = delta_rms
        result[f"{role}_adapter_ratio"] = (
            delta_rms / max(raw_rms, 1e-8)
        )
        result[f"{role}_adapter_delta_std"] = float(
            base.skill_standard_deviation(
                role_delta, role_mask
            ).item()
        )

    return result


def compact(metrics: Mapping[str, float]) -> str:
    def g(name: str, default: float = float("nan")) -> float:
        return float(metrics.get(name, default))

    return (
        f"action={g('action_mse'):.5f} "
        f"obs={g('observer_action_mse'):.5f} "
        f"drone={g('drone_action_mse'):.5f} "
        f"task={g('task_contrastive_loss'):.4f} "
        f"task_acc={g('task_contrastive_accuracy'):.3f} "
        f"value={g('value_loss'):.4f} "
        f"planner={g('planner_loss'):.4f}"
    )


def progress_write(message: str) -> None:
    """Print a persistent log line without overwriting active progress bars."""
    if tqdm is not None:
        tqdm.write(message)
    else:
        print(message, flush=True)


# ---------------------------------------------------------------------------
# Train / validation
# ---------------------------------------------------------------------------

def train_epoch(
    model: JointHeterogeneousCrossAgentAdapterHiSSD,
    loaders: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    *,
    epoch: int,
    batches_per_source: int | None,
    task_only: bool,
    show_progress: bool = False,
) -> tuple[
    dict[str, float],
    dict[str, dict[str, float]],
]:
    model.train()

    total = defaultdict(float)
    per_source = {
        name: defaultdict(float)
        for name in loaders
    }
    counts = defaultdict(int)
    processed = 0

    iterator = balanced_source_batches(
        dict(loaders),
        seed=args.seed + epoch,
        batches_per_source=batches_per_source,
    )

    effective_bps = (
        max(len(loader) for loader in loaders.values())
        if batches_per_source is None
        else batches_per_source
    )
    train_total = effective_bps * len(loaders)
    if args.max_train_batches is not None:
        train_total = min(train_total, args.max_train_batches)
    progress = None
    if tqdm is not None and show_progress:
        progress = tqdm(
            total=train_total,
            desc=f"epoch {epoch} train",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
            position=1,
        )

    for source_name, raw_batch in iterator:
        if (
            args.max_train_batches is not None
            and processed >= args.max_train_batches
        ):
            break

        batch = prepare_joint_batch(
            raw_batch, device
        )

        controller_loss, controller_metrics = (
            joint_controller_objective(
                model,
                batch,
                args,
                task_only=task_only,
            )
        )
        controller_metrics["controller_grad_norm"] = (
            base.optimize(
                controller_loss,
                model,
                optimizer,
                args.grad_clip,
            )
        )

        if task_only:
            with torch.no_grad():
                _, value_metrics = joint_value_objective(
                    model, batch, args
                )
                _, planner_metrics = (
                    joint_planner_objective(
                        model, batch, args
                    )
                )
            value_metrics["value_grad_norm"] = 0.0
            planner_metrics["planner_grad_norm"] = 0.0
            planner_metrics["planner_update_skipped"] = 0.0
        else:
            value_loss, value_metrics = (
                joint_value_objective(
                    model, batch, args
                )
            )
            value_metrics["value_grad_norm"] = (
                base.optimize(
                    value_loss,
                    model,
                    optimizer,
                    args.grad_clip,
                )
            )

            with base.strict_planner_math(device):
                planner_loss, planner_metrics = (
                    joint_planner_objective(
                        model, batch, args
                    )
                )
                planner_grad = base.optimize(
                    planner_loss,
                    model,
                    optimizer,
                    args.grad_clip,
                    skip_nonfinite=True,
                )

            planner_metrics[
                "planner_update_skipped"
            ] = float(
                not math.isfinite(planner_grad)
            )
            planner_metrics["planner_grad_norm"] = (
                planner_grad
                if math.isfinite(planner_grad)
                else 0.0
            )
            model.update_targets(args.target_tau)

        merged = {
            **controller_metrics,
            **value_metrics,
            **planner_metrics,
        }
        add_metrics(total, merged)
        add_metrics(
            per_source[source_name], merged
        )
        counts[source_name] += 1
        processed += 1
        if progress is not None:
            progress.update(1)
            progress.set_postfix(
                source=source_name,
                action=f"{controller_metrics['action_mse']:.4f}",
                refresh=False,
            )

    if progress is not None:
        progress.close()

    return (
        average_metrics(total, processed),
        {
            name: average_metrics(
                per_source[name], counts[name]
            )
            for name in loaders
            if counts[name] > 0
        },
    )


@torch.inference_mode()
def validate_one_source(
    model: JointHeterogeneousCrossAgentAdapterHiSSD,
    loader,
    device: torch.device,
    args: argparse.Namespace,
    *,
    source_name: str | None = None,
    progress=None,
) -> dict[str, float]:
    model.eval()
    accumulator = defaultdict(float)
    count = 0

    for raw_batch in loader:
        if (
            args.max_val_batches is not None
            and count >= args.max_val_batches
        ):
            break

        batch = prepare_joint_batch(
            raw_batch, device
        )
        _, controller = joint_controller_objective(
            model, batch, args
        )
        _, value = joint_value_objective(
            model, batch, args
        )
        _, planner = joint_planner_objective(
            model, batch, args
        )

        add_metrics(
            accumulator,
            {**controller, **value, **planner},
        )
        count += 1
        if progress is not None:
            progress.update(1)
            progress.set_postfix(
                source=source_name or "?",
                refresh=False,
            )

    return average_metrics(
        accumulator, count
    )


@torch.inference_mode()
def validate_all_sources(
    model: JointHeterogeneousCrossAgentAdapterHiSSD,
    loaders: Mapping[str, Any],
    device: torch.device,
    args: argparse.Namespace,
    *,
    epoch: int | None = None,
    show_progress: bool = False,
) -> tuple[
    dict[str, float],
    dict[str, dict[str, float]],
]:
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

    by_source = {}
    for name, loader in loaders.items():
        by_source[name] = validate_one_source(
            model,
            loader,
            device,
            args,
            source_name=name,
            progress=progress,
        )

    if progress is not None:
        progress.close()
    return equal_source_average(by_source), by_source


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    model: JointHeterogeneousCrossAgentAdapterHiSSD,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    args: argparse.Namespace,
    cli: argparse.Namespace,
    train_metrics: Mapping[str, float],
    train_by_source: Mapping[str, Mapping[str, float]],
    val_metrics: Mapping[str, float],
    val_by_source: Mapping[str, Mapping[str, float]],
    adapter_metrics: Mapping[str, float],
    population_counts: Mapping[str, Mapping[str, int]],
) -> None:
    path.parent.mkdir(
        parents=True, exist_ok=True
    )

    torch.save(
        {
            "format_version": 1,
            "model_type": (
                "hemac_joint_heterogeneous_hissd_"
                "cross_agent_context_adapter"
            ),
            "model_config": model.config(),
            "training_variant": "scratch_no_bc_initialization",
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
                "adapter_count": 1,
                "adapter_input": (
                    "[raw_common_skill, "
                    "encoded_agent_observation, "
                    "cross_agent_context]"
                ),
                "adapter_form": (
                    "agent_self_attention_plus_residual_mlp"
                ),
                "cross_agent_attention": True,
                "cross_agent_context_input": (
                    "[encoded_agent_observation, raw_common_skill]"
                ),
                "cross_agent_context_dim": (
                    cli.cross_agent_context_dim
                ),
                "cross_agent_attention_heads": (
                    cli.cross_agent_attention_heads
                ),
                "cross_agent_attention_dropout": (
                    cli.cross_agent_attention_dropout
                ),
                "adapter_hidden_dim": (
                    cli.adapter_hidden_dim
                ),
                "adapter_output": (
                    "cross_agent_conditioned_common_skill"
                ),
                "role_embedding": False,
                "agent_id_embedding": False,
                "population_embedding": False,
                "planner_uses_adapted_common_skill": True,
                "controller_uses_adapted_common_skill": True,
                "role_balanced_action_loss": True,
                "role_balanced_planner_local_loss": True,
            },
            "initialization": "scratch",
            "bc_initialization": None,
            "adapter_metrics": dict(
                adapter_metrics
            ),
            "training_metrics": dict(
                train_metrics
            ),
            "training_metrics_by_source": {
                name: dict(metrics)
                for name, metrics
                in train_by_source.items()
            },
            "validation_metrics": dict(
                val_metrics
            ),
            "validation_metrics_by_source": {
                name: dict(metrics)
                for name, metrics
                in val_by_source.items()
            },
            "multisource": {
                "task_identity": "difficulty_only",
                "configuration_balancing": (
                    "equal_batches_per_source"
                ),
                "outcome_balanced_sampling": bool(
                    cli.balanced_sampling
                ),
                "population_counts": {
                    name: dict(counts)
                    for name, counts
                    in population_counts.items()
                },
                "sources": {
                    "3-1": {
                        "manifest": str(
                            cli.manifest_31
                        ),
                        "data_root": str(
                            cli.data_root_31
                        ),
                    },
                    "4-2": {
                        "manifest": str(
                            cli.manifest_42
                        ),
                        "data_root": str(
                            cli.data_root_42
                        ),
                    },
                },
            },
            "training_args": vars(args),
        },
        path,
    )



def role_config_from_sample(
    sample: Mapping[str, Any],
    role: str,
) -> dict[str, Any]:
    """Infer the matched role I/O architecture directly from offline data."""
    payload = sample[role]
    observations = payload["observations"]
    actions = payload["actions"]
    return {
        "global_map_channels": int(
            observations["global_map"].shape[-3]
        ),
        "local_map_channels": int(
            observations["local_map"].shape[-3]
        ),
        "global_map_size": tuple(
            int(x)
            for x in observations["global_map"].shape[-2:]
        ),
        "local_map_size": tuple(
            int(x)
            for x in observations["local_map"].shape[-2:]
        ),
        "action_history_shape": tuple(
            int(x)
            for x in observations["action_history"].shape[-2:]
        ),
        # Match the role encoders used in the BC-initialized runs.
        "hidden_sizes": (96, 96),
        "activation": "relu",
        "action_dim": int(actions.shape[-1]),
    }


def validate_role_schemas_from_data(
    datasets: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Require D3O1/D4O2 to expose the same per-role I/O schema."""
    configs_by_source: dict[
        str, dict[str, dict[str, Any]]
    ] = {}
    for source_name, dataset in datasets.items():
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
                f"{reference_name}={reference}, "
                f"{source_name}={config}"
            )
    return reference



def build_source_rollout_configs(
    cli: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    """Build canonical D1 source environments for diagnostics only."""
    _, template31 = rollout_eval.load_env_template(
        cli.data_root_31
    )
    _, template42 = rollout_eval.load_env_template(
        cli.data_root_42
    )
    return {
        "D3O1": rollout_eval.build_target_config(
            template31,
            difficulty=1,
            n_drones=3,
            n_observers=1,
        ),
        "D4O2": rollout_eval.build_target_config(
            template42,
            difficulty=1,
            n_drones=4,
            n_observers=2,
        ),
    }


@torch.inference_mode()
def evaluate_source_rollouts(
    model: JointHeterogeneousCrossAgentAdapterHiSSD,
    rollout_configs: Mapping[str, Mapping[str, Any]],
    *,
    device: torch.device,
    episodes: int,
    seed_base: int,
    epoch: int | None = None,
    show_progress: bool = False,
) -> dict[str, dict[str, float]]:
    """Evaluate true mission success on source populations with fixed seeds."""
    was_training = model.training
    model.eval()
    progress = None
    if tqdm is not None and show_progress:
        progress = tqdm(
            total=int(episodes) * len(rollout_configs),
            desc=f"epoch {epoch or '?'} rollout",
            unit="episode",
            dynamic_ncols=True,
            leave=False,
            position=1,
        )
    try:
        results: dict[str, dict[str, float]] = {}
        for source_name, config in rollout_configs.items():
            records = []
            for episode_index in range(int(episodes)):
                records.append(rollout_eval.run_episode(
                    config,
                    model,
                    seed=int(seed_base)
                    + 100_000
                    + episode_index,
                    difficulty=1,
                    device=device,
                ))
                if progress is not None:
                    progress.update(1)
                    progress.set_postfix(
                        source=source_name,
                        refresh=False,
                    )
            results[source_name] = (
                rollout_eval.aggregate(records)
            )
        return results
    finally:
        if progress is not None:
            progress.close()
        model.train(was_training)


def append_rollout_history(
    output_dir: Path,
    *,
    epoch: int,
    results: Mapping[str, Mapping[str, float]],
    episodes: int,
    seed_base: int,
) -> None:
    payload = {
        "epoch": int(epoch),
        "episodes_per_source": int(episodes),
        "seed_base": int(seed_base),
        "seed_formula": (
            "seed_base + 100000 + episode_index"
        ),
        "sources": {
            name: {
                key: float(value)
                for key, value in metrics.items()
            }
            for name, metrics in results.items()
        },
    }
    path = (
        output_dir
        / "source_rollout_history.jsonl"
    )
    with path.open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(
            json.dumps(
                payload, sort_keys=True
            )
            + "\n"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cli = parse_args()

    if cli.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if cli.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if cli.sequence_length <= 0:
        raise ValueError(
            "--sequence-length must be positive."
        )
    if cli.adapter_hidden_dim <= 0:
        raise ValueError(
            "--adapter-hidden-dim must be positive."
        )
    if cli.cross_agent_context_dim <= 0:
        raise ValueError(
            "--cross-agent-context-dim must be positive."
        )
    if cli.cross_agent_attention_heads <= 0:
        raise ValueError(
            "--cross-agent-attention-heads must be positive."
        )
    if (
        cli.cross_agent_context_dim
        % cli.cross_agent_attention_heads
        != 0
    ):
        raise ValueError(
            "--cross-agent-context-dim must be divisible by "
            "--cross-agent-attention-heads."
        )
    if not 0.0 <= cli.cross_agent_attention_dropout < 1.0:
        raise ValueError(
            "--cross-agent-attention-dropout must be in [0, 1)."
        )
    if cli.batches_per_source < 0:
        raise ValueError(
            "--batches-per-source cannot be negative."
        )
    if cli.early_stopping_patience < 0:
        raise ValueError(
            "--early-stopping-patience cannot be negative."
        )
    if cli.rollout_eval_every < 0:
        raise ValueError(
            "--rollout-eval-every cannot be negative."
        )
    if cli.rollout_eval_episodes <= 0:
        raise ValueError(
            "--rollout-eval-episodes must be positive."
        )

    args = configure_training_args(cli)
    multi.seed_everything(cli.seed)
    device = multi.resolve_device(cli.device)

    if hasattr(base, "configure_gpu_backend"):
        base.configure_gpu_backend(device)

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

    tasks31, names31 = multi.manifest_info(
        cli.manifest_31
    )
    tasks42, names42 = multi.manifest_info(
        cli.manifest_42
    )
    if tasks31 != tasks42:
        raise ValueError(
            "D3O1/D4O2 manifests use different source "
            f"difficulties: {tasks31} vs {tasks42}."
        )
    if not tasks31:
        raise ValueError(
            "No source difficulties found."
        )
    if tasks31 != list(
        range(1, len(tasks31) + 1)
    ):
        raise ValueError(
            "Source difficulty IDs must be contiguous "
            f"from 1; got {tasks31}."
        )
    if names31 != names42:
        raise ValueError(
            "D3O1/D4O2 descriptor schemas differ."
        )

    pin_memory = device.type == "cuda"
    train_datasets: dict[str, Any] = {}
    train_loaders: dict[str, Any] = {}
    val_datasets: dict[str, Any] = {}
    val_loaders: dict[str, Any] = {}

    for index, (name, spec) in enumerate(
        source_specs.items()
    ):
        (
            train_datasets[name],
            train_loaders[name],
        ) = create_joint_dataloader(
            spec,
            "source_train",
            cli,
            seed=cli.seed + index * 1000,
            pin_memory=pin_memory,
        )
        (
            val_datasets[name],
            val_loaders[name],
        ) = create_joint_dataloader(
            spec,
            "source_val",
            cli,
            seed=(
                cli.seed
                + 10_000
                + index * 1000
            ),
            pin_memory=pin_memory,
        )

    sample31 = train_datasets["3-1"][0]
    central_shape = sample31[
        "global_state"
    ]["central_map"].shape

    args.task_descriptor_names = names31
    args.training_tasks = tasks31
    args.held_out_tasks = []

    descriptor_mean, descriptor_scale = (
        multi.combined_descriptor_statistics(
            train_datasets,
            args.descriptor_scale_floor,
        )
    )
    args.task_descriptor_mean = (
        descriptor_mean.tolist()
    )
    args.task_descriptor_scale = (
        descriptor_scale.tolist()
    )

    role_configs = validate_role_schemas_from_data(
        train_datasets
    )
    model = JointHeterogeneousCrossAgentAdapterHiSSD(
            role_configs=role_configs,
            central_map_channels=int(
                central_shape[-3]
            ),
            central_map_size=tuple(
                int(x)
                for x in central_shape[-2:]
            ),
            reference_observer_count=int(
                sample31["observer"][
                    "actions"
                ].shape[-2]
            ),
            reference_drone_count=int(
                sample31["drone"][
                    "actions"
                ].shape[-2]
            ),
            hidden_dim=args.hidden_dim,
            skill_dim=args.skill_dim,
            transformer_heads=(
                args.transformer_heads
            ),
            common_skill_adapter_hidden_dim=(
                cli.adapter_hidden_dim
            ),
            cross_agent_context_dim=(
                cli.cross_agent_context_dim
            ),
            cross_agent_attention_heads=(
                cli.cross_agent_attention_heads
            ),
            cross_agent_attention_dropout=(
                cli.cross_agent_attention_dropout
            ),
            contrastive_from_action_skill=False,
            task_context_pooling=False,
            task_descriptor_dim=(
                int(
                    sample31[
                        "task_descriptor"
                    ].numel()
                )
                if args.descriptor_enabled
                else 0
            ),
            task_prior_count=(
                len(tasks31)
                if args.task_contrastive_enabled
                else 0
            ),
            task_dropout=args.task_dropout,
            task_feature_deltas=True,
            learned_task_classifier=(
                args.task_contrastive_enabled
            ),
            normalize_task_context=False,
            task_running_statistics=True,
            direct_task_summary=False,
            task_action_residual=False,
            skill_structure="split",
        ).to(device)

    population_counts = (
        validate_joint_source_compatibility(
            train_datasets, model
        )
    )

    expected_prior_count = (
        len(tasks31)
        if args.task_contrastive_enabled
        else 0
    )
    if (
        model.task_prior_count
        != expected_prior_count
    ):
        raise RuntimeError(
            "Unexpected task prior count: "
            f"model={model.task_prior_count}, "
            f"expected={expected_prior_count}."
        )

    optimizer, base_params, task_params = (
        multi.optimizer_for_model(
            model, args
        )
    )

    adapter_params = sum(
        parameter.numel()
        for parameter
        in model.common_skill_adapter.parameters()
    )
    common_params = sum(
        parameter.numel()
        for parameter
        in model.common_skill_encoder.parameters()
    )

    initial_raw_batch = next(
        iter(train_loaders["3-1"])
    )
    init_diag = initialization_diagnostics(
        model,
        initial_raw_batch,
        device,
    )
    if (
        init_diag["adapter_identity_error"]
        > 1e-8
    ):
        raise RuntimeError(
            "Adapter is not identity at initialization: "
            f"{init_diag['adapter_identity_error']:.3e}"
        )

    cli.output_dir.mkdir(
        parents=True, exist_ok=True
    )

    batches_per_source = (
        None
        if cli.batches_per_source <= 0
        else cli.batches_per_source
    )
    effective_bps = (
        max(
            len(loader)
            for loader
            in train_loaders.values()
        )
        if batches_per_source is None
        else batches_per_source
    )

    print(
        f"device={device} "
        f"source_populations={population_counts}",
        flush=True,
    )
    print(
        "train_windows="
        + str(
            {
                name: len(dataset)
                for name, dataset
                in train_datasets.items()
            }
        )
        + " val_windows="
        + str(
            {
                name: len(dataset)
                for name, dataset
                in val_datasets.items()
            }
        ),
        flush=True,
    )
    print(
        f"balanced_batches_per_source={effective_bps} "
        f"source_updates_per_epoch={2 * effective_bps}",
        flush=True,
    )
    print(
        f"outcome_balanced_sampling={cli.balanced_sampling} "
        f"(True matches current comparison protocol)",
        flush=True,
    )
    print(
        f"base_params={sum(p.numel() for p in base_params):,} "
        f"task_params={sum(p.numel() for p in task_params):,} "
        f"common_encoder_params={common_params:,} "
        f"adapter_params={adapter_params:,}",
        flush=True,
    )
    print(
        "initialization=scratch "
        "(NO BC checkpoint; role encoders/action decoders random-init) "
        f"adapter_identity={init_diag['adapter_identity_error']:.3e}",
        flush=True,
    )
    print(
        "architecture="
        "role_specific_obs_encoder + "
        "shared_common_skill_encoder + "
        "ONE cross-agent SelfAttention([h_j,c_j]) + "
        "A([c_i,h_i,g_i]) residual adapter + "
        "role_specific_action_decoder",
        flush=True,
    )
    print(
        "loss_balancing="
        "0.5*observer_action + 0.5*drone_action; "
        "planner local prediction also role-balanced",
        flush=True,
    )
    print(
        f"descriptor_scale="
        f"{[round(x, 4) for x in args.task_descriptor_scale]} "
        f"task_ids=difficulty_only:{tasks31} "
        f"early_stopping_patience="
        f"{args.early_stopping_patience}",
        flush=True,
    )

    rollout_configs = None
    if cli.rollout_eval_every > 0:
        rollout_configs = (
            build_source_rollout_configs(cli)
        )
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
    aux_enabled = bool(
        args.descriptor_enabled
        or args.task_contrastive_enabled
    )

    # Fixed batch: adapter magnitude is comparable across epochs.
    diagnostic_raw_batch = initial_raw_batch

    show_progress = tqdm is not None and not cli.no_progress
    epoch_progress = range(1, cli.epochs + 1)
    if show_progress:
        epoch_progress = tqdm(
            epoch_progress,
            total=cli.epochs,
            desc="joint cross-agent HiSSD scratch",
            unit="epoch",
            dynamic_ncols=True,
            position=0,
        )

    for epoch in epoch_progress:
        task_only = (
            epoch <= args.task_warmup_epochs
        )

        (
            train_metrics,
            train_by_source,
        ) = train_epoch(
            model,
            train_loaders,
            optimizer,
            device,
            args,
            epoch=epoch,
            batches_per_source=batches_per_source,
            task_only=task_only,
            show_progress=show_progress,
        )

        (
            val_metrics,
            val_by_source,
        ) = validate_all_sources(
            model,
            val_loaders,
            device,
            args,
            epoch=epoch,
            show_progress=show_progress,
        )

        combined_loss, task_loss = (
            base.validation_losses(
                val_metrics, args
            )
        )

        adapter_metrics = adapter_diagnostics(
            model,
            diagnostic_raw_batch,
            device,
        )

        rollout_results = None
        should_rollout = (
            rollout_configs is not None
            and (
                epoch % cli.rollout_eval_every == 0
                or epoch == cli.epochs
            )
        )
        if should_rollout:
            rollout_results = (
                evaluate_source_rollouts(
                    model,
                    rollout_configs,
                    device=device,
                    episodes=cli.rollout_eval_episodes,
                    seed_base=cli.rollout_eval_seed_base,
                    epoch=epoch,
                    show_progress=show_progress,
                )
            )
            append_rollout_history(
                cli.output_dir,
                epoch=epoch,
                results=rollout_results,
                episodes=cli.rollout_eval_episodes,
                seed_base=cli.rollout_eval_seed_base,
            )

        line = (
            f"epoch={epoch:03d} "
            f"phase={'task_warmup' if task_only else 'joint'} "
            f"train[{compact(train_metrics)}] "
            f"val[{compact(val_metrics)}] "
            f"adapter={adapter_metrics['adapter_delta_rms']:.5f} "
            f"ratio={adapter_metrics['adapter_to_common_ratio']:.4f} "
            f"obs_delta={adapter_metrics['observer_adapter_delta_rms']:.5f} "
            f"drone_delta={adapter_metrics['drone_adapter_delta_rms']:.5f} "
            + " ".join(
                f"{name}[{compact(metrics)}]"
                for name, metrics
                in val_by_source.items()
            )
        )
        if rollout_results is not None:
            s31 = rollout_results["D3O1"]["success"]
            s42 = rollout_results["D4O2"]["success"]
            source_success = 0.5 * (s31 + s42)
            line += (
                f" rollout[D3O1_succ={s31:.3f} "
                f"D4O2_succ={s42:.3f} "
                f"mean_succ={source_success:.3f} "
                f"D3O1_crash="
                f"{rollout_results['D3O1']['fatal_crash']:.3f} "
                f"D4O2_crash="
                f"{rollout_results['D4O2']['fatal_crash']:.3f}]"
            )
        progress_write(line)
        if show_progress and hasattr(epoch_progress, "set_postfix"):
            postfix = {
                "train": f"{train_metrics['action_mse']:.4f}",
                "val": f"{val_metrics['action_mse']:.4f}",
            }
            if rollout_results is not None:
                postfix["success"] = f"{source_success:.3f}"
            epoch_progress.set_postfix(postfix, refresh=False)

        save_checkpoint(
            cli.output_dir
            / "hissd_joint_cross_agent_adapter_scratch_last.pt",
            model,
            optimizer,
            epoch=epoch,
            args=args,
            cli=cli,
            train_metrics=train_metrics,
            train_by_source=train_by_source,
            val_metrics=val_metrics,
            val_by_source=val_by_source,
            adapter_metrics=adapter_metrics,
            population_counts=population_counts,
        )

        if combined_loss < best_combined:
            best_combined = combined_loss
            save_checkpoint(
                cli.output_dir
                / "hissd_joint_cross_agent_adapter_scratch_best.pt",
                model,
                optimizer,
                epoch=epoch,
                args=args,
                cli=cli,
                train_metrics=train_metrics,
                train_by_source=train_by_source,
                val_metrics=val_metrics,
                val_by_source=val_by_source,
                    adapter_metrics=adapter_metrics,
                population_counts=population_counts,
            )

        selection = (
            task_loss
            if aux_enabled
            else combined_loss
        )
        if selection < best_task:
            best_task = selection
            best_task_epoch = epoch
            no_improve = 0
            save_checkpoint(
                cli.output_dir
                / "hissd_joint_cross_agent_adapter_scratch_best_task.pt",
                model,
                optimizer,
                epoch=epoch,
                args=args,
                cli=cli,
                train_metrics=train_metrics,
                train_by_source=train_by_source,
                val_metrics=val_metrics,
                val_by_source=val_by_source,
                    adapter_metrics=adapter_metrics,
                population_counts=population_counts,
            )
        else:
            no_improve += 1

        patience = int(
            args.early_stopping_patience
        )
        if (
            patience > 0
            and no_improve >= patience
        ):
            progress_write(
                "Early stopping: no selected validation "
                f"improvement for {patience} epochs "
                f"(best epoch={best_task_epoch})."
            )
            break

    if show_progress and hasattr(epoch_progress, "close"):
        epoch_progress.close()

    progress_write(
        f"done best_combined={best_combined:.6f} "
        f"best_task={best_task:.6f}"
        f"@{best_task_epoch}"
    )


if __name__ == "__main__":
    main()
