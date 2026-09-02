"""Jointly fine-tune HiSSD drones and a MAPPO observer residual with PPO.

The pretrained drone/observer encoders and base controllers stay frozen. PPO
updates only small task-conditioned action residuals, their exploration
standard deviations, and a centralized team-value head.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch
import torch.nn.functional as F
from ray.tune.registry import register_env
from torch import nn
from torch.distributions import Normal


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from hemac import HeMAC_v0
from hemac.rllib_policy import register_hemac_rllib_models
from skill_discovery.collect_offline_data import (
    CHECKPOINT_PATH as DEFAULT_MAPPO_CHECKPOINT,
    ENV_NAME,
    _local_reward_after_step,
    _team_reward,
    build_global_central_map,
    build_collection_env_config,
    convert_observation,
    env_creator,
    get_core_env,
    load_inference_algorithm,
)
from skill_discovery.evaluate_drone_bc import drone_action_scale
from skill_discovery.hissd_models import HeMACHISSD
from skill_discovery.models import ObserverTaskResidualPolicy
from skill_discovery.analyze_learning_efficiency import append_curve_points
from skill_discovery.visualize_hissd_skills import load_hissd_model, resolve_device


DEFAULT_HISSD_CHECKPOINT = (
    PROJECT_ROOT
    / "src/skill_discovery/checkpoints/hissd_checkpoints_v22/hissd_adapted_best.pt"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "src/skill_discovery/checkpoints/hissd_joint_online_checkpoints"
)
STAGE_MIXTURES = {
    4: ((4, 1.0),),
    5: ((4, 0.30), (5, 0.70)),
    6: ((4, 0.15), (5, 0.25), (6, 0.60)),
}


class OnlineValueHead(nn.Module):
    """Central team-value estimate from pooled frozen HiSSD representations."""

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hissd-checkpoint", type=Path, default=DEFAULT_HISSD_CHECKPOINT)
    parser.add_argument("--mappo-checkpoint", type=Path, default=DEFAULT_MAPPO_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-stage", type=int, choices=(4, 5, 6), default=4)
    parser.add_argument("--end-stage", type=int, choices=(4, 5, 6), default=6)
    parser.add_argument("--iterations-per-stage", type=int, default=40)
    parser.add_argument("--episodes-per-iteration", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--observer-actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--log-std-lr", type=float, default=2e-5)
    parser.add_argument("--log-std-init", type=float, default=-2.0)
    parser.add_argument("--observer-log-std-init", type=float, default=-2.0)
    parser.add_argument("--log-std-min", type=float, default=-3.0)
    parser.add_argument("--log-std-max", type=float, default=-0.7)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.15)
    parser.add_argument("--value-coeff", type=float, default=0.5)
    parser.add_argument("--entropy-coeff", type=float, default=0.002)
    parser.add_argument("--anchor-coeff", type=float, default=0.02)
    parser.add_argument(
        "--observer-residual-scale",
        type=float,
        default=0.35,
        help="Maximum observer residual as a fraction of its action range.",
    )
    parser.add_argument("--reward-scale", type=float, default=100.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--method-name",
        default="hissd_joint_online",
        help="Unique learning-curve method label for ablation comparisons.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--learning-curve-output",
        type=Path,
        help="Defaults to learning_curve_seed_<seed>.json under --output-dir.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.start_stage > args.end_stage:
        raise ValueError("--start-stage cannot exceed --end-stage.")
    for name in (
        "iterations_per_stage",
        "episodes_per_iteration",
        "eval_every",
        "eval_episodes",
        "ppo_epochs",
        "minibatch_size",
        "actor_lr",
        "observer_actor_lr",
        "critic_lr",
        "log_std_lr",
        "gamma",
        "gae_lambda",
        "clip_ratio",
        "reward_scale",
        "max_grad_norm",
        "observer_residual_scale",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if not 0.0 < args.gamma <= 1.0 or not 0.0 < args.gae_lambda <= 1.0:
        raise ValueError("--gamma and --gae-lambda must be in (0, 1].")
    if args.log_std_min >= args.log_std_max:
        raise ValueError("--log-std-min must be lower than --log-std-max.")
    args.method_name = args.method_name.strip()
    if not args.method_name:
        raise ValueError("--method-name cannot be empty.")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def drone_observation_batch(
    env,
    drone_ids: list[str],
    action_scale: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    converted = [convert_observation(env.observe(agent_id), "drone") for agent_id in drone_ids]
    return {
        "global_map": torch.from_numpy(
            np.stack([item["global_map"] for item in converted])
        ).unsqueeze(0).to(device),
        "local_map": torch.from_numpy(
            np.stack([item["local_map"] for item in converted])
        ).unsqueeze(0).to(device),
        "action_history": (
            torch.from_numpy(
                np.stack([item["action_history"] for item in converted])
            ).unsqueeze(0).to(device)
            / action_scale
        ),
    }


def observer_observation_batch(
    observation: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Convert one observer observation without changing MAPPO input scales."""
    converted = convert_observation(observation, "observer")
    return {
        "global_map": torch.from_numpy(converted["global_map"])
        .unsqueeze(0)
        .to(device),
        "local_map": torch.from_numpy(converted["local_map"])
        .unsqueeze(0)
        .to(device),
        "action_history": torch.from_numpy(converted["action_history"])
        .unsqueeze(0)
        .to(device),
    }


def observer_residual_action(
    baseline_action: np.ndarray,
    raw_residual: torch.Tensor,
    action_space,
    residual_scale: float,
) -> np.ndarray:
    """Apply a bounded residual to a baseline action in normalized space."""
    high = np.asarray(action_space.high, dtype=np.float32)
    baseline = np.asarray(baseline_action, dtype=np.float32).reshape(-1)
    normalized = baseline / high
    delta = torch.tanh(raw_residual).detach().cpu().numpy() * residual_scale
    action = np.clip(normalized + delta, -1.0, 1.0) * high
    return np.ascontiguousarray(
        np.clip(action, action_space.low, action_space.high), dtype=np.float32
    )


def squashed_log_prob(
    distribution: Normal,
    raw_action: torch.Tensor,
) -> torch.Tensor:
    correction = 2.0 * (
        math.log(2.0) - raw_action - F.softplus(-2.0 * raw_action)
    )
    return (distribution.log_prob(raw_action) - correction).sum(dim=(-1, -2))


def sample_stage_difficulty(stage: int, rng: random.Random) -> int:
    mixture = STAGE_MIXTURES[stage]
    draw = rng.random()
    cumulative = 0.0
    for difficulty, probability in mixture:
        cumulative += probability
        if draw <= cumulative:
            return difficulty
    return mixture[-1][0]


def rollout_episode(
    *,
    env,
    observer_algo,
    model: HeMACHISSD,
    value_head: OnlineValueHead,
    anchor_head: nn.Module,
    log_std: torch.Tensor,
    observer_residual_policy: ObserverTaskResidualPolicy,
    observer_anchor_head: nn.Module,
    observer_log_std: torch.Tensor,
    observer_residual_scale: float,
    action_scale: float,
    reward_scale: float,
    seed: int,
    device: torch.device,
    stochastic: bool,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Run one AEC episode and retain joint drone PPO transitions."""
    env.reset(seed=seed)
    core_env = get_core_env(env)
    agent_order = list(env.possible_agents)
    agent_index = {agent_id: index for index, agent_id in enumerate(agent_order)}
    observer_ids = [a for a in agent_order if a.startswith("observer_")]
    drone_ids = [a for a in agent_order if a.startswith("drone_")]
    observer_indices = [agent_index[a] for a in observer_ids]
    drone_indices = [agent_index[a] for a in drone_ids]
    if len(observer_ids) != 1:
        raise ValueError(
            "Joint online fine-tuning currently requires exactly one observer."
        )
    last_agent_id = agent_order[-1]
    valid_mask = torch.ones(1, len(drone_ids), dtype=torch.bool, device=device)
    recurrent_state = model.initial_inference_state(device=device)
    transitions: list[dict[str, Any]] = []
    cached_actions: dict[str, np.ndarray] = {}
    cycle_rewards: np.ndarray | None = None
    cycle_mask: np.ndarray | None = None
    shared_success_reward = 0.0
    final_info: dict[str, Any] = {}
    cycle_count = 0

    for agent_id in env.agent_iter():
        _, _, termination, truncation, info = env.last()
        if info:
            final_info.update(info)
        if termination or truncation:
            env.step(None)
            continue

        if not cached_actions:
            observations = drone_observation_batch(
                env, drone_ids, action_scale, device
            )
            with torch.no_grad():
                outputs, recurrent_state = model.inference_step(
                    observations, valid_mask, recurrent_state
                )
                task_skill = outputs["task_skills"][0]
                action_logits = outputs["action_logits"][0]
                direct_logits = model.action_decoder.task_action_residual_head(
                    task_skill
                )
                base_logits = action_logits - direct_logits
                anchor_logits = base_logits + anchor_head(task_skill)
                observer_id = observer_ids[0]
                observer_baseline_action = np.asarray(
                    observer_algo.compute_single_action(
                        observation=env.observe(observer_id),
                        policy_id="observer_policy",
                        explore=False,
                    ),
                    dtype=np.float32,
                ).reshape(-1)
                observer_observation = observer_observation_batch(
                    env.observe(observer_id), device
                )
                pooled_task_skill = task_skill.mean(dim=0, keepdim=True)
                observer_residual_mean, observer_features = observer_residual_policy(
                    observer_observation["global_map"],
                    observer_observation["local_map"],
                    observer_observation["action_history"],
                    pooled_task_skill,
                )
                observer_anchor_mean = observer_anchor_head(
                    torch.cat((observer_features, pooled_task_skill), dim=-1)
                )
                critic_features = torch.cat(
                    (
                        outputs["observation_features"][0].mean(dim=0),
                        outputs["common_skills"][0].mean(dim=0),
                        task_skill.mean(dim=0),
                        observer_features[0],
                        model.central_state_encoder(
                            torch.from_numpy(build_global_central_map(core_env))
                            .unsqueeze(0)
                            .to(device)
                        )[0],
                    ),
                    dim=-1,
                )
                value = value_head(critic_features.unsqueeze(0))[0]
                distribution = Normal(action_logits, log_std.exp())
                raw_action = distribution.sample() if stochastic else action_logits
                normalized_action = torch.tanh(raw_action)
                drone_log_prob = squashed_log_prob(distribution, raw_action)
                observer_distribution = Normal(
                    observer_residual_mean, observer_log_std.exp()
                )
                observer_raw_action = (
                    observer_distribution.sample()
                    if stochastic
                    else observer_residual_mean
                )
                observer_log_prob = observer_distribution.log_prob(
                    observer_raw_action
                ).sum()
                old_log_prob = drone_log_prob + observer_log_prob

            cached_actions = {}
            for drone_index, drone_id in enumerate(drone_ids):
                action_space = env.action_space(drone_id)
                action = normalized_action[drone_index].cpu().numpy() * action_scale
                cached_actions[drone_id] = np.ascontiguousarray(
                    np.clip(action, action_space.low, action_space.high),
                    dtype=np.float32,
                )
            cached_actions[observer_id] = observer_residual_action(
                observer_baseline_action,
                observer_raw_action[0],
                env.action_space(observer_id),
                observer_residual_scale,
            )
            if stochastic:
                transitions.append(
                    {
                        "task_skill": task_skill.cpu(),
                        "base_logits": base_logits.cpu(),
                        "anchor_logits": anchor_logits.cpu(),
                        "raw_action": raw_action.cpu(),
                        "old_log_prob": old_log_prob.cpu(),
                        "observer_features": observer_features[0].cpu(),
                        "observer_task_skill": pooled_task_skill[0].cpu(),
                        "observer_raw_action": observer_raw_action[0].cpu(),
                        "observer_anchor_mean": observer_anchor_mean[0].cpu(),
                        "critic_features": critic_features.cpu(),
                        "value": value.cpu(),
                    }
                )
            cycle_rewards = np.zeros(len(agent_order), dtype=np.float32)
            cycle_mask = np.zeros(len(agent_order), dtype=np.bool_)
            shared_success_reward = 0.0

        index = agent_index[agent_id]
        cycle_mask[index] = True
        env.step(cached_actions[agent_id])
        for reward_agent_id, reward_index in agent_index.items():
            cycle_rewards[reward_index] += _local_reward_after_step(
                core_env, reward_agent_id
            )
        shared_success_reward = max(
            shared_success_reward,
            float(core_env.global_reward),
        )
        cycle_finished = (
            agent_id == last_agent_id
            or bool(core_env.terminate)
            or bool(core_env.truncate)
        )
        if not cycle_finished:
            continue
        team_reward = _team_reward(
            cycle_rewards,
            cycle_mask,
            observer_indices,
            drone_indices,
            shared_success_reward,
        )
        if stochastic:
            transitions[-1]["reward"] = float(team_reward / reward_scale)
        cycle_count += 1
        cached_actions = {}
        cycle_rewards = None
        cycle_mask = None

    if hasattr(core_env, "build_episode_info"):
        final_info.update(core_env.build_episode_info())
    metrics = {
        "success": float(bool(final_info.get("success", core_env.mission_success))),
        "goal_found": float(bool(final_info.get("goal_found", core_env.found_goal))),
        "fatal_crash": float(bool(final_info.get("fatal_crash", core_env.collided))),
        "drone_crash": float(bool(final_info.get("drone_crash", core_env.drone_crash))),
        "observer_crash": float(
            bool(final_info.get("observer_crash", core_env.observer_crash))
        ),
        "coverage": float(core_env.current_coverage_ratio()),
        "cycles": float(cycle_count),
    }
    return transitions, metrics


def add_gae(
    transitions: list[dict[str, Any]],
    gamma: float,
    gae_lambda: float,
) -> None:
    advantage = 0.0
    next_value = 0.0
    for transition in reversed(transitions):
        value = float(transition["value"])
        delta = transition["reward"] + gamma * next_value - value
        advantage = delta + gamma * gae_lambda * advantage
        transition["advantage"] = advantage
        transition["return"] = advantage + value
        next_value = value


def ppo_update(
    model: HeMACHISSD,
    observer_residual_policy: ObserverTaskResidualPolicy,
    value_head: OnlineValueHead,
    log_std: nn.Parameter,
    observer_log_std: nn.Parameter,
    optimizer: torch.optim.Optimizer,
    transitions: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    fields = (
        "task_skill",
        "base_logits",
        "anchor_logits",
        "raw_action",
        "old_log_prob",
        "observer_features",
        "observer_task_skill",
        "observer_raw_action",
        "observer_anchor_mean",
        "critic_features",
        "value",
    )
    data = {
        name: torch.stack([item[name] for item in transitions]).to(device)
        for name in fields
    }
    data["advantage"] = torch.tensor(
        [item["advantage"] for item in transitions],
        dtype=torch.float32,
        device=device,
    )
    data["return"] = torch.tensor(
        [item["return"] for item in transitions],
        dtype=torch.float32,
        device=device,
    )
    data["advantage"] = (
        data["advantage"] - data["advantage"].mean()
    ) / data["advantage"].std(unbiased=False).clamp_min(1e-6)

    accumulator: dict[str, float] = defaultdict(float)
    updates = 0
    count = len(transitions)
    for _ in range(args.ppo_epochs):
        order = torch.randperm(count, device=device)
        for start in range(0, count, args.minibatch_size):
            indices = order[start : start + args.minibatch_size]
            logits = data["base_logits"][indices] + (
                model.action_decoder.task_action_residual_head(
                    data["task_skill"][indices]
                )
            )
            distribution = Normal(logits, log_std.exp())
            drone_log_prob = squashed_log_prob(
                distribution, data["raw_action"][indices]
            )
            observer_mean = observer_residual_policy.residual_head(
                torch.cat(
                    (
                        data["observer_features"][indices],
                        data["observer_task_skill"][indices],
                    ),
                    dim=-1,
                )
            )
            observer_distribution = Normal(observer_mean, observer_log_std.exp())
            observer_log_prob = observer_distribution.log_prob(
                data["observer_raw_action"][indices]
            ).sum(dim=-1)
            log_prob = drone_log_prob + observer_log_prob
            ratio = (log_prob - data["old_log_prob"][indices]).exp()
            advantage = data["advantage"][indices]
            surrogate = torch.minimum(
                ratio * advantage,
                ratio.clamp(1.0 - args.clip_ratio, 1.0 + args.clip_ratio)
                * advantage,
            )
            policy_loss = -surrogate.mean()
            values = value_head(data["critic_features"][indices])
            value_loss = F.smooth_l1_loss(values, data["return"][indices])
            entropy = (
                distribution.entropy().sum(dim=(-1, -2))
                + observer_distribution.entropy().sum(dim=-1)
            ).mean()
            drone_anchor_loss = F.mse_loss(
                torch.tanh(logits),
                torch.tanh(data["anchor_logits"][indices]),
            )
            observer_anchor_loss = F.mse_loss(
                torch.tanh(observer_mean),
                torch.tanh(data["observer_anchor_mean"][indices]),
            )
            anchor_loss = drone_anchor_loss + observer_anchor_loss
            loss = (
                policy_loss
                + args.value_coeff * value_loss
                - args.entropy_coeff * entropy
                + args.anchor_coeff * anchor_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                [
                    *model.action_decoder.task_action_residual_head.parameters(),
                    *observer_residual_policy.residual_head.parameters(),
                    *value_head.parameters(),
                    log_std,
                    observer_log_std,
                ],
                args.max_grad_norm,
            )
            optimizer.step()
            with torch.no_grad():
                log_std.clamp_(args.log_std_min, args.log_std_max)
                observer_log_std.clamp_(args.log_std_min, args.log_std_max)
            accumulator["policy_loss"] += float(policy_loss.detach())
            accumulator["value_loss"] += float(value_loss.detach())
            accumulator["entropy"] += float(entropy.detach())
            accumulator["anchor_loss"] += float(anchor_loss.detach())
            accumulator["drone_anchor_loss"] += float(drone_anchor_loss.detach())
            accumulator["observer_anchor_loss"] += float(
                observer_anchor_loss.detach()
            )
            accumulator["approx_kl"] += float(
                (data["old_log_prob"][indices] - log_prob).mean().detach()
            )
            accumulator["clip_fraction"] += float(
                ((ratio - 1.0).abs() > args.clip_ratio).float().mean().detach()
            )
            accumulator["grad_norm"] += float(grad_norm)
            updates += 1
    return {name: value / max(updates, 1) for name, value in accumulator.items()}


def average_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    return {
        name: float(np.mean([entry[name] for entry in metrics]))
        for name in metrics[0]
    }


def evaluation_score(metrics: dict[str, float]) -> float:
    return (
        metrics["success"]
        + 0.10 * metrics["goal_found"]
        - 0.25 * metrics["fatal_crash"]
    )


def evaluate_policy(
    *,
    difficulty: int,
    checkpoint_env_config: dict[str, Any],
    observer_algo,
    model: HeMACHISSD,
    value_head: OnlineValueHead,
    anchor_head: nn.Module,
    log_std: torch.Tensor,
    observer_residual_policy: ObserverTaskResidualPolicy,
    observer_anchor_head: nn.Module,
    observer_log_std: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate deterministic actions on fixed seeds for comparable selection."""
    metrics = []
    env_config = build_collection_env_config(checkpoint_env_config, difficulty)
    for episode_index in range(args.eval_episodes):
        env = HeMAC_v0.env(**env_config)
        try:
            _, result = rollout_episode(
                env=env,
                observer_algo=observer_algo,
                model=model,
                value_head=value_head,
                anchor_head=anchor_head,
                log_std=log_std,
                observer_residual_policy=observer_residual_policy,
                observer_anchor_head=observer_anchor_head,
                observer_log_std=observer_log_std,
                observer_residual_scale=args.observer_residual_scale,
                action_scale=drone_action_scale(env_config),
                reward_scale=args.reward_scale,
                seed=args.seed + difficulty * 1_000_000 + episode_index,
                device=device,
                stochastic=False,
            )
        finally:
            env.close()
        metrics.append(result)
    return average_metrics(metrics)


def save_checkpoint(
    path: Path,
    model: HeMACHISSD,
    source_payload: dict[str, Any],
    value_head: OnlineValueHead,
    log_std: nn.Parameter,
    observer_residual_policy: ObserverTaskResidualPolicy,
    observer_log_std: nn.Parameter,
    optimizer: torch.optim.Optimizer,
    *,
    stage: int,
    iteration: int,
    stage_joint_env_steps: int,
    total_joint_env_steps: int,
    metrics: dict[str, float],
    args: argparse.Namespace,
) -> None:
    payload = dict(source_payload)
    payload.update(
        {
            "model_config": model.config(),
            "model_state_dict": model.state_dict(),
            "online_value_state_dict": value_head.state_dict(),
            "online_log_std": log_std.detach().cpu(),
            "online_observer_residual_config": observer_residual_policy.config(),
            "online_observer_residual_state_dict": (
                observer_residual_policy.state_dict()
            ),
            "online_observer_log_std": observer_log_std.detach().cpu(),
            "online_observer_residual_scale": args.observer_residual_scale,
            "online_optimizer_state_dict": optimizer.state_dict(),
            "online_finetuning": {
                "method": "joint_observer_drone_task_residual_team_ppo",
                "curve_method": args.method_name,
                "source_checkpoint": str(args.hissd_checkpoint.resolve()),
                "observer_base_checkpoint": str(args.mappo_checkpoint.resolve()),
                "stage": stage,
                "iteration": iteration,
                "stage_joint_env_steps": stage_joint_env_steps,
                "total_joint_env_steps": total_joint_env_steps,
                "stage_mixture": STAGE_MIXTURES[stage],
                "metrics": metrics,
                "hyperparameters": vars(args),
            },
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    model, source_payload = load_hissd_model(args.hissd_checkpoint, device)
    model.enable_task_action_residual()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    task_head = model.action_decoder.task_action_residual_head
    for parameter in task_head.parameters():
        parameter.requires_grad_(True)
    model.eval()
    anchor_head = copy.deepcopy(task_head).to(device).eval()
    for parameter in anchor_head.parameters():
        parameter.requires_grad_(False)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=1)
    register_hemac_rllib_models()
    register_env(ENV_NAME, env_creator)
    observer_algo = load_inference_algorithm(args.mappo_checkpoint.resolve())
    checkpoint_env_config = getattr(observer_algo.config, "env_config", {}) or {}
    rllib_observer_model = observer_algo.get_policy("observer_policy").model
    observer_residual_policy = ObserverTaskResidualPolicy(
        global_map_channels=rllib_observer_model.global_map_channels,
        local_map_channels=rllib_observer_model.local_map_channels,
        task_skill_dim=model.skill_dim,
        action_dim=3,
    ).to(device)
    observer_residual_policy.initialize_encoder_from_rllib(rllib_observer_model)
    for parameter in observer_residual_policy.encoder.parameters():
        parameter.requires_grad_(False)
    observer_residual_policy.eval()
    observer_anchor_head = copy.deepcopy(
        observer_residual_policy.residual_head
    ).to(device).eval()
    for parameter in observer_anchor_head.parameters():
        parameter.requires_grad_(False)

    critic_input_dim = (
        model.observation_encoder.output_dim
        + 2 * model.skill_dim
        + observer_residual_policy.encoder.output_dim
        + model.hidden_dim
    )
    value_head = OnlineValueHead(critic_input_dim).to(device)
    log_std = nn.Parameter(
        torch.full((model.agent_count, model.action_dim), args.log_std_init, device=device)
    )
    observer_log_std = nn.Parameter(
        torch.full((model.action_dim,), args.observer_log_std_init, device=device)
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": task_head.parameters(), "lr": args.actor_lr},
            {
                "params": observer_residual_policy.residual_head.parameters(),
                "lr": args.observer_actor_lr,
            },
            {"params": value_head.parameters(), "lr": args.critic_lr},
            {"params": [log_std], "lr": args.log_std_lr},
            {"params": [observer_log_std], "lr": args.log_std_lr},
        ],
        weight_decay=1e-5,
    )
    print(
        "Joint PPO trainable parameters: "
        f"drone_residual={sum(p.numel() for p in task_head.parameters())}, "
        "observer_residual="
        f"{sum(p.numel() for p in observer_residual_policy.residual_head.parameters())}, "
        f"critic={sum(p.numel() for p in value_head.parameters())}"
    )
    rng = random.Random(args.seed)
    global_iteration = 0
    total_joint_env_steps = 0
    learning_curve_path = (
        args.learning_curve_output.expanduser().resolve()
        if args.learning_curve_output is not None
        else args.output_dir / f"learning_curve_seed_{args.seed}.json"
    )
    try:
        for stage in range(args.start_stage, args.end_stage + 1):
            stage_joint_env_steps = 0
            print(f"\nStarting online PPO stage {stage}: mixture={STAGE_MIXTURES[stage]}")
            baseline = evaluate_policy(
                difficulty=stage,
                checkpoint_env_config=checkpoint_env_config,
                observer_algo=observer_algo,
                model=model,
                value_head=value_head,
                anchor_head=anchor_head,
                log_std=log_std,
                observer_residual_policy=observer_residual_policy,
                observer_anchor_head=observer_anchor_head,
                observer_log_std=observer_log_std,
                args=args,
                device=device,
            )
            stage_best_score = evaluation_score(baseline)
            stage_best_state: dict[str, Any] = {
                "model": copy.deepcopy(model.state_dict()),
                "value": copy.deepcopy(value_head.state_dict()),
                "log_std": log_std.detach().clone(),
                "observer": copy.deepcopy(observer_residual_policy.state_dict()),
                "observer_log_std": observer_log_std.detach().clone(),
                "optimizer": copy.deepcopy(optimizer.state_dict()),
            }
            print(
                f"BASELINE stage={stage} success={baseline['success']:.3f} "
                f"goal={baseline['goal_found']:.3f} "
                f"fatal={baseline['fatal_crash']:.3f} "
                f"score={stage_best_score:.4f}"
            )
            append_curve_points(
                learning_curve_path,
                [
                    {
                        "method": args.method_name,
                        "seed": args.seed,
                        "difficulty": stage,
                        "joint_env_steps": stage_joint_env_steps,
                        "total_joint_env_steps": total_joint_env_steps,
                        "iteration": global_iteration,
                        "success_rate": baseline["success"],
                        "goal_found_rate": baseline["goal_found"],
                        "fatal_crash_rate": baseline["fatal_crash"],
                        "drone_crash_rate": baseline["drone_crash"],
                        "observer_crash_rate": baseline["observer_crash"],
                        "mean_coverage_ratio": baseline["coverage"],
                        "mean_cycles": baseline["cycles"],
                    }
                ],
                metadata={
                    "evaluation_episodes": args.eval_episodes,
                    "source_checkpoint": str(args.hissd_checkpoint.resolve()),
                    "curve_method": args.method_name,
                    "success_definition": "observer_goal_arrival",
                    "training_step_definition": (
                        "sum of world cycles from target training episodes; "
                        "evaluation cycles excluded"
                    ),
                },
            )
            save_checkpoint(
                args.output_dir / f"hissd_online_stage_{stage:02d}_best.pt",
                model,
                source_payload,
                value_head,
                log_std,
                observer_residual_policy,
                observer_log_std,
                optimizer,
                stage=stage,
                iteration=global_iteration,
                stage_joint_env_steps=stage_joint_env_steps,
                total_joint_env_steps=total_joint_env_steps,
                metrics=baseline,
                args=args,
            )
            if stage == args.end_stage:
                save_checkpoint(
                    args.output_dir / "hissd_online_best.pt",
                    model,
                    source_payload,
                    value_head,
                    log_std,
                    observer_residual_policy,
                    observer_log_std,
                    optimizer,
                    stage=stage,
                    iteration=global_iteration,
                    stage_joint_env_steps=stage_joint_env_steps,
                    total_joint_env_steps=total_joint_env_steps,
                    metrics=baseline,
                    args=args,
                )
            for stage_iteration in range(1, args.iterations_per_stage + 1):
                global_iteration += 1
                trajectories: list[dict[str, Any]] = []
                train_episode_metrics = []
                difficulty_counts: dict[int, int] = defaultdict(int)
                for episode_index in range(args.episodes_per_iteration):
                    difficulty = sample_stage_difficulty(stage, rng)
                    difficulty_counts[difficulty] += 1
                    env_config = build_collection_env_config(
                        checkpoint_env_config, difficulty
                    )
                    env = HeMAC_v0.env(**env_config)
                    try:
                        episode, metrics = rollout_episode(
                            env=env,
                            observer_algo=observer_algo,
                            model=model,
                            value_head=value_head,
                            anchor_head=anchor_head,
                            log_std=log_std,
                            observer_residual_policy=observer_residual_policy,
                            observer_anchor_head=observer_anchor_head,
                            observer_log_std=observer_log_std,
                            observer_residual_scale=args.observer_residual_scale,
                            action_scale=drone_action_scale(env_config),
                            reward_scale=args.reward_scale,
                            seed=args.seed + global_iteration * 10_000 + episode_index,
                            device=device,
                            stochastic=True,
                        )
                    finally:
                        env.close()
                    add_gae(episode, args.gamma, args.gae_lambda)
                    trajectories.extend(episode)
                    train_episode_metrics.append(metrics)
                    episode_joint_steps = int(round(metrics["cycles"]))
                    stage_joint_env_steps += episode_joint_steps
                    total_joint_env_steps += episode_joint_steps
                update_metrics = ppo_update(
                    model,
                    observer_residual_policy,
                    value_head,
                    log_std,
                    observer_log_std,
                    optimizer,
                    trajectories,
                    args,
                    device,
                )
                train_metrics = average_metrics(train_episode_metrics)
                print(
                    f"stage={stage} iter={stage_iteration:03d}/"
                    f"{args.iterations_per_stage} samples={len(trajectories)} "
                    f"mix={dict(difficulty_counts)} "
                    f"success={train_metrics['success']:.3f} "
                    f"crash={train_metrics['fatal_crash']:.3f} "
                    f"policy={update_metrics['policy_loss']:.4f} "
                    f"value={update_metrics['value_loss']:.4f} "
                    f"kl={update_metrics['approx_kl']:.5f} "
                    f"clip={update_metrics['clip_fraction']:.3f} "
                    f"anchor={update_metrics['anchor_loss']:.5f} "
                    f"drone_std={float(log_std.mean().detach()):.3f} "
                    f"observer_std={float(observer_log_std.mean().detach()):.3f}"
                )
                save_checkpoint(
                    args.output_dir / "hissd_online_last.pt",
                    model,
                    source_payload,
                    value_head,
                    log_std,
                    observer_residual_policy,
                    observer_log_std,
                    optimizer,
                    stage=stage,
                    iteration=global_iteration,
                    stage_joint_env_steps=stage_joint_env_steps,
                    total_joint_env_steps=total_joint_env_steps,
                    metrics={**train_metrics, **update_metrics},
                    args=args,
                )
                if stage_iteration % args.eval_every != 0:
                    continue
                evaluation = evaluate_policy(
                    difficulty=stage,
                    checkpoint_env_config=checkpoint_env_config,
                    observer_algo=observer_algo,
                    model=model,
                    value_head=value_head,
                    anchor_head=anchor_head,
                    log_std=log_std,
                    observer_residual_policy=observer_residual_policy,
                    observer_anchor_head=observer_anchor_head,
                    observer_log_std=observer_log_std,
                    args=args,
                    device=device,
                )
                score = evaluation_score(evaluation)
                print(
                    f"EVAL stage={stage} success={evaluation['success']:.3f} "
                    f"goal={evaluation['goal_found']:.3f} "
                    f"fatal={evaluation['fatal_crash']:.3f} "
                    f"drone_crash={evaluation['drone_crash']:.3f} "
                    f"observer_crash={evaluation['observer_crash']:.3f} "
                    f"coverage={evaluation['coverage']:.3f} score={score:.4f}"
                )
                append_curve_points(
                    learning_curve_path,
                    [
                        {
                            "method": args.method_name,
                            "seed": args.seed,
                            "difficulty": stage,
                            "joint_env_steps": stage_joint_env_steps,
                            "total_joint_env_steps": total_joint_env_steps,
                            "iteration": global_iteration,
                            "success_rate": evaluation["success"],
                            "goal_found_rate": evaluation["goal_found"],
                            "fatal_crash_rate": evaluation["fatal_crash"],
                            "drone_crash_rate": evaluation["drone_crash"],
                            "observer_crash_rate": evaluation["observer_crash"],
                            "mean_coverage_ratio": evaluation["coverage"],
                            "mean_cycles": evaluation["cycles"],
                        }
                    ],
                )
                if score > stage_best_score:
                    stage_best_score = score
                    stage_best_state = {
                        "model": copy.deepcopy(model.state_dict()),
                        "value": copy.deepcopy(value_head.state_dict()),
                        "log_std": log_std.detach().clone(),
                        "observer": copy.deepcopy(
                            observer_residual_policy.state_dict()
                        ),
                        "observer_log_std": observer_log_std.detach().clone(),
                        "optimizer": copy.deepcopy(optimizer.state_dict()),
                    }
                    save_checkpoint(
                        args.output_dir / f"hissd_online_stage_{stage:02d}_best.pt",
                        model,
                        source_payload,
                        value_head,
                        log_std,
                        observer_residual_policy,
                        observer_log_std,
                        optimizer,
                        stage=stage,
                        iteration=global_iteration,
                        stage_joint_env_steps=stage_joint_env_steps,
                        total_joint_env_steps=total_joint_env_steps,
                        metrics=evaluation,
                        args=args,
                    )
                    if stage == args.end_stage:
                        save_checkpoint(
                            args.output_dir / "hissd_online_best.pt",
                            model,
                            source_payload,
                            value_head,
                            log_std,
                            observer_residual_policy,
                            observer_log_std,
                            optimizer,
                            stage=stage,
                            iteration=global_iteration,
                            stage_joint_env_steps=stage_joint_env_steps,
                            total_joint_env_steps=total_joint_env_steps,
                            metrics=evaluation,
                            args=args,
                        )
            model.load_state_dict(stage_best_state["model"])
            value_head.load_state_dict(stage_best_state["value"])
            observer_residual_policy.load_state_dict(stage_best_state["observer"])
            with torch.no_grad():
                log_std.copy_(stage_best_state["log_std"])
                observer_log_std.copy_(stage_best_state["observer_log_std"])
            optimizer.load_state_dict(stage_best_state["optimizer"])
        save_checkpoint(
            args.output_dir / "hissd_online_final.pt",
            model,
            source_payload,
            value_head,
            log_std,
            observer_residual_policy,
            observer_log_std,
            optimizer,
            stage=args.end_stage,
            iteration=global_iteration,
            stage_joint_env_steps=stage_joint_env_steps,
            total_joint_env_steps=total_joint_env_steps,
            metrics={},
            args=args,
        )
    finally:
        observer_algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
