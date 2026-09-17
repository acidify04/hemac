"""Adapt a homogeneous drone skill-VAE policy with parameter-shared PPO.

The default mode freezes the offline skill policy and learns a target residual
actor, while a training-only central critic uses the world map for CTDE credit
assignment. The earlier joint head fine-tuning remains available as an option.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from hemac import HeMAC_v0
from hemac.curriculum_config import OBSTACLE_CURRICULUM_LEVELS
from skill_discovery.analyze_learning_efficiency import append_curve_points
from skill_discovery.collect_offline_data import (
    _local_reward_after_step,
    agent_found_goal,
    build_global_central_map,
    get_core_env,
)
from skill_discovery.drone_skill_vae import load_drone_skill_vae
from skill_discovery.evaluate_drone_bc import drone_action_scale
from skill_discovery.finetune_hissd_drone_online import (
    add_per_drone_gae,
    build_env_config,
    configure_gpu_backend,
    evaluation_score,
    load_checkpoint_env_config,
    per_drone_squashed_log_prob,
)
from skill_discovery.finetune_hissd_online import (
    OnlineValueHead,
    average_metrics,
    drone_observation_batch,
)
from skill_discovery.hissd_models import CentralStateEncoder


DEFAULT_VAE_CHECKPOINT = (
    PROJECT_ROOT
    / "src/skill_discovery/checkpoints/drone_skill_vae/"
    "drone_skill_vae_best.pt"
)
DEFAULT_MAPPO_CHECKPOINT = (
    PROJECT_ROOT / "src/train/drone_mappo_coverage60_checkpoints/checkpoint_07800"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "src/skill_discovery/checkpoints/drone_skill_vae_online"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vae-checkpoint", type=Path, default=DEFAULT_VAE_CHECKPOINT)
    parser.add_argument("--mappo-checkpoint", type=Path, default=DEFAULT_MAPPO_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--difficulty",
        type=int,
        choices=range(1, len(OBSTACLE_CURRICULUM_LEVELS) + 1),
        required=True,
    )
    parser.add_argument("--success-min-coverage-ratio", type=float, default=0.6)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--episodes-per-iteration", type=int, default=8)
    parser.add_argument(
        "--train-batch-joint-steps",
        type=int,
        default=5000,
        help=(
            "Collect at least this many joint environment cycles before each "
            "PPO update. Set to 0 to use --episodes-per-iteration instead."
        ),
    )
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument(
        "--joint-step-budget",
        type=int,
        help=(
            "Stop after this many target joint environment cycles. When set, "
            "this takes precedence over --iterations."
        ),
    )
    parser.add_argument(
        "--eval-every-joint-steps",
        type=int,
        help=(
            "Evaluate whenever this many target joint cycles have elapsed. "
            "Must be used together with --joint-step-budget."
        ),
    )
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument(
        "--eval-seed-base",
        type=int,
        help=(
            "Seed base for the fixed validation environments. If omitted, the "
            "training seed is used for backward compatibility."
        ),
    )
    parser.add_argument(
        "--test-seed-base",
        type=int,
        help="Independent seed base for one held-out final test evaluation.",
    )
    parser.add_argument(
        "--test-episodes",
        type=int,
        default=0,
        help="Held-out episodes evaluated once after selecting the best checkpoint.",
    )
    parser.add_argument("--ppo-epochs", type=int, default=5)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--skill-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--log-std-lr", type=float, default=2e-5)
    parser.add_argument("--log-std-init", type=float, default=-2.0)
    parser.add_argument("--log-std-min", type=float, default=-3.0)
    parser.add_argument("--log-std-max", type=float, default=-0.7)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coeff", type=float, default=0.5)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--anchor-coeff", type=float, default=0.001)
    parser.add_argument("--skill-anchor-coeff", type=float, default=0.001)
    parser.add_argument("--skill-warmup-iterations", type=int, default=0)
    parser.add_argument(
        "--full-base-freeze-iterations",
        type=int,
        default=0,
        help=(
            "Keep the BC base head fixed for the first N full-skill PPO "
            "iterations so target adaptation cannot immediately bypass z."
        ),
    )
    parser.add_argument(
        "--full-base-freeze-joint-steps",
        type=int,
        default=0,
        help=(
            "Keep the BC base head fixed for this many initial target cycles "
            "in full-skill mode. This is ignored by no_skill."
        ),
    )
    parser.add_argument(
        "--skill-mode",
        choices=("full", "no_skill"),
        default="full",
        help=(
            "full uses the recurrent VAE latent and residual; no_skill removes "
            "both while retaining the same BC actor, critic, and PPO update."
        ),
    )
    parser.add_argument(
        "--ppo-actor-mode",
        choices=("skill_support", "joint_finetune"),
        default="skill_support",
        help=(
            "skill_support freezes the offline policy and trains a matched PPO "
            "residual; joint_finetune retains the earlier end-to-end head update."
        ),
    )
    parser.add_argument("--online-residual-scale", type=float, default=0.25)
    parser.add_argument(
        "--train-base-action-head",
        action="store_true",
        help=(
            "Also adapt the BC action head in joint_finetune mode. The matched "
            "skill_support control uses its separate residual actor instead."
        ),
    )
    parser.add_argument("--reward-scale", type=float, default=100.0)
    parser.add_argument(
        "--shared-terminal-crash-penalty",
        type=float,
        default=300.0,
        help=(
            "Team penalty assigned to every drone return when any drone causes "
            "a terminal crash. The crashing drone is not penalized twice."
        ),
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--method-name", default="drone_skill_vae_online")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use reproducible CUDA kernels and disable TF32/benchmark selection.",
    )
    parser.add_argument("--learning-curve-output", type=Path)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "iterations",
        "episodes_per_iteration",
        "eval_every",
        "eval_episodes",
        "ppo_epochs",
        "minibatch_size",
        "actor_lr",
        "skill_lr",
        "critic_lr",
        "log_std_lr",
        "gamma",
        "gae_lambda",
        "clip_ratio",
        "reward_scale",
        "shared_terminal_crash_penalty",
        "online_residual_scale",
        "max_grad_norm",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.train_batch_joint_steps < 0:
        raise ValueError("--train-batch-joint-steps cannot be negative.")
    if not 0.0 < args.success_min_coverage_ratio <= 1.0:
        raise ValueError("--success-min-coverage-ratio must be in (0, 1].")
    if args.log_std_min >= args.log_std_max:
        raise ValueError("--log-std-min must be lower than --log-std-max.")
    if args.anchor_coeff < 0 or args.skill_anchor_coeff < 0:
        raise ValueError("Anchor coefficients cannot be negative.")
    if (
        args.skill_warmup_iterations < 0
        or args.full_base_freeze_iterations < 0
        or args.full_base_freeze_joint_steps < 0
    ):
        raise ValueError("Skill/base warm-up iteration counts cannot be negative.")
    step_options = (
        args.joint_step_budget is not None,
        args.eval_every_joint_steps is not None,
    )
    if any(step_options) and not all(step_options):
        raise ValueError(
            "--joint-step-budget and --eval-every-joint-steps must be used together."
        )
    if args.joint_step_budget is not None:
        if args.joint_step_budget <= 0 or args.eval_every_joint_steps <= 0:
            raise ValueError(
                "Joint-step budget and evaluation interval must be positive."
            )
        if args.eval_every_joint_steps > args.joint_step_budget:
            raise ValueError(
                "--eval-every-joint-steps cannot exceed --joint-step-budget."
            )
    for name in ("eval_seed_base", "test_seed_base"):
        value = getattr(args, name)
        if value is not None and value < 0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative.")
    if args.test_episodes < 0:
        raise ValueError("--test-episodes cannot be negative.")
    if (args.test_seed_base is None) != (args.test_episodes == 0):
        raise ValueError(
            "--test-seed-base and a positive --test-episodes must be used together."
        )
    if (
        args.eval_seed_base is not None
        and args.test_seed_base is not None
        and args.eval_seed_base == args.test_seed_base
    ):
        raise ValueError("Validation and test seed bases must be different.")
    if (
        args.ppo_actor_mode == "joint_finetune"
        and args.skill_mode == "no_skill"
        and not args.train_base_action_head
    ):
        raise ValueError(
            "--skill-mode no_skill requires --train-base-action-head so the "
            "control actor can learn rather than only adapting exploration noise."
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def apply_shared_terminal_crash_penalty(
    cycle_rewards: np.ndarray,
    *,
    fatal_crash: bool,
    penalty: float,
) -> np.ndarray:
    """Propagate a terminal team failure without double-penalizing its cause."""
    if not fatal_crash:
        return cycle_rewards
    return np.minimum(cycle_rewards, -abs(float(penalty)))


class OnlineResidualPolicy(nn.Module):
    """Small target PPO actor placed on top of a frozen offline action prior."""

    def __init__(
        self,
        observation_dim: int,
        skill_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim + skill_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self, observation_features: torch.Tensor, skills: torch.Tensor
    ) -> torch.Tensor:
        inputs = torch.cat((observation_features, skills), dim=-1)
        return torch.tanh(self.network(inputs))


def configure_backend(device: torch.device, deterministic: bool) -> None:
    if not deterministic:
        configure_gpu_backend(device)
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    if device.type != "cuda":
        return
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def rollout_episode(
    *,
    env,
    model,
    online_residual_policy: OnlineResidualPolicy,
    central_encoder: CentralStateEncoder,
    value_head: OnlineValueHead,
    anchor_base_head: nn.Module,
    anchor_mu: nn.Module,
    anchor_skill_head: nn.Module,
    log_std: torch.Tensor,
    action_scale: float,
    reward_scale: float,
    shared_terminal_crash_penalty: float,
    seed: int,
    device: torch.device,
    stochastic: bool,
    skill_mode: str,
    ppo_actor_mode: str,
    online_residual_scale: float,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    env.reset(seed=seed)
    core_env = get_core_env(env)
    drone_ids = list(env.possible_agents)
    if not drone_ids or any(not name.startswith("drone_") for name in drone_ids):
        raise ValueError("The skill-VAE online runner requires a drone-only env.")
    last_agent_id = drone_ids[-1]
    state = model.initial_inference_state(len(drone_ids), device=device)
    transitions: list[dict[str, Any]] = []
    cached_actions: dict[str, np.ndarray] = {}
    cycle_rewards: np.ndarray | None = None
    cycle_mask: np.ndarray | None = None
    shared_success_reward = 0.0
    cycle_count = 0
    episode_return = 0.0
    skill_action_delta_sum = torch.zeros((), device=device)
    skill_latent_std_sum = torch.zeros((), device=device)
    skill_switch_count = 0

    for agent_id in env.agent_iter():
        _, _, termination, truncation, _ = env.last()
        if termination or truncation:
            env.step(None)
            continue

        if not cached_actions:
            observations = drone_observation_batch(env, drone_ids, action_scale, device)
            with torch.no_grad():
                outputs, state = model.inference_step(observations, state)
                features = outputs["observation_features"][0]
                contexts = outputs["skill_context"][0]
                if skill_mode == "full":
                    skills = outputs["skills"][0]
                    offline_logits = outputs["action_logits"][0]
                    anchor_skills = anchor_mu(contexts)
                    anchor_residual = model.skill_residual(
                        features,
                        anchor_skills,
                        head=anchor_skill_head,
                    )
                    anchor_logits = (
                        anchor_base_head(features)
                        + model.residual_logit_scale * anchor_residual
                    )
                    skill_switch_count += int(outputs["skill_switched"])
                else:
                    skills = torch.zeros_like(outputs["skills"][0])
                    offline_logits = model.base_action_head(features)
                    anchor_skills = torch.zeros_like(skills)
                    anchor_logits = anchor_base_head(features)
                if ppo_actor_mode == "skill_support":
                    logits = offline_logits + online_residual_scale * (
                        online_residual_policy(features, skills)
                    )
                    anchor_logits = offline_logits
                else:
                    logits = offline_logits
                base_actions = torch.tanh(model.base_action_head(features))
                policy_actions = torch.tanh(logits)
                skill_action_delta_sum += (
                    policy_actions - base_actions
                ).abs().mean()
                skill_latent_std_sum += skills.std(
                    dim=0, unbiased=False
                ).mean()
                central_map = torch.from_numpy(
                    build_global_central_map(core_env)
                ).unsqueeze(0).to(device)
                central = central_encoder(central_map)[0]
                critic_features = torch.cat(
                    (
                        features,
                        skills,
                        central.unsqueeze(0).expand(len(drone_ids), -1),
                    ),
                    dim=-1,
                )
                value = value_head(critic_features)
                distribution = Normal(logits, log_std.exp())
                raw_action = distribution.sample() if stochastic else logits
                normalized_action = torch.tanh(raw_action)
                old_log_prob = per_drone_squashed_log_prob(distribution, raw_action)

            for index, drone_id in enumerate(drone_ids):
                action_space = env.action_space(drone_id)
                action = normalized_action[index].cpu().numpy() * action_scale
                cached_actions[drone_id] = np.ascontiguousarray(
                    np.clip(action, action_space.low, action_space.high),
                    dtype=np.float32,
                )
            if stochastic:
                transitions.append(
                    {
                        "observation_features": features.cpu(),
                        "skill_context": contexts.cpu(),
                        "anchor_skill": anchor_skills.cpu(),
                        "anchor_logits": anchor_logits.cpu(),
                        "central_map": central_map[0].cpu(),
                        "raw_action": raw_action.cpu(),
                        "old_log_prob": old_log_prob.cpu(),
                        "value": value.cpu(),
                    }
                )
            cycle_rewards = np.zeros(len(drone_ids), dtype=np.float32)
            cycle_mask = np.zeros(len(drone_ids), dtype=np.bool_)
            shared_success_reward = 0.0

        drone_index = drone_ids.index(agent_id)
        cycle_mask[drone_index] = True
        env.step(cached_actions[agent_id])
        for reward_index, reward_agent_id in enumerate(drone_ids):
            cycle_rewards[reward_index] += _local_reward_after_step(
                core_env, reward_agent_id
            )
        shared_success_reward = max(shared_success_reward, float(core_env.global_reward))
        cycle_finished = (
            agent_id == last_agent_id
            or bool(core_env.terminate)
            or bool(core_env.truncate)
        )
        if not cycle_finished:
            continue
        cycle_rewards = apply_shared_terminal_crash_penalty(
            cycle_rewards,
            fatal_crash=bool(core_env.collided),
            penalty=shared_terminal_crash_penalty,
        )
        if stochastic:
            transitions[-1]["agent_mask"] = torch.from_numpy(cycle_mask.copy())
            transitions[-1]["reward"] = torch.from_numpy(
                (cycle_rewards + shared_success_reward) / reward_scale
            )
        if np.any(cycle_mask):
            episode_return += (
                float(cycle_rewards[cycle_mask].mean()) + shared_success_reward
            )
        cycle_count += 1
        cached_actions = {}
        cycle_rewards = None
        cycle_mask = None

    goal_found = any(agent_found_goal(core_env, drone_id) for drone_id in drone_ids)
    coverage = float(core_env.current_coverage_ratio())
    reward_coverage = float(core_env.current_drone_reward_coverage_ratio())
    fatal_crash = bool(core_env.collided)
    # The environment excludes base sectors from drone-only success coverage.
    # Use its terminal decision rather than reclassifying with full-map coverage.
    success = bool(core_env.mission_success)
    return transitions, {
        "success": float(success),
        "goal_found": float(goal_found),
        "fatal_crash": float(fatal_crash),
        "drone_crash": float(bool(core_env.drone_crash)),
        "observer_crash": 0.0,
        "coverage": coverage,
        "reward_coverage": reward_coverage,
        "cycles": float(cycle_count),
        "episode_return": float(episode_return),
        "skill_action_delta": float(
            (skill_action_delta_sum / max(cycle_count, 1)).cpu()
        ),
        "skill_latent_std": float(
            (skill_latent_std_sum / max(cycle_count, 1)).cpu()
        ),
        "skill_switch_rate": float(skill_switch_count / max(cycle_count, 1)),
    }


def ppo_update(
    model,
    online_residual_policy: OnlineResidualPolicy,
    central_encoder: CentralStateEncoder,
    value_head: OnlineValueHead,
    log_std: nn.Parameter,
    optimizer: torch.optim.Optimizer,
    transitions: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
    *,
    adapt_skill: bool,
    skill_mode: str,
    ppo_actor_mode: str,
    online_residual_scale: float,
) -> dict[str, float]:
    if not transitions:
        raise RuntimeError("No PPO transitions were collected.")
    names = (
        "observation_features",
        "skill_context",
        "anchor_skill",
        "anchor_logits",
        "central_map",
        "raw_action",
        "old_log_prob",
        "agent_mask",
    )
    data = {
        name: torch.stack([transition[name] for transition in transitions]).to(device)
        for name in names
    }
    advantages = torch.stack(
        [torch.as_tensor(item["advantage"]) for item in transitions]
    ).to(device)
    returns = torch.stack(
        [torch.as_tensor(item["return"]) for item in transitions]
    ).to(device)
    mask = data["agent_mask"].bool()
    valid_advantages = advantages[mask]
    advantages = torch.where(
        mask,
        (advantages - valid_advantages.mean())
        / valid_advantages.std(unbiased=False).clamp_min(1e-6),
        0.0,
    )
    count = len(transitions)
    totals: dict[str, float] = defaultdict(float)
    updates = 0
    for _ in range(args.ppo_epochs):
        permutation = torch.randperm(count, device=device)
        for start in range(0, count, args.minibatch_size):
            indices = permutation[start : start + args.minibatch_size]
            contexts = data["skill_context"][indices]
            if skill_mode == "no_skill":
                skills = torch.zeros_like(data["anchor_skill"][indices])
            elif adapt_skill and ppo_actor_mode == "joint_finetune":
                skills = model.posterior_mu(contexts)
            else:
                skills = data["anchor_skill"][indices]
            features = data["observation_features"][indices]
            if ppo_actor_mode == "skill_support":
                logits = data["anchor_logits"][indices] + online_residual_scale * (
                    online_residual_policy(features, skills)
                )
            elif skill_mode == "full":
                logits = model.decode_logits(features, skills)
            else:
                logits = model.base_action_head(features)
            distribution = Normal(logits, log_std.exp())
            log_prob = per_drone_squashed_log_prob(
                distribution, data["raw_action"][indices]
            )
            ratio = torch.exp(log_prob - data["old_log_prob"][indices])
            advantage = advantages[indices]
            unclipped = ratio * advantage
            clipped = ratio.clamp(
                1.0 - args.clip_ratio, 1.0 + args.clip_ratio
            ) * advantage
            minibatch_mask = mask[indices]
            policy_loss = -torch.minimum(unclipped, clipped)[minibatch_mask].mean()

            central = central_encoder(data["central_map"][indices])
            critic_features = torch.cat(
                (
                    features,
                    skills,
                    central.unsqueeze(1).expand(-1, features.shape[1], -1),
                ),
                dim=-1,
            )
            values = value_head(critic_features)
            value_loss = F.mse_loss(
                values[minibatch_mask], returns[indices][minibatch_mask]
            )
            entropy = distribution.entropy().sum(dim=-1)[minibatch_mask].mean()
            action_mask = minibatch_mask.unsqueeze(-1).expand_as(logits)
            anchor_loss = (
                torch.tanh(logits) - torch.tanh(data["anchor_logits"][indices])
            ).square()[action_mask].mean()
            skill_mask = minibatch_mask.unsqueeze(-1).expand_as(skills)
            skill_anchor_loss = (
                skills - data["anchor_skill"][indices]
            ).square()[skill_mask].mean()
            loss = (
                policy_loss
                + args.value_coeff * value_loss
                - args.entropy_coeff * entropy
                + args.anchor_coeff * anchor_loss
                + args.skill_anchor_coeff * skill_anchor_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            trainable = [
                *central_encoder.parameters(),
                *value_head.parameters(),
                log_std,
            ]
            if ppo_actor_mode == "skill_support":
                trainable.extend(online_residual_policy.parameters())
            else:
                trainable.extend(model.skill_action_head.parameters())
                if args.train_base_action_head:
                    trainable.extend(model.base_action_head.parameters())
                if adapt_skill:
                    trainable.extend(model.posterior_mu.parameters())
            grad_norm = nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                log_std.clamp_(args.log_std_min, args.log_std_max)
            totals["policy_loss"] += float(policy_loss.detach())
            totals["value_loss"] += float(value_loss.detach())
            totals["entropy"] += float(entropy.detach())
            totals["anchor_loss"] += float(anchor_loss.detach())
            totals["skill_anchor_loss"] += float(skill_anchor_loss.detach())
            totals["approx_kl"] += float(
                (data["old_log_prob"][indices] - log_prob)[minibatch_mask]
                .mean()
                .detach()
            )
            totals["clip_fraction"] += float(
                ((ratio - 1.0).abs() > args.clip_ratio)[minibatch_mask]
                .float()
                .mean()
                .detach()
            )
            totals["grad_norm"] += float(grad_norm)
            updates += 1
    return {name: value / max(updates, 1) for name, value in totals.items()}


def evaluate_policy(
    *,
    checkpoint_env_config: dict[str, Any],
    model,
    online_residual_policy: OnlineResidualPolicy,
    central_encoder: CentralStateEncoder,
    value_head: OnlineValueHead,
    anchor_base_head: nn.Module,
    anchor_mu: nn.Module,
    anchor_skill_head: nn.Module,
    log_std: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    episode_count: int | None = None,
    seed_base: int | None = None,
) -> dict[str, float]:
    config = build_env_config(
        checkpoint_env_config,
        args.difficulty,
        args.success_min_coverage_ratio,
    )
    results = []
    episode_count = args.eval_episodes if episode_count is None else episode_count
    seed_base = args.seed if seed_base is None else seed_base
    for episode_index in range(episode_count):
        env = HeMAC_v0.env(**config)
        try:
            _, metrics = rollout_episode(
                env=env,
                model=model,
                online_residual_policy=online_residual_policy,
                central_encoder=central_encoder,
                value_head=value_head,
                anchor_base_head=anchor_base_head,
                anchor_mu=anchor_mu,
                anchor_skill_head=anchor_skill_head,
                log_std=log_std,
                action_scale=drone_action_scale(config),
                reward_scale=args.reward_scale,
                shared_terminal_crash_penalty=(
                    args.shared_terminal_crash_penalty
                ),
                seed=seed_base + args.difficulty * 1_000_000 + episode_index,
                device=device,
                stochastic=False,
                skill_mode=args.skill_mode,
                ppo_actor_mode=args.ppo_actor_mode,
                online_residual_scale=args.online_residual_scale,
            )
        finally:
            env.close()
        results.append(metrics)
    return average_metrics(results)


def curve_point(
    args: argparse.Namespace,
    metrics: dict[str, float],
    joint_env_steps: int,
    iteration: int,
    scheduled_joint_env_steps: int | None = None,
) -> dict[str, Any]:
    point = {
        "method": args.method_name,
        "seed": args.seed,
        "difficulty": args.difficulty,
        "joint_env_steps": joint_env_steps,
        "iteration": iteration,
        "success_rate": metrics["success"],
        "goal_found_rate": metrics["goal_found"],
        "fatal_crash_rate": metrics["fatal_crash"],
        "drone_crash_rate": metrics["drone_crash"],
        "observer_crash_rate": 0.0,
        "mean_coverage_ratio": metrics["coverage"],
        "mean_drone_reward_coverage_ratio": metrics["reward_coverage"],
        "mean_cycles": metrics["cycles"],
        "mean_episode_return": metrics["episode_return"],
        "mean_skill_action_delta": metrics["skill_action_delta"],
        "mean_skill_latent_std": metrics["skill_latent_std"],
        "mean_skill_switch_rate": metrics["skill_switch_rate"],
    }
    if scheduled_joint_env_steps is not None:
        point["scheduled_joint_env_steps"] = scheduled_joint_env_steps
    return point


def save_checkpoint(
    path: Path,
    model,
    online_residual_policy: OnlineResidualPolicy,
    source_payload: dict[str, Any],
    central_encoder: CentralStateEncoder,
    value_head: OnlineValueHead,
    log_std: nn.Parameter,
    optimizer: torch.optim.Optimizer,
    *,
    iteration: int,
    joint_env_steps: int,
    metrics: dict[str, float],
    args: argparse.Namespace,
) -> None:
    payload = dict(source_payload)
    payload.update(
        {
            "model_config": model.config(),
            "model_state_dict": model.state_dict(),
            "online_residual_policy_state_dict": (
                online_residual_policy.state_dict()
            ),
            "online_central_encoder_config": {
                "channels": central_encoder.channels,
                "map_size": central_encoder.map_size,
                "hidden_dim": central_encoder.hidden_dim,
            },
            "online_central_encoder_state_dict": central_encoder.state_dict(),
            "online_value_state_dict": value_head.state_dict(),
            "online_drone_log_std": log_std.detach().cpu(),
            "online_optimizer_state_dict": optimizer.state_dict(),
            "online_finetuning": {
                "method": "homogeneous_skill_vae_parameter_sharing_ppo",
                "skill_mode": args.skill_mode,
                "ppo_actor_mode": args.ppo_actor_mode,
                "difficulty": args.difficulty,
                "iteration": iteration,
                "joint_env_steps": joint_env_steps,
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
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    configure_backend(device, args.deterministic)
    model, source_payload = load_drone_skill_vae(args.vae_checkpoint, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    joint_finetune = args.ppo_actor_mode == "joint_finetune"
    for parameter in model.skill_action_head.parameters():
        parameter.requires_grad_(joint_finetune)
    initial_base_trainable = args.train_base_action_head and not (
        args.skill_mode == "full" and args.full_base_freeze_iterations > 0
    )
    for parameter in model.base_action_head.parameters():
        parameter.requires_grad_(joint_finetune and initial_base_trainable)
    for parameter in model.posterior_mu.parameters():
        parameter.requires_grad_(False)
    model.eval()
    anchor_base_head = copy.deepcopy(model.base_action_head).to(device).eval()
    anchor_mu = copy.deepcopy(model.posterior_mu).to(device).eval()
    anchor_skill_head = copy.deepcopy(model.skill_action_head).to(device).eval()
    for module in (anchor_base_head, anchor_mu, anchor_skill_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    online_residual_policy = OnlineResidualPolicy(
        model.observation_dim,
        model.latent_dim,
        model.action_dim,
    ).to(device)

    resolved_mappo, checkpoint_env_config = load_checkpoint_env_config(
        args.mappo_checkpoint
    )
    config = build_env_config(
        checkpoint_env_config,
        args.difficulty,
        args.success_min_coverage_ratio,
    )
    probe_env = HeMAC_v0.env(**config)
    try:
        probe_env.reset(seed=args.seed)
        central_shape = build_global_central_map(get_core_env(probe_env)).shape
    finally:
        probe_env.close()
    central_encoder = CentralStateEncoder(
        central_shape[0], tuple(central_shape[-2:]), hidden_dim=64
    ).to(device)
    critic_input_dim = model.observation_dim + model.latent_dim + 64
    value_head = OnlineValueHead(critic_input_dim).to(device)
    log_std = nn.Parameter(
        torch.full((model.action_dim,), args.log_std_init, device=device)
    )
    optimizer = torch.optim.AdamW(
        [
            {
                "params": online_residual_policy.parameters(),
                "lr": args.actor_lr,
            },
            {"params": model.base_action_head.parameters(), "lr": args.actor_lr},
            {"params": model.skill_action_head.parameters(), "lr": args.actor_lr},
            {"params": model.posterior_mu.parameters(), "lr": args.skill_lr},
            {"params": central_encoder.parameters(), "lr": args.critic_lr},
            {"params": value_head.parameters(), "lr": args.critic_lr},
            {"params": [log_std], "lr": args.log_std_lr},
        ],
        weight_decay=1e-5,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    curve_path = (
        args.learning_curve_output.expanduser().resolve()
        if args.learning_curve_output is not None
        else args.output_dir / f"learning_curve_seed_{args.seed}.json"
    )
    print(
        f"Drone skill-VAE PPO difficulty={args.difficulty}, seed={args.seed}, "
        f"skill_mode={args.skill_mode}, train_base={args.train_base_action_head}, "
        f"ppo_actor_mode={args.ppo_actor_mode}, "
        f"base_freeze_iterations={args.full_base_freeze_iterations}, "
        f"base_freeze_steps={args.full_base_freeze_joint_steps}, "
        f"train_batch_steps={args.train_batch_joint_steps}, "
        f"joint_step_budget={args.joint_step_budget}, "
        f"eval_step_interval={args.eval_every_joint_steps}, "
        f"deterministic={args.deterministic}, "
        f"checkpoint={args.vae_checkpoint}, env_checkpoint={resolved_mappo}"
    )
    baseline = evaluate_policy(
        checkpoint_env_config=checkpoint_env_config,
        model=model,
        online_residual_policy=online_residual_policy,
        central_encoder=central_encoder,
        value_head=value_head,
        anchor_base_head=anchor_base_head,
        anchor_mu=anchor_mu,
        anchor_skill_head=anchor_skill_head,
        log_std=log_std,
        args=args,
        device=device,
        seed_base=args.eval_seed_base,
    )
    joint_env_steps = 0
    append_curve_points(
        curve_path,
        [curve_point(args, baseline, 0, 0)],
        metadata={
            "evaluation_episodes": args.eval_episodes,
            "validation_seed_base": (
                args.seed if args.eval_seed_base is None else args.eval_seed_base
            ),
            "test_seed_base": args.test_seed_base,
            "test_episodes": args.test_episodes,
            "requested_joint_step_budget": args.joint_step_budget,
            "evaluation_joint_step_interval": args.eval_every_joint_steps,
            "source_checkpoint": str(args.vae_checkpoint.resolve()),
            "environment_checkpoint": str(resolved_mappo),
            "curve_method": args.method_name,
            "skill_model": "single_latent_homogeneous_drone_vae",
            "skill_mode": args.skill_mode,
            "ppo_actor_mode": args.ppo_actor_mode,
            "online_residual_scale": args.online_residual_scale,
            "skill_duration": model.skill_duration,
            "decoder_observation_conditioned": (
                model.decoder_observation_conditioned
            ),
            "train_base_action_head": args.train_base_action_head,
            "full_base_freeze_iterations": args.full_base_freeze_iterations,
            "full_base_freeze_joint_steps": args.full_base_freeze_joint_steps,
            "train_batch_joint_steps": args.train_batch_joint_steps,
            "shared_terminal_crash_penalty": (
                args.shared_terminal_crash_penalty
            ),
            "deterministic": args.deterministic,
            "success_definition": "environment_mission_success",
            "success_coverage_definition": (
                "drone_reward_coverage_excluding_base_sectors"
            ),
            "success_min_coverage_ratio": args.success_min_coverage_ratio,
            "training_step_definition": (
                "sum of world cycles from target training episodes; "
                "evaluation cycles excluded"
            ),
            "evaluation_reward_definition": (
                "unscaled episode sum over world cycles of mean active-drone "
                "local reward plus one shared success reward"
            ),
        },
    )
    best_score = evaluation_score(baseline)
    best_state = {
        "model": copy.deepcopy(model.state_dict()),
        "online_residual": copy.deepcopy(online_residual_policy.state_dict()),
        "central": copy.deepcopy(central_encoder.state_dict()),
        "value": copy.deepcopy(value_head.state_dict()),
        "log_std": log_std.detach().clone(),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
    }
    print(
        f"BASELINE success={baseline['success']:.3f} "
        f"goal={baseline['goal_found']:.3f} crash={baseline['fatal_crash']:.3f} "
        f"coverage={baseline['coverage']:.3f} "
        f"reward_coverage={baseline['reward_coverage']:.3f} "
        f"return={baseline['episode_return']:.2f} "
        f"skill_delta={baseline['skill_action_delta']:.4f} "
        f"skill_switch={baseline['skill_switch_rate']:.3f}"
    )
    save_checkpoint(
        args.output_dir / "drone_skill_vae_online_best.pt",
        model,
        online_residual_policy,
        source_payload,
        central_encoder,
        value_head,
        log_std,
        optimizer,
        iteration=0,
        joint_env_steps=0,
        metrics=baseline,
        args=args,
    )

    iteration = 0
    next_eval_step = args.eval_every_joint_steps
    while (
        joint_env_steps < args.joint_step_budget
        if args.joint_step_budget is not None
        else iteration < args.iterations
    ):
        iteration += 1
        base_is_frozen = args.skill_mode == "full" and (
            iteration <= args.full_base_freeze_iterations
            or joint_env_steps < args.full_base_freeze_joint_steps
        )
        base_trainable = (
            joint_finetune and args.train_base_action_head and not base_is_frozen
        )
        for parameter in model.base_action_head.parameters():
            parameter.requires_grad_(base_trainable)
        trajectories: list[dict[str, Any]] = []
        episode_metrics = []
        batch_start_steps = joint_env_steps
        episode_index = 0
        while True:
            env = HeMAC_v0.env(**config)
            try:
                episode, metrics = rollout_episode(
                    env=env,
                    model=model,
                    online_residual_policy=online_residual_policy,
                    central_encoder=central_encoder,
                    value_head=value_head,
                    anchor_base_head=anchor_base_head,
                    anchor_mu=anchor_mu,
                    anchor_skill_head=anchor_skill_head,
                    log_std=log_std,
                    action_scale=drone_action_scale(config),
                    reward_scale=args.reward_scale,
                    shared_terminal_crash_penalty=(
                        args.shared_terminal_crash_penalty
                    ),
                    seed=args.seed + iteration * 10_000 + episode_index,
                    device=device,
                    stochastic=True,
                    skill_mode=args.skill_mode,
                    ppo_actor_mode=args.ppo_actor_mode,
                    online_residual_scale=args.online_residual_scale,
                )
            finally:
                env.close()
            add_per_drone_gae(episode, args.gamma, args.gae_lambda)
            trajectories.extend(episode)
            episode_metrics.append(metrics)
            joint_env_steps += int(round(metrics["cycles"]))
            episode_index += 1
            if (
                args.joint_step_budget is not None
                and joint_env_steps >= args.joint_step_budget
            ):
                break
            if args.train_batch_joint_steps > 0:
                if (
                    joint_env_steps - batch_start_steps
                    >= args.train_batch_joint_steps
                ):
                    break
            elif episode_index >= args.episodes_per_iteration:
                break
        adapt_skill = (
            joint_finetune
            and args.skill_mode == "full"
            and iteration > args.skill_warmup_iterations
        )
        for parameter in model.posterior_mu.parameters():
            parameter.requires_grad_(adapt_skill)
        update = ppo_update(
            model,
            online_residual_policy,
            central_encoder,
            value_head,
            log_std,
            optimizer,
            trajectories,
            args,
            device,
            adapt_skill=adapt_skill,
            skill_mode=args.skill_mode,
            ppo_actor_mode=args.ppo_actor_mode,
            online_residual_scale=args.online_residual_scale,
        )
        train = average_metrics(episode_metrics)
        progress_target = (
            f"steps_target={args.joint_step_budget}"
            if args.joint_step_budget is not None
            else f"iterations_target={args.iterations}"
        )
        print(
            f"iter={iteration:03d} steps={joint_env_steps} {progress_target} "
            f"success={train['success']:.3f} crash={train['fatal_crash']:.3f} "
            f"episodes={len(episode_metrics)} "
            f"return={train['episode_return']:.2f} "
            f"skill_delta={train['skill_action_delta']:.4f} "
            f"skill_switch={train['skill_switch_rate']:.3f} "
            f"policy={update['policy_loss']:.4f} value={update['value_loss']:.4f} "
            f"kl={update['approx_kl']:.5f} clip={update['clip_fraction']:.3f} "
            f"skill_mode={args.skill_mode} "
            f"actor_mode={args.ppo_actor_mode} "
            f"skill_update={'on' if adapt_skill else 'off'} "
            f"base_update={'on' if base_trainable else 'off'}"
        )
        save_checkpoint(
            args.output_dir / "drone_skill_vae_online_last.pt",
            model,
            online_residual_policy,
            source_payload,
            central_encoder,
            value_head,
            log_std,
            optimizer,
            iteration=iteration,
            joint_env_steps=joint_env_steps,
            metrics={**train, **update},
            args=args,
        )
        if args.joint_step_budget is not None:
            should_evaluate = joint_env_steps >= next_eval_step
            scheduled_eval_step = min(next_eval_step, args.joint_step_budget)
        else:
            should_evaluate = iteration % args.eval_every == 0
            scheduled_eval_step = None
        if not should_evaluate:
            continue
        evaluation = evaluate_policy(
            checkpoint_env_config=checkpoint_env_config,
            model=model,
            online_residual_policy=online_residual_policy,
            central_encoder=central_encoder,
            value_head=value_head,
            anchor_base_head=anchor_base_head,
            anchor_mu=anchor_mu,
            anchor_skill_head=anchor_skill_head,
            log_std=log_std,
            args=args,
            device=device,
            seed_base=args.eval_seed_base,
        )
        append_curve_points(
            curve_path,
            [
                curve_point(
                    args,
                    evaluation,
                    joint_env_steps,
                    iteration,
                    scheduled_joint_env_steps=scheduled_eval_step,
                )
            ],
        )
        if args.joint_step_budget is not None:
            while next_eval_step <= joint_env_steps:
                next_eval_step += args.eval_every_joint_steps
        score = evaluation_score(evaluation)
        print(
            f"EVAL success={evaluation['success']:.3f} "
            f"goal={evaluation['goal_found']:.3f} "
            f"crash={evaluation['fatal_crash']:.3f} "
            f"coverage={evaluation['coverage']:.3f} "
            f"reward_coverage={evaluation['reward_coverage']:.3f} "
            f"return={evaluation['episode_return']:.2f} "
            f"skill_delta={evaluation['skill_action_delta']:.4f} score={score:.4f}"
        )
        if score > best_score:
            best_score = score
            best_state = {
                "model": copy.deepcopy(model.state_dict()),
                "online_residual": copy.deepcopy(
                    online_residual_policy.state_dict()
                ),
                "central": copy.deepcopy(central_encoder.state_dict()),
                "value": copy.deepcopy(value_head.state_dict()),
                "log_std": log_std.detach().clone(),
                "optimizer": copy.deepcopy(optimizer.state_dict()),
            }
            save_checkpoint(
                args.output_dir / "drone_skill_vae_online_best.pt",
                model,
                online_residual_policy,
                source_payload,
                central_encoder,
                value_head,
                log_std,
                optimizer,
                iteration=iteration,
                joint_env_steps=joint_env_steps,
                metrics=evaluation,
                args=args,
            )

    model.load_state_dict(best_state["model"])
    online_residual_policy.load_state_dict(best_state["online_residual"])
    central_encoder.load_state_dict(best_state["central"])
    value_head.load_state_dict(best_state["value"])
    with torch.no_grad():
        log_std.copy_(best_state["log_std"])
    optimizer.load_state_dict(best_state["optimizer"])
    save_checkpoint(
        args.output_dir / "drone_skill_vae_online_final.pt",
        model,
        online_residual_policy,
        source_payload,
        central_encoder,
        value_head,
        log_std,
        optimizer,
        iteration=iteration,
        joint_env_steps=joint_env_steps,
        metrics={"best_score": best_score},
        args=args,
    )
    if args.test_seed_base is not None:
        test_metrics = evaluate_policy(
            checkpoint_env_config=checkpoint_env_config,
            model=model,
            online_residual_policy=online_residual_policy,
            central_encoder=central_encoder,
            value_head=value_head,
            anchor_base_head=anchor_base_head,
            anchor_mu=anchor_mu,
            anchor_skill_head=anchor_skill_head,
            log_std=log_std,
            args=args,
            device=device,
            episode_count=args.test_episodes,
            seed_base=args.test_seed_base,
        )
        test_payload = {
            "method": args.method_name,
            "training_seed": args.seed,
            "difficulty": args.difficulty,
            "test_seed_base": args.test_seed_base,
            "test_episodes": args.test_episodes,
            "selected_validation_score": best_score,
            "training_joint_env_steps": joint_env_steps,
            "metrics": test_metrics,
        }
        test_path = args.output_dir / "final_test_metrics.json"
        temporary = test_path.with_suffix(test_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(test_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(test_path)
        print(
            f"FINAL TEST success={test_metrics['success']:.3f} "
            f"goal={test_metrics['goal_found']:.3f} "
            f"crash={test_metrics['fatal_crash']:.3f} "
            f"return={test_metrics['episode_return']:.2f} path={test_path}"
        )
    print(f"Saved learning curve: {curve_path}")


if __name__ == "__main__":
    main()
