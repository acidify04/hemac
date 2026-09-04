"""Fine-tune HiSSD drone residuals with PPO and record learning efficiency.

This runner is intentionally separate from joint observer-drone fine-tuning.
It keeps every pretrained encoder and the BC-compatible action path frozen and
updates only the task-conditioned drone residual, exploration variance, and a
centralized team-value head.
"""

from __future__ import annotations

import argparse
import copy
import math
import pickle
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
    _team_reward,
    agent_found_goal,
    build_collection_env_config,
    build_global_central_map,
    get_core_env,
    resolve_algorithm_checkpoint,
)
from skill_discovery.evaluate_drone_bc import drone_action_scale
from skill_discovery.finetune_hissd_online import (
    OnlineValueHead,
    add_gae,
    average_metrics,
    drone_observation_batch,
    seed_everything,
)
from skill_discovery.visualize_hissd_skills import load_hissd_model, resolve_device


DEFAULT_HISSD_CHECKPOINT = (
    PROJECT_ROOT
    / "src/skill_discovery/checkpoints/"
    "hissd_drone_d12_t34_adapted_stable_v3/hissd_adapted_best.pt"
)
DEFAULT_MAPPO_CHECKPOINT = (
    PROJECT_ROOT / "src/train/drone_mappo_coverage60_checkpoints/checkpoint_07800"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "src/skill_discovery/checkpoints/hissd_drone_online"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hissd-checkpoint", type=Path, default=DEFAULT_HISSD_CHECKPOINT)
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
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--log-std-lr", type=float, default=2e-5)
    parser.add_argument("--log-std-init", type=float, default=-2.0)
    parser.add_argument("--log-std-min", type=float, default=-3.0)
    parser.add_argument("--log-std-max", type=float, default=-0.7)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.15)
    parser.add_argument("--value-coeff", type=float, default=0.5)
    parser.add_argument("--entropy-coeff", type=float, default=0.002)
    parser.add_argument("--anchor-coeff", type=float, default=0.02)
    parser.add_argument("--reward-scale", type=float, default=100.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--method-name", default="hissd_adapted_online")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--learning-curve-output", type=Path)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "iterations",
        "episodes_per_iteration",
        "eval_every",
        "eval_episodes",
        "ppo_epochs",
        "minibatch_size",
        "actor_lr",
        "critic_lr",
        "log_std_lr",
        "gamma",
        "gae_lambda",
        "clip_ratio",
        "reward_scale",
        "max_grad_norm",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if not 0.0 < args.success_min_coverage_ratio <= 1.0:
        raise ValueError("--success-min-coverage-ratio must be in (0, 1].")
    if not 0.0 < args.gamma <= 1.0 or not 0.0 < args.gae_lambda <= 1.0:
        raise ValueError("--gamma and --gae-lambda must be in (0, 1].")
    if args.log_std_min >= args.log_std_max:
        raise ValueError("--log-std-min must be lower than --log-std-max.")
    args.method_name = args.method_name.strip()
    if not args.method_name:
        raise ValueError("--method-name cannot be empty.")


def load_checkpoint_env_config(path: Path) -> tuple[Path, dict[str, Any]]:
    """Read only RLlib's environment configuration without starting Ray."""
    checkpoint = resolve_algorithm_checkpoint(path)
    state_path = checkpoint / "algorithm_state.pkl"
    with state_path.open("rb") as file:
        state = pickle.load(file)
    config = state.get("config", {})
    env_config = dict(config.get("env_config", {}) or {})
    if int(env_config.get("n_observers", 0)) != 0:
        raise ValueError(
            "Drone-only learning efficiency requires an RLlib checkpoint with "
            "n_observers=0. Use finetune_hissd_online.py for the joint mission."
        )
    if int(env_config.get("n_drones", 0)) <= 0:
        raise ValueError("The RLlib checkpoint does not configure any drones.")
    return checkpoint, env_config


def per_drone_squashed_log_prob(
    distribution: Normal,
    raw_action: torch.Tensor,
) -> torch.Tensor:
    """Return one tanh-corrected log probability per shared-policy drone."""
    correction = 2.0 * (
        math.log(2.0) - raw_action - F.softplus(-2.0 * raw_action)
    )
    return (distribution.log_prob(raw_action) - correction).sum(dim=-1)


def build_env_config(
    checkpoint_env_config: dict[str, Any],
    difficulty: int,
    success_min_coverage_ratio: float,
) -> dict[str, Any]:
    config = build_collection_env_config(checkpoint_env_config, difficulty)
    config["n_observers"] = 0
    config["drone_only_success_min_coverage_ratio"] = success_min_coverage_ratio
    return config


def rollout_episode(
    *,
    env,
    model,
    value_head: OnlineValueHead,
    anchor_head: nn.Module,
    log_std: torch.Tensor,
    action_scale: float,
    reward_scale: float,
    success_min_coverage_ratio: float,
    seed: int,
    device: torch.device,
    stochastic: bool,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    env.reset(seed=seed)
    core_env = get_core_env(env)
    agent_order = list(env.possible_agents)
    drone_ids = [agent_id for agent_id in agent_order if agent_id.startswith("drone_")]
    if len(drone_ids) != len(agent_order):
        raise ValueError("Drone-only fine-tuning received a non-drone agent.")
    drone_indices = list(range(len(drone_ids)))
    last_agent_id = agent_order[-1]
    valid_mask = torch.ones(1, len(drone_ids), dtype=torch.bool, device=device)
    recurrent_state = model.initial_inference_state(device=device)
    transitions: list[dict[str, Any]] = []
    cached_actions: dict[str, np.ndarray] = {}
    cycle_rewards: np.ndarray | None = None
    cycle_mask: np.ndarray | None = None
    shared_success_reward = 0.0
    cycle_count = 0

    for agent_id in env.agent_iter():
        _, _, termination, truncation, _ = env.last()
        if termination or truncation:
            env.step(None)
            continue

        if not cached_actions:
            observations = drone_observation_batch(env, drone_ids, action_scale, device)
            with torch.no_grad():
                outputs, recurrent_state = model.inference_step(
                    observations, valid_mask, recurrent_state
                )
                task_skill = outputs["task_skills"][0]
                action_logits = outputs["action_logits"][0]
                direct_logits = model.action_decoder.task_action_residual_head(task_skill)
                base_logits = action_logits - direct_logits
                anchor_logits = base_logits + anchor_head(task_skill)
                central_embedding = model.central_state_encoder(
                    torch.from_numpy(build_global_central_map(core_env))
                    .unsqueeze(0)
                    .to(device)
                )[0]
                critic_features = torch.cat(
                    (
                        outputs["observation_features"][0].mean(dim=0),
                        outputs["common_skills"][0].mean(dim=0),
                        task_skill.mean(dim=0),
                        central_embedding,
                    ),
                    dim=-1,
                )
                value = value_head(critic_features.unsqueeze(0))[0]
                distribution = Normal(action_logits, log_std.exp())
                raw_action = distribution.sample() if stochastic else action_logits
                normalized_action = torch.tanh(raw_action)
                old_log_prob = per_drone_squashed_log_prob(distribution, raw_action)

            for drone_index, drone_id in enumerate(drone_ids):
                action_space = env.action_space(drone_id)
                action = normalized_action[drone_index].cpu().numpy() * action_scale
                cached_actions[drone_id] = np.ascontiguousarray(
                    np.clip(action, action_space.low, action_space.high),
                    dtype=np.float32,
                )
            if stochastic:
                transitions.append(
                    {
                        "task_skill": task_skill.cpu(),
                        "base_logits": base_logits.cpu(),
                        "anchor_logits": anchor_logits.cpu(),
                        "raw_action": raw_action.cpu(),
                        "old_log_prob": old_log_prob.cpu(),
                        "critic_features": critic_features.cpu(),
                        "value": value.cpu(),
                    }
                )
            cycle_rewards = np.zeros(len(agent_order), dtype=np.float32)
            cycle_mask = np.zeros(len(agent_order), dtype=np.bool_)
            shared_success_reward = 0.0

        index = drone_ids.index(agent_id)
        cycle_mask[index] = True
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
        team_reward = _team_reward(
            cycle_rewards,
            cycle_mask,
            [],
            drone_indices,
            shared_success_reward,
        )
        if stochastic:
            transitions[-1]["reward"] = float(team_reward / reward_scale)
        cycle_count += 1
        cached_actions = {}
        cycle_rewards = None
        cycle_mask = None

    drone_goal_found = any(agent_found_goal(core_env, drone_id) for drone_id in drone_ids)
    coverage = float(core_env.current_coverage_ratio())
    success = drone_goal_found and coverage + 1e-6 >= success_min_coverage_ratio
    metrics = {
        "success": float(success),
        "goal_found": float(drone_goal_found),
        "fatal_crash": float(bool(core_env.collided)),
        "drone_crash": float(bool(core_env.drone_crash)),
        "observer_crash": 0.0,
        "coverage": coverage,
        "cycles": float(cycle_count),
    }
    return transitions, metrics


def ppo_update(
    model,
    value_head: OnlineValueHead,
    log_std: nn.Parameter,
    optimizer: torch.optim.Optimizer,
    transitions: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    if not transitions:
        raise RuntimeError("No drone PPO transitions were collected.")
    data = {
        key: torch.stack([transition[key] for transition in transitions]).to(device)
        for key in (
            "task_skill",
            "base_logits",
            "anchor_logits",
            "raw_action",
            "old_log_prob",
            "critic_features",
        )
    }
    advantages = torch.tensor(
        [transition["advantage"] for transition in transitions],
        dtype=torch.float32,
        device=device,
    )
    returns = torch.tensor(
        [transition["return"] for transition in transitions],
        dtype=torch.float32,
        device=device,
    )
    advantages = (advantages - advantages.mean()) / advantages.std(
        unbiased=False
    ).clamp_min(1e-6)
    transition_count = len(transitions)
    accumulator: dict[str, float] = defaultdict(float)
    updates = 0
    task_head = model.action_decoder.task_action_residual_head
    for _ in range(args.ppo_epochs):
        permutation = torch.randperm(transition_count, device=device)
        for start in range(0, transition_count, args.minibatch_size):
            indices = permutation[start : start + args.minibatch_size]
            logits = data["base_logits"][indices] + task_head(
                data["task_skill"][indices]
            )
            distribution = Normal(logits, log_std.exp())
            log_prob = per_drone_squashed_log_prob(
                distribution, data["raw_action"][indices]
            )
            ratio = torch.exp(log_prob - data["old_log_prob"][indices])
            advantage = advantages[indices, None]
            unclipped = ratio * advantage
            clipped = ratio.clamp(
                1.0 - args.clip_ratio, 1.0 + args.clip_ratio
            ) * advantage
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value = value_head(data["critic_features"][indices])
            value_loss = F.mse_loss(value, returns[indices])
            entropy = distribution.entropy().sum(dim=-1).mean()
            anchor_loss = F.mse_loss(
                torch.tanh(logits), torch.tanh(data["anchor_logits"][indices])
            )
            loss = (
                policy_loss
                + args.value_coeff * value_loss
                - args.entropy_coeff * entropy
                + args.anchor_coeff * anchor_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [*task_head.parameters(), *value_head.parameters(), log_std],
                args.max_grad_norm,
            )
            optimizer.step()
            with torch.no_grad():
                log_std.clamp_(args.log_std_min, args.log_std_max)
            accumulator["policy_loss"] += float(policy_loss.detach())
            accumulator["value_loss"] += float(value_loss.detach())
            accumulator["entropy"] += float(entropy.detach())
            accumulator["anchor_loss"] += float(anchor_loss.detach())
            accumulator["approx_kl"] += float(
                (data["old_log_prob"][indices] - log_prob).mean().detach()
            )
            accumulator["clip_fraction"] += float(
                ((ratio - 1.0).abs() > args.clip_ratio).float().mean().detach()
            )
            accumulator["grad_norm"] += float(grad_norm)
            updates += 1
    return {name: value / max(updates, 1) for name, value in accumulator.items()}


def evaluation_score(metrics: dict[str, float]) -> float:
    return metrics["success"] + 0.10 * metrics["goal_found"] - 0.25 * metrics["fatal_crash"]


def evaluate_policy(
    *,
    checkpoint_env_config: dict[str, Any],
    model,
    value_head: OnlineValueHead,
    anchor_head: nn.Module,
    log_std: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    config = build_env_config(
        checkpoint_env_config,
        args.difficulty,
        args.success_min_coverage_ratio,
    )
    metrics = []
    for episode_index in range(args.eval_episodes):
        env = HeMAC_v0.env(**config)
        try:
            _, result = rollout_episode(
                env=env,
                model=model,
                value_head=value_head,
                anchor_head=anchor_head,
                log_std=log_std,
                action_scale=drone_action_scale(config),
                reward_scale=args.reward_scale,
                success_min_coverage_ratio=args.success_min_coverage_ratio,
                seed=args.seed + args.difficulty * 1_000_000 + episode_index,
                device=device,
                stochastic=False,
            )
        finally:
            env.close()
        metrics.append(result)
    return average_metrics(metrics)


def save_checkpoint(
    path: Path,
    model,
    source_payload: dict[str, Any],
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
            "online_value_state_dict": value_head.state_dict(),
            "online_drone_log_std": log_std.detach().cpu(),
            "online_finetuning": {
                "method": "drone_task_residual_parameter_sharing_ppo",
                "difficulty": args.difficulty,
                "success_definition": "drone_goal_found_and_coverage",
                "success_min_coverage_ratio": args.success_min_coverage_ratio,
                "iteration": iteration,
                "joint_env_steps": joint_env_steps,
                "metrics": metrics,
                "hyperparameters": vars(args),
            },
            "online_optimizer_state_dict": optimizer.state_dict(),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def curve_point(
    args: argparse.Namespace,
    metrics: dict[str, float],
    joint_env_steps: int,
    iteration: int,
) -> dict[str, Any]:
    return {
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
        "mean_cycles": metrics["cycles"],
    }


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

    resolved_mappo, checkpoint_env_config = load_checkpoint_env_config(
        args.mappo_checkpoint
    )
    critic_input_dim = (
        model.observation_encoder.output_dim + 2 * model.skill_dim + model.hidden_dim
    )
    value_head = OnlineValueHead(critic_input_dim).to(device)
    log_std = nn.Parameter(
        torch.full(
            (model.agent_count, model.action_dim), args.log_std_init, device=device
        )
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": task_head.parameters(), "lr": args.actor_lr},
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
        f"Drone-only PPO difficulty={args.difficulty}, seed={args.seed}, "
        f"checkpoint={args.hissd_checkpoint}, env_checkpoint={resolved_mappo}"
    )
    print(
        f"trainable drone_residual={sum(p.numel() for p in task_head.parameters()):,}, "
        f"critic={sum(p.numel() for p in value_head.parameters()):,}"
    )

    baseline = evaluate_policy(
        checkpoint_env_config=checkpoint_env_config,
        model=model,
        value_head=value_head,
        anchor_head=anchor_head,
        log_std=log_std,
        args=args,
        device=device,
    )
    best_score = evaluation_score(baseline)
    best_state = {
        "model": copy.deepcopy(model.state_dict()),
        "value": copy.deepcopy(value_head.state_dict()),
        "log_std": log_std.detach().clone(),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
    }
    joint_env_steps = 0
    append_curve_points(
        curve_path,
        [curve_point(args, baseline, joint_env_steps, 0)],
        metadata={
            "evaluation_episodes": args.eval_episodes,
            "source_checkpoint": str(args.hissd_checkpoint.resolve()),
            "environment_checkpoint": str(resolved_mappo),
            "curve_method": args.method_name,
            "success_definition": "drone_goal_found_and_coverage",
            "success_min_coverage_ratio": args.success_min_coverage_ratio,
            "training_step_definition": (
                "sum of world cycles from target training episodes; "
                "evaluation cycles excluded"
            ),
        },
    )
    print(
        f"BASELINE success={baseline['success']:.3f} "
        f"goal={baseline['goal_found']:.3f} crash={baseline['fatal_crash']:.3f} "
        f"coverage={baseline['coverage']:.3f}"
    )
    save_checkpoint(
        args.output_dir / "hissd_drone_online_best.pt",
        model,
        source_payload,
        value_head,
        log_std,
        optimizer,
        iteration=0,
        joint_env_steps=0,
        metrics=baseline,
        args=args,
    )

    config = build_env_config(
        checkpoint_env_config,
        args.difficulty,
        args.success_min_coverage_ratio,
    )
    for iteration in range(1, args.iterations + 1):
        trajectories: list[dict[str, Any]] = []
        episode_metrics = []
        for episode_index in range(args.episodes_per_iteration):
            env = HeMAC_v0.env(**config)
            try:
                episode, metrics = rollout_episode(
                    env=env,
                    model=model,
                    value_head=value_head,
                    anchor_head=anchor_head,
                    log_std=log_std,
                    action_scale=drone_action_scale(config),
                    reward_scale=args.reward_scale,
                    success_min_coverage_ratio=args.success_min_coverage_ratio,
                    seed=args.seed + iteration * 10_000 + episode_index,
                    device=device,
                    stochastic=True,
                )
            finally:
                env.close()
            add_gae(episode, args.gamma, args.gae_lambda)
            trajectories.extend(episode)
            episode_metrics.append(metrics)
            joint_env_steps += int(round(metrics["cycles"]))
        update = ppo_update(
            model, value_head, log_std, optimizer, trajectories, args, device
        )
        train = average_metrics(episode_metrics)
        print(
            f"iter={iteration:03d}/{args.iterations} steps={joint_env_steps} "
            f"success={train['success']:.3f} crash={train['fatal_crash']:.3f} "
            f"policy={update['policy_loss']:.4f} value={update['value_loss']:.4f} "
            f"kl={update['approx_kl']:.5f} clip={update['clip_fraction']:.3f} "
            f"std={float(log_std.mean().detach()):.3f}"
        )
        save_checkpoint(
            args.output_dir / "hissd_drone_online_last.pt",
            model,
            source_payload,
            value_head,
            log_std,
            optimizer,
            iteration=iteration,
            joint_env_steps=joint_env_steps,
            metrics={**train, **update},
            args=args,
        )
        if iteration % args.eval_every != 0:
            continue
        evaluation = evaluate_policy(
            checkpoint_env_config=checkpoint_env_config,
            model=model,
            value_head=value_head,
            anchor_head=anchor_head,
            log_std=log_std,
            args=args,
            device=device,
        )
        score = evaluation_score(evaluation)
        append_curve_points(
            curve_path,
            [curve_point(args, evaluation, joint_env_steps, iteration)],
        )
        print(
            f"EVAL success={evaluation['success']:.3f} "
            f"goal={evaluation['goal_found']:.3f} "
            f"crash={evaluation['fatal_crash']:.3f} "
            f"coverage={evaluation['coverage']:.3f} score={score:.4f}"
        )
        if score > best_score:
            best_score = score
            best_state = {
                "model": copy.deepcopy(model.state_dict()),
                "value": copy.deepcopy(value_head.state_dict()),
                "log_std": log_std.detach().clone(),
                "optimizer": copy.deepcopy(optimizer.state_dict()),
            }
            save_checkpoint(
                args.output_dir / "hissd_drone_online_best.pt",
                model,
                source_payload,
                value_head,
                log_std,
                optimizer,
                iteration=iteration,
                joint_env_steps=joint_env_steps,
                metrics=evaluation,
                args=args,
            )

    model.load_state_dict(best_state["model"])
    value_head.load_state_dict(best_state["value"])
    with torch.no_grad():
        log_std.copy_(best_state["log_std"])
    optimizer.load_state_dict(best_state["optimizer"])
    save_checkpoint(
        args.output_dir / "hissd_drone_online_final.pt",
        model,
        source_payload,
        value_head,
        log_std,
        optimizer,
        iteration=args.iterations,
        joint_env_steps=joint_env_steps,
        metrics={"best_score": best_score},
        args=args,
    )
    print(f"Saved learning curve: {curve_path}")


if __name__ == "__main__":
    main()
