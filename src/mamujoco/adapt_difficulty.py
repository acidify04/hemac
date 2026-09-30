"""Independently adapt HiSSD task context and residual actors to D3 or D4."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.distributions import Normal
from torch.utils.data import default_collate

from .dataset import MultiTaskTrajectoryDataset
from .difficulty_protocol import (
    add_difficulty_arguments,
    adaptation_replay_ids,
    adaptation_evaluation_steps,
    protocol_from_args,
    reward_auc,
    validate_adaptation_counts,
)
from .env import (
    add_environment_version_argument,
    make_env,
    recorded_environment_version,
)
from .evaluate_zero_shot import load_model
from .happo import _episode_done, resolve_device, seed_everything
from .metrics import EpisodeMetrics
from .models import CentralCritic
from .offline_models import supervised_contrastive_loss
from .tasks import AGENT_IDS


DEFAULT_DATA_ROOT = Path("src/mamujoco/offline_data")
DEFAULT_OUTPUT_ROOT = Path("src/mamujoco/checkpoints/adaptation")


class DifficultyResidualActors(nn.Module):
    """Agent-specific zero-initialized residuals conditioned on inferred context."""

    def __init__(self, observation_dims, action_dims, skill_dim: int, hidden_dim: int):
        super().__init__()
        self.heads = nn.ModuleDict()
        for agent in AGENT_IDS:
            head = nn.Sequential(
                nn.Linear(observation_dims[agent] + skill_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dims[agent]),
            )
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
            self.heads[agent] = head
        self.log_std = nn.ParameterDict(
            {
                agent: nn.Parameter(torch.full((action_dims[agent],), -1.0))
                for agent in AGENT_IDS
            }
        )

    def residuals(self, observations, task_skills):
        return {
            agent: self.heads[agent](
                torch.cat((observations[agent], task_skills[:, :, index]), dim=-1)
            )
            for index, agent in enumerate(AGENT_IDS)
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_DATA_ROOT / "difficulty/manifest.json")
    parser.add_argument("--target-difficulty", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--adaptation-budget", type=int, default=500_000)
    parser.add_argument("--eval-interval", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--target-replay-ratio", type=float, default=0.5)
    parser.add_argument("--source-replay-ratios", nargs=2, type=float, default=(0.25, 0.25))
    parser.add_argument("--history-length", type=int, default=32)
    parser.add_argument("--max-cycles", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.001)
    parser.add_argument("--contrastive-weight", type=float, default=2.0)
    parser.add_argument("--source-anchor-weight", type=float, default=1.0)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--gradient-norm-clip", type=float, default=10.0)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    add_environment_version_argument(parser)
    add_difficulty_arguments(parser)
    return parser.parse_args()


def _squashed_log_prob(distribution: Normal, raw_action: torch.Tensor) -> torch.Tensor:
    correction = 2.0 * (
        math.log(2.0) - raw_action - functional.softplus(-2.0 * raw_action)
    )
    return (distribution.log_prob(raw_action) - correction).sum(dim=-1)


def _atanh(action: torch.Tensor) -> torch.Tensor:
    action = action.clamp(-0.999999, 0.999999)
    return 0.5 * (torch.log1p(action) - torch.log1p(-action))


def _task_outputs(model, batch):
    return model.task_encoder(
        batch["history_observations"],
        batch["history_valid"],
        batch["active_agents"],
    )


def _pooled_query(query: torch.Tensor, active_agents: torch.Tensor) -> torch.Tensor:
    numeric = active_agents.to(query.dtype)
    return (
        query.mean(dim=1) * numeric.unsqueeze(-1)
    ).sum(dim=1) / numeric.sum(dim=1, keepdim=True).clamp_min(1.0)


def _to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def _sample_source(dataset, count: int, task_id: int, rng: random.Random, device):
    items = [dataset[rng.randrange(len(dataset))] for _ in range(count)]
    batch = default_collate(items)
    batch["task_id"] = torch.full((count,), task_id, dtype=torch.long)
    return _to_device(batch, device)


def _finalize_gae(transitions: list[dict], gamma: float, gae_lambda: float) -> None:
    advantage = 0.0
    next_value = 0.0
    for transition in reversed(transitions):
        continuation = 0.0 if transition["done"] else 1.0
        delta = transition["reward"] + gamma * continuation * next_value - transition["value"]
        advantage = delta + gamma * gae_lambda * continuation * advantage
        transition["advantage"] = torch.tensor(advantage, dtype=torch.float32)
        transition["return"] = torch.tensor(advantage + transition["value"], dtype=torch.float32)
        next_value = transition["value"]


@torch.no_grad()
def _rollout_episode(
    env, model, residual, critic, *, seed: int, max_steps: int, device
) -> list[dict]:
    observations, _ = env.reset(seed=seed)
    hidden = model.initial_inference_state(1, device=device)
    transitions = []
    while observations:
        tensor_obs = {
            agent: torch.as_tensor(value, dtype=torch.float32, device=device).reshape(1, -1)
            for agent, value in observations.items()
        }
        state = torch.as_tensor(env.state(), dtype=torch.float32, device=device).reshape(1, -1)
        base_actions, next_hidden = model.act_step(tensor_obs, hidden)
        task_skill = next_hidden["last_task_skill"]
        residual_input = {agent: value.unsqueeze(1) for agent, value in tensor_obs.items()}
        residual_values = residual.residuals(residual_input, task_skill.unsqueeze(1))
        raw_actions = {}
        old_log_probs = {}
        numpy_actions = {}
        base_raw = {}
        for agent in AGENT_IDS:
            base_raw[agent] = _atanh(base_actions[agent])
            mean = base_raw[agent] + residual_values[agent][:, 0]
            distribution = Normal(mean, residual.log_std[agent].exp().expand_as(mean))
            raw = distribution.sample()
            action = torch.tanh(raw)
            raw_actions[agent] = raw
            old_log_probs[agent] = _squashed_log_prob(distribution, raw)
            numpy_actions[agent] = action.squeeze(0).cpu().numpy().astype(np.float32)
        value = float(critic(state).item())
        next_observations, rewards, terminations, truncations, _ = env.step(numpy_actions)
        done = _episode_done(terminations, truncations)
        try:
            next_state = torch.as_tensor(env.state(), dtype=torch.float32).reshape(1, -1)
        except Exception:
            next_state = state.cpu()
        transitions.append(
            {
                "observations": {
                    agent: tensor_obs[agent].cpu() for agent in AGENT_IDS
                },
                "history_observations": {
                    agent: next_hidden["task_observation_history"][agent].cpu()
                    for agent in AGENT_IDS
                },
                "history_valid": next_hidden["task_history_valid"].cpu(),
                "actions": {
                    agent: torch.tanh(raw_actions[agent]).cpu() for agent in AGENT_IDS
                },
                "raw_actions": {agent: raw_actions[agent].cpu() for agent in AGENT_IDS},
                "old_log_probs": {
                    agent: old_log_probs[agent].cpu() for agent in AGENT_IDS
                },
                "base_raw": {agent: base_raw[agent].cpu() for agent in AGENT_IDS},
                "states": state.cpu(),
                "next_states": next_state,
                "rewards": torch.tensor([float(rewards[AGENT_IDS[0]])]),
                "terminations": torch.tensor([any(terminations.values())]),
                "valid": torch.tensor([True]),
                "active_agents": torch.ones(len(AGENT_IDS), dtype=torch.bool),
                "task_id": torch.tensor(2, dtype=torch.long),
                "reward": float(rewards[AGENT_IDS[0]]),
                "value": value,
                "done": done,
            }
        )
        hidden = next_hidden
        observations = next_observations
        if done:
            break
        if len(transitions) >= max_steps:
            break
    return transitions


def _adaptation_update(
    model,
    teacher_task_encoder,
    residual,
    critic,
    optimizer,
    target_batch,
    d1_batch,
    d2_batch,
    args,
):
    target_task, target_query = _task_outputs(model, target_batch)
    target_residual = residual.residuals(target_batch["observations"], target_task)
    advantages = target_batch["advantage"]
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
    actor_losses = []
    entropies = []
    for agent in AGENT_IDS:
        mean = target_batch["base_raw"][agent] + target_residual[agent]
        distribution = Normal(mean, residual.log_std[agent].exp().expand_as(mean))
        log_prob = _squashed_log_prob(distribution, target_batch["raw_actions"][agent])
        ratio = torch.exp(log_prob - target_batch["old_log_probs"][agent])
        objective = ratio * advantages.unsqueeze(1)
        clipped = ratio.clamp(1.0 - args.clip_ratio, 1.0 + args.clip_ratio) * advantages.unsqueeze(1)
        actor_losses.append(-torch.minimum(objective, clipped).mean())
        entropies.append(distribution.entropy().sum(dim=-1).mean())
    actor_loss = torch.stack(actor_losses).mean()
    entropy = torch.stack(entropies).mean()
    value = critic(target_batch["states"]).squeeze(-1)
    value_loss = (value - target_batch["return"].unsqueeze(1)).pow(2).mean()

    source_outputs = []
    source_anchor_losses = []
    source_residual_losses = []
    for source_batch in (d1_batch, d2_batch):
        source_task, source_query = _task_outputs(model, source_batch)
        with torch.no_grad():
            teacher_task, teacher_query = teacher_task_encoder(
                source_batch["history_observations"],
                source_batch["history_valid"],
                source_batch["active_agents"],
            )
        source_anchor_losses.append(
            0.5 * ((source_task - teacher_task).pow(2).mean() + (source_query - teacher_query).pow(2).mean())
        )
        source_residual = residual.residuals(source_batch["observations"], source_task)
        source_residual_losses.append(
            torch.stack([value.pow(2).mean() for value in source_residual.values()]).mean()
        )
        source_outputs.append((source_query, source_batch))
    contexts = [_pooled_query(target_query, target_batch["active_agents"])]
    labels = [target_batch["task_id"]]
    for source_query, source_batch in source_outputs:
        contexts.append(_pooled_query(source_query, source_batch["active_agents"]))
        labels.append(source_batch["task_id"])
    contrastive_loss = supervised_contrastive_loss(
        torch.cat(contexts),
        torch.cat(labels),
        args.contrastive_temperature,
    )
    source_anchor = torch.stack(source_anchor_losses).mean()
    source_residual = torch.stack(source_residual_losses).mean()
    loss = (
        actor_loss
        + args.value_coefficient * value_loss
        - args.entropy_coefficient * entropy
        + args.contrastive_weight * contrastive_loss
        + args.source_anchor_weight * (source_anchor + source_residual)
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    torch.nn.utils.clip_grad_norm_(parameters, args.gradient_norm_clip)
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "actor_loss": float(actor_loss.detach()),
        "value_loss": float(value_loss.detach()),
        "contrastive_loss": float(contrastive_loss.detach()),
        "source_anchor_loss": float(source_anchor.detach()),
    }


@torch.no_grad()
def _evaluate(env, model, residual, *, episodes: int, seed_base: int, device):
    records = []
    model.eval()
    residual.eval()
    for episode in range(episodes):
        observations, _ = env.reset(seed=seed_base + episode)
        hidden = model.initial_inference_state(1, device=device)
        metrics = EpisodeMetrics()
        metrics.begin(env.x_position)
        while observations:
            tensor_obs = {
                agent: torch.as_tensor(value, dtype=torch.float32, device=device).reshape(1, -1)
                for agent, value in observations.items()
            }
            base_actions, hidden = model.act_step(tensor_obs, hidden)
            task_skill = hidden["last_task_skill"].unsqueeze(1)
            residual_values = residual.residuals(
                {agent: value.unsqueeze(1) for agent, value in tensor_obs.items()},
                task_skill,
            )
            actions = {
                agent: torch.tanh(
                    _atanh(base_actions[agent]) + residual_values[agent][:, 0]
                ).squeeze(0).cpu().numpy().astype(np.float32)
                for agent in AGENT_IDS
            }
            observations, rewards, terminations, truncations, infos = env.step(actions)
            metrics.update(rewards, infos)
            if _episode_done(terminations, truncations):
                break
        records.append(metrics.finalize())
    model.task_encoder.train()
    residual.train()
    summary = {}
    for name in ("episode_return", "forward_velocity", "control_cost"):
        values = [float(record[name]) for record in records]
        summary[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
    return summary, records


def _validate_args(args, protocol):
    steps = adaptation_evaluation_steps(args.adaptation_budget, args.eval_interval)
    if args.target_difficulty not in protocol.target_ids:
        raise ValueError("Target adaptation may only use a configured target difficulty")
    ratios = (args.target_replay_ratio, *args.source_replay_ratios)
    if not math.isclose(sum(ratios), 1.0, abs_tol=1e-9):
        raise ValueError("target/source replay ratios must sum to one")
    counts = tuple(int(round(args.batch_size * ratio)) for ratio in ratios)
    validate_adaptation_counts(*counts, args.batch_size)
    if counts != (64, 32, 32) and args.batch_size == 128 and ratios == (0.5, 0.25, 0.25):
        raise AssertionError("Default adaptation batch must be 64/32/32")
    return steps, counts


def main() -> None:
    args = parse_args()
    protocol = protocol_from_args(args)
    evaluation_steps, counts = _validate_args(args, protocol)
    target_id, source_ids = adaptation_replay_ids(
        protocol, args.target_difficulty
    )
    target_count, d1_count, d2_count = counts
    seed_everything(args.seed)
    rng = random.Random(args.seed)
    device = resolve_device(args.device)
    model, source_checkpoint = load_model(
        args.checkpoint, algorithm="hissd", suite="difficulty", device=device
    )
    checkpoint_version = recorded_environment_version(source_checkpoint)
    if checkpoint_version != args.environment_version:
        raise ValueError(
            f"Checkpoint uses {checkpoint_version!r}, expected "
            f"{args.environment_version!r}"
        )
    if model.history_length != args.history_length:
        raise ValueError(
            f"Checkpoint history length is {model.history_length}, requested {args.history_length}"
        )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.task_encoder.parameters():
        parameter.requires_grad_(True)
    teacher_task_encoder = copy.deepcopy(model.task_encoder).eval()
    for parameter in teacher_task_encoder.parameters():
        parameter.requires_grad_(False)
    residual = DifficultyResidualActors(
        model.observation_dims,
        model.action_dims,
        model.skill_dim,
        model.hidden_dim,
    ).to(device)
    critic = CentralCritic(model.state_dim, hidden_dim=model.hidden_dim).to(device)
    optimizer = torch.optim.Adam(
        [
            *model.task_encoder.parameters(),
            *residual.parameters(),
            *critic.parameters(),
        ],
        lr=args.learning_rate,
    )
    d1_dataset = MultiTaskTrajectoryDataset(
        args.manifest,
        sequence_length=1,
        history_length=args.history_length,
        allowed_tasks=(source_ids[0],),
    )
    d2_dataset = MultiTaskTrajectoryDataset(
        args.manifest,
        sequence_length=1,
        history_length=args.history_length,
        allowed_tasks=(source_ids[1],),
    )
    if args.target_difficulty in {
        task_name
        for dataset in (d1_dataset, d2_dataset)
        for task_name in dataset.task_to_id
    }:
        raise ValueError("Target difficulty leaked into source replay")
    target_task = protocol.task(target_id)
    env = make_env(
        target_task,
        environment_version=args.environment_version,
        max_cycles=args.max_cycles,
    )
    output_dir = (
        args.output_root
        / args.target_difficulty
        / f"seed_{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pending: deque[dict] = deque()
    curve = []
    raw_evaluations = {}
    update_count = 0
    adaptation_step = 0
    try:
        for evaluation_step in evaluation_steps:
            while adaptation_step < evaluation_step:
                remaining = evaluation_step - adaptation_step
                episode = _rollout_episode(
                    env,
                    model,
                    residual,
                    critic,
                    seed=args.seed * 10_000_000 + adaptation_step,
                    max_steps=remaining,
                    device=device,
                )
                _finalize_gae(episode, args.gamma, args.gae_lambda)
                pending.extend(episode)
                adaptation_step += len(episode)
                while len(pending) >= target_count:
                    target_items = [pending.popleft() for _ in range(target_count)]
                    target_batch = _to_device(default_collate(target_items), device)
                    target_batch["advantage"] = target_batch["advantage"].float()
                    target_batch["return"] = target_batch["return"].float()
                    d1_batch = _sample_source(d1_dataset, d1_count, 0, rng, device)
                    d2_batch = _sample_source(d2_dataset, d2_count, 1, rng, device)
                    for _ in range(args.ppo_epochs):
                        _adaptation_update(
                            model,
                            teacher_task_encoder,
                            residual,
                            critic,
                            optimizer,
                            target_batch,
                            d1_batch,
                            d2_batch,
                            args,
                        )
                    update_count += 1
            summary, records = _evaluate(
                env,
                model,
                residual,
                episodes=args.eval_episodes,
                seed_base=(
                    args.seed * 100_000_000
                    + int(args.target_difficulty.removeprefix("D")) * 1_000_000
                    + evaluation_step
                ),
                device=device,
            )
            row = {
                "method": "hissd_based_difficulty_generalization",
                "seed": args.seed,
                "target_difficulty": args.target_difficulty,
                "actuator_strength": target_task.actuator_scale,
                "adaptation_step": evaluation_step,
                "episode_return_mean": summary["episode_return"]["mean"],
                "episode_return_std": summary["episode_return"]["std"],
                "forward_velocity_mean": summary["forward_velocity"]["mean"],
                "forward_velocity_std": summary["forward_velocity"]["std"],
                "control_cost_mean": summary["control_cost"]["mean"],
                "control_cost_std": summary["control_cost"]["std"],
                "num_eval_episodes": args.eval_episodes,
                "target_gradient_updates": update_count,
            }
            curve.append(row)
            raw_evaluations[str(evaluation_step)] = records
            print(
                f"target={args.target_difficulty} seed={args.seed} "
                f"step={evaluation_step} return={row['episode_return_mean']:.2f}"
            )
    finally:
        env.close()
    raw_auc, normalized_auc = reward_auc(
        [row["adaptation_step"] for row in curve],
        [row["episode_return_mean"] for row in curve],
    )
    summary = {
        "method": "hissd_based_difficulty_generalization",
        "seed": args.seed,
        "target_difficulty": args.target_difficulty,
        "actuator_strength": target_task.actuator_scale,
        "adaptation_budget": args.adaptation_budget,
        "zero_shot_return": curve[0]["episode_return_mean"],
        "final_500k_return": curve[-1]["episode_return_mean"],
        "raw_reward_auc": raw_auc,
        "normalized_reward_auc": normalized_auc,
        "evaluation_points": len(curve),
        "target_gradient_updates_at_step_zero": curve[0]["target_gradient_updates"],
    }
    payload = {
        "summary": summary,
        "curve": curve,
        "raw_episodes": raw_evaluations,
        "difficulty_protocol": protocol.to_dict(),
        "environment_version": args.environment_version,
        "adaptation_batch_counts": {
            args.target_difficulty: target_count,
            protocol.source_ids[0]: d1_count,
            protocol.source_ids[1]: d2_count,
        },
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_step": source_checkpoint.get("step"),
        "cross_target_data_used": [],
        "config": vars(args),
    }
    with (output_dir / "adaptation_results.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True, default=str)
    with (output_dir / "adaptation_curve.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(curve)
    torch.save(
        {
            "model": model.state_dict(),
            "residual_actors": residual.state_dict(),
            "critic": critic.state_dict(),
            "summary": summary,
            "config": vars(args),
        },
        output_dir / "adapted_final.pt",
    )
    print(f"Saved adaptation results: {output_dir}")


if __name__ == "__main__":
    main()
