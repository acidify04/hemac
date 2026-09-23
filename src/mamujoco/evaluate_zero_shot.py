"""Evaluate source and unseen MaMuJoCo tasks without target adaptation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .env import (
    add_environment_version_argument,
    make_env,
    recorded_environment_version,
)
from .happo import _episode_done, resolve_device
from .metrics import EpisodeMetrics
from .offline_models import build_offline_model
from .tasks import AGENT_IDS, SUPPORTED_SUITES, list_tasks
from .difficulty_protocol import add_difficulty_arguments, protocol_from_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("hissd", "skill_vae"), required=True)
    parser.add_argument("--suite", choices=SUPPORTED_SUITES, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--episodes",
        type=int,
        default=8,
        help="Eight episodes for each of four training seeds gives 32 paper-style runs.",
    )
    parser.add_argument("--seed-base", type=int, default=20_000_000)
    parser.add_argument("--max-cycles", type=int, default=1000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    add_environment_version_argument(parser)
    add_difficulty_arguments(parser)
    return parser.parse_args()


def load_model(path: Path, *, algorithm: str, suite: str, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("algorithm") != algorithm:
        raise ValueError(
            f"Checkpoint algorithm is {payload.get('algorithm')!r}, expected {algorithm!r}."
        )
    if payload.get("suite") != suite:
        raise ValueError(
            f"Checkpoint suite is {payload.get('suite')!r}, expected {suite!r}."
        )
    model = build_offline_model(
        algorithm,
        payload["observation_dims"],
        payload["action_dims"],
        payload["state_dim"],
        num_source_tasks=payload.get("model_config", {}).get(
            "num_source_tasks", 5
        ),
        history_length=payload.get("model_config", {}).get("history_length", 32),
        training_sequence_length=payload.get("model_config", {}).get(
            "training_sequence_length",
            payload.get("config", {}).get("sequence_length", 1),
        ),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload


@torch.no_grad()
def evaluate_episode(env, model, *, seed: int, device, render: bool):
    observations, _ = env.reset(seed=seed)
    hidden = model.initial_inference_state(1, device=device)
    metrics = EpisodeMetrics()
    if hasattr(env, "x_position"):
        metrics.begin(env.x_position)
    disabled_agent = getattr(getattr(env, "task", None), "disabled_agent", None)
    active_agents = torch.tensor(
        [[agent != disabled_agent for agent in AGENT_IDS]],
        dtype=torch.bool,
        device=device,
    )
    while observations:
        tensor_obs = {
            agent: torch.as_tensor(value, dtype=torch.float32, device=device).reshape(1, -1)
            for agent, value in observations.items()
        }
        actions, hidden = model.act_step(
            tensor_obs, hidden, active_agents=active_agents
        )
        numpy_actions = {
            agent: actions[agent].squeeze(0).cpu().numpy().astype(np.float32)
            for agent in AGENT_IDS
        }
        observations, rewards, terminations, truncations, infos = env.step(
            numpy_actions
        )
        metrics.update(rewards, infos)
        if render:
            env.render()
        if _episode_done(terminations, truncations):
            break
    return metrics.finalize()


def summarize(records):
    keys = (
        "episode_return",
        "forward_distance",
        "forward_reward",
        "forward_velocity",
        "control_cost",
        "episode_length",
    )
    return {
        key: {
            "mean": float(np.mean([record[key] for record in records])),
            "std": float(np.std([record[key] for record in records], ddof=1))
            if len(records) > 1
            else 0.0,
        }
        for key in keys
    }


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    device = resolve_device(args.device)
    model, checkpoint = load_model(
        args.checkpoint,
        algorithm=args.algorithm,
        suite=args.suite,
        device=device,
    )
    checkpoint_version = recorded_environment_version(checkpoint)
    if checkpoint_version != args.environment_version:
        raise ValueError(
            f"Checkpoint uses {checkpoint_version!r}, expected "
            f"{args.environment_version!r}"
        )
    results = {
        "algorithm": args.algorithm,
        "suite": args.suite,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": checkpoint.get("step"),
        "primary_metric": "episode_return",
        "zero_shot": True,
        "target_gradient_updates": 0,
        "environment_version": args.environment_version,
        "episodes_per_task": args.episodes,
        "source": {},
        "target": {},
    }
    if args.suite == "difficulty":
        protocol = protocol_from_args(args)
        all_tasks = protocol.tasks
        results["difficulty_protocol"] = protocol.to_dict()
    else:
        all_tasks = list_tasks(args.suite)
    for task_index, task in enumerate(all_tasks):
        env = make_env(
            task,
            environment_version=args.environment_version,
            render_mode="human" if args.render else None,
            max_cycles=args.max_cycles,
        )
        try:
            records = [
                evaluate_episode(
                    env,
                    model,
                    seed=args.seed_base + task_index * 1_000_000 + episode,
                    device=device,
                    render=args.render,
                )
                for episode in range(args.episodes)
            ]
        finally:
            env.close()
        summary = summarize(records)
        results[task.split][task.name] = {
            "task": task.to_dict(),
            "metrics": summary,
            "episodes": records,
            "episode_seeds": [
                args.seed_base + task_index * 1_000_000 + episode
                for episode in range(args.episodes)
            ],
        }
        print(
            f"[{task.split}] {task.name}: "
            f"return={summary['episode_return']['mean']:.2f}±"
            f"{summary['episode_return']['std']:.2f}, "
            f"distance={summary['forward_distance']['mean']:.2f}"
        )
    output = args.output or args.checkpoint.parent / "zero_shot_evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, sort_keys=True)
    print(f"Saved evaluation: {output}")


if __name__ == "__main__":
    main()
