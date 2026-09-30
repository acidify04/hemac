"""Collect the paper-scale 100 HAPPO trajectories per source task."""

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
from .models import load_happo_checkpoint
from .tasks import SUPPORTED_SUITES, get_task, list_tasks, task_names


DEFAULT_CHECKPOINT_ROOT = Path("src/mamujoco/checkpoints/happo")
DEFAULT_OUTPUT_ROOT = Path("src/mamujoco/offline_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUPPORTED_SUITES, required=True)
    parser.add_argument(
        "--task",
        default="all",
        help="One source task or 'all' for every source task.",
    )
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--checkpoint-seed", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--trajectories-per-task", type=int, default=100)
    parser.add_argument("--max-cycles", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=100_000)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    add_environment_version_argument(parser)
    return parser.parse_args()


@torch.no_grad()
def collect_trajectory(env, actors, *, seed: int, deterministic: bool, device):
    observations, _ = env.reset(seed=seed)
    agent_ids = actors.agent_ids
    obs_rows = {agent: [] for agent in agent_ids}
    next_obs_rows = {agent: [] for agent in agent_ids}
    action_rows = {agent: [] for agent in agent_ids}
    state_rows = []
    next_state_rows = []
    reward_rows = []
    termination_rows = []
    truncation_rows = []
    metrics = EpisodeMetrics()
    if hasattr(env, "x_position"):
        metrics.begin(env.x_position)

    while observations:
        tensor_obs = {
            agent: torch.as_tensor(value, dtype=torch.float32, device=device).reshape(1, -1)
            for agent, value in observations.items()
        }
        state = np.asarray(env.state(), dtype=np.float32).copy()
        actions, _, _ = actors.act(tensor_obs, deterministic=deterministic)
        numpy_actions = {
            agent: actions[agent].squeeze(0).cpu().numpy().astype(np.float32)
            for agent in agent_ids
        }
        next_observations, rewards, terminations, truncations, infos = env.step(
            numpy_actions
        )
        try:
            next_state = np.asarray(env.state(), dtype=np.float32).copy()
        except Exception:
            next_state = state.copy()
        done = _episode_done(terminations, truncations)
        for agent in agent_ids:
            current = np.asarray(observations[agent], dtype=np.float32)
            following = np.asarray(
                next_observations.get(agent, np.zeros_like(current)), dtype=np.float32
            )
            obs_rows[agent].append(torch.from_numpy(current.copy()))
            next_obs_rows[agent].append(torch.from_numpy(following.copy()))
            action_rows[agent].append(
                torch.from_numpy(env.last_executed_actions[agent].copy())
            )
        state_rows.append(torch.from_numpy(state))
        next_state_rows.append(torch.from_numpy(next_state))
        reward_rows.append(float(rewards[agent_ids[0]]))
        termination_rows.append(any(terminations.values()))
        truncation_rows.append(any(truncations.values()))
        metrics.update(rewards, infos)
        observations = next_observations
        if done:
            break

    return {
        "observations": {a: torch.stack(rows) for a, rows in obs_rows.items()},
        "next_observations": {
            a: torch.stack(rows) for a, rows in next_obs_rows.items()
        },
        "actions": {a: torch.stack(rows) for a, rows in action_rows.items()},
        "states": torch.stack(state_rows),
        "next_states": torch.stack(next_state_rows),
        "rewards": torch.tensor(reward_rows, dtype=torch.float32),
        "terminations": torch.tensor(termination_rows, dtype=torch.bool),
        "truncations": torch.tensor(truncation_rows, dtype=torch.bool),
        "metrics": metrics.finalize(),
    }


def checkpoint_path(
    root: Path, environment_version: str, suite: str, task: str, seed: int
) -> Path:
    return root / environment_version / suite / task / f"seed_{seed}" / "best.pt"


def main() -> None:
    args = parse_args()
    if args.trajectories_per_task <= 0:
        raise ValueError("--trajectories-per-task must be positive")
    if args.task == "all":
        tasks = list_tasks(args.suite, "source")
    else:
        if args.task not in task_names(args.suite, "source"):
            raise ValueError(f"Unknown source task: {args.task}")
        tasks = (get_task(args.suite, args.task),)
    device = resolve_device(args.device)
    manifest_path = args.output_root / args.suite / "manifest.json"
    manifest = {
        "format_version": 1,
        "suite": args.suite,
        "environment_version": args.environment_version,
        "backend": f"gymnasium_robotics.mamujoco_v1/{args.environment_version}",
        "agent_conf": "6x1",
        "agent_obsk": 1,
        "trajectories_per_source_task": args.trajectories_per_task,
        "tasks": {},
    }
    if args.task != "all" and manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as file:
            existing = json.load(file)
        if existing.get("suite") != args.suite:
            raise ValueError(f"Existing manifest has a different suite: {manifest_path}")
        if existing.get("environment_version") != args.environment_version:
            raise ValueError(
                f"Existing manifest has a different environment: {manifest_path}"
            )
        manifest["tasks"].update(existing.get("tasks", {}))
    for task_index, task in enumerate(tasks):
        path = checkpoint_path(
            args.checkpoint_root,
            args.environment_version,
            args.suite,
            task.name,
            args.checkpoint_seed,
        )
        actors, _, checkpoint = load_happo_checkpoint(path, device=device)
        checkpoint_version = recorded_environment_version(
            checkpoint.get("metadata", {})
        )
        if checkpoint_version != args.environment_version:
            raise ValueError(
                f"Checkpoint {path} uses {checkpoint_version!r}, "
                f"expected {args.environment_version!r}"
            )
        env = make_env(
            task,
            environment_version=args.environment_version,
            max_cycles=args.max_cycles,
        )
        task_dir = args.output_root / args.suite / task.name
        task_dir.mkdir(parents=True, exist_ok=True)
        entries = []
        try:
            for episode_index in range(args.trajectories_per_task):
                seed = args.seed + task_index * 1_000_000 + episode_index
                trajectory = collect_trajectory(
                    env,
                    actors,
                    seed=seed,
                    deterministic=not args.stochastic,
                    device=device,
                )
                trajectory["metadata"] = {
                    "suite": args.suite,
                    "task": task.to_dict(),
                    "seed": seed,
                    "behavior_checkpoint": str(path.resolve()),
                    "behavior_checkpoint_metadata": checkpoint.get("metadata", {}),
                    "agent_conf": "6x1",
                    "agent_obsk": 1,
                    "environment_version": args.environment_version,
                }
                episode_path = task_dir / f"episode_{episode_index:04d}.pt"
                torch.save(trajectory, episode_path)
                entries.append(
                    {
                        "path": str(episode_path.relative_to(args.output_root)),
                        "seed": seed,
                        "transitions": int(trajectory["rewards"].shape[0]),
                        **trajectory["metrics"],
                    }
                )
                print(
                    f"[{task.name}] {episode_index + 1}/{args.trajectories_per_task} "
                    f"return={trajectory['metrics']['episode_return']:.2f}"
                )
        finally:
            env.close()
        manifest["tasks"][task.name] = {
            "spec": task.to_dict(),
            "episodes": entries,
        }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
