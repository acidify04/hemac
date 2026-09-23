"""Collect 100 High/Mid/Low trajectories for each source difficulty."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .collect_offline_data import collect_trajectory
from .difficulty_protocol import QUALITY_FRACTIONS, add_difficulty_arguments, protocol_from_args
from .env import (
    add_environment_version_argument,
    make_env,
    recorded_environment_version,
)
from .happo import resolve_device
from .models import load_happo_checkpoint


DEFAULT_CHECKPOINT_ROOT = Path("src/mamujoco/checkpoints/happo")
DEFAULT_OUTPUT_ROOT = Path("src/mamujoco/offline_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--checkpoint-seed", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--episodes-per-quality", type=int, default=100)
    parser.add_argument("--max-cycles", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=100_000)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    add_environment_version_argument(parser)
    add_difficulty_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes_per_quality <= 0:
        raise ValueError("--episodes-per-quality must be positive")
    protocol = protocol_from_args(args)
    selection_path = args.selection or (
        args.checkpoint_root
        / args.environment_version
        / "difficulty"
        / f"quality_selection_seed_{args.checkpoint_seed}.json"
    )
    with selection_path.open("r", encoding="utf-8") as file:
        selection = json.load(file)
    if selection.get("difficulty_protocol") != protocol.to_dict():
        raise ValueError("Quality selection and requested difficulty protocol differ")
    if selection.get("environment_version") != args.environment_version:
        raise ValueError("Quality selection and requested environment version differ")
    device = resolve_device(args.device)
    manifest = {
        "format_version": 1,
        "suite": "difficulty",
        "environment_version": args.environment_version,
        "backend": f"gymnasium_robotics.mamujoco_v1/{args.environment_version}",
        "agent_conf": "6x1",
        "agent_obsk": 1,
        "difficulty_protocol": protocol.to_dict(),
        "episodes_per_quality": args.episodes_per_quality,
        "qualities": list(QUALITY_FRACTIONS),
        "tasks": {},
    }
    for task_index, difficulty in enumerate(protocol.source_ids):
        task = protocol.task(difficulty)
        task_entries = []
        for quality_index, quality in enumerate(QUALITY_FRACTIONS):
            selected = selection["qualities"][difficulty][quality]
            checkpoint_path = Path(selected["checkpoint_path"])
            actors, _, checkpoint = load_happo_checkpoint(
                checkpoint_path, device=device
            )
            checkpoint_version = recorded_environment_version(
                checkpoint.get("metadata", {})
            )
            if checkpoint_version != args.environment_version:
                raise ValueError(
                    f"Checkpoint {checkpoint_path} uses {checkpoint_version!r}, "
                    f"expected {args.environment_version!r}"
                )
            env = make_env(
                task,
                environment_version=args.environment_version,
                max_cycles=args.max_cycles,
            )
            quality_dir = args.output_root / "difficulty" / difficulty / quality
            quality_dir.mkdir(parents=True, exist_ok=True)
            try:
                for episode_index in range(args.episodes_per_quality):
                    seed = (
                        args.seed
                        + task_index * 1_000_000
                        + quality_index * 100_000
                        + episode_index
                    )
                    trajectory = collect_trajectory(
                        env,
                        actors,
                        seed=seed,
                        deterministic=not args.stochastic,
                        device=device,
                    )
                    trajectory["metadata"] = {
                        "suite": "difficulty",
                        "task": task.to_dict(),
                        "difficulty": difficulty,
                        "quality": quality,
                        "seed": seed,
                        "behavior_checkpoint": str(checkpoint_path.resolve()),
                        "behavior_checkpoint_metadata": checkpoint.get("metadata", {}),
                        "quality_selection": selected,
                        "agent_conf": "6x1",
                        "agent_obsk": 1,
                        "environment_version": args.environment_version,
                    }
                    episode_path = quality_dir / f"episode_{episode_index:04d}.pt"
                    torch.save(trajectory, episode_path)
                    task_entries.append(
                        {
                            "path": str(episode_path.relative_to(args.output_root)),
                            "seed": seed,
                            "quality": quality,
                            "transitions": int(trajectory["rewards"].shape[0]),
                            "checkpoint_step": selected["checkpoint_step"],
                            "checkpoint_path": str(checkpoint_path.resolve()),
                            "evaluation_return": selected["evaluation_return"],
                            "target_fraction": selected["target_fraction"],
                            **trajectory["metrics"],
                        }
                    )
                    print(
                        f"[{difficulty}/{quality}] "
                        f"{episode_index + 1}/{args.episodes_per_quality} "
                        f"return={trajectory['metrics']['episode_return']:.2f}"
                    )
            finally:
                env.close()
        manifest["tasks"][difficulty] = {
            "spec": task.to_dict(),
            "episodes": task_entries,
        }
    manifest_path = args.output_root / "difficulty" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
    print(f"Saved source-only difficulty manifest: {manifest_path}")


if __name__ == "__main__":
    main()
