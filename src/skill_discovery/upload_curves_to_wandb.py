from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb


def get_points(payload):
    if isinstance(payload, list):
        return payload

    for key in ("points", "curve", "records", "history"):
        value = payload.get(key)
        if isinstance(value, list):
            return value

    raise ValueError("Could not find learning-curve points in JSON")


def get_step(point):
    # HiSSD may store the intended evaluation grid separately.
    value = point.get("scheduled_joint_env_steps")
    if isinstance(value, (int, float)):
        return int(value)

    for key in (
        "joint_env_steps",
        "environment_steps",
        "env_steps",
        "timesteps_total",
    ):
        value = point.get(key)
        if isinstance(value, (int, float)):
            return int(value)

    raise KeyError(f"No environment-step key in {point.keys()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--project", default="hemac-target-learning")
    parser.add_argument("--method")
    parser.add_argument("--difficulty", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    payload = json.loads(args.file.read_text())
    points = get_points(payload)

    if not points:
        raise ValueError("Learning curve contains no points")

    first = points[0]

    method = args.method or first.get("method", "unknown")
    difficulty = (
        args.difficulty
        if args.difficulty is not None
        else int(first["difficulty"])
    )
    seed = (
        args.seed
        if args.seed is not None
        else int(first["seed"])
    )

    run = wandb.init(
        project=args.project,
        name=f"{method}_D{difficulty}_seed{seed}",
        group=f"D{difficulty}_{method}",
        config={
            "method": method,
            "difficulty": difficulty,
            "seed": seed,
            "curve_file": str(args.file),
        },
    )

    wandb.define_metric("target_env_steps")

    metric_keys = (
        "success_rate",
        "goal_found_rate",
        "fatal_crash_rate",
        "drone_crash_rate",
        "mean_coverage_ratio",
        "mean_cycles",
        "episode_return",
        "validation_score",
    )

    for key in metric_keys:
        wandb.define_metric(key, step_metric="target_env_steps")

    for point in points:
        row = {
            "target_env_steps": get_step(point),
        }

        for key in metric_keys:
            value = point.get(key)
            if isinstance(value, (int, float)):
                row[key] = float(value)

        wandb.log(row)

    run.finish()


if __name__ == "__main__":
    main()
