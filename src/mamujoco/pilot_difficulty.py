"""Run the full 4x4 HAPPO difficulty cross-evaluation pilot."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path

import numpy as np

from .difficulty_protocol import (
    DEFAULT_EXPERIMENT_SEEDS,
    add_difficulty_arguments,
    protocol_from_args,
)
from .env import add_environment_version_argument, make_env
from .happo import collect_episode, resolve_device
from .models import load_happo_checkpoint


DEFAULT_CHECKPOINT_ROOT = Path("src/mamujoco/checkpoints/pilot_happo")
DEFAULT_OUTPUT_ROOT = Path("src/mamujoco/outputs/difficulty/pilot_cross")
METRICS = ("episode_return", "forward_velocity", "control_cost")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--checkpoint-name", default="best.pt")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_EXPERIMENT_SEEDS)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=30_000_000)
    parser.add_argument("--max-cycles", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    add_environment_version_argument(parser)
    add_difficulty_arguments(parser)
    return parser.parse_args()


def cross_evaluation_pairs(
    difficulty_ids: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    """Return every train/evaluation pair once in row-major matrix order."""
    return tuple(product(difficulty_ids, repeat=2))


def _metric_summary(records: list[dict]) -> dict[str, float]:
    result = {}
    for metric in METRICS:
        values = [float(record[metric]) for record in records]
        result[f"{metric}_mean"] = float(np.mean(values))
        result[f"{metric}_std"] = (
            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        )
    return result


def build_metric_matrix(
    records: list[dict],
    difficulty_ids: tuple[str, ...],
    metric: str,
) -> dict[str, object]:
    """Build a labeled matrix with training rows and evaluation columns."""
    values = []
    for training_difficulty in difficulty_ids:
        row = []
        for evaluation_difficulty in difficulty_ids:
            matching = [
                float(record[f"{metric}_mean"])
                for record in records
                if record["training_difficulty"] == training_difficulty
                and record["evaluation_difficulty"] == evaluation_difficulty
            ]
            if not matching:
                raise ValueError(
                    "Missing cross-evaluation pair: "
                    f"{training_difficulty}->{evaluation_difficulty}"
                )
            row.append(float(np.mean(matching)))
        values.append(row)
    return {
        "row_axis": "training_difficulty",
        "column_axis": "evaluation_difficulty",
        "rows": list(difficulty_ids),
        "columns": list(difficulty_ids),
        "values": values,
    }


def _write_matrix_csv(path: Path, matrix: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["training_difficulty", *matrix["columns"]])
        for label, values in zip(matrix["rows"], matrix["values"]):
            writer.writerow([label, *values])


def main() -> None:
    args = parse_args()
    if args.episodes <= 0 or not args.seeds:
        raise ValueError("--episodes and --seeds must be non-empty and positive")
    protocol = protocol_from_args(args)
    difficulty_ids = tuple(task.name for task in protocol.tasks)
    expected_pairs = cross_evaluation_pairs(difficulty_ids)
    device = resolve_device(args.device)
    summary_records = []
    raw_records = []
    for seed in args.seeds:
        for training_difficulty, evaluation_difficulty in expected_pairs:
            training_task = protocol.task(training_difficulty)
            evaluation_task = protocol.task(evaluation_difficulty)
            checkpoint_path = (
                args.checkpoint_root
                / args.environment_version
                / "difficulty"
                / training_difficulty
                / f"seed_{seed}"
                / args.checkpoint_name
            )
            actors, critic, checkpoint = load_happo_checkpoint(
                checkpoint_path, device=device
            )
            checkpoint_version = checkpoint.get("metadata", {}).get(
                "environment_version"
            )
            if checkpoint_version != args.environment_version:
                raise ValueError(
                    f"Checkpoint {checkpoint_path} uses {checkpoint_version!r}, "
                    f"expected {args.environment_version!r}"
                )
            env = make_env(
                evaluation_task,
                environment_version=args.environment_version,
                max_cycles=args.max_cycles,
            )
            try:
                episodes = [
                    collect_episode(
                        env,
                        actors,
                        critic,
                        seed=(
                            args.seed_base
                            + seed * 10_000_000
                            + int(training_difficulty[1:]) * 1_000_000
                            + int(evaluation_difficulty[1:]) * 100_000
                            + episode
                        ),
                        deterministic=True,
                        device=device,
                    ).metrics
                    for episode in range(args.episodes)
                ]
            finally:
                env.close()
            row = {
                "environment_version": args.environment_version,
                "training_difficulty": training_difficulty,
                "training_strength": training_task.actuator_scale,
                "evaluation_difficulty": evaluation_difficulty,
                "evaluation_strength": evaluation_task.actuator_scale,
                "seed": seed,
                **_metric_summary(episodes),
                "num_eval_episodes": args.episodes,
            }
            summary_records.append(row)
            for episode_index, episode_metrics in enumerate(episodes):
                raw_records.append(
                    {
                        "environment_version": args.environment_version,
                        "training_difficulty": training_difficulty,
                        "training_strength": training_task.actuator_scale,
                        "evaluation_difficulty": evaluation_difficulty,
                        "evaluation_strength": evaluation_task.actuator_scale,
                        "seed": seed,
                        "episode_index": episode_index,
                        **episode_metrics,
                    }
                )
            print(
                f"seed={seed} {training_difficulty}->{evaluation_difficulty} "
                f"return={row['episode_return_mean']:.2f}"
            )

    matrices = {
        metric: build_metric_matrix(summary_records, difficulty_ids, metric)
        for metric in METRICS
    }
    pair_summaries = []
    for training_difficulty, evaluation_difficulty in expected_pairs:
        matching = [
            record
            for record in raw_records
            if record["training_difficulty"] == training_difficulty
            and record["evaluation_difficulty"] == evaluation_difficulty
        ]
        pair_summaries.append(
            {
                "environment_version": args.environment_version,
                "training_difficulty": training_difficulty,
                "training_strength": protocol.task(training_difficulty).actuator_scale,
                "evaluation_difficulty": evaluation_difficulty,
                "evaluation_strength": protocol.task(evaluation_difficulty).actuator_scale,
                **_metric_summary(matching),
                "num_seeds": len(args.seeds),
                "num_eval_episodes": len(matching),
            }
        )
    result = {
        "method": "happo_difficulty_cross_evaluation",
        "environment_version": args.environment_version,
        "agent_conf": "6x1",
        "agent_obsk": 1,
        "difficulty_protocol": protocol.to_dict(),
        "seeds": list(args.seeds),
        "episodes_per_seed": args.episodes,
        "unique_train_evaluation_combinations": len(expected_pairs),
        "summary_records": summary_records,
        "raw_episodes": raw_records,
        "matrices": matrices,
        "diagonal": [
            row
            for row in pair_summaries
            if row["training_difficulty"] == row["evaluation_difficulty"]
        ],
        "off_diagonal": [
            row
            for row in pair_summaries
            if row["training_difficulty"] != row["evaluation_difficulty"]
        ],
    }
    output_dir = args.output_dir or DEFAULT_OUTPUT_ROOT / args.environment_version
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "cross_evaluation.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, sort_keys=True)
    with (output_dir / "cross_evaluation.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_records[0]))
        writer.writeheader()
        writer.writerows(summary_records)
    with (output_dir / "cross_evaluation_raw.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(raw_records[0]))
        writer.writeheader()
        writer.writerows(raw_records)
    for metric, matrix in matrices.items():
        _write_matrix_csv(output_dir / f"matrix_{metric}.csv", matrix)
    print(f"Saved 4x4 cross-evaluation: {output_dir}")


if __name__ == "__main__":
    main()
