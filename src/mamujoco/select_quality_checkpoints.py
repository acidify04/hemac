"""Select High/Mid/Low HAPPO checkpoints from evaluation-return curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .difficulty_protocol import QUALITY_FRACTIONS, add_difficulty_arguments, protocol_from_args
from .env import add_environment_version_argument


DEFAULT_CHECKPOINT_ROOT = Path("src/mamujoco/checkpoints/happo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path)
    add_environment_version_argument(parser)
    add_difficulty_arguments(parser)
    return parser.parse_args()


def select_curve_points(records: list[dict]) -> dict[str, dict]:
    """Select the closest recorded return for each requested quality fraction."""
    evaluated = [row for row in records if "eval/episode_return" in row]
    if not evaluated:
        raise ValueError("Training history contains no evaluation records")
    maximum = max(float(row["eval/episode_return"]) for row in evaluated)
    selected = {}
    for quality, fraction in QUALITY_FRACTIONS.items():
        target_return = fraction * maximum
        row = min(
            evaluated,
            key=lambda item: abs(float(item["eval/episode_return"]) - target_return),
        )
        selected[quality] = {
            "target_fraction": fraction,
            "target_return": target_return,
            "checkpoint_step": int(row["environment_steps"]),
            "evaluation_return": float(row["eval/episode_return"]),
        }
    return selected


def main() -> None:
    args = parse_args()
    protocol = protocol_from_args(args)
    result = {
        "format_version": 1,
        "suite": "difficulty",
        "seed": args.seed,
        "environment_version": args.environment_version,
        "difficulty_protocol": protocol.to_dict(),
        "rmax": {},
        "qualities": {},
    }
    for difficulty in protocol.source_ids:
        directory = (
            args.checkpoint_root
            / args.environment_version
            / "difficulty"
            / difficulty
            / f"seed_{args.seed}"
        )
        history_path = directory / "training_metrics.jsonl"
        if not history_path.is_file():
            raise FileNotFoundError(f"Missing HAPPO history: {history_path}")
        with history_path.open("r", encoding="utf-8") as file:
            records = [json.loads(line) for line in file if line.strip()]
        selections = select_curve_points(records)
        result["rmax"][difficulty] = selections["high"]["target_return"]
        for selection in selections.values():
            checkpoint = directory / f"checkpoint_{selection['checkpoint_step']:09d}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Missing selected checkpoint: {checkpoint}")
            selection["checkpoint_path"] = str(checkpoint.resolve())
        result["qualities"][difficulty] = selections
    output = args.output or (
        args.checkpoint_root
        / args.environment_version
        / "difficulty"
        / f"quality_selection_seed_{args.seed}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, sort_keys=True)
    print(f"Saved quality selections: {output}")


if __name__ == "__main__":
    main()
