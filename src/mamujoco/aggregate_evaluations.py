"""Pool four-seed zero-shot evaluations into the paper-style 32 runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRIC_NAMES = (
    "episode_return",
    "forward_distance",
    "forward_reward",
    "forward_velocity",
    "control_cost",
    "episode_length",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payloads = []
    for path in args.inputs:
        with path.open("r", encoding="utf-8") as file:
            payloads.append(json.load(file))
    identities = {
        (p["algorithm"], p["suite"], p.get("environment_version"))
        for p in payloads
    }
    if len(identities) != 1:
        raise ValueError(f"Cannot aggregate mixed experiments: {identities}")
    algorithm, suite, environment_version = identities.pop()
    result = {
        "algorithm": algorithm,
        "suite": suite,
        "environment_version": environment_version,
        "primary_metric": "episode_return",
        "zero_shot": True,
        "input_files": [str(path.resolve()) for path in args.inputs],
        "training_seed_count": len(payloads),
        "source": {},
        "target": {},
    }
    for split in ("source", "target"):
        task_names = payloads[0][split].keys()
        for task_name in task_names:
            episodes = [
                episode
                for payload in payloads
                for episode in payload[split][task_name]["episodes"]
            ]
            result[split][task_name] = {
                "task": payloads[0][split][task_name]["task"],
                "runs": len(episodes),
                "metrics": {
                    metric: {
                        "mean": float(np.mean([row[metric] for row in episodes])),
                        "std": float(np.std([row[metric] for row in episodes], ddof=1))
                        if len(episodes) > 1
                        else 0.0,
                    }
                    for metric in METRIC_NAMES
                },
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, sort_keys=True)
    runs_per_task = sorted({row["runs"] for row in result["target"].values()})
    print(f"Saved target evaluation ({runs_per_task} runs per task): {args.output}")


if __name__ == "__main__":
    main()
