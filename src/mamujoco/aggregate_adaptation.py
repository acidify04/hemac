"""Aggregate five-seed D3/D4 adaptation curves and reward AUC summaries."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


CURVE_METRICS = (
    "episode_return_mean",
    "forward_velocity_mean",
    "control_cost_mean",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payloads = []
    for path in args.inputs:
        with path.open("r", encoding="utf-8") as file:
            payloads.append(json.load(file))
    grouped = defaultdict(list)
    for payload in payloads:
        grouped[payload["summary"]["target_difficulty"]].append(payload)
    environment_versions = {
        payload.get("environment_version") for payload in payloads
    }
    if len(environment_versions) != 1:
        raise ValueError(
            f"Cannot aggregate mixed environment versions: {environment_versions}"
        )
    result = {
        "method": "hissd_based_difficulty_generalization",
        "environment_version": environment_versions.pop(),
        "targets": {},
        "input_files": [str(path.resolve()) for path in args.inputs],
    }
    csv_rows = []
    for target, target_payloads in sorted(grouped.items()):
        seeds = [int(payload["summary"]["seed"]) for payload in target_payloads]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate seeds for {target}: {seeds}")
        reference_steps = [row["adaptation_step"] for row in target_payloads[0]["curve"]]
        for payload in target_payloads[1:]:
            if [row["adaptation_step"] for row in payload["curve"]] != reference_steps:
                raise ValueError(f"Mismatched evaluation schedule for {target}")
        curve = []
        for row_index, step in enumerate(reference_steps):
            aggregate = {
                "method": "hissd_based_difficulty_generalization",
                "target_difficulty": target,
                "adaptation_step": step,
                "seed_count": len(seeds),
            }
            for metric in CURVE_METRICS:
                values = [
                    float(payload["curve"][row_index][metric])
                    for payload in target_payloads
                ]
                base_name = metric.removesuffix("_mean")
                aggregate[f"{base_name}_mean"] = float(np.mean(values))
                aggregate[f"{base_name}_std_across_seeds"] = (
                    float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                )
            curve.append(aggregate)
            csv_rows.append(aggregate)
        auc_values = [
            float(payload["summary"]["normalized_reward_auc"])
            for payload in target_payloads
        ]
        result["targets"][target] = {
            "seeds": sorted(seeds),
            "curve": curve,
            "zero_shot_return": curve[0]["episode_return_mean"],
            "final_return": curve[-1]["episode_return_mean"],
            "normalized_reward_auc_mean": float(np.mean(auc_values)),
            "normalized_reward_auc_std": (
                float(np.std(auc_values, ddof=1)) if len(auc_values) > 1 else 0.0
            ),
            "per_seed_normalized_reward_auc": auc_values,
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "adaptation_summary.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, sort_keys=True)
    if csv_rows:
        with (args.output_dir / "adaptation_summary.csv").open(
            "w", encoding="utf-8", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
    print(f"Saved adaptation aggregation: {args.output_dir}")


if __name__ == "__main__":
    main()
