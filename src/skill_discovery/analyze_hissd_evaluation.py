"""Compute paired uncertainty estimates for HiSSD rollout evaluation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = PROJECT_ROOT / "src/skill_discovery/hissd_rollout_results.json"
RATE_METRICS = (
    "success",
    "drone_task_success",
    "goal_found",
    "drone_goal_found",
    "fatal_crash",
    "drone_crash",
    "observer_crash",
)
CONTINUOUS_METRICS = ("coverage_ratio", "cycles")


def parse_args() -> argparse.Namespace:
    """Parse saved rollout and bootstrap settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def paired_metric_difference(
    hissd_values: np.ndarray,
    baseline_values: np.ndarray,
    *,
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> dict[str, float | int | bool]:
    """Return paired mean difference and a percentile bootstrap interval."""
    differences = hissd_values.astype(np.float64) - baseline_values.astype(np.float64)
    episode_count = differences.size
    sampled_indices = rng.integers(
        0,
        episode_count,
        size=(bootstrap_samples, episode_count),
    )
    bootstrap_means = differences[sampled_indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, (0.025, 0.975))
    return {
        "hissd_mean": float(hissd_values.mean()),
        "baseline_mean": float(baseline_values.mean()),
        "difference": float(differences.mean()),
        "ci95_low": float(lower),
        "ci95_high": float(upper),
        "ci_excludes_zero": bool(lower > 0.0 or upper < 0.0),
        "episodes": episode_count,
    }


def analyze_results(
    episodes: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Compare HiSSD and each baseline using exactly matched seeds."""
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive.")
    by_controller = {
        controller: {
            (int(item["difficulty"]), int(item["seed"])): item
            for item in episodes
            if item["controller"] == controller
        }
        for controller in {item["controller"] for item in episodes}
    }
    if "hissd" not in by_controller:
        raise ValueError("Results do not contain a HiSSD controller.")

    output: dict[str, Any] = {}
    rng = np.random.default_rng(seed)
    difficulties = sorted({int(item["difficulty"]) for item in episodes})
    preferred_order = (
        "hissd_reference",
        "hissd_no_task",
        "hissd_no_common",
        "hissd_base",
        "bc",
        "mappo",
    )
    baselines = [name for name in preferred_order if name in by_controller]
    baselines.extend(
        sorted(name for name in by_controller if name not in {"hissd", *baselines})
    )
    for difficulty in difficulties:
        for baseline in baselines:
            keys = sorted(
                key
                for key in by_controller["hissd"]
                if key[0] == difficulty and key in by_controller[baseline]
            )
            if not keys:
                continue
            comparison = {}
            for metric in (*RATE_METRICS, *CONTINUOUS_METRICS):
                def metric_value(item: dict[str, Any]) -> Any:
                    if metric == "success":
                        return item.get("mission_success", item["success"])
                    if metric == "drone_task_success":
                        return item.get("drone_task_success", item["success"])
                    return item[metric]

                hissd_values = np.asarray(
                    [metric_value(by_controller["hissd"][key]) for key in keys]
                )
                baseline_values = np.asarray(
                    [metric_value(by_controller[baseline][key]) for key in keys]
                )
                comparison[metric] = paired_metric_difference(
                    hissd_values,
                    baseline_values,
                    rng=rng,
                    bootstrap_samples=bootstrap_samples,
                )
            name = f"difficulty_{difficulty}/hissd_vs_{baseline}"
            output[name] = comparison
            success = comparison["success"]
            crash = comparison["fatal_crash"]
            print(
                f"PAIRED difficulty={difficulty} baseline={baseline} "
                f"success_delta={success['difference']:+.3f} "
                f"CI95=[{success['ci95_low']:+.3f},{success['ci95_high']:+.3f}] "
                f"fatal_crash_delta={crash['difference']:+.3f} "
                f"CI95=[{crash['ci95_low']:+.3f},{crash['ci95_high']:+.3f}]"
            )
    return output


def main() -> None:
    """Load one evaluator JSON and save paired statistical diagnostics."""
    args = parse_args()
    results_path = args.results.expanduser().resolve()
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    analysis = analyze_results(
        payload.get("episodes", []),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else results_path.with_name(f"{results_path.stem}_paired.json")
    )
    output_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(f"Saved paired evaluation analysis: {output_path}")


if __name__ == "__main__":
    main()
