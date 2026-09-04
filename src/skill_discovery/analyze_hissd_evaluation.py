"""Compute paired uncertainty estimates for HiSSD rollout evaluation results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = PROJECT_ROOT / "src/skill_discovery/hissd_rollout_results.json"
RATE_METRICS = (
    "success",
    "mission_success",
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
    parser.add_argument(
        "--plot",
        type=Path,
        help="Optionally save absolute success rates and paired differences.",
    )
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
                    if metric == "mission_success":
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


def plot_results(
    payload: dict[str, Any],
    analysis: dict[str, Any],
    output_path: Path,
) -> None:
    """Plot task-selected success rates and paired HiSSD differences."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summaries = payload.get("summaries", {})
    parsed = []
    for key, summary in summaries.items():
        if "/difficulty_" not in key:
            continue
        controller, difficulty_text = key.split("/difficulty_", 1)
        parsed.append((controller, int(difficulty_text), summary))
    difficulties = sorted({difficulty for _, difficulty, _ in parsed})
    preferred = ("hissd", "hissd_reference", "bc", "mappo")
    available = {controller for controller, _, _ in parsed}
    controllers = [controller for controller in preferred if controller in available]
    controllers.extend(sorted(available.difference(controllers)))

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    width = 0.8 / max(len(controllers), 1)
    positions = np.arange(len(difficulties), dtype=np.float64)
    colors = dict(zip(controllers, plt.get_cmap("tab10").colors))
    for controller_index, controller in enumerate(controllers):
        values = []
        errors = []
        for difficulty in difficulties:
            summary = next(
                item
                for name, task, item in parsed
                if name == controller and task == difficulty
            )
            rate = float(summary["success_rate"])
            episodes = max(int(summary.get("episodes", 1)), 1)
            values.append(rate)
            errors.append(1.96 * math.sqrt(rate * (1.0 - rate) / episodes))
        offsets = positions - 0.4 + width / 2 + controller_index * width
        axes[0].bar(
            offsets,
            values,
            width=width,
            yerr=errors,
            capsize=3,
            label=controller,
            color=colors[controller],
        )
    axes[0].set_xticks(positions, [f"D{difficulty}" for difficulty in difficulties])
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Task-selected success rate")
    axes[0].set_title("Checkpoint rollout success")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    labels = []
    differences = []
    lower_errors = []
    upper_errors = []
    for key, comparison in analysis.items():
        difficulty_text, baseline = key.split("/hissd_vs_", 1)
        success = comparison["success"]
        labels.append(f"D{difficulty_text.removeprefix('difficulty_')} vs {baseline}")
        difference = float(success["difference"])
        differences.append(difference)
        lower_errors.append(difference - float(success["ci95_low"]))
        upper_errors.append(float(success["ci95_high"]) - difference)
    y_positions = np.arange(len(labels), dtype=np.float64)
    axes[1].errorbar(
        differences,
        y_positions,
        xerr=np.asarray([lower_errors, upper_errors]),
        fmt="o",
        capsize=4,
        color="#176b87",
    )
    axes[1].axvline(0.0, color="#9b2c2c", linestyle="--", linewidth=1.2)
    axes[1].set_yticks(y_positions, labels)
    axes[1].set_xlabel("HiSSD success-rate difference (paired 95% CI)")
    axes[1].set_title("Paired comparison on identical seeds")
    axes[1].grid(axis="x", alpha=0.25)

    success_name = payload.get("success_definition", "configured task")
    figure.suptitle(f"HiSSD evaluation: {success_name}")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved paired evaluation plot: {output_path}")


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
    if args.plot is not None:
        plot_results(payload, analysis, args.plot.expanduser().resolve())


if __name__ == "__main__":
    main()
