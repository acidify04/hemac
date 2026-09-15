"""Plot full-skill versus no-skill evaluation metrics over env steps."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from skill_discovery.analyze_learning_efficiency import (
    load_curve,
    mean_ci95,
    paired_randomization_statistics,
)


METRICS = (
    ("success_rate", "Evaluation success rate", (0.0, 1.0)),
    ("mean_episode_return", "Mean evaluation episode return", None),
)
MODE_LABELS = {
    "full": "Skill",
    "no_skill": "No skill",
}
MODE_COLORS = {
    "full": "#087e8b",
    "no_skill": "#d1495b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curves", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid-points", type=int, default=101)
    return parser.parse_args()


def skill_mode(payload: dict[str, Any], point: dict[str, Any]) -> str:
    mode = str(payload.get("metadata", {}).get("skill_mode", ""))
    if mode in MODE_LABELS:
        return mode
    method = str(point.get("method", ""))
    if "no_skill" in method:
        return "no_skill"
    if "full" in method:
        return "full"
    return method


def interpolate_metric(
    points: list[dict[str, Any]], metric: str, step: float
) -> float:
    ordered = sorted(points, key=lambda point: int(point["joint_env_steps"]))
    samples = [
        (int(point["joint_env_steps"]), float(point[metric]))
        for point in ordered
        if metric in point
    ]
    if not samples:
        raise KeyError(metric)
    if step <= samples[0][0]:
        return samples[0][1]
    if step >= samples[-1][0]:
        return samples[-1][1]
    for (x0, y0), (x1, y1) in zip(samples, samples[1:]):
        if x0 <= step <= x1:
            fraction = (step - x0) / max(x1 - x0, 1)
            return y0 + fraction * (y1 - y0)
    raise ValueError(f"Step {step} is outside the recorded curve.")


def normalized_auc(grid: list[float], values: list[float]) -> float:
    """Average metric value over the shared environment-step budget."""
    budget = grid[-1]
    if budget <= 0:
        return values[-1]
    area = sum(
        (right_step - left_step) * (left_value + right_value) / 2.0
        for left_step, right_step, left_value, right_value in zip(
            grid, grid[1:], values, values[1:]
        )
    )
    return area / budget


def main() -> None:
    args = parse_args()
    if args.grid_points < 2:
        raise ValueError("--grid-points must be at least 2.")

    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for raw_path in args.curves:
        path = raw_path.expanduser().resolve()
        payload = load_curve(path)
        for point in payload["points"]:
            mode = skill_mode(payload, point)
            grouped[(mode, int(point["difficulty"]), int(point["seed"]))].append(
                point
            )
    if not grouped:
        raise ValueError("No learning-curve points were found.")

    difficulties = sorted({key[1] for key in grouped})
    modes = [mode for mode in MODE_LABELS if any(key[0] == mode for key in grouped)]
    if not modes:
        modes = sorted({key[0] for key in grouped})

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        len(METRICS),
        len(difficulties),
        figsize=(5.5 * len(difficulties), 4.0 * len(METRICS)),
        squeeze=False,
    )
    availability: dict[str, dict[str, int]] = {}
    metric_summaries: dict[str, dict[str, Any]] = {
        metric: {} for metric, _, _ in METRICS
    }
    for column, difficulty in enumerate(difficulties):
        run_groups = {
            mode: [
                sorted(points, key=lambda point: int(point["joint_env_steps"]))
                for (group_mode, group_difficulty, _), points in grouped.items()
                if group_mode == mode and group_difficulty == difficulty
            ]
            for mode in modes
        }
        all_runs = [run for runs in run_groups.values() for run in runs]
        budget = min(int(run[-1]["joint_env_steps"]) for run in all_runs)
        grid = [index * budget / (args.grid_points - 1) for index in range(args.grid_points)]

        for row, (metric, ylabel, limits) in enumerate(METRICS):
            axis = axes[row][column]
            plotted = False
            availability.setdefault(metric, {})[str(difficulty)] = 0
            per_mode_seed: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
            for mode in run_groups:
                metric_entries = sorted(
                    (
                        seed,
                        sorted(
                            points,
                            key=lambda point: int(point["joint_env_steps"]),
                        ),
                    )
                    for (group_mode, group_difficulty, seed), points in grouped.items()
                    if group_mode == mode
                    and group_difficulty == difficulty
                    and all(metric in point for point in points)
                )
                metric_runs = [run for _, run in metric_entries]
                availability[metric][str(difficulty)] += len(metric_runs)
                if not metric_runs:
                    continue
                values_by_step = [
                    [interpolate_metric(run, metric, step) for run in metric_runs]
                    for step in grid
                ]
                for (seed, run) in metric_entries:
                    run_values = [interpolate_metric(run, metric, step) for step in grid]
                    per_mode_seed[mode][seed] = {
                        "step_normalized_auc": normalized_auc(grid, run_values),
                        "initial": run_values[0],
                        "final": run_values[-1],
                        "gain_auc": normalized_auc(grid, run_values)
                        - run_values[0],
                        "final_gain": run_values[-1] - run_values[0],
                    }
                means = [statistics.fmean(values) for values in values_by_step]
                intervals = [mean_ci95(values) for values in values_by_step]
                color = MODE_COLORS.get(mode)
                axis.plot(
                    [step / 1000.0 for step in grid],
                    means,
                    color=color,
                    linewidth=2.2,
                    label=f"{MODE_LABELS.get(mode, mode)} (n={len(metric_runs)})",
                )
                if len(metric_runs) > 1:
                    low = [interval["ci95_low"] for interval in intervals]
                    high = [interval["ci95_high"] for interval in intervals]
                    if limits is not None:
                        low = [max(limits[0], value) for value in low]
                        high = [min(limits[1], value) for value in high]
                    axis.fill_between(
                        [step / 1000.0 for step in grid],
                        low,
                        high,
                        color=color,
                        alpha=0.16,
                    )
                plotted = True
            if not plotted:
                axis.text(
                    0.5,
                    0.5,
                    "Reward was not recorded in these curves.\n"
                    "Re-run the paired online experiment.",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                    color="#6b7280",
                )
            axis.set_title(f"Difficulty {difficulty}")
            axis.set_xlabel("Target joint env steps (thousands)")
            axis.set_ylabel(ylabel)
            if limits is not None:
                axis.set_ylim(*limits)
            axis.grid(alpha=0.22)
            if plotted:
                axis.legend(frameon=False)

            aggregates = {}
            for mode, seed_values in per_mode_seed.items():
                aggregates[mode] = {
                    "seeds": len(seed_values),
                    **{
                        name: mean_ci95(
                            [values[name] for values in seed_values.values()]
                        )
                        for name in (
                            "step_normalized_auc",
                            "gain_auc",
                            "initial",
                            "final",
                            "final_gain",
                        )
                    },
                }
            paired = None
            if "full" in per_mode_seed and "no_skill" in per_mode_seed:
                paired_seeds = sorted(
                    set(per_mode_seed["full"]) & set(per_mode_seed["no_skill"])
                )
                paired = {"seeds": len(paired_seeds), "difference": "full-no_skill"}
                for name in (
                    "step_normalized_auc",
                    "gain_auc",
                    "initial",
                    "final",
                    "final_gain",
                ):
                    differences = [
                        per_mode_seed["full"][seed][name]
                        - per_mode_seed["no_skill"][seed][name]
                        for seed in paired_seeds
                    ]
                    if differences:
                        paired[name] = {
                            **mean_ci95(differences),
                            **paired_randomization_statistics(differences),
                        }
            metric_summaries[metric][str(difficulty)] = {
                "budget": budget,
                "aggregates": aggregates,
                "paired": paired,
            }

    figure.suptitle("Skill versus no-skill learning curves", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)

    metadata_path = output.with_name(f"{output.stem}_metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "curves": [str(path.expanduser().resolve()) for path in args.curves],
                "difficulties": difficulties,
                "modes": modes,
                "metric_run_counts": availability,
                "metric_summaries": metric_summaries,
                "confidence_interval": "two-sided 95% Student-t across seeds",
                "step_unit": "joint_environment_cycle",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved skill-control curves: {output}")
    print(f"Saved curve metadata: {metadata_path}")
    if not any(availability["mean_episode_return"].values()):
        print(
            "Reward is absent from the supplied legacy curves; success was plotted, "
            "but reward requires re-running online collection with the updated runner."
        )


if __name__ == "__main__":
    main()
