"""Build and analyze target-task success learning curves.

The statistical unit is one training seed. Environment interactions are counted
as joint world cycles; evaluation rollouts must not be included in that count.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


FORMAT_VERSION = 1
STEP_UNIT = "joint_environment_cycle"
DEFAULT_THRESHOLDS = {4: 0.4, 5: 0.2, 6: 0.1}
SUMMARY_METRICS = (
    "success_rate",
    "goal_found_rate",
    "drone_goal_found_rate",
    "fatal_crash_rate",
    "drone_crash_rate",
    "observer_crash_rate",
    "mean_coverage_ratio",
    "mean_cycles",
)


def canonical_success_definition(value: Any) -> str | None:
    """Normalize equivalent task labels emitted by different evaluators."""
    if value is None:
        return None
    return {
        "mission": "observer_goal_arrival",
        "drone": "drone_goal_found_and_coverage",
    }.get(str(value), str(value))


def parse_thresholds(values: Iterable[str]) -> dict[int, float]:
    """Parse repeated `difficulty=rate` threshold arguments."""
    thresholds: dict[int, float] = {}
    for value in values:
        try:
            difficulty_text, rate_text = value.split("=", 1)
            difficulty = int(difficulty_text)
            rate = float(rate_text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid threshold {value!r}; expected DIFFICULTY=RATE."
            ) from exc
        if difficulty <= 0 or not 0.0 <= rate <= 1.0:
            raise argparse.ArgumentTypeError(
                f"Invalid threshold {value!r}; difficulty must be positive and rate in [0, 1]."
            )
        thresholds[difficulty] = rate
    return thresholds or dict(DEFAULT_THRESHOLDS)


def parse_baselines(values: Iterable[str]) -> dict[str, str]:
    """Parse repeated `controller=method` static-baseline arguments."""
    baselines: dict[str, str] = {}
    for value in values:
        try:
            controller, method = value.split("=", 1)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid baseline {value!r}; expected CONTROLLER=METHOD."
            ) from exc
        controller = controller.strip()
        method = method.strip()
        if not controller or not method:
            raise argparse.ArgumentTypeError(
                f"Invalid baseline {value!r}; names cannot be empty."
            )
        baselines[controller] = method
    if not baselines:
        raise argparse.ArgumentTypeError("At least one --baseline is required.")
    return baselines


def load_curve(path: Path) -> dict[str, Any]:
    """Load one standardized learning-curve file."""
    if not path.is_file():
        raise FileNotFoundError(f"Learning-curve file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("format_version", 0)) != FORMAT_VERSION:
        raise ValueError(f"Unsupported learning-curve format: {path}")
    if payload.get("step_unit") != STEP_UNIT:
        raise ValueError(
            f"{path} uses {payload.get('step_unit')!r}; expected {STEP_UNIT!r}."
        )
    if not isinstance(payload.get("points"), list):
        raise ValueError(f"Learning-curve file has no points list: {path}")
    return payload


def save_curve(path: Path, payload: dict[str, Any]) -> None:
    """Atomically save a standardized learning curve."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_curve_points(
    path: Path,
    points: Iterable[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> None:
    """Insert or replace evaluation points identified by method/seed/task/step."""
    if path.is_file():
        payload = load_curve(path)
    else:
        payload = {
            "format_version": FORMAT_VERSION,
            "step_unit": STEP_UNIT,
            "metadata": {},
            "points": [],
        }
    if metadata:
        payload["metadata"].update(metadata)

    keyed = {
        (
            str(point["method"]),
            int(point["seed"]),
            int(point["difficulty"]),
            int(point["joint_env_steps"]),
        ): point
        for point in payload["points"]
    }
    for raw_point in points:
        point = dict(raw_point)
        key = (
            str(point["method"]),
            int(point["seed"]),
            int(point["difficulty"]),
            int(point["joint_env_steps"]),
        )
        if key[3] < 0:
            raise ValueError("joint_env_steps cannot be negative.")
        success_rate = float(point["success_rate"])
        if not 0.0 <= success_rate <= 1.0:
            raise ValueError(f"Invalid success rate in point {key}: {success_rate}")
        point.update(
            {
                "method": key[0],
                "seed": key[1],
                "difficulty": key[2],
                "joint_env_steps": key[3],
                "success_rate": success_rate,
            }
        )
        keyed[key] = point
    payload["points"] = sorted(
        keyed.values(),
        key=lambda point: (
            point["method"],
            point["seed"],
            point["difficulty"],
            point["joint_env_steps"],
        ),
    )
    save_curve(path, payload)


def evaluation_points(
    evaluation_path: Path,
    controller: str,
    method: str,
    seed: int,
    joint_env_steps: int,
) -> list[dict[str, Any]]:
    """Convert one endpoint evaluation JSON into standardized curve points."""
    payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
    summaries = payload.get("summaries", {})
    prefix = f"{controller}/difficulty_"
    points = []
    for key, summary in summaries.items():
        if not key.startswith(prefix):
            continue
        difficulty = int(key.removeprefix(prefix))
        point = {
            "method": method,
            "seed": seed,
            "difficulty": difficulty,
            "joint_env_steps": joint_env_steps,
            "evaluation_source": str(evaluation_path.resolve()),
        }
        for metric in SUMMARY_METRICS:
            if metric in summary:
                point[metric] = float(summary[metric])
        if "success_rate" not in point:
            raise ValueError(f"Summary {key!r} has no success_rate.")
        points.append(point)
    if not points:
        raise ValueError(
            f"No summaries for controller {controller!r} in {evaluation_path}."
        )
    return points


def reference_curve_budgets(
    curve_paths: Iterable[Path],
    seed: int,
) -> dict[int, int]:
    """Return each difficulty's final online step for one training seed."""
    budgets: dict[int, int] = {}
    for path in curve_paths:
        for point in load_curve(path)["points"]:
            if int(point["seed"]) != seed:
                continue
            difficulty = int(point["difficulty"])
            step = int(point["joint_env_steps"])
            budgets[difficulty] = max(budgets.get(difficulty, 0), step)
    if not budgets:
        raise ValueError(f"No reference curves were found for seed {seed}.")
    invalid = [difficulty for difficulty, budget in budgets.items() if budget <= 0]
    if invalid:
        raise ValueError(
            "Reference curves need positive-step evaluations for difficulties "
            f"{invalid}."
        )
    return budgets


def static_baseline_points(
    evaluation_path: Path,
    baselines: dict[str, str],
    seed: int,
    budgets: dict[int, int],
) -> list[dict[str, Any]]:
    """Create constant checkpoint baselines over matching online budgets."""
    points: list[dict[str, Any]] = []
    for controller, method in baselines.items():
        endpoints = evaluation_points(
            evaluation_path,
            controller,
            method,
            seed,
            joint_env_steps=0,
        )
        matched = 0
        for endpoint in endpoints:
            difficulty = int(endpoint["difficulty"])
            if difficulty not in budgets:
                continue
            endpoint["static_checkpoint_baseline"] = True
            points.append(endpoint)
            points.append(
                {
                    **endpoint,
                    "joint_env_steps": budgets[difficulty],
                }
            )
            matched += 1
        if matched == 0:
            raise ValueError(
                f"Controller {controller!r} has no difficulties matching the "
                "reference curves."
            )
    return points


def interpolate_at(points: list[dict[str, Any]], step: int) -> float:
    """Linearly interpolate success at a requested evaluation budget."""
    for point in points:
        if point["joint_env_steps"] == step:
            return float(point["success_rate"])
    for left, right in zip(points, points[1:]):
        x0 = int(left["joint_env_steps"])
        x1 = int(right["joint_env_steps"])
        if x0 < step < x1:
            ratio = (step - x0) / (x1 - x0)
            return float(left["success_rate"]) + ratio * (
                float(right["success_rate"]) - float(left["success_rate"])
            )
    raise ValueError(f"Step {step} is outside the recorded curve.")


def run_metrics(
    points: list[dict[str, Any]],
    budget: int,
    threshold: float,
) -> dict[str, Any]:
    """Calculate normalized AUC and first observed threshold crossing."""
    ordered = sorted(points, key=lambda point: int(point["joint_env_steps"]))
    if int(ordered[0]["joint_env_steps"]) != 0:
        raise ValueError("Every learning curve must include a step-0 evaluation.")
    if int(ordered[-1]["joint_env_steps"]) < budget:
        raise ValueError(
            f"Curve ends at {ordered[-1]['joint_env_steps']}, before budget {budget}."
        )
    clipped = [point for point in ordered if int(point["joint_env_steps"]) < budget]
    clipped.append(
        {
            "joint_env_steps": budget,
            "success_rate": interpolate_at(ordered, budget),
        }
    )
    area = 0.0
    for left, right in zip(clipped, clipped[1:]):
        width = int(right["joint_env_steps"]) - int(left["joint_env_steps"])
        area += width * (
            float(left["success_rate"]) + float(right["success_rate"])
        ) / 2.0
    reached = next(
        (
            int(point["joint_env_steps"])
            for point in ordered
            if int(point["joint_env_steps"]) <= budget
            and float(point["success_rate"]) >= threshold
        ),
        None,
    )
    initial_success_rate = float(ordered[0]["success_rate"])
    final_success_rate = float(clipped[-1]["success_rate"])
    success_auc = area / budget
    return {
        "budget": budget,
        "initial_success_rate": initial_success_rate,
        "success_auc": success_auc,
        "success_gain_auc": success_auc - initial_success_rate,
        "final_success_rate": final_success_rate,
        "final_success_gain": final_success_rate - initial_success_rate,
        "threshold": threshold,
        "first_threshold_step": reached,
        "threshold_reached": reached is not None,
        "capped_threshold_step": reached if reached is not None else budget,
        "evaluation_points": len(clipped),
    }


def mean_ci95(values: list[float]) -> dict[str, float]:
    """Return mean, sample deviation, and a small-sample t confidence interval."""
    mean = statistics.fmean(values)
    if len(values) < 2:
        return {"mean": mean, "std": 0.0, "ci95_low": mean, "ci95_high": mean}
    std = statistics.stdev(values)
    # Two-sided 95% Student-t critical values; use the normal limit above 30 df.
    critical = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    }.get(len(values) - 1, 1.96)
    margin = critical * std / math.sqrt(len(values))
    return {
        "mean": mean,
        "std": std,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
    }


def analyze(
    curve_paths: list[Path],
    thresholds: dict[int, float],
    requested_budget: int | None,
) -> dict[str, Any]:
    """Analyze all method/seed/difficulty curves on common per-task budgets."""
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for path in curve_paths:
        for point in load_curve(path)["points"]:
            difficulty = int(point["difficulty"])
            if difficulty not in thresholds:
                continue
            key = (
                str(point["method"]),
                int(point["seed"]),
                difficulty,
            )
            grouped[key].append(point)
    if not grouped:
        raise ValueError("No learning-curve points were found.")

    budgets = {}
    for difficulty in sorted({key[2] for key in grouped}):
        endings = [
            max(int(point["joint_env_steps"]) for point in points)
            for key, points in grouped.items()
            if key[2] == difficulty
        ]
        budget = requested_budget if requested_budget is not None else min(endings)
        if budget <= 0:
            raise ValueError(
                f"Difficulty {difficulty} has no positive common budget; "
                "one endpoint evaluation is not enough to calculate AUC."
            )
        budgets[difficulty] = budget

    per_run = []
    for (method, seed, difficulty), points in sorted(grouped.items()):
        threshold = thresholds.get(difficulty)
        if threshold is None:
            raise ValueError(f"No success threshold configured for difficulty {difficulty}.")
        per_run.append(
            {
                "method": method,
                "seed": seed,
                "difficulty": difficulty,
                **run_metrics(points, budgets[difficulty], threshold),
            }
        )

    aggregate_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for result in per_run:
        aggregate_groups[(result["method"], result["difficulty"])].append(result)
    aggregates = []
    for (method, difficulty), results in sorted(aggregate_groups.items()):
        reached_steps = [
            float(result["first_threshold_step"])
            for result in results
            if result["threshold_reached"]
        ]
        aggregates.append(
            {
                "method": method,
                "difficulty": difficulty,
                "seeds": len(results),
                "budget": budgets[difficulty],
                "threshold": thresholds[difficulty],
                "success_auc": mean_ci95(
                    [float(result["success_auc"]) for result in results]
                ),
                "success_gain_auc": mean_ci95(
                    [float(result["success_gain_auc"]) for result in results]
                ),
                "initial_success_rate": mean_ci95(
                    [float(result["initial_success_rate"]) for result in results]
                ),
                "final_success_rate": mean_ci95(
                    [float(result["final_success_rate"]) for result in results]
                ),
                "final_success_gain": mean_ci95(
                    [float(result["final_success_gain"]) for result in results]
                ),
                "threshold_reached_fraction": (
                    len(reached_steps) / len(results)
                ),
                "first_threshold_step_reached_only": (
                    mean_ci95(reached_steps) if reached_steps else None
                ),
                "capped_threshold_step": mean_ci95(
                    [float(result["capped_threshold_step"]) for result in results]
                ),
            }
        )
    per_run_lookup = {
        (result["method"], result["difficulty"], result["seed"]): result
        for result in per_run
    }
    pairwise = []
    methods = sorted({result["method"] for result in per_run})
    for difficulty in sorted(budgets):
        for method_a, method_b in itertools.combinations(methods, 2):
            seeds = sorted(
                seed
                for method, task, seed in per_run_lookup
                if method == method_a
                and task == difficulty
                and (method_b, difficulty, seed) in per_run_lookup
            )
            if not seeds:
                continue
            comparison = {
                "method_a": method_a,
                "method_b": method_b,
                "difference_definition": "method_a - method_b",
                "difficulty": difficulty,
                "seeds": len(seeds),
            }
            for metric in (
                "success_auc",
                "success_gain_auc",
                "final_success_rate",
                "final_success_gain",
                "capped_threshold_step",
            ):
                differences = [
                    float(per_run_lookup[(method_a, difficulty, seed)][metric])
                    - float(per_run_lookup[(method_b, difficulty, seed)][metric])
                    for seed in seeds
                ]
                comparison[metric] = mean_ci95(differences)
            pairwise.append(comparison)
    return {
        "format_version": FORMAT_VERSION,
        "step_unit": STEP_UNIT,
        "budgets": {str(key): value for key, value in budgets.items()},
        "thresholds": {str(key): value for key, value in thresholds.items()},
        "per_run": per_run,
        "aggregates": aggregates,
        "pairwise": pairwise,
    }


def write_analysis(path: Path, analysis: dict[str, Any]) -> None:
    """Write JSON plus a compact aggregate CSV next to it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    csv_path = path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            (
                "method", "difficulty", "seeds", "budget", "threshold",
                "success_auc_mean", "success_auc_std", "success_auc_ci95_low",
                "success_auc_ci95_high", "final_success_mean",
                "success_gain_auc_mean", "initial_success_mean",
                "final_success_gain_mean",
                "threshold_reached_fraction", "capped_threshold_step_mean",
            )
        )
        for row in analysis["aggregates"]:
            writer.writerow(
                (
                    row["method"], row["difficulty"], row["seeds"], row["budget"],
                    row["threshold"], row["success_auc"]["mean"],
                    row["success_auc"]["std"], row["success_auc"]["ci95_low"],
                    row["success_auc"]["ci95_high"],
                    row["final_success_rate"]["mean"],
                    row["success_gain_auc"]["mean"],
                    row["initial_success_rate"]["mean"],
                    row["final_success_gain"]["mean"],
                    row["threshold_reached_fraction"],
                    row["capped_threshold_step"]["mean"],
                )
            )


def plot_learning_curves(
    curve_paths: list[Path],
    analysis: dict[str, Any],
    output_path: Path,
) -> tuple[Path, Path]:
    """Plot seed-aggregated success curves and metric summary bars."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thresholds = {
        int(key): float(value) for key, value in analysis["thresholds"].items()
    }
    budgets = {
        int(key): int(value) for key, value in analysis["budgets"].items()
    }
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for path in curve_paths:
        for point in load_curve(path)["points"]:
            difficulty = int(point["difficulty"])
            if difficulty not in thresholds:
                continue
            grouped[(str(point["method"]), int(point["seed"]), difficulty)].append(
                point
            )

    difficulties = sorted(budgets)
    methods = sorted({key[0] for key in grouped})
    colors = dict(zip(methods, plt.get_cmap("tab10").colors))
    figure, axes = plt.subplots(
        1,
        len(difficulties),
        figsize=(5.2 * len(difficulties), 4.4),
        squeeze=False,
        sharey=True,
    )
    for axis, difficulty in zip(axes[0], difficulties):
        budget = budgets[difficulty]
        grid = [round(index * budget / 100) for index in range(101)]
        for method in methods:
            seed_curves = [
                sorted(points, key=lambda point: int(point["joint_env_steps"]))
                for (name, _, task), points in grouped.items()
                if name == method and task == difficulty
            ]
            if not seed_curves:
                continue
            values_by_step = [
                [interpolate_at(points, step) for points in seed_curves]
                for step in grid
            ]
            means = [statistics.fmean(values) for values in values_by_step]
            intervals = [mean_ci95(values) for values in values_by_step]
            axis.plot(
                [step / 1000.0 for step in grid],
                means,
                label=method,
                color=colors[method],
                linewidth=2.0,
            )
            if len(seed_curves) > 1:
                axis.fill_between(
                    [step / 1000.0 for step in grid],
                    [interval["ci95_low"] for interval in intervals],
                    [interval["ci95_high"] for interval in intervals],
                    color=colors[method],
                    alpha=0.16,
                )
        axis.axhline(
            thresholds[difficulty],
            color="#9b2c2c",
            linestyle="--",
            linewidth=1.4,
            label="threshold",
        )
        axis.set_title(f"Difficulty {difficulty}")
        axis.set_xlabel("Target joint env steps (thousands)")
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.25)
    axes[0][0].set_ylabel("Evaluation success rate")
    handles, labels = axes[0][-1].get_legend_handles_labels()
    figure.suptitle("Target-task learning curves", y=0.99)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=max(len(labels), 1),
    )
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    aggregates = analysis["aggregates"]
    summary_path = output_path.with_name(f"{output_path.stem}_metrics.png")
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    width = 0.8 / max(len(methods), 1)
    positions = list(range(len(difficulties)))
    for method_index, method in enumerate(methods):
        rows = {
            int(row["difficulty"]): row
            for row in aggregates
            if row["method"] == method
        }
        offsets = [
            position - 0.4 + width / 2 + method_index * width
            for position in positions
        ]
        auc_values = [
            rows[difficulty]["success_auc"]["mean"]
            for difficulty in difficulties
        ]
        gain_auc_values = [
            rows[difficulty]["success_gain_auc"]["mean"]
            for difficulty in difficulties
        ]
        step_values = [
            rows[difficulty]["capped_threshold_step"]["mean"] / 1000.0
            for difficulty in difficulties
        ]
        axes[0].bar(
            offsets,
            auc_values,
            width=width,
            label=method,
            color=colors[method],
        )
        axes[1].bar(
            offsets,
            gain_auc_values,
            width=width,
            label=method,
            color=colors[method],
        )
        axes[2].bar(
            offsets,
            step_values,
            width=width,
            label=method,
            color=colors[method],
        )
    for axis in axes:
        axis.set_xticks(positions, [f"D{difficulty}" for difficulty in difficulties])
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_title("Normalized Success AUC")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("AUC")
    axes[1].set_title("Success Gain AUC (AUC - Step-0 Success)")
    axes[1].set_ylabel("Gain AUC")
    axes[1].axhline(0.0, color="#333333", linewidth=1.0)
    axes[2].set_title("Steps to Threshold (capped at budget)")
    axes[2].set_ylabel("Joint env steps (thousands)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=max(len(methods), 1),
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    figure.savefig(summary_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path, summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    add = subparsers.add_parser("add-evaluation")
    add.add_argument("--curve", type=Path, required=True)
    add.add_argument("--evaluation", type=Path, required=True)
    add.add_argument("--controller", required=True)
    add.add_argument("--method", required=True)
    add.add_argument("--seed", type=int, required=True)
    add.add_argument("--joint-env-steps", type=int, required=True)

    static = subparsers.add_parser("add-static-baselines")
    static.add_argument("--curve", type=Path, required=True)
    static.add_argument("--evaluation", type=Path, required=True)
    static.add_argument(
        "--baseline",
        action="append",
        default=[],
        metavar="CONTROLLER=METHOD",
        help="For example: --baseline bc=bc_checkpoint.",
    )
    static.add_argument("--reference-curves", type=Path, nargs="+", required=True)
    static.add_argument("--seed", type=int, required=True)
    static.add_argument(
        "--task-definition",
        choices=("mission", "drone"),
        required=True,
        help="Reject evaluation JSON generated with a different success condition.",
    )

    calculate = subparsers.add_parser("analyze")
    calculate.add_argument("--curves", type=Path, nargs="+", required=True)
    calculate.add_argument(
        "--threshold",
        action="append",
        default=[],
        metavar="DIFFICULTY=RATE",
    )
    calculate.add_argument("--budget", type=int)
    calculate.add_argument("--output", type=Path, required=True)
    calculate.add_argument(
        "--plot",
        type=Path,
        help="Defaults to the JSON output path with a .png suffix.",
    )
    calculate.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "add-evaluation":
        if args.joint_env_steps < 0:
            raise ValueError("--joint-env-steps cannot be negative.")
        points = evaluation_points(
            args.evaluation.expanduser().resolve(),
            args.controller,
            args.method,
            args.seed,
            args.joint_env_steps,
        )
        curve_path = args.curve.expanduser().resolve()
        append_curve_points(curve_path, points)
        print(f"Added {len(points)} points to {curve_path}")
        return

    if args.command == "add-static-baselines":
        evaluation_path = args.evaluation.expanduser().resolve()
        reference_paths = [
            path.expanduser().resolve() for path in args.reference_curves
        ]
        budgets = reference_curve_budgets(reference_paths, args.seed)
        baselines = parse_baselines(args.baseline)
        points = static_baseline_points(
            evaluation_path,
            baselines,
            args.seed,
            budgets,
        )
        curve_path = args.curve.expanduser().resolve()
        evaluation_payload = json.loads(
            evaluation_path.read_text(encoding="utf-8")
        )
        actual_definition = canonical_success_definition(
            evaluation_payload.get("success_definition")
        )
        expected_definition = canonical_success_definition(args.task_definition)
        if actual_definition != expected_definition:
            raise ValueError(
                "Evaluation success definition does not match the requested task: "
                f"file={actual_definition!r}, requested={expected_definition!r}."
            )
        append_curve_points(
            curve_path,
            points,
            metadata={
                "static_checkpoint_baseline": True,
                "success_definition": actual_definition,
                "reference_curves": [str(path) for path in reference_paths],
            },
        )
        print(
            f"Added {len(points)} static baseline points to {curve_path}; "
            f"budgets={budgets}"
        )
        return

    if args.budget is not None and args.budget <= 0:
        raise ValueError("--budget must be positive.")
    thresholds = parse_thresholds(args.threshold)
    result = analyze(
        [path.expanduser().resolve() for path in args.curves],
        thresholds,
        args.budget,
    )
    output = args.output.expanduser().resolve()
    write_analysis(output, result)
    print(f"Saved learning-efficiency analysis: {output}")
    print(f"Saved aggregate CSV: {output.with_suffix('.csv')}")
    if not args.no_plot:
        plot_path = (
            args.plot.expanduser().resolve()
            if args.plot is not None
            else output.with_suffix(".png")
        )
        curves_plot, metrics_plot = plot_learning_curves(
            [path.expanduser().resolve() for path in args.curves],
            result,
            plot_path,
        )
        print(f"Saved learning curves: {curves_plot}")
        print(f"Saved metric comparison: {metrics_plot}")


if __name__ == "__main__":
    main()
