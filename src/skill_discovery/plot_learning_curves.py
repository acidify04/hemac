#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

STEP_KEYS = (
    "scheduled_joint_env_steps",
    "joint_env_steps",
    "environment_steps",
    "env_steps",
    "timesteps_total",
)

METHOD_ALIASES = {
    "hissd": "HiSSD",
    "hissd_pure_exact": "HiSSD",
    "hissd_skill_supported_ppo": "HiSSD",
    "bc": "BC",
    "bc_init_ppo": "BC",
    "mappo": "MAPPO",
    "pure_rl_source_init": "MAPPO",
    "rl_scratch": "RL scratch",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Plot target-task learning curves from HeMAC/HiSSD JSON logs. "
            "The input directory is scanned recursively. Runs are grouped "
            "by method and seed; multiple seeds are shown as mean ± 95% CI."
        )
    )
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--difficulty", type=int, default=3)
    p.add_argument("--metric", default="success_rate")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--title", default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument(
        "--show-seeds",
        action="store_true",
        help="Also draw individual seed curves faintly.",
    )
    p.add_argument(
        "--no-ci",
        action="store_true",
        help="Disable 95%% CI shading.",
    )
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


def find_points(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if not isinstance(payload, dict):
        return []

    for key in ("points", "curve", "records", "history"):
        value = payload.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]

    candidates = [
        value
        for value in payload.values()
        if isinstance(value, list)
        and value
        and all(isinstance(x, dict) for x in value)
    ]
    return candidates[0] if len(candidates) == 1 else []


def get_step(point: dict[str, Any]) -> int | None:
    for key in STEP_KEYS:
        value = point.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return int(value)
    return None


def normalize_method(value: Any, path: Path) -> str:
    if isinstance(value, str) and value.strip():
        raw = value.strip()
        return METHOD_ALIASES.get(raw.lower(), raw)

    lower_path = str(path).lower()
    if "hissd" in lower_path:
        return "HiSSD"
    if "/bc" in lower_path or "_bc" in lower_path or "bc_" in lower_path:
        return "BC"
    if "mappo" in lower_path or "pure_rl" in lower_path:
        return "MAPPO"
    if "scratch" in lower_path:
        return "RL scratch"
    return path.stem


def infer_seed(point: dict[str, Any], payload: Any, path: Path) -> int | str:
    value = point.get("seed")
    if isinstance(value, (int, str)):
        return value

    if isinstance(payload, dict):
        meta = payload.get("metadata")
        if isinstance(meta, dict) and isinstance(meta.get("seed"), (int, str)):
            return meta["seed"]

    match = re.search(r"seed[_-]?(\d+)", str(path), flags=re.I)
    return int(match.group(1)) if match else path.stem


def infer_difficulty(point: dict[str, Any], payload: Any) -> int | None:
    value = point.get("difficulty")
    if isinstance(value, (int, float)):
        return int(value)

    if isinstance(payload, dict):
        meta = payload.get("metadata")
        if isinstance(meta, dict):
            value = meta.get("difficulty")
            if isinstance(value, (int, float)):
                return int(value)
    return None


def load_runs(
    input_dir: Path,
    difficulty: int,
    metric: str,
    max_steps: int | None,
) -> dict[str, dict[int | str, list[tuple[int, float]]]]:
    runs: dict[str, dict[int | str, list[tuple[int, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    json_paths = sorted(input_dir.rglob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No JSON files found under {input_dir}")

    accepted_files = 0
    for path in json_paths:
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue

        points = find_points(payload)
        if not points:
            continue

        accepted_any = False
        for point in points:
            d = infer_difficulty(point, payload)
            if d is not None and d != difficulty:
                continue

            value = point.get(metric)
            if not isinstance(value, (int, float)):
                continue

            step = get_step(point)
            if step is None or (max_steps is not None and step > max_steps):
                continue

            method = normalize_method(point.get("method"), path)
            seed = infer_seed(point, payload, path)
            runs[method][seed].append((step, float(value)))
            accepted_any = True

        if accepted_any:
            accepted_files += 1

    if not runs:
        raise RuntimeError(
            f"No usable D{difficulty} points with metric={metric!r} found under {input_dir}"
        )

    for method, seed_runs in runs.items():
        for seed, pairs in seed_runs.items():
            by_step: dict[int, float] = {}
            for step, value in pairs:
                by_step[step] = value
            seed_runs[seed] = sorted(by_step.items())

    print(f"Scanned {len(json_paths)} JSON files; accepted {accepted_files}.")
    for method, seed_runs in sorted(runs.items()):
        detail = ", ".join(
            f"{seed}({len(points)} pts)"
            for seed, points in sorted(seed_runs.items(), key=lambda x: str(x[0]))
        )
        print(f"{method}: {len(seed_runs)} seed(s) - {detail}")

    return runs


def aggregate_on_common_steps(
    seed_runs: dict[int | str, list[tuple[int, float]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    step_sets = [{step for step, _ in pairs} for pairs in seed_runs.values() if pairs]
    if not step_sets:
        return np.array([]), np.array([]), np.array([]), 0

    common_steps = sorted(set.intersection(*step_sets))
    if not common_steps:
        return np.array([]), np.array([]), np.array([]), len(step_sets)

    rows = []
    for pairs in seed_runs.values():
        values = dict(pairs)
        rows.append([values[s] for s in common_steps])

    arr = np.asarray(rows, dtype=float)
    mean = arr.mean(axis=0)
    if arr.shape[0] > 1:
        sem = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
        ci95 = 1.96 * sem
    else:
        ci95 = np.zeros_like(mean)

    return np.asarray(common_steps), mean, ci95, arr.shape[0]


def pretty_metric(metric: str) -> str:
    names = {
        "success_rate": "Success rate",
        "fatal_crash_rate": "Fatal crash rate",
        "drone_crash_rate": "Drone crash rate",
        "goal_found_rate": "Goal-found rate",
        "mean_coverage_ratio": "Mean coverage ratio",
        "reward_coverage": "Reward coverage",
        "reward_coverage_rate": "Reward coverage",
        "episode_return": "Episode return",
        "validation_score": "Validation score",
    }
    return names.get(metric, metric.replace("_", " ").title())


def main() -> None:
    args = parse_args()
    runs = load_runs(args.input_dir, args.difficulty, args.metric, args.max_steps)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    for method in sorted(runs):
        seed_runs = runs[method]

        if args.show_seeds:
            for _, pairs in sorted(seed_runs.items(), key=lambda x: str(x[0])):
                x = np.asarray([p[0] for p in pairs])
                y = np.asarray([p[1] for p in pairs])
                ax.plot(x, y, linewidth=1.0, alpha=0.22, label="_nolegend_")

        x, mean, ci95, n = aggregate_on_common_steps(seed_runs)
        if len(x) == 0:
            for seed, pairs in sorted(seed_runs.items(), key=lambda x: str(x[0])):
                x_seed = np.asarray([p[0] for p in pairs])
                y_seed = np.asarray([p[1] for p in pairs])
                ax.plot(
                    x_seed,
                    y_seed,
                    marker="o",
                    markersize=3,
                    linewidth=1.8,
                    label=f"{method} seed {seed}",
                )
            continue

        label = method if n == 1 else f"{method} (n={n})"
        line, = ax.plot(
            x,
            mean,
            marker="o",
            markersize=3.5,
            linewidth=2.0,
            label=label,
        )

        if n > 1 and not args.no_ci:
            ax.fill_between(
                x,
                mean - ci95,
                mean + ci95,
                alpha=0.18,
                color=line.get_color(),
                linewidth=0,
            )

    ax.set_xlabel("Target environment steps")
    ax.set_ylabel(pretty_metric(args.metric))
    ax.set_title(args.title or f"D{args.difficulty} target learning curve")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    if args.metric in {
        "success_rate",
        "fatal_crash_rate",
        "drone_crash_rate",
        "goal_found_rate",
        "mean_coverage_ratio",
        "reward_coverage",
        "reward_coverage_rate",
    }:
        ax.set_ylim(-0.02, 1.02)

    if args.max_steps is not None:
        ax.set_xlim(0, args.max_steps)
    else:
        ax.set_xlim(left=0)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
