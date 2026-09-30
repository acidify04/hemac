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

ALIASES = {
    "hissd": "HiSSD",
    "hissd_pure_exact": "HiSSD",
    "hissd_skill_supported_ppo": "HiSSD",
    "bc": "BC",
    "bc_init_ppo": "BC",
    "mappo": "MAPPO",
    "pure_rl_source_init": "MAPPO",
    "rl_scratch": "RL scratch",
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path)
    p.add_argument(
        "--curve",
        action="append",
        default=[],
        metavar="METHOD=PATH",
        help="Explicit curve JSON, e.g. MAPPO=.../mappo_curve.json. Repeatable.",
    )
    p.add_argument("--difficulty", type=int, default=3)
    p.add_argument("--metric", default="success_rate")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--title")
    p.add_argument("--show-seeds", action="store_true")
    p.add_argument("--no-ci", action="store_true")
    p.add_argument("--max-steps", type=int)
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()

def find_points(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("points", "curve", "records", "history"):
        v = payload.get(key)
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
    lists = [
        v for v in payload.values()
        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v)
    ]
    return lists[0] if len(lists) == 1 else []

def get_step(point: dict[str, Any]) -> int | None:
    for key in STEP_KEYS:
        v = point.get(key)
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            return int(v)
    return None

def normalize_method(name: str | None, path: Path) -> str:
    if name:
        return ALIASES.get(name.strip().lower(), name.strip())
    s = str(path).lower()
    if "hissd" in s:
        return "HiSSD"
    if re.search(r"(^|[/_-])bc([/_-]|$)", s):
        return "BC"
    if "mappo" in s or "pure_rl" in s:
        return "MAPPO"
    if "scratch" in s:
        return "RL scratch"
    return path.stem

def infer_seed(point: dict[str, Any], payload: Any, path: Path):
    if isinstance(point.get("seed"), (int, str)):
        return point["seed"]
    if isinstance(payload, dict):
        meta = payload.get("metadata")
        if isinstance(meta, dict) and isinstance(meta.get("seed"), (int, str)):
            return meta["seed"]
    m = re.search(r"seed[_-]?(\d+)", str(path), re.I)
    return int(m.group(1)) if m else path.stem

def infer_diff(point: dict[str, Any], payload: Any):
    if isinstance(point.get("difficulty"), (int, float)):
        return int(point["difficulty"])
    if isinstance(payload, dict):
        meta = payload.get("metadata")
        if isinstance(meta, dict) and isinstance(meta.get("difficulty"), (int, float)):
            return int(meta["difficulty"])
    return None

def parse_explicit(curves: list[str]) -> list[tuple[str, Path]]:
    out = []
    for item in curves:
        if "=" not in item:
            raise ValueError(f"--curve must be METHOD=PATH, got {item!r}")
        method, raw = item.split("=", 1)
        out.append((method.strip(), Path(raw).expanduser()))
    return out

def collect(args):
    runs = defaultdict(lambda: defaultdict(list))
    sources: list[tuple[str | None, Path]] = []

    if args.curve:
        sources.extend(parse_explicit(args.curve))
    elif args.input_dir:
        sources.extend((None, p) for p in sorted(args.input_dir.rglob("*.json")))
    else:
        raise ValueError("Provide --input-dir or one/more --curve METHOD=PATH arguments.")

    used_files = []
    for forced_method, path in sources:
        if not path.exists():
            print(f"WARNING missing: {path}")
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception as e:
            print(f"WARNING unreadable JSON {path}: {e}")
            continue

        points = find_points(payload)
        accepted = 0
        for point in points:
            d = infer_diff(point, payload)
            if d is not None and d != args.difficulty:
                continue
            val = point.get(args.metric)
            step = get_step(point)
            if not isinstance(val, (int, float)) or step is None:
                continue
            if args.max_steps is not None and step > args.max_steps:
                continue
            method = forced_method or normalize_method(point.get("method"), path)
            seed = infer_seed(point, payload, path)
            runs[method][seed].append((int(step), float(val)))
            accepted += 1

        if accepted:
            used_files.append((forced_method, path, accepted))

    # sort + dedup by step
    for method in runs:
        for seed in runs[method]:
            d = {}
            for step, value in runs[method][seed]:
                d[step] = value
            runs[method][seed] = sorted(d.items())

    return runs, used_files

def aggregate(seed_runs):
    sets = [{s for s, _ in pts} for pts in seed_runs.values() if pts]
    if not sets:
        return np.array([]), np.array([]), np.array([]), 0
    common = sorted(set.intersection(*sets))
    if not common:
        return np.array([]), np.array([]), np.array([]), len(sets)
    arr = np.array([[dict(pts)[s] for s in common] for pts in seed_runs.values()], float)
    mean = arr.mean(0)
    if arr.shape[0] > 1:
        ci = 1.96 * arr.std(0, ddof=1) / np.sqrt(arr.shape[0])
    else:
        ci = np.zeros_like(mean)
    return np.array(common), mean, ci, arr.shape[0]

def pretty_metric(metric):
    return {
        "success_rate": "Success rate",
        "fatal_crash_rate": "Fatal crash rate",
        "goal_found_rate": "Goal-found rate",
        "mean_coverage_ratio": "Mean coverage ratio",
        "episode_return": "Episode return",
    }.get(metric, metric.replace("_", " ").title())

def main():
    args = parse_args()
    runs, used = collect(args)
    if not runs:
        raise SystemExit("No usable curve data found.")

    print("=== Loaded curves ===")
    for method, seed_runs in sorted(runs.items()):
        all_steps = sorted({s for pts in seed_runs.values() for s, _ in pts})
        npoints = sum(len(pts) for pts in seed_runs.values())
        print(
            f"{method:12s} seeds={len(seed_runs)} total_points={npoints} "
            f"step_range={all_steps[0]}..{all_steps[-1]}"
        )

    max_by_method = {
        method: max(s for pts in seed_runs.values() for s, _ in pts)
        for method, seed_runs in runs.items()
    }
    global_max = max(max_by_method.values())
    for method, mx in max_by_method.items():
        if global_max > 0 and mx < 0.5 * global_max:
            print(
                f"WARNING: {method} only reaches {mx:,} steps while another method "
                f"reaches {global_max:,}. Curves are not on comparable budgets."
            )

    if "RL scratch" in runs:
        print(
            "WARNING: 'RL scratch' is random initialization. "
            "If the intended baseline is the source MAPPO policy, rerun with "
            "--initialization mappo."
        )

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    for method in sorted(runs):
        seed_runs = runs[method]

        if args.show_seeds:
            for seed, pts in seed_runs.items():
                x = np.array([s for s, _ in pts])
                y = np.array([v for _, v in pts])
                ax.plot(x, y, linewidth=1.0, alpha=0.22, label="_nolegend_")

        x, mean, ci, n = aggregate(seed_runs)
        if len(x) == 0:
            for seed, pts in seed_runs.items():
                xx = np.array([s for s, _ in pts])
                yy = np.array([v for _, v in pts])
                ax.plot(xx, yy, marker="o", linewidth=1.8, label=f"{method} seed {seed}")
            continue

        label = method if n == 1 else f"{method} (n={n})"
        line, = ax.plot(x, mean, marker="o", markersize=3.5, linewidth=2.0, label=label)

        if n > 1 and not args.no_ci:
            ax.fill_between(
                x, mean - ci, mean + ci,
                alpha=0.18, color=line.get_color(), linewidth=0
            )

    ax.set_xlabel("Target environment steps")
    ax.set_ylabel(pretty_metric(args.metric))
    ax.set_title(args.title or f"D{args.difficulty} target learning curve")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    if args.metric in {
        "success_rate", "fatal_crash_rate", "goal_found_rate", "mean_coverage_ratio"
    }:
        ax.set_ylim(-0.02, 1.02)

    ax.set_xlim(left=0)
    if args.max_steps is not None:
        ax.set_xlim(0, args.max_steps)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()
