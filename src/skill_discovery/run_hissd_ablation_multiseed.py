"""Train and evaluate HiSSD task-auxiliary ablations over multiple seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = PROJECT_ROOT / "src/skill_discovery/train_hissd.py"
EVALUATE_SCRIPT = PROJECT_ROOT / "src/skill_discovery/evaluate_hissd.py"
DEFAULT_CHECKPOINT_ROOT = (
    PROJECT_ROOT / "src/skill_discovery/checkpoints/hissd_ablation_multiseed"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "src/skill_discovery/outputs/hissd_ablation_multiseed"
)
ABLATIONS = (
    "full",
    "no_descriptor",
    "no_task_contrast",
    "no_task_auxiliary",
)
METRICS = (
    "success",
    "goal_found",
    "drone_goal_found",
    "fatal_crash",
    "drone_crash",
    "observer_crash",
    "coverage_ratio",
    "cycles",
)
SCOPES = {
    "source": (1, 2, 3),
    "target": (4, 5, 6),
    "all": (1, 2, 3, 4, 5, 6),
}


def parse_args() -> argparse.Namespace:
    """Parse the shared training and evaluation settings for the suite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=(2026, 2027, 2028))
    parser.add_argument(
        "--ablations",
        nargs="+",
        choices=ABLATIONS,
        default=ABLATIONS,
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("train", "evaluate", "summarize"),
        default=("train", "evaluate", "summarize"),
    )
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--bc-checkpoint", type=Path)
    parser.add_argument("--mappo-checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--eval-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--difficulties",
        type=int,
        nargs="+",
        choices=range(1, 7),
        default=(1, 2, 3, 4, 5, 6),
    )
    parser.add_argument("--base-seed", type=int, default=200_000)
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable one TensorBoard directory per run; disabled by default.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run stages even when their expected output already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and paths without training, evaluation, or writes.",
    )
    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    """Reject settings that would make runs incomparable or invalid."""
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must not contain duplicates.")
    if len(set(args.ablations)) != len(args.ablations):
        raise ValueError("--ablations must not contain duplicates.")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("--epochs and --batch-size must be positive.")
    if args.sequence_length <= 0 or args.stride <= 0:
        raise ValueError("--sequence-length and --stride must be positive.")
    if args.num_workers < 0 or args.episodes <= 0:
        raise ValueError("--num-workers must be nonnegative and --episodes positive.")
    if "summarize" in args.stages and "full" not in args.ablations:
        raise ValueError("Summary comparisons require the full ablation baseline.")
    if "summarize" in args.stages and set(args.difficulties) != set(range(1, 7)):
        raise ValueError("Summary scopes require --difficulties 1 2 3 4 5 6.")


def run_path(root: Path, seed: int, ablation: str) -> Path:
    """Return the isolated directory for one training run."""
    return root / f"seed_{seed}" / ablation


def evaluation_path(root: Path, seed: int, ablation: str) -> Path:
    """Return the evaluation JSON path for one trained model."""
    return root / f"seed_{seed}" / f"{ablation}.json"


def train_command(args: argparse.Namespace, seed: int, ablation: str) -> list[str]:
    """Build one fixed-duration training command."""
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--ablation",
        ablation,
        "--seed",
        str(seed),
        "--epochs",
        str(args.epochs),
        "--early-stopping-patience",
        "0",
        "--batch-size",
        str(args.batch_size),
        "--sequence-length",
        str(args.sequence_length),
        "--stride",
        str(args.stride),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--output-dir",
        str(run_path(args.checkpoint_root, seed, ablation)),
    ]
    if not args.tensorboard:
        command.append("--no-tensorboard")
    for flag, value in (
        ("--manifest", args.manifest),
        ("--data-root", args.data_root),
        ("--bc-checkpoint", args.bc_checkpoint),
    ):
        if value is not None:
            command.extend((flag, str(value)))
    return command


def evaluate_command(args: argparse.Namespace, seed: int, ablation: str) -> list[str]:
    """Build one evaluation command using the final fixed-epoch checkpoint."""
    command = [
        sys.executable,
        str(EVALUATE_SCRIPT),
        "--hissd-checkpoint",
        str(run_path(args.checkpoint_root, seed, ablation) / "hissd_last.pt"),
        "--difficulties",
        *(str(value) for value in args.difficulties),
        "--episodes",
        str(args.episodes),
        "--base-seed",
        str(args.base_seed),
        "--device",
        args.eval_device,
        "--output-json",
        str(evaluation_path(args.output_root, seed, ablation)),
        "--quiet",
    ]
    if args.mappo_checkpoint is not None:
        command.extend(("--mappo-checkpoint", str(args.mappo_checkpoint)))
    if args.bc_checkpoint is not None:
        command.extend(("--bc-checkpoint", str(args.bc_checkpoint)))
    return command


def execute(command: list[str], *, dry_run: bool) -> None:
    """Print and optionally execute one child process."""
    print(f"$ {shlex.join(command)}", flush=True)
    if dry_run:
        return
    environment = os.environ.copy()
    environment.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


def load_episode_metrics(path: Path) -> list[dict[str, Any]]:
    """Load the HiSSD controller's episode records from one evaluation."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    episodes = [
        episode
        for episode in payload.get("episodes", ())
        if episode.get("controller") == "hissd"
    ]
    if not episodes:
        raise ValueError(f"Evaluation has no HiSSD episodes: {path}")
    return episodes


def scope_metrics(
    episodes: list[dict[str, Any]], difficulties: tuple[int, ...]
) -> dict[str, float]:
    """Average all requested metrics over one difficulty scope."""
    selected = [
        episode for episode in episodes if int(episode["difficulty"]) in difficulties
    ]
    if not selected:
        return {metric: math.nan for metric in METRICS}
    return {
        metric: float(np.mean([float(episode[metric]) for episode in selected]))
        for metric in METRICS
    }


def confidence_interval(values: np.ndarray) -> tuple[float, float]:
    """Return a training-seed t interval for one scalar metric."""
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return math.nan, math.nan
    standard_error = stats.sem(finite)
    if not np.isfinite(standard_error) or standard_error == 0.0:
        mean = float(finite.mean())
        return mean, mean
    low, high = stats.t.interval(
        0.95,
        df=finite.size - 1,
        loc=float(finite.mean()),
        scale=float(standard_error),
    )
    return float(low), float(high)


def summarize_results(args: argparse.Namespace) -> dict[str, Any]:
    """Aggregate metrics with the training seed as the statistical unit."""
    per_run: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for seed in args.seeds:
        seed_key = str(seed)
        per_run[seed_key] = {}
        for ablation in args.ablations:
            path = evaluation_path(args.output_root, seed, ablation)
            if not path.is_file():
                raise FileNotFoundError(f"Missing evaluation result: {path}")
            episodes = load_episode_metrics(path)
            per_run[seed_key][ablation] = {
                scope: scope_metrics(episodes, difficulties)
                for scope, difficulties in SCOPES.items()
            }

    aggregates: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for ablation in args.ablations:
        aggregates[ablation] = {}
        for scope in SCOPES:
            aggregates[ablation][scope] = {}
            for metric in METRICS:
                values = np.asarray(
                    [
                        per_run[str(seed)][ablation][scope][metric]
                        for seed in args.seeds
                    ],
                    dtype=np.float64,
                )
                low, high = confidence_interval(values)
                metric_summary: dict[str, Any] = {
                    "values": values.tolist(),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                    "ci95": [low, high],
                }
                if ablation != "full":
                    full_values = np.asarray(
                        [
                            per_run[str(seed)]["full"][scope][metric]
                            for seed in args.seeds
                        ],
                        dtype=np.float64,
                    )
                    differences = values - full_values
                    diff_low, diff_high = confidence_interval(differences)
                    if differences.size < 2 or np.allclose(differences, 0.0):
                        p_value = 1.0
                    elif np.allclose(differences, differences[0]):
                        p_value = 0.0
                    else:
                        p_value = float(stats.ttest_rel(values, full_values).pvalue)
                    metric_summary["vs_full"] = {
                        "paired_differences": differences.tolist(),
                        "mean_difference": float(differences.mean()),
                        "ci95": [diff_low, diff_high],
                        "paired_t_pvalue": p_value,
                    }
                aggregates[ablation][scope][metric] = metric_summary
                rows.append(
                    {
                        "ablation": ablation,
                        "scope": scope,
                        "metric": metric,
                        "mean": metric_summary["mean"],
                        "std": metric_summary["std"],
                        "ci95_low": low,
                        "ci95_high": high,
                        "vs_full_mean_difference": metric_summary.get(
                            "vs_full", {}
                        ).get("mean_difference"),
                        "vs_full_pvalue": metric_summary.get("vs_full", {}).get(
                            "paired_t_pvalue"
                        ),
                    }
                )

    payload = {
        "statistical_unit": "training_seed",
        "seeds": list(args.seeds),
        "ablations": list(args.ablations),
        "epochs": args.epochs,
        "checkpoint_selection": "hissd_last.pt at fixed epoch count",
        "evaluation": {
            "difficulties": list(args.difficulties),
            "episodes_per_difficulty": args.episodes,
            "base_seed": args.base_seed,
        },
        "per_run": per_run,
        "aggregates": aggregates,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "multiseed_summary.json"
    summary_path.write_text(
        json.dumps(payload, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    csv_path = args.output_root / "multiseed_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Summary JSON: {summary_path}")
    print(f"Summary CSV:  {csv_path}")
    for ablation in args.ablations:
        success = aggregates[ablation]["all"]["success"]
        message = (
            f"{ablation:22s} success={success['mean']:.3f} "
            f"std={success['std']:.3f}"
        )
        if ablation != "full":
            comparison = success["vs_full"]
            message += (
                f" diff={comparison['mean_difference']:+.3f} "
                f"p={comparison['paired_t_pvalue']:.4f}"
            )
        print(message)
    return payload


def main() -> None:
    """Run requested suite stages in deterministic seed/mode order."""
    args = parse_args()
    args.checkpoint_root = args.checkpoint_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()

    if "train" in args.stages:
        for seed in args.seeds:
            for ablation in args.ablations:
                checkpoint = run_path(
                    args.checkpoint_root, seed, ablation
                ) / "hissd_last.pt"
                if checkpoint.is_file() and not args.force:
                    print(f"[skip train] {checkpoint}")
                    continue
                execute(train_command(args, seed, ablation), dry_run=args.dry_run)

    if "evaluate" in args.stages:
        for seed in args.seeds:
            for ablation in args.ablations:
                output = evaluation_path(args.output_root, seed, ablation)
                if output.is_file() and not args.force:
                    print(f"[skip evaluate] {output}")
                    continue
                checkpoint = run_path(
                    args.checkpoint_root, seed, ablation
                ) / "hissd_last.pt"
                if not checkpoint.is_file() and not args.dry_run:
                    raise FileNotFoundError(f"Missing fixed-epoch checkpoint: {checkpoint}")
                if not args.dry_run:
                    output.parent.mkdir(parents=True, exist_ok=True)
                execute(evaluate_command(args, seed, ablation), dry_run=args.dry_run)

    if "summarize" in args.stages:
        if args.dry_run:
            print(f"[dry-run summarize] {args.output_root / 'multiseed_summary.json'}")
        else:
            summarize_results(args)


if __name__ == "__main__":
    main()
