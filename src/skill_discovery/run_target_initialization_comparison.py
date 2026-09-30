"""Compare four target-task initializations and upload aligned curves to W&B.

The four methods are:
1. the MAPPO checkpoint used to collect the offline data, continued with PPO;
2. behavior-cloning initialization, continued with the same RLlib PPO baseline;
3. frozen source HiSSD evaluated zero-shot throughout the target budget;
4. source HiSSD fine-tuned online with the HiSSD PPO support path.

Runs are executed sequentially to avoid Ray/custom-PPO GPU contention. Their
training step zero, budget, evaluation schedule, and evaluation seeds are
aligned, so they remain directly comparable as learning curves.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_SCRIPT = PROJECT_ROOT / "src/skill_discovery/finetune_drone_baseline_online.py"
HISSD_SCRIPT = PROJECT_ROOT / "src/skill_discovery/finetune_hissd_drone_online.py"
ANALYZE_SCRIPT = PROJECT_ROOT / "src/skill_discovery/analyze_learning_efficiency.py"
PLOT_SCRIPT = PROJECT_ROOT / "src/skill_discovery/plot_skill_control_curves.py"

DEFAULT_MAPPO_CHECKPOINT = (
    PROJECT_ROOT / "src/train/drone_mappo_coverage60_checkpoints/checkpoint_07800"
)
DEFAULT_BC_CHECKPOINT = (
    PROJECT_ROOT
    / "src/skill_discovery/checkpoints/bc_drone_d12_t34_cp7800/drone_bc_best.pt"
)
DEFAULT_HISSD_CHECKPOINT = (
    PROJECT_ROOT
    / "src/skill_discovery/checkpoints/"
    "hissd_drone_d12_t34_realized_stable_v2/hissd_best.pt"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "src/skill_discovery/outputs/learning_efficiency/drone_d12_t34/"
    "target_initialization_comparison"
)
METHODS = (
    "mappo_checkpoint",
    "behavior_cloning",
    "hissd_zero_shot",
    "hissd_finetuned",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--difficulty", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--mappo-checkpoint", type=Path, default=DEFAULT_MAPPO_CHECKPOINT)
    parser.add_argument("--bc-checkpoint", type=Path, default=DEFAULT_BC_CHECKPOINT)
    parser.add_argument("--hissd-checkpoint", type=Path, default=DEFAULT_HISSD_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--joint-step-budget", type=int, default=500_000)
    parser.add_argument("--eval-every-joint-steps", type=int, default=25_000)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--eval-seed-base", type=int, default=100_000_000)
    parser.add_argument("--train-batch-joint-steps", type=int, default=5_000)
    parser.add_argument("--success-threshold", type=float, default=0.6)
    parser.add_argument("--num-env-runners", type=int, default=4)
    parser.add_argument("--num-gpus", type=float, default=1.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--wandb-project", default="HeMAC-Target-Learning-Efficiency")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars and print child output directly.",
    )
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Skip training and upload already existing curve files.",
    )
    args = parser.parse_args()
    positive = (
        "difficulty",
        "joint_step_budget",
        "eval_every_joint_steps",
        "eval_episodes",
        "train_batch_joint_steps",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.joint_step_budget % args.train_batch_joint_steps:
        raise ValueError("The budget must be divisible by --train-batch-joint-steps.")
    if args.eval_every_joint_steps % args.train_batch_joint_steps:
        raise ValueError(
            "The evaluation interval must be divisible by --train-batch-joint-steps."
        )
    if not 0.0 <= args.success_threshold <= 1.0:
        raise ValueError("--success-threshold must be in [0, 1].")
    return args


def curve_path(root: Path, method: str, difficulty: int, seed: int) -> Path:
    return root / method / f"d{difficulty}_seed{seed}" / f"learning_curve_seed_{seed}.json"


def progress_write(message: str) -> None:
    if tqdm is not None:
        tqdm.write(message)
    else:
        print(message, flush=True)


def run_command(
    command: list[str],
    log_path: Path,
    *,
    dry_run: bool,
    progress_label: str | None = None,
    progress_total: int | None = None,
) -> None:
    progress_write(f"$ {shlex.join(command)}")
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.setdefault("PYTHONUNBUFFERED", "1")
    environment.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    environment.setdefault("MPLCONFIGDIR", "/tmp/hemac_matplotlib_cache")
    recent_lines: deque[str] = deque(maxlen=40)
    step_progress = (
        tqdm(
            total=progress_total,
            desc=progress_label,
            unit="step",
            dynamic_ncols=True,
            leave=True,
            position=1,
        )
        if tqdm is not None and progress_label is not None and progress_total is not None
        else None
    )
    displayed_steps = 0
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            clean = line.replace("\r", "").strip()
            if not clean:
                continue
            recent_lines.append(clean)
            step_match = re.search(r"\bsteps=(\d+)\b", clean)
            if step_match is not None and step_progress is not None:
                current_steps = min(int(step_match.group(1)), progress_total)
                if current_steps > displayed_steps:
                    step_progress.update(current_steps - displayed_steps)
                    displayed_steps = current_steps
            should_display = (
                step_progress is None
                or clean.startswith(("BASELINE", "EVAL", "Saved", "TEST"))
                or "Traceback" in clean
                or "Error" in clean
                or "WARNING" in clean
            )
            if should_display:
                progress_write(clean)
        return_code = process.wait()
    if step_progress is not None:
        if return_code == 0 and displayed_steps < progress_total:
            step_progress.update(progress_total - displayed_steps)
        step_progress.close()
    if return_code:
        progress_write(f"[{progress_label or 'command'} failed] Last log lines:")
        for line in recent_lines:
            progress_write(line)
        raise subprocess.CalledProcessError(return_code, command)


def baseline_command(args: argparse.Namespace, method: str, output: Path) -> list[str]:
    initialization = "mappo" if method == "mappo_checkpoint" else "bc"
    return [
        sys.executable,
        str(BASELINE_SCRIPT),
        "--initialization",
        initialization,
        "--method-name",
        method,
        "--mappo-checkpoint",
        str(args.mappo_checkpoint),
        "--bc-checkpoint",
        str(args.bc_checkpoint),
        "--output-dir",
        str(output.parent),
        "--learning-curve-output",
        str(output),
        "--difficulty",
        str(args.difficulty),
        "--iterations",
        str(args.joint_step_budget // args.train_batch_joint_steps),
        "--train-batch-size",
        str(args.train_batch_joint_steps),
        "--minibatch-size",
        str(min(1024, args.train_batch_joint_steps)),
        "--eval-every",
        str(args.eval_every_joint_steps // args.train_batch_joint_steps),
        "--eval-episodes",
        str(args.eval_episodes),
        "--eval-seed-base",
        str(args.eval_seed_base),
        "--num-env-runners",
        str(args.num_env_runners),
        "--num-gpus",
        str(args.num_gpus),
        "--seed",
        str(args.seed),
    ]


def hissd_command(args: argparse.Namespace, output: Path) -> list[str]:
    return [
        sys.executable,
        str(HISSD_SCRIPT),
        "--hissd-checkpoint",
        str(args.hissd_checkpoint),
        "--mappo-checkpoint",
        str(args.mappo_checkpoint),
        "--output-dir",
        str(output.parent),
        "--learning-curve-output",
        str(output),
        "--method-name",
        "hissd_finetuned",
        "--skill-mode",
        "full",
        "--difficulty",
        str(args.difficulty),
        "--joint-step-budget",
        str(args.joint_step_budget),
        "--train-batch-joint-steps",
        str(args.train_batch_joint_steps),
        "--eval-every-joint-steps",
        str(args.eval_every_joint_steps),
        "--eval-episodes",
        str(args.eval_episodes),
        "--eval-seed-base",
        str(args.eval_seed_base),
        "--base-unfreeze-after-joint-steps",
        str(args.train_batch_joint_steps),
        "--encoder-unfreeze-after-joint-steps",
        str(3 * args.train_batch_joint_steps),
        "--skill-prior-init",
        "0.10",
        "--skill-prior-max",
        "0.25",
        "--credit-assignment",
        "agent",
        "--no-use-pretrained-value",
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--deterministic",
    ]


def synthesize_zero_shot_curve(
    finetuned_path: Path,
    output: Path,
    *,
    budget: int,
    interval: int,
) -> None:
    payload = json.loads(finetuned_path.read_text(encoding="utf-8"))
    baseline = min(payload["points"], key=lambda point: int(point["joint_env_steps"]))
    keep = {
        key: value
        for key, value in baseline.items()
        if not key.startswith("ppo_")
        and not key.startswith("train_")
        and not key.startswith("best_checkpoint_")
    }
    points = []
    for step in range(0, budget + 1, interval):
        point = dict(keep)
        point.update(
            method="hissd_zero_shot",
            joint_env_steps=step,
            iteration=0,
            frozen_zero_shot=1.0,
        )
        points.append(point)
    metadata = dict(payload.get("metadata", {}))
    metadata.update(
        curve_method="hissd_zero_shot",
        skill_mode="full",
        frozen_zero_shot=True,
        requested_joint_step_budget=budget,
        evaluation_joint_step_interval=interval,
        note="One fixed zero-shot evaluation repeated as a horizontal reference.",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "format_version": payload["format_version"],
                "step_unit": payload["step_unit"],
                "metadata": metadata,
                "points": points,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def upload_curve_to_wandb(
    path: Path,
    args: argparse.Namespace,
    *,
    method: str,
    group: str,
) -> None:
    if args.wandb_mode == "disabled":
        return
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb is required unless --wandb-mode disabled is used.") from exc

    payload = json.loads(path.read_text(encoding="utf-8"))
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=group,
        name=f"{method}-d{args.difficulty}-seed{args.seed}",
        job_type=method,
        mode=args.wandb_mode,
        config={
            "method": method,
            "difficulty": args.difficulty,
            "seed": args.seed,
            "joint_step_budget": args.joint_step_budget,
            "eval_every_joint_steps": args.eval_every_joint_steps,
            "eval_episodes": args.eval_episodes,
            "eval_seed_base": args.eval_seed_base,
            "mappo_checkpoint": str(args.mappo_checkpoint),
            "bc_checkpoint": str(args.bc_checkpoint),
            "hissd_checkpoint": str(args.hissd_checkpoint),
            "curve_metadata": payload.get("metadata", {}),
        },
        reinit=True,
    )
    run.define_metric("joint_env_steps")
    run.define_metric("eval/*", step_metric="joint_env_steps")
    metric_keys = {
        "success_rate": "eval/success_rate",
        "goal_found_rate": "eval/goal_found_rate",
        "fatal_crash_rate": "eval/fatal_crash_rate",
        "drone_crash_rate": "eval/drone_crash_rate",
        "mean_coverage_ratio": "eval/coverage_ratio",
        "mean_episode_return": "eval/episode_return",
    }
    for point in sorted(payload["points"], key=lambda item: item["joint_env_steps"]):
        actual_step = int(point["joint_env_steps"])
        display_step = int(point.get("scheduled_joint_env_steps", actual_step))
        row: dict[str, Any] = {
            "joint_env_steps": display_step,
            "actual_joint_env_steps": actual_step,
        }
        for source, target in metric_keys.items():
            if source in point and math.isfinite(float(point[source])):
                row[target] = float(point[source])
        run.log(row)
    artifact = wandb.Artifact(
        f"{method}-d{args.difficulty}-seed{args.seed}-curve",
        type="learning-curve",
    )
    artifact.add_file(str(path), name=path.name)
    run.log_artifact(artifact)
    run.finish()


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.expanduser().resolve()
    args.mappo_checkpoint = args.mappo_checkpoint.expanduser().resolve()
    args.bc_checkpoint = args.bc_checkpoint.expanduser().resolve()
    args.hissd_checkpoint = args.hissd_checkpoint.expanduser().resolve()
    for path in (args.mappo_checkpoint, args.bc_checkpoint, args.hissd_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_root.mkdir(parents=True, exist_ok=True)
    group = args.wandb_group or (
        f"d{args.difficulty}-seed{args.seed}-"
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )

    paths = {
        method: curve_path(args.output_root, method, args.difficulty, args.seed)
        for method in METHODS
    }
    overall_progress = (
        tqdm(
            total=2 * len(args.methods) + 2,
            desc=f"D{args.difficulty} four-method comparison",
            unit="job",
            dynamic_ncols=True,
            position=0,
        )
        if tqdm is not None and not args.no_progress and not args.dry_run
        else None
    )
    if args.force and not args.upload_only and not args.dry_run:
        for path in paths.values():
            path.unlink(missing_ok=True)

    if not args.upload_only:
        for method in args.methods:
            if method == "hissd_zero_shot":
                continue
            output = paths[method]
            if output.is_file() and not args.force:
                progress_write(f"[SKIP] Existing curve: {output}")
                if overall_progress is not None:
                    overall_progress.update(1)
                continue
            command = (
                hissd_command(args, output)
                if method == "hissd_finetuned"
                else baseline_command(args, method, output)
            )
            run_command(
                command,
                args.output_root / "logs" / f"{method}.log",
                dry_run=args.dry_run,
                progress_label=(None if args.no_progress else method),
                progress_total=args.joint_step_budget,
            )
            if overall_progress is not None:
                overall_progress.update(1)

        if "hissd_zero_shot" in args.methods and not args.dry_run:
            source = paths["hissd_finetuned"]
            if not source.is_file():
                raise FileNotFoundError(
                    "HiSSD fine-tuned curve is needed to extract the matched "
                    f"zero-shot baseline: {source}"
                )
            synthesize_zero_shot_curve(
                source,
                paths["hissd_zero_shot"],
                budget=args.joint_step_budget,
                interval=args.eval_every_joint_steps,
            )
            progress_write("[DONE] Synthesized frozen HiSSD zero-shot reference.")
            if overall_progress is not None:
                overall_progress.update(1)

    selected_paths = [paths[method] for method in args.methods]
    if args.dry_run:
        return
    for path in selected_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing curve for comparison: {path}")

    comparison = args.output_root / "comparison.json"
    plot = args.output_root / "comparison.png"
    run_command(
        [
            sys.executable,
            str(ANALYZE_SCRIPT),
            "analyze",
            "--curves",
            *(str(path) for path in selected_paths),
            "--threshold",
            f"{args.difficulty}={args.success_threshold}",
            "--budget",
            str(args.joint_step_budget),
            "--output",
            str(comparison),
        ],
        args.output_root / "logs/analyze.log",
        dry_run=False,
    )
    if overall_progress is not None:
        overall_progress.update(1)
    run_command(
        [
            sys.executable,
            str(PLOT_SCRIPT),
            "--curves",
            *(str(path) for path in selected_paths),
            "--output",
            str(plot),
        ],
        args.output_root / "logs/plot.log",
        dry_run=False,
    )
    if overall_progress is not None:
        overall_progress.update(1)
    for method, path in zip(args.methods, selected_paths):
        upload_curve_to_wandb(path, args, method=method, group=group)
        if overall_progress is not None:
            overall_progress.update(1)
    if overall_progress is not None:
        overall_progress.close()
    progress_write(f"Comparison: {comparison}")
    progress_write(f"Plot: {plot}")
    progress_write(f"W&B project/group: {args.wandb_project}/{group}")


if __name__ == "__main__":
    main()
