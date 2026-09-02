"""Run target online-finetuning ablations and compare learning efficiency."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINETUNE_SCRIPT = PROJECT_ROOT / "src/skill_discovery/finetune_hissd_online.py"
ANALYZE_SCRIPT = PROJECT_ROOT / "src/skill_discovery/analyze_learning_efficiency.py"
DEFAULT_CHECKPOINT_ROOT = (
    PROJECT_ROOT / "src/skill_discovery/checkpoints/hissd_ablation_multiseed"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "src/skill_discovery/outputs/learning_efficiency/hissd_ablation"
)
DEFAULT_MAPPO_CHECKPOINT = (
    PROJECT_ROOT / "src/train/mappo_checkpoints/checkpoint_19000"
)
ABLATIONS = (
    "full",
    "no_descriptor",
    "no_task_contrast",
    "no_task_auxiliary",
)
METHOD_NAMES = {
    "full": "hissd_full",
    "no_descriptor": "hissd_no_descriptor",
    "no_task_contrast": "hissd_no_task_classifier",
    "no_task_auxiliary": "hissd_no_task_auxiliary",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=(2026, 2027, 2028, 2029, 2030)
    )
    parser.add_argument(
        "--ablations", nargs="+", choices=ABLATIONS, default=ABLATIONS
    )
    parser.add_argument(
        "--difficulties", type=int, nargs="+", choices=(4, 5, 6), default=(4, 5, 6)
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("online", "analyze"),
        default=("online", "analyze"),
    )
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--mappo-checkpoint", type=Path, default=DEFAULT_MAPPO_CHECKPOINT)
    parser.add_argument(
        "--source-checkpoint-name",
        choices=("hissd_last.pt", "hissd_best.pt", "hissd_best_task.pt"),
        default="hissd_last.pt",
        help="Use hissd_last.pt for fixed-epoch, seed-matched ablation fairness.",
    )
    parser.add_argument("--iterations-per-stage", type=int, default=40)
    parser.add_argument("--episodes-per-iteration", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--thresholds", nargs="+", default=("4=0.4", "5=0.2", "6=0.1")
    )
    parser.add_argument("--budget", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "iterations_per_stage",
        "episodes_per_iteration",
        "eval_every",
        "eval_episodes",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.budget is not None and args.budget <= 0:
        raise ValueError("--budget must be positive.")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must not contain duplicates.")
    if len(set(args.ablations)) != len(args.ablations):
        raise ValueError("--ablations must not contain duplicates.")
    if len(set(args.difficulties)) != len(args.difficulties):
        raise ValueError("--difficulties must not contain duplicates.")


def run_directory(args: argparse.Namespace, ablation: str, difficulty: int, seed: int) -> Path:
    return args.output_root / ablation / f"d{difficulty}_seed{seed}"


def curve_path(args: argparse.Namespace, ablation: str, difficulty: int, seed: int) -> Path:
    return run_directory(args, ablation, difficulty, seed) / (
        f"learning_curve_seed_{seed}.json"
    )


def source_checkpoint(args: argparse.Namespace, ablation: str, seed: int) -> Path:
    return (
        args.checkpoint_root
        / f"seed_{seed}"
        / ablation
        / args.source_checkpoint_name
    )


def curve_is_complete(path: Path, expected_iteration: int) -> bool:
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return max(
        (int(point.get("iteration", -1)) for point in payload.get("points", ())),
        default=-1,
    ) >= expected_iteration


def finetune_command(
    args: argparse.Namespace,
    ablation: str,
    difficulty: int,
    seed: int,
) -> list[str]:
    return [
        sys.executable,
        str(FINETUNE_SCRIPT),
        "--hissd-checkpoint",
        str(source_checkpoint(args, ablation, seed)),
        "--mappo-checkpoint",
        str(args.mappo_checkpoint),
        "--method-name",
        METHOD_NAMES[ablation],
        "--start-stage",
        str(difficulty),
        "--end-stage",
        str(difficulty),
        "--iterations-per-stage",
        str(args.iterations_per_stage),
        "--episodes-per-iteration",
        str(args.episodes_per_iteration),
        "--eval-every",
        str(args.eval_every),
        "--eval-episodes",
        str(args.eval_episodes),
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--output-dir",
        str(run_directory(args, ablation, difficulty, seed)),
        "--learning-curve-output",
        str(curve_path(args, ablation, difficulty, seed)),
    ]


def analyze_command(args: argparse.Namespace, curves: list[Path]) -> list[str]:
    command = [
        sys.executable,
        str(ANALYZE_SCRIPT),
        "analyze",
        "--curves",
        *(str(path) for path in curves),
    ]
    for threshold in args.thresholds:
        command.extend(("--threshold", threshold))
    if args.budget is not None:
        command.extend(("--budget", str(args.budget)))
    command.extend(
        (
            "--output",
            str(args.output_root / "learning_efficiency_ablation.json"),
        )
    )
    return command


def execute(command: list[str], dry_run: bool) -> None:
    print(f"$ {shlex.join(command)}", flush=True)
    if dry_run:
        return
    environment = os.environ.copy()
    environment.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


def main() -> None:
    args = parse_args()
    args.checkpoint_root = args.checkpoint_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.mappo_checkpoint = args.mappo_checkpoint.expanduser().resolve()

    curves = [
        curve_path(args, ablation, difficulty, seed)
        for ablation in args.ablations
        for difficulty in args.difficulties
        for seed in args.seeds
    ]
    if "online" in args.stages:
        for ablation in args.ablations:
            for difficulty in args.difficulties:
                for seed in args.seeds:
                    checkpoint = source_checkpoint(args, ablation, seed)
                    if not checkpoint.is_file():
                        raise FileNotFoundError(f"Source checkpoint not found: {checkpoint}")
                    output = curve_path(args, ablation, difficulty, seed)
                    if not args.force and curve_is_complete(
                        output, args.iterations_per_stage
                    ):
                        print(f"SKIP complete curve: {output}")
                        continue
                    execute(
                        finetune_command(args, ablation, difficulty, seed),
                        args.dry_run,
                    )

    if "analyze" in args.stages:
        if not args.dry_run:
            missing = [path for path in curves if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"Cannot analyze before {len(missing)} curves are generated; "
                    f"first missing path: {missing[0]}"
                )
        execute(analyze_command(args, curves), args.dry_run)


if __name__ == "__main__":
    main()
