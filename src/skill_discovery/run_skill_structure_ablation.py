"""Run the four-way HiSSD skill-structure learning-efficiency ablation."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = PROJECT_ROOT / "src/skill_discovery/train_hissd.py"
ADAPT_SCRIPT = PROJECT_ROOT / "src/skill_discovery/adapt_hissd_target.py"
ONLINE_SCRIPT = (
    PROJECT_ROOT / "src/skill_discovery/finetune_hissd_drone_online.py"
)
ANALYZE_SCRIPT = (
    PROJECT_ROOT / "src/skill_discovery/analyze_learning_efficiency.py"
)
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "src/skill_discovery/offline_data_drone_d12_t34_cp7800/"
    "drone_task_dataset_splits.json"
)
DEFAULT_BC_CHECKPOINT = (
    PROJECT_ROOT
    / "src/skill_discovery/checkpoints/bc_drone_d12_t34_cp7800/"
    "drone_bc_best.pt"
)
DEFAULT_MAPPO_CHECKPOINT = (
    PROJECT_ROOT / "src/train/drone_mappo_coverage60_checkpoints/checkpoint_07800"
)
DEFAULT_CHECKPOINT_ROOT = (
    PROJECT_ROOT / "src/skill_discovery/checkpoints/hissd_skill_structure_ablation"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "src/skill_discovery/outputs/learning_efficiency/drone_d12_t34/"
    "skill_structure_ablation"
)
SKILL_STRUCTURES = ("common_only", "task_only", "split", "shared")
MODEL_VARIANTS = ("source", "adapted")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("train", "adapt", "online", "analyze"),
        default=("train", "adapt", "online", "analyze"),
    )
    parser.add_argument(
        "--structures", nargs="+", choices=SKILL_STRUCTURES, default=SKILL_STRUCTURES
    )
    parser.add_argument(
        "--variants", nargs="+", choices=MODEL_VARIANTS, default=MODEL_VARIANTS
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=(2026, 2027, 2028, 2029, 2030)
    )
    parser.add_argument("--difficulties", type=int, nargs="+", default=(3, 4))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--bc-checkpoint", type=Path, default=DEFAULT_BC_CHECKPOINT)
    parser.add_argument(
        "--mappo-checkpoint", type=Path, default=DEFAULT_MAPPO_CHECKPOINT
    )
    parser.add_argument(
        "--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-epochs", type=int, default=70)
    parser.add_argument("--adapt-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--dataset-cache-size",
        type=int,
        default=16,
        help="Number of mmap-backed episode files cached by each DataLoader worker.",
    )
    parser.add_argument(
        "--parallel-train-jobs",
        type=int,
        default=1,
        help="Concurrent source learners. Keep at 1 on GPUs with limited VRAM.",
    )
    parser.add_argument(
        "--parallel-adapt-jobs",
        type=int,
        default=1,
        help="Concurrent adaptation learners. Keep at 1 on GPUs with limited VRAM.",
    )
    parser.add_argument(
        "--parallel-online-jobs",
        type=int,
        default=2,
        help="Concurrent CPU-heavy online PPO experiments.",
    )
    parser.add_argument(
        "--torch-cpu-threads",
        type=int,
        default=2,
        help="CPU threads available to PyTorch inside each child process.",
    )
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--episodes-per-iteration", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument(
        "--checkpoint-selection",
        choices=("last", "best"),
        default="last",
        help="Use fixed-epoch last checkpoints by default for a fair ablation.",
    )
    parser.add_argument("--thresholds", nargs="+", default=("3=0.6", "4=0.5"))
    parser.add_argument("--budget", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "source_epochs",
        "adapt_epochs",
        "batch_size",
        "sequence_length",
        "iterations",
        "episodes_per_iteration",
        "eval_every",
        "eval_episodes",
        "dataset_cache_size",
        "parallel_train_jobs",
        "parallel_adapt_jobs",
        "parallel_online_jobs",
        "torch_cpu_threads",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    if args.budget is not None and args.budget <= 0:
        raise ValueError("--budget must be positive.")
    for name in ("structures", "variants", "seeds", "difficulties"):
        values = getattr(args, name)
        if len(values) != len(set(values)):
            raise ValueError(f"--{name} must not contain duplicates.")


def source_dir(args: argparse.Namespace, structure: str, seed: int) -> Path:
    return args.checkpoint_root / structure / f"seed_{seed}" / "source"


def adapted_dir(args: argparse.Namespace, structure: str, seed: int) -> Path:
    return args.checkpoint_root / structure / f"seed_{seed}" / "adapted"


def checkpoint_path(
    args: argparse.Namespace, structure: str, seed: int, variant: str
) -> Path:
    if variant == "source":
        name = f"hissd_{args.checkpoint_selection}.pt"
        return source_dir(args, structure, seed) / name
    name = f"hissd_adapted_{args.checkpoint_selection}.pt"
    return adapted_dir(args, structure, seed) / name


def progress_checkpoint_path(
    args: argparse.Namespace, structure: str, seed: int, variant: str
) -> Path:
    """Return the last-epoch checkpoint used to detect or resume partial jobs."""
    if variant == "source":
        return source_dir(args, structure, seed) / "hissd_last.pt"
    return adapted_dir(args, structure, seed) / "hissd_adapted_last.pt"


def checkpoint_epoch(path: Path) -> int:
    """Read a checkpoint epoch, treating missing or invalid files as incomplete."""
    if not path.is_file():
        return 0
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return int(payload.get("epoch", 0))
    except Exception as error:
        print(f"WARNING: cannot inspect checkpoint {path}: {error}", flush=True)
        return 0


def online_dir(
    args: argparse.Namespace,
    structure: str,
    variant: str,
    difficulty: int,
    seed: int,
) -> Path:
    return args.output_root / variant / structure / f"d{difficulty}_seed{seed}"


def curve_path(
    args: argparse.Namespace,
    structure: str,
    variant: str,
    difficulty: int,
    seed: int,
) -> Path:
    return online_dir(args, structure, variant, difficulty, seed) / (
        f"learning_curve_seed_{seed}.json"
    )


def curve_is_complete(path: Path, expected_iteration: int) -> bool:
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return max(
        (int(point.get("iteration", -1)) for point in payload.get("points", ())),
        default=-1,
    ) >= expected_iteration


def train_command(
    args: argparse.Namespace,
    structure: str,
    seed: int,
    resume_checkpoint: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--manifest",
        str(args.manifest),
        "--bc-checkpoint",
        str(args.bc_checkpoint),
        "--output-dir",
        str(source_dir(args, structure, seed)),
        "--skill-structure",
        structure,
        "--epochs",
        str(args.source_epochs),
        "--early-stopping-patience",
        "0",
        "--batch-size",
        str(args.batch_size),
        "--sequence-length",
        str(args.sequence_length),
        "--stride",
        str(args.sequence_length),
        "--num-workers",
        str(args.num_workers),
        "--dataset-cache-size",
        str(args.dataset_cache_size),
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--no-tensorboard",
    ]
    if resume_checkpoint is not None:
        command.extend(("--resume-checkpoint", str(resume_checkpoint)))
    return command


def adapt_command(
    args: argparse.Namespace,
    structure: str,
    seed: int,
    resume_checkpoint: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(ADAPT_SCRIPT),
        "--checkpoint",
        str(checkpoint_path(args, structure, seed, "source")),
        "--manifest",
        str(args.manifest),
        "--output-dir",
        str(adapted_dir(args, structure, seed)),
        "--epochs",
        str(args.adapt_epochs),
        "--early-stopping-patience",
        "0",
        "--batch-size",
        str(args.batch_size),
        "--sequence-length",
        str(args.sequence_length),
        "--stride",
        str(args.sequence_length),
        "--num-workers",
        str(args.num_workers),
        "--dataset-cache-size",
        str(args.dataset_cache_size),
        "--seed",
        str(seed),
        "--device",
        args.device,
    ]
    if resume_checkpoint is not None:
        command.extend(("--resume-checkpoint", str(resume_checkpoint)))
    return command


def online_command(
    args: argparse.Namespace,
    structure: str,
    variant: str,
    difficulty: int,
    seed: int,
) -> list[str]:
    output_dir = online_dir(args, structure, variant, difficulty, seed)
    return [
        sys.executable,
        str(ONLINE_SCRIPT),
        "--hissd-checkpoint",
        str(checkpoint_path(args, structure, seed, variant)),
        "--mappo-checkpoint",
        str(args.mappo_checkpoint),
        "--output-dir",
        str(output_dir),
        "--learning-curve-output",
        str(curve_path(args, structure, variant, difficulty, seed)),
        "--method-name",
        f"hissd_{structure}_{variant}",
        "--difficulty",
        str(difficulty),
        "--iterations",
        str(args.iterations),
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
    command.extend(("--output", str(args.output_root / "comparison.json")))
    return command


def execute(
    command: list[str],
    *,
    label: str,
    dry_run: bool,
    cpu_threads: int,
    log_dir: Path,
) -> None:
    print(f"[START {label}] $ {shlex.join(command)}", flush=True)
    if dry_run:
        return
    environment = os.environ.copy()
    environment.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    environment.setdefault("PYTHONUNBUFFERED", "1")
    environment["OMP_NUM_THREADS"] = str(cpu_threads)
    environment["MKL_NUM_THREADS"] = str(cpu_threads)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{label.replace(':', '_')}.log"
    with log_path.open("w", encoding="utf-8") as log_file:
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
            log_file.write(line)
            log_file.flush()
            print(f"[{label}] {line}", end="", flush=True)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    print(f"[DONE  {label}]", flush=True)


def execute_many(
    jobs: list[tuple[str, list[str]]],
    *,
    max_workers: int,
    dry_run: bool,
    cpu_threads: int,
    log_dir: Path,
) -> None:
    """Run independent experiments concurrently with bounded resource use."""
    if not jobs:
        return
    if dry_run or max_workers == 1:
        for label, command in jobs:
            execute(
                command,
                label=label,
                dry_run=dry_run,
                cpu_threads=cpu_threads,
                log_dir=log_dir,
            )
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                execute,
                command,
                label=label,
                dry_run=False,
                cpu_threads=cpu_threads,
                log_dir=log_dir,
            ): label
            for label, command in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            label = futures[future]
            try:
                future.result()
            except Exception as error:
                raise RuntimeError(f"Parallel job failed: {label}") from error


def require_file(path: Path, stage: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Cannot run {stage}; required file is missing: {path}")


def main() -> None:
    args = parse_args()
    for name in ("manifest", "bc_checkpoint", "mappo_checkpoint"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.checkpoint_root = args.checkpoint_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()

    if "train" in args.stages:
        jobs = []
        for structure in args.structures:
            for seed in args.seeds:
                progress = progress_checkpoint_path(
                    args, structure, seed, "source"
                )
                completed_epoch = checkpoint_epoch(progress)
                if completed_epoch >= args.source_epochs and not args.force:
                    print(
                        f"SKIP complete source checkpoint (epoch "
                        f"{completed_epoch}/{args.source_epochs}): {progress}"
                    )
                    continue
                resume = (
                    progress
                    if completed_epoch > 0 and not args.force
                    else None
                )
                if resume is not None:
                    print(
                        f"RESUME partial source checkpoint (epoch "
                        f"{completed_epoch}/{args.source_epochs}): {resume}"
                    )
                jobs.append(
                    (
                        f"train:{structure}:seed{seed}",
                        train_command(args, structure, seed, resume),
                    )
                )
        execute_many(
            jobs,
            max_workers=args.parallel_train_jobs,
            dry_run=args.dry_run,
            cpu_threads=args.torch_cpu_threads,
            log_dir=args.output_root / "logs",
        )

    if "adapt" in args.stages:
        jobs = []
        for structure in args.structures:
            for seed in args.seeds:
                source = checkpoint_path(args, structure, seed, "source")
                if not args.dry_run:
                    require_file(source, "adapt")
                progress = progress_checkpoint_path(
                    args, structure, seed, "adapted"
                )
                completed_epoch = checkpoint_epoch(progress)
                if completed_epoch >= args.adapt_epochs and not args.force:
                    print(
                        f"SKIP complete adapted checkpoint (epoch "
                        f"{completed_epoch}/{args.adapt_epochs}): {progress}"
                    )
                    continue
                resume = (
                    progress
                    if completed_epoch > 0 and not args.force
                    else None
                )
                if resume is not None:
                    print(
                        f"RESUME partial adapted checkpoint (epoch "
                        f"{completed_epoch}/{args.adapt_epochs}): {resume}"
                    )
                jobs.append(
                    (
                        f"adapt:{structure}:seed{seed}",
                        adapt_command(args, structure, seed, resume),
                    )
                )
        execute_many(
            jobs,
            max_workers=args.parallel_adapt_jobs,
            dry_run=args.dry_run,
            cpu_threads=args.torch_cpu_threads,
            log_dir=args.output_root / "logs",
        )

    curves = [
        curve_path(args, structure, variant, difficulty, seed)
        for structure in args.structures
        for variant in args.variants
        for difficulty in args.difficulties
        for seed in args.seeds
    ]
    if "online" in args.stages:
        jobs = []
        for structure in args.structures:
            for variant in args.variants:
                for difficulty in args.difficulties:
                    for seed in args.seeds:
                        checkpoint = checkpoint_path(
                            args, structure, seed, variant
                        )
                        if not args.dry_run:
                            require_file(checkpoint, "online")
                        curve = curve_path(
                            args, structure, variant, difficulty, seed
                        )
                        if not args.force and curve_is_complete(
                            curve, args.iterations
                        ):
                            print(f"SKIP complete curve: {curve}")
                            continue
                        jobs.append(
                            (
                                f"online:{structure}:{variant}:d{difficulty}:seed{seed}",
                                online_command(
                                    args, structure, variant, difficulty, seed
                                ),
                            )
                        )
        execute_many(
            jobs,
            max_workers=args.parallel_online_jobs,
            dry_run=args.dry_run,
            cpu_threads=args.torch_cpu_threads,
            log_dir=args.output_root / "logs",
        )

    if "analyze" in args.stages:
        if not args.dry_run:
            missing = [path for path in curves if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"Cannot analyze before {len(missing)} curves are generated; "
                    f"first missing path: {missing[0]}"
                )
        execute(
            analyze_command(args, curves),
            label="analyze",
            dry_run=args.dry_run,
            cpu_threads=args.torch_cpu_threads,
            log_dir=args.output_root / "logs",
        )


if __name__ == "__main__":
    main()
