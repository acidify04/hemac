"""Run paired skill-transfer versus no-skill PPO experiments.

Both conditions use the same offline checkpoint, BC action head, centralized
critic architecture, rollout budget, and seeds. By default the full condition
uses a short skill-first base-head freeze; set --full-base-freeze-iterations 0
to recover the strict architecture-only full/no-skill control.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ONLINE_SCRIPT = (
    PROJECT_ROOT / "src/skill_discovery/finetune_drone_skill_vae_online.py"
)
ANALYZE_SCRIPT = (
    PROJECT_ROOT / "src/skill_discovery/analyze_learning_efficiency.py"
)
PLOT_SCRIPT = PROJECT_ROOT / "src/skill_discovery/plot_skill_control_curves.py"
DEFAULT_VAE_CHECKPOINT = (
    PROJECT_ROOT
    / "src/skill_discovery/checkpoints/drone_skill_vae_chunked_k8/"
    "drone_skill_vae_best.pt"
)
DEFAULT_MAPPO_CHECKPOINT = (
    PROJECT_ROOT / "src/train/drone_mappo_coverage60_checkpoints/checkpoint_07800"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "src/skill_discovery/outputs/learning_efficiency/drone_d12_t34/"
    "vae_chunked_skill_control"
)
MODES = ("full", "no_skill")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("online", "analyze"),
        default=("online", "analyze"),
    )
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=(2026, 2027, 2028, 2029, 2030)
    )
    parser.add_argument("--difficulties", type=int, nargs="+", default=(3, 4))
    parser.add_argument("--vae-checkpoint", type=Path, default=DEFAULT_VAE_CHECKPOINT)
    parser.add_argument(
        "--mappo-checkpoint", type=Path, default=DEFAULT_MAPPO_CHECKPOINT
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--episodes-per-iteration", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--skill-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--log-std-lr", type=float, default=2e-5)
    parser.add_argument("--clip-ratio", type=float, default=0.15)
    parser.add_argument("--entropy-coeff", type=float, default=0.002)
    parser.add_argument("--anchor-coeff", type=float, default=0.01)
    parser.add_argument("--skill-anchor-coeff", type=float, default=0.01)
    parser.add_argument("--skill-warmup-iterations", type=int, default=0)
    parser.add_argument("--full-base-freeze-iterations", type=int, default=10)
    parser.add_argument("--parallel-jobs", type=int, default=1)
    parser.add_argument("--torch-cpu-threads", type=int, default=2)
    parser.add_argument("--thresholds", nargs="+", default=("3=0.6", "4=0.5"))
    parser.add_argument("--budget", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "iterations",
        "episodes_per_iteration",
        "eval_every",
        "eval_episodes",
        "ppo_epochs",
        "minibatch_size",
        "actor_lr",
        "skill_lr",
        "critic_lr",
        "log_std_lr",
        "clip_ratio",
        "parallel_jobs",
        "torch_cpu_threads",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.anchor_coeff < 0 or args.skill_anchor_coeff < 0:
        raise ValueError("Anchor coefficients cannot be negative.")
    if args.skill_warmup_iterations < 0 or args.full_base_freeze_iterations < 0:
        raise ValueError("Skill/base warm-up iteration counts cannot be negative.")
    if args.budget is not None and args.budget <= 0:
        raise ValueError("--budget must be positive.")
    for name in ("modes", "seeds", "difficulties"):
        values = getattr(args, name)
        if len(values) != len(set(values)):
            raise ValueError(f"--{name} must not contain duplicates.")


def experiment_dir(
    args: argparse.Namespace, mode: str, difficulty: int, seed: int
) -> Path:
    return args.output_root / mode / f"d{difficulty}_seed{seed}"


def curve_path(
    args: argparse.Namespace, mode: str, difficulty: int, seed: int
) -> Path:
    return experiment_dir(args, mode, difficulty, seed) / (
        f"learning_curve_seed_{seed}.json"
    )


def curve_is_complete(path: Path, expected_iteration: int) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return max(
        (int(point.get("iteration", -1)) for point in payload.get("points", ())),
        default=-1,
    ) >= expected_iteration


def online_command(
    args: argparse.Namespace, mode: str, difficulty: int, seed: int
) -> list[str]:
    output_dir = experiment_dir(args, mode, difficulty, seed)
    method_name = f"vae_{mode}_controlled"
    if mode == "full" and args.full_base_freeze_iterations > 0:
        method_name = "vae_full_skill_first"
    return [
        sys.executable,
        str(ONLINE_SCRIPT),
        "--vae-checkpoint",
        str(args.vae_checkpoint),
        "--mappo-checkpoint",
        str(args.mappo_checkpoint),
        "--output-dir",
        str(output_dir),
        "--learning-curve-output",
        str(curve_path(args, mode, difficulty, seed)),
        "--method-name",
        method_name,
        "--skill-mode",
        mode,
        "--train-base-action-head",
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
        "--ppo-epochs",
        str(args.ppo_epochs),
        "--minibatch-size",
        str(args.minibatch_size),
        "--actor-lr",
        str(args.actor_lr),
        "--skill-lr",
        str(args.skill_lr),
        "--critic-lr",
        str(args.critic_lr),
        "--log-std-lr",
        str(args.log_std_lr),
        "--clip-ratio",
        str(args.clip_ratio),
        "--entropy-coeff",
        str(args.entropy_coeff),
        "--anchor-coeff",
        str(args.anchor_coeff),
        "--skill-anchor-coeff",
        str(args.skill_anchor_coeff),
        "--skill-warmup-iterations",
        str(args.skill_warmup_iterations),
        "--full-base-freeze-iterations",
        str(args.full_base_freeze_iterations),
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--deterministic" if args.deterministic else "--no-deterministic",
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


def plot_command(args: argparse.Namespace, curves: list[Path]) -> list[str]:
    return [
        sys.executable,
        str(PLOT_SCRIPT),
        "--curves",
        *(str(path) for path in curves),
        "--output",
        str(args.output_root / "success_reward_curves.png"),
    ]


def run_command(
    command: list[str], *, label: str, dry_run: bool, cpu_threads: int, log_dir: Path
) -> None:
    print(f"[START {label}] $ {shlex.join(command)}", flush=True)
    if dry_run:
        return
    environment = os.environ.copy()
    environment.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    environment.setdefault("PYTHONUNBUFFERED", "1")
    environment.setdefault("MPLCONFIGDIR", "/tmp/hemac_matplotlib_cache")
    environment["PYTHONHASHSEED"] = "0"
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


def run_many(
    jobs: list[tuple[str, list[str]]],
    *,
    max_workers: int,
    dry_run: bool,
    cpu_threads: int,
    log_dir: Path,
) -> None:
    if dry_run or max_workers == 1:
        for label, command in jobs:
            run_command(
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
                run_command,
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


def main() -> None:
    args = parse_args()
    args.vae_checkpoint = args.vae_checkpoint.expanduser().resolve()
    args.mappo_checkpoint = args.mappo_checkpoint.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if not args.dry_run:
        for path in (args.vae_checkpoint, args.mappo_checkpoint):
            if not path.exists():
                raise FileNotFoundError(f"Required checkpoint is missing: {path}")

    curves = [
        curve_path(args, mode, difficulty, seed)
        for mode in args.modes
        for difficulty in args.difficulties
        for seed in args.seeds
    ]
    if "online" in args.stages:
        jobs = []
        for mode in args.modes:
            for difficulty in args.difficulties:
                for seed in args.seeds:
                    curve = curve_path(args, mode, difficulty, seed)
                    if curve_is_complete(curve, args.iterations) and not args.force:
                        print(f"SKIP complete curve: {curve}", flush=True)
                        continue
                    if curve.is_file() and not args.dry_run:
                        # A restarted run must not retain stale points with a
                        # different realized world-cycle budget.
                        curve.unlink()
                    jobs.append(
                        (
                            f"{mode}:d{difficulty}:seed{seed}",
                            online_command(args, mode, difficulty, seed),
                        )
                    )
        run_many(
            jobs,
            max_workers=args.parallel_jobs,
            dry_run=args.dry_run,
            cpu_threads=args.torch_cpu_threads,
            log_dir=args.output_root / "logs",
        )

    if "analyze" in args.stages:
        if not args.dry_run:
            incomplete = [
                path
                for path in curves
                if not curve_is_complete(path, args.iterations)
            ]
            if incomplete:
                formatted = "\n".join(f"  - {path}" for path in incomplete)
                raise FileNotFoundError(
                    "Cannot analyze an incomplete paired experiment:\n" + formatted
                )
        run_command(
            analyze_command(args, curves),
            label="analyze",
            dry_run=args.dry_run,
            cpu_threads=args.torch_cpu_threads,
            log_dir=args.output_root / "logs",
        )
        run_command(
            plot_command(args, curves),
            label="plot-success-reward",
            dry_run=args.dry_run,
            cpu_threads=args.torch_cpu_threads,
            log_dir=args.output_root / "logs",
        )


if __name__ == "__main__":
    main()
