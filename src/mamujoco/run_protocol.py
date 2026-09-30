"""Run or print the integrated MaMuJoCo transfer protocols."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from .difficulty_protocol import (
    DEFAULT_EXPERIMENT_SEEDS,
    add_difficulty_arguments,
    protocol_from_args,
)
from .env import add_environment_version_argument
from .tasks import SUPPORTED_SUITES, list_tasks


LEGACY_SUITES = ("dynamics", "joint_disable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=(*SUPPORTED_SUITES, "both"), default="both")
    parser.add_argument(
        "--stage",
        choices=(
            "pilot", "happo", "select", "collect", "offline",
            "zero_shot", "adapt", "evaluate", "all",
        ),
        default="all",
    )
    parser.add_argument(
        "--algorithms", nargs="+", choices=("hissd", "skill_vae"),
        default=("hissd", "skill_vae"),
    )
    parser.add_argument("--offline-seeds", nargs="+", type=int, default=(1, 2, 3, 4))
    parser.add_argument(
        "--experiment-seeds", nargs="+", type=int,
        default=DEFAULT_EXPERIMENT_SEEDS,
    )
    parser.add_argument("--behavior-seed", type=int, default=1)
    parser.add_argument("--max-cycles", type=int, default=1000)
    parser.add_argument("--happo-total-env-steps", type=int, default=2_000_000)
    parser.add_argument("--happo-episodes-per-update", type=int, default=8)
    parser.add_argument("--happo-eval-every-updates", type=int, default=10)
    parser.add_argument("--happo-eval-episodes", type=int, default=8)
    parser.add_argument("--trajectories-per-task", type=int, default=100)
    parser.add_argument("--episodes-per-quality", type=int, default=100)
    parser.add_argument("--offline-training-steps", type=int, default=1_000_000)
    parser.add_argument("--eval-episodes-per-seed", type=int, default=8)
    parser.add_argument("--difficulty-eval-episodes", type=int, default=20)
    parser.add_argument("--adaptation-budget", type=int, default=500_000)
    parser.add_argument("--adaptation-eval-interval", type=int, default=10_000)
    parser.add_argument("--adaptation-batch-size", type=int, default=128)
    parser.add_argument("--target-replay-ratio", type=float, default=0.5)
    parser.add_argument(
        "--source-replay-ratios", nargs=2, type=float, default=(0.25, 0.25)
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    add_environment_version_argument(parser)
    add_difficulty_arguments(parser)
    return parser.parse_args()


def execute(command: list[str], *, dry_run: bool) -> None:
    print("$", shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def module_command(module: str, *arguments) -> list[str]:
    return [sys.executable, "-m", module, *(str(value) for value in arguments)]


def difficulty_arguments(args) -> list[str]:
    return [
        "--difficulty-strengths", *map(str, args.difficulty_strengths),
        "--source-difficulties", *args.source_difficulties,
        "--target-difficulties", *args.target_difficulties,
    ]


def run_legacy_suite(args, suite: str, stages: tuple[str, ...]) -> None:
    if "happo" in stages:
        for task in list_tasks(suite, "source"):
            execute(
                module_command(
                    "mamujoco.train_happo", "--suite", suite,
                    "--task", task.name, "--seed", args.behavior_seed,
                    "--total-env-steps", args.happo_total_env_steps,
                    "--max-cycles", args.max_cycles,
                    "--episodes-per-update", args.happo_episodes_per_update,
                    "--eval-every-updates", args.happo_eval_every_updates,
                    "--eval-episodes", args.happo_eval_episodes,
                    "--environment-version", args.environment_version,
                    "--device", args.device,
                ),
                dry_run=args.dry_run,
            )
    if "collect" in stages:
        execute(
            module_command(
                "mamujoco.collect_offline_data", "--suite", suite,
                "--task", "all", "--checkpoint-seed", args.behavior_seed,
                "--trajectories-per-task", args.trajectories_per_task,
                "--max-cycles", args.max_cycles,
                "--environment-version", args.environment_version,
                "--device", args.device,
            ),
            dry_run=args.dry_run,
        )
    if "offline" in stages:
        for algorithm in args.algorithms:
            for seed in args.offline_seeds:
                execute(
                    module_command(
                        "mamujoco.train_offline", "--algorithm", algorithm,
                        "--suite", suite, "--seed", seed,
                        "--training-steps", args.offline_training_steps,
                        "--device", args.device,
                    ),
                    dry_run=args.dry_run,
                )
    if "zero_shot" in stages or "evaluate" in stages:
        for algorithm in args.algorithms:
            evaluation_paths = []
            for seed in args.offline_seeds:
                checkpoint_dir = (
                    Path("src/mamujoco/checkpoints/offline")
                    / suite / algorithm / f"seed_{seed}"
                )
                output = checkpoint_dir / "zero_shot_evaluation.json"
                evaluation_paths.append(output)
                if "zero_shot" in stages:
                    execute(
                        module_command(
                            "mamujoco.evaluate_zero_shot", "--algorithm", algorithm,
                            "--suite", suite, "--checkpoint", checkpoint_dir / "latest.pt",
                            "--episodes", args.eval_episodes_per_seed,
                            "--max-cycles", args.max_cycles,
                            "--environment-version", args.environment_version,
                            "--seed-base", 20_000_000 + seed * 1_000_000,
                            "--output", output, "--device", args.device,
                        ),
                        dry_run=args.dry_run,
                    )
            if "evaluate" in stages:
                aggregate_output = (
                    Path("src/mamujoco/outputs") / suite / algorithm
                    / (
                        "zero_shot_"
                        f"{len(args.offline_seeds) * args.eval_episodes_per_seed}_runs.json"
                    )
                )
                execute(
                    module_command(
                        "mamujoco.aggregate_evaluations", *evaluation_paths,
                        "--output", aggregate_output,
                    ),
                    dry_run=args.dry_run,
                )


def run_difficulty_suite(args, stages: tuple[str, ...]) -> None:
    protocol = protocol_from_args(args)
    protocol_cli = difficulty_arguments(args)
    if "happo" in stages:
        for task in protocol.tasks:
            if task.split != "source":
                continue
            execute(
                module_command(
                    "mamujoco.train_happo", "--suite", "difficulty",
                    "--task", task.name, "--seed", args.behavior_seed,
                    "--total-env-steps", args.happo_total_env_steps,
                    "--max-cycles", args.max_cycles,
                    "--episodes-per-update", args.happo_episodes_per_update,
                    "--eval-every-updates", args.happo_eval_every_updates,
                    "--eval-episodes", args.happo_eval_episodes,
                    "--environment-version", args.environment_version,
                    "--device", args.device, *protocol_cli,
                ),
                dry_run=args.dry_run,
            )
    if "pilot" in stages:
        pilot_root = Path("src/mamujoco/checkpoints/pilot_happo")
        for task in protocol.tasks:
            for seed in args.experiment_seeds:
                execute(
                    module_command(
                        "mamujoco.train_happo", "--suite", "difficulty",
                        "--task", task.name, "--seed", seed,
                        "--allow-target-task",
                        "--output-root", pilot_root,
                        "--total-env-steps", args.happo_total_env_steps,
                        "--max-cycles", args.max_cycles,
                        "--episodes-per-update", args.happo_episodes_per_update,
                        "--eval-every-updates", args.happo_eval_every_updates,
                        "--eval-episodes", args.happo_eval_episodes,
                        "--environment-version", args.environment_version,
                        "--device", args.device, *protocol_cli,
                    ),
                    dry_run=args.dry_run,
                )
        execute(
            module_command(
                "mamujoco.pilot_difficulty", "--checkpoint-root", pilot_root,
                "--seeds", *args.experiment_seeds,
                "--episodes", args.difficulty_eval_episodes,
                "--max-cycles", args.max_cycles,
                "--environment-version", args.environment_version,
                "--device", args.device, *protocol_cli,
            ),
            dry_run=args.dry_run,
        )
    if "select" in stages:
        execute(
            module_command(
                "mamujoco.select_quality_checkpoints", "--seed", args.behavior_seed,
                "--environment-version", args.environment_version,
                *protocol_cli,
            ),
            dry_run=args.dry_run,
        )
    if "collect" in stages:
        execute(
            module_command(
                "mamujoco.collect_difficulty_data",
                "--checkpoint-seed", args.behavior_seed,
                "--episodes-per-quality", args.episodes_per_quality,
                "--max-cycles", args.max_cycles,
                "--environment-version", args.environment_version,
                "--device", args.device, *protocol_cli,
            ),
            dry_run=args.dry_run,
        )
    if "offline" in stages:
        for algorithm in args.algorithms:
            for seed in args.experiment_seeds:
                execute(
                    module_command(
                        "mamujoco.train_offline", "--algorithm", algorithm,
                        "--suite", "difficulty", "--seed", seed,
                        "--training-steps", args.offline_training_steps,
                        "--device", args.device,
                    ),
                    dry_run=args.dry_run,
                )
    if "zero_shot" in stages:
        for algorithm in args.algorithms:
            for seed in args.experiment_seeds:
                checkpoint_dir = (
                    Path("src/mamujoco/checkpoints/offline/difficulty")
                    / algorithm / f"seed_{seed}"
                )
                execute(
                    module_command(
                        "mamujoco.evaluate_zero_shot", "--algorithm", algorithm,
                        "--suite", "difficulty", "--checkpoint", checkpoint_dir / "latest.pt",
                        "--episodes", args.difficulty_eval_episodes,
                        "--max-cycles", args.max_cycles,
                        "--environment-version", args.environment_version,
                        "--seed-base", 20_000_000 + seed * 1_000_000,
                        "--output", checkpoint_dir / "zero_shot_evaluation.json",
                        "--device", args.device, *protocol_cli,
                    ),
                    dry_run=args.dry_run,
                )
    if "adapt" in stages:
        for seed in args.experiment_seeds:
            checkpoint = (
                Path("src/mamujoco/checkpoints/offline/difficulty/hissd")
                / f"seed_{seed}" / "latest.pt"
            )
            for target in protocol.target_ids:
                execute(
                    module_command(
                        "mamujoco.adapt_difficulty", "--checkpoint", checkpoint,
                        "--target-difficulty", target, "--seed", seed,
                        "--adaptation-budget", args.adaptation_budget,
                        "--eval-interval", args.adaptation_eval_interval,
                        "--eval-episodes", args.difficulty_eval_episodes,
                        "--batch-size", args.adaptation_batch_size,
                        "--max-cycles", args.max_cycles,
                        "--environment-version", args.environment_version,
                        "--target-replay-ratio", args.target_replay_ratio,
                        "--source-replay-ratios", *args.source_replay_ratios,
                        "--device", args.device, *protocol_cli,
                    ),
                    dry_run=args.dry_run,
                )
    if "evaluate" in stages:
        for algorithm in args.algorithms:
            zero_shot_inputs = [
                Path("src/mamujoco/checkpoints/offline/difficulty")
                / algorithm
                / f"seed_{seed}"
                / "zero_shot_evaluation.json"
                for seed in args.experiment_seeds
            ]
            execute(
                module_command(
                    "mamujoco.aggregate_evaluations",
                    *zero_shot_inputs,
                    "--output",
                    Path("src/mamujoco/outputs/difficulty")
                    / algorithm
                    / (
                        "zero_shot_"
                        f"{len(args.experiment_seeds) * args.difficulty_eval_episodes}_runs.json"
                    ),
                ),
                dry_run=args.dry_run,
            )
        inputs = [
            Path("src/mamujoco/checkpoints/adaptation")
            / target / f"seed_{seed}" / "adaptation_results.json"
            for target in protocol.target_ids
            for seed in args.experiment_seeds
        ]
        execute(
            module_command(
                "mamujoco.aggregate_adaptation", *inputs,
                "--output-dir", "src/mamujoco/outputs/difficulty/adaptation",
            ),
            dry_run=args.dry_run,
        )


def main() -> None:
    args = parse_args()
    suites = LEGACY_SUITES if args.suite == "both" else (args.suite,)
    stages = (
        (
            "happo", "pilot", "select", "collect", "offline",
            "zero_shot", "adapt", "evaluate",
        )
        if args.stage == "all"
        else (args.stage,)
    )
    for suite in suites:
        if suite == "difficulty":
            run_difficulty_suite(args, stages)
        else:
            run_legacy_suite(args, suite, stages)


if __name__ == "__main__":
    main()
