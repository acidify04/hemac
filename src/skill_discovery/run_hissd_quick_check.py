"""Run a short fixed-budget success/reward check for the existing HiSSD model."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
from pathlib import Path

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ONLINE_SCRIPT = PROJECT_ROOT / "src/skill_discovery/finetune_hissd_drone_online.py"
ANALYZE_SCRIPT = PROJECT_ROOT / "src/skill_discovery/analyze_learning_efficiency.py"
PLOT_SCRIPT = PROJECT_ROOT / "src/skill_discovery/plot_skill_control_curves.py"
DEFAULT_HISSD_CHECKPOINT = (
    PROJECT_ROOT
    / "src/skill_discovery/checkpoints/"
    "hissd_drone_d12_t34_adapted_stable_v4/hissd_adapted_best.pt"
)
DEFAULT_MAPPO_CHECKPOINT = (
    PROJECT_ROOT / "src/train/drone_mappo_coverage60_checkpoints/checkpoint_07800"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "src/skill_discovery/outputs/learning_efficiency/drone_d12_t34/"
    "hissd_skill_support_quick_check_v9"
)
MODES = ("full", "no_skill")
DEFAULT_SUCCESS_THRESHOLDS = {3: 0.6, 4: 0.5}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hissd-checkpoint", type=Path, default=DEFAULT_HISSD_CHECKPOINT)
    parser.add_argument("--mappo-checkpoint", type=Path, default=DEFAULT_MAPPO_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--difficulty", type=int, default=4)
    parser.add_argument(
        "--success-threshold",
        type=float,
        help="Learning-efficiency threshold; defaults to D3=0.6, D4=0.5.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--joint-step-budget", type=int, default=50_000)
    parser.add_argument("--eval-every-joint-steps", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--eval-seed-base", type=int, default=100_000_000)
    parser.add_argument("--train-batch-joint-steps", type=int, default=8_000)
    parser.add_argument("--ppo-epochs", type=int, default=5)
    parser.add_argument("--minibatch-size", type=int, default=1_024)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--base-head-lr", type=float, default=3e-5)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--base-unfreeze-after-joint-steps", type=int, default=16_000)
    parser.add_argument("--encoder-unfreeze-after-joint-steps", type=int, default=32_000)
    parser.add_argument(
        "--skill-prior-init",
        type=float,
        default=0.10,
        help="Initial HiSSD decoder contribution for full mode; no_skill stays zero.",
    )
    parser.add_argument("--skill-gate-lr", type=float, default=3e-5)
    parser.add_argument("--skill-residual-lr", type=float, default=3e-5)
    parser.add_argument("--skill-prior-max", type=float, default=0.25)
    parser.add_argument(
        "--use-pretrained-value",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the offline team-value baseline. Disabled by default because "
            "agent credit requires per-drone value targets."
        ),
    )
    parser.add_argument(
        "--credit-assignment",
        choices=("agent", "team"),
        default="agent",
        help=(
            "agent matches the original shared-policy MAPPO reward semantics; "
            "team broadcasts one averaged advantage to all drones."
        ),
    )
    parser.add_argument(
        "--shared-terminal-crash-penalty",
        type=float,
        default=0.0,
        help="Use zero to match the original MAPPO per-agent collision reward.",
    )
    parser.add_argument(
        "--rollback-score-drop",
        type=float,
        default=0.15,
        help=(
            "Restore the best policy after a sustained validation drop of this size. "
            "Set to zero to disable."
        ),
    )
    parser.add_argument("--rollback-patience", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.difficulty <= 0:
        raise ValueError("--difficulty must be positive.")
    if args.success_threshold is None:
        args.success_threshold = DEFAULT_SUCCESS_THRESHOLDS.get(args.difficulty, 0.5)
    if not 0.0 <= args.success_threshold <= 1.0:
        raise ValueError("--success-threshold must be in [0, 1].")
    for name in (
        "joint_step_budget",
        "eval_every_joint_steps",
        "eval_episodes",
        "train_batch_joint_steps",
        "ppo_epochs",
        "minibatch_size",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.actor_lr <= 0.0:
        raise ValueError("--actor-lr must be positive.")
    if args.base_head_lr <= 0.0 or args.encoder_lr <= 0.0:
        raise ValueError("Base-head and encoder learning rates must be positive.")
    if args.base_unfreeze_after_joint_steps < 0:
        raise ValueError("--base-unfreeze-after-joint-steps cannot be negative.")
    if args.encoder_unfreeze_after_joint_steps < 0:
        raise ValueError("--encoder-unfreeze-after-joint-steps cannot be negative.")
    if args.skill_gate_lr <= 0.0:
        raise ValueError("--skill-gate-lr must be positive.")
    if args.skill_residual_lr <= 0.0:
        raise ValueError("--skill-residual-lr must be positive.")
    if not 0.0 <= args.skill_prior_init < 1.0:
        raise ValueError("--skill-prior-init must be in [0, 1).")
    if not 0.0 < args.skill_prior_max < 1.0:
        raise ValueError("--skill-prior-max must be in (0, 1).")
    if args.skill_prior_init > args.skill_prior_max:
        raise ValueError("--skill-prior-init cannot exceed --skill-prior-max.")
    if args.shared_terminal_crash_penalty < 0.0:
        raise ValueError("--shared-terminal-crash-penalty cannot be negative.")
    if args.rollback_score_drop < 0.0:
        raise ValueError("--rollback-score-drop cannot be negative.")
    if args.rollback_patience <= 0:
        raise ValueError("--rollback-patience must be positive.")
    if args.eval_every_joint_steps > args.joint_step_budget:
        raise ValueError("Evaluation interval cannot exceed the joint-step budget.")
    return args


def run(
    command: list[str],
    *,
    label: str,
    log_path: Path,
    dry_run: bool,
    progress=None,
) -> None:
    message = f"[START {label}] $ {shlex.join(command)}"
    progress.write(message) if progress is not None else print(message, flush=True)
    if dry_run:
        if progress is not None:
            progress.update(1)
        return
    environment = os.environ.copy()
    environment.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    environment.setdefault("PYTHONUNBUFFERED", "1")
    environment.setdefault("MPLCONFIGDIR", "/tmp/hemac_matplotlib_cache")
    # Environment construction is intentionally quiet: repeated INFO logs otherwise
    # repaint child tqdm bars. Users can still opt in with LOGLEVEL=INFO.
    environment.setdefault("LOGLEVEL", "WARNING")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if progress is not None:
        progress.clear()
    with log_path.open("wb") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
        assert process.stdout is not None
        while True:
            chunk = os.read(process.stdout.fileno(), 4096)
            if not chunk:
                break
            output = getattr(sys.stdout, "buffer", sys.stdout)
            if output is sys.stdout:
                output.write(chunk.decode("utf-8", errors="replace"))
            else:
                output.write(chunk)
            output.flush()
            log_file.write(chunk)
            log_file.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    message = f"[DONE  {label}]"
    progress.write(message) if progress is not None else print(message, flush=True)
    if progress is not None:
        progress.update(1)
        progress.set_postfix(job=label, refresh=False)
        progress.refresh()


def write_report(curve_paths: list[Path], comparison_path: Path, output: Path) -> None:
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    efficiency_by_method = {
        row["method"]: row for row in comparison.get("per_run", [])
    }
    runs = []
    for curve_path in curve_paths:
        curve = json.loads(curve_path.read_text(encoding="utf-8"))
        points = curve["points"]
        ppo_points = [point for point in points if "ppo_approx_kl" in point]
        kls = [float(point["ppo_approx_kl"]) for point in ppo_points]
        clips = [float(point["ppo_clip_fraction"]) for point in ppo_points]
        skill_deltas = [
            float(point.get("ppo_skill_action_delta", 0.0)) for point in ppo_points
        ]
        skill_gates = [
            float(point.get("ppo_skill_prior_gate", 0.0)) for point in ppo_points
        ]
        skill_retentions = [
            float(point.get("ppo_skill_prior_retention", 0.0))
            for point in ppo_points
        ]
        mean_kl = statistics.fmean(kls) if kls else 0.0
        mean_clip = statistics.fmean(clips) if clips else 0.0
        finite = all(math.isfinite(value) for value in (*kls, *clips))
        first, last = points[0], points[-1]
        evaluation_episodes = int(
            curve.get("metadata", {}).get("evaluation_episodes", 0)
        )
        final_success = float(last["success_rate"])
        final_standard_error = (
            math.sqrt(
                final_success * (1.0 - final_success) / evaluation_episodes
            )
            if evaluation_episodes > 0
            else float("nan")
        )
        evaluated_points = points[1:] if len(points) > 1 else points
        window = min(3, len(evaluated_points))
        early_success = statistics.fmean(
            float(point["success_rate"]) for point in evaluated_points[:window]
        )
        late_success = statistics.fmean(
            float(point["success_rate"]) for point in evaluated_points[-window:]
        )
        best = max(
            points,
            key=lambda point: (
                float(point["success_rate"])
                + 0.10 * float(point["goal_found_rate"])
                - 0.25 * float(point["fatal_crash_rate"])
            ),
        )
        efficiency = efficiency_by_method.get(first["method"], {})
        runs.append(
            {
                "method": first["method"],
                "skill_mode": curve.get("metadata", {}).get("skill_mode"),
                "matched_bc_initial_policy": bool(
                    curve.get("metadata", {})
                    .get("online_adaptation", {})
                    .get("matched_bc_initial_policy", False)
                ),
                "initial_success_rate": float(first["success_rate"]),
                "final_success_rate": float(last["success_rate"]),
                "final_success_gain": float(last["success_rate"])
                - float(first["success_rate"]),
                "best_success_rate": float(best["success_rate"]),
                "best_success_step": int(best["joint_env_steps"]),
                "selected_checkpoint_success_rate": float(best["success_rate"]),
                "selected_checkpoint_step": int(best["joint_env_steps"]),
                "evaluation_episodes": evaluation_episodes,
                "final_success_standard_error": final_standard_error,
                "final_success_ci95": [
                    max(0.0, final_success - 1.96 * final_standard_error),
                    min(1.0, final_success + 1.96 * final_standard_error),
                ],
                "early_evaluation_success_mean": early_success,
                "late_evaluation_success_mean": late_success,
                "late_minus_early_success": late_success - early_success,
                "rollback_count": sum(
                    int(point.get("rollback_triggered", 0.0)) for point in points
                ),
                "initial_episode_return": float(first["mean_episode_return"]),
                "final_episode_return": float(last["mean_episode_return"]),
                "success_auc": efficiency.get("success_auc"),
                "success_gain_auc": efficiency.get("success_gain_auc"),
                "mean_ppo_approx_kl": mean_kl,
                "mean_ppo_clip_fraction": mean_clip,
                "mean_skill_action_delta": (
                    statistics.fmean(skill_deltas) if skill_deltas else 0.0
                ),
                "final_skill_prior_gate": skill_gates[-1] if skill_gates else 0.0,
                "final_skill_prior_retention": (
                    skill_retentions[-1] if skill_retentions else 0.0
                ),
                "mechanically_healthy": bool(
                    ppo_points
                    and finite
                    and -0.01 <= mean_kl <= 0.05
                    and 0.0 <= mean_clip <= 0.40
                ),
            }
        )
    by_mode = {run["skill_mode"]: run for run in runs}
    paired = None
    if "full" in by_mode and "no_skill" in by_mode:
        paired = {
            "initial_success_difference": (
                by_mode["full"]["initial_success_rate"]
                - by_mode["no_skill"]["initial_success_rate"]
            ),
            "success_auc_difference": (
                by_mode["full"]["success_auc"] - by_mode["no_skill"]["success_auc"]
            ),
            "success_gain_auc_difference": (
                by_mode["full"]["success_gain_auc"]
                - by_mode["no_skill"]["success_gain_auc"]
            ),
            "final_success_difference": (
                by_mode["full"]["final_success_rate"]
                - by_mode["no_skill"]["final_success_rate"]
            ),
        }
        paired["matched_initial_policy"] = bool(
            by_mode["full"]["matched_bc_initial_policy"]
            and by_mode["no_skill"]["matched_bc_initial_policy"]
        )
    report = {
        "purpose": "paired HiSSD skill/no-skill pilot, not a significance test",
        "runs": runs,
        "paired": paired,
    }
    report["skill_direction_promising"] = bool(
        paired
        and paired["success_auc_difference"] > 0.0
        and paired["final_success_difference"] >= 0.0
    )
    report["metric_interpretation"] = {
        "success_auc_difference": (
            "Primary transfer-efficiency metric; includes any zero-shot advantage "
            "from the pretrained skill prior."
        ),
        "success_gain_auc_difference": (
            "Secondary online-learning metric; subtracts each method's initial "
            "success and therefore excludes zero-shot transfer."
        ),
        "matched_initial_policy": (
            "Expected to be false when --skill-prior-init is nonzero."
        ),
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("HISSD QUICK CHECK", json.dumps(report, indent=2), flush=True)
    print(f"Saved quick-check report: {output}", flush=True)


def main() -> None:
    args = parse_args()
    checkpoint = args.hissd_checkpoint.expanduser().resolve()
    env_checkpoint = args.mappo_checkpoint.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"HiSSD checkpoint not found: {checkpoint}")
    if not env_checkpoint.exists():
        raise FileNotFoundError(f"Environment checkpoint not found: {env_checkpoint}")

    curve_paths = []
    run_specs = []
    for mode in args.modes:
        run_dir = output_root / mode / f"d{args.difficulty}_seed{args.seed}"
        curve_path = run_dir / f"learning_curve_seed_{args.seed}.json"
        curve_paths.append(curve_path)
        run_specs.append((mode, run_dir, curve_path))
    comparison_path = output_root / "comparison.json"
    plot_path = output_root / "success_reward_curves.png"
    report_path = output_root / "quick_check_report.json"
    output_root.mkdir(parents=True, exist_ok=True)
    if args.force and not args.dry_run:
        for path in (*curve_paths, comparison_path, plot_path, report_path):
            path.unlink(missing_ok=True)

    overall_progress = (
        tqdm(
            total=len(run_specs) + 2,
            desc=f"quick-check D{args.difficulty}",
            unit="job",
            dynamic_ncols=True,
            position=1,
        )
        if tqdm is not None
        else None
    )
    try:
        for mode, run_dir, curve_path in run_specs:
            method = (
                "hissd_skill_supported_ppo"
                if mode == "full"
                else "hissd_no_skill_ppo_control"
            )
            online = [
            sys.executable,
            str(ONLINE_SCRIPT),
            "--hissd-checkpoint",
            str(checkpoint),
            "--mappo-checkpoint",
            str(env_checkpoint),
            "--output-dir",
            str(run_dir),
            "--learning-curve-output",
            str(curve_path),
            "--method-name",
            method,
            "--skill-mode",
            mode,
            "--difficulty",
            str(args.difficulty),
            "--train-batch-joint-steps",
            str(args.train_batch_joint_steps),
            "--joint-step-budget",
            str(args.joint_step_budget),
            "--eval-every-joint-steps",
            str(args.eval_every_joint_steps),
            "--eval-episodes",
            str(args.eval_episodes),
            "--eval-seed-base",
            str(args.eval_seed_base),
            "--ppo-epochs",
            str(args.ppo_epochs),
            "--minibatch-size",
            str(args.minibatch_size),
            "--actor-lr",
            str(args.actor_lr),
            "--base-head-lr",
            str(args.base_head_lr),
            "--encoder-lr",
            str(args.encoder_lr),
            "--base-unfreeze-after-joint-steps",
            str(args.base_unfreeze_after_joint_steps),
            "--encoder-unfreeze-after-joint-steps",
            str(args.encoder_unfreeze_after_joint_steps),
            "--skill-prior-init",
            str(args.skill_prior_init),
            "--skill-gate-lr",
            str(args.skill_gate_lr),
            "--skill-residual-lr",
            str(args.skill_residual_lr),
            "--skill-prior-max",
            str(args.skill_prior_max),
            "--gamma",
            "0.995",
            "--clip-ratio",
            "0.2",
            "--target-kl",
            "0.02",
            "--entropy-coeff",
            "0.01",
            "--anchor-coeff",
            "0.001",
            "--skill-anchor-coeff",
            "0.001",
            "--context-warmup-iterations",
            "0",
            "--shared-terminal-crash-penalty",
            str(args.shared_terminal_crash_penalty),
            "--rollback-score-drop",
            str(args.rollback_score_drop),
            "--rollback-min-best-success",
            str(args.success_threshold),
            "--rollback-patience",
            str(args.rollback_patience),
            "--credit-assignment",
            args.credit_assignment,
            (
                "--use-pretrained-value"
                if args.use_pretrained_value
                else "--no-use-pretrained-value"
            ),
            "--device",
            args.device,
            "--deterministic",
            ]
            run(
                online,
                label=f"hissd-{mode}",
                log_path=output_root / f"logs/{mode}.log",
                dry_run=args.dry_run,
                progress=overall_progress,
            )
        analyze = [
        sys.executable,
        str(ANALYZE_SCRIPT),
        "analyze",
        "--curves",
        *(str(path) for path in curve_paths),
        "--threshold",
        f"{args.difficulty}={args.success_threshold}",
        "--budget",
        str(args.joint_step_budget),
        "--output",
        str(comparison_path),
        ]
        plot = [
        sys.executable,
        str(PLOT_SCRIPT),
        "--curves",
        *(str(path) for path in curve_paths),
        "--output",
        str(plot_path),
        ]

        run(
            analyze,
            label="analyze",
            log_path=output_root / "logs/analyze.log",
            dry_run=args.dry_run,
            progress=overall_progress,
        )
        run(
            plot,
            label="plot",
            log_path=output_root / "logs/plot.log",
            dry_run=args.dry_run,
            progress=overall_progress,
        )
        if not args.dry_run:
            write_report(curve_paths, comparison_path, report_path)
    finally:
        if overall_progress is not None:
            overall_progress.close()


if __name__ == "__main__":
    main()
