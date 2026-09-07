"""Train MAPPO/BC drone baselines online and record crash-safe curves."""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.registry import register_env


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from hemac import HeMAC_v0
from hemac.curriculum_config import OBSTACLE_CURRICULUM_LEVELS
from hemac.rllib_policy import (
    drone_policy_model_config,
    register_hemac_rllib_models,
)
from skill_discovery.analyze_learning_efficiency import append_curve_points
from skill_discovery.collect_offline_data import ENV_NAME, env_creator
from skill_discovery.evaluate_drone_bc import (
    DEFAULT_BC_CHECKPOINT,
    drone_action_scale,
    run_episode,
    summarize,
)
from skill_discovery.finetune_hissd_drone_online import (
    DEFAULT_MAPPO_CHECKPOINT,
    build_env_config,
    load_checkpoint_env_config,
)
from skill_discovery.models import DroneBehaviorCloningPolicy


DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "src/skill_discovery/outputs/learning_efficiency/drone_baselines"
)


def get_env_runner_group(algo):
    """Return the old-stack worker group without importing the training script."""
    return getattr(algo, "env_runner_group", None) or getattr(algo, "workers", None)


def load_policy_weights_from_checkpoint(
    checkpoint_dir: Path,
    policy_id: str,
) -> dict[str, Any]:
    """Load one policy state directly from an RLlib algorithm checkpoint."""
    state_path = checkpoint_dir / "policies" / policy_id / "policy_state.pkl"
    if not state_path.is_file():
        raise FileNotFoundError(f"Policy checkpoint not found: {state_path}")
    with state_path.open("rb") as file:
        state = pickle.load(file)
    weights = state.get("weights")
    if not isinstance(weights, dict) or not weights:
        raise ValueError(f"Checkpoint has no {policy_id} weights: {state_path}")
    return weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--initialization",
        choices=("scratch", "mappo", "bc"),
        required=True,
        help=(
            "scratch=random MAPPO, mappo=checkpoint policy with a fresh PPO "
            "optimizer, bc=BC actor copied into the MAPPO architecture."
        ),
    )
    parser.add_argument("--mappo-checkpoint", type=Path, default=DEFAULT_MAPPO_CHECKPOINT)
    parser.add_argument("--bc-checkpoint", type=Path, default=DEFAULT_BC_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--difficulty",
        type=int,
        choices=range(1, len(OBSTACLE_CURRICULUM_LEVELS) + 1),
        required=True,
    )
    parser.add_argument("--success-min-coverage-ratio", type=float, default=0.6)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--train-batch-size", type=int, default=1200)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--ppo-epochs", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--num-env-runners", type=int, default=4)
    parser.add_argument("--rollout-fragment-length", type=int, default=100)
    parser.add_argument("--sample-timeout-s", type=float, default=300.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coeff", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--method-name")
    parser.add_argument("--num-gpus", type=float, default=1.0)
    parser.add_argument("--learning-curve-output", type=Path)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "iterations",
        "train_batch_size",
        "minibatch_size",
        "ppo_epochs",
        "eval_every",
        "eval_episodes",
        "rollout_fragment_length",
        "sample_timeout_s",
        "lr",
        "gamma",
        "gae_lambda",
        "clip_ratio",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.num_env_runners < 0:
        raise ValueError("--num-env-runners cannot be negative.")
    if args.minibatch_size > args.train_batch_size:
        raise ValueError("--minibatch-size cannot exceed --train-batch-size.")
    if not 0.0 < args.success_min_coverage_ratio <= 1.0:
        raise ValueError("--success-min-coverage-ratio must be in (0, 1].")
    if args.method_name is None:
        args.method_name = {
            "scratch": "mappo_scratch",
            "mappo": "mappo_pretrained",
            "bc": "bc_init_ppo",
        }[args.initialization]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_algorithm(args: argparse.Namespace, env_config: dict[str, Any]):
    temp_env = env_creator(env_config)
    try:
        observation_space = temp_env.observation_space["drone_0"]
        action_space = temp_env.action_space["drone_0"]
    finally:
        temp_env.close()

    model_config = drone_policy_model_config()
    if args.initialization == "bc":
        model_config["custom_model_config"]["action_history_scale"] = (
            drone_action_scale(env_config)
        )
        model_config["custom_model_config"]["vector_last"] = True

    policies = {
        "drone_policy": (
            None,
            observation_space,
            action_space,
            {"model": model_config},
        )
    }

    def policy_mapping_fn(agent_id, episode, **kwargs):
        del agent_id, episode, kwargs
        return "drone_policy"

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .framework("torch")
        .callbacks(DefaultCallbacks)
        .environment(env=ENV_NAME, env_config=env_config)
        .env_runners(
            num_env_runners=args.num_env_runners,
            num_envs_per_env_runner=1,
            rollout_fragment_length=args.rollout_fragment_length,
            sample_timeout_s=args.sample_timeout_s,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["drone_policy"],
        )
        .resources(num_gpus=args.num_gpus)
        .training(
            train_batch_size=args.train_batch_size,
            minibatch_size=args.minibatch_size,
            num_epochs=args.ppo_epochs,
            lr=args.lr,
            gamma=args.gamma,
            lambda_=args.gae_lambda,
            grad_clip=1.0,
            clip_param=args.clip_ratio,
            entropy_coeff=args.entropy_coeff,
            kl_target=0.01,
        )
        .debugging(seed=args.seed, log_level="WARN")
    )
    return config.build()


def load_bc_checkpoint(path: Path, device: torch.device) -> DroneBehaviorCloningPolicy:
    payload = torch.load(path.expanduser().resolve(), map_location=device, weights_only=False)
    if payload.get("model_type") != "drone_behavior_cloning":
        raise ValueError(f"Not a drone BC checkpoint: {path}")
    model = DroneBehaviorCloningPolicy(**payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state_dict"])
    return model


def initialize_actor(args: argparse.Namespace, algo) -> None:
    if args.initialization == "scratch":
        return
    if args.initialization == "mappo":
        weights = load_policy_weights_from_checkpoint(
            args.mappo_checkpoint.expanduser().resolve(),
            "drone_policy",
        )
        algo.set_weights({"drone_policy": weights})
    else:
        policy = algo.get_policy("drone_policy")
        model = policy.model
        device = next(model.parameters()).device
        bc = load_bc_checkpoint(args.bc_checkpoint, device)
        model.global_map_encoder.load_state_dict(
            bc.encoder.global_map_encoder.state_dict()
        )
        model.local_map_encoder.load_state_dict(
            bc.encoder.local_map_encoder.state_dict()
        )
        model.encoder.load_state_dict(bc.encoder.fusion.state_dict())
        model.policy_head.load_state_dict(bc.action_head.state_dict())
        algo.set_weights({"drone_policy": policy.get_weights()})

    runner_group = get_env_runner_group(algo)
    if runner_group is not None and hasattr(runner_group, "sync_weights"):
        runner_group.sync_weights(policies=["drone_policy"])


def evaluate(algo, env_config: dict[str, Any], args: argparse.Namespace) -> dict[str, float]:
    env = HeMAC_v0.env(**env_config)
    results = []
    try:
        for episode_index in range(args.eval_episodes):
            results.append(
                run_episode(
                    env,
                    algo,
                    None,
                    controller="mappo",
                    difficulty=args.difficulty,
                    seed=args.seed + args.difficulty * 1_000_000 + episode_index,
                    action_scale=1.0,
                    device=torch.device("cpu"),
                    task_definition="drone",
                    success_min_coverage_ratio=args.success_min_coverage_ratio,
                )
            )
    finally:
        env.close()
    summary = summarize(results)
    return {
        "success": float(summary["success_rate"]),
        "goal_found": float(summary["drone_goal_found_rate"]),
        "fatal_crash": float(summary["fatal_crash_rate"]),
        "drone_crash": float(summary["drone_crash_rate"]),
        "coverage": float(summary["mean_coverage_ratio"]),
        "cycles": float(summary["mean_cycles"]),
    }


def curve_point(
    args: argparse.Namespace,
    metrics: dict[str, float],
    env_steps: int,
    iteration: int,
) -> dict[str, Any]:
    return {
        "method": args.method_name,
        "seed": args.seed,
        "difficulty": args.difficulty,
        "joint_env_steps": int(env_steps),
        "iteration": int(iteration),
        "success_rate": metrics["success"],
        "goal_found_rate": metrics["goal_found"],
        "fatal_crash_rate": metrics["fatal_crash"],
        "drone_crash_rate": metrics["drone_crash"],
        "observer_crash_rate": 0.0,
        "mean_coverage_ratio": metrics["coverage"],
        "mean_cycles": metrics["cycles"],
    }


def sampled_env_steps(result: dict[str, Any]) -> int:
    for container in (result, result.get("env_runners", {})):
        for key in (
            "num_env_steps_sampled_lifetime",
            "num_env_steps_sampled",
        ):
            if key in container:
                return int(container[key])
    raise KeyError("RLlib result has no lifetime environment-step counter.")


def training_mean_reward(result: dict[str, Any]) -> float | None:
    """Read episode reward from both current and legacy RLlib result layouts."""
    for container in (result.get("env_runners", {}), result):
        value = container.get("episode_reward_mean")
        if value is not None:
            return float(value)
    return None


def serializable_hyperparameters(args: argparse.Namespace) -> dict[str, Any]:
    """Convert path-valued CLI arguments before writing curve metadata."""
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    curve_path = (
        args.learning_curve_output.expanduser().resolve()
        if args.learning_curve_output is not None
        else args.output_dir / f"learning_curve_seed_{args.seed}.json"
    )
    resolved_mappo, checkpoint_env_config = load_checkpoint_env_config(
        args.mappo_checkpoint
    )
    env_config = build_env_config(
        checkpoint_env_config,
        args.difficulty,
        args.success_min_coverage_ratio,
    )
    env_config["render_mode"] = None
    env_config["log_step_rewards"] = False

    ray.init(ignore_reinit_error=True, include_dashboard=False)
    register_hemac_rllib_models()
    register_env(ENV_NAME, env_creator)
    algo = build_algorithm(args, env_config)
    try:
        initialize_actor(args, algo)
        baseline = evaluate(algo, env_config, args)
        append_curve_points(
            curve_path,
            [curve_point(args, baseline, 0, 0)],
            metadata={
                "initialization": args.initialization,
                "evaluation_episodes": args.eval_episodes,
                "environment_checkpoint": str(resolved_mappo),
                "bc_checkpoint": str(args.bc_checkpoint.expanduser().resolve()),
                "success_definition": (
                    "drone_goal_found_and_coverage_without_fatal_crash"
                ),
                "success_min_coverage_ratio": args.success_min_coverage_ratio,
                "training_step_definition": "RLlib sampled environment steps",
                "hyperparameters": serializable_hyperparameters(args),
            },
        )
        print(
            f"BASELINE method={args.method_name} success={baseline['success']:.3f} "
            f"crash={baseline['fatal_crash']:.3f}"
        )
        initial_sampled = None
        for iteration in range(1, args.iterations + 1):
            result = algo.train()
            lifetime_steps = sampled_env_steps(result)
            if initial_sampled is None:
                initial_sampled = max(0, lifetime_steps - args.train_batch_size)
            env_steps = max(0, lifetime_steps - initial_sampled)
            mean_reward = training_mean_reward(result)
            reward_text = "N/A" if mean_reward is None else f"{mean_reward:.3f}"
            print(
                f"iter={iteration:03d}/{args.iterations} steps={env_steps} "
                f"reward={reward_text}"
            )
            if iteration % args.eval_every != 0:
                continue
            metrics = evaluate(algo, env_config, args)
            append_curve_points(
                curve_path,
                [curve_point(args, metrics, env_steps, iteration)],
            )
            print(
                f"EVAL success={metrics['success']:.3f} "
                f"goal={metrics['goal_found']:.3f} "
                f"crash={metrics['fatal_crash']:.3f}"
            )
        checkpoint = algo.save(str(args.output_dir / "final_checkpoint"))
        checkpoint_path = getattr(checkpoint, "checkpoint", checkpoint)
        print(f"Saved final checkpoint: {checkpoint_path}")
        print(f"Saved learning curve: {curve_path}")
    finally:
        algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
