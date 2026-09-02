"""Evaluate a drone BC checkpoint with an optional frozen MAPPO observer."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch
from ray.tune.registry import register_env


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from hemac import HeMAC_v0
from hemac.curriculum_config import OBSTACLE_CURRICULUM_LEVELS
from hemac.rllib_policy import register_hemac_rllib_models
from skill_discovery.collect_offline_data import (
    CHECKPOINT_PATH as DEFAULT_MAPPO_CHECKPOINT,
)
from skill_discovery.collect_offline_data import (
    ENV_NAME,
    agent_found_goal,
    build_collection_env_config,
    convert_observation,
    env_creator,
    get_core_env,
    load_inference_algorithm,
)
from skill_discovery.models import DroneBehaviorCloningPolicy
from skill_discovery.drone_task import (
    DRONE_SKILL_SUCCESS_MIN_COVERAGE_RATIO,
    classify_drone_skill_outcome,
)


DEFAULT_BC_CHECKPOINT = (
    PROJECT_ROOT / "src/skill_discovery/checkpoints/bc_checkpoints/drone_bc_best.pt"
)


@dataclass(frozen=True)
class EpisodeResult:
    """Store task-level metrics from one environment rollout."""

    controller: str
    difficulty: int
    seed: int
    cycles: int
    success: bool
    goal_found: bool
    drone_goal_found: bool
    fatal_crash: bool
    drone_crash: bool
    observer_crash: bool
    coverage_ratio: float
    mission_success: bool = False
    drone_task_success: bool = False


def parse_args() -> argparse.Namespace:
    """Parse checkpoints, held-out seeds, and comparison settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc-checkpoint", type=Path, default=DEFAULT_BC_CHECKPOINT)
    parser.add_argument(
        "--mappo-checkpoint",
        type=Path,
        default=DEFAULT_MAPPO_CHECKPOINT,
        help="Provides the frozen observer policy and optional drone baseline.",
    )
    parser.add_argument(
        "--difficulties",
        type=int,
        nargs="+",
        choices=range(1, len(OBSTACLE_CURRICULUM_LEVELS) + 1),
        default=(1, 2, 3),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of held-out seeds evaluated per difficulty and controller.",
    )
    parser.add_argument("--base-seed", type=int, default=100_000)
    parser.add_argument(
        "--task-definition",
        choices=("mission", "drone"),
        default="mission",
        help="Report success using observer arrival or the drone exploration task.",
    )
    parser.add_argument(
        "--success-min-coverage-ratio",
        type=float,
        default=DRONE_SKILL_SUCCESS_MIN_COVERAGE_RATIO,
    )
    parser.add_argument(
        "--compare-mappo",
        action="store_true",
        help="Also evaluate the original MAPPO drone policy on identical seeds.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print summaries only instead of every episode.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    """Select the requested inference device."""
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def load_bc_policy(
    checkpoint_path: Path,
    device: torch.device,
) -> DroneBehaviorCloningPolicy:
    """Reconstruct a BC policy from its standalone checkpoint."""
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"BC checkpoint not found: {checkpoint_path}")
    payload = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if payload.get("model_type") != "drone_behavior_cloning":
        raise ValueError(f"Not a drone BC checkpoint: {checkpoint_path}")
    model_config = payload.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError(f"BC checkpoint has no model_config: {checkpoint_path}")
    policy = DroneBehaviorCloningPolicy(**model_config).to(device)
    policy.load_state_dict(payload["model_state_dict"])
    policy.eval()
    print(
        f"Loaded BC checkpoint epoch={payload.get('epoch')} "
        f"val_mse={payload.get('metrics', {}).get('validation', {}).get('mse')}"
    )
    return policy


def drone_action_scale(env_config: dict[str, Any]) -> float:
    """Read the scalar action normalization used by the offline Dataset."""
    drone_config = env_config.get("drone_config") or {}
    scale = float(drone_config.get("drone_max_speed", 25.0))
    if scale <= 0:
        raise ValueError(f"drone_max_speed must be positive, got {scale}.")
    return scale


def compute_bc_action(
    policy: DroneBehaviorCloningPolicy,
    observation: dict[str, Any],
    *,
    action_scale: float,
    action_space,
    device: torch.device,
) -> np.ndarray:
    """Convert one raw drone observation into a physical environment action."""
    converted = convert_observation(observation, "drone")
    global_map = torch.from_numpy(converted["global_map"]).unsqueeze(0).to(device)
    local_map = torch.from_numpy(converted["local_map"]).unsqueeze(0).to(device)
    action_history = (
        torch.from_numpy(converted["action_history"]).unsqueeze(0).to(device)
        / action_scale
    )
    with torch.inference_mode():
        normalized_action = policy(
            global_map,
            local_map,
            action_history,
        )[0]
    action = normalized_action.cpu().numpy().astype(np.float32) * action_scale
    return np.ascontiguousarray(
        np.clip(action, action_space.low, action_space.high),
        dtype=np.float32,
    )


def compute_cycle_actions(
    env,
    algo,
    bc_policy: DroneBehaviorCloningPolicy,
    controller: str,
    action_scale: float,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Compute all actions from one shared AEC cycle-start state."""
    actions = {}
    for agent_id in env.possible_agents:
        observation = env.observe(agent_id)
        if agent_id.startswith("observer_"):
            action = algo.compute_single_action(
                observation=observation,
                policy_id="observer_policy",
                explore=False,
            )
        elif controller == "mappo":
            action = algo.compute_single_action(
                observation=observation,
                policy_id="drone_policy",
                explore=False,
            )
        elif agent_id.startswith("drone_"):
            action = compute_bc_action(
                bc_policy,
                observation,
                action_scale=action_scale,
                action_space=env.action_space(agent_id),
                device=device,
            )
        else:
            raise ValueError(f"Unsupported agent ID: {agent_id!r}")
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        actions[agent_id] = np.ascontiguousarray(action_array)
    return actions


def run_episode(
    env,
    algo,
    bc_policy: DroneBehaviorCloningPolicy,
    *,
    controller: str,
    difficulty: int,
    seed: int,
    action_scale: float,
    device: torch.device,
    task_definition: str = "mission",
    success_min_coverage_ratio: float = DRONE_SKILL_SUCCESS_MIN_COVERAGE_RATIO,
) -> EpisodeResult:
    """Run one deterministic drone episode with any configured observer."""
    env.reset(seed=seed)
    core_env = get_core_env(env)
    last_agent_id = env.possible_agents[-1]
    cached_actions: dict[str, np.ndarray] = {}
    final_info: dict[str, Any] = {}
    cycles = 0

    for agent_id in env.agent_iter():
        _, _, termination, truncation, info = env.last()
        if info:
            final_info.update(info)
        if termination or truncation:
            env.step(None)
            continue

        if not cached_actions:
            cached_actions = compute_cycle_actions(
                env,
                algo,
                bc_policy,
                controller,
                action_scale,
                device,
            )
        env.step(cached_actions[agent_id])
        if (
            agent_id == last_agent_id
            or bool(core_env.terminate)
            or bool(core_env.truncate)
        ):
            cycles += 1
            cached_actions = {}

    if hasattr(core_env, "build_episode_info"):
        final_info.update(core_env.build_episode_info())
    drone_goal_found = any(
        agent_id.startswith("drone_") and agent_found_goal(core_env, agent_id)
        for agent_id in env.possible_agents
    )
    coverage_method = getattr(core_env, "current_coverage_ratio", None)
    coverage_ratio = float(coverage_method()) if callable(coverage_method) else 0.0
    drone_task_success = classify_drone_skill_outcome(
        drone_goal_found,
        coverage_ratio,
        success_min_coverage_ratio,
    ) == "success"
    mission_success = bool(final_info.get("success", core_env.mission_success))
    task_success = (
        drone_task_success if task_definition == "drone" else mission_success
    )
    return EpisodeResult(
        controller=controller,
        difficulty=difficulty,
        seed=seed,
        cycles=cycles,
        success=task_success,
        mission_success=mission_success,
        drone_task_success=drone_task_success,
        goal_found=bool(final_info.get("goal_found", core_env.found_goal)),
        drone_goal_found=drone_goal_found,
        fatal_crash=bool(final_info.get("fatal_crash", core_env.collided)),
        drone_crash=bool(final_info.get("drone_crash", core_env.drone_crash)),
        observer_crash=bool(
            final_info.get("observer_crash", core_env.observer_crash)
        ),
        coverage_ratio=coverage_ratio,
    )


def summarize(results: list[EpisodeResult]) -> dict[str, float | int]:
    """Aggregate task-level rates and means for one controller/difficulty."""
    if not results:
        raise ValueError("Cannot summarize an empty rollout list.")
    count = len(results)

    def rate(name: str) -> float:
        return sum(bool(getattr(result, name)) for result in results) / count

    return {
        "episodes": count,
        "success_rate": rate("success"),
        "mission_success_rate": rate("mission_success"),
        "drone_task_success_rate": rate("drone_task_success"),
        "goal_found_rate": rate("goal_found"),
        "drone_goal_found_rate": rate("drone_goal_found"),
        "fatal_crash_rate": rate("fatal_crash"),
        "drone_crash_rate": rate("drone_crash"),
        "observer_crash_rate": rate("observer_crash"),
        "mean_coverage_ratio": round(
            sum(r.coverage_ratio for r in results) / count,
            12,
        ),
        "mean_cycles": round(sum(r.cycles for r in results) / count, 12),
    }


def print_summary(controller: str, difficulty: int, summary: dict[str, Any]) -> None:
    """Print the metrics needed to judge whether BC survives online rollout."""
    print(
        f"SUMMARY controller={controller} difficulty={difficulty} "
        f"episodes={summary['episodes']} "
        f"success={summary['success_rate']:.3f} "
        f"drone_task_success={summary['drone_task_success_rate']:.3f} "
        f"goal_found={summary['goal_found_rate']:.3f} "
        f"drone_goal_found={summary['drone_goal_found_rate']:.3f} "
        f"fatal_crash={summary['fatal_crash_rate']:.3f} "
        f"drone_crash={summary['drone_crash_rate']:.3f} "
        f"observer_crash={summary['observer_crash_rate']:.3f} "
        f"coverage={summary['mean_coverage_ratio']:.3f} "
        f"cycles={summary['mean_cycles']:.1f}"
    )


def main() -> None:
    """Evaluate BC drones and optionally compare the MAPPO expert."""
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if not 0.0 <= args.success_min_coverage_ratio <= 1.0:
        raise ValueError("--success-min-coverage-ratio must be in [0, 1].")
    device = resolve_device(args.device)
    bc_policy = load_bc_policy(args.bc_checkpoint, device)

    ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=1)
    register_hemac_rllib_models()
    register_env(ENV_NAME, env_creator)
    algo = load_inference_algorithm(args.mappo_checkpoint.expanduser().resolve())
    checkpoint_env_config = getattr(algo.config, "env_config", {}) or {}
    controllers = ("bc", "mappo") if args.compare_mappo else ("bc",)
    all_results: list[EpisodeResult] = []
    summaries: dict[str, dict[str, Any]] = {}

    try:
        for difficulty in args.difficulties:
            env_config = build_collection_env_config(
                checkpoint_env_config,
                difficulty,
            )
            action_scale = drone_action_scale(env_config)
            print(
                f"Difficulty {difficulty}: obstacles="
                f"{env_config.get('min_obstacles')}-{env_config.get('max_obstacles')}, "
                f"static={env_config.get('n_static_obstacles')}, "
                f"speed={env_config.get('obstacle_min_speed')}-"
                f"{env_config.get('obstacle_max_speed')}, "
                f"drone_action_scale={action_scale:g}"
            )
            for controller in controllers:
                env = HeMAC_v0.env(**env_config)
                controller_results = []
                try:
                    for episode_index in range(args.episodes):
                        seed = args.base_seed + difficulty * 100_000 + episode_index
                        result = run_episode(
                            env,
                            algo,
                            bc_policy,
                            controller=controller,
                            difficulty=difficulty,
                            seed=seed,
                            action_scale=action_scale,
                            device=device,
                            task_definition=args.task_definition,
                            success_min_coverage_ratio=(
                                args.success_min_coverage_ratio
                            ),
                        )
                        controller_results.append(result)
                        all_results.append(result)
                        if not args.quiet:
                            print(
                                f"episode={episode_index + 1:03d}/{args.episodes} "
                                f"controller={controller} difficulty={difficulty} "
                                f"seed={seed} success={int(result.success)} "
                                f"drone_goal_found={int(result.drone_goal_found)} "
                                f"drone_crash={int(result.drone_crash)} "
                                f"coverage={result.coverage_ratio:.3f} "
                                f"cycles={result.cycles}"
                            )
                finally:
                    env.close()
                summary = summarize(controller_results)
                summaries[f"{controller}/difficulty_{difficulty}"] = summary
                print_summary(controller, difficulty, summary)
    finally:
        algo.stop()
        ray.shutdown()

    if args.output_json is not None:
        output_path = args.output_json.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "success_definition": args.task_definition,
                    "success_min_coverage_ratio": (
                        args.success_min_coverage_ratio
                    ),
                    "bc_checkpoint": str(args.bc_checkpoint.expanduser().resolve()),
                    "mappo_checkpoint": str(
                        args.mappo_checkpoint.expanduser().resolve()
                    ),
                    "summaries": summaries,
                    "episodes": [asdict(result) for result in all_results],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Saved rollout results: {output_path}")


if __name__ == "__main__":
    main()
