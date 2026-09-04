"""Evaluate HiSSD drones online with an optional frozen MAPPO observer."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
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
from skill_discovery.evaluate_drone_bc import (
    DEFAULT_BC_CHECKPOINT,
    EpisodeResult,
    compute_bc_action,
    drone_action_scale,
    load_bc_policy,
    print_summary,
    summarize,
)
from skill_discovery.drone_task import (
    DRONE_SKILL_SUCCESS_MIN_COVERAGE_RATIO,
    classify_drone_skill_outcome,
)
from skill_discovery.hissd_models import HeMACHISSD
from skill_discovery.models import ObserverTaskResidualPolicy
from skill_discovery.visualize_hissd_skills import load_hissd_model, resolve_device


DEFAULT_HISSD_CHECKPOINT = (
    PROJECT_ROOT
    / "src/skill_discovery/checkpoints/hissd_joint_online_checkpoints/hissd_online_best.pt"
)


def parse_args() -> argparse.Namespace:
    """Parse checkpoints, task difficulties, and comparison settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hissd-checkpoint", type=Path, default=DEFAULT_HISSD_CHECKPOINT)
    parser.add_argument(
        "--reference-hissd-checkpoint",
        type=Path,
        help="Optionally compare the adapted model with a source HiSSD checkpoint.",
    )
    parser.add_argument("--mappo-checkpoint", type=Path, default=DEFAULT_MAPPO_CHECKPOINT)
    parser.add_argument("--bc-checkpoint", type=Path, default=DEFAULT_BC_CHECKPOINT)
    parser.add_argument(
        "--difficulties",
        type=int,
        nargs="+",
        choices=range(1, len(OBSTACLE_CURRICULUM_LEVELS) + 1),
        default=(1, 2, 3, 4, 5, 6),
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=200_000)
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
        "--compare-baselines",
        action="store_true",
        help="Also evaluate BC and MAPPO drones on exactly the same seeds.",
    )
    parser.add_argument(
        "--compare-mappo",
        action="store_true",
        help="Evaluate only the MAPPO checkpoint baseline in addition to HiSSD.",
    )
    parser.add_argument(
        "--compare-bc",
        action="store_true",
        help="Evaluate only the BC checkpoint baseline in addition to HiSSD.",
    )
    parser.add_argument(
        "--ablate-skills",
        action="store_true",
        help=(
            "Also evaluate HiSSD with half task/common residual, without task skill, "
            "without common skill, and with only its BC-compatible base action head."
        ),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _drone_observation_batch(
    env,
    drone_ids: list[str],
    action_scale: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Convert one shared cycle state to `[1,A,...]` model tensors."""
    converted = [convert_observation(env.observe(agent_id), "drone") for agent_id in drone_ids]
    return {
        "global_map": torch.from_numpy(
            np.stack([item["global_map"] for item in converted])
        ).unsqueeze(0).to(device),
        "local_map": torch.from_numpy(
            np.stack([item["local_map"] for item in converted])
        ).unsqueeze(0).to(device),
        "action_history": (
            torch.from_numpy(
                np.stack([item["action_history"] for item in converted])
            ).unsqueeze(0).to(device)
            / action_scale
        ),
    }


def load_observer_residual(
    payload: dict[str, Any], device: torch.device
) -> tuple[ObserverTaskResidualPolicy | None, float]:
    """Load the optional observer residual embedded by joint online PPO."""
    config = payload.get("online_observer_residual_config")
    state = payload.get("online_observer_residual_state_dict")
    if config is None or state is None:
        return None, 0.0
    policy = ObserverTaskResidualPolicy(**config).to(device)
    policy.load_state_dict(state)
    policy.eval()
    scale = float(payload.get("online_observer_residual_scale", 0.35))
    return policy, scale


def _observer_observation_batch(
    observation: dict[str, np.ndarray], device: torch.device
) -> dict[str, torch.Tensor]:
    converted = convert_observation(observation, "observer")
    return {
        name: torch.from_numpy(converted[name]).unsqueeze(0).to(device)
        for name in ("global_map", "local_map", "action_history")
    }


@torch.inference_mode()
def compute_hissd_actions(
    env,
    model: HeMACHISSD,
    state: dict[str, torch.Tensor],
    *,
    action_scale: float,
    device: torch.device,
    skill_ablation: str | None = None,
) -> tuple[
    dict[str, np.ndarray], dict[str, torch.Tensor], torch.Tensor
]:
    """Infer actions jointly for all drones and advance recurrent skill state."""
    drone_ids = [
        agent_id for agent_id in env.possible_agents if agent_id.startswith("drone_")
    ]
    if len(drone_ids) != model.agent_count:
        raise ValueError(
            f"Checkpoint expects {model.agent_count} drones, environment has "
            f"{len(drone_ids)}."
        )
    observations = _drone_observation_batch(env, drone_ids, action_scale, device)
    valid_mask = torch.ones(1, model.agent_count, dtype=torch.bool, device=device)
    outputs, next_state = model.inference_step(observations, valid_mask, state)
    observer_task_skill = outputs["task_skills"].mean(dim=1)
    if skill_ablation is None:
        normalized_action_tensor = outputs["actions"]
    else:
        features = model.encode_observations(observations)
        common_skills = outputs["common_skills"]
        task_skills = outputs["task_skills"]
        if skill_ablation == "task":
            task_skills = torch.zeros_like(task_skills)
            observer_task_skill = torch.zeros_like(observer_task_skill)
            normalized_action_tensor = model.decode_actions(
                features, common_skills, task_skills
            )
        elif skill_ablation == "half_task":
            observer_task_skill = 0.5 * observer_task_skill
            no_task_actions = model.decode_actions(
                features, common_skills, torch.zeros_like(task_skills)
            )
            normalized_action_tensor = no_task_actions + 0.5 * (
                outputs["actions"] - no_task_actions
            )
        elif skill_ablation == "common":
            common_skills = torch.zeros_like(common_skills)
            normalized_action_tensor = model.decode_actions(
                features, common_skills, task_skills
            )
        elif skill_ablation == "half_common":
            no_common_actions = model.decode_actions(
                features, torch.zeros_like(common_skills), task_skills
            )
            normalized_action_tensor = no_common_actions + 0.5 * (
                outputs["actions"] - no_common_actions
            )
        elif skill_ablation == "all":
            observer_task_skill = torch.zeros_like(observer_task_skill)
            normalized_action_tensor = torch.tanh(
                model.action_decoder.base_action_head(features)
            )
        else:
            raise ValueError(f"Unsupported skill ablation: {skill_ablation!r}")
    normalized_actions = normalized_action_tensor[0].cpu().numpy()
    actions = {}
    for index, agent_id in enumerate(drone_ids):
        action_space = env.action_space(agent_id)
        action = normalized_actions[index].astype(np.float32) * action_scale
        actions[agent_id] = np.ascontiguousarray(
            np.clip(action, action_space.low, action_space.high),
            dtype=np.float32,
        )
    return actions, next_state, observer_task_skill


def compute_cycle_actions(
    env,
    algo,
    hissd_model: HeMACHISSD,
    hissd_state: dict[str, torch.Tensor],
    *,
    controller: str,
    action_scale: float,
    device: torch.device,
    bc_policy=None,
    observer_residual_policy: ObserverTaskResidualPolicy | None = None,
    observer_residual_scale: float = 0.0,
) -> tuple[dict[str, np.ndarray], dict[str, torch.Tensor]]:
    """Compute one observer action and all drone actions from one world state."""
    if controller.startswith("hissd"):
        skill_ablation = {
            "hissd": None,
            "hissd_reference": None,
            "hissd_no_task": "task",
            "hissd_half_task": "half_task",
            "hissd_no_common": "common",
            "hissd_half_common": "half_common",
            "hissd_base": "all",
        }.get(controller)
        if controller not in {"hissd", "hissd_reference"} and skill_ablation is None:
            raise ValueError(f"Unsupported HiSSD controller: {controller!r}")
        actions, hissd_state, observer_task_skill = compute_hissd_actions(
            env,
            hissd_model,
            hissd_state,
            action_scale=action_scale,
            device=device,
            skill_ablation=skill_ablation,
        )
    else:
        actions = {}
        observer_task_skill = None
        for agent_id in env.possible_agents:
            if not agent_id.startswith("drone_"):
                continue
            observation = env.observe(agent_id)
            if controller == "mappo":
                action = algo.compute_single_action(
                    observation=observation,
                    policy_id="drone_policy",
                    explore=False,
                )
            elif controller == "bc" and bc_policy is not None:
                action = compute_bc_action(
                    bc_policy,
                    observation,
                    action_scale=action_scale,
                    action_space=env.action_space(agent_id),
                    device=device,
                )
            else:
                raise ValueError(f"Unsupported controller: {controller!r}")
            actions[agent_id] = np.ascontiguousarray(
                np.asarray(action, dtype=np.float32).reshape(-1)
            )

    for agent_id in env.possible_agents:
        if not agent_id.startswith("observer_"):
            continue
        action = algo.compute_single_action(
            observation=env.observe(agent_id),
            policy_id="observer_policy",
            explore=False,
        )
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        use_observer_residual = (
            observer_residual_policy is not None
            and observer_task_skill is not None
            and controller not in {"hissd_reference", "hissd_base"}
        )
        if use_observer_residual:
            observation = _observer_observation_batch(env.observe(agent_id), device)
            residual_mean, _ = observer_residual_policy(
                observation["global_map"],
                observation["local_map"],
                observation["action_history"],
                observer_task_skill,
            )
            action_space = env.action_space(agent_id)
            high = np.asarray(action_space.high, dtype=np.float32)
            normalized = action / high
            delta = (
                torch.tanh(residual_mean[0]).detach().cpu().numpy()
                * observer_residual_scale
            )
            action = np.clip(normalized + delta, -1.0, 1.0) * high
            action = np.clip(action, action_space.low, action_space.high)
        actions[agent_id] = np.ascontiguousarray(action, dtype=np.float32)
    return actions, hissd_state


def run_episode(
    env,
    algo,
    hissd_model: HeMACHISSD,
    *,
    controller: str,
    difficulty: int,
    seed: int,
    action_scale: float,
    device: torch.device,
    bc_policy=None,
    observer_residual_policy: ObserverTaskResidualPolicy | None = None,
    observer_residual_scale: float = 0.0,
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
    hissd_state = hissd_model.initial_inference_state(device=device)

    for agent_id in env.agent_iter():
        _, _, termination, truncation, info = env.last()
        if info:
            final_info.update(info)
        if termination or truncation:
            env.step(None)
            continue
        if not cached_actions:
            cached_actions, hissd_state = compute_cycle_actions(
                env,
                algo,
                hissd_model,
                hissd_state,
                controller=controller,
                action_scale=action_scale,
                device=device,
                bc_policy=bc_policy,
                observer_residual_policy=observer_residual_policy,
                observer_residual_scale=observer_residual_scale,
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
        observer_crash=bool(final_info.get("observer_crash", core_env.observer_crash)),
        coverage_ratio=coverage_ratio,
    )


def main() -> None:
    """Evaluate HiSSD and optional baselines on matched environment seeds."""
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if not 0.0 <= args.success_min_coverage_ratio <= 1.0:
        raise ValueError("--success-min-coverage-ratio must be in [0, 1].")
    device = resolve_device(args.device)
    hissd_model, hissd_payload = load_hissd_model(args.hissd_checkpoint, device)
    observer_residual_policy, observer_residual_scale = load_observer_residual(
        hissd_payload, device
    )
    controllers = ["hissd"]
    reference_hissd_model = None
    if args.reference_hissd_checkpoint is not None:
        reference_hissd_model, reference_payload = load_hissd_model(
            args.reference_hissd_checkpoint, device
        )
        controllers.append("hissd_reference")
        print(
            "Loaded reference HiSSD checkpoint "
            f"epoch={reference_payload.get('epoch')}"
        )
    if args.ablate_skills:
        controllers.extend(
            (
                "hissd_half_task",
                "hissd_no_task",
                "hissd_half_common",
                "hissd_no_common",
                "hissd_base",
            )
        )
    if args.compare_baselines or args.compare_bc:
        controllers.append("bc")
    if args.compare_baselines or args.compare_mappo:
        controllers.append("mappo")
    controllers = tuple(controllers)
    bc_policy = load_bc_policy(args.bc_checkpoint, device) if "bc" in controllers else None
    print(
        f"Loaded HiSSD checkpoint epoch={hissd_payload.get('epoch')} on {device}; "
        f"controllers={','.join(controllers)}; "
        f"observer_residual={observer_residual_policy is not None}"
    )

    ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=1)
    register_hemac_rllib_models()
    register_env(ENV_NAME, env_creator)
    algo = load_inference_algorithm(args.mappo_checkpoint.expanduser().resolve())
    checkpoint_env_config = getattr(algo.config, "env_config", {}) or {}
    all_results: list[EpisodeResult] = []
    summaries: dict[str, dict[str, Any]] = {}

    try:
        for difficulty in args.difficulties:
            env_config = build_collection_env_config(checkpoint_env_config, difficulty)
            action_scale = drone_action_scale(env_config)
            print(
                f"Difficulty {difficulty}: obstacles="
                f"{env_config.get('min_obstacles')}-{env_config.get('max_obstacles')}, "
                f"static={env_config.get('n_static_obstacles')}, "
                f"speed={env_config.get('obstacle_min_speed')}-"
                f"{env_config.get('obstacle_max_speed')}"
            )
            for controller in controllers:
                env = HeMAC_v0.env(**env_config)
                controller_results = []
                controller_model = (
                    reference_hissd_model
                    if controller == "hissd_reference"
                    else hissd_model
                )
                try:
                    for episode_index in range(args.episodes):
                        seed = args.base_seed + difficulty * 100_000 + episode_index
                        result = run_episode(
                            env,
                            algo,
                            controller_model,
                            controller=controller,
                            difficulty=difficulty,
                            seed=seed,
                            action_scale=action_scale,
                            device=device,
                            bc_policy=bc_policy,
                            observer_residual_policy=observer_residual_policy,
                            observer_residual_scale=observer_residual_scale,
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
                                f"success={int(result.success)} "
                                f"drone_task_success={int(result.drone_task_success)} "
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
                    "success_definition": (
                        "observer_goal_arrival"
                        if args.task_definition == "mission"
                        else "drone_goal_found_and_coverage"
                    ),
                    "drone_task_success_definition": (
                        "drone_goal_found and full_map_coverage_ratio >= "
                        f"{args.success_min_coverage_ratio}"
                    ),
                    "hissd_half_task_definition": (
                        "no_task_action + 0.5 * (full_hissd_action - no_task_action)"
                    ),
                    "hissd_half_common_definition": (
                        "no_common_action + 0.5 * "
                        "(full_hissd_action - no_common_action)"
                    ),
                    "hissd_checkpoint": str(args.hissd_checkpoint.expanduser().resolve()),
                    "observer_residual_enabled": observer_residual_policy is not None,
                    "observer_residual_scale": observer_residual_scale,
                    "reference_hissd_checkpoint": (
                        str(args.reference_hissd_checkpoint.expanduser().resolve())
                        if args.reference_hissd_checkpoint is not None
                        else None
                    ),
                    "mappo_checkpoint": str(args.mappo_checkpoint.expanduser().resolve()),
                    "bc_checkpoint": str(args.bc_checkpoint.expanduser().resolve()),
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
