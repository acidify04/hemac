"""Collect episode-separated offline trajectories from a trained MAPPO policy.

The saved maps use PyTorch's CNN layout (batch, channels, height, width).
Each episode is written independently so interrupted collection keeps all
previously completed episodes.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.utils.checkpoints import get_checkpoint_info
from ray.tune.registry import register_env


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from hemac import HeMAC_v0
from hemac.rllib_policy import register_hemac_rllib_models


# Collection settings. These can be edited directly for repeated experiments.
CHECKPOINT_PATH = PROJECT_ROOT / "src/train/mappo_checkpoints/checkpoint_19000"
OUTPUT_DIR = PROJECT_ROOT / "src/skill_discovery/offline_data"
NUM_EPISODES = 100
MAX_COLLECTION_ATTEMPTS = 10000
BASE_SEED = 0
EXPLORE = False
ENV_NAME = "hemac_asymmetric_env"

ACTION_HISTORY_LENGTH = 5
ACTION_DIM = 3

CHANNEL_NAMES = {
    "observer": {
        "global_map": (
            "coverage",
            "boundary",
            "obstacle",
            "warning",
            "drones",
            "goal",
        ),
        "local_map": (
            "coverage",
            "boundary",
            "obstacle",
            "warning",
            "drones",
            "goal",
        ),
    },
    "drone": {
        "global_map": (
            "coverage",
            "boundary",
            "obstacle",
            "warning",
            "other_drones",
            "observer",
            "goal",
        ),
        "local_map": (
            "coverage",
            "boundary",
            "obstacle",
            "warning",
            "other_drones",
            "observer",
            "goal",
        ),
        "central_map": (
            "coverage",
            "boundary",
            "obstacle",
            "warning",
            "all_drones",
            "focal_drone",
            "observer",
            "goal",
        ),
    },
}


def parse_args() -> argparse.Namespace:
    """Parse optional one-off overrides while keeping editable globals."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--num-episodes", type=int, default=NUM_EPISODES)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=MAX_COLLECTION_ATTEMPTS,
        help="Maximum total rollout attempts used to find qualifying episodes.",
    )
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument(
        "--explore",
        action=argparse.BooleanOptionalAction,
        default=EXPLORE,
        help="Sample from the learned action distribution instead of using deterministic actions.",
    )
    return parser.parse_args()


def policy_id_for_agent(agent_id: str) -> str:
    """Map a HeMAC agent ID to its checkpoint policy ID."""
    if agent_id.startswith("observer_"):
        return "observer_policy"
    if agent_id.startswith("drone_"):
        return "drone_policy"
    raise ValueError(f"No trained policy is configured for agent {agent_id!r}.")


def env_creator(config):
    """Recreate the environment from the config embedded in the checkpoint."""
    return PettingZooEnv(HeMAC_v0.env(**dict(config)))


def group_name_for_agent(agent_id: str) -> str:
    """Return the policy-level dataset group for one agent."""
    if agent_id.startswith("observer_"):
        return "observer"
    if agent_id.startswith("drone_"):
        return "drone"
    raise ValueError(f"Unsupported agent ID: {agent_id!r}")


def get_core_env(env):
    """Return the inner HeMAC instance below PettingZoo wrappers and RawEnv."""
    aec_env = env.unwrapped
    return getattr(aec_env, "env", aec_env)


def agent_found_goal(core_env, agent_id: str) -> bool:
    """Read one agent's persistent goal-discovery flag from the core environment."""
    agent_index = core_env.agent_name_mapping.get(agent_id)
    if agent_index is None:
        return False
    agent = core_env.agents_list[agent_index]
    return bool(getattr(agent, "found_goal", False))


def map_to_chw(value: Any, key: str) -> np.ndarray:
    """Convert one HWC map observation to contiguous float32 CHW format."""
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"{key} must be HWC, got shape {array.shape}.")
    return np.ascontiguousarray(array.transpose(2, 0, 1))


def action_history_to_matrix(value: Any) -> np.ndarray:
    """Restore the flattened environment history as [5, 3]."""
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    expected_size = ACTION_HISTORY_LENGTH * ACTION_DIM
    if array.size != expected_size:
        raise ValueError(
            f"Action history must contain {expected_size} values, got {array.size}."
        )
    return np.ascontiguousarray(
        array.reshape(ACTION_HISTORY_LENGTH, ACTION_DIM)
    )


def convert_observation(observation: dict[str, Any], group_name: str) -> dict[str, np.ndarray]:
    """Convert an environment observation to the persisted tensor schema."""
    converted = {
        "global_map": map_to_chw(observation["global_map"], "global_map"),
        "local_map": map_to_chw(observation["local_map"], "local_map"),
        "action_history": action_history_to_matrix(observation["vector"]),
    }
    if group_name == "drone":
        converted["central_map"] = map_to_chw(
            observation["central_map"],
            "central_map",
        )
        converted["central_vector"] = np.ascontiguousarray(
            np.asarray(observation["central_vector"], dtype=np.float32).reshape(-1)
        )
    return converted


def empty_episode_buffers() -> dict[str, list[dict[str, Any]]]:
    """Create mutable policy-level transition buffers."""
    return {"observer": [], "drone": []}


def load_inference_algorithm(checkpoint_path: Path) -> Algorithm:
    """Restore checkpoint state without creating unused rollout workers or a GPU learner."""
    checkpoint_info = get_checkpoint_info(str(checkpoint_path))
    state = Algorithm._checkpoint_info_to_algorithm_state(
        checkpoint_info=checkpoint_info,
    )
    config = state.get("config")
    if config is None or not hasattr(config, "env_runners"):
        raise TypeError(f"Checkpoint does not contain a valid RLlib config: {checkpoint_path}")

    config.env_runners(
        num_env_runners=0,
        num_envs_per_env_runner=1,
        create_local_env_runner=True,
    )
    config.resources(num_gpus=0)
    state["config"] = config
    return Algorithm.from_state(state)


def stack_observations(
    records: list[dict[str, Any]],
    observation_key: str,
) -> dict[str, torch.Tensor]:
    """Stack one side of all transitions into a tensor dictionary."""
    observation_names = list(records[0][observation_key])
    return {
        name: torch.from_numpy(
            np.stack([record[observation_key][name] for record in records], axis=0)
        )
        for name in observation_names
    }


def tensorize_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert collected records for one policy into an offline-learning batch."""
    if not records:
        return {
            "agent_id": [],
            "episode_step": torch.empty((0,), dtype=torch.int64),
            "observations": {},
            "next_observations": {},
            "actions": torch.empty((0, ACTION_DIM), dtype=torch.float32),
            "rewards": torch.empty((0,), dtype=torch.float32),
            "terminated": torch.empty((0,), dtype=torch.bool),
            "truncated": torch.empty((0,), dtype=torch.bool),
        }

    return {
        "agent_id": [record["agent_id"] for record in records],
        "episode_step": torch.tensor(
            [record["episode_step"] for record in records],
            dtype=torch.int64,
        ),
        "observations": stack_observations(records, "observation"),
        "next_observations": stack_observations(records, "next_observation"),
        "actions": torch.from_numpy(
            np.stack([record["action"] for record in records], axis=0)
        ),
        "rewards": torch.tensor(
            [record["reward"] for record in records],
            dtype=torch.float32,
        ),
        "terminated": torch.tensor(
            [record["terminated"] for record in records],
            dtype=torch.bool,
        ),
        "truncated": torch.tensor(
            [record["truncated"] for record in records],
            dtype=torch.bool,
        ),
    }


def collect_episode(
    algo: Algorithm,
    env,
    seed: int,
    explore: bool,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Collect one episode and align each reward with the action that caused it."""
    env.reset(seed=seed)
    buffers = empty_episode_buffers()
    pending_by_agent: dict[str, dict[str, Any]] = {}
    agent_steps = {agent_id: 0 for agent_id in env.possible_agents}
    final_info: dict[str, Any] = {}
    core_env = get_core_env(env)
    drone_goal_found = False

    for agent_id in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if info:
            final_info = dict(info)

        group_name = group_name_for_agent(agent_id)
        converted_observation = convert_observation(observation, group_name)

        previous_record = pending_by_agent.pop(agent_id, None)
        if previous_record is not None:
            previous_record["next_observation"] = converted_observation
            previous_record["reward"] = float(reward)
            previous_record["terminated"] = bool(termination)
            previous_record["truncated"] = bool(truncation)

        if termination or truncation:
            env.step(None)
            continue

        action = algo.compute_single_action(
            observation=observation,
            policy_id=policy_id_for_agent(agent_id),
            explore=explore,
        )
        action_array = np.ascontiguousarray(
            np.asarray(action, dtype=np.float32).reshape(-1)
        )
        if action_array.shape != (ACTION_DIM,):
            raise ValueError(
                f"Expected a {ACTION_DIM}D action for {agent_id}, got {action_array.shape}."
            )

        record = {
            "agent_id": agent_id,
            "episode_step": agent_steps[agent_id],
            "observation": converted_observation,
            "action": action_array,
            "next_observation": None,
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
        }
        buffers[group_name].append(record)
        pending_by_agent[agent_id] = record
        agent_steps[agent_id] += 1
        env.step(action_array)
        if group_name == "drone":
            drone_goal_found = drone_goal_found or agent_found_goal(
                core_env,
                agent_id,
            )

    if pending_by_agent:
        missing = ", ".join(sorted(pending_by_agent))
        raise RuntimeError(f"Episode ended before transitions were finalized for: {missing}")

    if hasattr(core_env, "build_episode_info"):
        final_info.update(core_env.build_episode_info())
    final_info["drone_goal_found"] = drone_goal_found or any(
        agent.__class__.__name__ == "Drone"
        and bool(getattr(agent, "found_goal", False))
        for agent in getattr(core_env, "agents_list", ())
    )

    return buffers, final_info


def episode_is_eligible(final_info: dict[str, Any]) -> bool:
    """Return whether an episode is a successful drone-discovery demonstration."""
    return bool(final_info.get("drone_goal_found", False)) and bool(
        final_info.get("success", False)
    )


def save_episode(
    output_path: Path,
    episode_index: int,
    attempt_index: int,
    seed: int,
    checkpoint_path: Path,
    buffers: dict[str, list[dict[str, Any]]],
    final_info: dict[str, Any],
) -> dict[str, Any]:
    """Atomically save one episode as a PyTorch dictionary."""
    payload = {
        "metadata": {
            "format_version": 1,
            "episode_index": episode_index,
            "collection_attempt": attempt_index,
            "seed": seed,
            "checkpoint": str(checkpoint_path),
            "map_layout": "BCHW",
            "action_history_layout": "B53",
            "channel_names": CHANNEL_NAMES,
            "final_info": final_info,
        },
        "observer": tensorize_group(buffers["observer"]),
        "drone": tensorize_group(buffers["drone"]),
    }
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, output_path)
    return payload


def print_saved_shapes(output_path: Path, payload: dict[str, Any]) -> None:
    """Print compact shape information for one saved episode."""
    shape_parts = []
    for group_name in ("observer", "drone"):
        observations = payload[group_name]["observations"]
        formatted = ", ".join(
            f"{key}={tuple(value.shape)}" for key, value in observations.items()
        )
        shape_parts.append(f"{group_name}[{formatted}]")
    print(f"Saved {output_path}: " + " | ".join(shape_parts))


def main() -> None:
    """Load the MAPPO checkpoint and collect all requested episodes."""
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not (checkpoint_path / "algorithm_state.pkl").is_file():
        raise FileNotFoundError(f"Invalid RLlib checkpoint: {checkpoint_path}")
    if args.num_episodes <= 0:
        raise ValueError("--num-episodes must be positive.")
    if args.max_attempts < args.num_episodes:
        raise ValueError("--max-attempts must be at least --num-episodes.")

    output_dir.mkdir(parents=True, exist_ok=True)
    ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=1)
    register_hemac_rllib_models()
    register_env(ENV_NAME, env_creator)

    algo = load_inference_algorithm(checkpoint_path)
    checkpoint_env_config = getattr(algo.config, "env_config", {}) or {}
    collection_env_config = dict(checkpoint_env_config)
    collection_env_config["render_mode"] = None
    collection_env_config["log_step_rewards"] = False
    env = HeMAC_v0.env(**collection_env_config)

    try:
        saved_episodes = 0
        attempted_episodes = 0
        while (
            saved_episodes < args.num_episodes
            and attempted_episodes < args.max_attempts
        ):
            attempt_index = attempted_episodes
            seed = args.base_seed + attempt_index
            attempted_episodes += 1
            buffers, final_info = collect_episode(
                algo,
                env,
                seed=seed,
                explore=args.explore,
            )

            if not episode_is_eligible(final_info):
                if attempted_episodes == 1 or attempted_episodes % 10 == 0:
                    print(
                        "Collection progress: "
                        f"saved={saved_episodes}/{args.num_episodes}, "
                        f"attempted={attempted_episodes}, "
                        f"last_drone_goal_found={final_info.get('drone_goal_found', False)}, "
                        f"last_success={final_info.get('success', False)}"
                    )
                continue

            episode_index = saved_episodes
            output_path = output_dir / f"episode_{episode_index:06d}.pt"
            payload = save_episode(
                output_path,
                episode_index=episode_index,
                attempt_index=attempt_index,
                seed=seed,
                checkpoint_path=checkpoint_path,
                buffers=buffers,
                final_info=final_info,
            )
            saved_episodes += 1
            print_saved_shapes(output_path, payload)

        if saved_episodes < args.num_episodes:
            raise RuntimeError(
                "Could not collect enough qualifying episodes: "
                f"saved {saved_episodes}/{args.num_episodes} after "
                f"{attempted_episodes} attempts. Increase --max-attempts or "
                "use a checkpoint with a higher success rate."
            )
    finally:
        env.close()
        algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
