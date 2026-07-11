"""
새 학습: python src/train/train.py --num-iterations 1000
최신 체크포인트 재개: python src/train/train.py --resume-from latest
특정 체크포인트 재개: python src/train/train.py --load-checkpoint src/train/hemac_checkpoints/checkpoint_07300
"""
import argparse
import logging
import pickle
from collections import deque
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import wandb
from PIL import Image

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from hemac import HeMAC_v0
from hemac.helpers.logger import LOGGER
from hemac.rllib_policy import (
    drone_policy_model_config,
    observer_policy_model_config,
    get_policy_log_std_stats,
    register_hemac_rllib_models,
)


DRONE_START_POSITIONS = [
    [130.0, 870.0, 5.0],
    [170.0, 870.0, 5.0],
    [150.0, 830.0, 5.0],
]

GOAL_CONFIG = {
    "speed": 0,
    "spawn_mode": "random",
    "boundary_margin": 140,
    "spawn_quadrant": "bottom_right",
}

TRAIN_DIR = Path(__file__).resolve().parent
TRAIN_NUM_DRONES = len(DRONE_START_POSITIONS)
FROZEN_OBSERVER_CHECKPOINT = TRAIN_DIR / "observer_checkpoints" / "checkpoint_10000"

VISUALIZATION_LOG_INTERVAL = 500
EVAL_LOG_INTERVAL = 100
VIDEO_FPS = 12
VIDEO_SEED = 0
VIDEO_OUTPUT_DIR = Path("./wandb_media")
PPO_ENTROPY_COEFF = 0.01
NUM_ENV_RUNNERS = 10
ROLLOUT_FRAGMENT_LENGTH = 100
SAMPLE_TIMEOUT_S = 180.0
CURRICULUM_COVERAGE_LEVELS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
CURRICULUM_PROMOTION_SUCCESS_RATE = 0.8
CURRICULUM_STABILITY_WINDOW = 1
CURRENT_DRONE_SUCCESS_MIN_COVERAGE_RATIO = CURRICULUM_COVERAGE_LEVELS[0]
DEFAULT_CHECKPOINT_DIR = TRAIN_DIR / "hemac_checkpoints"
DEFAULT_WANDB_PROJECT = "HeMAC-RL"
DEFAULT_WANDB_RUN_NAME = "PPO-Agent-Training-With-Moving-Obstacles"
DEFAULT_NUM_ITERATIONS = 10_000_000_000
DEFAULT_CHECKPOINT_INTERVAL = 100
DEFAULT_NUM_GPUS = 1
OBSTACLE_CURRICULUM_LEVELS = [
    {
        "min_obstacles": 1,
        "max_obstacles": 2,
        "obstacle_min_speed": 1,
        "obstacle_max_speed": 1,
    },
    {
        "min_obstacles": 2,
        "max_obstacles": 3,
        "obstacle_min_speed": 1,
        "obstacle_max_speed": 2,
    },
    {
        "min_obstacles": 3,
        "max_obstacles": 4,
        "obstacle_min_speed": 1,
        "obstacle_max_speed": 3,
    },
    {
        "min_obstacles": 4,
        "max_obstacles": 5,
        "obstacle_min_speed": 2,
        "obstacle_max_speed": 4,
    },
    {
        "min_obstacles": 5,
        "max_obstacles": 6,
        "obstacle_min_speed": 2,
        "obstacle_max_speed": 5,
    },
    {
        "min_obstacles": 6,
        "max_obstacles": 8,
        "obstacle_min_speed": 2,
        "obstacle_max_speed": 6,
    },
    {
        "min_obstacles": 7,
        "max_obstacles": 10,
        "obstacle_min_speed": 3,
        "obstacle_max_speed": 7,
    },
]
CURRENT_OBSTACLE_DIFFICULTY = dict(OBSTACLE_CURRICULUM_LEVELS[0])


class CoverageCurriculum:
    """Promote the drone-only success target once evaluation success is high enough."""

    def __init__(self, levels, promotion_success_rate=0.8, stability_window=3):
        if not levels:
            raise ValueError("Coverage curriculum requires at least one level.")
        self.levels = [float(level) for level in levels]
        self.promotion_success_rate = float(promotion_success_rate)
        self.recent_success_rates = deque(maxlen=max(int(stability_window), 1))
        self.stage_index = 0

    @property
    def current_coverage_ratio(self):
        return self.levels[self.stage_index]

    @property
    def stage_number(self):
        return self.stage_index + 1

    @property
    def num_stages(self):
        return len(self.levels)

    @property
    def is_finished(self):
        return self.stage_index >= len(self.levels) - 1

    @property
    def recent_success_mean(self):
        if not self.recent_success_rates:
            return 0.0
        return float(np.mean(self.recent_success_rates))

    @property
    def recent_success_min(self):
        if not self.recent_success_rates:
            return 0.0
        return float(np.min(self.recent_success_rates))

    def record_success(self, success_rate):
        """Promote immediately when the supplied evaluation success crosses the threshold."""
        self.recent_success_rates.append(float(success_rate))
        if self.is_finished:
            return False
        if float(success_rate) < self.promotion_success_rate:
            return False

        self.stage_index += 1
        self.recent_success_rates.clear()
        return True


class ObstacleDifficultyCurriculum:
    """Promote obstacle count/speed once evaluation success is high enough."""

    def __init__(self, levels, promotion_success_rate=0.8, stability_window=5):
        if not levels:
            raise ValueError("Obstacle curriculum requires at least one level.")
        self.levels = [_normalize_obstacle_difficulty(level) for level in levels]
        self.promotion_success_rate = float(promotion_success_rate)
        self.recent_success_rates = deque(maxlen=max(int(stability_window), 1))
        self.stage_index = 0

    @property
    def current_level(self):
        return dict(self.levels[self.stage_index])

    @property
    def stage_number(self):
        return self.stage_index + 1

    @property
    def num_stages(self):
        return len(self.levels)

    @property
    def is_finished(self):
        return self.stage_index >= len(self.levels) - 1

    @property
    def recent_success_mean(self):
        if not self.recent_success_rates:
            return 0.0
        return float(np.mean(self.recent_success_rates))

    @property
    def recent_success_min(self):
        if not self.recent_success_rates:
            return 0.0
        return float(np.min(self.recent_success_rates))

    def record_success(self, success_rate):
        """Promote immediately when the supplied evaluation success crosses the threshold."""
        self.recent_success_rates.append(float(success_rate))
        if self.is_finished:
            return False
        if float(success_rate) < self.promotion_success_rate:
            return False

        self.stage_index += 1
        self.recent_success_rates.clear()
        return True


def set_current_drone_success_min_coverage_ratio(value):
    """Update the default success threshold used by newly-created environments."""
    global CURRENT_DRONE_SUCCESS_MIN_COVERAGE_RATIO
    CURRENT_DRONE_SUCCESS_MIN_COVERAGE_RATIO = float(value)


def _normalize_obstacle_difficulty(level):
    """Return a sanitized obstacle-difficulty config."""
    if not isinstance(level, dict):
        raise TypeError(f"Obstacle difficulty level must be a dict, got {type(level)!r}.")

    min_obstacles = max(int(level.get("min_obstacles", 0)), 0)
    max_obstacles = max(int(level.get("max_obstacles", min_obstacles)), min_obstacles)
    obstacle_min_speed = max(int(level.get("obstacle_min_speed", 1)), 1)
    obstacle_max_speed = max(int(level.get("obstacle_max_speed", obstacle_min_speed)), obstacle_min_speed)
    return {
        "min_obstacles": min_obstacles,
        "max_obstacles": max_obstacles,
        "obstacle_min_speed": obstacle_min_speed,
        "obstacle_max_speed": obstacle_max_speed,
    }


def set_current_obstacle_difficulty(level):
    """Update the default obstacle difficulty used by newly-created environments."""
    global CURRENT_OBSTACLE_DIFFICULTY
    CURRENT_OBSTACLE_DIFFICULTY = _normalize_obstacle_difficulty(level)


def _iter_wrapped_envs(env):
    """Yield one environment plus any nested wrappers/unwrapped instances."""
    queue = [env]
    visited = set()

    while queue:
        current = queue.pop(0)
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        yield current

        for attr_name in ("env", "unwrapped"):
            nested = getattr(current, attr_name, None)
            if nested is not None and nested is not current:
                queue.append(nested)


def _set_drone_success_ratio_on_env(env, coverage_ratio):
    """Push the curriculum target through wrappers down to the live HeMAC env."""
    updated = 0
    for current in _iter_wrapped_envs(env):

        if hasattr(current, "drone_only_success_min_coverage_ratio"):
            current.drone_only_success_min_coverage_ratio = float(coverage_ratio)
            updated += 1

        kwargs = getattr(current, "_kwargs", None)
        if isinstance(kwargs, dict):
            kwargs["drone_only_success_min_coverage_ratio"] = float(coverage_ratio)

    return updated


def _set_obstacle_difficulty_on_env(env, level):
    """Push obstacle difficulty through wrappers down to the live HeMAC env."""
    difficulty = _normalize_obstacle_difficulty(level)
    updated = 0

    for current in _iter_wrapped_envs(env):
        applied = False
        if hasattr(current, "set_obstacle_difficulty"):
            current.set_obstacle_difficulty(**difficulty)
            applied = True
        else:
            if hasattr(current, "min_obstacles"):
                current.min_obstacles = int(difficulty["min_obstacles"])
                applied = True
            if hasattr(current, "max_obstacles"):
                current.max_obstacles = int(difficulty["max_obstacles"])
                applied = True
            world = getattr(current, "world", None)
            if world is not None and hasattr(world, "set_obstacle_speed_range"):
                world.set_obstacle_speed_range(
                    difficulty["obstacle_min_speed"],
                    difficulty["obstacle_max_speed"],
                )
                applied = True

        kwargs = getattr(current, "_kwargs", None)
        if isinstance(kwargs, dict):
            kwargs.update(difficulty)

        if applied:
            updated += 1

    return updated


def get_env_runner_group(algo):
    """Return RLlib's env-runner group across minor API variations."""
    env_runner_group = getattr(algo, "env_runner_group", None)
    if env_runner_group is not None:
        return env_runner_group

    workers_attr = getattr(algo, "workers", None)
    if callable(workers_attr):
        try:
            return workers_attr()
        except TypeError:
            return None
    return workers_attr


def apply_curriculum_to_algo(algo, coverage_ratio):
    """Update both current workers and future rollouts to the new curriculum target."""
    set_current_drone_success_min_coverage_ratio(coverage_ratio)

    env_runner_group = get_env_runner_group(algo)

    updated_count = 0
    if env_runner_group is not None and hasattr(env_runner_group, "foreach_env"):
        results = env_runner_group.foreach_env(
            lambda env: _set_drone_success_ratio_on_env(env, coverage_ratio)
        )
        for worker_results in results:
            if isinstance(worker_results, list):
                updated_count += sum(int(value) for value in worker_results)
            else:
                updated_count += int(worker_results)

    env_config = getattr(algo.config, "env_config", None)
    if isinstance(env_config, dict):
        env_config["drone_only_success_min_coverage_ratio"] = float(coverage_ratio)

    return updated_count


def apply_obstacle_curriculum_to_algo(algo, level):
    """Update both current workers and future rollouts to the new obstacle difficulty."""
    difficulty = _normalize_obstacle_difficulty(level)
    set_current_obstacle_difficulty(difficulty)

    env_runner_group = get_env_runner_group(algo)

    updated_count = 0
    if env_runner_group is not None and hasattr(env_runner_group, "foreach_env"):
        results = env_runner_group.foreach_env(
            lambda env: _set_obstacle_difficulty_on_env(env, difficulty)
        )
        for worker_results in results:
            if isinstance(worker_results, list):
                updated_count += sum(int(value) for value in worker_results)
            else:
                updated_count += int(worker_results)

    env_config = getattr(algo.config, "env_config", None)
    if isinstance(env_config, dict):
        env_config.update(difficulty)

    return updated_count


def build_env_config(render_mode=None):
    """Return the shared environment config for training and evaluation."""
    obstacle_difficulty = dict(CURRENT_OBSTACLE_DIFFICULTY)
    env_config = {
        "n_observers": 1,
        "observer_speed": 10,
        "n_drones": TRAIN_NUM_DRONES,
        "n_provisioners": 0,
        "known_goals": False,
        "max_cycles": 400,
        "drone_config": {
            "drone_max_speed": 25,
            "drone_max_thrust": 8,
            "drones_starting_pos": DRONE_START_POSITIONS,
        },
        "min_obstacles": obstacle_difficulty["min_obstacles"],
        "max_obstacles": obstacle_difficulty["max_obstacles"],
        "obstacle_min_speed": obstacle_difficulty["obstacle_min_speed"],
        "obstacle_max_speed": obstacle_difficulty["obstacle_max_speed"],
        "poi_config": [GOAL_CONFIG],
        "drone_only_success_min_coverage_ratio": CURRENT_DRONE_SUCCESS_MIN_COVERAGE_RATIO,
        "drone_only_success_reward": 300.0,
    }
    if render_mode is not None:
        env_config["render_mode"] = render_mode
    return env_config


def extract_final_info_from_episode(episode):
    """Extract final per-episode info from RLlib's episode bookkeeping."""
    agent_ids = []
    if hasattr(episode, "get_agents"):
        agent_ids = list(episode.get_agents())
    elif hasattr(episode, "agent_rewards"):
        agent_ids = [key[0] for key in episode.agent_rewards.keys()]

    for agent_id in agent_ids:
        info = episode.last_info_for(agent_id) or {}
        if info and any(key in info for key in ("success", "fatal_crash", "timeout")):
            return info

    return {}


def extract_final_info_from_wrapped_env(env):
    """Extract final info by walking through env wrappers when callbacks miss it."""
    envs_to_check = [env]
    visited = set()

    while envs_to_check:
        current = envs_to_check.pop(0)
        if id(current) in visited:
            continue
        visited.add(id(current))

        infos = getattr(current, "infos", None)
        if isinstance(infos, dict):
            for info in infos.values():
                if info and any(key in info for key in ("success", "fatal_crash", "timeout")):
                    return info

        if hasattr(current, "build_episode_info"):
            return current.build_episode_info()

        if hasattr(current, "env") and current.env is not None:
            envs_to_check.append(current.env)
        if hasattr(current, "unwrapped") and current.unwrapped is not current:
            envs_to_check.append(current.unwrapped)

    return {}


def run_rollout(algo, seed, render_mode=None, capture_frames=False, explore=False):
    """Run one rollout and return its final info plus optional frames."""
    env = HeMAC_v0.env(**build_env_config(render_mode=render_mode))
    env.reset(seed=seed)

    frames = []
    frame_stride = max(len(getattr(env, "possible_agents", [])), 1)
    turn_idx = 0

    try:
        if capture_frames:
            initial_frame = env.render()
            if initial_frame is not None:
                frames.append(initial_frame)

        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                action = None
            else:
                if "observer" in agent_id:
                    policy_id = "observer_policy"
                elif "drone" in agent_id:
                    policy_id = "drone_policy"
                else:
                    policy_id = None

                if policy_id is None:
                    action = env.action_space(agent_id).sample()
                else:
                    action = algo.compute_single_action(
                        observation=observation,
                        policy_id=policy_id,
                        explore=explore,
                    )

            env.step(action)
            turn_idx += 1

            if capture_frames and turn_idx % frame_stride == 0:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

        final_info = extract_final_info_from_wrapped_env(env)
    finally:
        env.close()

    return final_info, frames


def save_frames_as_gif(frames, iteration):
    """Save rollout frames as a GIF file and return its path."""
    VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gif_path = VIDEO_OUTPUT_DIR / f"policy_rollout_iter_{iteration:05d}_{timestamp}.gif"

    pil_frames = [Image.fromarray(frame.astype(np.uint8)) for frame in frames]
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=max(int(1000 / VIDEO_FPS), 1),
        loop=0,
    )
    return gif_path


def collect_visualization_video(algo, iteration, seed=VIDEO_SEED):
    """Run one evaluation rollout and return a WandB-compatible video artifact."""
    final_info, frames = run_rollout(
        algo,
        seed=seed,
        render_mode="rgb_array",
        capture_frames=True,
        explore=False,
    )

    if not frames:
        return None

    gif_path = save_frames_as_gif(frames, iteration)
    return wandb.Video(str(gif_path), format="gif")


def collect_eval_success_rate(algo, num_episodes=5, seed=VIDEO_SEED, explore=False, seeds=None):
    """Run evaluation episodes and return average success rate."""
    successes = []
    rollout_seeds = seeds if seeds is not None else [seed + episode_idx for episode_idx in range(num_episodes)]
    for rollout_seed in rollout_seeds:
        final_info, _ = run_rollout(
            algo,
            seed=rollout_seed,
            render_mode=None,
            capture_frames=False,
            explore=explore,
        )
        successes.append(1.0 if final_info.get("success", False) else 0.0)
    return float(np.mean(successes)) if successes else 0.0


def collect_eval_drone_crash_rate(algo, num_episodes=5, seed=VIDEO_SEED, explore=False, seeds=None):
    """Run evaluation episodes and return average drone-crash rate."""
    crash_flags = []
    rollout_seeds = seeds if seeds is not None else [seed + episode_idx for episode_idx in range(num_episodes)]
    for rollout_seed in rollout_seeds:
        final_info, _ = run_rollout(
            algo,
            seed=rollout_seed,
            render_mode=None,
            capture_frames=False,
            explore=explore,
        )
        crash_flags.append(1.0 if final_info.get("drone_crash", False) else 0.0)
    return float(np.mean(crash_flags)) if crash_flags else 0.0


class HeMACCallbacks(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        # Prefer RLlib's episode bookkeeping at episode end. The wrapped env may
        # already be transitioning to the next reset by the time this callback runs.
        final_info = extract_final_info_from_episode(episode)

        if not final_info:
            final_info = extract_final_info_from_wrapped_env(base_env.get_sub_environments()[env_index])

        # 최종 값 추출 (어느 방법으로든 찾지 못한 경우 99999.0 등 기본값)
        area = final_info.get("explored_area", 0.0)
        coverage_ratio = final_info.get("coverage_ratio", 0.0)
        goal_found_step = final_info.get("goal_found_step", 0.0)
        success_step = final_info.get("success_step", 0.0)
        steps_after_goal_found = final_info.get("steps_after_goal_found", 0.0)

        # wandb 및 터미널 출력용 custom_metrics 할당
        episode.custom_metrics["explored_area"] = float(area)
        episode.custom_metrics["coverage_ratio"] = float(coverage_ratio)
        # episode.custom_metrics["goal_found_step"] = float(goal_found_step)
        episode.custom_metrics["success_step"] = float(success_step)
        # episode.custom_metrics["steps_after_goal_found"] = float(steps_after_goal_found)
        episode.custom_metrics["success_rate"] = 1.0 if final_info.get("success", False) else 0.0
        episode.custom_metrics["goal_found_rate"] = 1.0 if final_info.get("goal_found", False) else 0.0
        episode.custom_metrics["success_after_goal_found_rate"] = (
            1.0 if final_info.get("success_after_goal_found", False) else 0.0
        )
        episode.custom_metrics["crash_rate"] = 1.0 if final_info.get("fatal_crash", False) else 0.0
        # episode.custom_metrics["timeout_rate"] = 1.0 if final_info.get("timeout", False) else 0.0
        episode.custom_metrics["drone_crash_rate"] = 1.0 if final_info.get("drone_crash", False) else 0.0
        episode.custom_metrics["observer_crash_rate"] = 1.0 if final_info.get("observer_crash", False) else 0.0
        episode.custom_metrics["drone_crash_to_obstacle_rate"] = 1.0 if final_info.get("drone_crash_to_obstacle", False) else 0.0
        episode.custom_metrics["observer_crash_to_obstacle_rate"] = 1.0 if final_info.get("observer_crash_to_obstacle", False) else 0.0


def env_creator(config):
    env_config = build_env_config()
    if config:
        env_config.update(config)
    return PettingZooEnv(HeMAC_v0.env(**env_config))


def load_policy_weights_from_checkpoint(checkpoint_dir, policy_id):
    """Load one policy's weights from an RLlib checkpoint directory."""
    policy_state_path = Path(checkpoint_dir) / "policies" / policy_id / "policy_state.pkl"
    if not policy_state_path.is_file():
        raise FileNotFoundError(
            f"Policy checkpoint not found for {policy_id}: {policy_state_path}"
        )

    with policy_state_path.open("rb") as file_obj:
        policy_state = pickle.load(file_obj)

    weights = policy_state.get("weights")
    if not isinstance(weights, dict) or not weights:
        raise ValueError(
            f"Checkpoint {policy_state_path} does not contain valid weights for {policy_id}."
        )
    return weights


def restore_frozen_observer_policy(algo, checkpoint_dir):
    """Load observer weights from checkpoint and sync them to every worker."""
    observer_weights = load_policy_weights_from_checkpoint(
        checkpoint_dir, "observer_policy"
    )
    algo.set_weights({"observer_policy": observer_weights})

    env_runner_group = get_env_runner_group(algo)
    if env_runner_group is not None and hasattr(env_runner_group, "sync_weights"):
        env_runner_group.sync_weights(
            policies=["observer_policy"],
            timeout_seconds=max(float(SAMPLE_TIMEOUT_S), 30.0),
        )

    return Path(checkpoint_dir)


def parse_args():
    """Parse CLI arguments for training."""
    parser = argparse.ArgumentParser(description="Train HeMAC PPO policies.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory where training checkpoints are saved.",
    )
    parser.add_argument(
        "--resume-from",
        "--load-checkpoint",
        dest="resume_from",
        default=None,
        help="Checkpoint directory to restore before continuing training. Use 'latest' to load the newest checkpoint under --checkpoint-dir.",
    )
    parser.add_argument(
        "--restore-observer-from",
        type=Path,
        default=None,
        help="Observer checkpoint directory to load before training starts.",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=DEFAULT_NUM_ITERATIONS,
        help="Additional PPO training iterations to run.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL,
        help="Checkpoint save interval in training iterations.",
    )
    parser.add_argument(
        "--video-log-interval",
        type=int,
        default=VISUALIZATION_LOG_INTERVAL,
        help="Evaluation/video logging interval in training iterations.",
    )
    parser.add_argument(
        "--wandb-project",
        default=DEFAULT_WANDB_PROJECT,
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=DEFAULT_WANDB_RUN_NAME,
        help="Weights & Biases run name.",
    )
    parser.add_argument(
        "--num-env-runners",
        type=int,
        default=NUM_ENV_RUNNERS,
        help="Number of RLlib environment runners.",
    )
    parser.add_argument(
        "--rollout-fragment-length",
        type=int,
        default=ROLLOUT_FRAGMENT_LENGTH,
        help="Rollout fragment length for PPO sampling.",
    )
    parser.add_argument(
        "--sample-timeout-s",
        type=float,
        default=SAMPLE_TIMEOUT_S,
        help="RLlib sample timeout in seconds.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=DEFAULT_NUM_GPUS,
        help="Number of GPUs requested by RLlib.",
    )
    return parser.parse_args()


def resolve_checkpoint_path(checkpoint_path, checkpoint_dir):
    """Resolve a checkpoint path or the newest checkpoint under the checkpoint dir."""
    if checkpoint_path is None:
        return None

    if checkpoint_path == "latest":
        candidate_root = Path(checkpoint_dir)
    else:
        candidate_root = Path(checkpoint_path)

    if candidate_root.is_dir() and candidate_root.name.startswith("checkpoint_"):
        return candidate_root.resolve()

    if not candidate_root.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {candidate_root}")

    checkpoints = sorted(
        (
            path for path in candidate_root.iterdir()
            if path.is_dir() and path.name.startswith("checkpoint_")
        ),
        key=lambda path: path.stat().st_mtime,
    )
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoint_* directory found under: {candidate_root}"
        )
    return checkpoints[-1].resolve()


def _find_coverage_stage_index(levels, coverage_ratio):
    """Return the curriculum stage index that matches the given coverage ratio."""
    target_ratio = float(coverage_ratio)
    for stage_index, level in enumerate(levels):
        if np.isclose(float(level), target_ratio):
            return stage_index
    raise ValueError(
        f"Coverage ratio {target_ratio} does not match any configured curriculum stage."
    )


def _find_obstacle_stage_index(levels, obstacle_level):
    """Return the obstacle-curriculum stage index that matches the given level."""
    normalized_target = _normalize_obstacle_difficulty(obstacle_level)
    for stage_index, level in enumerate(levels):
        if _normalize_obstacle_difficulty(level) == normalized_target:
            return stage_index
    raise ValueError(
        "Obstacle difficulty "
        f"{normalized_target} does not match any configured obstacle curriculum stage."
    )


def initialize_curricula_from_env_config(env_config):
    """Build curriculum objects that match the current environment config."""
    obstacle_curriculum = ObstacleDifficultyCurriculum(
        levels=OBSTACLE_CURRICULUM_LEVELS,
        promotion_success_rate=CURRICULUM_PROMOTION_SUCCESS_RATE,
        stability_window=CURRICULUM_STABILITY_WINDOW,
    )
    current_obstacle_level = {
        "min_obstacles": env_config.get(
            "min_obstacles",
            OBSTACLE_CURRICULUM_LEVELS[0]["min_obstacles"],
        ),
        "max_obstacles": env_config.get(
            "max_obstacles",
            OBSTACLE_CURRICULUM_LEVELS[0]["max_obstacles"],
        ),
        "obstacle_min_speed": env_config.get(
            "obstacle_min_speed",
            OBSTACLE_CURRICULUM_LEVELS[0]["obstacle_min_speed"],
        ),
        "obstacle_max_speed": env_config.get(
            "obstacle_max_speed",
            OBSTACLE_CURRICULUM_LEVELS[0]["obstacle_max_speed"],
        ),
    }
    obstacle_curriculum.stage_index = _find_obstacle_stage_index(
        OBSTACLE_CURRICULUM_LEVELS,
        current_obstacle_level,
    )
    set_current_obstacle_difficulty(obstacle_curriculum.current_level)
    env_config.update(obstacle_curriculum.current_level)

    coverage_curriculum = None
    curriculum_enabled = env_config.get("n_observers", 0) == 0 and env_config.get("n_drones", 0) > 0
    if curriculum_enabled:
        coverage_curriculum = CoverageCurriculum(
            levels=CURRICULUM_COVERAGE_LEVELS,
            promotion_success_rate=CURRICULUM_PROMOTION_SUCCESS_RATE,
            stability_window=CURRICULUM_STABILITY_WINDOW,
        )
        current_coverage_ratio = env_config.get(
            "drone_only_success_min_coverage_ratio",
            CURRICULUM_COVERAGE_LEVELS[0],
        )
        coverage_curriculum.stage_index = _find_coverage_stage_index(
            CURRICULUM_COVERAGE_LEVELS,
            current_coverage_ratio,
        )
        set_current_drone_success_min_coverage_ratio(
            coverage_curriculum.current_coverage_ratio
        )
        env_config["drone_only_success_min_coverage_ratio"] = (
            coverage_curriculum.current_coverage_ratio
        )
    else:
        set_current_drone_success_min_coverage_ratio(
            env_config.get(
                "drone_only_success_min_coverage_ratio",
                CURRICULUM_COVERAGE_LEVELS[0],
            )
        )

    return obstacle_curriculum, coverage_curriculum


def main():
    args = parse_args()
    LOGGER.setLevel(logging.WARNING)
    ray.init()
    register_hemac_rllib_models()

    checkpoint_dir = args.checkpoint_dir.resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resume_checkpoint_path = resolve_checkpoint_path(args.resume_from, checkpoint_dir)

    env_name = "hemac_asymmetric_env"
    register_env(env_name, env_creator)

    if resume_checkpoint_path is not None:
        print(f"체크포인트 로드 중: {resume_checkpoint_path}")
        algo = Algorithm.from_checkpoint(str(resume_checkpoint_path))
        restored_env_config = getattr(algo.config, "env_config", None)
        if not isinstance(restored_env_config, dict):
            raise TypeError(
                "Restored algorithm does not expose a valid env_config dictionary."
            )
        env_config = build_env_config()
        env_config.update(restored_env_config)
    else:
        env_config = build_env_config()

        temp_env = env_creator(env_config)
        obs_space = temp_env.observation_space
        act_space = temp_env.action_space
        temp_env.close()

        policies = {}
        if env_config.get("n_observers", 0) > 0:
            policies["observer_policy"] = (
                None,
                obs_space["observer_0"],
                act_space["observer_0"],
                {"model": observer_policy_model_config()},
            )
        if env_config.get("n_drones", 0) > 0:
            policies["drone_policy"] = (
                None,
                obs_space["drone_0"],
                act_space["drone_0"],
                {"model": drone_policy_model_config()},
            )

        def policy_mapping_fn(agent_id, episode, **kwargs):
            del episode, kwargs
            if "observer" in agent_id and "observer_policy" in policies:
                return "observer_policy"
            if "drone" in agent_id and "drone_policy" in policies:
                return "drone_policy"
            raise ValueError(f"No policy configured for agent_id={agent_id}")

        config = (
            PPOConfig()
            .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
            .framework("torch")
            .callbacks(HeMACCallbacks)
            .environment(env=env_name, env_config=env_config)
            .env_runners(
                num_env_runners=args.num_env_runners,
                rollout_fragment_length=args.rollout_fragment_length,
                sample_timeout_s=args.sample_timeout_s,
            )
            .multi_agent(
                policies=policies,
                policy_mapping_fn=policy_mapping_fn,
                policies_to_train=["observer_policy", "drone_policy"],
            )
            .resources(num_gpus=args.num_gpus)
            .training(
                train_batch_size=8000,
                minibatch_size=512,
                num_epochs=5,
                lr_schedule=[
                    [0, 3e-4],           # [수정] 초기 학습률 증가 (기존 5e-5)
                    [500 * 8000, 1e-4],  # [수정] 중간 학습률 조정
                    [10000 * 8000, 1e-5]
                ],
                gamma=0.995,
                grad_clip=1.0,
                clip_param=0.2,
                entropy_coeff=PPO_ENTROPY_COEFF,
                kl_target=0.01,
            )
            .debugging(log_level="WARN")
        )

        print("RLlib PPO 알고리즘 빌드 중...")
        algo = config.build()

    obstacle_curriculum, coverage_curriculum = initialize_curricula_from_env_config(
        env_config
    )

    if args.restore_observer_from is not None:
        observer_checkpoint_path = restore_frozen_observer_policy(
            algo,
            args.restore_observer_from,
        )
        print(f"[observer] loaded frozen observer policy from {observer_checkpoint_path}")

    obstacle_updated_envs = apply_obstacle_curriculum_to_algo(
        algo, obstacle_curriculum.current_level
    )
    if obstacle_updated_envs <= 0:
        print("[warn] obstacle curriculum target was not applied to any live env at startup.")
    current_obstacle_level = obstacle_curriculum.current_level
    print(
        "[obstacle curriculum] start stage "
        f"{obstacle_curriculum.stage_number}/{obstacle_curriculum.num_stages} "
        f"(obstacles={current_obstacle_level['min_obstacles']}-{current_obstacle_level['max_obstacles']}, "
        f"speed={current_obstacle_level['obstacle_min_speed']}-{current_obstacle_level['obstacle_max_speed']}, "
        f"updated_envs={obstacle_updated_envs})"
    )

    coverage_updated_envs = 0
    if coverage_curriculum is not None:
        coverage_updated_envs = apply_curriculum_to_algo(
            algo, coverage_curriculum.current_coverage_ratio
        )
        if coverage_updated_envs <= 0:
            print("[warn] curriculum target was not applied to any live env at startup.")
        print(
            "[curriculum] start stage "
            f"{coverage_curriculum.stage_number}/{coverage_curriculum.num_stages} "
            f"(coverage >= {coverage_curriculum.current_coverage_ratio:.1f}, "
            f"updated_envs={coverage_updated_envs})"
        )
    else:
        print("[curriculum] disabled because this run includes an observer.")

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            **algo.config.to_dict(),
            "resume_from": str(resume_checkpoint_path) if resume_checkpoint_path is not None else None,
            "restore_observer_from": (
                str(args.restore_observer_from)
                if args.restore_observer_from is not None
                else None
            ),
        },
    )

    print("학습 루프 시작...")
    start_iteration = int(getattr(algo, "iteration", 0))

    for i in range(args.num_iterations):
        result = algo.train()
        iteration = int(
            result.get("training_iteration", start_iteration + i + 1)
        )
        mean_reward = result.get('env_runners', {}).get('episode_reward_mean', result.get('episode_reward_mean', 0))
        obstacle_curriculum_promoted = False
        obstacle_curriculum_updated_envs = 0
        coverage_curriculum_promoted = False
        coverage_curriculum_updated_envs = 0
        eval_success_rate = None
        
        print(f"\n--- Iteration {iteration} ---")
        print(f"Mean Reward: {mean_reward:.2f}")

        custom_metrics = result.get('custom_metrics', {})
        if not custom_metrics:
            custom_metrics = result.get('env_runners', {}).get('custom_metrics', {})
        visible_custom_metrics = {
            key: value for key, value in custom_metrics.items() if "min" not in key and "max" not in key
        }
            
        policy_rewards = result.get('policy_reward_mean', {})
        if not policy_rewards:
            policy_rewards = result.get('env_runners', {}).get('policy_reward_mean', {})
        drone_log_std_stats = get_policy_log_std_stats(algo, "drone_policy") or {}
        rollout_success_rate = float(custom_metrics.get("success_rate_mean", 0.0))

        print(f">>> [디버깅] custom_metrics: {visible_custom_metrics}")

        log_payload = {
            "iteration": iteration,
            "reward/mean_reward": mean_reward,
            "reward/observer_policy": policy_rewards.get("observer_policy", 0),
            "reward/drone_policy": policy_rewards.get("drone_policy", 0),
            "model/drone_log_std_mean": drone_log_std_stats.get("mean", 0.0),
            # "model/drone_log_std_min": drone_log_std_stats.get("min", 0.0),
            # "model/drone_log_std_max": drone_log_std_stats.get("max", 0.0),
            "model/entropy_coeff": PPO_ENTROPY_COEFF,
            "metrics/rollout_success_rate": rollout_success_rate,
            "metrics/goal_found_rate": custom_metrics.get("goal_found_rate_mean", 0),
            "metrics/crash_rate": custom_metrics.get("crash_rate_mean", 0),
            "metrics/drone_crash_rate": custom_metrics.get("drone_crash_rate_mean", 0),
            "metrics/observer_crash_rate": custom_metrics.get("observer_crash_rate_mean", 0),
            "metrics/explored_area": custom_metrics.get("explored_area_mean", 0),
            "metrics/coverage_ratio": custom_metrics.get("coverage_ratio_mean", 0),
            # "metrics/timeout_rate": custom_metrics.get("timeout_rate_mean", 0),
            "metrics/success_after_goal_found_rate": custom_metrics.get("success_after_goal_found_rate_mean", 0),
            # "metrics/goal_found_step": custom_metrics.get("goal_found_step_mean", 0),
            "metrics/success_step": custom_metrics.get("success_step_mean", 0),
            # "metrics/steps_after_goal_found": custom_metrics.get("steps_after_goal_found_mean", 0),
            "metrics/drone_crash_to_obstacle_rate": custom_metrics.get("drone_crash_to_obstacle_rate_mean", 0),
            "metrics/observer_crash_to_obstacle_rate": custom_metrics.get("observer_crash_to_obstacle_rate_mean", 0),
            "curriculum/stage_number": (
                coverage_curriculum.stage_number if coverage_curriculum is not None else 0.0
            ),
            "curriculum/num_stages": (
                coverage_curriculum.num_stages if coverage_curriculum is not None else 0.0
            ),
            "curriculum/current_coverage_target": (
                coverage_curriculum.current_coverage_ratio if coverage_curriculum is not None else 0.0
            ),
            "curriculum/recent_success_mean": (
                coverage_curriculum.recent_success_mean if coverage_curriculum is not None else 0.0
            ),
            "curriculum/recent_success_min": (
                coverage_curriculum.recent_success_min if coverage_curriculum is not None else 0.0
            ),
            "curriculum/promotion_success_rate": (
                CURRICULUM_PROMOTION_SUCCESS_RATE if coverage_curriculum is not None else 0.0
            ),
            "curriculum/stability_window": (
                CURRICULUM_STABILITY_WINDOW if coverage_curriculum is not None else 0.0
            ),
            "curriculum/is_finished": (
                1.0 if coverage_curriculum is not None and coverage_curriculum.is_finished else 0.0
            ),
            "obstacle_curriculum/stage_number": obstacle_curriculum.stage_number,
            "obstacle_curriculum/num_stages": obstacle_curriculum.num_stages,
            "obstacle_curriculum/recent_success_mean": obstacle_curriculum.recent_success_mean,
            "obstacle_curriculum/recent_success_min": obstacle_curriculum.recent_success_min,
            "obstacle_curriculum/promotion_success_rate": CURRICULUM_PROMOTION_SUCCESS_RATE,
            "obstacle_curriculum/stability_window": CURRICULUM_STABILITY_WINDOW,
            "obstacle_curriculum/is_finished": 1.0 if obstacle_curriculum.is_finished else 0.0,
            "obstacle_curriculum/min_obstacles": obstacle_curriculum.current_level["min_obstacles"],
            "obstacle_curriculum/max_obstacles": obstacle_curriculum.current_level["max_obstacles"],
            "obstacle_curriculum/min_speed": obstacle_curriculum.current_level["obstacle_min_speed"],
            "obstacle_curriculum/max_speed": obstacle_curriculum.current_level["obstacle_max_speed"],
        }


        if iteration % args.video_log_interval == 0:
            try:
                video = collect_visualization_video(algo, iteration=iteration, seed=VIDEO_SEED)
                if video is not None:
                    log_payload["visualization/policy_rollout"] = video
            except Exception as exc:
                print(f"[warn] visualization logging skipped at iteration {iteration}: {exc}")


        if iteration % EVAL_LOG_INTERVAL == 0:
            try:
                current_eval_seeds = np.random.randint(0, 100000, size=10).tolist()
                
                eval_success_rate = collect_eval_success_rate(
                    algo,
                    num_episodes=10,
                    seeds=current_eval_seeds,
                    explore=False,
                )
                log_payload["metrics/eval_success_rate"] = eval_success_rate

                eval_success_rate_stochastic = collect_eval_success_rate(
                    algo,
                    num_episodes=10,
                    seeds=current_eval_seeds,
                    explore=True,
                )
                log_payload["metrics/eval_success_rate_stochastic"] = eval_success_rate_stochastic
            except Exception as exc:
                print(f"[warn] eval success logging skipped at iteration {iteration}: {exc}")


        if iteration % EVAL_LOG_INTERVAL == 0:
            try:
                current_eval_seeds = np.random.randint(0, 100000, size=10).tolist()
                
                eval_drone_crash_rate = collect_eval_drone_crash_rate(
                    algo,
                    num_episodes=10,
                    seeds=current_eval_seeds,
                    explore=False,
                )
                log_payload["metrics/eval_drone_crash_rate"] = eval_drone_crash_rate

                eval_drone_crash_rate_stochastic = collect_eval_drone_crash_rate(
                    algo,
                    num_episodes=10,
                    seeds=current_eval_seeds,
                    explore=True,
                )
                log_payload["metrics/eval_drone_crash_rate_stochastic"] = eval_drone_crash_rate_stochastic
            except Exception as exc:
                print(f"[warn] eval drone crash logging skipped at iteration {iteration}: {exc}")

        if eval_success_rate is not None:
            obstacle_curriculum_promoted = obstacle_curriculum.record_success(eval_success_rate)
            if obstacle_curriculum_promoted:
                obstacle_curriculum_updated_envs = apply_obstacle_curriculum_to_algo(
                    algo, obstacle_curriculum.current_level
                )
                if obstacle_curriculum_updated_envs <= 0:
                    print("[warn] obstacle curriculum promoted, but no live env received the new obstacle target.")
                current_obstacle_level = obstacle_curriculum.current_level
                print(
                    "[obstacle curriculum] promoted to stage "
                    f"{obstacle_curriculum.stage_number}/{obstacle_curriculum.num_stages} "
                    f"(eval_success_rate={eval_success_rate:.2f}, "
                    f"obstacles={current_obstacle_level['min_obstacles']}-{current_obstacle_level['max_obstacles']}, "
                    f"speed={current_obstacle_level['obstacle_min_speed']}-{current_obstacle_level['obstacle_max_speed']}, "
                    f"updated_envs={obstacle_curriculum_updated_envs})"
                )

            if coverage_curriculum is not None:
                coverage_curriculum_promoted = coverage_curriculum.record_success(eval_success_rate)
                if coverage_curriculum_promoted:
                    coverage_curriculum_updated_envs = apply_curriculum_to_algo(
                        algo, coverage_curriculum.current_coverage_ratio
                    )
                    if coverage_curriculum_updated_envs <= 0:
                        print("[warn] curriculum promoted, but no live env received the new coverage target.")
                    print(
                        "[curriculum] promoted to stage "
                        f"{coverage_curriculum.stage_number}/{coverage_curriculum.num_stages} "
                        f"(eval_success_rate={eval_success_rate:.2f}, "
                        f"coverage >= {coverage_curriculum.current_coverage_ratio:.1f}, "
                        f"updated_envs={coverage_curriculum_updated_envs})"
                    )

        log_payload["curriculum/stage_number"] = (
            coverage_curriculum.stage_number if coverage_curriculum is not None else 0.0
        )
        log_payload["curriculum/current_coverage_target"] = (
            coverage_curriculum.current_coverage_ratio if coverage_curriculum is not None else 0.0
        )
        log_payload["curriculum/recent_success_mean"] = (
            coverage_curriculum.recent_success_mean if coverage_curriculum is not None else 0.0
        )
        log_payload["curriculum/recent_success_min"] = (
            coverage_curriculum.recent_success_min if coverage_curriculum is not None else 0.0
        )
        log_payload["curriculum/just_promoted"] = 1.0 if coverage_curriculum_promoted else 0.0
        log_payload["curriculum/updated_envs"] = float(coverage_curriculum_updated_envs)
        log_payload["obstacle_curriculum/just_promoted"] = 1.0 if obstacle_curriculum_promoted else 0.0
        log_payload["obstacle_curriculum/updated_envs"] = float(obstacle_curriculum_updated_envs)

        wandb.log(log_payload)
        
        if iteration % args.save_every == 0:
            iter_checkpoint_dir = checkpoint_dir / f"checkpoint_{iteration:05d}"
            algo.save(str(iter_checkpoint_dir))
            print(f"Checkpoint 저장 완료: {iter_checkpoint_dir}")

    wandb.finish()
    ray.shutdown()

if __name__ == "__main__":
    main()
