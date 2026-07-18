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
from ray.rllib.utils.schedules import PiecewiseSchedule
import wandb
from PIL import Image

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from hemac import HeMAC_v0
from hemac.helpers.logger import LOGGER
from hemac.rllib_policy import (
    DRONE_LOG_STD_INIT,
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
    "spawn_quadrant": ["bottom_right", "bottom_left", "top_right"],
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
PPO_INITIAL_LR = 3e-4
PPO_MID_LR = 1e-4
PPO_FINAL_LR = 1e-5
PPO_TRAIN_BATCH_SIZE = 8000
PPO_LR_MID_OFFSET = 500 * PPO_TRAIN_BATCH_SIZE
PPO_LR_FINAL_OFFSET = 10000 * PPO_TRAIN_BATCH_SIZE
# LOG_STD_MAX_INCREASE_PER_OPTIMIZER_STEP = 1e-5
NUM_ENV_RUNNERS = 6
ROLLOUT_FRAGMENT_LENGTH = 100
SAMPLE_TIMEOUT_S = 300.0
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
        "max_obstacles": 5,
        "obstacle_min_speed": 1,
        "obstacle_max_speed": 3,
    },
    {
        "min_obstacles": 5,
        "max_obstacles": 7,
        "obstacle_min_speed": 2,
        "obstacle_max_speed": 5,
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


def build_curriculum_lr_schedule(start_timestep):
    """Build an LR schedule whose decay restarts at a curriculum promotion."""
    start_timestep = max(int(start_timestep), 0)
    schedule = [[0, PPO_INITIAL_LR]]
    if start_timestep > 0:
        schedule.append([start_timestep, PPO_INITIAL_LR])
    schedule.extend(
        [
            [start_timestep + PPO_LR_MID_OFFSET, PPO_MID_LR],
            [start_timestep + PPO_LR_FINAL_OFFSET, PPO_FINAL_LR],
        ]
    )
    return schedule


def reset_log_std_growth_limiter_reference(policy):
    """Align a policy's limiter reference with its current log_std value."""
    limiter_state = getattr(policy, "_hemac_log_std_limiter_state", None)
    model = getattr(policy, "model", None)
    log_std_parameter = getattr(model, "log_std", None)
    if limiter_state is None or log_std_parameter is None:
        return False

    limiter_state["previous"].copy_(log_std_parameter.detach())
    return True


# def install_log_std_growth_limiter(
#     policy,
#     max_increase=LOG_STD_MAX_INCREASE_PER_OPTIMIZER_STEP,
# ):
#     """Limit upward log_std movement after every optimizer update."""
#     model = getattr(policy, "model", None)
#     log_std_parameter = getattr(model, "log_std", None)
#     if log_std_parameter is None:
#         return False

#     existing_state = getattr(policy, "_hemac_log_std_limiter_state", None)
#     if existing_state is not None:
#         existing_state["max_increase"] = float(max_increase)
#         reset_log_std_growth_limiter_reference(policy)
#         return True

#     with log_std_parameter.no_grad():
#         log_std_parameter.clamp_(min=model.log_std_min, max=model.log_std_max)

#     limiter_state = {
#         "previous": log_std_parameter.detach().clone(),
#         "max_increase": max(float(max_increase), 0.0),
#     }
#     hook_handles = []

#     def limit_log_std_growth(optimizer, args, kwargs):
#         del optimizer, args, kwargs
#         current = log_std_parameter.data
#         max_allowed = limiter_state["previous"] + limiter_state["max_increase"]
#         current.copy_(current.minimum(max_allowed))
#         current.clamp_(min=model.log_std_min, max=model.log_std_max)
#         limiter_state["previous"].copy_(current)

#     for optimizer in getattr(policy, "_optimizers", []):
#         contains_log_std = any(
#             any(parameter is log_std_parameter for parameter in group["params"])
#             for group in optimizer.param_groups
#         )
#         if contains_log_std:
#             hook_handles.append(optimizer.register_step_post_hook(limit_log_std_growth))

#     if not hook_handles:
#         return False

#     policy._hemac_log_std_limiter_state = limiter_state
#     policy._hemac_log_std_limiter_handles = hook_handles
#     return True


# def install_algorithm_log_std_growth_limiters(algo):
#     """Install log_std growth limiters on all continuous trainable policies."""
#     installed_policy_ids = []
#     for policy_id in ("observer_policy", "drone_policy"):
#         try:
#             policy = algo.get_policy(policy_id)
#         except Exception:
#             policy = None
#         if policy is not None and install_log_std_growth_limiter(policy):
#             installed_policy_ids.append(policy_id)
#     return installed_policy_ids


def reset_policy_training_parameters(policy, start_timestep):
    """Restart one policy's optimization schedule and exploration scale."""
    lr_schedule = build_curriculum_lr_schedule(start_timestep)
    policy._lr_schedule = PiecewiseSchedule(
        lr_schedule,
        outside_value=lr_schedule[-1][-1],
        framework=None,
    )
    policy.cur_lr = float(PPO_INITIAL_LR)

    # Entropy is currently constant, but assigning it explicitly makes the
    # reset robust if a resumed checkpoint contains an older decay schedule.
    policy._entropy_coeff_schedule = None
    policy.entropy_coeff = float(PPO_ENTROPY_COEFF)

    model = getattr(policy, "model", None)
    log_std_parameter = getattr(model, "log_std", None)
    log_std_stats = None
    if model is not None and hasattr(model, "reset_log_std"):
        log_std_stats = model.reset_log_std(DRONE_LOG_STD_INIT)
        # reset_log_std_growth_limiter_reference(policy)

    for optimizer in getattr(policy, "_optimizers", []):
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = float(PPO_INITIAL_LR)
        if log_std_parameter is not None:
            # Adam momentum from the previous stage can immediately undo the
            # reset, so clear only the state associated with log_std.
            optimizer.state.pop(log_std_parameter, None)

    policy_config = getattr(policy, "config", None)
    if isinstance(policy_config, dict):
        policy_config["lr"] = float(PPO_INITIAL_LR)
        policy_config["lr_schedule"] = lr_schedule
        policy_config["entropy_coeff"] = float(PPO_ENTROPY_COEFF)
        policy_config["entropy_coeff_schedule"] = None

    return {
        "learning_rate": float(PPO_INITIAL_LR),
        "entropy_coeff": float(PPO_ENTROPY_COEFF),
        "log_std": log_std_stats,
    }


def reset_curriculum_training_parameters(algo):
    """Reset trainable policy hyperparameters after a difficulty promotion."""
    policies = {}
    for policy_id in ("observer_policy", "drone_policy"):
        try:
            policy = algo.get_policy(policy_id)
        except Exception:
            policy = None
        if policy is not None:
            policies[policy_id] = policy

    if not policies:
        return {}

    start_timestep = max(
        int(getattr(policy, "global_timestep", 0))
        for policy in policies.values()
    )
    reset_stats = {
        policy_id: reset_policy_training_parameters(policy, start_timestep)
        for policy_id, policy in policies.items()
    }

    lr_schedule = build_curriculum_lr_schedule(start_timestep)
    algo_config = getattr(algo, "config", None)
    if isinstance(algo_config, dict):
        algo_config["lr"] = float(PPO_INITIAL_LR)
        algo_config["lr_schedule"] = lr_schedule
        algo_config["entropy_coeff"] = float(PPO_ENTROPY_COEFF)
        algo_config["entropy_coeff_schedule"] = None
    elif algo_config is not None:
        # RLlib freezes the live AlgorithmConfig. Replace it with an updated
        # frozen copy so the restarted schedule is persisted in checkpoints.
        updated_config = algo_config.copy(copy_frozen=False)
        updated_config.lr = float(PPO_INITIAL_LR)
        updated_config.lr_schedule = lr_schedule
        updated_config.entropy_coeff = float(PPO_ENTROPY_COEFF)
        updated_config.entropy_coeff_schedule = None
        updated_config.freeze()
        algo.config = updated_config

    env_runner_group = get_env_runner_group(algo)
    if env_runner_group is not None and hasattr(env_runner_group, "sync_weights"):
        env_runner_group.sync_weights(
            policies=list(policies),
            timeout_seconds=max(float(SAMPLE_TIMEOUT_S), 30.0),
        )

    return {
        "start_timestep": start_timestep,
        "policies": reset_stats,
    }


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
        "max_cycles": 300,
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


def run_rollout(algo, seed, render_mode=None, capture_frames=False, explore=False, env=None):
    """Run one rollout and return its final info plus optional frames."""
    owns_env = env is None
    if env is None:
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
        if owns_env:
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
    return collect_eval_metrics(
        algo,
        num_episodes=num_episodes,
        seed=seed,
        explore=explore,
        seeds=seeds,
    )["success_rate"]


def collect_eval_metrics(algo, num_episodes=5, seed=VIDEO_SEED, explore=False, seeds=None):
    """Run each evaluation rollout once and collect all episode-level rates."""
    successes = []
    crash_flags = []
    rollout_seeds = seeds if seeds is not None else [seed + episode_idx for episode_idx in range(num_episodes)]
    env = HeMAC_v0.env(**build_env_config(render_mode=None))
    try:
        for rollout_seed in rollout_seeds:
            final_info, _ = run_rollout(
                algo,
                seed=rollout_seed,
                render_mode=None,
                capture_frames=False,
                explore=explore,
                env=env,
            )
            successes.append(1.0 if final_info.get("success", False) else 0.0)
            crash_flags.append(1.0 if final_info.get("drone_crash", False) else 0.0)
    finally:
        env.close()
    return {
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "drone_crash_rate": float(np.mean(crash_flags)) if crash_flags else 0.0,
    }


def collect_eval_drone_crash_rate(algo, num_episodes=5, seed=VIDEO_SEED, explore=False, seeds=None):
    """Run evaluation episodes and return average drone-crash rate."""
    return collect_eval_metrics(
        algo,
        num_episodes=num_episodes,
        seed=seed,
        explore=explore,
        seeds=seeds,
    )["drone_crash_rate"]


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
        drone_reward_coverage_ratio = final_info.get("drone_reward_coverage_ratio", 0.0)
        drone_reward_explored_area = final_info.get("drone_reward_explored_area", 0.0)
        goal_found_step = final_info.get("goal_found_step", 0.0)
        success_step = final_info.get("success_step", 0.0)
        steps_after_goal_found = final_info.get("steps_after_goal_found", 0.0)

        # wandb 및 터미널 출력용 custom_metrics 할당
        episode.custom_metrics["explored_area"] = float(area)
        episode.custom_metrics["coverage_ratio"] = float(coverage_ratio)
        episode.custom_metrics["drone_reward_coverage_ratio"] = float(drone_reward_coverage_ratio)
        episode.custom_metrics["drone_reward_explored_area"] = float(drone_reward_explored_area)
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


def restore_algorithm_with_sampling_config(checkpoint_dir, args):
    """Restore training state while applying the current sampling settings."""
    state_path = Path(checkpoint_dir) / "algorithm_state.pkl"
    if not state_path.is_file():
        raise FileNotFoundError(f"Algorithm state not found: {state_path}")

    with state_path.open("rb") as file_obj:
        state = pickle.load(file_obj)

    checkpoint_config = state.get("config")
    if not isinstance(checkpoint_config, dict):
        raise TypeError(f"Checkpoint does not contain a valid config: {state_path}")

    restored_config = dict(checkpoint_config)
    restored_config.update(
        {
            "num_env_runners": int(args.num_env_runners),
            "rollout_fragment_length": int(args.rollout_fragment_length),
            "sample_timeout_s": float(args.sample_timeout_s),
        }
    )
    state["config"] = restored_config
    return Algorithm.from_state(state)


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
        help="Checkpoint directory to restore before continuing training. Use 'latest' to recursively load the newest checkpoint under --checkpoint-dir.",
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


def is_algorithm_checkpoint_dir(path):
    """Return True when the path looks like a top-level RLlib algorithm checkpoint."""
    checkpoint_dir = Path(path)
    return checkpoint_dir.is_dir() and (checkpoint_dir / "algorithm_state.pkl").is_file()


def resolve_checkpoint_path(checkpoint_path, checkpoint_dir):
    """Resolve a checkpoint path or the newest checkpoint under the checkpoint dir."""
    if checkpoint_path is None:
        return None

    if checkpoint_path == "latest":
        candidate_root = Path(checkpoint_dir)
    else:
        candidate_root = Path(checkpoint_path)

    if is_algorithm_checkpoint_dir(candidate_root):
        return candidate_root.resolve()

    if not candidate_root.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {candidate_root}")

    if candidate_root.is_file():
        if candidate_root.name == "algorithm_state.pkl" and is_algorithm_checkpoint_dir(candidate_root.parent):
            return candidate_root.parent.resolve()
        raise FileNotFoundError(
            f"Checkpoint file is not a valid RLlib algorithm checkpoint marker: {candidate_root}"
        )

    checkpoint_markers = sorted(
        (
            path for path in candidate_root.rglob("algorithm_state.pkl")
            if is_algorithm_checkpoint_dir(path.parent)
        ),
        key=lambda path: path.stat().st_mtime,
    )
    if not checkpoint_markers:
        raise FileNotFoundError(
            f"No RLlib algorithm checkpoint found under: {candidate_root}"
        )
    return checkpoint_markers[-1].parent.resolve()


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
    """Return the matching stage, or the nearest stage for an older checkpoint."""
    normalized_target = _normalize_obstacle_difficulty(obstacle_level)
    normalized_levels = [_normalize_obstacle_difficulty(level) for level in levels]
    for stage_index, level in enumerate(normalized_levels):
        if level == normalized_target:
            return stage_index

    difficulty_keys = (
        "min_obstacles",
        "max_obstacles",
        "obstacle_min_speed",
        "obstacle_max_speed",
    )
    nearest_stage_index = min(
        range(len(normalized_levels)),
        key=lambda stage_index: sum(
            abs(normalized_levels[stage_index][key] - normalized_target[key])
            for key in difficulty_keys
        ),
    )
    LOGGER.warning(
        "Checkpoint obstacle difficulty %s is not in the current curriculum; "
        "using nearest stage %s.",
        normalized_target,
        normalized_levels[nearest_stage_index],
    )
    return nearest_stage_index


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
        algo = restore_algorithm_with_sampling_config(resume_checkpoint_path, args)
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
                train_batch_size=PPO_TRAIN_BATCH_SIZE,
                minibatch_size=512,
                num_epochs=5,
                lr=PPO_INITIAL_LR,
                lr_schedule=build_curriculum_lr_schedule(0),
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

    print(
        "[sampling] "
        f"env_runners={args.num_env_runners}, "
        f"fragment_length={args.rollout_fragment_length}, "
        f"timeout={args.sample_timeout_s:.0f}s"
    )

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

    # log_std_limited_policies = install_algorithm_log_std_growth_limiters(algo)
    # print(
    #     "[model] log_std growth limit installed "
    #     f"(max increase/optimizer step={LOG_STD_MAX_INCREASE_PER_OPTIMIZER_STEP:.1e}, "
    #     f"policies={log_std_limited_policies})"
    # )

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
            # "log_std_max_increase_per_optimizer_step": (
            #     LOG_STD_MAX_INCREASE_PER_OPTIMIZER_STEP
            # ),
        },
    )

    print("학습 루프 시작...")
    start_iteration = int(getattr(algo, "iteration", 0))

    for i in range(args.num_iterations):
        result = algo.train()
        local_step = i + 1
        iteration = int(
            result.get("training_iteration", start_iteration + i + 1)
        )
        mean_reward = result.get('env_runners', {}).get('episode_reward_mean', result.get('episode_reward_mean', 0))
        obstacle_curriculum_promoted = False
        obstacle_curriculum_updated_envs = 0
        coverage_curriculum_promoted = False
        coverage_curriculum_updated_envs = 0
        curriculum_parameter_reset = {}
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
            "local_step": local_step,
            "reward/mean_reward": mean_reward,
            "reward/observer_policy": policy_rewards.get("observer_policy", 0),
            "reward/drone_policy": policy_rewards.get("drone_policy", 0),
            "model/drone_log_std_mean": drone_log_std_stats.get("mean", 0.0),
            # "model/drone_log_std_min": drone_log_std_stats.get("min", 0.0),
            # "model/drone_log_std_max": drone_log_std_stats.get("max", 0.0),
            "model/entropy_coeff": PPO_ENTROPY_COEFF,
            # "model/log_std_max_increase_per_optimizer_step": (
            #     LOG_STD_MAX_INCREASE_PER_OPTIMIZER_STEP
            # ),
            "metrics/rollout_success_rate": rollout_success_rate,
            "metrics/goal_found_rate": custom_metrics.get("goal_found_rate_mean", 0),
            "metrics/crash_rate": custom_metrics.get("crash_rate_mean", 0),
            "metrics/drone_crash_rate": custom_metrics.get("drone_crash_rate_mean", 0),
            "metrics/observer_crash_rate": custom_metrics.get("observer_crash_rate_mean", 0),
            "metrics/explored_area": custom_metrics.get("explored_area_mean", 0),
            "metrics/coverage_ratio": custom_metrics.get("coverage_ratio_mean", 0),
            "metrics/drone_reward_explored_area": custom_metrics.get("drone_reward_explored_area_mean", 0),
            "metrics/drone_reward_coverage_ratio": custom_metrics.get("drone_reward_coverage_ratio_mean", 0),
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
                visualization_seed = int(np.random.randint(0, 100000))
                video = collect_visualization_video(
                    algo,
                    iteration=iteration,
                    seed=visualization_seed,
                )
                if video is not None:
                    log_payload["visualization/policy_rollout"] = video
                    log_payload["visualization/seed"] = visualization_seed
            except Exception as exc:
                print(f"[warn] visualization logging skipped at iteration {iteration}: {exc}")


        if iteration % EVAL_LOG_INTERVAL == 0:
            try:
                current_eval_seeds = np.random.randint(0, 100000, size=20).tolist()

                deterministic_eval = collect_eval_metrics(
                    algo,
                    num_episodes=20,
                    seeds=current_eval_seeds,
                    explore=False,
                )
                eval_success_rate = deterministic_eval["success_rate"]
                log_payload["metrics/eval_success_rate"] = eval_success_rate
                log_payload["metrics/eval_drone_crash_rate"] = deterministic_eval["drone_crash_rate"]

                stochastic_eval = collect_eval_metrics(
                    algo,
                    num_episodes=20,
                    seeds=current_eval_seeds,
                    explore=True,
                )
                log_payload["metrics/eval_success_rate_stochastic"] = stochastic_eval["success_rate"]
                log_payload["metrics/eval_drone_crash_rate_stochastic"] = stochastic_eval["drone_crash_rate"]
            except Exception as exc:
                print(f"[warn] evaluation logging skipped at iteration {iteration}: {exc}")

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

        if obstacle_curriculum_promoted or coverage_curriculum_promoted:
            curriculum_parameter_reset = reset_curriculum_training_parameters(algo)
            reset_policy_stats = curriculum_parameter_reset.get("policies", {})
            reset_summary = ", ".join(
                f"{policy_id}: lr={stats['learning_rate']:.1e}, "
                f"entropy={stats['entropy_coeff']:.3g}, "
                f"log_std={stats['log_std']['mean']:.2f}"
                for policy_id, stats in reset_policy_stats.items()
                if stats.get("log_std") is not None
            )
            print(
                "[curriculum] optimizer/exploration parameters reset at timestep "
                f"{curriculum_parameter_reset.get('start_timestep', 0)}"
                + (f" ({reset_summary})" if reset_summary else "")
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
        log_payload["curriculum/parameters_reset"] = 1.0 if curriculum_parameter_reset else 0.0
        if curriculum_parameter_reset:
            log_payload["model/learning_rate_after_curriculum_reset"] = PPO_INITIAL_LR
            log_payload["model/entropy_coeff_after_curriculum_reset"] = PPO_ENTROPY_COEFF
            for policy_id, stats in curriculum_parameter_reset.get("policies", {}).items():
                log_std_stats = stats.get("log_std")
                if log_std_stats is not None:
                    log_payload[f"model/{policy_id}_log_std_after_curriculum_reset"] = (
                        log_std_stats["mean"]
                    )

        wandb.log(log_payload, step=local_step)
        
        if iteration % args.save_every == 0:
            iter_checkpoint_dir = checkpoint_dir / f"checkpoint_{iteration:05d}"
            algo.save(str(iter_checkpoint_dir))
            print(f"Checkpoint 저장 완료: {iter_checkpoint_dir}")

    wandb.finish()
    ray.shutdown()

if __name__ == "__main__":
    main()
