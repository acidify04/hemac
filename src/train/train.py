import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import ray
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import wandb
from PIL import Image

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
}

VIDEO_LOG_INTERVAL = 100
VIDEO_FPS = 12
VIDEO_SEED = 0
VIDEO_OUTPUT_DIR = Path("./wandb_media")
PPO_ENTROPY_COEFF = 0.0005
NUM_ENV_RUNNERS = 12


def build_env_config(render_mode=None):
    """Return the shared environment config for training and evaluation."""
    env_config = {
        "n_observers": 1,
        "observer_speed": 5,
        "n_drones": 3,
        "n_provisioners": 0,
        "known_goals": False,
        "max_cycles": 500,
        "drone_config": {
            "drone_max_speed": 25,
            "drone_max_thrust": 8,
            "drones_starting_pos": DRONE_START_POSITIONS,
        },
        "min_obstacles": 5,
        "max_obstacles": 7,
        "poi_config": [GOAL_CONFIG],
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
    return PettingZooEnv(HeMAC_v0.env(**build_env_config()))


def main():
    LOGGER.setLevel(logging.WARNING)
    ray.init()
    register_hemac_rllib_models()
    
    env_name = "hemac_asymmetric_env"
    register_env(env_name, env_creator)

    temp_env = env_creator({})
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space

    policies = {
        "observer_policy": (
            None,
            obs_space["observer_0"],
            act_space["observer_0"],
            {"model": observer_policy_model_config()},
        ),
        "drone_policy": (
            None,
            obs_space["drone_0"],
            act_space["drone_0"],
            {"model": drone_policy_model_config()},
        ),
    }

    def policy_mapping_fn(agent_id, episode, **kwargs):
        if "observer" in agent_id: return "observer_policy"
        elif "drone" in agent_id: return "drone_policy"

    config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .framework("torch")
        .callbacks(HeMACCallbacks)
        .environment(env=env_name)
        .env_runners(num_env_runners=NUM_ENV_RUNNERS)
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
        .resources(num_gpus=1)
        .training(
            train_batch_size=8000, 
            lr_schedule=[
                [0, 5e-5],
                [500 * 8000, 2.5e-5],
                [10000 * 8000, 5e-6]
            ],
            gamma=0.99, 
            grad_clip=1.0, 
            clip_param=0.2,
            entropy_coeff=PPO_ENTROPY_COEFF,
            kl_target=0.01,
        )
        .debugging(log_level="WARN")
    )

    print("RLlib PPO 알고리즘 빌드 중...")
    algo = config.build()
    
    checkpoint_dir = "./hemac_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    wandb.init(project="HeMAC-RL", name="PPO-Asymmetric-Training", config=config.to_dict())

    print("학습 루프 시작...")
    num_iterations = 10000 
    
    for i in range(num_iterations):
        result = algo.train()
        mean_reward = result.get('env_runners', {}).get('episode_reward_mean', result.get('episode_reward_mean', 0))
        
        print(f"\n--- Iteration {i+1} ---")
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

        print(f">>> [디버깅] custom_metrics: {visible_custom_metrics}")

        log_payload = {
            "iteration": i + 1,
            "reward/mean_reward": mean_reward,
            "reward/observer_policy": policy_rewards.get("observer_policy", 0),
            "reward/drone_policy": policy_rewards.get("drone_policy", 0),
            "model/drone_log_std_mean": drone_log_std_stats.get("mean", 0.0),
            # "model/drone_log_std_min": drone_log_std_stats.get("min", 0.0),
            # "model/drone_log_std_max": drone_log_std_stats.get("max", 0.0),
            "model/entropy_coeff": PPO_ENTROPY_COEFF,
            "metrics/rollout_success_rate": custom_metrics.get("success_rate_mean", 0),
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
        }

        if (i + 1) % VIDEO_LOG_INTERVAL == 0:
            try:
                video = collect_visualization_video(algo, iteration=i + 1, seed=VIDEO_SEED)
                if video is not None:
                    log_payload["visualization/policy_rollout"] = video
            except Exception as exc:
                print(f"[warn] visualization logging skipped at iteration {i + 1}: {exc}")

        if (i + 1) % VIDEO_LOG_INTERVAL == 0:
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
                print(f"[warn] eval success logging skipped at iteration {i + 1}: {exc}")

        if (i + 1) % VIDEO_LOG_INTERVAL == 0:
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
                print(f"[warn] eval drone crash logging skipped at iteration {i + 1}: {exc}")

        wandb.log(log_payload)
        
        if (i + 1) % 100 == 0:
            # i+1:05d는 숫자를 5자리(예: 00500)로 포맷팅하여 정렬이 잘 되게 합니다.
            iter_checkpoint_dir = os.path.join(checkpoint_dir, f"checkpoint_{i+1:05d}")
            algo.save(iter_checkpoint_dir)
            print(f"Checkpoint 저장 완료: {iter_checkpoint_dir}")

    wandb.finish()
    ray.shutdown()

if __name__ == "__main__":
    main()
