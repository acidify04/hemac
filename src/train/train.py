import os
import math
import ray
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import wandb

# 수정한 HeMAC 환경 임포트
from hemac import HeMAC_v0

class HeMACCallbacks(DefaultCallbacks):
    # RLlib은 콜백 에러를 숨기므로 try-except로 안전망을 구축합니다.
    def on_episode_start(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        try:
            episode.user_data["min_drone_dist"] = float('inf')
            episode.user_data["min_obs_dist"] = float('inf')
            episode.user_data["explored_grids"] = set()
        except Exception as e:
            print(f"[Callback Error Start] {e}")

    def on_episode_step(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        try:
            env = base_env.get_sub_environments()[env_index]
            if hasattr(env, "unwrapped"): env = env.unwrapped
            while hasattr(env, "env") or hasattr(env, "par_env"):
                env = getattr(env, "env", getattr(env, "par_env", env))

            if not hasattr(env, 'goals') or len(env.goals) == 0: return
            goal = env.goals[0]

            agents = getattr(env, 'world', env).agents if hasattr(getattr(env, 'world', env), 'agents') else getattr(env, 'agents', [])
            if type(agents) is dict: agents = list(agents.values())

            for agent in agents:
                if isinstance(agent, str) or not hasattr(agent, 'x') or not hasattr(agent, 'y'): continue
                dist_to_goal = math.hypot(agent.x - goal.x, agent.y - goal.y)
                agent_name = getattr(agent, "name", getattr(agent, "id", ""))

                if "observer" in agent_name:
                    if dist_to_goal < episode.user_data["min_obs_dist"]:
                        episode.user_data["min_obs_dist"] = dist_to_goal
                elif "drone" in agent_name:
                    if dist_to_goal < episode.user_data["min_drone_dist"]:
                        episode.user_data["min_drone_dist"] = dist_to_goal
                    episode.user_data["explored_grids"].add((int(agent.x // 20), int(agent.y // 20)))
        except Exception:
            pass # 스텝마다 에러 출력 시 터미널이 마비되므로 패스

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        try:
            # 1. 무조건 계산되는 거리 및 탐색 지표
            min_drone = episode.user_data.get("min_drone_dist", 0)
            min_obs = episode.user_data.get("min_obs_dist", 0)
            
            if min_drone == float('inf'): min_drone = 0
            if min_obs == float('inf'): min_obs = 0
            area = len(episode.user_data.get("explored_grids", set())) * 400

            episode.custom_metrics["min_drone_dist"] = min_drone
            episode.custom_metrics["min_obs_dist"] = min_obs
            episode.custom_metrics["explored_area"] = area

            # 2. 안전한 Info 딕셔너리 추출
            info = episode.last_info_for("observer_0") or {}
            episode.custom_metrics["success_rate"] = 1.0 if info.get("success", False) else 0.0
            episode.custom_metrics["crash_rate"] = 1.0 if info.get("fatal_crash", False) else 0.0
        except Exception as e:
            print(f"[Callback Error End] {e}")


def env_creator(config):
    env_config = {
        "n_observers": 1,
        "observer_speed": 5, 
        "n_drones": 3,
        "n_provisioners": 0,
        "drone_config": {
            "drone_max_speed": 25,
            "drone_max_thrust": 8,
            "drones_starting_pos": [],
        },
        "max_obstacles": 5,
        "poi_config": [{"speed": 0}] 
    }
    env = HeMAC_v0.env(**env_config)
    return PettingZooEnv(env)


def main():
    ray.init()
    
    env_name = "hemac_asymmetric_env"
    register_env(env_name, env_creator)

    temp_env = env_creator({})
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space

    policies = {
        "observer_policy": (None, obs_space["observer_0"], act_space["observer_0"], {}),
        "drone_policy": (None, obs_space["drone_0"], act_space["drone_0"], {}) 
    }

    def policy_mapping_fn(agent_id, episode, **kwargs):
        if "observer" in agent_id:
            return "observer_policy"
        elif "drone" in agent_id:
            return "drone_policy"

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .callbacks(HeMACCallbacks)
        .environment(env=env_name)
        .env_runners(num_env_runners=4) 
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
        .training(
            train_batch_size=4000,
            lr=5e-5,
            gamma=0.99, 
        )
        .debugging(log_level="WARN")
    )

    print("RLlib PPO 알고리즘 빌드 중...")
    algo = config.build()
    
    checkpoint_dir = "./hemac_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    wandb.init(
        project="HeMAC-RL",               
        name="PPO-Asymmetric-Training",   
        config=config.to_dict()           
    )

    print("학습 루프 시작...")
    num_iterations = 10000 
    
    for i in range(num_iterations):
        result = algo.train()
        
        mean_reward = result.get('env_runners', {}).get('episode_reward_mean', result.get('episode_reward_mean', 0))
        
        print(f"--- Iteration {i+1} ---")
        print(f"Mean Reward: {mean_reward:.2f}")
        
        def get_metrics_anywhere(d, target_key="custom_metrics"):
            if target_key in d and d[target_key]: return d[target_key]
            for k, v in d.items():
                if isinstance(v, dict):
                    found = get_metrics_anywhere(v, target_key)
                    if found: return found
            return {}

        custom_metrics = get_metrics_anywhere(result, "custom_metrics")
        policy_rewards = get_metrics_anywhere(result, "policy_reward_mean")
        
        # 만약 policy_rewards가 비어있다면, 기존 방식으로 안전하게 가져옵니다.
        if not policy_rewards:
            policy_rewards = result.get('env_runners', {}).get('policy_reward_mean', {})

        print(f">>> [디버깅] custom_metrics 딕셔너리 내용: {custom_metrics}")

        wandb.log({
            "iteration": i + 1,
            "reward/mean_reward": mean_reward,
            "reward/observer_policy": policy_rewards.get("observer_policy", 0),
            "reward/drone_policy": policy_rewards.get("drone_policy", 0),
            "metrics/success_rate": custom_metrics.get("success_rate_mean", 0),
            "metrics/crash_rate": custom_metrics.get("crash_rate_mean", 0),
            "metrics/min_drone_dist": custom_metrics.get("min_drone_dist_mean", 0),
            "metrics/min_obs_dist": custom_metrics.get("min_obs_dist_mean", 0),
            "metrics/explored_area": custom_metrics.get("explored_area_mean", 0),
        })
        
        if (i + 1) % 500 == 0:
            checkpoint_path = algo.save(checkpoint_dir)
            print(f"==> Checkpoint saved at: {checkpoint_path}")

    wandb.finish()
    ray.shutdown()

if __name__ == "__main__":
    main()