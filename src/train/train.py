import os
import ray
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import wandb

from hemac import HeMAC_v0

class HeMACCallbacks(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        final_info = {}
        
        # [방법 1] 정석적인 RLlib 데이터 추출
        # AEC 환경에서는 마지막 턴을 수행한 에이전트(예: drone_2)의 info에 값이 들어갑니다.
        # 따라서 현재 에피소드에 참여한 모든 에이전트의 info를 뒤져서 값을 찾습니다.
        agents = []
        if hasattr(episode, "get_agents"):
            agents = episode.get_agents()
        elif hasattr(episode, "agent_rewards"):
            agents = [key[0] for key in episode.agent_rewards.keys()]

        for agent_id in agents:
            info = episode.last_info_for(agent_id) or {}
            if "min_drone_dist" in info:
                final_info = info
                break
        print(f'final_info1: {final_info}')
        
        # [방법 2] 만약 위 방법으로 못 찾았다면, 래퍼를 완전히 뜯어보는 BFS 탐색을 수행합니다.
        if not final_info:
            print(final_info)
            env = base_env.get_sub_environments()[env_index]
            envs_to_check = [env]
            
            while envs_to_check:
                curr = envs_to_check.pop(0)
                
                # 우리가 찾는 원본 HeMAC 클래스인지 확인
                if hasattr(curr, "min_drone_dist"):
                    final_info = {
                        "min_drone_dist": float(curr.min_drone_dist),
                        "min_obs_dist": float(curr.min_obs_dist),
                        "explored_area": float(len(curr.explored_grids) * 400),
                        "success": (float(curr.min_obs_dist) < 50),
                        "fatal_crash": curr.collided
                    }
                    print(f'final_info2: {final_info}')
                    break
                
                # 하위 래퍼 탐색 대기열 추가
                if hasattr(curr, "env") and curr.env is not None:
                    envs_to_check.append(curr.env)
                if hasattr(curr, "unwrapped") and curr.unwrapped is not curr:
                    envs_to_check.append(curr.unwrapped)

        # 최종 값 추출 (어느 방법으로든 찾지 못한 경우 99999.0 등 기본값)
        min_drone = final_info.get("min_drone_dist", 99999.0)
        min_obs = final_info.get("min_obs_dist", 99999.0)
        area = final_info.get("explored_area", 0.0)
        
        # 초기값 보정
        if min_drone == 99999.0: min_drone = 0.0
        if min_obs == 99999.0: min_obs = 0.0

        # wandb 및 터미널 출력용 custom_metrics 할당
        episode.custom_metrics["min_drone_dist"] = float(min_drone)
        episode.custom_metrics["min_obs_dist"] = float(min_obs)
        episode.custom_metrics["explored_area"] = float(area)
        episode.custom_metrics["success_rate"] = 1.0 if final_info.get("success", False) else 0.0
        episode.custom_metrics["crash_rate"] = 1.0 if final_info.get("fatal_crash", False) else 0.0


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
        "max_obstacles": 2,
        "poi_config": [{"speed": 0}] 
    }
    return PettingZooEnv(HeMAC_v0.env(**env_config))


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
        if "observer" in agent_id: return "observer_policy"
        elif "drone" in agent_id: return "drone_policy"

    config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .callbacks(HeMACCallbacks)
        .environment(env=env_name)
        .env_runners(num_env_runners=4) 
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
        .training(train_batch_size=4000, lr=5e-5, gamma=0.99, grad_clip=1.0, clip_param=0.2)
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
            
        policy_rewards = result.get('policy_reward_mean', {})
        if not policy_rewards:
            policy_rewards = result.get('env_runners', {}).get('policy_reward_mean', {})

        print(f">>> [디버깅] custom_metrics: {custom_metrics}")

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
            algo.save(checkpoint_dir)

    wandb.finish()
    ray.shutdown()

if __name__ == "__main__":
    main()