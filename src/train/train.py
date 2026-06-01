import os
import ray
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks

# 수정한 HeMAC 환경 임포트
from hemac import HeMAC_v0

from torch.utils.tensorboard import SummaryWriter

class HeMACCallbacks(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        # 우리가 원하는 경로에 직접 고정 폴더를 개설합니다.
        log_dir = "/home/cau/ray_results/HeMAC_Asymmetric_PPO"
        os.makedirs(log_dir, exist_ok=True)
        # 텐서보드 파일 작성기를 강제로 수동 생성합니다.
        self.writer = SummaryWriter(log_dir=log_dir)

    def on_episode_end(
        self,
        *,
        worker,
        base_env,
        policies,
        episode,
        env_index,
        **kwargs
    ):
        info = episode.last_info_for("observer_0")
        if info is not None:
            success = 1.0 if info.get("success", False) else 0.0
            crash = 1.0 if info.get("fatal_crash", False) else 0.0
            
            # 1. 기존 RLlib 내장 로깅 시스템에 기록
            episode.custom_metrics["success_rate"] = success
            episode.custom_metrics["crash_rate"] = crash

            # 2. [강제 주입] 텐서보드 파일에 직접 물리적으로 기록을 박아버립니다.
            # episode.length나 다른 고유한 타임스텝 기준으로 기록을 쌓습니다.
            total_steps = worker.policy_map["observer_policy"].global_timestep
            self.writer.add_scalar("custom_metrics/success_rate_mean", success, total_steps)
            self.writer.add_scalar("custom_metrics/crash_rate_mean", crash, total_steps)

def env_creator(config):
    env_config = {
        # 유인기 1대 (느리고 좁은 시야)
        "n_observers": 1,
        "observer_speed": 5, 

        # 무인기 3대 (빠르고 넓은 하향 시야)
        "n_drones": 3,
        "n_provisioners": 0,
        "drone_config": {
            "drone_max_speed": 25,
            "drone_max_thrust": 8,
            "drones_starting_pos": [],
        },
        
        # 맵 설정
        "max_obstacles": 5,
        "poi_config": [{"speed": 0}] # 1개의 목표 지점(도착지)
    }
    env = HeMAC_v0.env(**env_config)
    return PettingZooEnv(env)

def main():
    # 1. Ray 초기화
    ray.init()
    
    env_name = "hemac_asymmetric_env"
    register_env(env_name, env_creator)

    # 2. 임시 환경을 생성하여 관측(Obs) 및 행동(Action) 공간 추출
    temp_env = env_creator({})
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space

    # 3. 정책(Policy) 분리: 무인기용 1개, 유인기용 1개
    policies = {
        "observer_policy": (None, obs_space["observer_0"], act_space["observer_0"], {}),
        # drone_0의 공간을 기준으로 삼아 모든 무인기가 공유
        "drone_policy": (None, obs_space["drone_0"], act_space["drone_0"], {}) 
    }

    # 4. 에이전트 ID에 따라 어떤 정책을 사용할지 매핑하는 함수
    def policy_mapping_fn(agent_id, episode, **kwargs):
        if "observer" in agent_id:
            return "observer_policy"
        elif "drone" in agent_id:
            return "drone_policy"

    # 5. PPO 알고리즘 세팅
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
            gamma=0.99, # 미래 보상 할인율 (목표 도달까지 길게 보려면 높게 유지)
        )
        .debugging(log_level="WARN")
    )

    # 6. 알고리즘 빌드 및 학습 루프
    print("RLlib PPO 알고리즘 빌드 중...")
    algo = config.build()
    
    checkpoint_dir = "./hemac_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("학습 루프 시작...")
    num_iterations = 1000 # 전체 학습 횟수
    
    for i in range(num_iterations):
        result = algo.train()
        
        # 전체 팀의 평균 보상 및 정책별 보상 출력
        mean_reward = result.get('env_runners', {}).get('episode_reward_mean', result.get('episode_reward_mean', 0))
        
        print(f"--- Iteration {i+1} ---")
        print(f"Mean Reward: {mean_reward:.2f}")
        
        if 'policy_reward_mean' in result:
             print(f"Policy Rewards: {result['policy_reward_mean']}")
        
        # 10 이터레이션마다 모델 가중치 저장
        if (i + 1) % 100 == 0:
            # RLlib 최신 API에서는 save()가 checkpoint 경로를 반환합니다.
            checkpoint_path = algo.save(checkpoint_dir)
            print(f"==> Checkpoint saved at: {checkpoint_path}")

    # 학습 완료 후 종료
    ray.shutdown()

if __name__ == "__main__":
    main()