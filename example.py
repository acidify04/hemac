import os
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from hemac import HeMAC_v0
import time
def run_trained_model_simulation():
    # 1. Ray 및 가상환경 내 초기화
    ray.init(ignore_reinit_error=True)

    def env_creator(config):
        # 훈련 때 사용했던 동일한 스펙을 반환해야 합니다. (render_mode 제외)
        train_env_config = {
            "n_observers": 1,
            "observer_speed": 5, 
            "n_drones": 3,
            "n_provisioners": 0,
            "drone_config": {
                "drone_max_speed": 25,
                "drone_max_thrust": 8,
                # 수정 예시
                "drones_starting_pos": [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]],
            },
            "min_obstacles": 0,
            "max_obstacles": 0,
            "poi_config": [{"speed": 0}],
        }
        env = HeMAC_v0.env(**train_env_config)
        return PettingZooEnv(env)

    # 학습 때 사용했던 정확히 그 이름으로 등록합니다.
    register_env("hemac_asymmetric_env", env_creator)

    # 2. 저장된 체크포인트로부터 알고리즘(모델) 로드
    # 저장된 폴더 경로를 지정합니다. (예: ./hemac_checkpoints 하위의 실제 체크포인트 폴더)
    checkpoint_path = os.path.abspath("./src/train/hemac_checkpoints/checkpoint_00900")
    print(f"[{checkpoint_path}] 경로에서 학습된 모델을 불러오는 중...")
    algo = Algorithm.from_checkpoint(checkpoint_path)

    # 3. 평가용 비대칭 환경 구성 (학습 때 사용한 스펙과 완벽히 동일해야 합니다)
    env_config = {
        # 유인기 1대 (느린 속도)
        "n_observers": 1,
        "observer_speed": 5, 

        # 무인기 3대 (빠른 속도)
        "n_drones": 3,
        "n_provisioners": 0,
        "drone_config": {
            "drone_max_speed": 25,
            "drone_max_thrust": 8,
            "drones_starting_pos": [], 
        },
        
        # 맵 및 목적지 설정
        "min_obstacles": 0,
        "max_obstacles": 0,
        "poi_config": [{"speed": 0}],
        
        # [핵심] 화면 시각화 활성화
        "render_mode": "human" 
    }

    # 환경 생성 및 리셋
    env = HeMAC_v0.env(**env_config)
    env.reset(seed=0)

    print("시뮬레이션을 시작합니다. 창을 확인해 주세요.")

    # 4. 에이전트별 정책 매핑 헬퍼 함수
    def get_policy_id(agent_id):
        if "observer" in agent_id:
            return "observer_policy"
        elif "drone" in agent_id:
            return "drone_policy"
        return None


    print("AI 모델을 끄고 모든 에이전트가 무작위로 움직입니다.")
    
    # for agent in env.agent_iter():
    #     observation, reward, termination, truncation, info = env.last()
    #     if termination or truncation:
    #         action = None
    #     else:
    #         # this is where you would insert your policy
    #         action = env.action_space(agent).sample()
    #     env.step(action)
    #     env.render()
    #     time.sleep(0.01)
    #     # env.close()
    # time.sleep(2)
    # env.close()
    # ray.shutdown()
    # ray.shutdown()
    # # 5. 모델 기반 상호작용 루프 (사용자 제공 뼈대 유기적 수정)
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        
        if termination or truncation:
            action = None
        else:
            # 현재 에이전트 유형에 맞는 정책 네트워크(ID) 매핑
            policy_id = get_policy_id(agent)
            
            if policy_id:
                # 학습된 모델을 이용하여 최적의 행동 계산 (의사결정)
                action = algo.compute_single_action(
                    observation=observation,
                    policy_id=policy_id,
                    explore=False # 평가 모드이므로 무작위 탐색(Exploration)을 끕니다.
                )
            else:
                # 예외 처리용 폴백
                action = env.action_space(agent).sample()
                
        env.step(action)
        env.render()
        time.sleep(0.01)
        
    time.sleep(2)
    env.close()
    # ray.shutdown()

if __name__ == "__main__":
    run_trained_model_simulation()