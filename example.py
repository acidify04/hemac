import os
from datetime import datetime
from pathlib import Path
import random
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from hemac import HeMAC_v0
from hemac.rllib_policy import register_hemac_rllib_models
import time
import pygame
from PIL import Image


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

NUM_EVAL_SEEDS = 10
VISUALIZATION_DIR = Path("./visualization")


def find_latest_checkpoint():
    """Return the newest available checkpoint directory."""
    candidate_roots = [
        Path("./hemac_checkpoints"),
        Path("./src/train/hemac_checkpoints"),
    ]
    checkpoints = []
    for root in candidate_roots:
        if not root.exists():
            continue
        checkpoints.extend(path for path in root.iterdir() if path.is_dir() and path.name.startswith("checkpoint_"))

    if not checkpoints:
        raise FileNotFoundError("No checkpoint_* directory found in ./hemac_checkpoints or ./src/train/hemac_checkpoints")

    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def hold_window_open(seconds=5.0):
    """Keep the pygame window visible briefly after the episode ends."""
    end_time = time.time() + seconds
    while time.time() < end_time:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
        time.sleep(0.05)


def capture_pygame_frame():
    """Capture the current pygame display surface as an RGB array."""
    surface = pygame.display.get_surface()
    if surface is None:
        return None
    frame = pygame.surfarray.array3d(surface)
    return frame.transpose((1, 0, 2)).copy()


def save_gif(frames, eval_seed):
    """Save rollout frames as a GIF under visualization/."""
    if not frames:
        return None

    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gif_path = VISUALIZATION_DIR / f"example_seed_{eval_seed}_{timestamp}.gif"
    pil_frames = [Image.fromarray(frame.astype("uint8")) for frame in frames]
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=80,
        loop=0,
    )
    return gif_path


def run_single_episode(env, algo, eval_seed):
    """Run one evaluation episode and return the final info."""
    env.reset(seed=eval_seed)

    print(f"Evaluation seed: {eval_seed}")
    last_info = {}
    total_agent_turns = 0
    frames = []

    def get_policy_id(agent_id):
        if "observer" in agent_id:
            return "observer_policy"
        if "drone" in agent_id:
            return "drone_policy"
        return None

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        total_agent_turns += 1
        if info:
            last_info = info

        if termination or truncation:
            action = None
        else:
            policy_id = get_policy_id(agent)
            if policy_id:
                action = algo.compute_single_action(
                    observation=observation,
                    policy_id=policy_id,
                    explore=False,
                )
            else:
                action = env.action_space(agent).sample()

        env.step(action)
        env.render()
        frame = capture_pygame_frame()
        if frame is not None:
            frames.append(frame)
        time.sleep(0.01)

    print(f"Episode finished after {total_agent_turns} agent turns.")
    if last_info:
        print(
            "Final info:",
            {
                "success": last_info.get("success"),
                "goal_found": last_info.get("goal_found"),
                "fatal_crash": last_info.get("fatal_crash"),
                "drone_crash": last_info.get("drone_crash"),
                "observer_crash": last_info.get("observer_crash"),
                "min_drone_dist": last_info.get("min_drone_dist"),
                "min_obs_dist": last_info.get("min_obs_dist"),
                "explored_area": last_info.get("explored_area"),
            },
        )

    gif_path = save_gif(frames, eval_seed)
    if gif_path is not None:
        print(f"Saved GIF: {gif_path}")

    return last_info


def run_trained_model_simulation():
    # 1. Ray 및 가상환경 내 초기화
    ray.init(ignore_reinit_error=True)
    register_hemac_rllib_models()

    def env_creator(config):
        # 훈련 때 사용했던 동일한 스펙을 반환해야 합니다. (render_mode 제외)
        train_env_config = {
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
            "min_obstacles": 0,
            "max_obstacles": 0,
            "poi_config": [GOAL_CONFIG],
        }
        env = HeMAC_v0.env(**train_env_config)
        return PettingZooEnv(env)

    # 학습 때 사용했던 정확히 그 이름으로 등록합니다.
    register_env("hemac_asymmetric_env", env_creator)

    # 2. 저장된 체크포인트로부터 알고리즘(모델) 로드
    # 저장된 폴더 경로를 지정합니다. (예: ./hemac_checkpoints 하위의 실제 체크포인트 폴더)
    checkpoint_path = os.path.abspath(find_latest_checkpoint())
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
        "known_goals": False,
        "max_cycles": 500,
        "drone_config": {
            "drone_max_speed": 25,
            "drone_max_thrust": 8,
            "drones_starting_pos": DRONE_START_POSITIONS,
        },
        
        # 맵 및 목적지 설정
        "min_obstacles": 0,
        "max_obstacles": 0,
        "poi_config": [GOAL_CONFIG],
        
        # [핵심] 화면 시각화 활성화
        "render_mode": "human" 
    }

    # 환경 생성
    env = HeMAC_v0.env(**env_config)
    print("시뮬레이션을 시작합니다. 창을 확인해 주세요.")

    seed_base = random.randint(0, 9999)
    eval_seeds = [seed_base + offset for offset in range(NUM_EVAL_SEEDS)]
    print(f"Running {NUM_EVAL_SEEDS} evaluation seeds: {eval_seeds}")

    results = []
    for idx, eval_seed in enumerate(eval_seeds, start=1):
        print(f"\n=== Evaluation {idx}/{NUM_EVAL_SEEDS} ===")
        last_info = run_single_episode(env, algo, eval_seed)
        results.append(last_info)
        hold_window_open(seconds=1.0)

    success_count = sum(1 for info in results if info.get("success", False))
    goal_found_count = sum(1 for info in results if info.get("goal_found", False))
    drone_crash_count = sum(1 for info in results if info.get("drone_crash", False))
    observer_crash_count = sum(1 for info in results if info.get("observer_crash", False))

    print("\n=== Summary ===")
    print(f"Success: {success_count}/{NUM_EVAL_SEEDS}")
    print(f"Goal found: {goal_found_count}/{NUM_EVAL_SEEDS}")
    print(f"Drone crash: {drone_crash_count}/{NUM_EVAL_SEEDS}")
    print(f"Observer crash: {observer_crash_count}/{NUM_EVAL_SEEDS}")

    hold_window_open(seconds=5.0)
    env.close()
    # ray.shutdown()

if __name__ == "__main__":
    run_trained_model_simulation()
