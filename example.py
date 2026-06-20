import argparse
import os
from datetime import datetime
from pathlib import Path
import random
import math
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
import numpy as np
from hemac import HeMAC_v0
from hemac.rllib_policy import register_hemac_rllib_models
import time
import pygame
from PIL import Image

try:
    from pygame._sdl2.video import Renderer as SdlRenderer
    from pygame._sdl2.video import Texture as SdlTexture
    from pygame._sdl2.video import Window as SdlWindow
except Exception:
    SdlRenderer = None
    SdlTexture = None
    SdlWindow = None


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
OBS_GRID_SIZE = 20
SECTOR_FEATURE_COUNT = OBS_GRID_SIZE * OBS_GRID_SIZE + 4
RELATIVE_MAP_SIZE = OBS_GRID_SIZE * 2
AUTO_PLAY_DELAY_SECONDS = 0.08
OBS_PANEL_WIDTH = 300
OBS_PANEL_HEIGHT = 240
OBS_PANEL_MARGIN = 12
OBS_WINDOW_PADDING = 12
OBS_WINDOW_HEADER_HEIGHT = 42


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


def wait_for_playback(playback_mode, delay_seconds=AUTO_PLAY_DELAY_SECONDS):
    """Wait according to playback mode and stop on window close/escape."""
    window_close_event = getattr(pygame, "WINDOWCLOSE", None)
    if playback_mode == "auto":
        end_time = time.time() + delay_seconds
        while time.time() < end_time:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (window_close_event is not None and event.type == window_close_event):
                    return False
                if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
            time.sleep(0.001)
        return True

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (window_close_event is not None and event.type == window_close_event):
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    return True
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
        time.sleep(0.001)


class ObservationDebugWindow:
    """Separate SDL2 window used for observation debug panels."""

    def __init__(self):
        self.window = None
        self.renderer = None
        self.available = all(obj is not None for obj in (SdlWindow, SdlRenderer, SdlTexture))

    def ensure(self, size):
        """Create or resize the observation window."""
        if not self.available:
            return False

        normalized_size = (int(size[0]), int(size[1]))
        if self.window is None:
            self.window = SdlWindow(
                title="HeMAC Observation Debug",
                size=normalized_size,
                position=(40, 50),
            )
            self.renderer = SdlRenderer(self.window)
            self.renderer.draw_color = (5, 8, 12, 255)
        elif tuple(self.window.size) != normalized_size:
            self.window.size = normalized_size
        return True

    def present(self, surface):
        """Draw a pygame surface into the separate observation window."""
        if not self.ensure(surface.get_size()):
            return False
        texture = SdlTexture.from_surface(self.renderer, surface)
        self.renderer.clear()
        self.renderer.blit(texture)
        self.renderer.present()
        return True

    def close(self):
        """Close the observation window if it exists."""
        self.renderer = None
        if self.window is not None:
            self.window.destroy()
            self.window = None


def _extract_observation_debug(observation):
    """Split an observation into base features and sector-based features."""
    if isinstance(observation, dict):
        vector_obs = np.asarray(observation.get("vector", []), dtype=np.float32).reshape(-1)
        relative_map = np.asarray(observation.get("relative_map", []), dtype=np.float32)
        goal_relative_sector = None
        if vector_obs.size >= 2:
            goal_relative_sector = tuple(int(v) for v in vector_obs[-2:])
            vector_obs = vector_obs[:-2]
        return {
            "mode": "relative_map",
            "base_obs": vector_obs,
            "coverage_map": relative_map[:, :, 0] if relative_map.size else None,
            "valid_mask": relative_map[:, :, 1] if relative_map.size else None,
            "goal_relative_sector": goal_relative_sector,
        }

    obs = np.asarray(observation, dtype=np.float32).reshape(-1)
    if obs.size < SECTOR_FEATURE_COUNT:
        return {
            "mode": "flat",
            "base_obs": obs,
            "coverage_map": None,
            "self_sector": None,
            "goal_relative_sector": None,
        }

    base_obs = obs[:-SECTOR_FEATURE_COUNT]
    sector_obs = obs[-SECTOR_FEATURE_COUNT:]
    coverage_map = sector_obs[: OBS_GRID_SIZE * OBS_GRID_SIZE].reshape(OBS_GRID_SIZE, OBS_GRID_SIZE)

    return {
        "mode": "flat",
        "base_obs": base_obs,
        "coverage_map": coverage_map,
        "self_sector": tuple(int(v) for v in sector_obs[-4:-2]),
        "goal_relative_sector": tuple(int(v) for v in sector_obs[-2:]),
    }


def _base_observation_labels(agent_id, base_obs):
    """Return readable labels for the non-sector observation slice."""
    if "observer" in agent_id:
        labels = ["orientation", "dist_right", "dist_up", "dist_left", "dist_down"]
    else:
        labels = ["dist_right", "dist_up", "dist_left", "dist_down"]
        drone_pair_count = max((len(base_obs) - 4) // 2, 0)
        for idx in range(drone_pair_count):
            labels.extend([f"peer_{idx}_dx", f"peer_{idx}_dy"])

    if len(labels) < len(base_obs):
        labels.extend([f"pad_{idx}" for idx in range(len(base_obs) - len(labels))])

    return labels[: len(base_obs)]


def _format_action(action):
    """Format an action for the debug overlay."""
    if action is None:
        return "None"
    if isinstance(action, np.ndarray):
        values = action.tolist()
    elif isinstance(action, (list, tuple)):
        values = list(action)
    else:
        return str(action)
    return "[" + ", ".join(f"{float(v):.2f}" for v in values) + "]"


def draw_observation_panel(
    surface,
    agent_id,
    observation,
    reward,
    action,
    termination,
    truncation,
    panel_rect,
    is_active=False,
):
    """Draw one agent observation panel."""
    obs_debug = _extract_observation_debug(observation)
    base_obs = obs_debug["base_obs"]
    coverage_map = obs_debug["coverage_map"]
    goal_relative_sector = obs_debug.get("goal_relative_sector")
    self_sector = obs_debug.get("self_sector")
    valid_mask = obs_debug.get("valid_mask")
    is_relative_map = obs_debug["mode"] == "relative_map"
    panel_width = panel_rect.width
    panel_height = panel_rect.height
    heatmap_size = min(140, panel_height - 88)
    heatmap_cell = heatmap_size // OBS_GRID_SIZE

    panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
    panel.fill((8, 12, 16, 215))

    title_font = pygame.font.SysFont("Trebuchet MS", 16)
    body_font = pygame.font.SysFont("Trebuchet MS", 12)
    tiny_font = pygame.font.SysFont("Trebuchet MS", 11)

    title = title_font.render(f"Observation Debug: {agent_id}", True, (245, 248, 250))
    panel.blit(title, (10, 8))

    meta_lines = [
        f"obs len: {int(sum(np.asarray(v).size for v in observation.values())) if isinstance(observation, dict) else len(np.asarray(observation).reshape(-1))}",
        f"reward: {float(reward):.2f}",
        f"action: {_format_action(action)}",
        f"done: {termination or truncation}",
    ]
    for idx, line in enumerate(meta_lines):
        panel.blit(body_font.render(line, True, (215, 225, 235)), (10, 28 + idx * 14))

    heatmap_x = 10
    heatmap_y = 88
    pygame.draw.rect(panel, (32, 40, 52), pygame.Rect(heatmap_x - 2, heatmap_y - 2, heatmap_size + 4, heatmap_size + 4))

    if coverage_map is not None:
        map_size = RELATIVE_MAP_SIZE if is_relative_map else OBS_GRID_SIZE
        heatmap_cell = heatmap_size // map_size
        for display_row in range(map_size):
            grid_y = map_size - 1 - display_row
            for grid_x in range(map_size):
                if valid_mask is not None and valid_mask[grid_y, grid_x] <= 0.0:
                    color = (12, 16, 20)
                else:
                    coverage = float(coverage_map[grid_y, grid_x])
                    if coverage <= 0.0:
                        color = (26, 38, 56)
                    else:
                        intensity = min(max(coverage, 0.0), 1.0)
                        color = (
                            int(40 + 40 * intensity),
                            int(85 + 150 * intensity),
                            int(55 + 70 * intensity),
                        )
                rect = pygame.Rect(
                    heatmap_x + grid_x * heatmap_cell,
                    heatmap_y + display_row * heatmap_cell,
                    heatmap_cell,
                    heatmap_cell,
                )
                pygame.draw.rect(panel, color, rect)
                pygame.draw.rect(panel, (70, 82, 96), rect, width=1)

        if is_relative_map:
            self_map_x = OBS_GRID_SIZE
            self_map_y = OBS_GRID_SIZE
            marker_x = heatmap_x + self_map_x * heatmap_cell + heatmap_cell // 2
            marker_y = heatmap_y + (map_size - 1 - self_map_y) * heatmap_cell + heatmap_cell // 2
            pygame.draw.circle(panel, (255, 255, 255), (marker_x, marker_y), max(heatmap_cell // 3, 3), width=2)
            if goal_relative_sector is not None:
                goal_map_x = self_map_x + goal_relative_sector[0]
                goal_map_y = self_map_y + goal_relative_sector[1]
                if 0 <= goal_map_x < map_size and 0 <= goal_map_y < map_size:
                    goal_x = heatmap_x + goal_map_x * heatmap_cell + heatmap_cell // 2
                    goal_y = heatmap_y + (map_size - 1 - goal_map_y) * heatmap_cell + heatmap_cell // 2
                    pygame.draw.line(panel, (255, 105, 105), (goal_x - 4, goal_y - 4), (goal_x + 4, goal_y + 4), width=2)
                    pygame.draw.line(panel, (255, 105, 105), (goal_x + 4, goal_y - 4), (goal_x - 4, goal_y + 4), width=2)
        else:
            if self_sector is not None and self_sector[0] >= 0 and self_sector[1] >= 0:
                marker_x = heatmap_x + self_sector[0] * heatmap_cell + heatmap_cell // 2
                marker_y = heatmap_y + (map_size - 1 - self_sector[1]) * heatmap_cell + heatmap_cell // 2
                pygame.draw.circle(panel, (255, 255, 255), (marker_x, marker_y), max(heatmap_cell // 3, 3), width=2)

            if self_sector is not None and goal_relative_sector is not None:
                goal_sector = (
                    self_sector[0] + goal_relative_sector[0],
                    self_sector[1] + goal_relative_sector[1],
                )
                if 0 <= goal_sector[0] < map_size and 0 <= goal_sector[1] < map_size:
                    goal_x = heatmap_x + goal_sector[0] * heatmap_cell + heatmap_cell // 2
                    goal_y = heatmap_y + (map_size - 1 - goal_sector[1]) * heatmap_cell + heatmap_cell // 2
                    pygame.draw.line(panel, (255, 105, 105), (goal_x - 4, goal_y - 4), (goal_x + 4, goal_y + 4), width=2)
                    pygame.draw.line(panel, (255, 105, 105), (goal_x + 4, goal_y - 4), (goal_x - 4, goal_y + 4), width=2)

    panel.blit(tiny_font.render("relative map" if is_relative_map else "coverage map", True, (210, 220, 230)), (heatmap_x, heatmap_y - 16))
    panel.blit(tiny_font.render("self: white circle", True, (210, 220, 230)), (heatmap_x, heatmap_y + heatmap_size + 4))
    panel.blit(tiny_font.render("goal: red x", True, (210, 220, 230)), (heatmap_x, heatmap_y + heatmap_size + 16))

    labels = _base_observation_labels(agent_id, base_obs)
    for idx, (label, value) in enumerate(zip(labels, base_obs)):
        text = tiny_font.render(f"{label}: {float(value): .3f}", True, (220, 228, 236))
        panel.blit(text, (heatmap_x + heatmap_size + 14, 88 + idx * 12))

    if self_sector is not None:
        panel.blit(
            tiny_font.render(f"self sector: {self_sector}", True, (220, 228, 236)),
            (heatmap_x + heatmap_size + 14, 60),
        )
    if goal_relative_sector is not None:
        panel.blit(
            tiny_font.render(f"goal rel: {goal_relative_sector}", True, (255, 180, 180)),
            (heatmap_x + heatmap_size + 14, 72),
        )

    border_color = (255, 220, 120) if is_active else (90, 100, 116)
    pygame.draw.rect(panel, border_color, panel.get_rect(), width=2)
    surface.blit(panel, panel_rect.topleft)


def draw_observation_overlays(agent_debug_state, active_agent, playback_mode, observation_window=None):
    """Draw observation panels for all known agents."""
    agent_ids = list(agent_debug_state.keys())
    if not agent_ids:
        return

    columns = 2
    rows = int(math.ceil(len(agent_ids) / columns))
    canvas_width = columns * OBS_PANEL_WIDTH + (columns - 1) * OBS_PANEL_MARGIN + OBS_WINDOW_PADDING * 2
    canvas_height = rows * OBS_PANEL_HEIGHT + (rows - 1) * OBS_PANEL_MARGIN + OBS_WINDOW_PADDING * 2 + OBS_WINDOW_HEADER_HEIGHT
    canvas = pygame.Surface((canvas_width, canvas_height))
    canvas.fill((6, 10, 14))
    start_x = OBS_WINDOW_PADDING
    start_y = OBS_WINDOW_PADDING + OBS_WINDOW_HEADER_HEIGHT

    for idx, agent_id in enumerate(agent_ids):
        row = idx // columns
        col = idx % columns
        panel_rect = pygame.Rect(
            start_x + col * (OBS_PANEL_WIDTH + OBS_PANEL_MARGIN),
            start_y + row * (OBS_PANEL_HEIGHT + OBS_PANEL_MARGIN),
            OBS_PANEL_WIDTH,
            OBS_PANEL_HEIGHT,
        )
        state = agent_debug_state[agent_id]
        draw_observation_panel(
            surface=canvas,
            agent_id=agent_id,
            observation=state["observation"],
            reward=state["reward"],
            action=state["action"],
            termination=state["termination"],
            truncation=state["truncation"],
            panel_rect=panel_rect,
            is_active=(agent_id == active_agent),
        )

    help_font = pygame.font.SysFont("Trebuchet MS", 16)
    help_bg = pygame.Surface((canvas_width - OBS_WINDOW_PADDING * 2, 30), pygame.SRCALPHA)
    help_bg.fill((8, 12, 16, 190))
    canvas.blit(help_bg, (OBS_WINDOW_PADDING, 8))
    if playback_mode == "auto":
        help_label = "Auto playback | Esc/Q: quit"
    else:
        help_label = "Space: next step (all agents) | Esc/Q: quit"
    help_text = help_font.render(help_label, True, (240, 248, 255))
    canvas.blit(help_text, (OBS_WINDOW_PADDING + 8, 14))

    if observation_window is not None and observation_window.present(canvas):
        return

    surface = pygame.display.get_surface()
    if surface is None:
        return
    surface.blit(canvas, (0, 0))
    pygame.display.flip()


def run_single_episode(env, algo, eval_seed, playback_mode="step", observation_window=None):
    """Run one evaluation episode and return the final info."""
    env.reset(seed=eval_seed)

    print(f"Evaluation seed: {eval_seed}")
    last_info = {}
    total_agent_turns = 0
    frames = []
    possible_agents = list(getattr(env, "possible_agents", []))
    agent_debug_state = {
        agent_id: {
            "observation": None,
            "reward": 0.0,
            "action": None,
            "termination": False,
            "truncation": False,
        }
        for agent_id in possible_agents
    }
    agents_per_step = max(len(possible_agents), 1)

    def get_policy_id(agent_id):
        if "observer" in agent_id:
            return "observer_policy"
        if "drone" in agent_id:
            return "drone_policy"
        return None

    def refresh_agent_debug_state():
        rewards = getattr(env, "rewards", {})
        terminations = getattr(env, "terminations", {})
        truncations = getattr(env, "truncations", {})
        for agent_id, state in agent_debug_state.items():
            try:
                state["observation"] = env.observe(agent_id)
            except Exception:
                pass
            if isinstance(rewards, dict):
                state["reward"] = rewards.get(agent_id, state["reward"])
            if isinstance(terminations, dict):
                state["termination"] = terminations.get(agent_id, state["termination"])
            if isinstance(truncations, dict):
                state["truncation"] = truncations.get(agent_id, state["truncation"])

    def render_step_state():
        env.render()
        draw_observation_overlays(
            agent_debug_state,
            getattr(env, "agent_selection", None),
            playback_mode,
            observation_window=observation_window,
        )
        frame = capture_pygame_frame()
        if frame is not None:
            frames.append(frame)

    refresh_agent_debug_state()
    render_step_state()
    if not wait_for_playback(playback_mode):
        gif_path = save_gif(frames, eval_seed)
        if gif_path is not None:
            print(f"Saved GIF: {gif_path}")
        return last_info

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

        agent_debug_state[agent] = {
            "observation": observation,
            "reward": reward,
            "action": action,
            "termination": termination,
            "truncation": truncation,
        }
        env.step(action)

        if total_agent_turns % agents_per_step == 0:
            refresh_agent_debug_state()
            render_step_state()
            episode_done = all(
                agent_debug_state[agent_id]["termination"] or agent_debug_state[agent_id]["truncation"]
                for agent_id in possible_agents
            )
            if not episode_done and not wait_for_playback(playback_mode):
                break

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


def run_trained_model_simulation(playback_mode="step"):
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
    # checkpoint_path = os.path.abspath(find_latest_checkpoint())
    checkpoint_path = os.path.abspath("./src/train/hemac_checkpoints/checkpoint_00100")
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
    observation_window = ObservationDebugWindow()
    print(f"시뮬레이션을 시작합니다. 재생 모드: {playback_mode}")

    seed_base = random.randint(0, 9999)
    eval_seeds = [seed_base + offset for offset in range(NUM_EVAL_SEEDS)]
    print(f"Running {NUM_EVAL_SEEDS} evaluation seeds: {eval_seeds}")

    results = []
    for idx, eval_seed in enumerate(eval_seeds, start=1):
        print(f"\n=== Evaluation {idx}/{NUM_EVAL_SEEDS} ===")
        last_info = run_single_episode(
            env,
            algo,
            eval_seed,
            playback_mode=playback_mode,
            observation_window=observation_window,
        )
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
    observation_window.close()
    env.close()
    # ray.shutdown()

def parse_args():
    """Parse CLI arguments for example playback."""
    parser = argparse.ArgumentParser(description="Run a trained HeMAC policy visualization.")
    parser.add_argument(
        "--playback",
        choices=("step", "auto"),
        default="step",
        help="Visualization playback mode.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_trained_model_simulation(playback_mode=args.playback)
