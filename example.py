import argparse
import os
from datetime import datetime
from pathlib import Path
import pickle
import random
import math
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

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
    "boundary_margin": 60,
    "spawn_quadrant": ["bottom_right", "bottom_left", "top_right"],
}

CHECKPOINT_ROOTS = (
    PROJECT_ROOT / "hemac_checkpoints",
    PROJECT_ROOT / "src/train/hemac_checkpoints",
)

NUM_EVAL_SEEDS = 10
VISUALIZATION_DIR = Path("./visualization")
GLOBAL_MAP_SIZE = 40
LOCAL_MAP_SIZE = 20
ACTION_HISTORY_LENGTH = 5
ACTION_DIM = 3
AUTO_PLAY_DELAY_SECONDS = 0.02
OBS_PANEL_WIDTH = 640
OBS_PANEL_HEIGHT = 430
OBS_PANEL_MARGIN = 12
OBS_WINDOW_PADDING = 12
OBS_WINDOW_HEADER_HEIGHT = 42


def find_complete_checkpoints(required_policy_ids=("observer_policy", "drone_policy")):
    """Return complete checkpoints and their creation timestamps."""
    checkpoints = []
    for root in CHECKPOINT_ROOTS:
        if not root.exists():
            continue
        for state_path in root.rglob("algorithm_state.pkl"):
            checkpoint_dir = state_path.parent
            if not checkpoint_dir.name.startswith("checkpoint_"):
                continue
            if all(
                (checkpoint_dir / "policies" / policy_id / "policy_state.pkl").is_file()
                for policy_id in required_policy_ids
            ):
                checkpoints.append((state_path.stat().st_mtime_ns, checkpoint_dir))

    if not checkpoints:
        roots = ", ".join(str(root) for root in CHECKPOINT_ROOTS)
        raise FileNotFoundError(
            f"No complete checkpoint containing {required_policy_ids} found under: {roots}"
        )

    return checkpoints


def find_latest_checkpoint(required_policy_ids=("observer_policy", "drone_policy")):
    """Return the newest complete checkpoint found below the checkpoint roots."""
    checkpoints = find_complete_checkpoints(required_policy_ids)

    return max(checkpoints, key=lambda item: (item[0], str(item[1])))[1].resolve()


def find_checkpoint_by_iteration(
    iteration,
    required_policy_ids=("observer_policy", "drone_policy"),
):
    """Return the newest complete checkpoint matching one training iteration."""
    target_iteration = int(iteration)
    matches = []
    for created_at, checkpoint_dir in find_complete_checkpoints(required_policy_ids):
        suffix = checkpoint_dir.name.removeprefix("checkpoint_")
        try:
            checkpoint_iteration = int(suffix)
        except ValueError:
            continue
        if checkpoint_iteration == target_iteration:
            matches.append((created_at, checkpoint_dir))

    if not matches:
        roots = ", ".join(str(root) for root in CHECKPOINT_ROOTS)
        raise FileNotFoundError(
            f"No complete checkpoint found for iteration {target_iteration} under: {roots}"
        )

    return max(matches, key=lambda item: (item[0], str(item[1])))[1].resolve()


def load_policy_weights_from_checkpoint(checkpoint_dir, policy_id):
    """Load one policy's weight dict from an RLlib checkpoint directory."""
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


def restore_policy_from_checkpoint(algo, policy_id, checkpoint_dir):
    """Overwrite one policy inside a restored RLlib Algorithm."""
    weights = load_policy_weights_from_checkpoint(checkpoint_dir, policy_id)
    algo.set_weights({policy_id: weights})


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


def wait_for_playback(
    playback_state,
    delay_seconds=AUTO_PLAY_DELAY_SECONDS,
    on_mode_change=None,
):
    """Wait for the current playback mode while allowing live F1 toggling."""
    window_close_event = getattr(pygame, "WINDOWCLOSE", None)
    auto_deadline = time.monotonic() + delay_seconds
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (window_close_event is not None and event.type == window_close_event):
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                if event.key == pygame.K_F1:
                    playback_state["mode"] = (
                        "step" if playback_state["mode"] == "auto" else "auto"
                    )
                    print(f"Playback mode: {playback_state['mode']}")
                    auto_deadline = time.monotonic() + delay_seconds
                    if on_mode_change is not None:
                        on_mode_change()
                elif event.key == pygame.K_SPACE and playback_state["mode"] == "step":
                    return True

        if playback_state["mode"] == "auto" and time.monotonic() >= auto_deadline:
            return True
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


def _extract_observation_debug(agent_id, observation):
    """Normalize current observation formats into a debug-friendly dict."""
    if isinstance(observation, dict):
        global_map = np.asarray(observation.get("global_map", []), dtype=np.float32)
        local_map = np.asarray(observation.get("local_map", []), dtype=np.float32)
        central_map = np.asarray(observation.get("central_map", []), dtype=np.float32)
        central_vector = np.asarray(
            observation.get("central_vector", []),
            dtype=np.float32,
        ).reshape(-1)
        vector_obs = np.asarray(observation.get("vector", []), dtype=np.float32).reshape(-1)

        if global_map.ndim == 3 or local_map.ndim == 3:
            return {
                "mode": "multi_map",
                "vector_obs": vector_obs,
                "global_map": global_map if global_map.ndim == 3 else None,
                "local_map": local_map if local_map.ndim == 3 else None,
                "central_map": central_map if central_map.ndim == 3 else None,
                "central_vector": central_vector,
            }

        relative_map = np.asarray(observation.get("relative_map", []), dtype=np.float32)
        return {
            "mode": "legacy_relative_map",
            "vector_obs": vector_obs,
            "relative_map": relative_map if relative_map.ndim == 3 else None,
        }

    obs = np.asarray(observation, dtype=np.float32).reshape(-1)
    return {
        "mode": "flat",
        "vector_obs": obs,
        "global_map": None,
        "local_map": None,
        "central_map": None,
        "central_vector": np.empty((0,), dtype=np.float32),
    }


def _observation_size(observation):
    """Return the flattened size of an observation object."""
    if isinstance(observation, dict):
        return int(sum(np.asarray(value).size for value in observation.values()))
    return int(np.asarray(observation).size)


def _map_channel_labels(agent_id, map_kind, channel_count):
    """Return human-readable channel labels for the current observation schema."""
    if map_kind == "central_map":
        labels = [
            "coverage",
            "boundary",
            "obstacle",
            "warning",
            "drones",
            "focal_drone",
            "observer",
            "goal",
        ]
    elif "observer" in agent_id:
        labels = ["coverage", "boundary", "obstacle", "warning", "drones", "goal"]
    else:
        labels = ["coverage", "boundary", "obstacle", "warning", "other_drones", "observer", "goal"]

    if channel_count > len(labels):
        labels.extend([f"channel_{idx}" for idx in range(len(labels), channel_count)])
    return labels[:channel_count]


def _blend_color(base_color, overlay_color, alpha):
    """Alpha-blend two RGB colors."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    base = np.asarray(base_color, dtype=np.float32)
    overlay = np.asarray(overlay_color, dtype=np.float32)
    return tuple(int(v) for v in (base * (1.0 - alpha) + overlay * alpha))


def _draw_map_thumbnail(panel, map_array, agent_id, map_kind, top_left, max_size, title_font, tiny_font):
    """Draw a composite map thumbnail for the current observation."""
    title_x, title_y = top_left
    panel.blit(title_font.render(map_kind, True, (240, 245, 250)), (title_x, title_y))

    if map_array is None or map_array.ndim != 3 or map_array.shape[0] == 0 or map_array.shape[1] == 0:
        panel.blit(tiny_font.render("missing", True, (220, 228, 236)), (title_x, title_y + 16))
        return pygame.Rect(title_x, title_y + 18, 0, 0)

    map_height, map_width, channel_count = map_array.shape
    cell_size = max(1, max_size // max(map_height, map_width))
    draw_width = map_width * cell_size
    draw_height = map_height * cell_size
    heatmap_x = title_x
    heatmap_y = title_y + 18
    heatmap_rect = pygame.Rect(heatmap_x, heatmap_y, draw_width, draw_height)
    pygame.draw.rect(panel, (32, 40, 52), heatmap_rect.inflate(4, 4))

    channel_labels = _map_channel_labels(agent_id, map_kind, channel_count)
    channel_index = {label: idx for idx, label in enumerate(channel_labels)}
    coverage = map_array[:, :, channel_index["coverage"]] if "coverage" in channel_index else None
    boundary = map_array[:, :, channel_index["boundary"]] if "boundary" in channel_index else None
    obstacle = map_array[:, :, channel_index["obstacle"]] if "obstacle" in channel_index else None
    warning = map_array[:, :, channel_index["warning"]] if "warning" in channel_index else None
    drone_layer = map_array[:, :, channel_index["drones"]] if "drones" in channel_index else None
    if drone_layer is None and "other_drones" in channel_index:
        drone_layer = map_array[:, :, channel_index["other_drones"]]
    observer_layer = map_array[:, :, channel_index["observer"]] if "observer" in channel_index else None
    focal_drone_layer = (
        map_array[:, :, channel_index["focal_drone"]]
        if "focal_drone" in channel_index
        else None
    )
    goal_layer = map_array[:, :, channel_index["goal"]] if "goal" in channel_index else None

    for display_row in range(map_height):
        grid_y = map_height - 1 - display_row
        for grid_x in range(map_width):
            boundary_value = float(boundary[grid_y, grid_x]) if boundary is not None else 1.0
            coverage_value = float(coverage[grid_y, grid_x]) if coverage is not None else 0.0
            obstacle_value = float(obstacle[grid_y, grid_x]) if obstacle is not None else 0.0
            warning_value = float(warning[grid_y, grid_x]) if warning is not None else 0.0

            if boundary is not None and boundary_value <= 0.0 and obstacle_value <= 0.0 and warning_value <= 0.0:
                color = (10, 14, 18)
            else:
                if coverage is not None:
                    intensity = float(np.clip(coverage_value, 0.0, 1.0))
                    color = (
                        int(26 + 32 * boundary_value),
                        int(52 + 170 * intensity),
                        int(36 + 72 * intensity),
                    )
                else:
                    boundary_intensity = float(np.clip(boundary_value, 0.0, 1.0))
                    color = (
                        int(18 + 26 * boundary_intensity),
                        int(28 + 42 * boundary_intensity),
                        int(42 + 112 * boundary_intensity),
                    )
                if obstacle_value > 0.0:
                    color = _blend_color(color, (190, 72, 52), min(obstacle_value, 1.0))
                if warning_value > 0.0:
                    color = _blend_color(color, (255, 150, 150), min(0.65 * warning_value, 1.0))

            rect = pygame.Rect(
                heatmap_x + grid_x * cell_size,
                heatmap_y + display_row * cell_size,
                cell_size,
                cell_size,
            )
            pygame.draw.rect(panel, color, rect)

            center = (rect.centerx, rect.centery)
            marker_radius = max(1, cell_size // 3)
            if drone_layer is not None and float(drone_layer[grid_y, grid_x]) > 0.0:
                pygame.draw.circle(panel, (90, 230, 255), center, marker_radius)
            if observer_layer is not None and float(observer_layer[grid_y, grid_x]) > 0.0:
                pygame.draw.circle(panel, (255, 226, 120), center, marker_radius + 1, width=1)
            if focal_drone_layer is not None and float(focal_drone_layer[grid_y, grid_x]) > 0.0:
                pygame.draw.circle(panel, (120, 255, 145), center, marker_radius + 2, width=1)
            if goal_layer is not None and float(goal_layer[grid_y, grid_x]) > 0.0:
                pygame.draw.line(panel, (255, 110, 110), (center[0] - 3, center[1] - 3), (center[0] + 3, center[1] + 3), width=1)
                pygame.draw.line(panel, (255, 110, 110), (center[0] + 3, center[1] - 3), (center[0] - 3, center[1] + 3), width=1)

    pygame.draw.rect(panel, (70, 82, 96), heatmap_rect, width=1)
    return heatmap_rect


def _map_stat_lines(prefix, map_array, agent_id, map_kind):
    """Summarize map channels with compact numeric stats."""
    if map_array is None or map_array.ndim != 3:
        return [f"{prefix}: missing"]

    lines = []
    channel_labels = _map_channel_labels(agent_id, map_kind, map_array.shape[-1])
    for idx, label in enumerate(channel_labels):
        channel = map_array[:, :, idx]
        if label in {"coverage", "boundary", "obstacle", "warning"}:
            lines.append(f"{prefix}.{label}: mean {float(np.mean(channel)):.3f}")
        else:
            lines.append(f"{prefix}.{label}: sum {float(np.sum(channel)):.1f}")
    return lines


def _action_history_lines(vector_obs):
    """Format the previous 5-step action history."""
    if vector_obs.size == 0:
        return ["action_history: empty"]

    if vector_obs.size % ACTION_DIM != 0:
        preview = ", ".join(f"{float(value):.2f}" for value in vector_obs[: min(len(vector_obs), 6)])
        return [f"vector: [{preview}]"]

    history = vector_obs.reshape(-1, ACTION_DIM)
    lines = []
    total_steps = history.shape[0]
    for idx, action_values in enumerate(history):
        age = total_steps - idx
        lines.append(
            f"a[-{age}]: [{float(action_values[0]):.2f}, {float(action_values[1]):.2f}, {float(action_values[2]):.2f}]"
        )
    return lines


def _central_vector_lines(central_vector):
    """Format the MAPPO critic's normalized relative entity coordinates."""
    if central_vector.size == 0:
        return ["central_vector: unavailable"]
    if central_vector.size % 2 != 0:
        preview = ", ".join(
            f"{float(value):+.3f}" for value in central_vector[: min(central_vector.size, 8)]
        )
        return [f"central_vector: [{preview}]"]

    relative_positions = central_vector.reshape(-1, 2)
    if len(relative_positions) == 4:
        labels = ["d1", "d2", "obs", "goal"]
    else:
        labels = [f"entity_{idx}" for idx in range(len(relative_positions))]

    return [
        f"{label}: ({float(position[0]):+.3f}, {float(position[1]):+.3f})"
        for label, position in zip(labels, relative_positions)
    ]


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
    obs_debug = _extract_observation_debug(agent_id, observation)
    vector_obs = obs_debug.get("vector_obs", np.empty((0,), dtype=np.float32))
    global_map = obs_debug.get("global_map")
    local_map = obs_debug.get("local_map")
    central_map = obs_debug.get("central_map")
    central_vector = obs_debug.get(
        "central_vector",
        np.empty((0,), dtype=np.float32),
    )
    panel_width = panel_rect.width
    panel_height = panel_rect.height

    panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
    panel.fill((8, 12, 16, 215))

    title_font = pygame.font.SysFont("Trebuchet MS", 16)
    body_font = pygame.font.SysFont("Trebuchet MS", 12)
    tiny_font = pygame.font.SysFont("Trebuchet MS", 11)

    title = title_font.render(f"Observation Debug: {agent_id}", True, (245, 248, 250))
    panel.blit(title, (10, 8))

    meta_lines = [
        f"obs len: {_observation_size(observation)}",
        f"reward: {float(reward):.2f}",
        f"action: {_format_action(action)}",
        f"done: {termination or truncation}",
    ]
    for idx, line in enumerate(meta_lines):
        panel.blit(body_font.render(line, True, (215, 225, 235)), (10, 28 + idx * 14))

    if obs_debug["mode"] == "multi_map":
        global_rect = _draw_map_thumbnail(
            panel,
            global_map,
            agent_id,
            "global_map",
            (10, 88),
            128,
            body_font,
            tiny_font,
        )
        local_rect = _draw_map_thumbnail(
            panel,
            local_map,
            agent_id,
            "local_map",
            (150, 88),
            128,
            body_font,
            tiny_font,
        )

        central_rect = None
        if central_map is not None:
            central_rect = _draw_map_thumbnail(
                panel,
                central_map,
                agent_id,
                "central_map",
                (290, 88),
                128,
                body_font,
                tiny_font,
            )

        stat_lines = _map_stat_lines("g", global_map, agent_id, "global")
        stat_lines.extend(_map_stat_lines("l", local_map, agent_id, "local"))
        if central_map is not None:
            stat_lines.extend(_map_stat_lines("c", central_map, agent_id, "central_map"))
        stats_x = 430 if central_map is not None else 290
        panel.blit(tiny_font.render("channel stats", True, (240, 245, 250)), (stats_x, 88))
        for idx, line in enumerate(stat_lines):
            text = tiny_font.render(line, True, (220, 228, 236))
            panel.blit(text, (stats_x, 104 + idx * 11))

        action_lines = _action_history_lines(vector_obs)
        action_title_y = max(global_rect.bottom, local_rect.bottom) + 16
        panel.blit(tiny_font.render("previous 5-step actions", True, (240, 245, 250)), (10, action_title_y))
        for idx, line in enumerate(action_lines):
            panel.blit(tiny_font.render(line, True, (220, 228, 236)), (10, action_title_y + 14 + idx * 12))

        if central_rect is not None:
            central_vector_y = central_rect.bottom + 16
            panel.blit(
                tiny_font.render("central relative (dx, dy)", True, (240, 245, 250)),
                (150, central_vector_y),
            )
            for idx, line in enumerate(_central_vector_lines(central_vector)):
                panel.blit(
                    tiny_font.render(line, True, (220, 228, 236)),
                    (150, central_vector_y + 14 + idx * 12),
                )

        legend_y = panel_height - 38
        legend_lines = [
            "bg: coverage/boundary, red: obstacle, rose: warning",
            "cyan: drone, green ring: focal, gold: observer, pink x: goal",
        ]
        for idx, legend in enumerate(legend_lines):
            panel.blit(tiny_font.render(legend, True, (210, 220, 230)), (10, legend_y + idx * 12))
    else:
        panel.blit(
            body_font.render("Current debug panel expects global_map/local_map/vector.", True, (255, 190, 190)),
            (10, 96),
        )
        fallback_lines = _action_history_lines(vector_obs)
        for idx, line in enumerate(fallback_lines):
            panel.blit(tiny_font.render(line, True, (220, 228, 236)), (10, 118 + idx * 12))

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
        help_label = "Auto playback | F1: step mode | Esc/Q: quit"
    else:
        help_label = "Step playback | Space: next step | F1: auto mode | Esc/Q: quit"
    help_text = help_font.render(help_label, True, (240, 248, 255))
    canvas.blit(help_text, (OBS_WINDOW_PADDING + 8, 14))

    if observation_window is not None and observation_window.present(canvas):
        return

    surface = pygame.display.get_surface()
    if surface is None:
        return
    surface.blit(canvas, (0, 0))
    pygame.display.flip()


def run_single_episode(env, algo, eval_seed, playback_state, observation_window=None):
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

    def render_step_state(capture_frame=True):
        env.render()
        draw_observation_overlays(
            agent_debug_state,
            getattr(env, "agent_selection", None),
            playback_state["mode"],
            observation_window=observation_window,
        )
        if capture_frame:
            frame = capture_pygame_frame()
            if frame is not None:
                frames.append(frame)

    refresh_agent_debug_state()
    render_step_state()
    if not wait_for_playback(
        playback_state,
        on_mode_change=lambda: render_step_state(capture_frame=False),
    ):
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
            if not episode_done and not wait_for_playback(
                playback_state,
                on_mode_change=lambda: render_step_state(capture_frame=False),
            ):
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


def run_trained_model_simulation(playback_mode="step", checkpoint_iteration=None):
    # 1. Ray 및 가상환경 내 초기화
    ray.init(ignore_reinit_error=True)
    register_hemac_rllib_models()

    def env_creator(config):
        # 훈련 때 사용했던 동일한 스펙을 반환해야 합니다. (render_mode 제외)
        train_env_config = {
            "n_observers": 1,
            "observer_speed": 10,
            "n_drones": 3,
            "n_provisioners": 0,
            "known_goals": False,
            "max_cycles": 300,
            "drone_config": {
                "drone_max_speed": 25,
                "drone_max_thrust": 8,
                "drones_starting_pos": DRONE_START_POSITIONS,
            },
            "min_obstacles": 9,
            "max_obstacles": 9,
            "obstacle_min_speed": 3,
            "obstacle_max_speed": 7,
            "n_static_obstacles": 3,
            "poi_config": [GOAL_CONFIG],
            "log_step_rewards": True
        }
        env = HeMAC_v0.env(**train_env_config)
        return PettingZooEnv(env)

    # 학습 때 사용했던 정확히 그 이름으로 등록합니다.
    register_env("hemac_asymmetric_env", env_creator)

    # 2. 두 정책이 모두 저장된 가장 최근 체크포인트를 로드합니다.
    if checkpoint_iteration is None:
        checkpoint_path = find_latest_checkpoint()
        print(f"가장 최근 체크포인트를 불러오는 중: {checkpoint_path}")
    else:
        checkpoint_path = find_checkpoint_by_iteration(checkpoint_iteration)
        print(f"Iteration {checkpoint_iteration} 체크포인트를 불러오는 중: {checkpoint_path}")
    algo = Algorithm.from_checkpoint(str(checkpoint_path))
    restore_policy_from_checkpoint(algo, "observer_policy", checkpoint_path)
    restore_policy_from_checkpoint(algo, "drone_policy", checkpoint_path)
    print(f"[observer_policy] <- {checkpoint_path}")
    print(f"[drone_policy] <- {checkpoint_path}")

    # 3. 평가용 비대칭 환경 구성 (학습 때 사용한 스펙과 완벽히 동일해야 합니다)
    env_config = {
        # 유인기 1대 (느린 속도)
        "n_observers": 1,
        "observer_speed": 10,

        # 무인기 3대 (빠른 속도)
        "n_drones": 3,
        "n_provisioners": 0,
        "known_goals": False,
        "max_cycles": 300,
        "drone_config": {
            "drone_max_speed": 25,
            "drone_max_thrust": 8,
            "drones_starting_pos": DRONE_START_POSITIONS,
        },
        
        # 맵 및 목적지 설정
        "min_obstacles": 4,
        "max_obstacles": 5,
        "obstacle_min_speed": 2,
        "obstacle_max_speed": 3,
        "n_static_obstacles": 2,
        "poi_config": [GOAL_CONFIG],
        "log_step_rewards": True,

        # [핵심] 화면 시각화 활성화
        "render_mode": "human" 
    }

    # 환경 생성
    env = HeMAC_v0.env(**env_config)
    observation_window = ObservationDebugWindow()
    playback_state = {"mode": playback_mode}
    print(f"시뮬레이션을 시작합니다. 재생 모드: {playback_state['mode']} (F1로 전환)")

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
            playback_state=playback_state,
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
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=None,
        metavar="ITERATION",
        help="Load checkpoint_ITERATION. If omitted, load the newest checkpoint.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_trained_model_simulation(
        playback_mode=args.playback,
        checkpoint_iteration=args.checkpoint,
    )
