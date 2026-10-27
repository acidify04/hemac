import argparse
import copy
import os
from collections.abc import Mapping
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

from ray.rllib.policy.policy import Policy
from ray.rllib.utils.spaces.space_utils import unsquash_action
import numpy as np
from hemac import HeMAC_v0
from hemac.environment.world import World
from hemac.rllib_policy import register_hemac_rllib_models
import time
import pygame
from PIL import Image
from packaging.version import Version

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
    [130.0, 830.0, 5.0],
    [170.0, 870.0, 5.0],
    [170.0, 830.0, 5.0],
    # [150.0, 830.0, 5.0],
]

GOAL_CONFIG = {
    "speed": 0,
    "spawn_mode": "random",
    "boundary_margin": 60,
    "spawn_quadrant": ["bottom_right", "bottom_left", "top_right"],
}

TRAIN_ROOT = PROJECT_ROOT / "src/train"
CHECKPOINT_ROOTS = (
    PROJECT_ROOT / "hetero_drone_checkpoints",
    *(sorted(TRAIN_ROOT.glob("*checkpoints*")) if TRAIN_ROOT.is_dir() else ()),
)

NUM_EVAL_SEEDS = 10
DEFAULT_EVAL_MAX_CYCLES = 300
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

SENSOR_VARIATIONS = {
    "v1": {
        "name": "dual_omni_balanced",
        "description": "A: 360deg/125, B: 360deg/135",
        "A": {"range": 125.0, "model": "RoundCamera"},
        "B": {"range": 135.0, "model": "RoundCamera"},
    },
    "v2": {
        "name": "wide_b_scout",
        "description": "A: 360deg/125, B: forward 90deg/140",
        "A": {"range": 125.0, "model": "RoundCamera"},
        "B": {"range": 140.0, "model": "ForwardFacingCamera", "fov_deg": 90.0},
    },
    "v3": {
        "name": "dual_omni_wide_a",
        "description": "A: 360deg/140, B: 360deg/125",
        "A": {"range": 140.0, "model": "RoundCamera"},
        "B": {"range": 125.0, "model": "RoundCamera"},
    },
    "v4": {
        "name": "b_omni_a_forward",
        "description": "A: forward 180deg/130, B: 360deg/145",
        "A": {"range": 130.0, "model": "ForwardFacingCamera", "fov_deg": 180.0},
        "B": {"range": 145.0, "model": "RoundCamera"},
    },
    "v5": {
        "name": "complementary_sectors",
        "description": "A: forward 200deg/125, B: forward 75deg/140",
        "A": {"range": 125.0, "model": "ForwardFacingCamera", "fov_deg": 200.0},
        "B": {"range": 140.0, "model": "ForwardFacingCamera", "fov_deg": 75.0},
    },
}


class _LegacyPackagingVersion:
    """Temporary target for Version objects pickled by older packaging releases."""

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __setstate__(self, state):
        self.state = state


class _CheckpointUnpickler(pickle.Unpickler):
    """Unpickle RLlib state across the packaging.Version state-format change."""

    def find_class(self, module, name):
        if module == "packaging.version" and name == "Version":
            return _LegacyPackagingVersion
        return super().find_class(module, name)


def _restore_legacy_version(value):
    if not isinstance(value, _LegacyPackagingVersion):
        return value

    state = value.state
    if not isinstance(state, tuple) or len(state) < 2:
        raise ValueError(f"Unsupported legacy packaging.Version state: {state!r}")

    epoch, release = state[:2]
    release_text = ".".join(str(component) for component in release)
    version_text = f"{epoch}!{release_text}" if epoch else release_text
    return Version(version_text)


def _load_checkpoint_pickle(path):
    """Load one RLlib pickle, including checkpoints made with old packaging."""
    path = Path(path)
    try:
        with path.open("rb") as file_obj:
            return pickle.load(file_obj)
    except pickle.UnpicklingError as error:
        if "state is not a dictionary" not in str(error):
            raise

    with path.open("rb") as file_obj:
        state = _CheckpointUnpickler(file_obj).load()
    if isinstance(state, dict) and "checkpoint_version" in state:
        state["checkpoint_version"] = _restore_legacy_version(
            state["checkpoint_version"]
        )
    return state


def checkpoint_policy_ids(checkpoint_dir):
    """Return policy IDs that have a complete standalone policy state."""
    policies_dir = Path(checkpoint_dir) / "policies"
    if not policies_dir.is_dir():
        return set()
    return {
        policy_dir.name
        for policy_dir in policies_dir.iterdir()
        if policy_dir.is_dir() and (policy_dir / "policy_state.pkl").is_file()
    }


def _has_supported_drone_policy(policy_ids):
    return "drone_policy" in policy_ids or {
        "drone_a_policy",
        "drone_b_policy",
    }.issubset(policy_ids)


def find_complete_checkpoints(required_policy_ids=None):
    """Return complete legacy or type-specific drone checkpoints."""
    checkpoints = []
    for root in CHECKPOINT_ROOTS:
        if not root.exists():
            continue
        for state_path in root.rglob("algorithm_state.pkl"):
            checkpoint_dir = state_path.parent
            if not checkpoint_dir.name.startswith("checkpoint_"):
                continue
            policy_ids = checkpoint_policy_ids(checkpoint_dir)
            has_required_policies = (
                _has_supported_drone_policy(policy_ids)
                if required_policy_ids is None
                else set(required_policy_ids).issubset(policy_ids)
            )
            if has_required_policies:
                checkpoints.append((state_path.stat().st_mtime_ns, checkpoint_dir))

    if not checkpoints:
        roots = ", ".join(str(root) for root in CHECKPOINT_ROOTS)
        raise FileNotFoundError(
            "No complete legacy drone_policy or heterogeneous "
            f"drone_a_policy/drone_b_policy checkpoint found under: {roots}"
        )

    return checkpoints


def find_latest_checkpoint(required_policy_ids=None):
    """Return the newest complete checkpoint found below the checkpoint roots."""
    checkpoints = find_complete_checkpoints(required_policy_ids)

    return max(checkpoints, key=lambda item: (item[0], str(item[1])))[1].resolve()


def find_checkpoint_by_iteration(
    iteration,
    required_policy_ids=None,
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


def resolve_checkpoint(checkpoint):
    """Resolve an iteration number or an explicit Algorithm checkpoint path."""
    if checkpoint is None:
        return find_latest_checkpoint()

    candidate = Path(str(checkpoint)).expanduser()
    if candidate.exists():
        candidate = candidate.resolve()
        if candidate.parent.name == "policies":
            candidate = candidate.parent.parent
        if not (candidate / "algorithm_state.pkl").is_file():
            raise FileNotFoundError(
                f"Checkpoint directory has no algorithm_state.pkl: {candidate}"
            )
        policy_ids = checkpoint_policy_ids(candidate)
        if not _has_supported_drone_policy(policy_ids):
            raise FileNotFoundError(
                "Checkpoint has neither a shared drone_policy nor both "
                f"drone_a_policy/drone_b_policy: {candidate}"
            )
        return candidate

    try:
        return find_checkpoint_by_iteration(int(str(checkpoint)))
    except ValueError as error:
        raise ValueError(
            "--checkpoint must be an iteration number or checkpoint directory path."
        ) from error


def _uses_legacy_four_conv_encoder(weights):
    keys = tuple(weights)
    has_old_final_conv = any(
        key.endswith(("map_encoder.7.weight", "global_map_encoder.7.weight"))
        for key in keys
    )
    has_current_final_conv = any(
        key.endswith(("map_encoder.9.weight", "global_map_encoder.9.weight"))
        for key in keys
    )
    return has_old_final_conv and not has_current_final_conv


def _policy_original_space(policy):
    return getattr(policy.observation_space, "original_space", policy.observation_space)


def adapt_observation_for_policy(observation, policy):
    """Convert current observations to the schema stored in a checkpoint."""
    expected_space = _policy_original_space(policy)
    expected_spaces = getattr(expected_space, "spaces", None)
    if not isinstance(observation, dict) or expected_spaces is None:
        return observation

    expected_keys = set(expected_spaces)
    if "relative_map" in expected_keys and "relative_map" not in observation:
        global_map = np.asarray(observation["global_map"], dtype=np.float32)
        channel_count = int(expected_spaces["relative_map"].shape[-1])
        observation = dict(observation)
        observation["relative_map"] = global_map[:, :, :channel_count]

    missing = expected_keys.difference(observation)
    if missing:
        raise ValueError(
            f"Checkpoint observation keys are missing from the environment: {sorted(missing)}"
        )

    observation = dict(observation)
    for map_key in ("global_map", "local_map"):
        if map_key not in expected_spaces or map_key not in observation:
            continue
        current_map = np.asarray(observation[map_key])
        expected_shape = expected_spaces[map_key].shape
        if (
            current_map.ndim == 3
            and len(expected_shape) == 3
            and current_map.shape[-1] == 7
            and expected_shape[-1] == 6
        ):
            observation[map_key] = np.concatenate(
                (current_map[:, :, :5], current_map[:, :, 6:]), axis=-1
            )
    adapted = {
        key: np.asarray(observation[key], dtype=expected_spaces[key].dtype)
        for key in expected_spaces
    }
    mismatches = {
        key: (adapted[key].shape, expected_spaces[key].shape)
        for key in expected_spaces
        if adapted[key].shape != expected_spaces[key].shape
    }
    if mismatches:
        raise ValueError(f"Checkpoint observation shape mismatch: {mismatches}")
    return adapted


class DronePolicyRunner:
    """Minimal Algorithm-compatible wrapper around restored policy states."""

    def __init__(self, policies, env_config, n_drones=None, n_goals=1):
        self.policies = dict(policies)
        self.env_config = copy.deepcopy(env_config)
        if n_drones is None:
            n_drones = len(DRONE_START_POSITIONS)
        drone_policy = next(
            policy
            for policy_id, policy in self.policies.items()
            if policy_id.startswith("drone_")
        )
        original_space = _policy_original_space(drone_policy)
        spaces = getattr(original_space, "spaces", {})
        central_vector_space = spaces.get("central_vector")
        if central_vector_space is None:
            self.required_observer_count = 0
        else:
            entity_count = int(central_vector_space.shape[0]) // 2
            fixed_entity_count = max(int(n_drones) - 1, 0) + max(
                int(n_goals), 0
            )
            self.required_observer_count = max(
                entity_count - fixed_entity_count,
                0,
            )

    def policy_id_for_agent(self, agent_id):
        if str(agent_id).startswith("observer_"):
            return "observer_policy"
        if not str(agent_id).startswith("drone_"):
            return None

        if "drone_policy" in self.policies:
            return "drone_policy"

        try:
            drone_index = int(str(agent_id).rsplit("_", 1)[1])
        except (IndexError, ValueError) as error:
            raise ValueError(f"Cannot determine drone index from {agent_id!r}.") from error
        drone_config = self.env_config.get("drone_config", {}) or {}
        drone_types = list(drone_config.get("drone_types", []))
        if not 0 <= drone_index < len(drone_types):
            raise ValueError(
                f"No drone type configured for {agent_id!r}; types={drone_types}."
            )
        policy_id = f"drone_{str(drone_types[drone_index]).lower()}_policy"
        if policy_id not in self.policies:
            raise ValueError(
                f"Checkpoint has no {policy_id!r}; available={sorted(self.policies)}."
            )
        return policy_id

    def compute_single_action(
        self,
        observation,
        policy_id=None,
        agent_id=None,
        explore=False,
    ):
        if policy_id is None and agent_id is not None:
            policy_id = self.policy_id_for_agent(agent_id)
        policy = self.policies.get(policy_id)
        if policy is None:
            if policy_id == "observer_policy":
                return np.zeros((3,), dtype=np.float32)
            raise ValueError(
                f"Unsupported policy ID {policy_id!r}; "
                f"available={sorted(self.policies)}."
            )

        action, _, _ = policy.compute_single_action(
            adapt_observation_for_policy(observation, policy),
            explore=explore,
        )
        if policy.config.get("normalize_actions", False):
            action = unsquash_action(action, policy.action_space)
        return action


def _load_policy_from_checkpoint(checkpoint_dir, policy_id):
    """Restore one policy without rebuilding training-time Ray workers."""
    register_hemac_rllib_models()
    policy_state_path = (
        Path(checkpoint_dir) / "policies" / policy_id / "policy_state.pkl"
    )
    state = _load_checkpoint_pickle(policy_state_path)

    weights = state.get("weights") or {}
    encoder_variant = "current"
    if _uses_legacy_four_conv_encoder(weights):
        model_config = state["policy_spec"]["config"]["model"]
        model_config.setdefault("custom_model_config", {})[
            "encoder_variant"
        ] = "legacy_4conv"
        encoder_variant = "legacy_4conv"

    state.pop("_optimizer_variables", None)
    policy = Policy.from_state(state)
    print(
        f"[{policy_id}] <- {policy_state_path} "
        f"(encoder={encoder_variant})"
    )
    return policy


def load_drone_policy(checkpoint_dir, env_config=None):
    """Restore shared or heterogeneous drone policies from one checkpoint."""
    if env_config is None:
        env_config = load_checkpoint_env_config(checkpoint_dir)
    available_policy_ids = checkpoint_policy_ids(checkpoint_dir)
    if "drone_policy" in available_policy_ids:
        policy_ids = ["drone_policy"]
    else:
        policy_ids = ["drone_a_policy", "drone_b_policy"]
    if "observer_policy" in available_policy_ids:
        policy_ids.append("observer_policy")
    policies = {
        policy_id: _load_policy_from_checkpoint(checkpoint_dir, policy_id)
        for policy_id in policy_ids
    }
    if "observer_policy" not in policies and int(env_config.get("n_observers", 0)) > 0:
        print("[observer_policy] not found; observers will use zero actions.")
    n_drones = int(env_config.get("n_drones", len(DRONE_START_POSITIONS)))
    n_observers = int(env_config.get("n_observers", 0))
    n_goals = (
        n_observers
        if n_observers > 1
        else len(env_config.get("poi_config") or ())
    )
    return DronePolicyRunner(
        policies,
        env_config,
        n_drones=n_drones,
        n_goals=n_goals,
    )


def load_checkpoint_env_config(checkpoint_dir):
    """Return the environment configuration saved with an Algorithm checkpoint."""
    state_path = Path(checkpoint_dir) / "algorithm_state.pkl"
    checkpoint_state = _load_checkpoint_pickle(state_path)

    config = checkpoint_state.get("config") or {}
    if isinstance(config, Mapping):
        env_config = config.get("env_config")
    else:
        env_config = getattr(config, "env_config", None)
    if not isinstance(env_config, Mapping):
        raise ValueError(f"Checkpoint has no valid env_config: {state_path}")

    return copy.deepcopy(dict(env_config))


def load_current_drone_env_config():
    """Build the current drone-only environment without duplicating train config."""
    from train.train import build_env_config

    return build_env_config(drone_only=True)


def apply_sensor_variation(env_config, variation):
    """Apply a visualization-only A/B sensor preset without changing spaces."""
    config = copy.deepcopy(env_config)
    if variation == "original":
        return config, "original checkpoint/current sensor configuration"

    preset = SENSOR_VARIATIONS[variation]
    drone_config = copy.deepcopy(config.get("drone_config", {}) or {})
    type_configs = copy.deepcopy(drone_config.get("drone_type_configs", {}) or {})
    obstacle_max_speed = max(float(config.get("obstacle_max_speed", 0.0)), 0.0)

    for drone_type in ("A", "B"):
        sensor_spec = preset[drone_type]
        config_key = (
            drone_type
            if drone_type in type_configs
            else drone_type.lower()
            if drone_type.lower() in type_configs
            else drone_type
        )
        role_config = copy.deepcopy(type_configs.get(config_key, {}) or {})
        sensing_range = float(sensor_spec["range"])
        max_speed = float(
            role_config.get("max_speed", drone_config.get("drone_max_speed", 0.0))
        )
        safety_buffer = sensing_range - float(World.OBSTACLE_WARNING_RADIUS)
        minimum_buffer = max_speed + obstacle_max_speed
        if safety_buffer <= minimum_buffer:
            raise ValueError(
                f"Unsafe {variation}/{drone_type} sensor: range={sensing_range}, "
                f"warning_radius={World.OBSTACLE_WARNING_RADIUS}, "
                f"max_closing_step={minimum_buffer}."
            )

        sensor_params = {"sensing_range": sensing_range}
        if sensor_spec["model"] == "ForwardFacingCamera":
            sensor_params["hfov"] = math.radians(float(sensor_spec["fov_deg"]))
        role_config["sensing_range"] = sensing_range
        role_config["sensor"] = {
            "model": sensor_spec["model"],
            "params": sensor_params,
        }
        type_configs[config_key] = role_config

    drone_config["drone_type_configs"] = type_configs
    config["drone_config"] = drone_config
    return config, f"{variation}/{preset['name']}: {preset['description']}"


def print_sensor_variations():
    """Print visualization sensor presets and their safety interpretation."""
    print("original: use the environment sensor configuration without overrides")
    for variation, preset in SENSOR_VARIATIONS.items():
        print(f"{variation}: {preset['name']} | {preset['description']}")
    print(
        "All presets keep range 125-145 and leave at least 60 units between "
        "detection range and the obstacle warning-zone boundary."
    )


def load_policy_weights_from_checkpoint(checkpoint_dir, policy_id):
    """Load one policy's weight dict from an RLlib checkpoint directory."""
    policy_state_path = Path(checkpoint_dir) / "policies" / policy_id / "policy_state.pkl"
    if not policy_state_path.is_file():
        raise FileNotFoundError(
            f"Policy checkpoint not found for {policy_id}: {policy_state_path}"
        )

    policy_state = _load_checkpoint_pickle(policy_state_path)

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


def save_gif(frames, eval_seed, filename_prefix="example"):
    """Save rollout frames as a GIF under visualization/."""
    if not frames:
        return None

    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gif_path = VISUALIZATION_DIR / (
        f"{filename_prefix}_seed_{eval_seed}_{timestamp}.gif"
    )
    pil_frames = [
        frame.copy()
        if isinstance(frame, Image.Image)
        else Image.fromarray(frame.astype("uint8"))
        for frame in frames
    ]
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=80,
        loop=0,
    )
    return gif_path


def prepare_comparison_gif_frame(frame, max_width=640):
    """Downsample and palette-compress a frame for a multi-rollout GIF."""
    image = Image.fromarray(frame.astype("uint8"))
    max_width = max(int(max_width), 1)
    if image.width > max_width:
        ratio = max_width / image.width
        target_size = (max_width, max(1, round(image.height * ratio)))
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        image = image.resize(target_size, resampling)
    adaptive_palette = getattr(getattr(Image, "Palette", Image), "ADAPTIVE", 1)
    return image.convert("P", palette=adaptive_palette, colors=128)


def draw_sensor_variation_banner(label):
    """Identify the active sensor preset in the recorded main window."""
    if not label:
        return
    surface = pygame.display.get_surface()
    if surface is None:
        return
    font = pygame.font.SysFont("Trebuchet MS", 22, bold=True)
    text_surface = font.render(label, True, (255, 255, 255))
    padding_x = 14
    padding_y = 8
    banner = pygame.Surface(
        (
            text_surface.get_width() + padding_x * 2,
            text_surface.get_height() + padding_y * 2,
        ),
        pygame.SRCALPHA,
    )
    banner.fill((5, 10, 14, 220))
    banner.blit(text_surface, (padding_x, padding_y))
    surface.blit(banner, (12, 12))
    pygame.display.flip()


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
        labels = [
            "coverage",
            "boundary",
            "obstacle",
            "warning",
            "drones",
            "other_observers",
            "goal",
        ]
    else:
        labels = [
            "coverage",
            "boundary",
            "obstacle",
            "warning",
            "other_drones",
            "observer",
            "goal",
        ]

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


def run_single_episode(
    env,
    algo,
    eval_seed,
    playback_state,
    observation_window=None,
    visualization_label=None,
    shared_gif_frames=None,
    gif_frame_stride=1,
    gif_max_width=None,
    save_episode_gif=True,
    gif_filename_prefix="example",
):
    """Run one evaluation episode and return the final info."""
    env.reset(seed=eval_seed)

    print(f"Evaluation seed: {eval_seed}")
    last_info = {}
    total_agent_turns = 0
    frames = []
    rendered_steps = 0
    captured_frames = 0
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
        nonlocal rendered_steps, captured_frames
        env.render()
        draw_sensor_variation_banner(visualization_label)
        draw_observation_overlays(
            agent_debug_state,
            getattr(env, "agent_selection", None),
            playback_state["mode"],
            observation_window=observation_window,
        )
        if capture_frame:
            rendered_steps += 1
            if (rendered_steps - 1) % max(int(gif_frame_stride), 1) != 0:
                return
            frame = capture_pygame_frame()
            if frame is not None:
                if gif_max_width is not None:
                    frame = prepare_comparison_gif_frame(frame, gif_max_width)
                destination = shared_gif_frames if shared_gif_frames is not None else frames
                # Hold the first labeled frame briefly between comparison rollouts.
                if shared_gif_frames is not None and captured_frames == 0:
                    destination.extend(frame.copy() for _ in range(8))
                destination.append(frame)
                captured_frames += 1

    refresh_agent_debug_state()
    render_step_state()
    if not wait_for_playback(
        playback_state,
        on_mode_change=lambda: render_step_state(capture_frame=False),
    ):
        if save_episode_gif:
            gif_path = save_gif(
                frames,
                eval_seed,
                filename_prefix=gif_filename_prefix,
            )
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
            policy_id = algo.policy_id_for_agent(agent)
            if policy_id:
                action = algo.compute_single_action(
                    observation=observation,
                    policy_id=policy_id,
                    agent_id=agent,
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

    if save_episode_gif:
        gif_path = save_gif(
            frames,
            eval_seed,
            filename_prefix=gif_filename_prefix,
        )
        if gif_path is not None:
            print(f"Saved GIF: {gif_path}")

    return last_info


def run_trained_model_simulation(
    playback_mode="step",
    checkpoint_iteration=None,
    environment_mode="auto",
    max_cycles=DEFAULT_EVAL_MAX_CYCLES,
    sensor_variation="original",
    eval_episodes=NUM_EVAL_SEEDS,
    seed_base=None,
):
    # Standalone policy restoration avoids rebuilding training-time Ray workers.
    register_hemac_rllib_models()
    checkpoint_path = resolve_checkpoint(checkpoint_iteration)
    print(f"드론 체크포인트를 불러오는 중: {checkpoint_path}")
    checkpoint_env_config = load_checkpoint_env_config(checkpoint_path)
    policy_ids = checkpoint_policy_ids(checkpoint_path)
    use_current_environment = environment_mode == "current" or (
        environment_mode == "auto"
        and {"drone_a_policy", "drone_b_policy"}.issubset(policy_ids)
    )
    if use_current_environment:
        env_config = load_current_drone_env_config()
        env_config_source = "current train.py drone-only config"
    else:
        env_config = copy.deepcopy(checkpoint_env_config)
        env_config_source = "checkpoint env_config"
    env_config, sensor_description = apply_sensor_variation(
        env_config,
        sensor_variation,
    )
    algo = load_drone_policy(checkpoint_path, env_config)

    env_config["max_cycles"] = max(int(max_cycles), 1)
    env_config["render_mode"] = "human"
    env_config["log_step_rewards"] = True
    configured_observers = int(env_config.get("n_observers", 0))
    if configured_observers != algo.required_observer_count:
        raise ValueError(
            "Checkpoint env_config and drone central_vector disagree on observer "
            f"count: env={configured_observers}, vector={algo.required_observer_count}"
        )
    print(
        f"Environment config: drones={env_config.get('n_drones')}, "
        f"observers={configured_observers}, "
        f"max_cycles={env_config['max_cycles']}, "
        f"policies={sorted(algo.policies)}, source={env_config_source}"
    )
    print(f"Sensor variation: {sensor_description}")
    # 환경 생성
    env = HeMAC_v0.env(**env_config)
    observation_window = ObservationDebugWindow()
    playback_state = {"mode": playback_mode}
    print(f"시뮬레이션을 시작합니다. 재생 모드: {playback_state['mode']} (F1로 전환)")

    eval_episodes = max(int(eval_episodes), 1)
    seed_base = random.randint(0, 9999) if seed_base is None else int(seed_base)
    eval_seeds = [seed_base + offset for offset in range(eval_episodes)]
    print(f"Running {eval_episodes} evaluation seeds: {eval_seeds}")

    results = []
    for idx, eval_seed in enumerate(eval_seeds, start=1):
        print(f"\n=== Evaluation {idx}/{eval_episodes} ===")
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
    print(f"Success: {success_count}/{eval_episodes}")
    print(f"Goal found: {goal_found_count}/{eval_episodes}")
    print(f"Drone crash: {drone_crash_count}/{eval_episodes}")
    print(f"Observer crash: {observer_crash_count}/{eval_episodes}")

    hold_window_open(seconds=5.0)
    observation_window.close()
    env.close()
    # ray.shutdown()


def run_sensor_variation_comparison(
    playback_mode="auto",
    checkpoint_iteration=None,
    environment_mode="auto",
    max_cycles=DEFAULT_EVAL_MAX_CYCLES,
    seed=None,
    gif_frame_stride=3,
    gif_max_width=640,
):
    """Play original and v1-v5 sequentially and save one GIF per preset."""
    register_hemac_rllib_models()
    checkpoint_path = resolve_checkpoint(checkpoint_iteration)
    print(f"드론 체크포인트를 불러오는 중: {checkpoint_path}")
    checkpoint_env_config = load_checkpoint_env_config(checkpoint_path)
    policy_ids = checkpoint_policy_ids(checkpoint_path)
    use_current_environment = environment_mode == "current" or (
        environment_mode == "auto"
        and {"drone_a_policy", "drone_b_policy"}.issubset(policy_ids)
    )
    if use_current_environment:
        base_env_config = load_current_drone_env_config()
        env_config_source = "current train.py drone-only config"
    else:
        base_env_config = copy.deepcopy(checkpoint_env_config)
        env_config_source = "checkpoint env_config"

    algo = load_drone_policy(checkpoint_path, base_env_config)
    comparison_seed = random.randint(0, 9999) if seed is None else int(seed)
    variations = ("original", *SENSOR_VARIATIONS)
    results = []
    playback_state = {"mode": playback_mode}

    print(
        f"Sensor comparison: {' -> '.join(variations)}, seed={comparison_seed}, "
        f"source={env_config_source}"
    )
    for index, variation in enumerate(variations, start=1):
        env_config, sensor_description = apply_sensor_variation(
            base_env_config,
            variation,
        )
        env_config["max_cycles"] = max(int(max_cycles), 1)
        env_config["render_mode"] = "human"
        env_config["log_step_rewards"] = True
        configured_observers = int(env_config.get("n_observers", 0))
        if configured_observers != algo.required_observer_count:
            raise ValueError(
                "Checkpoint env_config and drone central_vector disagree on observer "
                f"count: env={configured_observers}, "
                f"vector={algo.required_observer_count}"
            )

        print(f"\n=== Sensor comparison {index}/{len(variations)} ===")
        print(f"Sensor variation: {sensor_description}")
        env = HeMAC_v0.env(**env_config)
        observation_window = ObservationDebugWindow()
        try:
            last_info = run_single_episode(
                env,
                algo,
                comparison_seed,
                playback_state=playback_state,
                observation_window=observation_window,
                visualization_label=f"{index}/6  {sensor_description}",
                gif_frame_stride=gif_frame_stride,
                gif_max_width=gif_max_width,
                save_episode_gif=True,
                gif_filename_prefix=f"sensor_variation_{variation}",
            )
            results.append((variation, last_info))
            hold_window_open(seconds=0.5)
        finally:
            observation_window.close()
            env.close()

    print("\n=== Sensor comparison summary ===")
    for variation, info in results:
        print(
            f"{variation}: success={info.get('success')}, "
            f"goal_found={info.get('goal_found')}, "
            f"explored_area={info.get('explored_area')}"
        )
    print(f"Saved 6 individual GIFs under: {VISUALIZATION_DIR.resolve()}")

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
        type=str,
        default=None,
        metavar="ITERATION_OR_PATH",
        help=(
            "Load a drone checkpoint by iteration or explicit checkpoint path. "
            "If omitted, load the newest checkpoint containing either the legacy "
            "drone_policy or both drone_a_policy/drone_b_policy."
        ),
    )
    parser.add_argument(
        "--environment",
        choices=("auto", "current", "checkpoint"),
        default="auto",
        help=(
            "Environment config source. auto uses the current train.py drone-only "
            "config for A/B checkpoints and the saved config for legacy checkpoints."
        ),
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=DEFAULT_EVAL_MAX_CYCLES,
        help=(
            "Maximum environment cycles per visualization episode "
            f"(default: {DEFAULT_EVAL_MAX_CYCLES})."
        ),
    )
    parser.add_argument(
        "--sensor-variation",
        choices=("original", *SENSOR_VARIATIONS),
        default="original",
        help=(
            "Visualization-only A/B sensor preset. It preserves observation "
            "tensor shapes so existing heterogeneous checkpoints can be loaded."
        ),
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=NUM_EVAL_SEEDS,
        help=(
            "Number of visualization episodes "
            f"(default: {NUM_EVAL_SEEDS}; use 1 for quick preset comparison)."
        ),
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=None,
        help="First evaluation seed; omit for a random seed base.",
    )
    parser.add_argument(
        "--list-sensor-variations",
        action="store_true",
        help="Print all sensor presets and exit.",
    )
    parser.add_argument(
        "--compare-sensor-variations",
        action="store_true",
        help=(
            "Play original and v1-v5 in order on the same seed and save one "
            "GIF for each preset."
        ),
    )
    parser.add_argument(
        "--gif-frame-stride",
        type=int,
        default=3,
        help="Capture every Nth rendered step in comparison GIF mode (default: 3).",
    )
    parser.add_argument(
        "--gif-max-width",
        type=int,
        default=640,
        help="Maximum comparison GIF width in pixels (default: 640).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.list_sensor_variations:
        print_sensor_variations()
        raise SystemExit(0)
    if args.compare_sensor_variations:
        run_sensor_variation_comparison(
            playback_mode=args.playback,
            checkpoint_iteration=args.checkpoint,
            environment_mode=args.environment,
            max_cycles=args.max_cycles,
            seed=args.seed_base,
            gif_frame_stride=args.gif_frame_stride,
            gif_max_width=args.gif_max_width,
        )
    else:
        run_trained_model_simulation(
            playback_mode=args.playback,
            checkpoint_iteration=args.checkpoint,
            environment_mode=args.environment,
            max_cycles=args.max_cycles,
            sensor_variation=args.sensor_variation,
            eval_episodes=args.eval_episodes,
            seed_base=args.seed_base,
        )
