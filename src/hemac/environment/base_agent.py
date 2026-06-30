"""Base agent module."""
from functools import lru_cache

import numpy as np
import pygame


class BaseAgent(pygame.sprite.Sprite):
    """Base agent class."""

    GRID_RESOLUTION = 20
    EXTRA_OBS_ROWS = 2
    RELATIVE_MAP_SIZE = GRID_RESOLUTION * 2
    GLOBAL_MAP_SIZE = 40
    LOCAL_MAP_SIZE = 20
    LOCAL_REAL_MAP_SIZE = 10
    ACTION_HISTORY_LENGTH = 5
    ACTION_DIM = 3

    def __init__(self):
        """Overwrite base class constructor."""
        super().__init__()
        self.latest_detected = np.empty((0, 2), dtype=np.int32)
        self.action_history = np.zeros(
            (self.ACTION_HISTORY_LENGTH, self.ACTION_DIM),
            dtype=np.float32,
        )

    @staticmethod
    @lru_cache(maxsize=None)
    def _detected_offsets(radius: int) -> np.ndarray:
        """Cache integer offsets inside a circular sensing range."""
        radius = max(0, int(radius))
        coords = np.arange(-radius, radius + 1, dtype=np.int32)
        grid_x, grid_y = np.meshgrid(coords, coords, indexing="xy")
        mask = (grid_x * grid_x) + (grid_y * grid_y) <= radius * radius
        offsets = np.column_stack((grid_x[mask], grid_y[mask])).astype(np.int32, copy=False)
        offsets.setflags(write=False)
        return offsets

    def update_detected_area(self, sensing_range: float) -> None:
        """Mark all integer coordinates inside the sensing range as detected."""
        center = np.array((int(self.x), int(self.y)), dtype=np.int32)
        self.latest_detected = self._detected_offsets(sensing_range) + center

    def _position_to_grid(self, x: float, y: float, world) -> tuple[int, int]:
        """Convert world coordinates to a coverage-grid sector."""
        width = max(world.area.width, 1)
        height = max(world.area.height, 1)
        clipped_x = min(max(int(x), 0), width - 1)
        clipped_y = min(max(int(y), 0), height - 1)
        grid_limit = max(int(world.coverage_grid_size) - 1, 0)
        grid_x = min(int(clipped_x / world.coverage_cell_width), grid_limit)
        grid_y = min(int(clipped_y / world.coverage_cell_height), grid_limit)
        return grid_x, grid_y

    def reset_action_history(self) -> None:
        """Reset the fixed-length action history buffer."""
        self.action_history.fill(0.0)

    def push_action_history(self, action) -> None:
        """Append one action to the fixed-length history buffer."""
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        history_entry = np.zeros((self.ACTION_DIM,), dtype=np.float32)
        copy_len = min(history_entry.size, action_array.size)
        if copy_len > 0:
            history_entry[:copy_len] = action_array[:copy_len]

        self.action_history[:-1] = self.action_history[1:]
        self.action_history[-1] = history_entry

    def build_action_history_vector(self) -> np.ndarray:
        """Return the previous actions as a flat vector."""
        return self.action_history.reshape(-1).astype(np.float32, copy=False)

    def build_entity_channel(self, world, positions) -> np.ndarray:
        """Return an occupancy map for the provided world-space positions."""
        channel = np.zeros(
            (world.coverage_grid_size, world.coverage_grid_size),
            dtype=np.float32,
        )
        if not positions:
            return channel

        positions_array = np.asarray(positions, dtype=np.float32).reshape(-1, 2)
        clipped_x = np.clip(positions_array[:, 0], 0.0, max(world.area.width - 1, 0))
        clipped_y = np.clip(positions_array[:, 1], 0.0, max(world.area.height - 1, 0))
        grid_x = np.minimum((clipped_x / world.coverage_cell_width).astype(np.int32), world.coverage_grid_size - 1)
        grid_y = np.minimum((clipped_y / world.coverage_cell_height).astype(np.int32), world.coverage_grid_size - 1)
        channel[grid_y, grid_x] = 1.0
        return channel

    def build_padded_entity_channel(self, world, positions, pad: int | None = None) -> np.ndarray:
        """Return a padded occupancy map for the provided world-space positions."""
        pad = world.relative_pad if pad is None else int(pad)
        channel = np.zeros(
            (world.coverage_grid_size + 2 * pad, world.coverage_grid_size + 2 * pad),
            dtype=np.float32,
        )
        if not positions:
            return channel

        positions_array = np.asarray(positions, dtype=np.float32).reshape(-1, 2)
        clipped_x = np.clip(positions_array[:, 0], 0.0, max(world.area.width - 1, 0))
        clipped_y = np.clip(positions_array[:, 1], 0.0, max(world.area.height - 1, 0))
        grid_x = np.minimum((clipped_x / world.coverage_cell_width).astype(np.int32), world.coverage_grid_size - 1)
        grid_y = np.minimum((clipped_y / world.coverage_cell_height).astype(np.int32), world.coverage_grid_size - 1)
        channel[grid_y + pad, grid_x + pad] = 1.0
        return channel

    def crop_padded_channel(self, world, padded_channel: np.ndarray, window_size: int) -> np.ndarray:
        """Crop a self-centered window from an already padded 2D map."""
        self_grid_x, self_grid_y = self._position_to_grid(self.x, self.y, world)
        half = window_size // 2
        start_y = self_grid_y + world.relative_pad - half
        start_x = self_grid_x + world.relative_pad - half
        end_y = start_y + window_size
        end_x = start_x + window_size
        return padded_channel[start_y:end_y, start_x:end_x]

    def build_multi_channel_view(
        self,
        world,
        padded_channels: list[np.ndarray],
        window_size: int,
        upsample: int = 1,
    ) -> np.ndarray:
        """Stack centered channel crops into one HWC observation tensor."""
        cropped_channels = [
            self.crop_padded_channel(world, channel, window_size)
            for channel in padded_channels
        ]
        stacked = np.stack(cropped_channels, axis=-1).astype(np.float32, copy=False)
        if upsample > 1:
            stacked = np.repeat(np.repeat(stacked, upsample, axis=0), upsample, axis=1)
        return stacked

    def build_global_map_view(self, world, padded_channels: list[np.ndarray]) -> np.ndarray:
        """Return a 40x40 self-centered global observation view."""
        return self.build_multi_channel_view(world, padded_channels, self.GLOBAL_MAP_SIZE)

    def build_local_map_view(self, world, padded_channels: list[np.ndarray]) -> np.ndarray:
        """Return a zoomed 20x20 local view from a narrower real window."""
        scale = max(self.LOCAL_MAP_SIZE // self.LOCAL_REAL_MAP_SIZE, 1)
        return self.build_multi_channel_view(
            world,
            padded_channels,
            self.LOCAL_REAL_MAP_SIZE,
            upsample=scale,
        )

    def build_sector_features(self, world) -> np.ndarray:
        """Build flattened sector coverage and sector-position features."""
        obs = np.full(self.GRID_RESOLUTION * self.GRID_RESOLUTION + 4, -1.0, dtype=np.float32)
        obs[: self.GRID_RESOLUTION * self.GRID_RESOLUTION] = world.coverage_map.reshape(-1)

        offset = self.GRID_RESOLUTION * self.GRID_RESOLUTION
        self_grid_x, self_grid_y = self._position_to_grid(self.x, self.y, world)
        obs[offset] = self_grid_x
        obs[offset + 1] = self_grid_y

        if world.goal_position is not None:
            goal_grid_x, goal_grid_y = self._position_to_grid(world.goal_position[0], world.goal_position[1], world)
            obs[offset + 2] = goal_grid_x - self_grid_x # goal 상대좌표 적용
            obs[offset + 3] = goal_grid_y - self_grid_y

        return obs

    def build_relative_sector_map(self, world) -> np.ndarray:
        """Build a self-centered 40x40x3 map of coverage, valid-search, and explored obstacles."""
        self_grid_x, self_grid_y = self._position_to_grid(self.x, self.y, world)
        start_y = self_grid_y
        start_x = self_grid_x
        end_y = start_y + self.RELATIVE_MAP_SIZE
        end_x = start_x + self.RELATIVE_MAP_SIZE

        relative_map = np.empty((self.RELATIVE_MAP_SIZE, self.RELATIVE_MAP_SIZE, 3), dtype=np.float32)
        relative_map[:, :, 0] = world.padded_coverage_map[start_y:end_y, start_x:end_x]
        relative_map[:, :, 1] = world.padded_search_mask[start_y:end_y, start_x:end_x]
        relative_map[:, :, 2] = world.padded_explored_obstacle_map[start_y:end_y, start_x:end_x]
        return relative_map

    @staticmethod
    def update_boundary_distance_channels(point, area, sensing_range, distances) -> None:
        """Update per-direction boundary distances for axis-aligned search areas."""
        minx, miny, maxx, maxy = area.bounds
        wall_distances = {
            "right": maxx - point.x,
            "up": maxy - point.y,
            "left": point.x - minx,
            "down": point.y - miny,
        }
        for direction, distance in wall_distances.items():
            if 0.0 <= distance < sensing_range:
                distances[direction] = min(distances[direction], float(distance))

    @staticmethod
    def update_boundary_distance_array(
        point_x: float,
        point_y: float,
        search_bounds: tuple[float, float, float, float],
        sensing_range: float,
        distances: np.ndarray,
    ) -> None:
        """Update cardinal boundary distances in-place for [right, up, left, down]."""
        minx, miny, maxx, maxy = search_bounds
        wall_distances = np.array(
            [maxx - point_x, maxy - point_y, point_x - minx, point_y - miny],
            dtype=np.float32,
        )
        valid = (wall_distances >= 0.0) & (wall_distances < sensing_range)
        distances[valid] = np.minimum(distances[valid], wall_distances[valid])

    @staticmethod
    def obstacle_distance_channels(
        px: float,
        py: float,
        sensing_range: float,
        obstacle_bounds: np.ndarray,
    ) -> np.ndarray:
        """Return nearest obstacle distances for [right, up, left, down] in game coordinates."""
        distances = np.full(4, float(sensing_range), dtype=np.float32)
        if obstacle_bounds.size == 0:
            return distances

        left = obstacle_bounds[:, 0]
        right = obstacle_bounds[:, 1]
        top = obstacle_bounds[:, 2]
        bottom = obstacle_bounds[:, 3]
        closest_x = np.clip(px, left, right)
        closest_y = np.clip(py, top, bottom)
        distance = np.hypot(closest_x - px, closest_y - py)
        within = distance < sensing_range
        if not np.any(within):
            return distances

        closest_x = closest_x[within]
        closest_y = closest_y[within]
        distance = distance[within]

        right_mask = closest_x > px
        if np.any(right_mask):
            distances[0] = float(np.min(distance[right_mask]))
        up_mask = closest_y < py
        if np.any(up_mask):
            distances[1] = float(np.min(distance[up_mask]))
        left_mask = closest_x < px
        if np.any(left_mask):
            distances[2] = float(np.min(distance[left_mask]))
        down_mask = closest_y > py
        if np.any(down_mask):
            distances[3] = float(np.min(distance[down_mask]))

        return distances

    def build_goal_relative_sector(self, world) -> np.ndarray:
        """Build a goal-sector offset relative to the agent sector."""
        goal_relative = np.zeros(2, dtype=np.float32)
        if world.goal_position is None:
            return goal_relative

        self_grid_x, self_grid_y = self._position_to_grid(self.x, self.y, world)
        goal_grid_x, goal_grid_y = self._position_to_grid(world.goal_position[0], world.goal_position[1], world)
        scale = max(self.GRID_RESOLUTION - 1, 1)
        goal_relative[0] = (goal_grid_x - self_grid_x) / scale
        goal_relative[1] = (goal_grid_y - self_grid_y) / scale
        return goal_relative

    def build_relative_agent_positions(self, world, agents) -> np.ndarray:
        """Build normalized relative positions for other observer/drone agents."""
        try:
            minx, miny, maxx, maxy = world.search_area.bounds
            norm = float(np.hypot(maxx - minx, maxy - miny))
        except Exception:
            norm = float(np.hypot(world.area.width, world.area.height))
        if norm <= 0:
            norm = 1.0

        relative_positions = []
        for agent in agents:
            if agent is self or agent.__class__.__name__ not in {"Drone", "Observer"}:
                continue

            dx = float(np.clip((agent.x - self.x) / norm, -1.0, 1.0))
            dy = float(np.clip((agent.y - self.y) / norm, -1.0, 1.0))
            relative_positions.extend((dx, dy))

        return np.array(relative_positions, dtype=np.float32)

    def draw(self, surface):
        """Abstract method to draw the agent. Must be implemented by child classes."""
        raise NotImplementedError("Child classes must implement the draw method.")

    def update(self, *args, **kwargs):
        """Abstract method to update the agent's state. Must be implemented by child classes."""
        raise NotImplementedError("Child classes must implement the update method.")

    def reset(self, *args, **kwargs):
        """Abstract method to reset the agent's state. Must be implemented by child classes."""
        raise NotImplementedError("Child classes must implement the reset method.")

    def observe(self, *args, **kwargs):
        """Abstract method to collect an observation. Must be implemented by child classes."""
        raise NotImplementedError("Child classes must implement the observe method.")
