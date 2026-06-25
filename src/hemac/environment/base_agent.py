"""Base agent module."""
from functools import lru_cache

import numpy as np
import pygame


class BaseAgent(pygame.sprite.Sprite):
    """Base agent class."""

    GRID_RESOLUTION = 20
    EXTRA_OBS_ROWS = 2
    RELATIVE_MAP_SIZE = GRID_RESOLUTION * 2

    def __init__(self):
        """Overwrite base class constructor."""
        super().__init__()
        self.latest_detected = np.empty((0, 2), dtype=np.int32)

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
        grid_x = min(int(clipped_x / world.coverage_cell_width), self.GRID_RESOLUTION - 1)
        grid_y = min(int(clipped_y / world.coverage_cell_height), self.GRID_RESOLUTION - 1)
        return grid_x, grid_y

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
        pad = self.GRID_RESOLUTION
        padded_coverage = np.pad(world.coverage_map, ((pad, pad), (pad, pad)), mode="constant", constant_values=0.0)
        padded_search_mask = np.pad(world.search_mask, ((pad, pad), (pad, pad)), mode="constant", constant_values=0.0)
        explored_obstacle_map = np.where(world.coverage_map > 0.0, world.obstacle_map, 0.0)
        padded_obstacles = np.pad(explored_obstacle_map, ((pad, pad), (pad, pad)), mode="constant", constant_values=0.0)

        start_y = self_grid_y
        start_x = self_grid_x
        end_y = start_y + self.RELATIVE_MAP_SIZE
        end_x = start_x + self.RELATIVE_MAP_SIZE

        relative_map = np.zeros((self.RELATIVE_MAP_SIZE, self.RELATIVE_MAP_SIZE, 3), dtype=np.float32)
        relative_map[:, :, 0] = padded_coverage[start_y:end_y, start_x:end_x]
        relative_map[:, :, 1] = padded_search_mask[start_y:end_y, start_x:end_x]
        relative_map[:, :, 2] = padded_obstacles[start_y:end_y, start_x:end_x]
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
