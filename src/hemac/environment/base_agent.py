"""Base agent module."""
from functools import lru_cache
from math import isqrt

import numpy as np
import pygame


class BaseAgent(pygame.sprite.Sprite):
    """Base agent class."""

    GRID_RESOLUTION = 20
    EXTRA_OBS_ROWS = 2

    def __init__(self):
        """Overwrite base class constructor."""
        super().__init__()
        self.latest_detected = set()

    @staticmethod
    @lru_cache(maxsize=None)
    def _detected_offsets(radius: int) -> tuple[tuple[int, int], ...]:
        """Cache integer offsets inside a circular sensing range."""
        radius = max(0, int(radius))
        radius_sq = radius * radius
        offsets = []
        for dx in range(-radius, radius + 1):
            max_dy = isqrt(radius_sq - dx * dx)
            offsets.extend((dx, dy) for dy in range(-max_dy, max_dy + 1))
        return tuple(offsets)

    def update_detected_area(self, sensing_range: float) -> None:
        """Mark all integer coordinates inside the sensing range as detected."""
        center_x = int(self.x)
        center_y = int(self.y)
        self.latest_detected = {
            (center_x + dx, center_y + dy)
            for dx, dy in self._detected_offsets(sensing_range)
        }
        self.detected.update(self.latest_detected)

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
            obs[offset + 2] = goal_grid_x
            obs[offset + 3] = goal_grid_y

        return obs

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
