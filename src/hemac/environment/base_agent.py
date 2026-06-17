"""Base agent module."""
from functools import lru_cache
from math import isqrt

import pygame


class BaseAgent(pygame.sprite.Sprite):
    """Base agent class."""

    def __init__(self):
        """Overwrite base class constructor."""
        super().__init__()

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
        self.detected.update(
            (center_x + dx, center_y + dy)
            for dx, dy in self._detected_offsets(sensing_range)
        )

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
