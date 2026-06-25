"""Tests for vectorized detected-area updates."""

from pathlib import Path
import sys

import numpy as np
import pygame
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hemac.environment.base_agent import BaseAgent
from hemac.environment.world import World


class DummyAgent(BaseAgent):
    """Minimal agent used to exercise detected-area updates."""

    def __init__(self, x: float, y: float):
        super().__init__()
        self.x = x
        self.y = y
        self.detected = set()

    def draw(self, surface):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self, *args, **kwargs):
        raise NotImplementedError

    def observe(self, *args, **kwargs):
        raise NotImplementedError


def test_update_detected_area_matches_integer_circle():
    """Detected coordinates should cover the integer points inside the radius."""
    agent = DummyAgent(x=10.4, y=20.6)

    agent.update_detected_area(2.9)

    detected_points = {tuple(point) for point in agent.latest_detected.tolist()}
    expected_points = {
        (10 + dx, 20 + dy)
        for dx in range(-2, 3)
        for dy in range(-2, 3)
        if dx * dx + dy * dy <= 4
    }
    assert detected_points == expected_points


def test_world_register_detected_points_accepts_numpy_arrays():
    """World coverage updates should work from vectorized point batches."""
    pygame.init()
    try:
        world = World(
            game_area=pygame.Rect(0, 0, 1000, 1000),
            geofence_area=[],
            search_area=Polygon(((100, 100), (900, 100), (900, 900), (100, 900))),
            randomizer=np.random.default_rng(7),
        )

        points = np.array(
            [
                [325, 325],
                [325, 325],
                [375, 325],
                [-1, 50],
            ],
            dtype=np.int32,
        )

        new_points = world.register_detected_points(points, return_new_points=True)

        assert {tuple(point) for point in new_points.tolist()} == {(325, 325), (375, 325)}
        assert len(world.detected) == 2
    finally:
        pygame.quit()
