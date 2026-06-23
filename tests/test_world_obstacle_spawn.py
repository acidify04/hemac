"""Tests for obstacle spawning rules around the base."""

from pathlib import Path
import sys

import numpy as np
import pygame
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hemac.environment.world import World


def test_generated_obstacles_stay_away_from_base():
    """Obstacles should spawn more than 150 units away from the base."""
    pygame.init()
    try:
        world = World(
            game_area=pygame.Rect(0, 0, 1000, 1000),
            geofence_area=[],
            search_area=Polygon(((100, 100), (900, 100), (900, 900), (100, 900))),
            randomizer=np.random.default_rng(7),
        )
        world.base.center = (150, 150)
        world.generate_obstacles(5)

        assert world.obstacles
        for obstacle in world.obstacles:
            assert world._rect_distance(obstacle, world.base) > world.BASE_OBSTACLE_CLEARANCE
    finally:
        pygame.quit()
