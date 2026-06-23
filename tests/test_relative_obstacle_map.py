"""Tests for obstacle visibility in relative observations."""

from pathlib import Path
import sys

import numpy as np
import pygame
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hemac.environment.base_agent import BaseAgent
from hemac.environment.world import World
from hemac.helpers.helper import world_ref_to_game_ref


class DummyAgent(BaseAgent):
    """Minimal agent used to exercise relative-map generation."""

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


def test_relative_map_marks_only_explored_obstacles():
    """Obstacle cells should appear in observations only after being explored."""
    pygame.init()
    try:
        world = World(
            game_area=pygame.Rect(0, 0, 1000, 1000),
            geofence_area=[],
            search_area=Polygon(((100, 100), (900, 100), (900, 900), (100, 900))),
            randomizer=np.random.default_rng(3),
        )

        agent = DummyAgent(x=325.0, y=325.0)

        explored_obstacle = pygame.Rect(0, 0, 24, 24)
        explored_obstacle.center = world_ref_to_game_ref((325.0, 325.0), world.area)

        hidden_obstacle = pygame.Rect(0, 0, 24, 24)
        hidden_obstacle.center = world_ref_to_game_ref((375.0, 325.0), world.area)

        world.obstacles = [explored_obstacle, hidden_obstacle]
        world._rebuild_obstacle_map()
        world.register_detected_points({(325, 325)})

        relative_map = agent.build_relative_sector_map(world)

        assert relative_map.shape == (agent.RELATIVE_MAP_SIZE, agent.RELATIVE_MAP_SIZE, 3)
        assert relative_map[agent.GRID_RESOLUTION, agent.GRID_RESOLUTION, 2] == 1.0
        assert relative_map[agent.GRID_RESOLUTION, agent.GRID_RESOLUTION + 1, 2] == 0.0
    finally:
        pygame.quit()
