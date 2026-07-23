import numpy as np
import pygame
from shapely import Polygon

from hemac.environment.poi import PointOfInterest
from hemac.helpers.helper import game_ref_to_world_ref


def test_goal_spawn_respects_curriculum_distance_from_base():
    area = pygame.Rect(0, 0, 1000, 1000)
    search_area = Polygon(((100, 100), (900, 100), (900, 900), (100, 900)))
    spawn_range = {"x_range": (100, 900), "y_range": (100, 900)}
    randomizer = np.random.default_rng(1234)
    goal = PointOfInterest(
        randomizer=randomizer,
        poi_config={
            "speed": 0,
            "spawn_mode": "random",
            "boundary_margin": 50,
            "spawn_quadrant": ["bottom_right", "bottom_left", "top_right"],
        },
        time_factor=1.0,
        area=area,
        spawn_range=spawn_range,
    )
    base_position = game_ref_to_world_ref((150, 150), area)

    for min_distance, max_distance in (
        (350.0, 450.0),
        (400.0, 520.0),
        (475.0, 600.0),
        (550.0, 675.0),
        (625.0, 750.0),
        (700.0, 825.0),
        (775.0, 925.0),
        (850.0, 1000.0),
    ):
        for _ in range(10):
            goal.spawn_poi(
                search_area,
                base_position=base_position,
                min_base_distance=min_distance,
                max_base_distance=max_distance,
            )
            distance = float(
                np.hypot(
                    goal.x - base_position[0],
                    goal.y - base_position[1],
                )
            )
            assert min_distance <= distance <= max_distance
