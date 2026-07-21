"""Tests for obstacle spawning rules around the base."""

from pathlib import Path
import sys
from types import SimpleNamespace

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


def test_generated_obstacles_do_not_overlap_blocked_rects():
    """Obstacles should avoid explicitly blocked rectangles such as goal positions."""
    pygame.init()
    try:
        world = World(
            game_area=pygame.Rect(0, 0, 1000, 1000),
            geofence_area=[],
            search_area=Polygon(((100, 100), (900, 100), (900, 900), (100, 900))),
            randomizer=np.random.default_rng(13),
        )
        blocked_goal_rect = pygame.Rect(0, 0, 40, 40)
        blocked_goal_rect.center = (500, 500)

        world.generate_obstacles(10, avoid_rects=[blocked_goal_rect])

        assert world.obstacles
        for obstacle in world.obstacles:
            assert not obstacle.colliderect(blocked_goal_rect)
    finally:
        pygame.quit()


def test_obstacles_avoid_base_quadrant_and_goal_warning_zone_while_moving():
    """Moving and static obstacle warning zones must remain in allowed space."""
    pygame.init()
    try:
        world = World(
            game_area=pygame.Rect(0, 0, 1000, 1000),
            geofence_area=[],
            search_area=Polygon(((100, 100), (900, 100), (900, 900), (100, 900))),
            randomizer=np.random.default_rng(19),
            obstacle_min_speed=3,
            obstacle_max_speed=7,
        )
        goal_rect = pygame.Rect(0, 0, 8, 8)
        goal_rect.center = (700, 700)
        goal = SimpleNamespace(x=700.0, y=300.0, rect=goal_rect)
        world.reset([goal])
        world.generate_obstacles(4, avoid_rects=[goal_rect], n_static_obstacles=1)

        assert len(world.obstacles) == 5
        assert np.count_nonzero(world.obstacle_is_static) == 1
        static_index = int(np.flatnonzero(world.obstacle_is_static)[0])
        static_position = world.obstacles[static_index].topleft
        forbidden_quadrant = world._second_quadrant_game_rect()

        for _ in range(200):
            world.update(world.area)
            assert world.obstacles[static_index].topleft == static_position
            for obstacle in world.obstacles:
                assert not world._warning_zone_overlaps_rect(obstacle, forbidden_quadrant)
                assert not world._warning_zone_overlaps_rect(obstacle, goal_rect)
    finally:
        pygame.quit()


def test_static_obstacle_uses_shared_obstacle_observation_channel():
    """A detected static obstacle should populate the regular obstacle belief map."""
    pygame.init()
    try:
        world = World(
            game_area=pygame.Rect(0, 0, 1000, 1000),
            geofence_area=[],
            search_area=Polygon(((100, 100), (900, 100), (900, 900), (100, 900))),
            randomizer=np.random.default_rng(23),
        )
        world.base.center = (150, 150)
        world.generate_obstacles(0, n_static_obstacles=1)

        assert len(world.obstacles) == 1
        assert world.obstacle_is_static.tolist() == [True]
        obstacle = world.obstacles[0]
        sensor = SimpleNamespace(
            pos=obstacle.center,
            sensing_range=10.0,
            hfov=None,
            theta=None,
            is_point_detected=lambda point: True,
        )
        agent = SimpleNamespace(sensor=sensor)

        assert world.update_obstacle_observations_for_agent(agent)
        assert world.actual_obstacle_confidences[0] == 1.0
        assert world._rect_key(obstacle) in world.observed_obstacle_confidences
        assert np.any(world.explored_obstacle_map > 0.0)
    finally:
        pygame.quit()


def test_moving_obstacle_chases_nearest_sensed_agent_at_lower_speed():
    """A moving obstacle should chase inside range without matching agent speed."""
    pygame.init()
    try:
        world = World(
            game_area=pygame.Rect(0, 0, 1000, 1000),
            geofence_area=[],
            search_area=Polygon(((100, 100), (900, 100), (900, 900), (100, 900))),
            randomizer=np.random.default_rng(29),
            obstacle_min_speed=7,
            obstacle_max_speed=7,
        )
        obstacle = pygame.Rect(600, 500, 1, 1)
        world.obstacles = [obstacle]
        world.obstacle_is_static = np.array([False])
        world._rebuild_obstacle_map()

        Observer = type("Observer", (), {})
        observer = Observer()
        observer.x = 650.0
        observer.y = 499.0
        observer.max_speed = 5.0
        distance_before = np.hypot(obstacle.x - observer.x, (world.area.height - obstacle.bottom) - observer.y)

        world.update(world.area, [observer])

        distance_after = np.hypot(obstacle.x - observer.x, (world.area.height - obstacle.bottom) - observer.y)
        assert distance_after < distance_before
        assert 0 < world.obstacle_move_speeds[0] < observer.max_speed
    finally:
        pygame.quit()


def test_goal_avoidance_overrides_chase_until_release_distance():
    """Goal avoidance should override a nearby chase target until distance 200."""
    pygame.init()
    try:
        world = World(
            game_area=pygame.Rect(0, 0, 1000, 1000),
            geofence_area=[],
            search_area=Polygon(((100, 100), (900, 100), (900, 900), (100, 900))),
            randomizer=np.random.default_rng(31),
            obstacle_min_speed=3,
            obstacle_max_speed=3,
        )
        goal_rect = pygame.Rect(0, 0, 8, 8)
        goal_rect.center = (700, 500)
        goal = SimpleNamespace(x=700.0, y=500.0, rect=goal_rect)
        world.reset([goal])
        obstacle = pygame.Rect(610, 500, 1, 1)
        world.obstacles = [obstacle]
        world.obstacle_is_static = np.array([False])
        world._rebuild_obstacle_map()

        Drone = type("Drone", (), {})
        drone = Drone()
        drone.x = 650.0
        drone.y = 499.0
        drone.max_speed = 25.0

        world.update(world.area, [drone])
        assert obstacle.x < 610
        assert world.obstacle_avoiding_goal[0]

        reached_release_distance = False
        remained_active_past_trigger = False
        for _ in range(100):
            center = np.asarray(world._warning_center_for_rect(obstacle))
            goal_distance = float(np.linalg.norm(center - np.asarray(goal_rect.center)))
            if 120.0 < goal_distance < 200.0:
                remained_active_past_trigger = remained_active_past_trigger or bool(
                    world.obstacle_avoiding_goal[0]
                )
            world.update(world.area, [drone])
            center = np.asarray(world._warning_center_for_rect(obstacle))
            goal_distance = float(np.linalg.norm(center - np.asarray(goal_rect.center)))
            if not world.obstacle_avoiding_goal[0]:
                reached_release_distance = goal_distance >= 200.0
                break

        assert remained_active_past_trigger
        assert reached_release_distance
    finally:
        pygame.quit()


def test_obstacle_chases_drone_diagonally_then_returns_to_pre_chase_position():
    """A lost drone should trigger return, followed by normal random movement."""
    pygame.init()
    try:
        world = World(
            game_area=pygame.Rect(0, 0, 1000, 1000),
            geofence_area=[],
            search_area=Polygon(((100, 100), (900, 100), (900, 900), (100, 900))),
            randomizer=np.random.default_rng(41),
            obstacle_min_speed=3,
            obstacle_max_speed=3,
        )
        obstacle = pygame.Rect(600, 500, 1, 1)
        world.obstacles = [obstacle]
        world.obstacle_is_static = np.array([False])
        world._rebuild_obstacle_map()
        pre_chase_position = obstacle.topleft
        pre_chase_center = np.asarray(world._warning_center_for_rect(obstacle))

        Drone = type("Drone", (), {})
        drone = Drone()
        drone.x = 650.0
        drone.y = 450.0
        drone.max_speed = 25.0

        world.update(world.area, [drone])

        assert obstacle.x > pre_chase_position[0]
        assert obstacle.y > pre_chase_position[1]
        np.testing.assert_array_equal(world.obstacle_chase_origins_game[0], pre_chase_center)

        drone.x = 850.0
        drone.y = 150.0
        returned = False
        for _ in range(100):
            world.update(world.area, [drone])
            if not np.all(np.isfinite(world.obstacle_chase_origins_game[0])):
                returned = obstacle.topleft == pre_chase_position
                break

        assert returned
        world.update(world.area, [drone])
        assert obstacle.topleft != pre_chase_position
    finally:
        pygame.quit()
