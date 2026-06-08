"""Helper functions."""
import pygame
from shapely import Point


def world_ref_to_game_ref(world_vec, game_area: pygame.Rect):
    """Convert a 2D vector (pos, vel or accel), from the world referential (ENU) to Pygame display referential."""
    height = game_area.height
    game_ref = [world_vec[0], height - world_vec[1]]
    return game_ref


def game_ref_to_world_ref(game_vec, game_area: pygame.Rect):
    """Convert a 2D vector (pos, vel or accel), from the game referential (computer graphics) to ENU referential."""
    height = game_area.height
    world_vec = [game_vec[0], height - game_vec[1]]
    return world_vec


def sample_point_in_rect(rect, randomizer):
    """Sample a random point inside a rect."""
    x, y, w, h = rect
    return randomizer.integers(x + 15, w - 15), randomizer.integers(y + 15, h - 15)


def sample_point_in_polygon(polygon, randomizer):
    """Sample a random point inside a polygon."""
    min_x, min_y, max_x, max_y = polygon.bounds

    while True:
        random_point = Point(randomizer.uniform(min_x, max_x), randomizer.uniform(min_y, max_y))
        if polygon.contains(random_point):
            return random_point.x, random_point.y
