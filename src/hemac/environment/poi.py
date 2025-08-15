"""POI class."""

import numpy as np
import pygame
from pymap3d import geodetic2enu
from shapely import Point

from hemac.helpers.helper import game_ref_to_world_ref, sample_point_in_rect, world_ref_to_game_ref


class PointOfInterest:
    """PointOfInterest class."""

    def __init__(
        self,
        randomizer: np.random.Generator,
        poi_config: dict,
        time_factor: float,
        area: pygame.Rect,
        spawn_range: dict,
    ) -> None:
        """Overwrite constructor."""
        self.area = area
        self.spawn_range = spawn_range

        self.config = {
            "dimension": poi_config.get("dimension") if poi_config and poi_config.get("dimension") else [8, 8],
            "speed": poi_config.get("speed") if poi_config and poi_config.get("speed") else 0,
            "variable_speed": poi_config.get("variable_speed")
            if poi_config and poi_config.get("variable_speed") is not None
            else True,
            "spawn_mode": poi_config.get("spawn_mode") if poi_config and poi_config.get("spawn_mode") else "random",
            "starting_pos_coordinates_type": poi_config.get("starting_pos_coordinates_type")
            if poi_config and poi_config.get("starting_pos_coordinates_type")
            else "cardinal",
            "starting_pos": poi_config.get("starting_pos")
            if poi_config and poi_config.get("starting_pos")
            else [300, 300],
            "draw_expected_position": poi_config.get("draw_expected_position")
            if poi_config and poi_config.get("draw_expected_position") is not None
            else False,
            "draw_uncertainty": poi_config.get("draw_uncertainty")
            if poi_config and poi_config.get("draw_uncertainty") is not None
            else False,
            "position_origin": poi_config.get("position_origin")
            if poi_config and poi_config.get("position_origin")
            else {"latitude": 0, "longitude": 0},
            "waypoints_coordinates_type": poi_config.get("waypoints_coordinates_type")
            if poi_config and poi_config.get("waypoints_coordinates_type")
            else "geo",
            "waypoints": poi_config.get("waypoints") if poi_config and poi_config.get("waypoints") else [],
        }
        self.randomizer = randomizer
        self.rect = None
        self.x = None
        self.y = None

        self.detected = False
        self.expected_rect = None
        self.uncertainty_width = None
        self.uncertainty_height = None
        self.uncertainty_angle = None
        self.expected_speed = None  # last_observed speed for position estimation
        self.potential_speed = 0  # maximum potential speed assuming a maximum acceleration for uncertainty tracking
        self.expected_orientation = None

        self.time_factor = time_factor

        self.speed = poi_config.get("speed", 12)
        self.orientation = randomizer.uniform() * np.pi * 2
        self.max_speed = 16
        self.max_acceleration = 0.5
        self.waypoints = self.get_waypoints()

        self.pause_interval = 5
        self.step_counter = 0

    def get_waypoints(self) -> list | None:
        """Get waypoints."""
        if len(self.config.get("waypoints")):
            if self.config.get("waypoints_coordinates_type") == "geo":
                _waypoints = []
                for waypoint in self.config.get("waypoints"):
                    x, y, *_ = list(
                        geodetic2enu(
                            waypoint[0],
                            waypoint[1],
                            0,
                            self.config.get("position_origin", {}).get("latitude"),
                            self.config.get("position_origin", {}).get("longitude"),
                            0,
                        )
                    )
                    wr_gr = world_ref_to_game_ref([x, y], self.area)
                    wp_pos = game_ref_to_world_ref([wr_gr[0], wr_gr[1]], self.area)

                    _waypoints.append([wp_pos[0], wp_pos[1]])
                return _waypoints
            else:
                _waypoints = []
                for waypoint in self.config.get("waypoints"):
                    wp_pos = world_ref_to_game_ref(waypoint, self.area)
                    _waypoints.append([wp_pos[0], wp_pos[1]])
                return _waypoints
        return None

    def spawn_poi(self, area, obstacles=None) -> list:
        """Spawn POI randomly inside patrolling area."""
        pos = [0, 0]
        max_attempts = 1000  # maximum number of attempts to avoid infinite loops
        attempts = 0
        inside_area = False
        no_collision = False

        # min_x, max_x = self.spawn_range["x_range"]
        # min_y, max_y = self.spawn_range["y_range"]

        while not (inside_area and no_collision) and attempts < max_attempts:
            attempts += 1

            if self.config.get("spawn_mode") == "random":
                pos = sample_point_in_rect(self.area, self.randomizer)
                # pos[0] = self.randomizer.integers(min_x, max_x)
                # pos[1] = self.randomizer.integers(min_y, max_y)
            elif self.config.get("spawn_mode") == "fixed" and len(self.config.get("starting_pos", [])):
                if self.config.get("starting_pos_coordinates_type") == "geo":
                    # we convert geo to cardinal position
                    pos[0], pos[1], *_ = list(
                        geodetic2enu(
                            self.config.get("starting_pos")[0],
                            self.config.get("starting_pos")[1],
                            0,
                            self.config.get("position_origin", {}).get("latitude"),
                            self.config.get("position_origin", {}).get("longitude"),
                            0,
                        )
                    )
                else:
                    pos = self.config.get("starting_pos")

            self.rect = pygame.Rect(
                pos[0], pos[1], self.config.get("dimension", [])[0], self.config.get("dimension", [])[1]
            )

            goal_pos_rect = world_ref_to_game_ref(self.rect.center, self.area)
            self.rect.x = goal_pos_rect[0]
            self.rect.y = goal_pos_rect[1]
            goal_pos = game_ref_to_world_ref(self.rect.center, self.area)
            self.x = goal_pos[0]
            self.y = goal_pos[1]

            # Check if POI is in area
            inside_area = area.contains(Point(goal_pos))

            # Check if POI collides with obstacles
            no_collision = not any(self.rect.colliderect(obstacle) for obstacle in (obstacles or []))

            # print(f"inside area: {inside_area}")

        if not (inside_area and no_collision):
            print("POI could not be positioned after several attempts.")

        return pos

    def move(self, obstacles, patrol_area):
        """Move POI."""
        # Small random modification for orientation, but restricted
        proposed_orientation = self.orientation + self.randomizer.normal(scale=0.02)

        # Calculating displacement as a function of speed and orientation
        dx = self.speed * np.cos(proposed_orientation) * self.time_factor
        dy = self.speed * np.sin(proposed_orientation) * self.time_factor
        # self.x += dx
        # self.y += dy
        proposed_x = self.x + dx
        proposed_y = self.y + dy

        proposed_rect = pygame.Rect(0, 0, self.rect.width, self.rect.height)
        proposed_rect.center = world_ref_to_game_ref([proposed_x, proposed_y], self.area)

        collision_detected = any(
            proposed_rect.colliderect(obstacle) for obstacle in obstacles
        ) or not patrol_area.contains(Point((proposed_x, proposed_y)))

        if not collision_detected:
            self.x = proposed_x
            self.y = proposed_y
            self.orientation = proposed_orientation

            # Limit movement within the defined area
            if self.x < self.spawn_range["x_range"][0] or self.x > self.spawn_range["x_range"][1]:
                self.orientation = np.pi - self.orientation
                self.x = np.clip(self.x, self.spawn_range["x_range"][0], self.spawn_range["x_range"][1])

            if self.y < self.spawn_range["y_range"][0] or self.y > self.spawn_range["y_range"][1]:
                self.orientation = -self.orientation
                self.y = np.clip(self.y, self.spawn_range["y_range"][0], self.spawn_range["y_range"][1])

            # Update position in Pygame
            self.rect.center = world_ref_to_game_ref([self.x, self.y], self.area)
        else:
            # If collision, change direction to avid obstacle
            self.orientation += np.pi / 2  # Turn 90° to attempt to bypass collision
            # print("Collision detected, changing direction.")

    def distance_traveled(self):
        """Calculate an estimate of the possible distance travelled by the goal.

        assuming linear movement with constant max acceleration
        the distance is scaled by 0.7, assuming some random movements in other directions. #TODO better model needed.
        """
        # Calculate the time to reach maximum speed
        t_accel = (self.max_speed - self.potential_speed) / self.max_acceleration

        if t_accel > 1:
            # The object does not reach maximum speed within 1 timestep
            distance = self.potential_speed + 0.5 * self.max_acceleration
            self.potential_speed += self.max_acceleration
        else:
            # The object reaches maximum speed within 1 timestep
            d1 = self.potential_speed * t_accel + 0.5 * self.max_acceleration * t_accel**2
            d2 = self.max_speed * (1 - t_accel)
            distance = d1 + d2

        return distance * 0.7

    def draw(self, screen):
        """Draw the POI."""
        pygame.draw.rect(screen, (0, 255, 0), self.rect)
        if self.detected:
            pygame.draw.rect(screen, (128, 255, 255), self.rect)
            if self.config.get("draw_expected_position"):
                # draw expected position
                pygame.draw.rect(screen, (255, 200, 0), self.expected_rect)
                # link to ground truth for visualisation
                pygame.draw.line(screen, (0, 255, 0), self.rect.center, self.expected_rect.center)
            # draw uncertainty circle
            if self.config.get("draw_uncertainty"):
                surface = pygame.Surface((self.uncertainty_width, self.uncertainty_height), pygame.SRCALPHA)
                surface.fill((0, 0, 0, 0))  # Fill the surface with transparent color
                pygame.draw.ellipse(
                    surface,
                    (255, 200, 0, 50),
                    (0, 0, self.uncertainty_width, self.uncertainty_height),
                )
                ellipsis_center = (
                    round(self.expected_rect.centerx),
                    round(self.expected_rect.centery),
                )
                rotated_surface = pygame.transform.rotate(surface, self.uncertainty_angle * 180 / np.pi)
                rotated_rect = rotated_surface.get_rect(center=ellipsis_center)
                screen.blit(rotated_surface, rotated_rect.topleft)

    def reset(self):
        """Reset POI."""
        self.detected = False
        self.expected_rect = None
        self.uncertainty_width = None
        self.uncertainty_height = None
        self.uncertainty_angle = None
        self.expected_speed = None  # last_observed speed for position estimation
        self.potential_speed = 0  # maximum potential speed assuming a maximum acceleration for uncertainty tracking
        self.expected_orientation = None
