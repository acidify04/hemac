"""Observer module."""

import os

import pygame
from .base_agent import BaseAgent
import numpy as np
import math
import gymnasium
from .sensors import ForwardFacingCamera, Sensor
from .world import world_ref_to_game_ref

from shapely.geometry import Point


class Observer(BaseAgent):
    """Observer class."""

    def __init__(self, dims, speed, observer_id=-1, sensor: Sensor = ForwardFacingCamera(), time_factor: int = 1, discrete_action_space: bool = False, comm_range = 150):
        """Overwrite constructor."""
        super().__init__()
        self.img = pygame.image.load(f"{os.path.dirname(__file__)}/img/observer.png")
        self.img = pygame.transform.scale(self.img, dims)
        self.base_img = self.img.copy()
        self.rect = self.img.get_rect()
        self.x = self.rect.x
        self.y = self.rect.y
        self.id = observer_id
        self.out_of_bound = False
        self.goal_in_view = False
        self.goal_estimation = None
        self.comm_range = comm_range
        self.detected = set()

        self.trajectory_len = 3

        self.time_factor = time_factor
        self.speed = speed  # fixed speed
        # rad, positive angle counter-clockwise (note that the world referential is the opposite: y-axis down)
        self.orientation = 0
        self.altitude = 100
        self.steering_angle = np.pi / 10  # angular velocity
        self.sensor = sensor
        self.sensing_range = sensor.sensing_range

        if discrete_action_space:
            self.action_space = gymnasium.spaces.Discrete(5)
            self.discrete_action_space = True
        else:
            self.action_space = gymnasium.spaces.Box(low=-100, high=100, shape=(3,))
            self.discrete_action_space = False
        """
        2D steering. 0: right, 1: left
        """
        base_obs_len = 11
        sector_obs_len = self.GRID_RESOLUTION * self.GRID_RESOLUTION + 4
        self.observation_space = gymnasium.spaces.Box(
            low=-19.0,
            high=19.0,
            shape=(base_obs_len + sector_obs_len,),
            dtype=np.float32,
        )
        """
        [POI, x_g, y_g, theta, x, y, _...]: POI is treated as a bool corresponding to the
        presence of in POI in the FOV (1000 = True, -1000 = False).
        [x_g, y_g] are the goal's absolute coordinates
        theta is the agent's absolute orientation
        [x, y] is the agent's absolute position
        _ is a placeholer to maintain consistent observation spaces between agents (when padding is required)
        """

    def reset(self, seed=None, options=None):
        """Reset observer."""
        self.sensor.update_poly_points((self.rect.centerx, self.rect.centery), self.orientation, self.altitude)
        self.out_of_bound = False
        self.goal_estimation = None
        self.detected = set()
        self.latest_detected = set()
        pass

    def sync_pose_state(self):
        """Sync state derived from the current spawn pose."""
        self.trajectory = [(self.x, self.y)] * self.trajectory_len
        self.sensor.update_poly_points((self.rect.centerx, self.rect.centery), self.orientation, self.altitude)

    def draw(self, screen):
        """Draw observer."""
        # draw observer
        previous_pos = self.rect.center
        self.img = pygame.transform.rotate(
            self.base_img, self.orientation * 180 / np.pi - 45
        )  # offset by 45 deg because of source image
        self.rect = self.img.get_rect()
        self.rect.center = previous_pos
        screen.blit(self.img, self.rect)
        self.sensor.draw_sensor(screen)

    def update(self, area, world, action, found_goal):
        """Update observer."""
        # action: > 0 : turn right, < 0 : turn left, 0: straight
        if self.discrete_action_space:
            action = self.discrete_to_continuous(action)

        if action[0] < 0:
            self.orientation += self.steering_angle
        elif action[0] > 0:
            self.orientation -= self.steering_angle
        self.orientation = self.orientation % (2 * np.pi)
        self.x += self.speed * np.cos(self.orientation) * self.time_factor
        self.y += self.speed * np.sin(self.orientation) * self.time_factor
        self.update_detected_area(self.sensing_range)

        newpos = self.rect.copy()
        rect_pos = world_ref_to_game_ref([self.x, self.y], world.area)
        newpos.centerx = rect_pos[0]
        newpos.centery = rect_pos[1]

        # communication only possible if near a building
        if self.goal_estimation is not None:
            if not world.obstacles:
                # print(f"no obstacles! infinite comm range")
                world.observer_communication = self.goal_estimation
            else:
                for obstacle in world.obstacles:
                    obstacle_pos = obstacle.center
                    if dist(obstacle_pos[0], obstacle_pos[1], self.rect.centerx, self.rect.centery) < self.comm_range:
                        world.observer_communication = self.goal_estimation
                        break

        # make sure the players stay inside the screen
        if area.contains(newpos):
            self.rect = newpos
        else:
            self.out_of_bound = True

        self.sensor.update_poly_points((self.rect.centerx, self.rect.centery), self.orientation, self.altitude)

    def discrete_to_continuous(self, action):
        """Convert discrete action to box space."""
        if action == 0:
            out = [1, 0, 0]
        elif action == 1:
            out = [-1, 0, 0]
        elif action == 2:
            out = [0, 0, 0]
        elif action == 3:
            out = [0, 0, 0]
        elif action == 4:
            out = [0, 0, 0]
        return out

    def process_collision(self, o_rect, o_speed):
        """Process a collision.

        Args:
        ----
            o_rect : Obstacle rect
            dx, dy : agent speed along single axis
            o_speed : Obstacle speed

        Returns:
        -------
            is_collision: 1 if agent collides with obstacle

        """
        if not self.rect.colliderect(o_rect):
            return False
        else:
            return True

    def get_fov_obs(self, world, goals) -> list:
        """Return observations given world and sensor.

        Args:
        ----
            world (_type_): Pygame object.
            goals (_type_): List of goals.

        Returns:
        -------
            list: Observations.

        """
        if goals:
            goal = goals[0]
            if self.sensor and hasattr(self.sensor, "is_point_detected"):
                if self.sensor.is_point_detected((goal.rect.x, goal.rect.y)):
                    self.goal_in_view = True
                    goal.detected = True
                    self.goal_estimation = (goal.x, goal.y)
                    world.goal_position = (goal.x, goal.y)

        distances = self.obstacles_in_quadrants(Point(self.x, self.y), world.search_area, world.obstacles, world)
        norm_dists = [float(np.clip(d / max(self.sensing_range, 1e-6), 0.0, 1.0)) for d in distances]

        orient_norm = float(((self.orientation % (2 * np.pi)) / np.pi) - 1.0)

        obs_vec = [orient_norm] + norm_dists
        while len(obs_vec) < 11:
            obs_vec.append(0.0)

        obs = np.array(obs_vec, dtype=np.float32)
        return np.concatenate((obs, self.build_sector_features(world))).astype(np.float32, copy=False)

    def observe(self, world, agents, goals):
        """Observe observer."""
        return self.get_fov_obs(world, goals)
    
    def obstacles_in_quadrants(self, point, area, obstacles, world):
        """Find distances to obstacles in the 4 quadrants."""
        pygame_area = world.area  # Needed for coordinate conversion
        px, py = world_ref_to_game_ref((point.x, point.y), pygame_area)

        # Initialize distances with sensing range
        distances = {
            "right": self.sensing_range,
            "up": self.sensing_range,
            "left": self.sensing_range,
            "down": self.sensing_range,
        }

        # --- Find closest point on each obstacle ---
        for obstacle in obstacles:
            closest_x, closest_y = obstacle.clamp(pygame.Rect(px, py, 0, 0)).topleft  # Closest point on rect
            distance = np.hypot(closest_x - px, closest_y - py)

            if distance < self.sensing_range:
                if closest_x > px:
                    distances["right"] = min(distances["right"], distance)
                if closest_y > py:
                    distances["down"] = min(distances["down"], distance)  # y is inverted in pygame
                if closest_x < px:
                    distances["left"] = min(distances["left"], distance)
                if closest_y < py:
                    distances["up"] = min(distances["up"], distance)  # y is inverted in pygame

        # --- Find closest point on the area boundary ---
        closest_point = area.boundary.interpolate(area.boundary.project(point))
        ax, ay = closest_point.x, closest_point.y
        distance = np.hypot(ax - point.x, ay - point.y)

        if distance < self.sensing_range:
            if ax > point.x:
                distances["right"] = min(distances["right"], distance)
            if ay > point.y:
                distances["up"] = min(distances["up"], distance)
            if ax < point.x:
                distances["left"] = min(distances["left"], distance)
            if ay < point.y:
                distances["down"] = min(distances["down"], distance)

        result = [dist for dist in distances.values()]

        return result

def dist(x1, y1, x2, y2):
    """Distance between two points."""
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
