"""Observer module."""

import os

import pygame
from .base_agent import BaseAgent
import numpy as np
import gymnasium
from .sensors import ForwardFacingCamera, Sensor
from .world import world_ref_to_game_ref


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

        self.time_factor = time_factor
        self.speed = speed  # fixed speed
        # rad, positive angle counter-clockwise (note that the world referential is the opposite: y-axis down)
        self.orientation = 0
        self.altitude = 100
        self.steering_angle = np.pi / 10  # angular velocity
        self.sensor = sensor

        if discrete_action_space:
            self.action_space = gymnasium.spaces.Discrete(5)
            self.discrete_action_space = True
        else:
            self.action_space = gymnasium.spaces.Box(low=-100, high=100, shape=(3,))
            self.discrete_action_space = False
        """
        2D steering. 0: right, 1: left
        """
        self.observation_space = gymnasium.spaces.Box(
            low=-10000, high=10000, shape=(11,)
        )  # handling the presence and position of 1 POI for now
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
        pass

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

    def update(self, area, world, action):
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

        newpos = self.rect.copy()
        rect_pos = world_ref_to_game_ref([self.x, self.y], world.area)
        newpos.centerx = rect_pos[0]
        newpos.centery = rect_pos[1]

        # communication only possible if near a building
        if self.goal_estimation is not None:
            if not world.obstacles:
                print(f"no obstacles! infinite comm range")
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
        obs = [-1000, 0, 0, self.orientation, self.x, self.y, 0, 0, 0, 0, 0]
        # print(f"observer obs: {obs}")
        for goal in goals:
            if self.sensor.is_point_detected((goal.rect.x, goal.rect.y)):
                self.goal_in_view = True
                goal.detected = True  # seen at least once
                self.goal_estimation = (
                    goal.x,
                    goal.y,
                )  # hardcoded communication for now
                obs = [1000, goal.x, goal.y, self.orientation, self.x, self.y, 0, 0, 0, 0, 0]
                return np.array(obs, np.float32)
            else:
                self.goal_in_view = False
        return np.array(obs, np.float32)

    def observe(self, world, agents, goals):
        """Observe observer."""
        return self.get_fov_obs(world, goals)


def dist(x1, y1, x2, y2):
    """Distance between two points."""
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


