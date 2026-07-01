"""Observer module."""

import math
import os

import gymnasium
import numpy as np
import pygame

from .base_agent import BaseAgent
from .sensors import ForwardFacingCamera, Sensor
from .world import world_ref_to_game_ref


class Observer(BaseAgent):
    """Observer class."""

    def __init__(
        self,
        dims,
        speed,
        observer_id=-1,
        sensor: Sensor = ForwardFacingCamera(),
        time_factor: int = 1,
        discrete_action_space: bool = False,
        comm_range=150,
    ):
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
        self.min_dist_record = float('inf')

        self.trajectory_len = 3

        self.time_factor = time_factor
        self.speed = float(speed)
        self.max_speed = float(speed)
        # rad, positive angle counter-clockwise (note that the world referential is the opposite: y-axis down)
        self.orientation = 0
        self.altitude = 100
        self.steering_angle = np.pi / 18  # angular velocity
        self.sensor = sensor
        self.sensor.sensing_range = max(float(self.sensor.sensing_range), 1.0)
        self.sensing_range = sensor.sensing_range

        if discrete_action_space:
            self.action_space = gymnasium.spaces.Discrete(5)
            self.discrete_action_space = True
        else:
            self.action_space = gymnasium.spaces.Box(
                low=-self.max_speed,
                high=self.max_speed,
                shape=(3,),
                dtype=np.float32,
            )
            self.discrete_action_space = False
        """
        2D velocity control compatible with the drone action format: [vx, vy, aux].
        """
        self.observation_space = gymnasium.spaces.Dict(
            {
                "global_map": gymnasium.spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.GLOBAL_MAP_SIZE, self.GLOBAL_MAP_SIZE, 5),
                    dtype=np.float32,
                ),
                "local_map": gymnasium.spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.LOCAL_MAP_SIZE, self.LOCAL_MAP_SIZE, 5),
                    dtype=np.float32,
                ),
                "vector": gymnasium.spaces.Box(
                    low=-self.max_speed,
                    high=self.max_speed,
                    shape=(self.ACTION_HISTORY_LENGTH * self.ACTION_DIM,),
                    dtype=np.float32,
                ),
            }
        )

    def reset(self, seed=None, options=None):
        """Reset observer."""
        self.out_of_bound = False
        self.goal_in_view = False
        self.goal_estimation = None
        self.orientation = 0.0
        self.detected = set()
        self.min_dist_record = float('inf')
        self.latest_detected = np.empty((0, 2), dtype=np.int32)
        self.reset_action_history()
        self.sensor.update_poly_points((self.rect.centerx, self.rect.centery), self.orientation, self.altitude)

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
        if self.discrete_action_space:
            action = self.discrete_to_continuous(action)

        vx = float(np.clip(action[0], -self.max_speed, self.max_speed))
        vy = float(np.clip(action[1], -self.max_speed, self.max_speed))
        self.push_action_history([vx, vy, float(action[2]) if len(action) > 2 else 0.0])
        if not np.isclose(vx, 0.0) or not np.isclose(vy, 0.0):
            self.orientation = math.atan2(vy, vx) % (2 * np.pi)
        self.x += vx * self.time_factor
        self.y += vy * self.time_factor
        self.update_detected_area(self.sensing_range)

        newpos = self.rect.copy()
        rect_pos = world_ref_to_game_ref([self.x, self.y], world.area)
        newpos.centerx = rect_pos[0]
        newpos.centery = rect_pos[1]

        # communication only possible if near a building
        if self.goal_estimation is not None:
            if world.obstacle_centers.size == 0:
                # print(f"no obstacles! infinite comm range")
                world.observer_communication = self.goal_estimation
            else:
                dx = world.obstacle_centers[:, 0] - float(self.rect.centerx)
                dy = world.obstacle_centers[:, 1] - float(self.rect.centery)
                if np.any((dx * dx + dy * dy) < self.comm_range * self.comm_range):
                    world.observer_communication = self.goal_estimation

        # make sure the players stay inside the screen
        if area.contains(newpos):
            self.rect = newpos
        else:
            self.out_of_bound = True

        self.sensor.update_poly_points((self.rect.centerx, self.rect.centery), self.orientation, self.altitude)

    def discrete_to_continuous(self, action):
        """Convert discrete action to box space."""
        if action == 0:
            out = [0, 0, 1]
        elif action == 1:
            out = [self.max_speed, self.max_speed, 0]
        elif action == 2:
            out = [self.max_speed, -self.max_speed, 0]
        elif action == 3:
            out = [-self.max_speed, self.max_speed, 0]
        else:
            out = [-self.max_speed, -self.max_speed, 0]
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

    def get_fov_obs(self, world, agents, goals) -> dict[str, np.ndarray]:
        """Return the observer's global/local-map observation."""
        if goals and self.sensor and hasattr(self.sensor, "is_point_detected"):
            for goal in goals:
                if self.sensor.is_point_detected((goal.rect.x, goal.rect.y)):
                    self.goal_in_view = True
                    goal.detected = True
                    self.goal_estimation = (goal.x, goal.y)
                    world.goal_position = (goal.x, goal.y)
                    break

        drone_positions = [
            (agent.x, agent.y)
            for agent in agents
            if agent.__class__.__name__ == "Drone"
        ]
        goal_positions = [(goal.x, goal.y) for goal in goals]

        padded_channels = [
            world.padded_coverage_map,
            world.padded_search_mask,
            world.padded_explored_obstacle_map,
            self.build_padded_entity_channel(world, drone_positions),
            self.build_padded_entity_channel(world, goal_positions),
        ]
        global_map = self.build_global_map_view(world, padded_channels)
        local_map = self.build_local_map_view(
            world,
            static_channels=[
                world.observation_coverage_map,
                world.search_mask,
                world.explored_obstacle_map,
            ],
            entity_position_groups=[
                drone_positions,
                goal_positions,
            ],
            sensing_range=self.sensing_range,
        )

        return {
            "global_map": global_map,
            "local_map": local_map,
            "vector": self.build_action_history_vector(),
        }

    def observe(self, world, agents, goals):
        """Observe observer."""
        return self.get_fov_obs(world, agents, goals)
    
    def obstacles_in_quadrants(self, world):
        """Find distances to obstacles in the 4 quadrants."""
        px, py = world_ref_to_game_ref((self.x, self.y), world.area)
        distances = self.obstacle_distance_channels(px, py, self.sensing_range, world.obstacle_bounds)
        self.update_boundary_distance_array(self.x, self.y, world.search_bounds, self.sensing_range, distances)
        return distances

def dist(x1, y1, x2, y2):
    """Distance between two points."""
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
