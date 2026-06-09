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

        self.trajectory_len = 3
        self.trajectory = []
        self.last_goal_distance = None
        self.last_base_distance = None
        self.last_boundary_distance = None
        self.last_frontier_distance = None

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
            # self.action_space = gymnasium.spaces.Box(low=-100, high=100, shape=(3,))
            # 드론과 옵저버 모두 동일하게 [-1.0, 1.0] 범위로 설정합니다.
            self.action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(3,))
            self.discrete_action_space = False
        """
        2D steering. 0: right, 1: left
        """
        self.observation_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(24,)
        )
        """
        Observation contains the mission phase, goal/frontier vectors,
        orientation as sin/cos, normalized position and boundary margins,
        aggregate coverage, and recent relative trajectory offsets.
        """

    def reset(self, seed=None, options=None):
        """Reset observer."""
        self.out_of_bound = False
        self.goal_in_view = False
        self.goal_estimation = None
        self.last_goal_distance = None
        self.last_base_distance = None
        self.last_boundary_distance = None
        self.last_frontier_distance = None
        self.orientation = 0.0
        self.trajectory = []

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

    def update(self, area, world, action):
        """Update observer."""
        # action: > 0 : turn right, < 0 : turn left, 0: straight
        if self.discrete_action_space:
            action = self.discrete_to_continuous(action)

        steering = float(np.clip(action[0], -1.0, 1.0))
        self.orientation -= steering * self.steering_angle
        self.orientation = self.orientation % (2 * np.pi)
        self.x += self.speed * np.cos(self.orientation) * self.time_factor
        self.y += self.speed * np.sin(self.orientation) * self.time_factor

        self.trajectory.append((self.x, self.y))
        if len(self.trajectory) > self.trajectory_len:
            self.trajectory.pop(0)

            
        newpos = self.rect.copy()
        rect_pos = world_ref_to_game_ref([self.x, self.y], world.area)
        newpos.centerx = rect_pos[0]
        newpos.centery = rect_pos[1]

        # communication only possible if near a building
        if self.goal_estimation is not None:
            if not world.obstacles:
                world.goal_known = True
                world.observer_communication = [self.goal_estimation[0], self.goal_estimation[1]]
            else:
                for obstacle in world.obstacles:
                    obstacle_pos = obstacle.center
                    if dist(obstacle_pos[0], obstacle_pos[1], self.rect.centerx, self.rect.centery) < self.comm_range:
                        world.goal_known = True
                        world.observer_communication = [self.goal_estimation[0], self.goal_estimation[1]]
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
            out = [1.0, 0, 0]
        elif action == 1:
            out = [0.5, 0, 0]
        elif action == 2:
            out = [0, 0, 0]
        elif action == 3:
            out = [-0.5, 0, 0]
        elif action == 4:
            out = [-1.0, 0, 0]
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
        if len(goals) > 0:
            goal = goals[0]
            if self.sensor.is_point_detected(goal.rect.center):
                self.goal_in_view = True
                goal.detected = True
                self.goal_estimation = (goal.x, goal.y)
                world.goal_known = True
                world.observer_communication = [goal.x, goal.y]
            else:
                self.goal_in_view = False

        minx, miny, maxx, maxy = world.search_area.bounds
        area_width = max(maxx - minx, 1.0)
        area_height = max(maxy - miny, 1.0)
        area_diag = max(np.hypot(area_width, area_height), 1.0)

        if world.goal_known:
            goal_x, goal_y = world.observer_communication
            goal_dx = goal_x - self.x
            goal_dy = goal_y - self.y
            goal_dist = np.hypot(goal_dx, goal_dy)
            goal_heading = np.arctan2(goal_dy, goal_dx)
            goal_heading_error = ((goal_heading - self.orientation + np.pi) % (2 * np.pi)) - np.pi
        else:
            goal_dx, goal_dy = 0.0, 0.0
            goal_dist = 0.0
            goal_heading_error = 0.0

        frontier_dx, frontier_dy = self.nearest_unexplored_vector(world)
        frontier_dist = np.hypot(frontier_dx, frontier_dy)

        boundary_obs = [
            np.clip((maxx - self.x) / area_width, 0.0, 1.0),
            np.clip((maxy - self.y) / area_height, 0.0, 1.0),
            np.clip((self.x - minx) / area_width, 0.0, 1.0),
            np.clip((self.y - miny) / area_height, 0.0, 1.0),
        ]
        boundary_clearance = min(maxx - self.x, maxy - self.y, self.x - minx, self.y - miny)
        coverage_ratio = (
            float(len(getattr(world, "explored_grids", set()) | getattr(world, "observer_explored_grids", set())))
            / float(max(len(getattr(world, "search_grid_centers", {})), 1))
        )
        base_obs = [
            1.0 if world.goal_known else -1.0,
            np.clip(goal_dx / area_width, -1.0, 1.0),
            np.clip(goal_dy / area_height, -1.0, 1.0),
            np.clip(goal_dist / area_diag, 0.0, 1.0),
            np.clip(goal_heading_error / np.pi, -1.0, 1.0),
            np.clip(frontier_dx / area_width, -1.0, 1.0),
            np.clip(frontier_dy / area_height, -1.0, 1.0),
            np.clip(frontier_dist / area_diag, 0.0, 1.0),
            float(np.sin(self.orientation)),
            float(np.cos(self.orientation)),
            np.clip(((self.x - minx) / area_width) * 2 - 1, -1.0, 1.0),
            np.clip(((self.y - miny) / area_height) * 2 - 1, -1.0, 1.0),
            np.clip(boundary_clearance / max(min(area_width, area_height), 1.0), 0.0, 1.0),
            np.clip(coverage_ratio * 2.0 - 1.0, -1.0, 1.0),
        ] + boundary_obs

        motion_scale = max(self.speed * self.time_factor * self.trajectory_len, 1.0)
        traj_obs = []
        for pos in self.trajectory:
            traj_obs.extend(
                [
                    np.clip((self.x - pos[0]) / motion_scale, -1.0, 1.0),
                    np.clip((self.y - pos[1]) / motion_scale, -1.0, 1.0),
                ]
            )

        obs = np.array(base_obs + traj_obs, dtype=np.float32)
        return np.clip(obs, -1.0, 1.0).astype(np.float32, copy=False)

    def observe(self, world, agents, goals):
        """Observe observer."""
        return self.get_fov_obs(world, goals)

    def nearest_unexplored_vector(self, world):
        """Return a vector toward the next unexplored portion of the search grid."""
        search_grid_centers = getattr(world, "search_grid_centers", {})
        explored = getattr(world, "explored_grids", set())
        observer_explored = getattr(world, "observer_explored_grids", set())
        if not search_grid_centers:
            return 0.0, 0.0

        cell_size = float(getattr(world, "exploration_cell_size", 20))
        local_radius_sq = float(max(self.speed * self.time_factor * 6.0, cell_size * 3.0) ** 2)

        nearest_dx = 0.0
        nearest_dy = 0.0
        nearest_dist_sq = float("inf")
        weighted_x = 0.0
        weighted_y = 0.0
        weight_total = 0.0

        for grid_key, (center_x, center_y) in search_grid_centers.items():
            if grid_key in explored or grid_key in observer_explored:
                continue

            dx = center_x - self.x
            dy = center_y - self.y
            dist_sq = dx * dx + dy * dy
            if dist_sq < nearest_dist_sq:
                nearest_dist_sq = dist_sq
                nearest_dx = dx
                nearest_dy = dy

            distance = np.sqrt(dist_sq)
            weight = 1.0 / max(distance, cell_size)
            weighted_x += dx * weight
            weighted_y += dy * weight
            weight_total += weight

        if nearest_dist_sq <= local_radius_sq:
            return nearest_dx, nearest_dy
        if weight_total > 0:
            return weighted_x / weight_total, weighted_y / weight_total
        return 0.0, 0.0


def dist(x1, y1, x2, y2):
    """Distance between two points."""
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
