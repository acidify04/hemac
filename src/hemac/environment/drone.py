"""Drone module."""

import math
import os
from random import randrange

import gymnasium
import numpy as np
import pygame
from pymap3d import geodetic2enu

from hemac.environment.base_agent import BaseAgent
from hemac.helpers.helper import world_ref_to_game_ref, game_ref_to_world_ref
from hemac.environment.sensors import DownwardFacingCamera, Sensor, RoundCamera

from shapely.geometry import Point


class UWB:
    """UWB (Ultra Wide Band) class."""

    def __init__(self, randomizer: np.random.Generator, max_range, noise=0.15, bad_read_frequency=0.01):
        """Overwrite constructor."""
        self.randomizer = randomizer
        self.bias = 0
        # self.noise = 0.0707 #within 10 cm 95% of the time
        # Based on https://doi.org/10.1016/j.measurement.2022.112276,
        # noise is ~15cm at a maximum 200m with the newest methods on UWB
        self.noise = noise  # within 10 cm 95% of the time
        self.bad_read_frequency = bad_read_frequency  # probability of very bad measurement (error > 1 m)
        self.max_range = max_range

    def measure(self, true_dist):
        """Measure distance between true and predicted distance."""
        error = self.randomizer.normal(self.bias, self.noise)
        if abs(error) > 1:
            error = np.random.choice([-1, 1])

        # Return the maximum distance if out of sight
        if true_dist > self.max_range:
            true_dist = self.max_range

        return true_dist + error


class IMU:
    """IMU class."""

    def __init__(self, randomizer: np.random.Generator, noise=0.15):
        """Overwrite constructor."""
        self.randomizer = randomizer
        self.bias = randomizer.normal(0, 0.025)
        self.variance = noise
        self.scale_error = randomizer.random() * 0.02 - 0.01
        self.measured_accel = [0, 0]

    def measure(self, ax, ay):
        """Measure acceleration."""
        self.measured_accel[0] = ax * (1 + self.scale_error) + self.randomizer.normal(self.bias, self.variance)
        self.measured_accel[1] = ay * (1 + self.scale_error) + self.randomizer.normal(self.bias, self.variance)
        return self.measured_accel


class Drone(BaseAgent):
    """Drone class."""

    def __init__(
        self,
        drone_config,
        number_of_drones,
        randomizer,
        world,
        drone_id=-1,
        sensor: Sensor = DownwardFacingCamera(0.7, 0.7),
        time_factor=0.8,
        num_discrete_actions=5,
    ):
        """Overwrite constructor."""
        super().__init__()

        self.id = drone_id
        self.out_of_bound = False
        self.time_factor = time_factor
        self.starting_pos = None
        self.has_custom_starting_pos = False

        ui_dims = 40
        dims = [1, 1]
        dims_meters = [40, 40]

        if drone_config:
            pixel_to_meter_ref = 1  # how many game pixel to represent a meter
            ui_dims = drone_config.get("drone_ui_dimension", 40)
            # dims_meters = [drone_config.get("drone_dimension")[0] / 100, drone_config.get("drone_dimension")[1] / 100]
            dims_meters = [
                drone_config.get("drone_dimension", [40, 40])[0] / 100,
                drone_config.get("drone_dimension", [40, 40])[1] / 100,
            ]
            dims = [math.ceil(pixel_to_meter_ref * dims_meters[0]), math.ceil(pixel_to_meter_ref * dims_meters[1])]
            self.max_speed = drone_config.get("drone_max_speed", 16)
            self.max_thrust = drone_config.get("drone_max_thrust", 4)
            self.altitude = drone_config.get("drone_altitude", 30)
            self.max_charge = drone_config.get("drone_max_charge", 9999)
            if len(drone_config.get("drones_starting_pos", [])) >= drone_id + 1:
                if drone_config.get("starting_pos_coordinates_type") == "geo":
                    # we convert geo to cardinal position
                    self.starting_pos = list(
                        geodetic2enu(
                            drone_config.get("drones_starting_pos")[drone_id][0],
                            drone_config.get("drones_starting_pos")[drone_id][1],
                            0,
                            drone_config.get("position_origin", {}).get("latitude"),
                            drone_config.get("position_origin", {}).get("longitude"),
                            0,
                        )
                    )
                    self.has_custom_starting_pos = True
                else:
                    self.starting_pos = drone_config.get("drones_starting_pos")[drone_id]
                    self.has_custom_starting_pos = True
            else:
                self.starting_pos = [0, 0]  # computer random position
        else:
            self.max_speed = 16
            self.max_thrust = 4
            self.altitude = 30
            self.max_charge = 60 * 32

        self.img = pygame.transform.scale(
            pygame.image.load(f"{os.path.dirname(__file__)}/img/drone.png"), [ui_dims, ui_dims]
        )
        self.drone_color = (randrange(0, 255), randrange(0, 255), randrange(0, 255))
        self.rect = pygame.Rect(0, 0, dims[0], dims[1])
        self.charging_distance = 50
        self.x = self.rect.x
        self.y = self.rect.y
        self.vx = 0
        self.vy = 0
        self.accel_x = 0
        self.accel_y = 0
        self.IMU = IMU(randomizer)
        self.UWB = UWB(randomizer, max_range=200)
        self.randomizer = randomizer
        self.sensor = sensor
        self.sensing_range = sensor.sensing_range if isinstance(sensor, RoundCamera) else 50
        self.orientation = 0.0
        self.carried_targets = 0
        self.carrying_capacity = 1

        self.world = world
        self.sensor.update_poly_points((self.rect.centerx, self.rect.centery), self.orientation, self.altitude)

        self.drone_config = drone_config
        self.number_of_drones = number_of_drones

        if drone_config.get("discrete_action_space", False):
            self.action_space = gymnasium.spaces.Discrete(5)
            self.discrete_action_space = True
        else:
            # self.action_space = gymnasium.spaces.Box(low=-self.max_speed, high=self.max_speed, shape=(3,))
            # 드론과 옵저버 모두 동일하게 [-1.0, 1.0] 범위로 설정합니다.
            self.action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(3,))
            self.discrete_action_space = False

        """
        action space: [wanted vx, wanted vy, recharge] where recharge is mapped to a bool for trying to recharge.
        """
        self.observation_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(53 + self.number_of_drones * 2,)
        )
        """
        Observation space: mission-phase flag, normalized relative quantities for
        goal/base/observer/frontier, compact vehicle state, local explored-tile
        mask, normalized distances to boundaries/obstacles, other-drone relative
        positions, and recent relative trajectory offsets.
        """
        self.charge_level = self.max_charge
        self.charging = False
        self.charging_point = (0, 0)
        self.goto_pos = [0, 0]

        self.trajectory_len = 3
        self.trajectory = []
        self.last_goal_distance = None
        self.last_observer_distance = None
        self.last_boundary_distance = None
        self.last_frontier_distance = None
        self.max_sector_progress = 0.0

        self.is_broken = False

    def reset(self, seed=None, options=None):
        """Reset drone."""
        self.charge_level = self.max_charge
        self.charging = False
        self.out_of_bound = False
        self.carried_targets = 0
        self.is_broken = False
        self.vx = 0.0
        self.vy = 0.0
        self.accel_x = 0.0
        self.accel_y = 0.0
        self.orientation = 0.0
        self.goto_pos = [0, 0]
        self.trajectory = []
        self.last_goal_distance = None
        self.last_observer_distance = None
        self.last_boundary_distance = None
        self.last_frontier_distance = None
        self.max_sector_progress = 0.0

    def sync_pose_state(self):
        """Sync state derived from the current spawn pose."""
        self.trajectory = [(self.x, self.y)] * self.trajectory_len
        self.sensor.update_poly_points((self.rect.centerx, self.rect.centery), self.orientation, self.altitude)

    def is_newly_explored(self, explored_grids):
        """탐색되지 않은 새로운 영역인지 확인합니다."""
        grid_key = (int(self.x // 20), int(self.y // 20))
        return grid_key not in explored_grids
    
    def draw(self, screen):
        """Draw drone."""
        # draw drone UI representation (not necessary accurate to real drone dimensions)
        img_pos = pygame.Rect(
            self.rect.left - self.img.get_width() / 2, self.rect.top - self.img.get_height() / 2, 0, 0
        )
        if self.carried_targets:
            for i in range(self.carried_targets):
                carried_target = pygame.Rect(self.rect.left + 5, self.rect.top + i * 5, 8, 8)
                pygame.draw.rect(screen, (128, 255, 255), carried_target)
        screen.blit(self.img, img_pos)
        font = pygame.font.SysFont("Trebuchet MS", 16)
        id_text = font.render(str(self.id), True, self.drone_color)
        screen.blit(id_text, self.rect.center)

        # draw drone real size
        shape_surf = pygame.Surface(self.rect.size, pygame.SRCALPHA)
        pygame.draw.rect(shape_surf, (255, 0, 0, 255), shape_surf.get_rect())
        screen.blit(shape_surf, self.rect)

        if self.goto_pos:
            pygame.draw.circle(screen, self.drone_color, self.goto_pos, 6)
            font = pygame.font.SysFont("Trebuchet MS", 16)
            id_text = font.render("+", True, (255, 255, 255))
            screen.blit(id_text, [self.goto_pos[0] - 5, self.goto_pos[1] - 10])

        # draw FOV
        self.sensor.draw_sensor(screen)

        if self.charging:
            pygame.draw.line(screen, (155, 255, 255), self.rect.center, self.charging_point, width=3)

    def update(self, area, world, action):
        """Update drone."""
        if self.is_broken:
            return

        if self.discrete_action_space:
            action = self.discrete_to_continuous(action)

        # if action[2] > 0:  # drone tries to recharge
        #     self.closest_point_in_base = closest_point_in_rect(world.base, self.rect.center)
        #     can_charge = self.charging_distance > dist(
        #         self.rect.x,
        #         self.rect.y,
        #         self.closest_point_in_base[0],
        #         self.closest_point_in_base[1],
        #     )
        #     if can_charge:
        #         if (self.closest_point_in_base == self.rect.center).all():
        #             self.charging_point = world.base.center
        #         else:
        #             self.charging_point = self.closest_point_in_base  # game ref
        #         self.charging = True
        #         self.charge_level += 9
        #         if self.charge_level > self.max_charge:
        #             self.charge_level = self.max_charge
        #         # print("charging at base!")
        #     else:  # check if provisioner near
        #         for id, coords in world.provisioners.items():
        #             if self.charging_distance > dist(self.x, self.y, coords[0], coords[1]):
        #                 self.charging_point = world_ref_to_game_ref(coords, area)
        #                 self.charging = True
        #                 self.charge_level += 9
        #                 if self.charge_level > self.max_charge:
        #                     self.charge_level = self.max_charge
        # else:
        #     self.charging = False

        # if not self.charging:  # drone wants to move (only if not currently charging)
        #     if self.charge_level > 0:
        #         self.charge_level -= 1
        self.previous_accel = [self.accel_x, self.accel_y]

        # compute target acceleration compensating for predicted drag (a = dV/dt + drag compensation)
        self.accel_x = (action[0] - self.vx) / self.time_factor + 0.02 * action[0] * abs(action[0])
        self.accel_y = (action[1] - self.vy) / self.time_factor + 0.02 * action[1] * abs(action[1])

        # for position control
        self.goto_pos = (int(self.rect.x + action[0]), int(self.rect.y - action[1]))

        # compute achievable acceleration given max thrust
        if np.linalg.norm([self.accel_x, self.accel_y]) > self.max_thrust:
            self.accel_x = self.accel_x / np.linalg.norm([self.accel_x, self.accel_y]) * self.max_thrust
            self.accel_y = self.accel_y / np.linalg.norm([self.accel_x, self.accel_y]) * self.max_thrust

        # compute actual acceleration given drag and wind
        self.accel_x -= 0.02 * self.vx * abs(self.vx) + self.randomizer.normal(0, 0.1)
        self.accel_y -= 0.02 * self.vy * abs(self.vy) + self.randomizer.normal(0, 0.1)

        # blend with previous acceleration to simulate delay
        self.accel_x = 0.6 * self.accel_x + 0.4 * self.previous_accel[0]
        self.accel_y = 0.6 * self.accel_y + 0.4 * self.previous_accel[1]

        # update position using the exact method (assuming constant acceleration)
        dx = self.vx * self.time_factor + 0.5 * self.accel_x * self.time_factor**2
        dy = self.vy * self.time_factor + 0.5 * self.accel_y * self.time_factor**2

        # move and update pygame coordinates
        self.x += dx
        self.y += dy
        self.trajectory.append((self.x, self.y))
        if len(self.trajectory) > self.trajectory_len:
            self.trajectory.pop(0)

        newpos = self.rect.copy()
        rect_pos = world_ref_to_game_ref([self.x, self.y], world.area)
        newpos.centerx = rect_pos[0]
        newpos.centery = rect_pos[1]

        # make sure the players stay inside the screen
        if area.contains(newpos):
            self.rect = newpos
        else:
            self.rect = newpos
            self.out_of_bound = True

        # update velocity
        self.vx = self.vx + self.accel_x * self.time_factor
        self.vy = self.vy + self.accel_y * self.time_factor
            # LOGGER.info(f"Velocity: {round((self.vx ** 2 + self.vy ** 2) ** 0.5)} m/s, Pos: {[self.x, self.y]}")
        # else:
        #     # print("drone has no energy!")
        #     pass

        self.sensor.update_poly_points((self.rect.centerx, self.rect.centery), self.orientation, self.altitude)

    def discrete_to_continuous(self, action):
        """Convert discrete action to box space."""
        # if action == 0:
        #     out = [0, 0, 1]
        if action == 0:
            out = [0, 0, 0]
        elif action == 1:
            out = [10, 10, 0]
        elif action == 2:
            out = [10, -10, 0]
        elif action == 3:
            out = [-10, 10, 0]
        elif action == 4:
            out = [-10, -10, 0]
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
        
        self.is_broken = True
        return True

    def observe(self, world, agents, poi) -> np.array:
        """Observe the world."""
        minx, miny, maxx, maxy = world.search_area.bounds
        area_width = max(maxx - minx, 1.0)
        area_height = max(maxy - miny, 1.0)
        area_diag = max(np.hypot(area_width, area_height), 1.0)
        observer = next((agent for agent in agents if agent.__class__.__name__ == "Observer"), None)

        if world.goal_known:
            goal_x, goal_y = world.observer_communication
            goal_dist = np.hypot(goal_x - self.x, goal_y - self.y)
            to_goal_x = np.clip((goal_x - self.x) / area_width, -1.0, 1.0)
            to_goal_y = np.clip((goal_y - self.y) / area_height, -1.0, 1.0)
        else:
            to_goal_x = 0.0
            to_goal_y = 0.0
            goal_dist = 0.0

        frontier_dx, frontier_dy = self.nearest_unexplored_vector(world, observer)
        frontier_x = np.clip(frontier_dx / area_width, -1.0, 1.0)
        frontier_y = np.clip(frontier_dy / area_height, -1.0, 1.0)
        frontier_dist = np.hypot(frontier_dx, frontier_dy)

        base_x, base_y = game_ref_to_world_ref(world.base.center, world.area)
        to_base_x = np.clip((base_x - self.x) / area_width, -1.0, 1.0)
        to_base_y = np.clip((base_y - self.y) / area_height, -1.0, 1.0)

        if observer is not None:
            to_observer_x = np.clip((observer.x - self.x) / area_width, -1.0, 1.0)
            to_observer_y = np.clip((observer.y - self.y) / area_height, -1.0, 1.0)
            observer_dist = np.hypot(observer.x - self.x, observer.y - self.y)
        else:
            to_observer_x = 0.0
            to_observer_y = 0.0
            observer_dist = 0.0

        patrol_pref_x, patrol_pref_y = self.preferred_patrol_vector(world, observer)
        if self.number_of_drones > 1:
            role_obs = np.clip((2.0 * self.id / (self.number_of_drones - 1)) - 1.0, -1.0, 1.0)
        else:
            role_obs = 0.0
        local_mask = self.local_exploration_mask(world)
        coverage_ratio = (
            float(len(getattr(world, "explored_grids", set()) | getattr(world, "observer_explored_grids", set())))
            / float(max(len(getattr(world, "search_grid_centers", {})), 1))
        )
        boundary_clearance = min(maxx - self.x, maxy - self.y, self.x - minx, self.y - miny)

        raw_distances = self.obstacles_in_quadrants(Point(self.x, self.y), world.search_area, world.obstacles)
        distances = np.array(
            [
                np.clip(raw_distances[0] / area_width, 0.0, 1.0),
                np.clip(raw_distances[1] / area_height, 0.0, 1.0),
                np.clip(raw_distances[2] / area_width, 0.0, 1.0),
                np.clip(raw_distances[3] / area_height, 0.0, 1.0),
            ],
            dtype=np.float32,
        )

        agents_rel_pos = [
            np.clip(coord / area_width, -1.0, 1.0) if idx % 2 == 0 else np.clip(coord / area_height, -1.0, 1.0)
            for agent in agents
            if isinstance(agent, Drone) and agent is not self
            for idx, coord in enumerate((agent.x - self.x, agent.y - self.y))
        ]
        motion_scale = max(self.max_speed * self.time_factor * self.trajectory_len, 1.0)
        traj_obs = []
        for pos in self.trajectory:
            traj_obs.extend(
                [
                    np.clip((self.x - pos[0]) / motion_scale, -1.0, 1.0),
                    np.clip((self.y - pos[1]) / motion_scale, -1.0, 1.0),
                ]
            )

        obs = np.array(
            [
                1.0 if world.goal_known else -1.0,
                to_goal_x,
                to_goal_y,
                np.clip(goal_dist / area_diag, 0.0, 1.0),
                self.charge_level / self.max_charge,
                to_base_x,
                to_base_y,
                to_observer_x,
                to_observer_y,
                np.clip(observer_dist / area_diag, 0.0, 1.0),
                frontier_x,
                frontier_y,
                np.clip(frontier_dist / area_diag, 0.0, 1.0),
                role_obs,
                patrol_pref_x,
                patrol_pref_y,
                np.clip(boundary_clearance / max(min(area_width, area_height), 1.0), 0.0, 1.0),
                np.clip(coverage_ratio * 2.0 - 1.0, -1.0, 1.0),
                np.clip(self.vx / max(self.max_speed, 1.0), -1.0, 1.0),
                np.clip(self.vy / max(self.max_speed, 1.0), -1.0, 1.0),
            ],
            dtype=np.float32,
        )
        obs = np.concatenate((obs, local_mask, distances, agents_rel_pos, traj_obs), dtype=np.float32)
        return np.clip(obs, -1.0, 1.0).astype(np.float32, copy=False)

    def local_exploration_mask(self, world, radius=2):
        """Return a local explored-tile mask around the drone."""
        search_grid_centers = getattr(world, "search_grid_centers", {})
        explored_grids = getattr(world, "explored_grids", set())
        observer_explored_grids = getattr(world, "observer_explored_grids", set())
        shared_explored = explored_grids | observer_explored_grids

        cell_size = getattr(self.world, "exploration_cell_size", 20)
        center_gx = int(self.x // cell_size)
        center_gy = int(self.y // cell_size)

        mask = []
        for gy in range(center_gy + radius, center_gy - radius - 1, -1):
            for gx in range(center_gx - radius, center_gx + radius + 1):
                grid_key = (gx, gy)
                if grid_key not in search_grid_centers:
                    mask.append(0.0)
                elif grid_key in shared_explored:
                    mask.append(1.0)
                else:
                    mask.append(-1.0)

        return np.array(mask, dtype=np.float32)

    def preferred_patrol_vector(self, world, observer=None):
        """Return the unit vector of this drone's preferred scouting sector."""
        minx, miny, maxx, maxy = world.search_area.bounds
        search_center_x = (minx + maxx) * 0.5
        search_center_y = (miny + maxy) * 0.5

        if observer is not None:
            origin_x, origin_y = observer.x, observer.y
        else:
            origin_x, origin_y = self.x, self.y

        center_angle = math.atan2(search_center_y - origin_y, search_center_x - origin_x)
        if self.number_of_drones <= 1:
            patrol_angle = center_angle
        else:
            fan_angles = np.linspace(-np.deg2rad(60), np.deg2rad(60), self.number_of_drones)
            patrol_angle = center_angle + float(fan_angles[self.id])

        return float(np.cos(patrol_angle)), float(np.sin(patrol_angle))

    def nearest_unexplored_vector(self, world, observer=None):
        """Return a relative vector toward a coverage target inside this drone's sector."""
        search_grid_centers = getattr(world, "search_grid_centers", {})
        explored_grids = getattr(world, "explored_grids", set())
        observer_explored_grids = getattr(world, "observer_explored_grids", set())
        cell_size = float(getattr(world, "exploration_cell_size", 20))

        patrol_pref_x, patrol_pref_y = self.preferred_patrol_vector(world, observer)
        preferred_angle = math.atan2(patrol_pref_y, patrol_pref_x)
        sector_half_width = np.deg2rad(30)
        local_radius_sq = float((self.sensing_range * 1.75) ** 2)

        if observer is not None:
            origin_x, origin_y = observer.x, observer.y
        else:
            origin_x, origin_y = self.x, self.y

        local_dx = 0.0
        local_dy = 0.0
        local_dist_sq = float("inf")
        sector_weighted_x = 0.0
        sector_weighted_y = 0.0
        sector_weight_total = 0.0
        fallback_weighted_x = 0.0
        fallback_weighted_y = 0.0
        fallback_weight_total = 0.0
        unsafe_sector_weighted_x = 0.0
        unsafe_sector_weighted_y = 0.0
        unsafe_sector_weight_total = 0.0

        minx, miny, maxx, maxy = world.search_area.bounds
        search_span = max(maxx - minx, maxy - miny, 1.0)
        safe_boundary_margin = max(self.sensing_range * 0.75, cell_size * 3.0)

        for grid_key, (center_x, center_y) in search_grid_centers.items():
            if grid_key in explored_grids or grid_key in observer_explored_grids:
                continue

            dx = center_x - self.x
            dy = center_y - self.y
            dist_sq = dx * dx + dy * dy
            if dist_sq < local_radius_sq and dist_sq < local_dist_sq:
                local_dist_sq = dist_sq
                local_dx = dx
                local_dy = dy

            radial_dx = center_x - origin_x
            radial_dy = center_y - origin_y
            radial_dist = math.hypot(radial_dx, radial_dy)
            radial_weight = 1.0 + (radial_dist / search_span)
            boundary_clearance = min(center_x - minx, maxx - center_x, center_y - miny, maxy - center_y)
            is_boundary_safe = boundary_clearance >= safe_boundary_margin

            fallback_weighted_x += dx * radial_weight
            fallback_weighted_y += dy * radial_weight
            fallback_weight_total += radial_weight

            sector_angle = math.atan2(center_y - origin_y, center_x - origin_x)
            angle_error = ((sector_angle - preferred_angle + math.pi) % (2 * math.pi)) - math.pi
            if abs(angle_error) > sector_half_width:
                continue

            alignment = math.cos(angle_error)
            sector_weight = radial_weight * (1.0 + 0.5 * alignment)
            if is_boundary_safe:
                sector_weighted_x += dx * sector_weight
                sector_weighted_y += dy * sector_weight
                sector_weight_total += sector_weight
            else:
                unsafe_sector_weighted_x += dx * sector_weight
                unsafe_sector_weighted_y += dy * sector_weight
                unsafe_sector_weight_total += sector_weight

        if local_dist_sq != float("inf"):
            return local_dx, local_dy

        if sector_weight_total > 0:
            return sector_weighted_x / sector_weight_total, sector_weighted_y / sector_weight_total

        if unsafe_sector_weight_total > 0:
            return (
                unsafe_sector_weighted_x / unsafe_sector_weight_total,
                unsafe_sector_weighted_y / unsafe_sector_weight_total,
            )

        if fallback_weight_total > 0:
            return fallback_weighted_x / fallback_weight_total, fallback_weighted_y / fallback_weight_total

        return 0.0, 0.0

    def obstacles_in_quadrants(self, point, area, obstacles):
        """Find distancs to obstacles in the 4 quadrants."""
        pygame_area = self.world.area  # Needed for coordinate conversion
        px, py = world_ref_to_game_ref((point.x, point.y), pygame_area)

        minx, miny, maxx, maxy = area.bounds

        # Use true search-area edge distances so the drone always knows how much
        # room remains in each direction, not only when a boundary is within
        # the local sensing radius.
        distances = {
            "right": maxx - point.x,
            "up": maxy - point.y,
            "left": point.x - minx,
            "down": point.y - miny,
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

        result = [dist for dist in distances.values()]

        return result


def dist(x1, y1, x2, y2):
    """Distance between two points."""
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def closest_point_in_rect(rect, point):
    """Find the closest point in a rectangle to a given point.

    :params:
    ----------
    rect (pygame.Rect): The rectangle.
    point (tuple): The point (x, y).

    :return:
    -------
    tuple: The closest point (x, y) in the rectangle to the given point.

    """
    closest_x = max(rect.left, min(point[0], rect.right))
    closest_y = max(rect.top, min(point[1], rect.bottom))

    to_closest_point = np.array([closest_x, closest_y])

    return to_closest_point
