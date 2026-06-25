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
                # print("starting pos set")
                self.has_custom_starting_pos = True
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
                else:
                    self.starting_pos = drone_config.get("drones_starting_pos")[drone_id]
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
        self.detected = set() # 이 drone이 탐색한 좌표
        self.found_goal = False # 이 drone이 goal을 찾았는지 여부

        self.world = world
        self.sensor.update_poly_points((self.rect.centerx, self.rect.centery), self.orientation, self.altitude)

        self.drone_config = drone_config
        self.number_of_drones = number_of_drones

        if drone_config.get("discrete_action_space", False):
            self.action_space = gymnasium.spaces.Discrete(5)
            self.discrete_action_space = True
        else:
            self.action_space = gymnasium.spaces.Box(low=-self.max_speed, high=self.max_speed, shape=(3,))
            self.discrete_action_space = False

        self.base_obs_len = 6 + self.number_of_drones * 2
        # add one extra slot for normalized self.id in the vector observation
        self.observation_space = gymnasium.spaces.Dict(
            {
                "vector": gymnasium.spaces.Box(
                    low=-19.0,
                    high=19.0,
                    shape=(self.base_obs_len + 3,),
                    dtype=np.float32,
                ),
                "relative_map": gymnasium.spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.RELATIVE_MAP_SIZE, self.RELATIVE_MAP_SIZE, 3),
                    dtype=np.float32,
                ),
            }
        )

        """
        action space: [wanted vx, wanted vy, recharge] where recharge is mapped to a bool for trying to recharge.
        """
        self.charge_level = self.max_charge
        self.charging = False
        self.charging_point = (0, 0)
        self.goto_pos = [0, 0]

        self.trajectory_len = 3

    def reset(self, seed=None, options=None):
        """Reset drone."""
        self.charge_level = self.max_charge
        self.charging = False
        self.charging_point = (0, 0)
        self.out_of_bound = False
        self.carried_targets = 0
        self.found_goal = False
        self.vx = 0.0
        self.vy = 0.0
        self.accel_x = 0.0
        self.accel_y = 0.0
        self.previous_accel = [0.0, 0.0]
        self.orientation = 0.0
        self.goto_pos = [0, 0]
        self.detected = set()
        self.latest_detected = np.empty((0, 2), dtype=np.int32)
        self.sensor.update_poly_points((self.rect.centerx, self.rect.centery), self.orientation, self.altitude)

    def sync_pose_state(self):
        """Sync state derived from the current spawn pose."""
        self.trajectory = [(self.x, self.y)] * self.trajectory_len
        self.sensor.update_poly_points((self.rect.centerx, self.rect.centery), self.orientation, self.altitude)

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

    def update(self, area, world, action, found_goal):
        """Update drone."""
        # if self.found_goal or found_goal:
        #     return
        if self.discrete_action_space:
            action = self.discrete_to_continuous(action)

        if action[2] > 0:  # drone tries to recharge
            # self.closest_point_in_base = closest_point_in_rect(world.base, self.rect.center)
            # can_charge = self.charging_distance > dist(
            #     self.rect.x,
            #     self.rect.y,
            #     self.closest_point_in_base[0],
            #     self.closest_point_in_base[1],
            # )
            # if can_charge:
            #     if (self.closest_point_in_base == self.rect.center).all():
            #         self.charging_point = world.base.center
            #     else:
            #         self.charging_point = self.closest_point_in_base  # game ref
            #     self.charging = True
            #     self.charge_level += 9
            #     if self.charge_level > self.max_charge:
            #         self.charge_level = self.max_charge
            #     # print("charging at base!")
            # else:  # check if provisioner near
            #     for id, coords in world.provisioners.items():
            #         if self.charging_distance > dist(self.x, self.y, coords[0], coords[1]):
            #             self.charging_point = world_ref_to_game_ref(coords, area)
            #             self.charging = True
            #             self.charge_level += 9
            #             if self.charge_level > self.max_charge:
            #                 self.charge_level = self.max_charge
            pass
        else:
            self.charging = False

        if not self.charging:  # drone wants to move (only if not currently charging)
            if self.charge_level > 0:
                self.charge_level -= 1
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
                newpos = self.rect.copy()
                rect_pos = world_ref_to_game_ref([self.x, self.y], world.area)
                self.update_detected_area(self.sensing_range)
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
            else:
                # print("drone has no energy!")
                pass

        self.sensor.update_poly_points((self.rect.centerx, self.rect.centery), self.orientation, self.altitude)

    def discrete_to_continuous(self, action):
        """Convert discrete action to box space."""
        if action == 0:
            out = [0, 0, 1]
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
        return True

    def observe(self, world, agents, poi) -> dict[str, np.ndarray]:
        """Observe the world."""
        try:
            minx, miny, maxx, maxy = world.search_area.bounds
            norm = math.hypot(maxx - minx, maxy - miny)
        except Exception:
            norm = 1.0
        if norm <= 0:
            norm = 1.0

        distances = self.obstacles_in_quadrants(Point(self.x, self.y), world.search_area, world.obstacles) # 시야 안의 obstacle, boundary까지의 거리
        norm_dists = [float(np.clip(d / max(self.sensing_range, 1e-6), 0.0, 1.0)) for d in distances]

        agents_rel_pos = []
        index = 0
        for ag in agents:
            if isinstance(ag, Drone):
                dx = float(np.clip((ag.x - self.x) / norm, -1.0, 1.0))
                dy = float(np.clip((ag.y - self.y) / norm, -1.0, 1.0))
                if index == 0 and dx == 0 and dy == 0: # 자신인 경우 pass
                    index += 1
                    continue
                agents_rel_pos.extend([dx, dy]) # 다른 에이전트의 상대 위치

        obs_raw = norm_dists + agents_rel_pos
        obs = np.array(obs_raw, dtype=np.float32)
        if obs.size < self.base_obs_len:
            obs = np.pad(obs, (0, self.base_obs_len - obs.size), mode="constant", constant_values=0.0)
        elif obs.size > self.base_obs_len:
            obs = obs[: self.base_obs_len]

        goal_relative = self.build_goal_relative_sector(world) if world.goal_known else np.zeros(2, dtype=np.float32)
        vector_obs = np.concatenate((obs, goal_relative)).astype(np.float32, copy=False)
        # normalized id in [0,1]
        try:
            id_norm = float(self.id) / max(1, self.number_of_drones)
        except Exception:
            id_norm = 0.0
        vector_obs = np.concatenate((vector_obs, np.array([id_norm], dtype=np.float32)))
        return {
            "vector": vector_obs,
            "relative_map": self.build_relative_sector_map(world), # 탐색률, map 경계, obstacle 위치 표시
        }

    def obstacles_in_quadrants(self, point, area, obstacles):
        """Find distancs to obstacles in the 4 quadrants."""
        pygame_area = self.world.area  # Needed for coordinate conversion
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

        self.update_boundary_distance_channels(point, area, self.sensing_range, distances)

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
