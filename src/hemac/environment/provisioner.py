"""Provisioner module."""

import math
import os
from random import randrange

import gymnasium
import numpy as np
import pygame
from .sensors import ForwardFacingCamera, Sensor

from .base_agent import BaseAgent
from hemac.helpers.helper import world_ref_to_game_ref
from .drone import Drone


class Provisioner(BaseAgent):
    """Provisioner class."""

    def __init__(
        self,
        provisioner_config,
        world,
        randomizer,
        provisioner_id=-1,
        time_factor=1,
        render_ratio: float = 1,
        sensor: Sensor = ForwardFacingCamera(hfov=np.pi / 2, sensing_range=25),
    ):
        """Overwrite constructor."""
        super().__init__()

        self.id = provisioner_id
        self.time_factor = time_factor
        self.starting_pos = None
        self.world = world
        ui_dims = 20
        dims = [1, 1]
        # rad, positive angle counter-clockwise (note that the world referential is the opposite: y-axis down)
        self.orientation = np.pi / 4
        self.altitude = 100
        self.steering_angle = np.pi / 8  # angular velocity
        self.sensor = sensor

        if provisioner_config:
            ui_dims = provisioner_config.get("ui_dimension", 20)

            dim_cfg = provisioner_config.get("dimension") if provisioner_config.get("dimension") else [32, 44]
            dims_meters = [dim_cfg[0] / 100, dim_cfg[1] / 100]

            dims = [math.ceil(dims_meters[0]), math.ceil(dims_meters[1])]
            self.max_speed = provisioner_config.get("max_speed", 8)
            self.max_charge = provisioner_config.get("max_charge", 9999)
            self.max_thrust = provisioner_config.get("max_thrust", 4)
            self.altitude = provisioner_config.get("altitude", 0)
            self.display_goto = provisioner_config.get("display_goto", True)
        else:
            self.max_speed = 8
            self.max_thrust = 4
            self.altitude = 0
            self.max_charge = 60 * 32
            self.display_goto = True
            ui_dims = 25

        self.img = pygame.transform.scale(
            pygame.image.load(f"{os.path.dirname(__file__)}/img/vehicle.png"), [ui_dims, ui_dims]
        )
        self.base_img = self.img.copy()
        self.Provisioner_color = (randrange(0, 255), randrange(0, 255), randrange(0, 255))
        self.rect = pygame.Rect(0, 0, dims[0], dims[1])
        self.x = self.rect.x
        self.y = self.rect.y
        self.last_node: int = None
        self.current_edge: tuple = None
        self.randomizer = randomizer
        self.render_ratio = render_ratio

        self.action_space = gymnasium.spaces.Discrete(5)
        """
        action space: {0, 1, 2, 3, 4} where :
        0: stay in current position
        1: drive east to next intersection
        2: drive north to next intersection
        3: drive west to next intersection
        4: drive south to next intersection
        Note: if the road for a given action doesn't exist, the agent stays in place
        """
        self.observation_space = gymnasium.spaces.Box(low=-10000, high=10000, shape=(8,))
        """
        observation space: [goal_x, goal_y, drone_x, drone_y, east, north, west, south] where:
        (goal_x, goal_y) is the observer communication
        (drone_x, drone_y) are the relative coordinates to the closest drone (capped between [-100, 100])
        east, north, west, south are boolean values set to 1 if the provisioner can drive in the given direction
        """
        self.goto_pos = [0, 0]

    def reset(self, seed=None, options=None):
        """Reset Provisioner."""
        # start in random node
        starting_node = np.random.randint(1, len(self.world.roads["nodes"]) + 1)
        self.x = self.world.roads["nodes"][starting_node][0]
        self.y = self.world.roads["nodes"][starting_node][1]
        self.world.provisioners[self.id] = (self.x, self.y)
        (self.rect.centerx, self.rect.centery) = world_ref_to_game_ref((self.x, self.y), self.world.area)
        self.sensor.update_poly_points((self.rect.centerx, self.rect.centery), self.orientation, self.altitude)
        self.out_of_bound = False

        # initialize current node/edge
        min_dist = np.inf
        closest_node = None
        for node, coords in self.world.roads["nodes"].items():
            dist_to_node = dist(self.x, self.y, coords[0], coords[1])
            if dist_to_node < min_dist:
                min_dist = dist_to_node
                closest_node = node
        if min_dist < 5:  # at an intersection!
            self.last_node = closest_node
        else:  # on a road
            destinations = [node for node in self.world.roads["adjacency_list"][closest_node]]
            min_dist = float("inf")
            closest_edge = None
            for destination in destinations:
                a, b = self.world.roads["nodes"][closest_node], self.world.roads["nodes"][destination]
                q = closest_point_on_segment(a, b, (self.x, self.y))
                d = dist((self.x, self.y), q)

                if d < min_dist:
                    min_dist = d
                    closest_edge = (closest_node, destination)
            self.current_edge = closest_edge

    def draw(self, screen):
        """Draw Provisioner."""
        # draw Provisioner UI representation (not necessary accurate to real Provisioner dimensions)
        previous_pos = self.rect.center
        self.img = pygame.transform.rotate(
            self.base_img, self.orientation * 180 / np.pi + 90
        )  # rotate because of source image
        self.rect = self.img.get_rect()
        self.rect.center = previous_pos
        screen.blit(self.img, self.rect)
        self.sensor.draw_sensor(screen)

    def update(self, area, world, action):
        """Update Provisioner."""
        if action == 0:  # don't move
            return
        else:
            # convert action into direction according to the following:
            # 1: 0 deg (right)
            # 2: 90 deg (top)
            # 3: 180 deg (left)
            # 4: 270 deg (down)
            target_angle = (action - 1) * 90

        # if at intersection, assign action to correct paths
        last_node_coords = world.roads["nodes"][self.last_node]
        if dist(self.x, self.y, last_node_coords[0], last_node_coords[1]) < 5:
            angles = {}

            adjacent_nodes = world.roads["adjacency_list"][self.last_node]
            for neighbor in adjacent_nodes:
                x2, y2 = world.roads["nodes"][neighbor]
                angle = np.rad2deg(math.atan2(y2 - last_node_coords[1], x2 - last_node_coords[0]))
                angles[neighbor] = angle

            # Find the neighbor with the closest angle
            angle_diffs = {
                neighbor: min(abs(angle - target_angle) % 360, 360 - abs(angle - target_angle) % 360)
                for neighbor, angle in angles.items()
            }
            closest_neighbor = min(angle_diffs, key=lambda n: angle_diffs[n])

            goal = world.roads["nodes"][closest_neighbor]
            target = closest_neighbor
            # update current edge
            self.current_edge = (self.last_node, target)
        else:  # if traveling between intersections, can only continue or go back
            start, end = self.current_edge
            start_coords = world.roads["nodes"][start]
            end_coords = world.roads["nodes"][end]
            angle_to_start = (
                abs(np.rad2deg(math.atan2(start_coords[1] - self.y, start_coords[0] - self.x)) - target_angle) % 360
            )
            start_diff = min(angle_to_start, 360 - angle_to_start)
            angle_to_end = (
                abs(np.rad2deg(math.atan2(end_coords[1] - self.y, end_coords[0] - self.x)) - target_angle) % 360
            )
            end_diff = min(angle_to_end, 360 - angle_to_end)
            if start_diff < end_diff:
                goal = start_coords  # go back
                target = start
            else:
                goal = end_coords  # continue
                target = end
            self.last_node = target
            # print(f"on edge {self.current_edge}, choices: {self.current_edge}, going to {target}, {goal}")

        angle_to_goal = math.atan2(goal[1] - self.y, goal[0] - self.x)
        dist_to_goal = dist(self.x, self.y, goal[0], goal[1])
        steer_command = ((angle_to_goal - self.orientation + np.pi) % (2 * np.pi)) - np.pi
        speed = min(self.max_speed, dist_to_goal / 2)
        if steer_command > 3 * np.pi / 5:
            steer_command = steer_command - np.pi  # complement angle
            speed *= -1
        elif steer_command < -3 * np.pi / 5:
            steer_command = steer_command + np.pi  # complement angle
            speed *= -1
        self.orientation += np.clip(steer_command, -self.steering_angle, self.steering_angle)
        self.orientation = self.orientation % (2 * np.pi)

        self.x += speed * np.cos(self.orientation) * self.time_factor
        self.y += speed * np.sin(self.orientation) * self.time_factor
        world.provisioners[self.id] = (self.x, self.y)

        newpos = self.rect.copy()
        rect_pos = world_ref_to_game_ref([self.x, self.y], world.area)
        newpos.centerx = rect_pos[0]
        newpos.centery = rect_pos[1]

        # make sure the players stay inside the screen
        if area.contains(newpos):
            self.rect = newpos
        else:
            self.out_of_bound = True

        self.sensor.update_poly_points((self.rect.centerx, self.rect.centery), self.orientation, self.altitude)

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

    def observe(self, world, agents, goals) -> np.array:
        """Observe world."""
        goal_x, goal_y = world.observer_communication
        dist_closest_drone = np.inf
        drone_x = self.x
        drone_y = self.y
        for agent in agents:
            if isinstance(agent, Drone):
                distance = dist(self.x, self.y, agent.x, agent.y)
                if distance < dist_closest_drone:
                    dist_closest_drone = distance
                    drone_x = agent.x
                    drone_y = agent.y

        drone_x = np.clip((drone_x - self.x), -100, 100)
        drone_y = np.clip((drone_y - self.y), -100, 100)

        possible_directions = [0, 0, 0, 0]
        last_node_coords = world.roads["nodes"][self.last_node]
        if dist(self.x, self.y, last_node_coords[0], last_node_coords[1]) < 5:  # at an intersection
            adjacent_nodes = world.roads["adjacency_list"][self.last_node]
            for neighbor in adjacent_nodes:
                x2, y2 = world.roads["nodes"][neighbor]
                angle = np.rad2deg(math.atan2(y2 - last_node_coords[1], x2 - last_node_coords[0]))
                for i, direction in enumerate((0, 90, 180, 270)):
                    if angle_difference(angle, direction) < 15:
                        possible_directions[i] = 1

        else:  # if traveling between intersections, can only continue or go back
            start, end = self.current_edge
            start_coords = world.roads["nodes"][start]
            end_coords = world.roads["nodes"][end]
            angle_to_start = np.rad2deg(math.atan2(start_coords[1] - self.y, start_coords[0] - self.x))
            angle_to_end = np.rad2deg(math.atan2(end_coords[1] - self.y, end_coords[0] - self.x))
            for i, direction in enumerate((0, 90, 180, 270)):
                if angle_difference(angle_to_start, direction) < 30:
                    possible_directions[i] = 1
                elif angle_difference(angle_to_end, direction) < 30:
                    possible_directions[i] = 1

        obs = [goal_x, goal_y, drone_x, drone_y] + possible_directions
        # print(f"provisionver obs: {obs}")
        return np.array(obs, np.float32)

    def get_location_obs(self, world, agents) -> np.array:
        """Return the location of the Provisioner relative to boundaries and base.

        Args:
        ----
            world (_type_): world of simulation.
            agents (_type_): list of Pygame Sprites

        Returns:
        -------
            np.array: Observation

        """
        return np.array([0, 0, 0, 0], np.float32)


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


def closest_point_on_segment(a, b, p):
    """Find the closest point on segment ab to point p."""
    a = np.array(a)
    b = np.array(b)
    p = np.array(p)

    ab = b - a  # Vector ab
    ap = p - a  # Vector ap

    # Project ap onto ab to find the closest point
    t = np.dot(ap, ab) / np.dot(ab, ab)

    # Clamp t to stay within segment [0,1]
    t = max(0, min(1, t))

    # Compute closest point on the segment
    closest_point = a + t * ab
    return tuple(closest_point)


def angle_difference(theta1, theta2):
    """Compute the diference between 2 angles in deg."""
    return abs(((theta2 - theta1 + 180) % (360)) - 180)
