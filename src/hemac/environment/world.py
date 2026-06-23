"""World module."""

import os
from datetime import datetime, UTC

import copy

import numpy as np
import pygame

from shapely.geometry import Point, Polygon

from hemac.helpers.helper import game_ref_to_world_ref, world_ref_to_game_ref, sample_point_in_polygon


class World(pygame.sprite.Sprite):
    """World class."""

    BASE_OBSTACLE_CLEARANCE = 150

    def __init__(
        self,
        game_area: pygame.Rect,
        geofence_area: list,
        search_area: Polygon,
        randomizer: np.random.Generator,
        time_factor: int = 1,
        initial_prior: bool = False,
    ):
        """Overwrite constructor."""
        self.area = game_area
        self.bg_image = pygame.transform.scale(
            pygame.image.load(f"{os.path.dirname(__file__)}/img/world_forest.jpg"), self.area.size
        )
        self.spawn_max_tries = 10000
        self.obstacles = []
        self.base = pygame.Rect(0, 0, 100, 100)
        self.search_area = search_area
        self.geofence_area = [world_ref_to_game_ref(coords, self.area) for coords in geofence_area]
        self.displayed_search_area = [
            world_ref_to_game_ref(coords, self.area) for coords in self.search_area.exterior.coords
        ]
        self.basex = self.base.x
        self.basey = self.base.y
        self.provisioners = {}
        self.randomizer = randomizer
        self.time_factor = time_factor
        self.timestep = 0
        self.simulation_start_time = datetime.now(UTC).timestamp()  # set to current timestamp
        self.observer_communication = [0.0, 0.0]
        self.goal_known = False
        self.goal_position = None
        self.coverage_grid_size = 20
        self.coverage_cell_width = self.area.width / self.coverage_grid_size
        self.coverage_cell_height = self.area.height / self.coverage_grid_size
        self.coverage_cell_area = self.coverage_cell_width * self.coverage_cell_height
        self.detected = set()
        self.coverage_counts = np.zeros((self.coverage_grid_size, self.coverage_grid_size), dtype=np.int32)
        self.coverage_map = np.zeros((self.coverage_grid_size, self.coverage_grid_size), dtype=np.float32)
        self.search_mask = np.zeros((self.coverage_grid_size, self.coverage_grid_size), dtype=np.float32)
        self.obstacle_map = np.zeros((self.coverage_grid_size, self.coverage_grid_size), dtype=np.float32)
        for grid_x in range(self.coverage_grid_size):
            for grid_y in range(self.coverage_grid_size):
                cell_center = (
                    (grid_x + 0.5) * self.coverage_cell_width,
                    (grid_y + 0.5) * self.coverage_cell_height,
                )
                self.search_mask[grid_y, grid_x] = float(self.search_area.covers(Point(cell_center)))

        # Road network data TODO: random generation
        nodes = {
            1: (200, 200),
            2: (200, 500),
            3: (300, 500),
            4: (400, 500),
            5: (400, 650),
            6: (400, 300),
            7: (600, 650),
        }

        # Each edge connects two nodes
        edges = [(1, 2), (2, 3), (3, 4), (4, 5), (4, 6), (5, 7)]
        adjacency_list = build_adjacency_dict(nodes, edges)
        self.roads = {"nodes": nodes, "edges": edges, "adjacency_list": adjacency_list}

    def reset(self, poi_list, seed=None, options=None):
        """Reset world."""
        self.timestep = 0
        self.observer_communication = [0.0, 0.0]
        self.goal_known = False
        self.goal_position = (poi_list[0].x, poi_list[0].y) if poi_list else None
        self.detected.clear()
        self.coverage_counts.fill(0)
        self.coverage_map.fill(0.0)
        self.obstacle_map.fill(0.0)
        self.base.center = (150, 150)
        # collision = True
        # while collision:
        #     self.base.center = world_ref_to_game_ref(
        #         sample_point_in_polygon(self.search_area, self.randomizer), self.area
        #     )
        #     for start_id, end_id in self.roads["edges"]:
        #         start = world_ref_to_game_ref(self.roads["nodes"][start_id], self.area)
        #         end = world_ref_to_game_ref(self.roads["nodes"][end_id], self.area)
        #         collision = self.base.clipline(start, end)
        #         if collision:
        #             break
        # TODO: re spawn base, roads and obstacles here?

    def register_detected_points(self, points):
        """Update the cached coverage map with newly detected coordinates."""
        cell_updates = {}
        for x, y in points:
            point = (int(x), int(y))
            if point in self.detected:
                continue
            if not (0 <= point[0] < self.area.width and 0 <= point[1] < self.area.height):
                continue

            self.detected.add(point)
            grid_x = min(int(point[0] / self.coverage_cell_width), self.coverage_grid_size - 1)
            grid_y = min(int(point[1] / self.coverage_cell_height), self.coverage_grid_size - 1)
            key = (grid_y, grid_x)
            cell_updates[key] = cell_updates.get(key, 0) + 1

        for (grid_y, grid_x), count in cell_updates.items():
            new_total = self.coverage_counts[grid_y, grid_x] + count
            self.coverage_counts[grid_y, grid_x] = new_total
            self.coverage_map[grid_y, grid_x] = min(new_total / self.coverage_cell_area, 1.0)

        return len(self.detected)

    def clear_obstacles(self):
        """Remove all obstacles from the world."""
        self.obstacles.clear()  # Clear the list of obstacles
        self.obstacle_map.fill(0.0)

    @staticmethod
    def _rect_distance(rect_a: pygame.Rect, rect_b: pygame.Rect) -> float:
        """Return the minimum Euclidean distance between two rectangles."""
        dx = max(rect_a.left - rect_b.right, rect_b.left - rect_a.right, 0)
        dy = max(rect_a.top - rect_b.bottom, rect_b.top - rect_a.bottom, 0)
        return float(np.hypot(dx, dy))

    def _grid_to_game_rect(self, grid_x: int, grid_y: int) -> pygame.Rect:
        """Convert a coverage-grid cell to the corresponding game-space rect."""
        left = int(round(grid_x * self.coverage_cell_width))
        right = int(round((grid_x + 1) * self.coverage_cell_width))
        top = int(round(self.area.height - (grid_y + 1) * self.coverage_cell_height))
        bottom = int(round(self.area.height - grid_y * self.coverage_cell_height))
        return pygame.Rect(left, top, max(right - left, 1), max(bottom - top, 1))

    def _rebuild_obstacle_map(self) -> None:
        """Refresh the obstacle occupancy grid used by agent observations."""
        self.obstacle_map.fill(0.0)
        for grid_x in range(self.coverage_grid_size):
            for grid_y in range(self.coverage_grid_size):
                if self.search_mask[grid_y, grid_x] <= 0.0:
                    continue
                cell_rect = self._grid_to_game_rect(grid_x, grid_y)
                if any(obstacle.colliderect(cell_rect) for obstacle in self.obstacles):
                    self.obstacle_map[grid_y, grid_x] = 1.0

    def generate_obstacles(self, n_obstacles):
        """Generate random obstacles."""
        for i in range(n_obstacles):
            w, h = self.randomizer.integers(10, 150), self.randomizer.integers(10, 150)
            obstacle = pygame.Rect(0, 0, w, h)
            valid_coord = False
            while not valid_coord:
                obstacle.center = world_ref_to_game_ref(
                    sample_point_in_polygon(self.search_area, self.randomizer), self.area
                )
                road_collision = True
                for start_id, end_id in self.roads["edges"]:
                    start = world_ref_to_game_ref(self.roads["nodes"][start_id], self.area)
                    end = world_ref_to_game_ref(self.roads["nodes"][end_id], self.area)
                    road_collision = obstacle.clipline(start, end)
                    if road_collision:
                        break
                base_clearance_ok = self._rect_distance(obstacle, self.base) > self.BASE_OBSTACLE_CLEARANCE
                if base_clearance_ok and not road_collision:
                    valid_coord = True
            self.obstacles.append(obstacle)
        self._rebuild_obstacle_map()

    def draw(self, screen):
        """Draw world."""
        screen.blit(self.bg_image, (0, 0))
        font = pygame.font.SysFont("Trebuchet MS", 25)
        # find the simulation date
        simulation_current_time = self.simulation_start_time + self.timestep * self.time_factor
        simulation_current_date = datetime.fromtimestamp(simulation_current_time).astimezone().isoformat()
        date_font = font.render(simulation_current_date, True, (0, 51, 0))
        screen.blit(date_font, [5, 5])

        if self.geofence_area:
            pygame.draw.polygon(screen, (78, 0, 200), self.geofence_area, 2)
        pygame.draw.polygon(screen, (30, 30, 0), self.displayed_search_area)
        pygame.draw.polygon(screen, (222, 0, 0), self.displayed_search_area, 2)

        # Base
        pygame.draw.rect(screen, (50, 50, 150), self.base)

        # Roads
        for start_id, end_id in self.roads["edges"]:
            start = world_ref_to_game_ref(self.roads["nodes"][start_id], self.area)
            end = world_ref_to_game_ref(self.roads["nodes"][end_id], self.area)
            draw_road(start, end, screen)

        # Obstacles
        for obstacle in self.obstacles:
            pygame.draw.rect(screen, (150, 0, 0), obstacle)

    def update(self, area):
        """Update world."""
        # increase timestep counter to know how many step were run
        self.timestep += 1
        pass

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

    def spawn_asset(self, asset, other_assets, avoid_world_obstacles=False, set_real_coordinates=False):
        """Spawned asset."""
        step = 0
        found_point = False
        temp_rect = copy.deepcopy(asset.rect)
        while step < self.spawn_max_tries and not found_point:
            step += 1
            (x, y) = sample_point_in_polygon(self.search_area, self.randomizer)
            temp_rect.x, temp_rect.y = world_ref_to_game_ref((x, y), self.area)
            safe = True
            # loop over world obstacles
            if avoid_world_obstacles:
                for obstacle in self.obstacles:
                    if obstacle.colliderect(temp_rect):
                        safe = False

            # loop over other rects
            for obstacle in other_assets:
                if obstacle.rect.colliderect(temp_rect):
                    safe = False

            if safe:
                found_point = True
                asset.rect = temp_rect
                if set_real_coordinates:
                    [asset.x, asset.y] = game_ref_to_world_ref(asset.rect.center, self.area)

        if step == self.spawn_max_tries:
            print(f"couldn't find valid spot for asset {asset}!")


def draw_road(start, end, screen):
    """Draw road network."""
    # Draw the road as a thick gray line
    pygame.draw.line(screen, (50, 50, 50), start, end, 20)

    # Add dashed center line (yellow)
    dash_length = 15
    total_length = ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5
    num_dashes = int(total_length // (dash_length * 2))

    for i in range(num_dashes):
        # Calculate dash positions
        t_start = i * 2 * dash_length / total_length
        t_end = (i * 2 + 1) * dash_length / total_length

        dash_start = (start[0] + (end[0] - start[0]) * t_start, start[1] + (end[1] - start[1]) * t_start)
        dash_end = (start[0] + (end[0] - start[0]) * t_end, start[1] + (end[1] - start[1]) * t_end)

        pygame.draw.line(screen, (255, 204, 0), dash_start, dash_end, 2)


def build_adjacency_dict(nodes, edges):
    """Compute adjacency list of the road network."""
    adjacency_dict = {node: set() for node in nodes}
    for u, v in edges:
        adjacency_dict[u].add(v)
        adjacency_dict[v].add(u)  # Undirected graph
    return adjacency_dict
