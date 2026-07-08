"""World module."""

import os
from functools import lru_cache
from datetime import datetime, UTC

import copy

import numpy as np
import pygame

from shapely.geometry import Point, Polygon
from shapely.prepared import prep
from shapely import contains_xy

from hemac.helpers.helper import game_ref_to_world_ref, world_ref_to_game_ref, sample_point_in_polygon


class World(pygame.sprite.Sprite):
    """World class."""

    BASE_OBSTACLE_CLEARANCE = 150
    OBSTACLE_WARNING_RADIUS = 60
    OBSTACLE_MOVE_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))
    OBSTACLE_DIRECTION_HOLD_STEPS = 50
    OBSTACLE_SPEED_HOLD_STEPS = 50
    OBSTACLE_MIN_SPEED = 3
    OBSTACLE_MAX_SPEED = 7
    OBSTACLE_WARNING_FILL = (255, 80, 80, 55)
    OBSTACLE_WARNING_OUTLINE = (255, 120, 120, 150)
    OBSERVED_OBSTACLE_DECAY = 0.98

    def __init__(
        self,
        game_area: pygame.Rect,
        geofence_area: list,
        search_area: Polygon,
        randomizer: np.random.Generator,
        time_factor: int = 1,
        initial_prior: bool = False,
        obstacle_min_speed: int | None = None,
        obstacle_max_speed: int | None = None,
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
        self.search_bounds = tuple(float(v) for v in self.search_area.bounds)
        self.search_area_is_rect = self._is_axis_aligned_rect(self.search_area)
        self.prepared_search_area = None if self.search_area_is_rect else prep(self.search_area)
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
        self.obstacle_min_speed = self.OBSTACLE_MIN_SPEED
        self.obstacle_max_speed = self.OBSTACLE_MAX_SPEED
        self.set_obstacle_speed_range(
            self.obstacle_min_speed if obstacle_min_speed is None else obstacle_min_speed,
            self.obstacle_max_speed if obstacle_max_speed is None else obstacle_max_speed,
        )
        self.simulation_start_time = datetime.now(UTC).timestamp()  # set to current timestamp
        self.observer_communication = [0.0, 0.0]
        self.goal_known = False
        self.goal_position = None
        self.coverage_grid_size = 20
        self.coverage_cell_width = self.area.width / self.coverage_grid_size
        self.coverage_cell_height = self.area.height / self.coverage_grid_size
        self.coverage_cell_area = self.coverage_cell_width * self.coverage_cell_height
        self.detected = set()
        self.detected_count = 0
        self.detected_mask = np.zeros((self.area.height, self.area.width), dtype=bool)
        self.search_pixel_mask = self._build_search_pixel_mask()
        self.search_pixel_mask_float = self.search_pixel_mask.astype(np.float32, copy=False)
        self.search_integral = self._build_integral_image(self.search_pixel_mask_float)
        self.obstacle_pixel_mask = np.zeros((self.area.height, self.area.width), dtype=np.float32)
        self.warning_pixel_mask = np.zeros((self.area.height, self.area.width), dtype=np.float32)
        self.observed_obstacle_pixel_mask = np.zeros((self.area.height, self.area.width), dtype=np.float32)
        self.observed_warning_pixel_mask = np.zeros((self.area.height, self.area.width), dtype=np.float32)
        self.coverage_counts = np.zeros((self.coverage_grid_size, self.coverage_grid_size), dtype=np.int32)
        self.coverage_map = np.zeros((self.coverage_grid_size, self.coverage_grid_size), dtype=np.float32)
        self.observation_coverage_map = np.zeros((self.coverage_grid_size, self.coverage_grid_size), dtype=np.float32)
        self.search_mask = np.zeros((self.coverage_grid_size, self.coverage_grid_size), dtype=np.float32)
        self.obstacle_map = np.zeros((self.coverage_grid_size, self.coverage_grid_size), dtype=np.float32)
        self.explored_obstacle_map = np.zeros((self.coverage_grid_size, self.coverage_grid_size), dtype=np.float32)
        self.warning_range_map = np.zeros((self.coverage_grid_size, self.coverage_grid_size), dtype=np.float32)
        self.explored_warning_range_map = np.zeros((self.coverage_grid_size, self.coverage_grid_size), dtype=np.float32)
        self.coverage_counts_flat = self.coverage_counts.reshape(-1)
        self.coverage_map_flat = self.coverage_map.reshape(-1)
        self.relative_pad = self.coverage_grid_size
        self.cell_game_rects = [
            [self._grid_to_game_rect(grid_x, grid_y) for grid_x in range(self.coverage_grid_size)]
            for grid_y in range(self.coverage_grid_size)
        ]
        for grid_x in range(self.coverage_grid_size):
            for grid_y in range(self.coverage_grid_size):
                self.search_mask[grid_y, grid_x] = self._cell_search_area_ratio(grid_x, grid_y)
        self.padded_search_mask = np.pad(
            self.search_mask,
            ((self.relative_pad, self.relative_pad), (self.relative_pad, self.relative_pad)),
            mode="constant",
            constant_values=0.0,
        )
        self.padded_coverage_map = np.pad(
            self.coverage_map,
            ((self.relative_pad, self.relative_pad), (self.relative_pad, self.relative_pad)),
            mode="constant",
            constant_values=0.0,
        )
        self.padded_explored_obstacle_map = np.pad(
            self.explored_obstacle_map,
            ((self.relative_pad, self.relative_pad), (self.relative_pad, self.relative_pad)),
            mode="constant",
            constant_values=0.0,
        )
        self.padded_explored_warning_range_map = np.pad(
            self.explored_warning_range_map,
            ((self.relative_pad, self.relative_pad), (self.relative_pad, self.relative_pad)),
            mode="constant",
            constant_values=0.0,
        )
        self.coverage_x_edges = np.rint(
            np.linspace(0.0, float(self.area.width), self.coverage_grid_size + 1)
        ).astype(np.int32)
        self.coverage_y_edges = np.rint(
            np.linspace(0.0, float(self.area.height), self.coverage_grid_size + 1)
        ).astype(np.int32)
        self.coverage_x_edges[0] = 0
        self.coverage_y_edges[0] = 0
        self.coverage_x_edges[-1] = self.area.width
        self.coverage_y_edges[-1] = self.area.height
        self._observed_obstacle_integral = None
        self._observed_warning_integral = None
        self.actual_obstacle_confidences = np.zeros((0,), dtype=np.float32)
        self.obstacle_move_direction_indices = np.zeros((0,), dtype=np.int8)
        self.obstacle_move_steps_remaining = np.zeros((0,), dtype=np.int32)
        self.obstacle_move_speeds = np.zeros((0,), dtype=np.int16)
        self.obstacle_speed_steps_remaining = np.zeros((0,), dtype=np.int32)
        self._last_obstacle_confidence_decay_timestep = -1
        self.observed_obstacle_confidences = {}
        self.last_observed_rect_keys_by_obstacle = []
        self.observed_obstacle_rects = []
        self.obstacle_bounds = np.empty((0, 4), dtype=np.int32)
        self.obstacle_centers = np.empty((0, 2), dtype=np.float32)
        self.obstacle_keys = set()
        self.revealed_warning_obstacles = np.zeros((0,), dtype=bool)
        minx, miny, maxx, maxy = self.search_bounds
        self.search_diagonal = float(np.hypot(maxx - minx, maxy - miny))

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

    @staticmethod
    def _build_integral_image(mask: np.ndarray) -> np.ndarray:
        """Return a summed-area table with one-cell zero padding."""
        float_mask = np.asarray(mask, dtype=np.float32)
        return np.pad(
            float_mask.cumsum(axis=0).cumsum(axis=1),
            ((1, 0), (1, 0)),
            mode="constant",
            constant_values=0.0,
        )

    @staticmethod
    def _integral_box_sums(integral: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray) -> np.ndarray:
        """Return per-bin sums from a summed-area table."""
        x0 = x_edges[:-1]
        x1 = x_edges[1:]
        y0 = y_edges[:-1]
        y1 = y_edges[1:]
        return (
            integral[y1[:, None], x1[None, :]]
            - integral[y0[:, None], x1[None, :]]
            - integral[y1[:, None], x0[None, :]]
            + integral[y0[:, None], x0[None, :]]
        )

    @staticmethod
    @lru_cache(maxsize=None)
    def _disk_offsets(radius: int) -> np.ndarray:
        """Cache integer offsets inside a circular warning radius."""
        radius = max(0, int(radius))
        coords = np.arange(-radius, radius + 1, dtype=np.int32)
        grid_x, grid_y = np.meshgrid(coords, coords, indexing="xy")
        mask = (grid_x * grid_x) + (grid_y * grid_y) <= radius * radius
        offsets = np.column_stack((grid_x[mask], grid_y[mask])).astype(np.int32, copy=False)
        offsets.setflags(write=False)
        return offsets

    def _rect_arrays(self, rects) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return left/top/width/height arrays for a rect collection."""
        rects = list(rects)
        if not rects:
            empty = np.empty((0,), dtype=np.int32)
            return empty, empty, empty, empty

        left = np.fromiter((rect.left for rect in rects), dtype=np.int32, count=len(rects))
        top = np.fromiter((rect.top for rect in rects), dtype=np.int32, count=len(rects))
        width = np.fromiter((rect.width for rect in rects), dtype=np.int32, count=len(rects))
        height = np.fromiter((rect.height for rect in rects), dtype=np.int32, count=len(rects))
        return left, top, width, height

    def _all_unit_rects(self, rects) -> bool:
        """Return True when every rect is a single world pixel."""
        _, _, width, height = self._rect_arrays(rects)
        if width.size == 0:
            return True
        return bool(np.all(width == 1) and np.all(height == 1))

    def _paint_disks_on_world_mask(
        self,
        mask: np.ndarray,
        centers_x: np.ndarray,
        centers_y: np.ndarray,
        radius: float,
        values: np.ndarray | None = None,
    ) -> None:
        """Rasterize many circular warning zones into a world-space pixel mask."""
        if centers_x.size == 0:
            return

        offsets = self._disk_offsets(int(np.ceil(max(float(radius), 0.0))))
        if offsets.size == 0:
            return

        center_x = np.asarray(np.rint(centers_x), dtype=np.int32).reshape(-1, 1)
        center_y = np.asarray(np.rint(centers_y), dtype=np.int32).reshape(-1, 1)
        x = center_x + offsets[None, :, 0]
        y = center_y + offsets[None, :, 1]
        valid = (x >= 0) & (x < self.area.width) & (y >= 0) & (y < self.area.height)
        if not np.any(valid):
            return
        if values is None:
            belief_values = np.ones_like(x, dtype=np.float32)
        else:
            base_values = np.asarray(values, dtype=np.float32).reshape(-1, 1)
            belief_values = np.broadcast_to(base_values, x.shape)
        np.maximum.at(mask, (y[valid], x[valid]), belief_values[valid])

    def _build_search_pixel_mask(self) -> np.ndarray:
        """Rasterize the search area at world-pixel resolution."""
        mask = np.zeros((self.area.height, self.area.width), dtype=np.float32)
        if self.search_area_is_rect:
            minx, miny, maxx, maxy = self.search_bounds
            x0 = max(int(np.floor(minx)), 0)
            x1 = min(int(np.ceil(maxx)), self.area.width)
            y0 = max(int(np.floor(miny)), 0)
            y1 = min(int(np.ceil(maxy)), self.area.height)
            if x1 > x0 and y1 > y0:
                mask[y0:y1, x0:x1] = 1.0
            return mask

        pixel_x = np.arange(self.area.width, dtype=np.float32) + 0.5
        pixel_y = np.arange(self.area.height, dtype=np.float32) + 0.5
        grid_x, grid_y = np.meshgrid(pixel_x, pixel_y, indexing="xy")
        return contains_xy(self.search_area, grid_x, grid_y).astype(np.float32)

    @staticmethod
    @lru_cache(maxsize=None)
    def _visible_local_mask(grid_size: int, sensing_range: float) -> np.ndarray:
        """Cache the circular visibility mask for local observation windows."""
        grid_size = max(int(grid_size), 1)
        sensing_range = max(float(sensing_range), 1e-6)
        cell_width = (2.0 * sensing_range) / float(grid_size)
        offsets = ((np.arange(grid_size, dtype=np.float32) + 0.5) * cell_width) - sensing_range
        rel_x, rel_y = np.meshgrid(offsets, offsets, indexing="xy")
        visible_mask = ((rel_x * rel_x) + (rel_y * rel_y) <= sensing_range * sensing_range)
        visible_mask.setflags(write=False)
        return visible_mask

    def _invalidate_local_area_integrals(
        self,
        *,
        observed_obstacle: bool = False,
        observed_warning: bool = False,
    ) -> None:
        """Invalidate lazily rebuilt integral images used by local observations."""
        if observed_obstacle:
            self._observed_obstacle_integral = None
        if observed_warning:
            self._observed_warning_integral = None

    def _get_observed_obstacle_integral(self) -> np.ndarray:
        """Return the cached integral image for observed obstacle confidence."""
        if self._observed_obstacle_integral is None:
            self._observed_obstacle_integral = self._build_integral_image(self.observed_obstacle_pixel_mask)
        return self._observed_obstacle_integral

    def _get_observed_warning_integral(self) -> np.ndarray:
        """Return the cached integral image for observed warning zones inside the search area."""
        if self._observed_warning_integral is None:
            warning_search = self.observed_warning_pixel_mask * self.search_pixel_mask_float
            self._observed_warning_integral = self._build_integral_image(warning_search)
        return self._observed_warning_integral

    def build_local_area_channels(
        self,
        center_x: float,
        center_y: float,
        sensing_range: float,
        grid_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return local coverage, boundary, obstacle, and warning channels over the sensing window."""
        grid_size = max(int(grid_size), 1)
        sensing_range = max(float(sensing_range), 1e-6)

        min_x = float(center_x) - sensing_range
        max_x = float(center_x) + sensing_range
        min_y = float(center_y) - sensing_range
        max_y = float(center_y) + sensing_range

        x_edges_world = np.linspace(min_x, max_x, grid_size + 1, dtype=np.float32)
        y_edges_world = np.linspace(min_y, max_y, grid_size + 1, dtype=np.float32)
        cell_width = (2.0 * sensing_range) / float(grid_size)
        cell_area = max(cell_width * cell_width, 1e-6)

        search_integral = self.search_integral
        obstacle_integral = self._get_observed_obstacle_integral()
        warning_integral = self._get_observed_warning_integral()
        crop_x0 = max(int(np.floor(min_x)), 0)
        crop_x1 = min(int(np.ceil(max_x)), self.area.width)
        crop_y0 = max(int(np.floor(min_y)), 0)
        crop_y1 = min(int(np.ceil(max_y)), self.area.height)

        detected_search_crop = (
            self.detected_mask[crop_y0:crop_y1, crop_x0:crop_x1].astype(np.float32, copy=False)
            * self.search_pixel_mask_float[crop_y0:crop_y1, crop_x0:crop_x1]
        )
        detected_integral = self._build_integral_image(detected_search_crop)

        x_edges = np.clip(np.rint(x_edges_world).astype(np.int32), 0, self.area.width)
        y_edges = np.clip(np.rint(y_edges_world).astype(np.int32), 0, self.area.height)
        x_edges = np.maximum.accumulate(x_edges)
        y_edges = np.maximum.accumulate(y_edges)
        x_edges[-1] = min(max(int(np.rint(max_x)), 0), self.area.width)
        y_edges[-1] = min(max(int(np.rint(max_y)), 0), self.area.height)
        x_edges_crop = np.clip(np.rint(x_edges_world).astype(np.int32), crop_x0, crop_x1) - crop_x0
        y_edges_crop = np.clip(np.rint(y_edges_world).astype(np.int32), crop_y0, crop_y1) - crop_y0
        x_edges_crop = np.maximum.accumulate(x_edges_crop)
        y_edges_crop = np.maximum.accumulate(y_edges_crop)
        x_edges_crop[-1] = max(min(int(np.rint(max_x)), crop_x1) - crop_x0, 0)
        y_edges_crop[-1] = max(min(int(np.rint(max_y)), crop_y1) - crop_y0, 0)

        search_sum = self._integral_box_sums(search_integral, x_edges, y_edges)
        obstacle_sum = self._integral_box_sums(obstacle_integral, x_edges, y_edges)
        warning_sum = self._integral_box_sums(warning_integral, x_edges, y_edges)
        detected_sum = self._integral_box_sums(detected_integral, x_edges_crop, y_edges_crop)

        boundary_channel = np.clip(search_sum / cell_area, 0.0, 1.0).astype(np.float32, copy=False)
        coverage_channel = np.zeros_like(boundary_channel)
        valid_search = search_sum > 0.0
        coverage_channel[valid_search] = np.clip(
            detected_sum[valid_search] / search_sum[valid_search],
            0.0,
            1.0,
        )

        obstacle_channel = np.zeros_like(boundary_channel)
        obstacle_channel[valid_search] = np.clip(
            obstacle_sum[valid_search] / search_sum[valid_search],
            0.0,
            1.0,
        )
        obstacle_channel[detected_sum <= 0.0] = 0.0

        warning_channel = np.zeros_like(boundary_channel)
        warning_channel[valid_search] = np.clip(
            warning_sum[valid_search] / search_sum[valid_search],
            0.0,
            1.0,
        )
        visible_mask = self._visible_local_mask(grid_size, round(float(sensing_range), 6))
        coverage_channel[~visible_mask] = 0.0
        boundary_channel[~visible_mask] = 0.0
        obstacle_channel[~visible_mask] = 0.0
        warning_channel[~visible_mask] = 0.0

        return coverage_channel, boundary_channel, obstacle_channel, warning_channel

    def _paint_disk_on_world_mask(
        self,
        mask: np.ndarray,
        center_x: float,
        center_y: float,
        radius: float,
        value: float = 1.0,
    ) -> None:
        """Rasterize one circular warning zone into a world-space pixel mask."""
        radius = max(float(radius), 0.0)
        if radius <= 0.0:
            return

        x0 = max(int(np.floor(center_x - radius)), 0)
        x1 = min(int(np.ceil(center_x + radius)) + 1, self.area.width)
        y0 = max(int(np.floor(center_y - radius)), 0)
        y1 = min(int(np.ceil(center_y + radius)) + 1, self.area.height)
        if x1 <= x0 or y1 <= y0:
            return

        sample_x = np.arange(x0, x1, dtype=np.float32) + 0.5
        sample_y = np.arange(y0, y1, dtype=np.float32)[:, None] + 0.5
        dx = sample_x[None, :] - float(center_x)
        dy = sample_y - float(center_y)
        disk = np.where(
            (dx * dx) + (dy * dy) <= radius * radius,
            np.float32(value),
            np.float32(0.0),
        ).astype(np.float32, copy=False)
        mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], disk)

    def _rebuild_warning_range_map(self) -> None:
        """Aggregate the warning pixel mask into the 20x20 coverage grid."""
        self.warning_range_map.fill(0.0)
        if not np.any(self.warning_pixel_mask):
            return

        visible_warning_mask = self.warning_pixel_mask * self.search_pixel_mask_float
        warning_integral = self._build_integral_image(visible_warning_mask)
        warning_sum = self._integral_box_sums(warning_integral, self.coverage_x_edges, self.coverage_y_edges)
        self.warning_range_map[:, :] = np.clip(
            warning_sum / max(self.coverage_cell_area, 1e-6),
            0.0,
            1.0,
        )

    def _rebuild_observed_warning_map(self) -> None:
        """Aggregate revealed warning zones into the observation grid."""
        self.explored_warning_range_map.fill(0.0)
        self.padded_explored_warning_range_map.fill(0.0)
        if not np.any(self.observed_warning_pixel_mask):
            return

        visible_warning_mask = self.observed_warning_pixel_mask * self.search_pixel_mask_float
        warning_integral = self._build_integral_image(visible_warning_mask)
        warning_sum = self._integral_box_sums(warning_integral, self.coverage_x_edges, self.coverage_y_edges)
        self.explored_warning_range_map[:, :] = np.clip(
            warning_sum / max(self.coverage_cell_area, 1e-6),
            0.0,
            1.0,
        )
        self.padded_explored_warning_range_map[
            self.relative_pad : self.relative_pad + self.coverage_grid_size,
            self.relative_pad : self.relative_pad + self.coverage_grid_size,
        ] = self.explored_warning_range_map

    def reveal_warning_zones_for_obstacles(self, obstacle_indices) -> bool:
        """Reveal warning circles for newly observed obstacles."""
        if not self.obstacles:
            return False

        new_indices = []
        for obstacle_idx in obstacle_indices:
            idx = int(obstacle_idx)
            if idx < 0 or idx >= len(self.obstacles):
                continue
            if idx < len(self.revealed_warning_obstacles) and self.revealed_warning_obstacles[idx]:
                continue
            new_indices.append(idx)

        if not new_indices:
            return False

        for idx in new_indices:
            obstacle = self.obstacles[idx]
            self._paint_disk_on_world_mask(
                self.observed_warning_pixel_mask,
                center_x=float(obstacle.centerx),
                center_y=float(self.area.height - obstacle.centery),
                radius=float(self.OBSTACLE_WARNING_RADIUS),
            )
            self.revealed_warning_obstacles[idx] = True

        self._rebuild_observed_warning_map()
        return True

    def reveal_warning_zones_for_agent(self, agent) -> bool:
        """Reveal warning zones only when an agent sensor sees an obstacle."""
        if not self.obstacles or not hasattr(agent, "sensor") or agent.sensor is None:
            return False
        if not hasattr(agent.sensor, "is_point_detected"):
            return False

        visible_indices = []
        for obstacle_idx, obstacle in enumerate(self.obstacles):
            if obstacle_idx < len(self.revealed_warning_obstacles) and self.revealed_warning_obstacles[obstacle_idx]:
                continue
            if agent.sensor.is_point_detected(obstacle.center):
                visible_indices.append(obstacle_idx)

        return self.reveal_warning_zones_for_obstacles(visible_indices)

    def is_in_warning_zone(self, x: float, y: float, *, require_detected: bool = False) -> bool:
        """Return True when a world-space point lies inside any obstacle warning circle."""
        if not (0.0 <= x < float(self.area.width) and 0.0 <= y < float(self.area.height)):
            return False

        grid_x = min(max(int(np.floor(x)), 0), self.area.width - 1)
        grid_y = min(max(int(np.floor(y)), 0), self.area.height - 1)
        if require_detected and not self.detected_mask[grid_y, grid_x]:
            return False
        return bool(self.warning_pixel_mask[grid_y, grid_x] > 0.0)

    def game_rect_intersects_warning_zone(self, rect: pygame.Rect | None) -> bool:
        """Return True when any part of a game-space rect overlaps a warning zone."""
        if rect is None:
            return False

        world_left = max(int(rect.left), 0)
        world_right = min(int(rect.right), self.area.width)
        world_min_y = max(int(self.area.height - rect.bottom), 0)
        world_max_y = min(int(self.area.height - rect.top), self.area.height)
        if world_right <= world_left or world_max_y <= world_min_y:
            return False

        return bool(np.any(self.warning_pixel_mask[world_min_y:world_max_y, world_left:world_right] > 0.0))

    @staticmethod
    def _is_axis_aligned_rect(polygon: Polygon) -> bool:
        """Return True when the polygon matches an axis-aligned rectangle."""
        coords = np.asarray(polygon.exterior.coords[:-1], dtype=np.float32)
        if len(coords) != 4:
            return False
        return len(np.unique(coords[:, 0])) == 2 and len(np.unique(coords[:, 1])) == 2

    def point_in_search_area(self, x: float, y: float) -> bool:
        """Return True when a world-space point lies inside the search area."""
        minx, miny, maxx, maxy = self.search_bounds
        if self.search_area_is_rect:
            return minx <= x <= maxx and miny <= y <= maxy
        return bool(self.prepared_search_area.covers(Point((x, y))))

    def _cell_search_area_ratio(self, grid_x: int, grid_y: int) -> float:
        """Return the fraction of one coverage cell that lies inside the search area."""
        if self.search_area_is_rect:
            minx, miny, maxx, maxy = self.search_bounds
            cell_min_x = grid_x * self.coverage_cell_width
            cell_max_x = (grid_x + 1) * self.coverage_cell_width
            cell_min_y = grid_y * self.coverage_cell_height
            cell_max_y = (grid_y + 1) * self.coverage_cell_height
            overlap_x = max(0.0, min(cell_max_x, maxx) - max(cell_min_x, minx))
            overlap_y = max(0.0, min(cell_max_y, maxy) - max(cell_min_y, miny))
            return float(np.clip((overlap_x * overlap_y) / max(self.coverage_cell_area, 1e-6), 0.0, 1.0))

        cell_polygon = self._rect_to_world_polygon(self.cell_game_rects[grid_y][grid_x])
        cell_area = float(cell_polygon.area)
        if cell_area <= 0.0:
            return 0.0
        intersection_area = float(self.search_area.intersection(cell_polygon).area)
        return float(np.clip(intersection_area / cell_area, 0.0, 1.0))

    def _refresh_obstacle_cache(self) -> None:
        """Refresh vectorized obstacle bounds/centers caches."""
        if not self.obstacles:
            self.obstacle_bounds = np.empty((0, 4), dtype=np.int32)
            self.obstacle_centers = np.empty((0, 2), dtype=np.float32)
            self.obstacle_keys = set()
            self.actual_obstacle_confidences = np.zeros((0,), dtype=np.float32)
            self.last_observed_rect_keys_by_obstacle = []
            self.obstacle_move_direction_indices = np.zeros((0,), dtype=np.int8)
            self.obstacle_move_steps_remaining = np.zeros((0,), dtype=np.int32)
            self.obstacle_move_speeds = np.zeros((0,), dtype=np.int16)
            self.obstacle_speed_steps_remaining = np.zeros((0,), dtype=np.int32)
            self.revealed_warning_obstacles = np.zeros((0,), dtype=bool)
            return

        self.obstacle_bounds = np.array(
            [[rect.left, rect.right, rect.top, rect.bottom] for rect in self.obstacles],
            dtype=np.int32,
        )
        self.obstacle_centers = np.array([rect.center for rect in self.obstacles], dtype=np.float32)
        self.obstacle_keys = {self._rect_key(rect) for rect in self.obstacles}
        obstacle_count = len(self.obstacles)
        if self.actual_obstacle_confidences.shape[0] != obstacle_count:
            previous = self.actual_obstacle_confidences
            self.actual_obstacle_confidences = np.zeros((obstacle_count,), dtype=np.float32)
            copy_len = min(previous.shape[0], obstacle_count)
            if copy_len > 0:
                self.actual_obstacle_confidences[:copy_len] = previous[:copy_len]
        if len(self.last_observed_rect_keys_by_obstacle) != obstacle_count:
            previous_keys = self.last_observed_rect_keys_by_obstacle
            self.last_observed_rect_keys_by_obstacle = [None] * obstacle_count
            copy_len = min(len(previous_keys), obstacle_count)
            if copy_len > 0:
                self.last_observed_rect_keys_by_obstacle[:copy_len] = previous_keys[:copy_len]
        self._sync_obstacle_motion_state(obstacle_count)
        self.revealed_warning_obstacles = np.zeros((len(self.obstacles),), dtype=bool)

    def _sync_obstacle_motion_state(self, obstacle_count: int | None = None) -> None:
        """Resize per-obstacle motion state arrays while preserving existing values."""
        if obstacle_count is None:
            obstacle_count = len(self.obstacles)
        obstacle_count = max(int(obstacle_count), 0)

        if obstacle_count == 0:
            self.obstacle_move_direction_indices = np.zeros((0,), dtype=np.int8)
            self.obstacle_move_steps_remaining = np.zeros((0,), dtype=np.int32)
            self.obstacle_move_speeds = np.zeros((0,), dtype=np.int16)
            self.obstacle_speed_steps_remaining = np.zeros((0,), dtype=np.int32)
            return

        if (
            self.obstacle_move_direction_indices.shape[0] == obstacle_count
            and self.obstacle_move_speeds.shape[0] == obstacle_count
        ):
            return

        previous_directions = self.obstacle_move_direction_indices
        previous_steps = self.obstacle_move_steps_remaining
        previous_speeds = self.obstacle_move_speeds
        previous_speed_steps = self.obstacle_speed_steps_remaining
        self.obstacle_move_direction_indices = np.full((obstacle_count,), -1, dtype=np.int8)
        self.obstacle_move_steps_remaining = np.zeros((obstacle_count,), dtype=np.int32)
        self.obstacle_move_speeds = np.zeros((obstacle_count,), dtype=np.int16)
        self.obstacle_speed_steps_remaining = np.zeros((obstacle_count,), dtype=np.int32)
        copy_len = min(previous_directions.shape[0], obstacle_count)
        if copy_len > 0:
            self.obstacle_move_direction_indices[:copy_len] = previous_directions[:copy_len]
            self.obstacle_move_steps_remaining[:copy_len] = previous_steps[:copy_len]
        speed_copy_len = min(previous_speeds.shape[0], obstacle_count)
        if speed_copy_len > 0:
            self.obstacle_move_speeds[:speed_copy_len] = previous_speeds[:speed_copy_len]
            self.obstacle_speed_steps_remaining[:speed_copy_len] = previous_speed_steps[:speed_copy_len]

    def _sample_obstacle_speed(self) -> int:
        """Sample a random obstacle speed in cells per world step."""
        min_speed = max(int(self.obstacle_min_speed), 1)
        max_speed = max(int(self.obstacle_max_speed), min_speed)
        return int(self.randomizer.integers(min_speed, max_speed + 1))

    def set_obstacle_speed_range(self, min_speed: int, max_speed: int) -> None:
        """Update the obstacle speed sampling range."""
        self.obstacle_min_speed = max(int(min_speed), 1)
        self.obstacle_max_speed = max(int(max_speed), self.obstacle_min_speed)
        if hasattr(self, "obstacle_move_speeds") and self.obstacle_move_speeds.size > 0:
            np.clip(
                self.obstacle_move_speeds,
                self.obstacle_min_speed,
                self.obstacle_max_speed,
                out=self.obstacle_move_speeds,
            )

    @staticmethod
    def _rect_key(rect: pygame.Rect) -> tuple[int, int, int, int]:
        """Return a hashable identifier for an obstacle rect."""
        return int(rect.left), int(rect.top), int(rect.width), int(rect.height)

    @staticmethod
    def _rect_from_key(rect_key: tuple[int, int, int, int]) -> pygame.Rect:
        """Recreate a rect from its cached key."""
        left, top, width, height = rect_key
        return pygame.Rect(int(left), int(top), int(width), int(height))

    def _sync_observed_obstacle_rects(self) -> None:
        """Keep the rect list synchronized with the confidence cache."""
        self.observed_obstacle_rects = [
            self._rect_from_key(rect_key)
            for rect_key, confidence in self.observed_obstacle_confidences.items()
            if confidence > 0.0
        ]

    def _clear_observed_obstacle_key(self, rect_key) -> bool:
        """Remove one remembered obstacle location from shared observation state."""
        removed = self.observed_obstacle_confidences.pop(rect_key, None) is not None
        if removed and self.last_observed_rect_keys_by_obstacle:
            self.last_observed_rect_keys_by_obstacle = [
                None if key == rect_key else key
                for key in self.last_observed_rect_keys_by_obstacle
            ]
        return removed

    def _decay_observed_obstacle_confidences(self) -> bool:
        """Apply one-step decay to the remembered obstacle confidences."""
        if not self.observed_obstacle_confidences:
            return False

        decay = float(self.OBSERVED_OBSTACLE_DECAY)
        self.observed_obstacle_confidences = {
            rect_key: float(confidence) * decay
            for rect_key, confidence in self.observed_obstacle_confidences.items()
            if confidence > 0.0
        }
        return True

    def _decay_shared_obstacle_confidences_once(self) -> bool:
        """Apply one decay step per world timestep to shared obstacle confidences."""
        if self._last_obstacle_confidence_decay_timestep == self.timestep:
            return False

        self._last_obstacle_confidence_decay_timestep = self.timestep
        changed = self._decay_observed_obstacle_confidences()
        if self.actual_obstacle_confidences.size > 0:
            if np.any(self.actual_obstacle_confidences > 0.0):
                changed = True
            self.actual_obstacle_confidences *= np.float32(self.OBSERVED_OBSTACLE_DECAY)
        return changed

    def _rebuild_observed_obstacle_knowledge(self) -> None:
        """Rebuild the shared obstacle/warning belief maps from last observed positions."""
        self.observed_obstacle_pixel_mask.fill(0.0)
        self.explored_obstacle_map.fill(0.0)
        self.padded_explored_obstacle_map.fill(0.0)
        self.observed_warning_pixel_mask.fill(0.0)
        self.explored_warning_range_map.fill(0.0)
        self.padded_explored_warning_range_map.fill(0.0)
        self._invalidate_local_area_integrals(observed_obstacle=True, observed_warning=True)
        self._sync_observed_obstacle_rects()
        if not self.observed_obstacle_rects:
            return

        if self._all_unit_rects(self.observed_obstacle_rects):
            left, top, _, _ = self._rect_arrays(self.observed_obstacle_rects)
            confidences = np.fromiter(
                (self.observed_obstacle_confidences.get(self._rect_key(rect), 0.0) for rect in self.observed_obstacle_rects),
                dtype=np.float32,
                count=len(self.observed_obstacle_rects),
            )
            bottom = top + 1
            world_y = self.area.height - bottom
            valid = (
                (left >= 0)
                & (left < self.area.width)
                & (world_y >= 0)
                & (world_y < self.area.height)
                & (confidences > 0.0)
            )
            if np.any(valid):
                valid_left = left[valid]
                valid_world_y = world_y[valid]
                valid_confidences = confidences[valid]
                np.maximum.at(
                    self.observed_obstacle_pixel_mask,
                    (valid_world_y, valid_left),
                    valid_confidences,
                )

                grid_x = np.minimum((valid_left / self.coverage_cell_width).astype(np.int32), self.coverage_grid_size - 1)
                grid_y = np.minimum((valid_world_y / self.coverage_cell_height).astype(np.int32), self.coverage_grid_size - 1)
                np.add.at(
                    self.explored_obstacle_map,
                    (grid_y, grid_x),
                    valid_confidences / max(self.coverage_cell_area, 1e-6),
                )
                np.clip(self.explored_obstacle_map, 0.0, 1.0, out=self.explored_obstacle_map)

                center_x = valid_left.astype(np.float32, copy=False)
                center_y = valid_world_y.astype(np.float32, copy=False)
                self._paint_disks_on_world_mask(
                    self.observed_warning_pixel_mask,
                    center_x,
                    center_y,
                    float(self.OBSTACLE_WARNING_RADIUS),
                    values=valid_confidences,
                )

            self.padded_explored_obstacle_map[
                self.relative_pad : self.relative_pad + self.coverage_grid_size,
                self.relative_pad : self.relative_pad + self.coverage_grid_size,
            ] = self.explored_obstacle_map
            self._rebuild_observed_warning_map()
            return

        for obstacle in self.observed_obstacle_rects:
            belief_value = float(self.observed_obstacle_confidences.get(self._rect_key(obstacle), 0.0))
            if belief_value <= 0.0:
                continue
            world_left = max(int(obstacle.left), 0)
            world_right = min(int(obstacle.right), self.area.width)
            world_min_y = max(int(self.area.height - obstacle.bottom), 0)
            world_max_y = min(int(self.area.height - obstacle.top), self.area.height)
            if world_right > world_left and world_max_y > world_min_y:
                current = self.observed_obstacle_pixel_mask[world_min_y:world_max_y, world_left:world_right]
                self.observed_obstacle_pixel_mask[world_min_y:world_max_y, world_left:world_right] = np.maximum(
                    current,
                    np.float32(belief_value),
                )

            self._paint_disk_on_world_mask(
                self.observed_warning_pixel_mask,
                center_x=float(obstacle.centerx),
                center_y=float(self.area.height - obstacle.centery),
                radius=float(self.OBSTACLE_WARNING_RADIUS),
                value=belief_value,
            )

            min_grid_x = max(int(obstacle.left / self.coverage_cell_width), 0)
            max_grid_x = min(
                int((max(obstacle.right - 1, obstacle.left)) / self.coverage_cell_width),
                self.coverage_grid_size - 1,
            )
            min_grid_y = max(int(world_min_y / self.coverage_cell_height), 0)
            max_grid_y = min(
                int((max(world_max_y - 1, world_min_y)) / self.coverage_cell_height),
                self.coverage_grid_size - 1,
            )

            for grid_y in range(min_grid_y, max_grid_y + 1):
                for grid_x in range(min_grid_x, max_grid_x + 1):
                    if self.search_mask[grid_y, grid_x] <= 0.0:
                        continue
                    cell_rect = self.cell_game_rects[grid_y][grid_x]
                    overlap_rect = obstacle.clip(cell_rect)
                    if overlap_rect.width <= 0 or overlap_rect.height <= 0:
                        continue
                    overlap_area = float(overlap_rect.width * overlap_rect.height)
                    self.explored_obstacle_map[grid_y, grid_x] += (
                        overlap_area / max(self.coverage_cell_area, 1e-6)
                    ) * belief_value

        np.clip(self.explored_obstacle_map, 0.0, 1.0, out=self.explored_obstacle_map)
        self.padded_explored_obstacle_map[
            self.relative_pad : self.relative_pad + self.coverage_grid_size,
            self.relative_pad : self.relative_pad + self.coverage_grid_size,
        ] = self.explored_obstacle_map
        self._rebuild_observed_warning_map()

    def update_obstacle_observations_for_agent(self, agent) -> bool:
        """Update obstacle beliefs from the current agent sensor view."""
        sensor = getattr(agent, "sensor", None)
        if sensor is None or not hasattr(sensor, "is_point_detected"):
            return False

        def visibility_mask(points: np.ndarray) -> np.ndarray:
            if points.size == 0:
                return np.zeros((0,), dtype=bool)

            pos = getattr(sensor, "pos", None)
            theta = getattr(sensor, "theta", None)
            sensing_range = getattr(sensor, "sensing_range", None)
            hfov = getattr(sensor, "hfov", None)

            if pos is not None and sensing_range is not None:
                delta_x = points[:, 0] - float(pos[0])
                delta_y_screen = points[:, 1] - float(pos[1])
                dist_sq = (delta_x * delta_x) + (delta_y_screen * delta_y_screen)
                in_range = dist_sq < float(sensing_range) * float(sensing_range)

                if hfov is None or theta is None:
                    return in_range

                delta_y_world = -delta_y_screen
                angles = np.arctan2(delta_y_world, delta_x)
                relative = (angles - float(theta) + np.pi) % (2 * np.pi) - np.pi
                return in_range & (np.abs(relative) < float(hfov) / 2.0)

            return np.array([bool(sensor.is_point_detected(tuple(point))) for point in points], dtype=bool)

        changed = self._decay_shared_obstacle_confidences_once()
        actual_obstacle_lookup = self.obstacle_keys
        observed_keys = list(self.observed_obstacle_confidences.keys())
        if observed_keys:
            observed_rects = [self._rect_from_key(rect_key) for rect_key in observed_keys]
            observed_centers = np.asarray([rect.center for rect in observed_rects], dtype=np.float32)
            observed_visible = visibility_mask(observed_centers)
            for rect_key, is_visible in zip(observed_keys, observed_visible):
                if is_visible and rect_key not in actual_obstacle_lookup:
                    changed = self._clear_observed_obstacle_key(rect_key) or changed

        if self.obstacles:
            obstacle_centers = np.asarray([obstacle.center for obstacle in self.obstacles], dtype=np.float32)
            obstacle_visible = visibility_mask(obstacle_centers)
            visible_current_keys = set()
            previous_keys_to_clear = set()

            for obstacle_idx, (obstacle, is_visible) in enumerate(zip(self.obstacles, obstacle_visible)):
                if not is_visible:
                    continue
                obstacle_key = self._rect_key(obstacle)
                visible_current_keys.add(obstacle_key)
                previous_key = None
                if obstacle_idx < len(self.last_observed_rect_keys_by_obstacle):
                    previous_key = self.last_observed_rect_keys_by_obstacle[obstacle_idx]
                if previous_key is not None and previous_key != obstacle_key:
                    previous_keys_to_clear.add(previous_key)

            for rect_key in previous_keys_to_clear - visible_current_keys:
                changed = self._clear_observed_obstacle_key(rect_key) or changed

            for obstacle_idx, (obstacle, is_visible) in enumerate(zip(self.obstacles, obstacle_visible)):
                if not is_visible:
                    continue
                obstacle_key = self._rect_key(obstacle)
                if obstacle_idx < len(self.last_observed_rect_keys_by_obstacle):
                    self.last_observed_rect_keys_by_obstacle[obstacle_idx] = obstacle_key
                previous_confidence = self.observed_obstacle_confidences.get(obstacle_key)
                if previous_confidence != 1.0:
                    self.observed_obstacle_confidences[obstacle_key] = 1.0
                    changed = True
                if obstacle_idx < self.actual_obstacle_confidences.shape[0] and self.actual_obstacle_confidences[obstacle_idx] != 1.0:
                    self.actual_obstacle_confidences[obstacle_idx] = 1.0
                    changed = True

        if changed:
            self._rebuild_observed_obstacle_knowledge()
        return changed

    def _refresh_padded_observation_maps(self, touched_flat: np.ndarray | None = None) -> None:
        """Keep cached padded observation maps in sync with coverage updates."""
        search_mask_positive = self.search_mask > 0.0

        if touched_flat is None:
            self.padded_coverage_map.fill(0.0)
            self.observation_coverage_map.fill(0.0)
            np.divide(
                self.coverage_map,
                self.search_mask,
                out=self.observation_coverage_map,
                where=search_mask_positive,
            )
            np.clip(self.observation_coverage_map, 0.0, 1.0, out=self.observation_coverage_map)
            self.padded_coverage_map[
                self.relative_pad : self.relative_pad + self.coverage_grid_size,
                self.relative_pad : self.relative_pad + self.coverage_grid_size,
            ] = self.observation_coverage_map
            return

        touched_flat = np.asarray(touched_flat, dtype=np.int32).reshape(-1)
        if touched_flat.size == 0:
            return
        grid_y, grid_x = np.divmod(touched_flat, self.coverage_grid_size)
        padded_y = grid_y + self.relative_pad
        padded_x = grid_x + self.relative_pad
        raw_search = self.search_mask[grid_y, grid_x]
        normalized_coverage = np.zeros_like(raw_search, dtype=np.float32)
        valid_search = raw_search > 0.0
        normalized_coverage[valid_search] = self.coverage_map[grid_y, grid_x][valid_search] / raw_search[valid_search]
        self.observation_coverage_map[grid_y, grid_x] = np.clip(normalized_coverage, 0.0, 1.0)
        self.padded_coverage_map[padded_y, padded_x] = self.observation_coverage_map[grid_y, grid_x]

    def reset(self, poi_list, seed=None, options=None):
        """Reset world."""
        self.timestep = 0
        self.observer_communication = [0.0, 0.0]
        self.goal_known = False
        self.goal_position = (poi_list[0].x, poi_list[0].y) if poi_list else None
        self.detected = set()
        self.detected_count = 0
        self.detected_mask.fill(False)
        self.coverage_counts.fill(0)
        self.coverage_map.fill(0.0)
        self.observation_coverage_map.fill(0.0)
        self.obstacle_map.fill(0.0)
        self.explored_obstacle_map.fill(0.0)
        self.warning_range_map.fill(0.0)
        self.explored_warning_range_map.fill(0.0)
        self.obstacle_pixel_mask.fill(0.0)
        self.warning_pixel_mask.fill(0.0)
        self.observed_obstacle_pixel_mask.fill(0.0)
        self.observed_warning_pixel_mask.fill(0.0)
        self.padded_coverage_map.fill(0.0)
        self.padded_explored_obstacle_map.fill(0.0)
        self.padded_explored_warning_range_map.fill(0.0)
        self.actual_obstacle_confidences = np.zeros((len(self.obstacles),), dtype=np.float32)
        self._last_obstacle_confidence_decay_timestep = -1
        self.observed_obstacle_confidences = {}
        self.last_observed_rect_keys_by_obstacle = [None] * len(self.obstacles)
        self.observed_obstacle_rects = []
        self._invalidate_local_area_integrals(observed_obstacle=True, observed_warning=True)
        self.revealed_warning_obstacles = np.zeros((len(self.obstacles),), dtype=bool)
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

    def register_detected_points(self, points, *, return_new_points: bool = False, assume_unique: bool = False):
        """Update the cached coverage map with newly detected coordinates."""
        if isinstance(points, np.ndarray):
            points_array = np.asarray(points, dtype=np.int32).reshape(-1, 2)
            if points_array.size == 0:
                return points_array if return_new_points else self.detected_count

            x = points_array[:, 0]
            y = points_array[:, 1]
            valid = (0 <= x) & (x < self.area.width) & (0 <= y) & (y < self.area.height)
            if not np.any(valid):
                empty = np.empty((0, 2), dtype=np.int32)
                return empty if return_new_points else self.detected_count

            valid_points = points_array[valid]
            if not assume_unique and len(valid_points) > 1:
                valid_points = np.unique(valid_points, axis=0)
            valid_x = valid_points[:, 0]
            valid_y = valid_points[:, 1]
            unseen = ~self.detected_mask[valid_y, valid_x]
            if not np.any(unseen):
                empty = np.empty((0, 2), dtype=np.int32)
                return empty if return_new_points else self.detected_count

            new_points = valid_points[unseen]
            new_x = new_points[:, 0]
            new_y = new_points[:, 1]
            self.detected_mask[new_y, new_x] = True
            self.detected_count += int(len(new_points))
            grid_x = np.minimum((new_x / self.coverage_cell_width).astype(np.int32), self.coverage_grid_size - 1)
            grid_y = np.minimum((new_y / self.coverage_cell_height).astype(np.int32), self.coverage_grid_size - 1)
            flat_cells = grid_y * self.coverage_grid_size + grid_x
            cell_counts = np.bincount(flat_cells, minlength=self.coverage_grid_size * self.coverage_grid_size)
            touched_flat = np.flatnonzero(cell_counts)
            self.coverage_counts_flat[touched_flat] += cell_counts[touched_flat]
            self.coverage_map_flat[touched_flat] = np.minimum(
                self.coverage_counts_flat[touched_flat] / self.coverage_cell_area,
                1.0,
            )
            self._refresh_padded_observation_maps(touched_flat)

            return new_points if return_new_points else self.detected_count

        cell_updates = {}
        for x, y in points:
            point = (int(x), int(y))
            if not (0 <= point[0] < self.area.width and 0 <= point[1] < self.area.height):
                continue
            if self.detected_mask[point[1], point[0]]:
                continue

            self.detected.add(point)
            self.detected_mask[point[1], point[0]] = True
            self.detected_count += 1
            grid_x = min(int(point[0] / self.coverage_cell_width), self.coverage_grid_size - 1)
            grid_y = min(int(point[1] / self.coverage_cell_height), self.coverage_grid_size - 1)
            key = (grid_y, grid_x)
            cell_updates[key] = cell_updates.get(key, 0) + 1

        for (grid_y, grid_x), count in cell_updates.items():
            new_total = self.coverage_counts[grid_y, grid_x] + count
            self.coverage_counts[grid_y, grid_x] = new_total
            self.coverage_map[grid_y, grid_x] = min(new_total / self.coverage_cell_area, 1.0)
        if cell_updates:
            touched_flat = np.array(
                [grid_y * self.coverage_grid_size + grid_x for grid_y, grid_x in cell_updates],
                dtype=np.int32,
            )
            self._refresh_padded_observation_maps(touched_flat)

        return self.detected_count

    def clear_obstacles(self):
        """Remove all obstacles from the world."""
        self.obstacles.clear()  # Clear the list of obstacles
        self.obstacle_map.fill(0.0)
        self.explored_obstacle_map.fill(0.0)
        self.warning_range_map.fill(0.0)
        self.explored_warning_range_map.fill(0.0)
        self.padded_explored_obstacle_map.fill(0.0)
        self.padded_explored_warning_range_map.fill(0.0)
        self.obstacle_pixel_mask.fill(0.0)
        self.warning_pixel_mask.fill(0.0)
        self.observed_obstacle_pixel_mask.fill(0.0)
        self.observed_warning_pixel_mask.fill(0.0)
        self.actual_obstacle_confidences = np.zeros((0,), dtype=np.float32)
        self._last_obstacle_confidence_decay_timestep = -1
        self.observed_obstacle_confidences = {}
        self.last_observed_rect_keys_by_obstacle = []
        self.observed_obstacle_rects = []
        self._invalidate_local_area_integrals(observed_obstacle=True, observed_warning=True)
        self._refresh_obstacle_cache()

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

    def _rect_to_world_polygon(self, rect: pygame.Rect) -> Polygon:
        """Convert a game-space rect into a world-space polygon."""
        return Polygon(
            [
                game_ref_to_world_ref(rect.topleft, self.area),
                game_ref_to_world_ref(rect.topright, self.area),
                game_ref_to_world_ref(rect.bottomright, self.area),
                game_ref_to_world_ref(rect.bottomleft, self.area),
            ]
        )

    def _rect_within_search_area(self, rect: pygame.Rect) -> bool:
        """Return True when the entire obstacle rect is inside the search area."""
        if self.search_area_is_rect:
            minx, miny, maxx, maxy = self.search_bounds
            world_left, world_top = game_ref_to_world_ref(rect.topleft, self.area)
            world_right, world_bottom = game_ref_to_world_ref(rect.bottomright, self.area)
            world_min_x = min(world_left, world_right)
            world_max_x = max(world_left, world_right)
            world_min_y = min(world_top, world_bottom)
            world_max_y = max(world_top, world_bottom)
            return minx <= world_min_x and world_max_x <= maxx and miny <= world_min_y and world_max_y <= maxy
        return bool(self.prepared_search_area.covers(self._rect_to_world_polygon(rect)))

    def _rebuild_obstacle_map(self) -> None:
        """Refresh the obstacle occupancy grid used by agent observations."""
        self.obstacle_map.fill(0.0)
        self.obstacle_pixel_mask.fill(0.0)
        self.warning_pixel_mask.fill(0.0)
        if not self.obstacles:
            self.warning_range_map.fill(0.0)
            self.actual_obstacle_confidences = np.zeros((0,), dtype=np.float32)
            self.observed_obstacle_confidences = {}
            self.observed_obstacle_rects = []
            self._rebuild_observed_obstacle_knowledge()
            self._refresh_obstacle_cache()
            self._refresh_padded_observation_maps()
            return

        if self._all_unit_rects(self.obstacles):
            left, top, _, _ = self._rect_arrays(self.obstacles)
            bottom = top + 1
            world_y = self.area.height - bottom
            valid = (left >= 0) & (left < self.area.width) & (world_y >= 0) & (world_y < self.area.height)
            if np.any(valid):
                valid_left = left[valid]
                valid_world_y = world_y[valid]
                self.obstacle_pixel_mask[valid_world_y, valid_left] = 1.0
                grid_x = np.minimum((valid_left / self.coverage_cell_width).astype(np.int32), self.coverage_grid_size - 1)
                grid_y = np.minimum((valid_world_y / self.coverage_cell_height).astype(np.int32), self.coverage_grid_size - 1)
                np.add.at(self.obstacle_map, (grid_y, grid_x), 1.0 / max(self.coverage_cell_area, 1e-6))
                np.clip(self.obstacle_map, 0.0, 1.0, out=self.obstacle_map)

                center_x = valid_left.astype(np.float32, copy=False)
                center_y = valid_world_y.astype(np.float32, copy=False)
                self._paint_disks_on_world_mask(
                    self.warning_pixel_mask,
                    center_x,
                    center_y,
                    float(self.OBSTACLE_WARNING_RADIUS),
                )

            self._rebuild_warning_range_map()
            self._refresh_obstacle_cache()
            self._refresh_padded_observation_maps()
            return

        for obstacle in self.obstacles:
            world_left = max(int(obstacle.left), 0)
            world_right = min(int(obstacle.right), self.area.width)
            world_min_y = max(int(self.area.height - obstacle.bottom), 0)
            world_max_y = min(int(self.area.height - obstacle.top), self.area.height)
            if world_right > world_left and world_max_y > world_min_y:
                self.obstacle_pixel_mask[world_min_y:world_max_y, world_left:world_right] = 1.0
            self._paint_disk_on_world_mask(
                self.warning_pixel_mask,
                center_x=float(obstacle.centerx),
                center_y=float(self.area.height - obstacle.centery),
                radius=float(self.OBSTACLE_WARNING_RADIUS),
            )

            min_grid_x = max(int(obstacle.left / self.coverage_cell_width), 0)
            max_grid_x = min(int((max(obstacle.right - 1, obstacle.left)) / self.coverage_cell_width), self.coverage_grid_size - 1)
            min_grid_y = max(int(world_min_y / self.coverage_cell_height), 0)
            max_grid_y = min(
                int((max(world_max_y - 1, world_min_y)) / self.coverage_cell_height),
                self.coverage_grid_size - 1,
            )

            for grid_y in range(min_grid_y, max_grid_y + 1):
                for grid_x in range(min_grid_x, max_grid_x + 1):
                    if self.search_mask[grid_y, grid_x] <= 0.0:
                        continue
                    cell_rect = self.cell_game_rects[grid_y][grid_x]
                    overlap_rect = obstacle.clip(cell_rect)
                    if overlap_rect.width <= 0 or overlap_rect.height <= 0:
                        continue
                    overlap_area = float(overlap_rect.width * overlap_rect.height)
                    self.obstacle_map[grid_y, grid_x] += overlap_area / max(self.coverage_cell_area, 1e-6)

        np.clip(self.obstacle_map, 0.0, 1.0, out=self.obstacle_map)
        self._rebuild_warning_range_map()
        self._refresh_obstacle_cache()
        self._refresh_padded_observation_maps()

    def generate_obstacles(self, n_obstacles, avoid_rects=None):
        """Generate random obstacles."""
        blocked_rects = [rect for rect in (avoid_rects or []) if rect is not None]
        for i in range(n_obstacles):
            # w, h = self.randomizer.integers(10, 150), self.randomizer.integers(10, 150)
            w, h = 1, 1
            obstacle = pygame.Rect(0, 0, w, h)
            valid_coord = False
            tries = 0
            while not valid_coord and tries < self.spawn_max_tries:
                tries += 1
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
                inside_search_area = self._rect_within_search_area(obstacle)
                overlaps_blocked_rect = any(obstacle.colliderect(blocked_rect) for blocked_rect in blocked_rects)
                if base_clearance_ok and not road_collision and inside_search_area and not overlaps_blocked_rect:
                    valid_coord = True
            if valid_coord:
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
        confidence_font = pygame.font.SysFont("Trebuchet MS", 14)
        obstacle_warning_overlay = pygame.Surface(self.area.size, pygame.SRCALPHA)
        for obstacle in self.obstacles:
            pygame.draw.circle(
                obstacle_warning_overlay,
                self.OBSTACLE_WARNING_FILL,
                obstacle.center,
                self.OBSTACLE_WARNING_RADIUS,
            )
            pygame.draw.circle(
                obstacle_warning_overlay,
                self.OBSTACLE_WARNING_OUTLINE,
                obstacle.center,
                self.OBSTACLE_WARNING_RADIUS,
                width=2,
            )
        screen.blit(obstacle_warning_overlay, (0, 0))

        for obstacle_idx, obstacle in enumerate(self.obstacles):
            pygame.draw.rect(screen, (150, 0, 0), obstacle)
            confidence = 0.0
            if obstacle_idx < self.actual_obstacle_confidences.shape[0]:
                confidence = float(self.actual_obstacle_confidences[obstacle_idx])
            confidence_text = confidence_font.render(f"{confidence:.2f}", True, (255, 245, 245))
            label_rect = confidence_text.get_rect(midbottom=(obstacle.centerx, obstacle.top - 4))
            background_rect = label_rect.inflate(6, 2)
            pygame.draw.rect(screen, (35, 10, 10), background_rect, border_radius=4)
            screen.blit(confidence_text, label_rect)

    def _move_obstacles_one_step(self) -> bool:
        """Move each obstacle while keeping piecewise-constant direction and speed."""
        if not self.obstacles:
            return False

        self._sync_obstacle_motion_state()
        obstacle_indices = self.randomizer.permutation(len(self.obstacles))
        original_rects = [obstacle.copy() for obstacle in self.obstacles]
        stationary_keys = {self._rect_key(rect) for rect in original_rects}
        planned_rects = [rect.copy() for rect in original_rects]
        planned_path_keys = set()
        next_direction_indices = self.obstacle_move_direction_indices.copy()
        next_steps_remaining = self.obstacle_move_steps_remaining.copy()
        next_speeds = self.obstacle_move_speeds.copy()
        next_speed_steps_remaining = self.obstacle_speed_steps_remaining.copy()
        moved = False

        for obstacle_idx in obstacle_indices:
            idx = int(obstacle_idx)
            current_rect = original_rects[idx]
            current_key = self._rect_key(current_rect)
            stationary_keys.discard(current_key)

            chosen_rect = current_rect.copy()
            chosen_direction_idx = -1
            chosen_steps_remaining = 0
            current_direction_idx = int(self.obstacle_move_direction_indices[idx])
            current_steps_remaining = int(self.obstacle_move_steps_remaining[idx])
            current_speed = int(self.obstacle_move_speeds[idx])
            current_speed_steps_remaining = int(self.obstacle_speed_steps_remaining[idx])
            if current_speed > 0 and current_speed_steps_remaining > 0:
                chosen_speed = current_speed
                chosen_speed_steps_remaining = current_speed_steps_remaining - 1
            else:
                chosen_speed = self._sample_obstacle_speed()
                chosen_speed_steps_remaining = self.OBSTACLE_SPEED_HOLD_STEPS - 1

            candidate_direction_indices = []
            if 0 <= current_direction_idx < len(self.OBSTACLE_MOVE_DELTAS) and current_steps_remaining > 0:
                candidate_direction_indices.append(current_direction_idx)

            for random_direction_idx in self.randomizer.permutation(len(self.OBSTACLE_MOVE_DELTAS)):
                direction_idx = int(random_direction_idx)
                if direction_idx in candidate_direction_indices:
                    continue
                if current_steps_remaining <= 0 and direction_idx == current_direction_idx:
                    continue
                candidate_direction_indices.append(direction_idx)

            for direction_idx in candidate_direction_indices:
                dx, dy = self.OBSTACLE_MOVE_DELTAS[direction_idx]
                candidate = current_rect.copy()
                candidate_path_keys = []
                path_is_valid = True
                for _ in range(chosen_speed):
                    candidate = candidate.move(dx, dy)
                    candidate_key = self._rect_key(candidate)
                    if candidate_key in stationary_keys or candidate_key in planned_path_keys:
                        path_is_valid = False
                        break
                    if not self._rect_within_search_area(candidate):
                        path_is_valid = False
                        break
                    candidate_path_keys.append(candidate_key)
                if not path_is_valid:
                    continue

                chosen_rect = candidate
                chosen_direction_idx = direction_idx
                if direction_idx == current_direction_idx and current_steps_remaining > 0:
                    chosen_steps_remaining = current_steps_remaining - 1
                else:
                    chosen_steps_remaining = self.OBSTACLE_DIRECTION_HOLD_STEPS - 1
                moved = moved or (candidate_key != current_key)
                planned_path_keys.update(candidate_path_keys)
                break

            planned_rects[idx] = chosen_rect
            next_direction_indices[idx] = chosen_direction_idx
            next_steps_remaining[idx] = chosen_steps_remaining
            next_speeds[idx] = chosen_speed
            next_speed_steps_remaining[idx] = chosen_speed_steps_remaining

        for obstacle, new_rect in zip(self.obstacles, planned_rects):
            obstacle.x = new_rect.x
            obstacle.y = new_rect.y
            obstacle.width = new_rect.width
            obstacle.height = new_rect.height

        self.obstacle_move_direction_indices = next_direction_indices
        self.obstacle_move_steps_remaining = next_steps_remaining
        self.obstacle_move_speeds = next_speeds
        self.obstacle_speed_steps_remaining = next_speed_steps_remaining
        if moved:
            self._rebuild_obstacle_map()
        return moved

    def update(self, area):
        """Update world."""
        # increase timestep counter to know how many step were run
        self.timestep += 1
        self._move_obstacles_one_step()

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
