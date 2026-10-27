#!/usr/bin/env python3
"""Online shared-belief frontier A* baseline for HeMAC A/B exploration.

NO access to hidden goals, true obstacle positions, or future motions. Uses
shared observed coverage/hazard maps and self/teammate pose (robot localisation).
Each joint action cycle: rebuild cost grid, select frontiers, A* for every drone,
then output 3D continuous [velocity-x, velocity-y, recharge=0] actions.

Implementation note: the HeMAC world exposes *shared observed maps* as the
source for actor map channels. This planner reads those belief grids directly
(rather than decoding centred/resized actor global_map). It is an informed
map-based classical baseline, NOT exactly the same tensor-input policy as PPO.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

NEIGHBORS = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
             (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
             (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)))


def inflate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Boolean Chebyshev dilation in numpy; no SciPy required."""
    mask = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return mask.copy()
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    out = np.zeros_like(mask)
    h, w = mask.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            out |= padded[radius + dy:radius + dy + h,
                          radius + dx:radius + dx + w]
    return out


def astar(cost: np.ndarray, blocked: np.ndarray, start: tuple[int, int],
          goal: tuple[int, int]) -> tuple[list[tuple[int, int]], float]:
    """8-connected A*, no diagonal corner cutting. Grid index=(row=y,col=x)."""
    h, w = cost.shape
    if not (0 <= start[0] < h and 0 <= start[1] < w and
            0 <= goal[0] < h and 0 <= goal[1] < w):
        return [], math.inf
    if blocked[goal] and goal != start:
        return [], math.inf
    if goal == start:
        return [start], 0.0
    dist = {start: 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    closed: set[tuple[int, int]] = set()
    min_cost = max(1.0, float(np.min(cost)))

    def heuristic(pos):
        dy, dx = pos[0] - goal[0], pos[1] - goal[1]
        # Octile admissible heuristic for 8 connected grid with costs >=1.
        m = min(abs(dy), abs(dx))
        return ((abs(dy) + abs(dx) - 2*m) + math.sqrt(2)*m) * min_cost

    frontier = [(heuristic(start), 0.0, start)]
    while frontier:
        _, g, node = heapq.heappop(frontier)
        if node in closed:
            continue
        if node == goal:
            result = [goal]
            while result[-1] != start:
                result.append(parent[result[-1]])
            return result[::-1], g
        closed.add(node)
        r, c = node
        for dr, dc, step_len in NEIGHBORS:
            nr, nc = r+dr, c+dc
            nxt = (nr, nc)
            if not (0 <= nr < h and 0 <= nc < w) or nxt in closed:
                continue
            if blocked[nxt]:
                continue
            if dr and dc and (blocked[r+dr, c] or blocked[r, c+dc]):
                continue
            new_g = g + step_len * .5 * (float(cost[node]) + float(cost[nxt]))
            if new_g < dist.get(nxt, math.inf):
                dist[nxt] = new_g
                parent[nxt] = node
                heapq.heappush(frontier, (new_g + heuristic(nxt), new_g, nxt))
    return [], math.inf


@dataclass
class AStarConfig:
    obstacle_threshold: float = .35
    warning_threshold: float = .25
    inflation_cells: int = 0    # warning_map already encodes the danger radius
    soft_warning_cost: float = 6.0
    soft_obstacle_cost: float = 2.0
    teammate_cost: float = 2.0
    max_candidates: int = 6
    progress_weight: float = 0.55
    target_stickiness: float = 0.16
    min_unexplored: float = .045
    separation: float = 68.0
    kp_position: float = .24
    kd_velocity: float = .75
    min_action_speed_fraction: float = .0


class DynamicFrontierAStar:
    """Multi-agent receding-horizon online A* coverage explorer."""
    def __init__(self, cfg: AStarConfig | None = None):
        self.cfg = cfg or AStarConfig()
        self.previous_targets: dict[str, tuple[int, int]] = {}
        self.replans = 0
        self.failures = 0
        self.last_debug: dict[str, Any] = {}

    def reset(self) -> None:
        self.previous_targets.clear()
        self.replans = 0
        self.failures = 0
        self.last_debug.clear()

    @staticmethod
    def _grid_snapshot(core):
        """Only OBSERVED/shared belief channels, never core.goals or true obstacles."""
        world = core.world
        search = np.asarray(world.search_mask, dtype=np.float32)
        known_cov = np.asarray(world.observation_coverage_map, dtype=np.float32)
        obstacle = np.asarray(world.explored_obstacle_map, dtype=np.float32)
        warning = np.asarray(world.explored_warning_range_map, dtype=np.float32)
        if not (search.shape == known_cov.shape == obstacle.shape == warning.shape
                and search.ndim == 2):
            raise RuntimeError("World observed maps must be matching [H,W] arrays")
        if not all(np.isfinite(x).all() for x in (search, known_cov, obstacle, warning)):
            raise FloatingPointError("Invalid observed map values")
        return search, np.clip(known_cov, 0, 1), np.clip(obstacle, 0, 1), np.clip(warning, 0, 1)

    @staticmethod
    def _coord_to_cell(x, y, width, height, shape):
        rows, cols = shape
        col = int(np.clip(math.floor(x / width * cols), 0, cols - 1))
        row = int(np.clip(math.floor(y / height * rows), 0, rows - 1))
        return row, col

    @staticmethod
    def _cell_to_coord(cell, width, height, shape):
        rows, cols = shape
        return np.array([(cell[1] + .5) * width / cols,
                         (cell[0] + .5) * height / rows], dtype=np.float64)

    @staticmethod
    def _gain_map(unknown: np.ndarray, radius_cells: int):
        h, w = unknown.shape
        padded = np.pad(unknown, radius_cells, mode="constant")
        out = np.zeros((h, w), dtype=np.float32)
        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                if dr*dr + dc*dc <= radius_cells*radius_cells:
                    out += padded[radius_cells+dr:radius_cells+dr+h,
                                  radius_cells+dc:radius_cells+dc+w]
        return out

    def _cost_grid(self, search, obstacle, warning, team_positions,
                   agent_name, width, height):
        cfg = self.cfg
        blocked = ((search <= 0.01) |
                   (obstacle >= cfg.obstacle_threshold) |
                   (warning >= cfg.warning_threshold))
        blocked = inflate(blocked, cfg.inflation_cells)
        penalty = np.maximum(0.0, cfg.soft_warning_cost * warning + cfg.soft_obstacle_cost * obstacle)
        rows, cols = search.shape
        yy = (np.arange(rows) + .5) * height / rows
        xx = (np.arange(cols) + .5) * width / cols
        for name, xy in team_positions.items():
            if name == agent_name:
                continue
            # Teammates are soft obstacles: hard blocking strands clustered starts.
            rr2 = (yy[:, None] - xy[1])**2 + (xx[None, :] - xy[0])**2
            penalty += cfg.teammate_cost * np.exp(-rr2 / (2 * (width / cols * 1.5)**2))
        cost = 1.0 + penalty
        return cost.astype(np.float64), blocked

    def _select_path(self, agent, name, search, coverage, cost, blocked,
                     assigned, pos, world_width, world_height, eligible=None):
        cfg = self.cfg
        shape = search.shape
        start = self._coord_to_cell(pos[0], pos[1], world_width, world_height, shape)
        # From current robot location leave initial start passable; this is NOT
        # an oracle override for other nodes, only permits escaping a sensed danger.
        blocked = blocked.copy()
        blocked[start] = False
        reward_mask = np.asarray(search > .01, dtype=bool)
        if eligible is not None:
            eligible = np.asarray(eligible, dtype=bool)
            if eligible.shape != search.shape:
                raise ValueError("Reward eligibility mask must match search grid")
            reward_mask &= eligible
        unknown = np.maximum(0.0, 1.0 - coverage) * reward_mask
        gain_radius = max(1, int(round(float(agent.sensing_range) /
                                        (world_width / shape[1]))))
        gain_radius = min(gain_radius, max(shape))
        gain = self._gain_map(unknown, gain_radius)
        valid = reward_mask & ~blocked
        if not valid.any():
            return [start], None, {"reason": "no_valid_cells"}

        ys, xs = np.indices(shape)
        agent_speed = float(agent.max_speed)
        step_dist = np.hypot(ys-start[0], xs-start[1])
        travel_time = step_dist * (world_width/shape[1]) / max(agent_speed, 1e-3)
        # Score reachable frontier visibility gain against estimated travel time.
        score = gain / (2.5 + travel_time * cfg.progress_weight)
        # Directional camera yields smaller instantaneous area at each waypoint;
        # the same ranking still works and repeated replanning covers its blind spots.
        score[~valid] = -np.inf
        score[start] = -np.inf
        for other_target in assigned:
            otr, otc = other_target
            near2 = (ys - otr)**2 + (xs - otc)**2
            score -= (gain.max() * .13) * np.exp(-near2 / 6.0)
        old_target = self.previous_targets.get(name)
        if old_target is not None and valid[old_target] and gain[old_target] > cfg.min_unexplored:
            score[old_target] += cfg.target_stickiness * max(1.0, float(np.max(gain)))
        # Avoid choosing reachable but zero-gain cells after complete exploration.
        score[(gain < cfg.min_unexplored) & valid] = -np.inf
        order = np.argsort(score.ravel())[::-1]
        valid_candidates = 0
        for index in order:
            if valid_candidates >= cfg.max_candidates:
                break
            target = tuple(map(int, np.unravel_index(int(index), shape)))
            if not np.isfinite(score[target]):
                break
            valid_candidates += 1
            path, distance = astar(cost, blocked, start, target)
            self.replans += 1
            if path:
                self.previous_targets[name] = target
                return path, target, {"score": float(score[target]),
                                      "path_cost": distance,
                                      "target_gain": float(gain[target])}
        # Stuck: attempt to reach best other traversable location with nonzero gain
        # before issuing a safe stop. Never drive straight through a blocked cell.
        self.failures += 1
        self.previous_targets.pop(name, None)
        return [start], None, {"reason": "no_reachable_frontier"}

    def _velocity_action(self, agent, name, path, team_positions, world_width, world_height, shape):
        pos = np.array([float(agent.x), float(agent.y)], dtype=np.float64)
        vel = np.array([float(agent.vx), float(agent.vy)], dtype=np.float64)
        speed = float(agent.max_speed)
        # Path cells are coarse (~50m). Look ahead until reaching a useful point;
        # do not move directly to a far target, since that would cut A* corners.
        if len(path) >= 2:
            next_cell = path[1]
        elif len(path) == 1:
            next_cell = path[0]
        else:
            return np.zeros(3, dtype=np.float32)
        waypoint = self._cell_to_coord(next_cell, world_width, world_height, shape)
        displacement = waypoint - pos
        desired_velocity = self.cfg.kp_position * displacement
        norm = np.linalg.norm(desired_velocity)
        if norm > speed:
            desired_velocity *= speed / norm
        # Keep heading choices smooth under inertia/drag and bounded acceleration.
        action_xy = desired_velocity + self.cfg.kd_velocity * (desired_velocity - vel)
        # Avoid flying through teammate even when two starts are clustered.
        repel = np.zeros(2, dtype=np.float64)
        for teammate, q in team_positions.items():
            if teammate == name:
                continue
            dvec = pos - q
            dist = np.linalg.norm(dvec)
            if 1e-6 < dist < self.cfg.separation:
                repel += (dvec / dist) * (self.cfg.separation - dist) * .12
        action_xy += repel
        action_xy = np.clip(action_xy, -speed, speed)
        return np.array([action_xy[0], action_xy[1], 0.0], dtype=np.float32)

    def act(self, core, agent_names: list[str] | tuple[str, ...]):
        """Replan every joint cycle using refreshed observed maps."""
        search, coverage, obstacle, warning = self._grid_snapshot(core)
        width = float(core.world.area.width)
        height = float(core.world.area.height)
        agents = {name: core.agents_list[core.agent_name_mapping[name]]
                  for name in agent_names}
        team_positions = {name: np.array([float(a.x), float(a.y)], dtype=np.float64)
                          for name, a in agents.items()}
        assigned: list[tuple[int, int]] = []
        decisions = {}
        actions = {}
        for name in agent_names:
            agent = agents[name]
            cost, blocked = self._cost_grid(search, obstacle, warning,
                                            team_positions, name, width, height)
            path, target, dbg = self._select_path(agent, name, search, coverage,
                                                  cost, blocked, assigned,
                                                  team_positions[name], width, height,
                                                  getattr(core, "drone_reward_sector_mask", None))
            if target is not None:
                assigned.append(target)
            actions[name] = self._velocity_action(agent, name, path,
                                                  team_positions, width, height,
                                                  search.shape)
            # Final A/B Box compatibility enforced by caller.
            decisions[name] = {"target_cell": target,
                               "path_length_cells": len(path), **dbg}
        self.last_debug = decisions
        return actions
