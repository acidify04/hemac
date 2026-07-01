"""HeMAC.

| Import               | `from HeMAC import HeMAC_v0` |
|----------------------|--------------------------------------------------------|
| Actions              | Discrete and Continuous                                |
| Parallel API         | Yes                                                    |
| Manual Control       | No                                                     |
| Agents               | `agents= ['observer_1', 'drone_1', ...]                |
| Agents               | 1-n                                                    |
| Action Shapes        | agent-dependent (see agents' class definitions)        |
| Action Values        | agent-dependent (see agents' class definitions)        |
| Observation Shapes   | agent-dependent (see agents' class definitions)        |
| Observation Values   | agent-dependent (see agents' class definitions)        |

Implementation of the Heterogeneous Multi-Agent Challenge
Authors:
Charles Dansereau

### Arguments

``` python
HeMAC_v0.env(max_cycles=900)
``

`max_cycles`:  after max_cycles steps all agents will return done

### Version History

* v0: Initial versions release (1.0.0)

"""

import os
import time

import gymnasium
import gymnasium.spaces
import numpy as np
import pygame
import math
import random
from gymnasium.utils import EzPickle, seeding
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector
from pettingzoo.utils.conversions import parallel_wrapper_fn
from pymap3d import geodetic2enu
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from .world import world_ref_to_game_ref, game_ref_to_world_ref

import hemac.environment.sensors as sensors
from hemac.helpers.logger import LOGGER
from hemac.helpers.reward_shaping import proximity_penalty_from_distances
from .drone import Drone
from .observer import Observer
from .provisioner import Provisioner
from .poi import PointOfInterest
from .world import World

FPS = 15

__all__ = ["env", "RawEnv", "parallel_env"]


class HeMAC:
    """HeMAC environment."""

    def __init__(
        self,
        randomizer,
        time_factor=1,
        area_size=(1000, 1000),
        max_cycles=300,
        render_mode=None,
        render_ratio=1,
        render_fps=30,
        observer_speed=10,
        n_observers=1,
        observer_sensor: dict = None,  # TODO: add observer config
        observer_comm_range=150,  # TODO: put in observer config
        n_drones=1,
        drone_sensor: dict = None,
        drone_config: dict = None,
        n_provisioners=1,
        provisioner_config: dict = None,
        provisioner_sensor: dict = None,  # TODO: move sensors in agent configs
        min_obstacles=2,
        max_obstacles=3,
        rescuing_targets=False,
        known_goals=False,
        geofence_config: dict = None,
        patrol_config: dict = None,
        poi_config: list = None,
        poi_spawn_range: dict = None,
        observer_heading_reward_scale: float = 0.07,
        drone_hazard_penalty_scale: float = 0.2,
        detection_distance_scale: float = 200.0,
        detection_per_point_base: float = 0.5,
        detection_max_total: float = 3.0,
        drone_only_success_min_coverage_ratio: float = 0.7,
        drone_only_success_reward: float = 300.0,
    ):
        self.number_of_POIs = len(poi_config) if poi_config and len(poi_config) else 0
        self.goals = []

        """Overwrite constructor."""
        super().__init__()
        LOGGER.info(f"""
            HeMAC Configuration:
            ---------------
            Time factor (seconds): {time_factor}
            Max Cycles: {max_cycles}
            Render Mode: {render_mode}
            Observer Speed: {observer_speed}
            Number of Observers: {n_observers}
            Drone config: {drone_config}
            Number of Drones: {n_drones}
            Numover of Provisioners: {n_provisioners}
            Min Obstacles: {min_obstacles}
            Max Obstacles: {max_obstacles}
            Known Goals: {known_goals}
            Geofence config: {geofence_config}
            Patrol config: {patrol_config}
            POI config: {poi_config}
            Drone-only success min coverage ratio: {drone_only_success_min_coverage_ratio}
            Drone-only success reward: {drone_only_success_reward}
            """)

        pygame.init()
        self.randomizer = randomizer
        self.time_factor = time_factor
        self.known_goals = known_goals
        self.rescuing_targets = rescuing_targets
        self.global_reward = 0
        self.observer_heading_reward_scale = observer_heading_reward_scale
        self.drone_hazard_penalty_scale = drone_hazard_penalty_scale
        # detection reward params
        self.detection_distance_scale = detection_distance_scale
        self.detection_per_point_base = detection_per_point_base
        self.detection_max_total = detection_max_total
        self.drone_only_success_min_coverage_ratio = float(
            max(0.0, min(drone_only_success_min_coverage_ratio, 1.0))
        )
        self.drone_only_success_reward = float(drone_only_success_reward)

        # players
        self.n_observers = n_observers
        self.n_drones = n_drones
        self.n_provisioners = n_provisioners
        self.num_agents = n_observers + n_drones + n_provisioners
        self.observer_size = 24 // render_ratio

        # self.agents are the keys of the agents, and self.agents_list contains the actual agents instances
        self.agents = ["observer_" + str(i) for i in range(self.n_observers)]
        self.agents = self.agents + ["drone_" + str(i) for i in range(self.n_drones)]
        self.agents = self.agents + ["provisioner_" + str(i) for i in range(self.n_provisioners)]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))
        self.agents_list = []
        self.detected = set()
        self.found_goal = False
        self.finished = False
        self.drone_crash = False
        self.observer_crash = False
        self.drone_crash_to_obstacle = False
        self.observer_crash_to_obstacle = False
        self.max_coverage_ratio = 0.0

        self.old_dist_to_goal = None

        # Display screen
        self.render_ratio = render_ratio
        self.s_width, self.s_height = area_size[0] // render_ratio, area_size[1] // render_ratio
        self.area = pygame.Rect(0, 0, self.s_width, self.s_height)
        self.research_area = pygame.Rect(
            0.1 * self.s_width, 0.1 * self.s_height, 0.8 * self.s_width, 0.8 * self.s_height
        )

        # geofence
        enu_geofence_area = []
        if geofence_config:
            if geofence_config.get("coordinates_type") == "geo":
                for coord in geofence_config.get("area"):
                    enu_geofence_area.append(
                        list(
                            geodetic2enu(
                                coord[0],
                                coord[1],
                                0,
                                geofence_config.get("position_origin", {}).get("latitude"),
                                geofence_config.get("position_origin", {}).get("longitude"),
                                0,
                            )
                        )
                    )
            else:
                enu_geofence_area = geofence_config.get("area")

        self.geofence_area = enu_geofence_area
        # Patrol bookkeeping
        if not patrol_config:
            patrol_config = {}
        self.patrol_benchmark = patrol_config.get("benchmark")
        if self.patrol_benchmark:
            if not patrol_config.get("area"):
                LOGGER.error("patrol area is None")
            enu_search_area = []
            if patrol_config.get("coordinates_type") == "geo":
                for coord in patrol_config.get("area"):
                    enu_search_area.append(
                        list(
                            geodetic2enu(
                                coord[0],
                                coord[1],
                                0,
                                patrol_config.get("position_origin", {}).get("latitude"),
                                patrol_config.get("position_origin", {}).get("longitude"),
                                0,
                            )
                        )
                    )
            else:
                enu_search_area = patrol_config.get("area")

            self.search_area = Polygon(enu_search_area)
        else:
            self.search_area = Polygon(
                (
                    self.research_area.topleft,
                    self.research_area.topright,
                    self.research_area.bottomright,
                    self.research_area.bottomleft,
                )
            )
        
        self.search_grid_rects = {}

        # init POI
        if poi_spawn_range is None:
            minx, miny, maxx, maxy = self.search_area.bounds
            poi_spawn_range = {"x_range": (minx, maxx), "y_range": (miny, maxy)}
            first_poi_config = next((cfg for cfg in (poi_config or []) if cfg), None)
            spawn_quadrant = first_poi_config.get("spawn_quadrant") if first_poi_config else None
            if spawn_quadrant == "bottom_right":
                midx = (minx + maxx) / 2.0
                midy = (miny + maxy) / 2.0
                # World Y grows upward while the rendered map grows downward, so
                # the screen's lower half maps to the lower world-Y range.
                poi_spawn_range = {"x_range": (midx, maxx), "y_range": (miny, midy)}
        self.poi_spawn_range = poi_spawn_range
        for i in range(self.number_of_POIs):
            _poi_config = poi_config[i] if poi_config and poi_config[i] else None
            self.goals.append(
                PointOfInterest(
                    randomizer=self.randomizer,
                    poi_config=_poi_config,
                    time_factor=time_factor,
                    area=self.area,
                    spawn_range=self.poi_spawn_range,
                )
            )
        # init World
        self.world = World(  # TODO: group args in world config
            game_area=self.area,
            geofence_area=self.geofence_area,
            search_area=self.search_area,
            randomizer=randomizer,
            time_factor=self.time_factor,
        )
        self.search_grid_rects = self._build_search_grid_cache()

        # init observers
        for i in range(self.n_observers):
            in_observer_sensor = self._get_sensor(observer_sensor)
            self.agents_list.append(
                Observer(
                    dims=(self.observer_size, self.observer_size),
                    speed=observer_speed,
                    observer_id=i,
                    sensor=in_observer_sensor,
                    time_factor=time_factor,
                    discrete_action_space=False,
                    comm_range=observer_comm_range,
                )
            )

        # init drones
        if drone_config is not None:
            if self.n_drones > len(drone_config.get("drones_starting_pos", 0)):
                LOGGER.warning(f"""Error in Drone Config, found {len(drone_config.get("drones_starting_pos", 0))}
                starting coordinates for {self.n_drones} drones""")
                drone_config["drones_starting_pos"] = []
        for i in range(self.n_drones):
            in_drone_sensor = self._get_sensor(drone_sensor)
            if drone_config is None:
                drone_config = {}
            self.agents_list.append(
                Drone(
                    drone_config=drone_config,
                    number_of_drones=self.n_drones,
                    randomizer=randomizer,
                    drone_id=i,
                    sensor=in_drone_sensor,
                    time_factor=self.time_factor,
                    world=self.world,
                )
            )

        # init provisioners
        for i in range(self.n_provisioners):
            in_provisioner_sensor = self._get_sensor(provisioner_sensor)
            self.agents_list.append(
                Provisioner(
                    provisioner_config=provisioner_config,
                    world=self.world,
                    randomizer=randomizer,
                    provisioner_id=i,
                    time_factor=self.time_factor,
                    render_ratio=render_ratio,
                    sensor=in_provisioner_sensor,
                )
            )

        # define action and observation spaces
        self.action_spaces = dict(zip(self.agents, [agent.action_space for agent in self.agents_list]))
        self.observation_spaces = dict(zip(self.agents, [agent.observation_space for agent in self.agents_list]))

        LOGGER.info(f"action spaces: {self.action_spaces}")
        LOGGER.info(f"observation spaces: {self.observation_spaces}")

        # define the global space of the environment or state
        self.state_space = gymnasium.spaces.MultiBinary(2)
        self.min_obstacles = min_obstacles
        self.max_obstacles = max_obstacles

        self.render_mode = render_mode
        self.screen = None

        self.max_cycles = max_cycles
        self.num_frames = 0

        # to follow consecutive time steps without seeing the POI
        self.steps_without_poi = 0

        # self.world.observer_communication = [0, 0]

        self.mission_success = False
        self.reinit()

        self.render_fps = render_fps
        if self.render_mode == "human":
            self.clock = pygame.time.Clock()

    def _get_sensor(self, sensor: dict) -> sensors.Sensor:
        """Instantiate a sensor from sensor dictionnary.

        Args:
        ----
            sensor (dict): Sensor name and params.
            module (str): Module where sensor object is located.

        Returns:
        -------
            Sensor: Instantiated sensor.

        """
        if sensor is not None and sensor.get("model") is not None:
            obs_sensor_params = sensor.get("params")
            sensor_class = getattr(sensors, sensor.get("model"), None)
            if sensor_class:
                in_sensor = sensor_class(**obs_sensor_params) if obs_sensor_params else sensor_class()
            else:
                in_sensor = sensors.RoundCamera()
        else:
            in_sensor = sensors.RoundCamera()

        return in_sensor

    def reinit(self):
        """Reinitialize the environment."""
        self.rewards = dict(zip(self.agents, [0.0] * len(self.agents)))
        self.terminations = dict(zip(self.agents, [False] * len(self.agents)))
        self.truncations = dict(zip(self.agents, [False] * len(self.agents)))
        self.infos = dict(zip(self.agents, [{}] * len(self.agents)))
        self.score = 0

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        # reset goals
        self.success_step = None
        self.mission_success = False
        for goal in self.goals:
            goal.spawn_poi(self.search_area)
            goal.reset()
        self.explored_grids = set()
        self.observer_explored_grids = set()
        self.world.explored_grids = self.explored_grids
        self.world.observer_explored_grids = self.observer_explored_grids

        if self.render_mode == "human":
            print("resetting world.")
        self.world.reset(self.goals)
        self._sync_goal_position()
        self.world.clear_obstacles()  # Clear obstacles at the start of each episode
        self.detection_reward = 0

        # spawn obstacles
        if self.max_obstacles > 0:  # TODO: reset all world components inside world reset() (obstacles, etc.)
            num_obstacles = self.randomizer.integers(self.min_obstacles, self.max_obstacles)
            goal_rects = [goal.rect for goal in self.goals if goal.rect is not None]
            self.world.generate_obstacles(num_obstacles, avoid_rects=goal_rects)

        # reset agents to initial state
        observer_spawned = False
        base_x, base_y = 0.0, 0.0

        # 1. 유인기 스폰 (기지 중앙 좌표를 훔쳐옴)
        for agent, name in zip(self.agents_list, self.agents):
            if "observer" in name:
                agent.reset()
                
                # 1) World에 있는 기지(Base)의 화면 좌표를 수학 좌표로 변환해 가져옴
                [base_world_x, base_world_y] = game_ref_to_world_ref(self.world.base.center, self.area)
                
                agent.x = base_world_x
                agent.y = base_world_y
                agent.z = 5.0
                
                # 2) 시각적 껍데기도 기지(Base) 정중앙에 완벽하게 일치시킴
                agent.rect.center = self.world.base.center
                agent.sync_pose_state()
                
                base_x, base_y = agent.x, agent.y
                observer_spawned = True
                break

        # 2. 무인기 스폰 (유인기를 중심으로 호위 대형)
        drone_offsets = [(0, 50), (-50, -50), (50, -50), (0, -50)]
        drone_idx = 0
        
        for agent, name in zip(self.agents_list, self.agents):
            if "drone" in name:
                agent.reset()
                
                if getattr(agent, "has_custom_starting_pos", False) and agent.starting_pos is not None:
                    agent.x = agent.starting_pos[0]
                    agent.y = agent.starting_pos[1]
                    agent.z = agent.starting_pos[2] if len(agent.starting_pos) > 2 else 5.0

                    rect_pos = world_ref_to_game_ref([agent.x, agent.y], self.area)
                    agent.rect.centerx = rect_pos[0]
                    agent.rect.centery = rect_pos[1]
                elif observer_spawned:
                    agent.x = base_x + drone_offsets[drone_idx % len(drone_offsets)][0]
                    agent.y = base_y + drone_offsets[drone_idx % len(drone_offsets)][1]
                    agent.z = 5.0
                    
                    rect_pos = world_ref_to_game_ref([agent.x, agent.y], self.area)
                    agent.rect.centerx = rect_pos[0]
                    agent.rect.centery = rect_pos[1]
                else:
                    self.world.spawn_asset(agent, self.agents_list, avoid_world_obstacles=True, set_real_coordinates=True)
                agent.sync_pose_state()
                
                drone_idx += 1

        self.terminate = False
        self.collided = False
        self.truncate = False
        self.found_goal = False
        self.detected = set()
        self.drone_crash = False
        self.observer_crash = False
        self.drone_crash_to_obstacle = False
        self.observer_crash_to_obstacle = False
        self.max_coverage_ratio = 0.0

        self.num_frames = 0
        self.old_dist_to_goal = None

        self.reinit()

        self.time = 1

        # Pygame surface required even for render_mode == None, as observations could be taken from pixel values
        # Observe
        if self.render_mode != "human":
            self.screen = pygame.Surface((self.s_width, self.s_height))
        if self.render_mode is not None:
            self.render()

    def close(self):
        """Close the environment."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            gymnasium.logger.warn("You are calling render method without specifying any render mode.")
            return

        if self.screen is None:
            if self.render_mode == "human":
                os.environ["SDL_VIDEO_WINDOW_POS"] = f"{pygame.display.Info().current_w - 50 - self.s_width},50"
                self.screen = pygame.display.set_mode((self.s_width, self.s_height))
                pygame.display.set_caption("HeMARL")
        self.draw()

        state = np.array(pygame.surfarray.pixels3d(self.screen))
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.render_fps)
        return np.transpose(state, axes=(1, 0, 2)) if self.render_mode == "rgb_array" else None

    def observe(self, agent):
        """Observe the agent."""
        current_agent = self.agents_list[self.agent_name_mapping[agent]]
        observation = current_agent.observe(self.world, self.agents_list, self.goals)
        # LOGGER.info(f"observation for {agent}: {observation}")
        return observation

    def state(self):
        """Return an observation of the global environment."""
        state = np.array([0, 0])
        return state

    def _sync_goal_position(self):
        """Keep the shared goal position aligned with the current goal state."""
        self.world.goal_position = (self.goals[0].x, self.goals[0].y) if self.goals else None

    def _update_detected_cache(self, agent):
        """Merge newly detected coordinates into the shared coverage cache."""
        latest_points = getattr(agent, "latest_detected", agent.detected)
        if isinstance(latest_points, np.ndarray):
            new_points = self.world.register_detected_points(
                latest_points,
                return_new_points=True,
                assume_unique=True,
            )
            if len(new_points) == 0:
                return new_points
            return new_points

        new_points = latest_points.difference(self.world.detected)
        if not new_points:
            return set()

        self.world.register_detected_points(new_points)
        return new_points
    
    def _build_search_grid_cache(self):
        """Build renderable cells aligned to the shared coverage grid."""
        cell_width = self.world.coverage_cell_width
        cell_height = self.world.coverage_cell_height
        grid_rects = {}
        for grid_x in range(self.world.coverage_grid_size):
            for grid_y in range(self.world.coverage_grid_size):
                if self.world.search_mask[grid_y, grid_x] <= 0.0:
                    continue
                left = int(round(grid_x * cell_width))
                right = int(round((grid_x + 1) * cell_width))
                top = int(round(self.area.height - (grid_y + 1) * cell_height))
                bottom = int(round(self.area.height - grid_y * cell_height))
                grid_rects[(grid_y, grid_x)] = pygame.Rect(
                    left,
                    top,
                    max(right - left, 1),
                    max(bottom - top, 1),
                )
        return grid_rects

    def draw_exploration_overlay(self):
        """Draw explored vs unexplored search cells."""
        overlay = pygame.Surface(self.area.size, pygame.SRCALPHA)
        unexplored_color = (38, 57, 84, 70)
        explored_base = (76, 196, 120)
        outline_color = (220, 235, 255, 25)

        for grid_key, rect in self.search_grid_rects.items():
            search_ratio = float(self.world.search_mask[grid_key])
            coverage_value = float(self.world.observation_coverage_map[grid_key])
            if search_ratio <= 0.0:
                cell_color = (0, 0, 0, 0)
            elif coverage_value > 0.0:
                explored_alpha = 80 + int(140 * min(coverage_value, 1.0))
                cell_color = (*explored_base, explored_alpha)
            else:
                unexplored_alpha = int(unexplored_color[-1] * search_ratio)
                cell_color = (*unexplored_color[:3], unexplored_alpha)
            pygame.draw.rect(overlay, cell_color, rect)
            pygame.draw.rect(overlay, outline_color, rect, width=1)

        self.screen.blit(overlay, (0, 0))

        legend_font = pygame.font.SysFont("Trebuchet MS", 16)
        legend_bg = pygame.Surface((190, 80), pygame.SRCALPHA)
        legend_bg.fill((8, 12, 16, 150))
        self.screen.blit(legend_bg, (12, 40))

        pygame.draw.rect(self.screen, (*explored_base, 220), pygame.Rect(22, 50, 18, 18))
        pygame.draw.rect(self.screen, unexplored_color, pygame.Rect(22, 74, 18, 18))

        explored_label = legend_font.render("explored", True, (240, 248, 255))
        unexplored_label = legend_font.render("unexplored", True, (240, 248, 255))
        self.screen.blit(explored_label, (48, 50))
        self.screen.blit(unexplored_label, (48, 74))

    def draw(self):
        """Draw the environment."""
        pygame.event.pump()
        self.world.draw(self.screen)
        self.draw_exploration_overlay()
        for agent in self.agents_list:
            agent.draw(self.screen)
        for goal in self.goals:
            goal.draw(self.screen)
    
    def current_explored_area(self):
        """Return explored area inside the active search area for this episode."""
        search_cell_coverage = np.minimum(self.world.coverage_map, self.world.search_mask)
        return float(np.sum(search_cell_coverage) * self.world.coverage_cell_area)

    def current_coverage_ratio(self):
        """Return the explored ratio inside the active search area."""
        total_search_area = float(self.search_area.area)
        if total_search_area <= 0:
            return 0.0
        explored_area = self.current_explored_area()
        return min(explored_area / total_search_area, 1.0)

    def is_drone_only_mode(self):
        """Return True when the mission is trained with drones only."""
        return self.n_observers == 0 and self.n_drones > 0

    def _mark_mission_success(self, active_agent=None, reward_dict=None, reward_bonus=0.0):
        """Mark the current episode as a success and stop it."""
        if self.mission_success:
            return False

        reward_bonus = float(reward_bonus)
        if reward_bonus != 0.0:
            self.global_reward += reward_bonus
            if reward_dict is not None and active_agent in reward_dict:
                reward_dict[active_agent].append(reward_bonus)

        self.success_step = self.num_frames
        self.mission_success = True
        self.terminate = True
        return True

    def _check_drone_only_mission_success(self, active_agent=None, reward_dict=None):
        """Succeed once drones have found the goal and explored enough area."""
        if not self.is_drone_only_mode():
            return False
        if self.terminate or self.mission_success or not self.found_goal:
            return False

        if self.current_coverage_ratio() < self.drone_only_success_min_coverage_ratio:
            return False

        return self._mark_mission_success(
            active_agent=active_agent,
            reward_dict=reward_dict,
            reward_bonus=self.drone_only_success_reward,
        )

    def _compute_drone_detection_reward(self, newly_detected_points) -> float:
        """Return a bounded, non-saturating reward for goal-proximal exploration."""
        if not self.goals:
            return 0.0

        if isinstance(newly_detected_points, np.ndarray):
            points = newly_detected_points.astype(np.float32, copy=False)
        else:
            points = np.asarray(list(newly_detected_points), dtype=np.float32)

        if points.size == 0:
            return 0.0
        if points.ndim != 2 or points.shape[1] != 2:
            points = np.reshape(points, (-1, 2))

        goal_positions = np.asarray([(goal.x, goal.y) for goal in self.goals], dtype=np.float32)
        deltas = points[:, None, :] - goal_positions[None, :, :]
        min_dists = np.linalg.norm(deltas, axis=2).min(axis=1)
        scale = max(float(self.detection_distance_scale), 1e-6)
        weights = np.clip((scale - min_dists) / scale, 0.0, 1.0)
        weighted_hits = float(np.sum(weights))
        if weighted_hits <= 0.0:
            return 0.0

        reward = float(self.detection_per_point_base) * math.log1p(weighted_hits)
        if self.detection_max_total > 0:
            reward = min(reward, float(self.detection_max_total))
        return reward

    def build_episode_info(self):
        """Build a final-episode info dict for metrics and evaluation."""
        coverage_ratio = self.current_coverage_ratio()
        total_explored = self.current_explored_area()
        self.max_coverage_ratio = max(self.max_coverage_ratio, coverage_ratio)

        # goal_found_step = self.goal_found_step if self.goal_found_step is not None else self.max_cycles
        success_step = self.success_step if self.success_step is not None else self.max_cycles
        # if self.goal_found_step is None:
        #     steps_after_goal_found = self.max_cycles
        # elif self.success_step is None:
        #     steps_after_goal_found = self.num_frames - self.goal_found_step
        # else:
        #     steps_after_goal_found = self.success_step - self.goal_found_step

        return {
            "success": bool(self.mission_success),
            "goal_found": bool(self.found_goal),
            # "goal_known": bool(self.world.goal_known),
            "fatal_crash": bool(self.collided),
            "timeout": bool(self.truncate and not self.terminate),
            "drone_crash": bool(self.drone_crash),
            "observer_crash": bool(self.observer_crash),
            "drone_crash_to_obstacle": bool(self.drone_crash_to_obstacle),
            "observer_crash_to_obstacle": bool(self.observer_crash_to_obstacle),
            # "min_drone_dist": float(self.min_drone_dist),
            # "min_obs_dist": float(self.min_obs_dist),
            "explored_area": total_explored,
            "total_explored": total_explored,
            "coverage_ratio": float(coverage_ratio),
            "max_coverage_ratio": float(self.max_coverage_ratio),
            # "goal_found_step": float(goal_found_step),
            "success_step": float(success_step),
            # "steps_after_goal_found": float(max(steps_after_goal_found, 0)),
            "success_after_goal_found": bool(self.mission_success),
        }

    def _propagate_episode_state(self, *, include_global_reward: bool):
        """Push current rewards, terminations, and infos to all agents."""
        episode_info = self.build_episode_info()
        for ag in self.agents:
            if include_global_reward:
                self.rewards[ag] += self._global_reward_for_agent(ag)
            self.terminations[ag] = self.terminate
            self.truncations[ag] = self.truncate
            self.infos[ag] = dict(episode_info)

    def finalize_episode(self):
        """Propagate the current end-of-episode state to every agent."""
        self._propagate_episode_state(include_global_reward=True)

    def _global_reward_for_agent(self, agent_name):
        """Split shared success reward by role for clearer credit assignment."""
        if "observer" in agent_name:
            return self.global_reward
        if "drone" in agent_name:
            return 0.25 * self.global_reward
        return 0.0

    def step(self, action, active_agent):
        """Execute a step."""
        if active_agent == self.agents[0]:
            self.global_reward = 0
        found_goal = False
        delivered_goal = False
        reward = 0
        reward_dict = dict(zip(self.agents, [[] for _ in self.agents])) # reward logging용 dictionary
        # LOGGER.info(f'reward_dict: {reward_dict}')

        agent = self.agents_list[self.agent_name_mapping[active_agent]]
        agent.update(self.area, self.world, action, self.found_goal)
        self.world.reveal_warning_zones_for_agent(agent)

        # Update position and uncertainty of objectives
        for goal in self.goals:
            goal.move(self.world.obstacles, self.search_area)
        self._sync_goal_position()

        # Specific actions for UAVs
        if "drone" in active_agent:
            # Collision check and map limits
            reward -= 0.05  # Step penalty
            reward_dict[active_agent].append(-0.05)

            if not self.world.point_in_search_area(agent.x, agent.y):  # 맵 밖으로 나간 경우
                self.collided = True
                self.drone_crash = True
                self.terminate = True
                reward -= 300
                reward_dict[active_agent].append(-300)
                if self.render_mode == "human" or self.render_mode == "rgb_array":
                    LOGGER.info(f"drone went out of search area. pos: {(agent.x, agent.y)}")
            elif self.world.is_in_warning_zone(agent.x, agent.y):  # 위험구역에 들어간 경우
                if random.random() < 0.5:  # 50% 확률로 충돌 처리
                    self.collided = True
                    self.drone_crash = True
                    self.drone_crash_to_warning_zone = True
                    self.terminate = True
                    reward -= 300  # Penalty for entering a warning zone
                    reward_dict[active_agent].append(-300)
                # else:
                #     reward -= 50  # Penalty for entering a warning zone without collision
                #     reward_dict[active_agent].append(-50)
                if self.render_mode == "human" or self.render_mode == "rgb_array":
                    LOGGER.info(f"drone entered a warning zone. pos: {(agent.x, agent.y)}")
            else:
                obstacle_idx = agent.rect.collidelist(self.world.obstacles)
                if obstacle_idx != -1:
                    obstacle = self.world.obstacles[obstacle_idx]
                    self.collided = True
                    self.drone_crash = True
                    self.drone_crash_to_obstacle = True
                    self.terminate = True
                    reward -= 300  # Penalty for collision with an obstacle
                    reward_dict[active_agent].append(-300)
                    if self.render_mode == "human" or self.render_mode == "rgb_array":
                        LOGGER.info(
                            f"agent {active_agent} collided with obstacle at position [x,y] = {obstacle.center}"
                        )

            safe_radius = 75.0 # drone sensing range로 설정
            drone_proximity_penalty = 0.0
            for other_agent in self.agents_list:
                if other_agent is agent or not isinstance(other_agent, Drone):
                    continue
                drone_distance = dist(other_agent.x, other_agent.y, agent.x, agent.y) # 다른 drone과의 거리
                if drone_distance < safe_radius:
                    normalized_gap = (safe_radius - drone_distance) / safe_radius
                    drone_proximity_penalty += 0.5 * (normalized_gap ** 2) # 드론 간의 근접 패널티
            if drone_proximity_penalty > 0:
                reward -= drone_proximity_penalty
                reward_dict[active_agent].append(-drone_proximity_penalty)

            # POI tracking reward calculation
            for goal in self.goals[:]:
                goal_dist = dist(goal.x, goal.y, agent.x, agent.y)
                if goal_dist < agent.sensing_range and not self.found_goal:
                    agent.found_goal = True
                    self.found_goal = True
                    reward += 20  # goal 탐색 시
                    reward_dict[active_agent].append(20)

            # proximity-weighted detection reward: closer detections to any goal give more reward
            newly_detected_points = self._update_detected_cache(agent)
            if len(newly_detected_points) > 0:
                # total_detection_reward = min(self._compute_drone_detection_reward(newly_detected_points), 0.1)
                total_detection_reward = len(newly_detected_points) / 50000
                reward += total_detection_reward
                reward_dict[active_agent].append(total_detection_reward)

            self._check_drone_only_mission_success(active_agent=active_agent, reward_dict=reward_dict)

        elif "observer" in active_agent:
            reward -= 0.05  # step penalty
            reward_dict[active_agent].append(-0.05)

            if not self.world.point_in_search_area(agent.x, agent.y):
                self.collided = True
                self.observer_crash = True
                self.terminate = True
                reward -= 300  # going outside of search area
                reward_dict[active_agent].append(-300)
                if self.render_mode == "human" or self.render_mode == "rgb_array":
                    LOGGER.info(f"drone went out of search area. pos: {(agent.x, agent.y)}")
            elif self.world.is_in_warning_zone(agent.x, agent.y):
                if random.random() < 0.5:  # 50% 확률로 충돌 처리
                    self.collided = True
                    self.observer_crash = True
                    self.terminate = True
                    self.observer_crash_to_obstacle = True
                    reward -= 300  # Penalty for entering a warning zone
                    reward_dict[active_agent].append(-300)
                # else:
                #     reward -= 50  # Penalty for entering a warning zone without collision
                #     reward_dict[active_agent].append(-50)
                if self.render_mode == "human" or self.render_mode == "rgb_array":
                    LOGGER.info(f"observer entered a warning zone. pos: {(agent.x, agent.y)}")
            else:
                obstacle_idx = agent.rect.collidelist(self.world.obstacles)
                if obstacle_idx != -1:
                    obstacle = self.world.obstacles[obstacle_idx]
                    self.collided = True
                    self.observer_crash = True
                    self.observer_crash_to_obstacle = True
                    self.terminate = True
                    reward -= 300  # Penalty for collision with an obstacle
                    reward_dict[active_agent].append(-300)
                    if self.render_mode == "human" or self.render_mode == "rgb_array":
                        LOGGER.info(
                            f"agent {active_agent} collided with obstacle at position [x,y] = {obstacle.center}"
                        )

            closest_goal = None
            current_dist = float('inf')
            for goal in self.goals[:]:
                goal_dist = dist(goal.x, goal.y, agent.x, agent.y)
                if goal_dist < current_dist:
                    current_dist = goal_dist
                    closest_goal = goal
                
                if closest_goal is not None:
                    # Initialize the running best distance on the first valid step.
                    if not np.isfinite(getattr(agent, "min_dist_record", np.inf)):
                        agent.min_dist_record = current_dist

                    # Reward only when the observer sets a new closest-distance record.
                    elif current_dist < agent.min_dist_record:
                        progress = agent.min_dist_record - current_dist
                        progress_reward = progress * 0.05
                        if np.isfinite(progress_reward) and progress_reward > 0:
                            reward += progress_reward
                            reward_dict[active_agent].append(progress_reward)

                        heading_reward = heading_alignment_reward(
                            agent.x, agent.y, agent.orientation, closest_goal.x, closest_goal.y, self.observer_heading_reward_scale
                        )
                        if heading_reward > 0:
                            reward += heading_reward
                            reward_dict[active_agent].append(heading_reward)

                        agent.min_dist_record = current_dist

                if goal_dist < agent.sensing_range:  # goal까지의 거리가 sensing range보다 가까워지면 발견
                    agent.found_goal = True
                    self.found_goal = True
                    reward += 300
                    reward_dict[active_agent].append(300)
                    self._mark_mission_success(
                        active_agent=active_agent,
                        reward_dict=reward_dict,
                    )

            newly_detected_count = self._update_detected_cache(agent)
            # if newly_detected_count > 0:
            #     detection_reward = math.sqrt(math.sqrt(newly_detected_count)) / 20
            #     reward += detection_reward
            #     reward_dict[active_agent].append(detection_reward)

        # individual reward
        self.rewards[active_agent] = reward
        LOGGER.info(f"reward for {active_agent} at step {self.num_frames}: {reward_dict[active_agent]}")
        self.finalize_episode()

        # Update environment and check end of episode
        if agent == self.agents_list[-1]:
            if self.collided:
                self.terminate = True
                if self.render_mode == "human":
                    LOGGER.info(f"BOOM! episode length: {self.num_frames + 1}")
                    self.render()
                    time.sleep(2)

            self.world.update(self.area)

            # Termination or continuation of the episode
            if not self.terminate:
                self.num_frames += 1
                self.truncate = self.num_frames >= self.max_cycles

            if self.terminate or self.truncate:
                pass

            # Refresh episode info after the frame counter/truncation state changes.
            self._propagate_episode_state(include_global_reward=False)

            if self.render_mode is not None:
                self.render()
                if self.render_mode == "human":
                    # input()  # Toggle: slow down simulation to make prints more readable
                    pass


def env(**kwargs):
    """Env."""
    env = RawEnv(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


class RawEnv(AECEnv, EzPickle):
    """Raw environment."""

    # class env(MultiAgentEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "HeMAC_v0",
        "is_parallelizable": True,
        "render_fps": FPS,
        "has_manual_policy": True,
    }

    def __init__(self, **kwargs):
        """Overwrite the default constructor."""
        EzPickle.__init__(self, **kwargs)
        self._kwargs = kwargs

        self.randomizer = None
        self._seed()
        self.env = HeMAC(self.randomizer, **self._kwargs)

        self.agents = self.env.agents[:]
        self.possible_agents = self.agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        # spaces
        self.action_spaces = self.env.action_spaces
        self.observation_spaces = self.env.observation_spaces
        self.state_space = self.env.state_space
        # dicts
        self.observations = {}
        self.rewards = self.env.rewards
        self.terminations = self.env.terminations
        self.truncations = self.env.truncations
        self.infos = self.env.infos

        self.score = self.env.score

        self.render_mode = self.env.render_mode
        self.screen = None

    def observation_space(self, agent):
        """Return observation space."""
        return self.observation_spaces[agent]

    def action_space(self, agent):
        """Return action space."""
        return self.action_spaces[agent]

    def _seed(self, seed=None):
        self.randomizer, seed = seeding.np_random(seed)

    def reset(self, seed=None, options=None):
        """Reset environment."""
        if seed is not None:
            self._seed(seed=seed)
        self.env.randomizer = self.randomizer
        self.env.reset()
        self.agents = self.possible_agents[:]
        self.agent_selection = self._agent_selector.reset()
        self.rewards = self.env.rewards
        self._cumulative_rewards = {a: 0 for a in self.agents}
        self.terminations = self.env.terminations
        self.truncations = self.env.truncations
        self.infos = self.env.infos

    def observe(self, agent):
        """Observe agent."""
        obs = self.env.observe(agent)

        if not self.observation_spaces[agent].contains(obs):
            raise Exception(f"obs for agent {agent} must be in {self.observation_spaces[agent]}. It is currently {obs}")

        return obs

    def state(self):
        """Return state."""
        state = self.env.state()
        return state

    def close(self):
        """Close environment."""
        self.env.close()

    def render(self):
        """Render environment."""
        return self.env.render()

    def step(self, action):
        """Step environment."""
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        if not self.action_spaces[agent].contains(action):
            raise Exception(
                f"Action for agent {agent} must be in {self.action_spaces[agent]}. It is currently {action}"
            )

        self.env.rewards = {a: 0 for a in self.agents}
        self.env.step(action, agent)

        # select next agent and observe
        self.agent_selection = self._agent_selector.next()
        self.rewards = self.env.rewards
        self.terminations = self.env.terminations
        self.truncations = self.env.truncations
        self.infos = self.env.infos

        self.score = self.env.score

        self._cumulative_rewards[agent] = 0
        self._accumulate_rewards()


def dist(x1, y1, x2, y2):
    """Return distance between two points."""
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def heading_alignment_reward(agent_x, agent_y, orientation, goal_x, goal_y, reward_scale):
    """Return a bounded shaping reward when the observer faces the goal."""
    if reward_scale <= 0:
        return 0.0

    dx = goal_x - agent_x
    dy = goal_y - agent_y
    if np.isclose(dx, 0.0) and np.isclose(dy, 0.0):
        return 0.0

    goal_heading = math.atan2(dy, dx)
    heading_error = math.atan2(math.sin(goal_heading - orientation), math.cos(goal_heading - orientation))
    return max(math.cos(heading_error), 0.0) * reward_scale


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
