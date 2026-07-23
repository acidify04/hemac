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
        n_static_obstacles=1,
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
        obstacle_min_speed: int | None = None,
        obstacle_max_speed: int | None = None,
        goal_min_base_distance: float = 0.0,
        goal_max_base_distance: float | None = None,
        log_step_rewards: bool = False,
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
            Static Obstacles: {n_static_obstacles}
            Known Goals: {known_goals}
            Geofence config: {geofence_config}
            Patrol config: {patrol_config}
            POI config: {poi_config}
            Drone-only success min coverage ratio: {drone_only_success_min_coverage_ratio}
            Drone-only success reward: {drone_only_success_reward}
            Obstacle speed range: ({obstacle_min_speed}, {obstacle_max_speed})
            Goal distance from base: ({goal_min_base_distance}, {goal_max_base_distance})
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
        self.goal_min_base_distance = max(float(goal_min_base_distance), 0.0)
        self.goal_max_base_distance = (
            None
            if goal_max_base_distance is None
            else max(float(goal_max_base_distance), self.goal_min_base_distance)
        )
        self.log_step_rewards = bool(log_step_rewards)

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
            obstacle_min_speed=obstacle_min_speed,
            obstacle_max_speed=obstacle_max_speed,
        )
        self.search_grid_rects = self._build_search_grid_cache()
        self.drone_reward_sector_mask = np.ones(
            (self.world.coverage_grid_size, self.world.coverage_grid_size),
            dtype=bool,
        )
        active_rows = np.flatnonzero(np.any(self.world.search_mask > 0.0, axis=1))
        active_columns = np.flatnonzero(np.any(self.world.search_mask > 0.0, axis=0))
        # Coverage y increases upward, so the final active rows are visually at the top.
        top_rows = active_rows[-4:]
        left_columns = active_columns[:4]
        self.drone_reward_sector_mask[np.ix_(top_rows, left_columns)] = False

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
        self.n_static_obstacles = max(int(n_static_obstacles), 0)

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

    def set_obstacle_difficulty(
        self,
        *,
        min_obstacles: int | None = None,
        max_obstacles: int | None = None,
        obstacle_min_speed: int | None = None,
        obstacle_max_speed: int | None = None,
        goal_min_base_distance: float | None = None,
        goal_max_base_distance: float | None = None,
    ) -> None:
        """Update obstacle and goal-spawn difficulty for subsequent resets."""
        if min_obstacles is not None or max_obstacles is not None:
            next_min_obstacles = self.min_obstacles if min_obstacles is None else int(min_obstacles)
            next_max_obstacles = self.max_obstacles if max_obstacles is None else int(max_obstacles)
            self.min_obstacles = max(next_min_obstacles, 0)
            self.max_obstacles = max(next_max_obstacles, self.min_obstacles)

        if obstacle_min_speed is not None or obstacle_max_speed is not None:
            current_min_speed = getattr(self.world, "obstacle_min_speed", None)
            current_max_speed = getattr(self.world, "obstacle_max_speed", None)
            self.world.set_obstacle_speed_range(
                current_min_speed if obstacle_min_speed is None else obstacle_min_speed,
                current_max_speed if obstacle_max_speed is None else obstacle_max_speed,
            )

        if goal_min_base_distance is not None:
            self.goal_min_base_distance = max(float(goal_min_base_distance), 0.0)
        if goal_max_base_distance is not None:
            self.goal_max_base_distance = max(
                float(goal_max_base_distance),
                self.goal_min_base_distance,
            )
        elif (
            self.goal_max_base_distance is not None
            and self.goal_max_base_distance < self.goal_min_base_distance
        ):
            self.goal_max_base_distance = self.goal_min_base_distance

    def _spawn_goal(self, goal, *, obstacles=None, warning_zone_checker=None):
        """Spawn one goal within the current curriculum distance from the base."""
        base_position = game_ref_to_world_ref(self.world.base.center, self.area)
        return goal.spawn_poi(
            self.search_area,
            obstacles=obstacles,
            warning_zone_checker=warning_zone_checker,
            base_position=base_position,
            min_base_distance=self.goal_min_base_distance,
            max_base_distance=self.goal_max_base_distance,
        )

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        # reset goals
        self.success_step = None
        self.mission_success = False
        for goal in self.goals:
            goal.reset()
        self.explored_grids = set()
        self.observer_explored_grids = set()
        self.world.explored_grids = self.explored_grids
        self.world.observer_explored_grids = self.observer_explored_grids

        if self.render_mode == "human":
            print("resetting world.")
        self.world.reset(self.goals)
        self.world.clear_obstacles()  # Clear obstacles at the start of each episode
        for goal in self.goals:
            self._spawn_goal(goal)
        self._sync_goal_position()
        self.detection_reward = 0

        # spawn obstacles
        min_obstacles = max(int(self.min_obstacles), 0)
        max_obstacles = max(int(self.max_obstacles), min_obstacles)
        if max_obstacles > 0 or self.n_static_obstacles > 0:
            num_obstacles = (
                int(self.randomizer.integers(min_obstacles, max_obstacles + 1))
                if max_obstacles > 0
                else 0
            )
            goal_rects = [goal.rect for goal in self.goals if goal.rect is not None]
            self.world.generate_obstacles(
                num_obstacles,
                avoid_rects=goal_rects,
                n_static_obstacles=self.n_static_obstacles,
            )
            for goal in self.goals:
                self._spawn_goal(
                    goal,
                    obstacles=self.world.obstacles,
                    warning_zone_checker=self.world.game_rect_intersects_warning_zone,
                )
            self._sync_goal_position()

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

    def set_randomizer(self, randomizer):
        """Propagate a reset seed to every component that samples randomness."""
        self.randomizer = randomizer
        self.world.randomizer = randomizer
        for goal in self.goals:
            goal.randomizer = randomizer
        for agent in self.agents_list:
            if hasattr(agent, "randomizer"):
                agent.randomizer = randomizer
            if hasattr(agent, "IMU"):
                agent.IMU.randomizer = randomizer
            if hasattr(agent, "UWB"):
                agent.UWB.randomizer = randomizer

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

    def _update_detected_cache(self, agent, counted_sector_mask=None):
        """Merge newly detected coordinates into the shared coverage cache."""
        detection_center = getattr(agent, "latest_detection_center", None)
        if detection_center is not None:
            return self.world.register_detected_disk(
                detection_center[0],
                detection_center[1],
                getattr(agent, "latest_detection_radius", 0),
                counted_sector_mask=counted_sector_mask,
            )

        latest_points = getattr(agent, "latest_detected", agent.detected)
        if isinstance(latest_points, np.ndarray):
            new_points = self.world.register_detected_points(
                latest_points,
                return_new_points=True,
                assume_unique=True,
            )
            return self._count_points_in_sectors(new_points, counted_sector_mask)

        new_points = latest_points.difference(self.world.detected)
        if not new_points:
            return 0

        self.world.register_detected_points(new_points)
        return self._count_points_in_sectors(new_points, counted_sector_mask)

    def _count_points_in_sectors(self, points, sector_mask):
        """Count points whose coverage-grid sectors are enabled by a boolean mask."""
        if sector_mask is None:
            return len(points)
        points_array = np.asarray(list(points) if not isinstance(points, np.ndarray) else points, dtype=np.int32)
        if points_array.size == 0:
            return 0
        points_array = points_array.reshape(-1, 2)
        pixel_x = np.clip(points_array[:, 0], 0, self.world.area.width - 1)
        pixel_y = np.clip(points_array[:, 1], 0, self.world.area.height - 1)
        grid_x = self.world.pixel_to_grid_x[pixel_x]
        grid_y = self.world.pixel_to_grid_y[pixel_y]
        return int(np.count_nonzero(sector_mask[grid_y, grid_x]))
    
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

    def current_drone_reward_explored_area(self):
        """Return explored area eligible for drone rewards, excluding the base sectors."""
        search_cell_coverage = np.minimum(self.world.coverage_map, self.world.search_mask)
        return float(
            np.sum(search_cell_coverage, where=self.drone_reward_sector_mask)
            * self.world.coverage_cell_area
        )

    def current_drone_reward_coverage_ratio(self):
        """Return drone-reward coverage over only eligible search sectors."""
        eligible_search_area = float(
            np.sum(self.world.search_mask, where=self.drone_reward_sector_mask)
            * self.world.coverage_cell_area
        )
        if eligible_search_area <= 0.0:
            return 0.0
        return min(self.current_drone_reward_explored_area() / eligible_search_area, 1.0)

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

        if self.current_drone_reward_coverage_ratio() < self.drone_only_success_min_coverage_ratio:
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
        drone_reward_coverage_ratio = self.current_drone_reward_coverage_ratio()
        drone_reward_explored_area = self.current_drone_reward_explored_area()
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
            "drone_reward_coverage_ratio": float(drone_reward_coverage_ratio),
            "drone_reward_explored_area": float(drone_reward_explored_area),
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

    @staticmethod
    def _points_near_segment(points, start, end, radius):
        """Return a mask for points within radius of a movement segment."""
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if points.size == 0:
            return np.zeros((0,), dtype=bool)

        start = np.asarray(start, dtype=np.float32)
        end = np.asarray(end, dtype=np.float32)
        segment = end - start
        segment_length_sq = float(np.dot(segment, segment))
        if segment_length_sq <= 1e-12:
            closest = np.broadcast_to(start, points.shape)
        else:
            relative = points - start
            projection = np.clip((relative @ segment) / segment_length_sq, 0.0, 1.0)
            closest = start + projection[:, None] * segment

        offsets = points - closest
        distance_sq = np.einsum("ij,ij->i", offsets, offsets)
        return distance_sq <= float(radius) ** 2

    def _agent_path_intersects_warning_zone(self, start, end):
        """Return True when an agent movement segment crosses a warning zone."""
        centers = self.world.obstacle_warning_centers_world
        if centers.size == 0:
            return False
        return bool(
            np.any(
                self._points_near_segment(
                    centers,
                    start,
                    end,
                    self.world.OBSTACLE_WARNING_RADIUS,
                )
            )
        )

    def _mark_agent_crash(self, agent):
        """Set the shared and role-specific crash state for one agent."""
        self.collided = True
        self.terminate = True
        if isinstance(agent, Drone):
            self.drone_crash = True
            self.drone_crash_to_obstacle = True
        elif isinstance(agent, Observer):
            self.observer_crash = True
            self.observer_crash_to_obstacle = True

    def _apply_obstacle_motion_collisions(self, previous_centers, reward_dict):
        """Crash agents crossed by obstacles during the latest world update."""
        current_centers = self.world.obstacle_warning_centers_world
        previous_centers = np.asarray(previous_centers, dtype=np.float32).reshape(-1, 2)
        if (
            previous_centers.size == 0
            or current_centers.size == 0
            or len(previous_centers) != len(current_centers)
            or not self.agents_list
        ):
            return []

        agent_positions = np.asarray(
            [(agent.x, agent.y) for agent in self.agents_list],
            dtype=np.float32,
        )
        hit_mask = np.zeros((len(self.agents_list),), dtype=bool)
        for previous_center, current_center in zip(previous_centers, current_centers):
            hit_mask |= self._points_near_segment(
                agent_positions,
                previous_center,
                current_center,
                self.world.OBSTACLE_WARNING_RADIUS,
            )

        crashed_agent_names = []
        for agent_idx in np.flatnonzero(hit_mask):
            agent_idx = int(agent_idx)
            agent = self.agents_list[agent_idx]
            agent_name = self.agents[agent_idx]
            self._mark_agent_crash(agent)
            self.rewards[agent_name] -= 300
            reward_dict[agent_name].append(-300)
            crashed_agent_names.append(agent_name)

        return crashed_agent_names

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
        previous_agent_position = (float(agent.x), float(agent.y))
        agent.update(self.area, self.world, action, self.found_goal)
        self.world.update_obstacle_observations_for_agent(agent)

        # Update position and uncertainty of objectives
        for goal in self.goals:
            goal.move(
                self.world.obstacles,
                self.search_area,
                warning_zone_checker=self.world.game_rect_intersects_warning_zone,
            )
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
            elif self._agent_path_intersects_warning_zone(
                previous_agent_position,
                (agent.x, agent.y),
            ):
                self._mark_agent_crash(agent)
                reward -= 300
                reward_dict[active_agent].append(-300)
                if self.render_mode == "human" or self.render_mode == "rgb_array":
                    LOGGER.info(f"drone crossed a warning zone. pos: {(agent.x, agent.y)}")

            if not self.terminate:
                safe_radius = 75.0
                drone_proximity_penalty = 0.0
                for other_agent in self.agents_list:
                    if other_agent is agent or not isinstance(other_agent, Drone):
                        continue
                    drone_distance = dist(other_agent.x, other_agent.y, agent.x, agent.y)
                    if drone_distance < safe_radius:
                        normalized_gap = (safe_radius - drone_distance) / safe_radius
                        drone_proximity_penalty += 0.5 * (normalized_gap ** 2)
                if drone_proximity_penalty > 0:
                    reward -= drone_proximity_penalty
                    reward_dict[active_agent].append(-drone_proximity_penalty)

                for goal in self.goals[:]:
                    goal_dist = dist(goal.x, goal.y, agent.x, agent.y)
                    if goal_dist < agent.sensing_range and not self.found_goal:
                        agent.found_goal = True
                        self.found_goal = True
                        reward += 20
                        reward_dict[active_agent].append(20)

                newly_detected_count = self._update_detected_cache(
                    agent,
                    counted_sector_mask=self.drone_reward_sector_mask,
                )
                if newly_detected_count > 0:
                    total_detection_reward = newly_detected_count / 50000
                    reward += total_detection_reward
                    reward_dict[active_agent].append(total_detection_reward)

                self._check_drone_only_mission_success(
                    active_agent=active_agent,
                    reward_dict=reward_dict,
                )

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
            elif self._agent_path_intersects_warning_zone(
                previous_agent_position,
                (agent.x, agent.y),
            ):
                self._mark_agent_crash(agent)
                reward -= 300
                reward_dict[active_agent].append(-300)
                if self.render_mode == "human" or self.render_mode == "rgb_array":
                    LOGGER.info(f"observer crossed a warning zone. pos: {(agent.x, agent.y)}")

            if not self.terminate:
                closest_goal = None
                current_dist = float("inf")
                for goal in self.goals[:]:
                    goal_dist = dist(goal.x, goal.y, agent.x, agent.y)
                    if goal_dist < current_dist:
                        current_dist = goal_dist
                        closest_goal = goal

                    if closest_goal is not None:
                        if not np.isfinite(getattr(agent, "min_dist_record", np.inf)):
                            agent.min_dist_record = current_dist
                        elif current_dist < agent.min_dist_record:
                            progress = agent.min_dist_record - current_dist
                            progress_reward = progress * 0.01
                            if np.isfinite(progress_reward) and progress_reward > 0:
                                reward += progress_reward
                                reward_dict[active_agent].append(progress_reward)
                            agent.min_dist_record = current_dist

                    if goal_dist < agent.sensing_range:
                        agent.found_goal = True
                        self.found_goal = True
                        self.global_reward += 300
                        reward_dict[active_agent].append(300)
                        success_marked = self._mark_mission_success(
                            active_agent=active_agent,
                            reward_dict=reward_dict,
                        )
                        if success_marked:
                            break

                if not self.terminate:
                    self._update_detected_cache(agent)

        # individual reward
        self.rewards[active_agent] = reward

        # Update environment and check end of episode
        if agent == self.agents_list[-1]:
            if not self.terminate:
                previous_obstacle_centers = self.world.obstacle_warning_centers_world.copy()
                self.world.update(self.area, self.agents_list)
                self._apply_obstacle_motion_collisions(
                    previous_obstacle_centers,
                    reward_dict,
                )

            # Termination or continuation of the episode
            if not self.terminate:
                self.num_frames += 1
                self.truncate = self.num_frames >= self.max_cycles

        if self.log_step_rewards:
            LOGGER.info(f"reward for {active_agent} at step {self.num_frames}: {reward_dict[active_agent]}")
        self.finalize_episode()

        if agent == self.agents_list[-1]:
            if self.collided and self.render_mode == "human":
                LOGGER.info(f"BOOM! episode length: {self.num_frames + 1}")
                self.render()
                time.sleep(2)
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
        self.env.set_randomizer(self.randomizer)
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
