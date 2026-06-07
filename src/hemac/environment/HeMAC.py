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
import math
from hemac.helpers.logger import LOGGER
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
        min_obstacles=0,
        max_obstacles=0,
        rescuing_targets=False,
        known_goals=False,
        geofence_config: dict = None,
        patrol_config: dict = None,
        poi_config: list = None,
        poi_spawn_range: dict = None,
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
            """)

        pygame.init()
        self.randomizer = randomizer
        self.time_factor = time_factor
        self.known_goals = known_goals
        self.rescuing_targets = rescuing_targets
        self.global_reward = 0

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

        self.exploration_cell_size = 20
        self.observer_exploration_cell_size = 30
        self.search_grid_rects = self._build_search_grid_cache(self.exploration_cell_size)
        self.search_grid_centers = {
            key: ((key[0] + 0.5) * self.exploration_cell_size, (key[1] + 0.5) * self.exploration_cell_size)
            for key in self.search_grid_rects
        }

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
        )

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
                    discrete_action_space=True,
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

        self.found_goal = False
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
        self.explored_grids = set() # 이거 안 지우면 다음 에피소드에서 정찰 보상 다 뺏김
        self.observer_explored_grids = set()
        self.min_drone_dist = 99999.0 # 거리 초기화 필수
        self.min_obs_dist = 99999.0
        self.found_goal = False
        
        # 2. 목표물 리셋
        for goal in self.goals:
            goal.detected_by_drone = False
        for goal in self.goals:
            goal.spawn_poi(self.search_area)
            goal.reset()

        if self.render_mode == "human":
            print("resetting world.")
        self.world.reset(self.goals)
        self.world.clear_obstacles()  # Clear obstacles at the start of each episode
        self.detection_reward = 0

        # spawn obstacles
        if self.max_obstacles > 0:  # TODO: reset all world components inside world reset() (obstacles, etc.)
            num_obstacles = self.randomizer.integers(self.min_obstacles, self.max_obstacles)
            self.world.generate_obstacles(num_obstacles)

        # reset agents to initial state
        # ---------------------------------------------------------
        # [수정 1] 유인기 및 무인기 동기화 스폰 로직
        # ---------------------------------------------------------
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
                
                if observer_spawned:
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

        self.num_frames = 0
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
        return observation

    def state(self):
        """Return an observation of the global environment."""
        state = np.array([0, 0])
        return state

    def _build_search_grid_cache(self, cell_size):
        """Build a cache of renderable search-area grid cells."""
        minx, miny, maxx, maxy = self.search_area.bounds
        grid_rects = {}
        for gx in range(int(minx // cell_size), int(maxx // cell_size) + 1):
            for gy in range(int(miny // cell_size), int(maxy // cell_size) + 1):
                world_center = ((gx + 0.5) * cell_size, (gy + 0.5) * cell_size)
                if not self.search_area.covers(Point(world_center)):
                    continue
                world_top_left = (gx * cell_size, (gy + 1) * cell_size)
                game_top_left = world_ref_to_game_ref(world_top_left, self.area)
                grid_rects[(gx, gy)] = pygame.Rect(
                    int(game_top_left[0]),
                    int(game_top_left[1]),
                    cell_size,
                    cell_size,
                )
        return grid_rects

    def draw_exploration_overlay(self):
        """Draw explored vs unexplored search cells."""
        overlay = pygame.Surface(self.area.size, pygame.SRCALPHA)
        unexplored_color = (38, 57, 84, 60)
        explored_color = (76, 196, 120, 110)
        outline_color = (220, 235, 255, 25)

        for grid_key, rect in self.search_grid_rects.items():
            cell_color = explored_color if grid_key in self.explored_grids else unexplored_color
            pygame.draw.rect(overlay, cell_color, rect)
            pygame.draw.rect(overlay, outline_color, rect, width=1)

        self.screen.blit(overlay, (0, 0))

        legend_font = pygame.font.SysFont("Trebuchet MS", 16)
        legend_bg = pygame.Surface((170, 58), pygame.SRCALPHA)
        legend_bg.fill((8, 12, 16, 150))
        self.screen.blit(legend_bg, (12, 40))

        pygame.draw.rect(self.screen, explored_color, pygame.Rect(22, 50, 18, 18))
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
    def count_nearby_drones(self, agent, radius=100):
        """특정 에이전트 주변의 드론 수를 셉니다."""
        count = 0
        for name, other_agent in zip(self.agents, self.agents_list):
            if "drone" in name and other_agent != agent:
                # 유클리드 거리 계산
                d = math.sqrt((agent.x - other_agent.x)**2 + (agent.y - other_agent.y)**2)
                if d < radius:
                    count += 1
        return count

    def drone_spacing_penalty(self, agent, radius=90):
        """Return a penalty when drones bunch up too tightly."""
        penalty = 0.0
        for name, other_agent in zip(self.agents, self.agents_list):
            if "drone" not in name or other_agent == agent:
                continue
            separation = dist(agent.x, agent.y, other_agent.x, other_agent.y)
            if separation < radius:
                penalty += 0.03
        return penalty

    def get_drone_exploration_cells(self, agent):
        """Return exploration-grid cells covered by a drone's circular sensing range."""
        cell_size = self.exploration_cell_size
        min_gx = int((agent.x - agent.sensing_range) // cell_size)
        max_gx = int((agent.x + agent.sensing_range) // cell_size)
        min_gy = int((agent.y - agent.sensing_range) // cell_size)
        max_gy = int((agent.y + agent.sensing_range) // cell_size)
        sensing_range_sq = agent.sensing_range ** 2

        covered_cells = set()
        for gx in range(min_gx, max_gx + 1):
            for gy in range(min_gy, max_gy + 1):
                grid_key = (gx, gy)
                if grid_key not in self.search_grid_centers:
                    continue
                center_x, center_y = self.search_grid_centers[grid_key]
                if (center_x - agent.x) ** 2 + (center_y - agent.y) ** 2 <= sensing_range_sq:
                    covered_cells.add(grid_key)
        return covered_cells

    def get_primary_observer(self):
        """Return the first observer in the environment."""
        for name, agent in zip(self.agents, self.agents_list):
            if "observer" in name:
                return agent
        return None

    def finalize_episode(self):
        """Propagate the current end-of-episode state to every agent."""
        for ag in self.agents:
            self.rewards[ag] += self.global_reward
            self.terminations[ag] = self.terminate
            self.truncations[ag] = self.truncate
            self.infos[ag] = {
                "success": self.found_goal,
                "fatal_crash": self.collided,
                "min_drone_dist": self.min_drone_dist,
                "min_obs_dist": self.min_obs_dist,
                "explored_area": len(self.explored_grids) * 400,
            }

    def step(self, action, active_agent):
        """Execute a step."""
        if active_agent == self.agents[0]:
            self.global_reward = 0
            
        reward = 0
        agent = self.agents_list[self.agent_name_mapping[active_agent]]
        observer = self.get_primary_observer()
        
        # 1. 에이전트 업데이트 및 맵 이탈 체크
        agent.update(self.area, self.world, action)
        
        # [안전 로직] 즉시 맵 이탈 확인
        if agent.out_of_bound:
            self.collided = True
            reward -= 20
            self.terminate = True
            self.rewards[active_agent] = reward
            self.finalize_episode()
            return

        # 목표 이동 (오브젝트 이동)
        for goal in self.goals:
            goal.move(self.world.obstacles, self.search_area)

        # ---------------------------------------------------------
        # [협업 로직] 역할별 보상 부여
        # ---------------------------------------------------------
        
        # Drone
        if "drone" in active_agent and not self.collided:
            covered_cells = self.get_drone_exploration_cells(agent)
            new_cells = covered_cells - self.explored_grids
            if new_cells:
                reward += min(0.004 * len(new_cells), 0.15)
                self.explored_grids.update(covered_cells)

            if observer is not None:
                observer_dist = dist(observer.x, observer.y, agent.x, agent.y)
                if self.world.goal_known:
                    if 80 <= observer_dist <= 180:
                        reward += 0.02
                    else:
                        reward -= min(abs(observer_dist - 130) * 0.0004, 0.03)
                else:
                    if 140 <= observer_dist <= 280:
                        reward += 0.01
                    elif observer_dist < 90:
                        reward -= 0.05
                    elif observer_dist > 340:
                        reward -= 0.02
                agent.last_observer_distance = observer_dist

            reward -= self.drone_spacing_penalty(agent)

            min_current_dist = min(
                [dist(goal.x, goal.y, agent.x, agent.y)
                for goal in self.goals]
            )
            self.min_drone_dist = min(
                self.min_drone_dist,
                min_current_dist
            )
            for goal in self.goals:
                d = dist(goal.x, goal.y, agent.x, agent.y)
                if d < agent.sensing_range:
                    goal.detected = True
                    self.world.goal_known = True
                    self.world.observer_communication = [goal.x, goal.y]
                    if not goal.detected_by_drone:
                        goal.detected_by_drone = True
                        reward += 5.0
                        self.global_reward += 15.0
                    else:
                        reward += 0.02

        # =========================================================
        # Observer : 최종 도착 담당
        # =========================================================
        elif "observer" in active_agent and not self.collided:
            actual_min_dist = min(
                [dist(goal.x, goal.y, agent.x, agent.y)
                for goal in self.goals]
            )
            self.min_obs_dist = min(
                self.min_obs_dist,
                actual_min_dist
            )

            reward -= 0.005

            if self.world.goal_known:
                reward += 0.01 * self.count_nearby_drones(agent, radius=220)
                goal_x, goal_y = self.world.observer_communication
                current_dist = dist(goal_x, goal_y, agent.x, agent.y)

                if agent.last_goal_distance is None:
                    agent.last_goal_distance = current_dist
                else:
                    delta = agent.last_goal_distance - current_dist
                    if delta > 0:
                        reward += delta * 0.08
                    else:
                        reward -= abs(delta) * 0.04
                    agent.last_goal_distance = current_dist

                if current_dist < 50:
                    self.found_goal = True
                    self.global_reward += 100
                    self.terminate = True
            else:
                agent.last_goal_distance = None
                observer_grid = (
                    int(agent.x // self.observer_exploration_cell_size),
                    int(agent.y // self.observer_exploration_cell_size),
                )
                if observer_grid not in self.observer_explored_grids:
                    self.observer_explored_grids.add(observer_grid)
                    reward += 0.05

                base_x, base_y = game_ref_to_world_ref(self.world.base.center, self.area)
                base_dist = dist(base_x, base_y, agent.x, agent.y)
                if agent.last_base_distance is None:
                    agent.last_base_distance = base_dist
                else:
                    delta_base = base_dist - agent.last_base_distance
                    if delta_base > 0 and base_dist < 320:
                        reward += delta_base * 0.01
                    elif delta_base < 0 and base_dist < 220:
                        reward -= abs(delta_base) * 0.01
                    agent.last_base_distance = base_dist

                if base_dist < 120:
                    reward -= 0.03

        # 최종 보상 합산
        self.rewards[active_agent] = reward

        # 에피소드 종료/Truncation 처리
        if agent == self.agents_list[-1]:
            if self.collided:
                self.terminate = True

            self.world.update(self.area)
            if not self.terminate:
                self.num_frames += 1
                self.truncate = self.num_frames >= self.max_cycles

            self.finalize_episode()

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
