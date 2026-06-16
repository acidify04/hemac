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
        self.detected = set()
        self.found_goal = False

        self.old_dist_to_goal = 1000

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

        self.num_frames = 0
        self.old_dist_to_goal = 1000

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

    def draw(self):
        """Draw the environment."""
        pygame.event.pump()
        self.world.draw(self.screen)
        for agent in self.agents_list:
            agent.draw(self.screen)
        for goal in self.goals:
            goal.draw(self.screen)

    def build_episode_info(self):
        """Build a final-episode info dict for metrics and evaluation."""
        # coverage_ratio = self.current_coverage_ratio()
        # total_explored = len(self.explored_grids | self.observer_explored_grids) * (self.exploration_cell_size ** 2)

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
            # "goal_found": bool(self.goal_found),
            # "goal_known": bool(self.world.goal_known),
            # "fatal_crash": bool(self.collided),
            # "timeout": bool(self.truncate and not self.terminate),
            # "drone_crash": bool(self.drone_crash),
            # "observer_crash": bool(self.observer_crash),
            # "min_drone_dist": float(self.min_drone_dist),
            # "min_obs_dist": float(self.min_obs_dist),
            # "explored_area": float(total_explored),
            # "coverage_ratio": float(coverage_ratio),
            # "max_coverage_ratio": float(max(self.max_coverage_ratio, coverage_ratio)),
            # "goal_found_step": float(goal_found_step),
            "success_step": float(success_step),
            # "steps_after_goal_found": float(max(steps_after_goal_found, 0)),
            # "success_after_goal_found": bool(self.mission_success and self.goal_found),
        }

    def finalize_episode(self):
        """Propagate the current end-of-episode state to every agent."""
        episode_info = self.build_episode_info()
        for ag in self.agents:
            self.rewards[ag] += self.global_reward
            self.terminations[ag] = self.terminate
            self.truncations[ag] = self.truncate
            self.infos[ag] = dict(episode_info)

    def step(self, action, active_agent):
        """Execute a step."""
        if active_agent == self.agents[0]:
            self.global_reward = 0
        found_goal = False
        delivered_goal = False
        reward = 0

        agent = self.agents_list[self.agent_name_mapping[active_agent]]
        agent.update(self.area, self.world, action, self.found_goal)

        # Update position and uncertainty of objectives
        for goal in self.goals:
            goal.move(self.world.obstacles, self.search_area)

        # Specific actions for UAVs
        if "drone" in active_agent:
            # Collision check and map limits
            reward -= 0.1  # Small step penalty to encourage efficiency
            if agent.out_of_bound:
                self.collided = True
                reward -= 100  # Penalty for leaving the map
                if self.render_mode == "human":
                    LOGGER.info(f"drone went out of bounds! pos: {(agent.x, agent.y)}")
            elif not self.search_area.covers(Point((agent.x, agent.y))):
                self.collided = True
                reward -= 100  # going outside of search area
                if self.render_mode == "human" or self.render_mode == "rgb_array":
                    LOGGER.info(f"drone went out of search area. pos: {(agent.x, agent.y)}")
            else:
                for obstacle in self.world.obstacles:
                    if agent.process_collision(obstacle, 0):
                        self.collided = True
                        reward -= 20  # Penalty for collision with an obstacle
                        if self.render_mode == "human" or self.render_mode == "rgb_array":
                            LOGGER.info(
                                f"agent {active_agent} collided with obstacle at position [x,y] = {obstacle.center}"
                            )
            boundary_dist = self.search_area.boundary.distance(Point((agent.x, agent.y)))
            reward += 0.05 * boundary_dist  # Reward for being farther from the boundary, encourages staying in the center of the search area

            # POI tracking reward calculation
            for goal in self.goals[:]:
                goal_dist = dist(goal.x, goal.y, agent.x, agent.y)
                reward -= math.sqrt(goal_dist)  # Exponential reward based on distance to the goal, encourages getting closer
                if goal_dist < agent.sensing_range and not self.found_goal:
                    if agent.carried_targets < agent.carrying_capacity:
                        agent.found_goal = True
                        self.found_goal = True
                        reward += 100  # Reward for finding a goal
                        # goal.spawn_poi(self.search_area)
                        # goal.reset()
                        if self.rescuing_targets:
                            agent.carried_targets += 1

            if self.rescuing_targets and agent.carried_targets:
                closest_point_to_base = closest_point_in_rect(self.world.base, agent.rect.center)
                if (
                    dist(closest_point_to_base[0], closest_point_to_base[1], agent.rect.x, agent.rect.y)
                    < agent.sensing_range
                ):
                    delivered_goal = 1 * agent.carried_targets
                    agent.carried_targets = 0
                else:
                    for friend in self.agents:
                        if "provisioner" in friend:
                            provisioner = self.agents_list[self.agent_name_mapping[friend]]
                            if dist(provisioner.x, provisioner.y, agent.x, agent.y) < agent.sensing_range:
                                delivered_goal = 1 * agent.carried_targets
                                print(f"agent dropped {agent.carried_targets} targets!")
                                agent.carried_targets = 0
                                break
            new_detected = self.detected.union(agent.detected)
            if len(self.detected) < len(new_detected):
                reward += 0.1 * (len(new_detected) - len(self.detected))  # Reward for newly detected goals
                self.detected = new_detected
                # print(f'drone detected: {self.detected}')

            # global reward
            if self.rescuing_targets:
                self.global_reward += 10 * self.found_goal + 25 * delivered_goal
            else:
                self.global_reward += 10 * self.found_goal

        elif "observer" in active_agent:
            reward -= 0.1  # Small step penalty to encourage efficiency
            # if agent.out_of_bound:
            #     self.collided = True
            #     reward -= 100  # Penalty for leaving the map
            #     if self.render_mode == "human":
            #         LOGGER.info(f"observer went out of bounds! pos: {(agent.x, agent.y)}")
            # elif agent.goal_in_view:
            #     reward = 0

            # if agent.out_of_bound:
            #     self.collided = True
            #     reward -= 100  # Penalty for leaving the map
            #     if self.render_mode == "human":
            #         LOGGER.info(f"observer went out of bounds! pos: {(agent.x, agent.y)}")
            # elif agent.goal_in_view:
            #     reward = 0

            # goal_x, goal_y = self.world.observer_communication
            # current_dist = dist(goal_x, goal_y, agent.x, agent.y)
            # # if not self.found_goal:
            # #     reward -= math.sqrt(math.sqrt(math.sqrt(current_dist))) 
            # # else:
            # #     reward -= math.sqrt(math.sqrt(current_dist))
            # goal_heading = np.arctan2(goal_y - agent.y, goal_x - agent.x)
            # heading_error = ((goal_heading - agent.orientation + np.pi) % (2 * np.pi)) - np.pi
            # success_radius = 80 if self.known_goals else 50

            # if self.known_goals:
            #     reward -= 0.01
            #     reward += 0.12 * np.cos(heading_error)
            # else:
            #     reward += 0.03 * np.cos(heading_error)

            # if current_dist < success_radius:
            #     self.mission_success = True
            #     reward += 200 # Task completion reward
            #     self.success_step = self.num_frames
            #     self.global_reward += 180 if self.known_goals else 120
            #     self.terminate = True
            
            # new_detected = self.detected.union(agent.detected)
            # if len(self.detected) < len(new_detected):
            #     reward += 5 * (len(new_detected) - len(self.detected))  # Reward for newly detected goals
            #     self.detected = new_detected
            #     # print(f'observer detected: {self.detected}')

            # boundary_dist = self.search_area.boundary.distance(Point((agent.x, agent.y)))
            # reward += 0.5 * boundary_dist  # Reward for being farther from the boundary
            if agent.out_of_bound:
                self.collided = True
                reward -= 100  # Penalty for leaving the map
                if self.render_mode == "human":
                    LOGGER.info(f"drone went out of bounds! pos: {(agent.x, agent.y)}")
            elif not self.search_area.covers(Point((agent.x, agent.y))):
                self.collided = True
                reward -= 100  # going outside of search area
                if self.render_mode == "human" or self.render_mode == "rgb_array":
                    LOGGER.info(f"drone went out of search area. pos: {(agent.x, agent.y)}")
            else:
                for obstacle in self.world.obstacles:
                    if agent.process_collision(obstacle, 0):
                        self.collided = True
                        reward -= 20  # Penalty for collision with an obstacle
                        if self.render_mode == "human" or self.render_mode == "rgb_array":
                            LOGGER.info(
                                f"agent {active_agent} collided with obstacle at position [x,y] = {obstacle.center}"
                            )
            boundary_dist = self.search_area.boundary.distance(Point((agent.x, agent.y)))
            reward += 0.05 * boundary_dist  # Reward for being farther from the boundary, encourages staying in the center of the search area

            # POI tracking reward calculation
            for goal in self.goals[:]:
                goal_dist = dist(goal.x, goal.y, agent.x, agent.y)
                reward -= math.sqrt(goal_dist) # Exponential reward based on distance to the goal, encourages getting closer
                if goal_dist < 50 and not self.found_goal: # goal까지의 거리가 sensing range보다 가까워지면 발견
                    agent.found_goal = True
                    self.found_goal = True
                    reward += 100  # Reward for finding a goal
                        # goal.spawn_poi(self.search_area)
                        # goal.reset()

            new_detected = self.detected.union(agent.detected)
            if len(self.detected) < len(new_detected):
                reward += 0.1 * (len(new_detected) - len(self.detected))  # Reward for newly detected goals
                self.detected = new_detected

        # individual reward
        self.rewards[active_agent] = reward
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

            # Distribution of awards and information to agents
            for i, ag in enumerate(self.agents):
                self.rewards[ag] += self.global_reward
                self.terminations[ag] = self.terminate
                self.truncations[ag] = self.truncate
                self.infos[ag] = {"success": self.found_goal}

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