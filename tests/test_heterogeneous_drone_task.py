"""Regression tests for the A/B heterogeneous drone-only MAPPO task."""

from pathlib import Path

import numpy as np
import pygame

from hemac import HeMAC_v0
from hemac.environment.sensors import ForwardFacingCamera, RoundCamera
from src.train.train import build_env_config, drone_policy_id_for_agent


def _make_env(*, max_cycles=3):
    config = build_env_config(drone_only=True)
    config.update(
        {
            "max_cycles": max_cycles,
            "min_obstacles": 0,
            "max_obstacles": 0,
            "n_static_obstacles": 0,
        }
    )
    env = HeMAC_v0.env(**config)
    env.reset(seed=17)
    return env, config


def test_drone_types_use_role_specific_motion_sensing_and_icons():
    env, config = _make_env()
    try:
        raw_env = env.unwrapped.env
        drones = raw_env.agents_list
        type_configs = config["drone_config"]["drone_type_configs"]

        assert [drone.drone_type for drone in drones] == ["A", "B", "B"]
        assert [drone.max_speed for drone in drones] == [10.0, 25.0, 25.0]
        assert [drone.sensing_range for drone in drones] == [
            type_configs["A"]["sensing_range"],
            type_configs["B"]["sensing_range"],
            type_configs["B"]["sensing_range"],
        ]
        assert isinstance(drones[0].sensor, RoundCamera)
        assert all(
            isinstance(drone.sensor, ForwardFacingCamera)
            for drone in drones[1:]
        )
        assert all(
            np.isclose(
                drone.sensor.hfov,
                type_configs["B"]["sensor"]["params"]["hfov"],
            )
            for drone in drones[1:]
        )
        assert (
            drones[0].sensing_range
            > raw_env.world.OBSTACLE_WARNING_RADIUS + drones[0].max_speed
        )
        assert all(
            drone.sensing_range
            > raw_env.world.OBSTACLE_WARNING_RADIUS + drone.max_speed
            for drone in drones[1:]
        )
        assert [Path(drone.icon_path).name for drone in drones] == [
            "drone_A.png",
            "drone_B.png",
            "drone_B.png",
        ]

        np.testing.assert_allclose(env.action_space("drone_0").high, 10.0)
        np.testing.assert_allclose(env.action_space("drone_1").high, 25.0)
        np.testing.assert_allclose(env.action_space("drone_2").high, 25.0)

        for agent_id in env.possible_agents:
            observation = env.observe(agent_id)
            assert observation["global_map"].shape == (40, 40, 7)
            assert observation["local_map"].shape == (20, 20, 7)
            assert env.observation_space(agent_id).contains(observation)
    finally:
        env.close()


def test_type_b_uses_the_configured_forward_field_of_view():
    env, _ = _make_env()
    try:
        drone_b = env.unwrapped.env.agents_list[1]
        sensor = drone_b.sensor
        sensor.update_poly_points((0.0, 0.0), 0.0)

        assert sensor.is_point_detected((100.0, 0.0))
        outside_angle = (sensor.hfov / 2.0) + 0.1
        outside_point = (
            100.0 * np.cos(outside_angle),
            -100.0 * np.sin(outside_angle),
        )
        assert not sensor.is_point_detected(outside_point)
    finally:
        env.close()


def test_type_b_sensor_heading_and_local_map_follow_motion_direction():
    env, _ = _make_env()
    try:
        raw_env = env.unwrapped.env
        drone_b = raw_env.agents_list[1]
        drone_b.update(
            raw_env.area,
            raw_env.world,
            np.asarray([0.0, drone_b.max_speed, 0.0], dtype=np.float32),
            False,
        )

        assert np.isclose(drone_b.orientation, np.pi / 2.0, atol=0.15)
        assert np.isclose(drone_b.sensor.theta, drone_b.orientation)

        observation = env.observe("drone_1")
        rel_x, rel_y, circular_mask = drone_b._local_sampling_template(
            drone_b.LOCAL_MAP_SIZE,
            drone_b.sensing_range,
        )
        angles = np.arctan2(rel_y, rel_x)
        relative_angles = (
            angles - drone_b.orientation + np.pi
        ) % (2 * np.pi) - np.pi
        visible = circular_mask & (
            np.abs(relative_angles) <= drone_b.sensor.hfov / 2.0
        )
        assert np.all(observation["local_map"][~visible] == 0.0)
    finally:
        env.close()


def test_type_a_and_type_b_use_separate_shared_policies():
    env, config = _make_env()
    try:
        assert drone_policy_id_for_agent("drone_0", config) == "drone_a_policy"
        assert drone_policy_id_for_agent("drone_1", config) == "drone_b_policy"
        assert drone_policy_id_for_agent("drone_2", config) == "drone_b_policy"
    finally:
        env.close()


def test_reward_config_matches_shared_shaping_without_terminal_bonus():
    env, config = _make_env()
    try:
        assert config["drone_step_penalty"] == 0.02
        assert config["drone_proximity_penalty_scale"] == 0.2
        assert config["drone_goal_found_reward"] == 100.0
        assert config["drone_exploration_reward_scale"] == 300.0
        assert config["drone_only_success_reward"] == 0.0
    finally:
        env.close()


def test_initial_sensing_is_registered_without_reward():
    env, _ = _make_env()
    try:
        raw_env = env.unwrapped.env

        assert raw_env.current_coverage_ratio() > 0.0
        assert all(reward == 0.0 for reward in raw_env.rewards.values())
    finally:
        env.close()


def test_goal_discovery_reward_is_shared_by_all_drones():
    env, _ = _make_env()
    try:
        raw_env = env.unwrapped.env
        active_drone = raw_env.agents_list[0]
        first_goal = raw_env.goals[0]
        raw_env.drone_proximity_penalty_scale = 0.0
        first_goal.x = active_drone.x
        first_goal.y = active_drone.y

        raw_env.step(np.zeros(3, dtype=np.float32), "drone_0")

        assert raw_env.found_goal
        assert first_goal.detected
        assert first_goal in raw_env.completed_goals
        for reward in raw_env.rewards.values():
            assert reward >= raw_env.drone_goal_found_reward
    finally:
        env.close()


def test_discovered_goal_route_detours_inside_explored_area_with_clearance():
    env, _ = _make_env()
    try:
        raw_env = env.unwrapped.env
        goal = raw_env.goals[0]
        goal.detected = True
        goal.rect.center = (800, 150)
        raw_env.completed_goals = [goal]
        raw_env.goals = []
        raw_env.world.base.center = (150, 150)
        raw_env.world.detected_search_mask.fill(True)
        # A dark top-down barrier forces the route to travel through the
        # explored opening below it instead of drawing a clipped straight line.
        raw_env.world.detected_search_mask[600:, 450:550] = False
        raw_env.world.observed_warning_pixel_mask[50:100, 50:100] = 1.0
        exploration_before_render = raw_env.world.detected_search_mask.copy()

        surface = pygame.Surface(raw_env.area.size, pygame.SRCALPHA)
        raw_env.draw_discovered_goal_routes(surface)
        np.testing.assert_array_equal(
            raw_env.world.detected_search_mask,
            exploration_before_render,
        )
        pixels = pygame.surfarray.array3d(surface)
        alpha = pygame.surfarray.array_alpha(surface)
        route_pixels = (
            np.all(pixels == np.asarray([255, 35, 35]), axis=2)
            & (alpha > 0)
        )

        assert np.any(route_pixels)
        assert not np.any(route_pixels[435:565, :415])
        crossing_y = np.flatnonzero(np.any(route_pixels[450:550], axis=0))
        assert crossing_y.size > 0
        assert int(crossing_y.min()) >= 420
    finally:
        env.close()


def test_three_goals_spawn_sequentially_across_diagonal_regions():
    env, _ = _make_env(max_cycles=20)
    try:
        raw_env = env.unwrapped.env
        drone = raw_env.agents_list[0]
        assert raw_env.sequential_goal_regions
        assert raw_env.number_of_POIs == 1

        for expected_region in range(3):
            assert len(raw_env.goals) == 1
            active_goal = raw_env.goals[0]
            assert active_goal.sequential_region_index == expected_region
            assert raw_env._position_in_diagonal_goal_region(
                (active_goal.x, active_goal.y),
                expected_region,
            )

            active_goal.x = drone.x
            active_goal.y = drone.y
            active_goal.rect.center = drone.rect.center
            raw_env.step(np.zeros(3, dtype=np.float32), "drone_0")

            assert active_goal in raw_env.completed_goals
            assert raw_env.sequential_goals_found == expected_region + 1
            assert env.observation_space("drone_0").contains(
                env.observe("drone_0")
            )

        assert raw_env.goals == []
        assert raw_env._required_goals_found()
        info = raw_env.build_episode_info()
        assert info["goal_found_count"] == 3
        assert info["goal_required_count"] == 3
        assert info["all_goals_found"]
    finally:
        env.close()


def test_new_coverage_reward_is_shared_and_ratio_normalized():
    env, _ = _make_env()
    try:
        raw_env = env.unwrapped.env
        active_drone = raw_env.agents_list[0]
        active_drone.x += 250.0
        raw_env.found_goal = True
        coverage_before = raw_env.current_drone_reward_coverage_ratio()

        raw_env.step(np.zeros(3, dtype=np.float32), "drone_0")

        coverage_after = raw_env.current_drone_reward_coverage_ratio()
        expected_reward = (
            raw_env.drone_exploration_reward_scale
            * (coverage_after - coverage_before)
        )
        assert coverage_after > coverage_before
        for reward in raw_env.rewards.values():
            np.testing.assert_allclose(reward, expected_reward)
    finally:
        env.close()


def test_goal_discovery_does_not_end_maximize_coverage_episode_early():
    env, _ = _make_env(max_cycles=5)
    try:
        raw_env = env.unwrapped.env
        raw_env.found_goal = True
        raw_env.sequential_goals_found = raw_env.sequential_goal_count
        raw_env.world.coverage_map[...] = raw_env.world.search_mask

        assert not raw_env._check_drone_only_mission_success()
        assert not raw_env.terminate
        assert not raw_env.truncate
        assert not raw_env.mission_success

        raw_env.num_frames = raw_env.max_cycles
        assert raw_env._finalize_maximize_coverage_task()
        raw_env.finalize_episode()

        assert raw_env.mission_success
        assert not raw_env.terminate
        assert raw_env.success_step == raw_env.max_cycles
        assert all(reward == 0.0 for reward in raw_env.rewards.values())
    finally:
        env.close()


def test_time_limit_is_failure_when_no_drone_found_goal():
    env, _ = _make_env(max_cycles=5)
    try:
        raw_env = env.unwrapped.env
        raw_env.num_frames = raw_env.max_cycles
        raw_env.truncate = True

        assert not raw_env._finalize_maximize_coverage_task()
        assert not raw_env.mission_success
        assert raw_env.build_episode_info()["timeout"]
    finally:
        env.close()
