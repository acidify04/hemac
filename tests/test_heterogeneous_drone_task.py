"""Regression tests for the A/B heterogeneous drone-only MAPPO task."""

from pathlib import Path

import numpy as np

from hemac import HeMAC_v0
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
    env, _ = _make_env()
    try:
        raw_env = env.unwrapped.env
        drones = raw_env.agents_list

        assert [drone.drone_type for drone in drones] == ["A", "B", "B"]
        assert [drone.max_speed for drone in drones] == [10.0, 25.0, 25.0]
        assert [drone.sensing_range for drone in drones] == [100.0, 40.0, 40.0]
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


def test_type_a_and_type_b_use_separate_shared_policies():
    env, config = _make_env()
    try:
        assert drone_policy_id_for_agent("drone_0", config) == "drone_a_policy"
        assert drone_policy_id_for_agent("drone_1", config) == "drone_b_policy"
        assert drone_policy_id_for_agent("drone_2", config) == "drone_b_policy"
    finally:
        env.close()


def test_reward_config_matches_individual_shaping_and_shared_success():
    env, config = _make_env()
    try:
        assert config["drone_step_penalty"] == 0.02
        assert config["drone_proximity_penalty_scale"] == 0.2
        assert config["drone_goal_found_reward"] == 100.0
        assert config["drone_exploration_reward_scale"] == 300.0
        assert config["drone_only_success_reward"] == 200.0
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


def test_goal_discovery_reward_is_individual():
    env, _ = _make_env()
    try:
        raw_env = env.unwrapped.env
        active_drone = raw_env.agents_list[0]
        raw_env.goals[0].x = active_drone.x
        raw_env.goals[0].y = active_drone.y

        raw_env.step(np.zeros(3, dtype=np.float32), "drone_0")

        assert raw_env.found_goal
        assert raw_env.rewards["drone_0"] > 99.0
        assert raw_env.rewards["drone_1"] == 0.0
        assert raw_env.rewards["drone_2"] == 0.0
    finally:
        env.close()


def test_new_coverage_reward_is_individual_and_ratio_normalized():
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
            - raw_env.drone_step_penalty
        )
        assert coverage_after > coverage_before
        np.testing.assert_allclose(raw_env.rewards["drone_0"], expected_reward)
        assert raw_env.rewards["drone_1"] == 0.0
        assert raw_env.rewards["drone_2"] == 0.0
    finally:
        env.close()


def test_goal_discovery_does_not_end_maximize_coverage_episode_early():
    env, _ = _make_env(max_cycles=5)
    try:
        raw_env = env.unwrapped.env
        raw_env.found_goal = True
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
        assert all(
            reward == raw_env.drone_only_success_reward
            for reward in raw_env.rewards.values()
        )
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
