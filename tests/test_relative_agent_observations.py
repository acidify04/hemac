"""Tests for relative agent positions in observations."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hemac import HeMAC_v0
from hemac.environment.world import world_ref_to_game_ref


def test_observations_include_other_drone_and_observer_positions():
    """Observer and drone observations should include normalized peer positions."""
    env = HeMAC_v0.env(
        n_observers=1,
        n_drones=2,
        n_provisioners=0,
        min_obstacles=0,
        max_obstacles=0,
        render_mode=None,
    )

    try:
        env.reset(seed=7)

        observer_obs = env.observe("observer_0")["vector"]
        drone_obs = env.observe("drone_0")["vector"]

        norm = np.hypot(800.0, 800.0)

        observer_peer_slice = observer_obs[5:-2]
        expected_observer_peers = np.array(
            [
                0.0,
                50.0 / norm,
                -50.0 / norm,
                -50.0 / norm,
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(observer_peer_slice, expected_observer_peers, atol=1e-6)

        drone_peer_slice = drone_obs[4:-2]
        expected_drone_peers = np.array(
            [
                0.0,
                -50.0 / norm,
                -50.0 / norm,
                -100.0 / norm,
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(drone_peer_slice, expected_drone_peers, atol=1e-6)
    finally:
        env.close()


def test_reset_initializes_every_observer_sensor_pose():
    """Every configured observer must be observable after each reset."""
    env = HeMAC_v0.env(
        n_observers=2,
        n_drones=1,
        n_provisioners=0,
        min_obstacles=0,
        max_obstacles=0,
        render_mode=None,
    )
    try:
        for seed in (7, 11, 19):
            env.reset(seed=seed)
            raw_env = env.unwrapped.env
            observers = [
                agent
                for agent in raw_env.agents_list
                if agent.__class__.__name__ == "Observer"
            ]
            assert len(observers) == 2
            assert observers[0].rect.center != observers[1].rect.center
            for index, observer in enumerate(observers):
                assert observer.sensor.pos == observer.rect.center
                observation = env.observe(f"observer_{index}")
                assert env.observation_space(f"observer_{index}").contains(observation)
    finally:
        env.close()


def test_all_observers_must_reach_goal_before_success():
    """Every observer must reach its own assigned goal before success."""
    env = HeMAC_v0.env(
        n_observers=2,
        n_drones=0,
        n_provisioners=0,
        min_obstacles=0,
        max_obstacles=0,
        n_static_obstacles=0,
        poi_config=[{"speed": 0, "starting_pos": [500, 500]}],
        render_mode=None,
    )
    try:
        env.reset(seed=23)
        raw_env = env.unwrapped.env
        observers = raw_env.agents_list
        assert len(raw_env.goals) == 2
        assert len(set(map(id, raw_env.observer_goal_assignments.values()))) == 2

        wrong_goal = raw_env.observer_goal_assignments["observer_1"]
        first_observer = observers[0]
        own_goal = raw_env.observer_goal_assignments["observer_0"]
        assert np.hypot(
            wrong_goal.x - own_goal.x,
            wrong_goal.y - own_goal.y,
        ) > first_observer.sensing_range
        first_observer.x = float(wrong_goal.x)
        first_observer.y = float(wrong_goal.y)
        first_observer.rect.center = world_ref_to_game_ref(
            (first_observer.x, first_observer.y), raw_env.area
        )
        first_observer.sync_pose_state()
        raw_env.step(np.zeros(3, dtype=np.float32), "observer_0")
        assert "observer_0" not in raw_env.observers_reached_goal

        for index, observer in enumerate(observers):
            observer_name = f"observer_{index}"
            goal = raw_env.observer_goal_assignments[observer_name]
            observer.x = float(goal.x)
            observer.y = float(goal.y)
            observer.rect.center = world_ref_to_game_ref(
                (observer.x, observer.y), raw_env.area
            )
            observer.sync_pose_state()
            raw_env.step(np.zeros(3, dtype=np.float32), observer_name)
            assert raw_env.mission_success is (index == len(observers) - 1)

        info = raw_env.build_episode_info()
        assert info["observer_goal_count"] == 2
        assert info["observer_goal_required"] == 2
    finally:
        env.close()


def test_observers_see_private_goals_while_drone_sees_all_goals():
    """Observer goal channels are private, while drone inputs contain every goal."""
    env = HeMAC_v0.env(
        n_observers=2,
        n_drones=1,
        n_provisioners=0,
        min_obstacles=0,
        max_obstacles=0,
        n_static_obstacles=0,
        poi_config=[{"speed": 0, "spawn_mode": "random"}],
        render_mode=None,
    )
    try:
        env.reset(seed=2026)
        raw_env = env.unwrapped.env

        assert len(raw_env.goals) == 2
        assert set(raw_env.observer_goal_assignments) == {
            "observer_0",
            "observer_1",
        }
        assert len(set(map(id, raw_env.observer_goal_assignments.values()))) == 2

        observer_0 = env.observe("observer_0")
        observer_1 = env.observe("observer_1")
        drone = env.observe("drone_0")
        assert np.count_nonzero(observer_0["global_map"][:, :, 6]) == 1
        assert np.count_nonzero(observer_1["global_map"][:, :, 6]) == 1
        assert np.count_nonzero(drone["global_map"][:, :, 6]) == 2
        assert drone["central_vector"].shape == (8,)
    finally:
        env.close()


def test_reached_observer_stops_and_each_goal_gives_150_global_reward():
    """Each observer goal gives 150 team reward and then freezes its observer."""
    env = HeMAC_v0.env(
        n_observers=2,
        n_drones=1,
        n_provisioners=0,
        min_obstacles=0,
        max_obstacles=0,
        n_static_obstacles=0,
        poi_config=[{"speed": 0, "spawn_mode": "random"}],
        render_mode=None,
    )
    try:
        env.reset(seed=2026)
        raw_env = env.unwrapped.env
        observer_0, observer_1 = raw_env.agents_list[:2]

        first_goal = raw_env.observer_goal_assignments["observer_0"]
        observer_0.x = float(first_goal.x)
        observer_0.y = float(first_goal.y)
        observer_0.rect.center = world_ref_to_game_ref(
            (observer_0.x, observer_0.y), raw_env.area
        )
        observer_0.sync_pose_state()
        raw_env.rewards = {agent: 0.0 for agent in raw_env.agents}
        raw_env.step(np.zeros(3, dtype=np.float32), "observer_0")

        assert np.isclose(raw_env.rewards["observer_0"], 149.95)
        assert np.isclose(raw_env.rewards["observer_1"], 150.0)
        assert np.isclose(raw_env.rewards["drone_0"], 150.0)

        stopped_position = (observer_0.x, observer_0.y)
        raw_env.rewards = {agent: 0.0 for agent in raw_env.agents}
        raw_env.step(np.array([10.0, 10.0, 0.0], dtype=np.float32), "observer_0")
        assert (observer_0.x, observer_0.y) == stopped_position
        assert all(np.isclose(reward, 0.0) for reward in raw_env.rewards.values())

        obstacle_center = np.asarray([stopped_position], dtype=np.float32)
        raw_env.world.obstacle_warning_centers_world = obstacle_center.copy()
        reward_log = {agent: [] for agent in raw_env.agents}
        crashed = raw_env._apply_obstacle_motion_collisions(
            obstacle_center.copy(),
            reward_log,
        )
        assert "observer_0" not in crashed
        assert not raw_env.terminate
        raw_env.world.obstacle_warning_centers_world = np.empty(
            (0, 2), dtype=np.float32
        )

        second_goal = raw_env.observer_goal_assignments["observer_1"]
        observer_1.x = float(second_goal.x)
        observer_1.y = float(second_goal.y)
        observer_1.rect.center = world_ref_to_game_ref(
            (observer_1.x, observer_1.y), raw_env.area
        )
        observer_1.sync_pose_state()
        raw_env.rewards = {agent: 0.0 for agent in raw_env.agents}
        raw_env.step(np.zeros(3, dtype=np.float32), "observer_1")

        assert raw_env.mission_success
        assert np.isclose(raw_env.rewards["observer_0"], 0.0)
        assert np.isclose(raw_env.rewards["observer_1"], 149.95)
        assert np.isclose(raw_env.rewards["drone_0"], 150.0)
    finally:
        env.close()


def test_observer_maps_include_only_other_observers():
    """Observer maps expose peers while excluding the focal observer itself."""
    env = HeMAC_v0.env(
        n_observers=2,
        n_drones=0,
        n_provisioners=0,
        min_obstacles=0,
        max_obstacles=0,
        n_static_obstacles=0,
        render_mode=None,
    )
    try:
        env.reset(seed=29)
        raw_env = env.unwrapped.env
        focal, peer = raw_env.agents_list
        peer.x = focal.x + min(focal.sensing_range / 2.0, 20.0)
        peer.y = focal.y
        peer.rect.center = world_ref_to_game_ref((peer.x, peer.y), raw_env.area)
        peer.sync_pose_state()

        observation = env.observe("observer_0")
        assert observation["global_map"].shape == (40, 40, 7)
        assert observation["local_map"].shape == (20, 20, 7)
        assert float(observation["global_map"][:, :, 5].sum()) == 1.0
        assert float(observation["local_map"][:, :, 5].sum()) == 1.0
    finally:
        env.close()


def test_single_observer_peer_channels_are_empty():
    """A lone observer receives an all-zero other-observer channel."""
    env = HeMAC_v0.env(
        n_observers=1,
        n_drones=0,
        n_provisioners=0,
        min_obstacles=0,
        max_obstacles=0,
        n_static_obstacles=0,
        render_mode=None,
    )
    try:
        env.reset(seed=31)
        observation = env.observe("observer_0")
        assert observation["global_map"].shape == (40, 40, 7)
        assert observation["local_map"].shape == (20, 20, 7)
        assert not np.any(observation["global_map"][:, :, 5])
        assert not np.any(observation["local_map"][:, :, 5])
    finally:
        env.close()
