import copy

import numpy as np
import torch

from hemac import HeMAC_v0
from hemac.rllib_policy import (
    MAPPOCentralizedCriticTorchModel,
    drone_policy_model_config,
)


def _make_env():
    env = HeMAC_v0.env(
        n_observers=1,
        n_drones=3,
        n_provisioners=0,
        min_obstacles=0,
        max_obstacles=0,
        n_static_obstacles=0,
        poi_config=[{"speed": 0, "spawn_mode": "random"}],
    )
    env.reset(seed=7)
    return env


def _batch_observation(observation):
    return {
        key: torch.as_tensor(value).unsqueeze(0)
        for key, value in observation.items()
    }


def test_drone_observation_contains_world_centered_critic_inputs():
    env = _make_env()
    try:
        observations = [env.observe(f"drone_{idx}") for idx in range(3)]
        shared_channel_indices = [0, 1, 2, 3, 4, 6, 7]

        for drone_idx, observation in enumerate(observations):
            assert env.observation_space(f"drone_{drone_idx}").contains(observation)
            assert observation["central_map"].shape == (20, 20, 8)
            assert observation["central_vector"].shape == (8,)
            assert np.all(np.abs(observation["central_vector"]) <= 1.0)
            assert np.count_nonzero(observation["central_map"][:, :, 5]) > 0

        for observation in observations[1:]:
            np.testing.assert_array_equal(
                observations[0]["central_map"][:, :, shared_channel_indices],
                observation["central_map"][:, :, shared_channel_indices],
            )

        raw_env = env.unwrapped.env
        focal_drone = next(
            agent
            for agent in raw_env.agents_list
            if agent.__class__.__name__ == "Drone" and agent.id == 0
        )
        other_drones = sorted(
            (
                agent
                for agent in raw_env.agents_list
                if agent.__class__.__name__ == "Drone" and agent is not focal_drone
            ),
            key=lambda agent: (
                (agent.x - focal_drone.x) ** 2 + (agent.y - focal_drone.y) ** 2
            ),
        )
        observer = next(
            agent
            for agent in raw_env.agents_list
            if agent.__class__.__name__ == "Observer"
        )
        goal = raw_env.goals[0]
        expected_relative_positions = np.asarray(
            [
                *(
                    coordinate
                    for drone in other_drones
                    for coordinate in (
                        (drone.x - focal_drone.x) / raw_env.area.width,
                        (drone.y - focal_drone.y) / raw_env.area.height,
                    )
                ),
                (observer.x - focal_drone.x) / raw_env.area.width,
                (observer.y - focal_drone.y) / raw_env.area.height,
                (goal.x - focal_drone.x) / raw_env.area.width,
                (goal.y - focal_drone.y) / raw_env.area.height,
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(
            observations[0]["central_vector"],
            expected_relative_positions,
        )
    finally:
        env.close()


def test_mappo_actor_ignores_central_input_while_critic_uses_it():
    env = _make_env()
    try:
        observation = env.observe("drone_0")
        model_config = drone_policy_model_config()
        torch.manual_seed(5)
        model = MAPPOCentralizedCriticTorchModel(
            env.observation_space("drone_0"),
            env.action_space("drone_0"),
            num_outputs=6,
            model_config={
                "custom_model_config": model_config["custom_model_config"],
            },
            name="mappo_separation_test",
        )
        assert model.central_critic_input_dim == 136

        logits_before, _ = model(
            {"obs": _batch_observation(observation)},
            [],
            None,
        )
        value_before = model.value_function().detach().clone()

        changed_observation = copy.deepcopy(observation)
        changed_observation["central_map"] = 1.0 - changed_observation["central_map"]
        changed_observation["central_vector"] = np.clip(
            -changed_observation["central_vector"] + 0.25,
            -1.0,
            1.0,
        )
        logits_after, _ = model(
            {"obs": _batch_observation(changed_observation)},
            [],
            None,
        )
        value_after = model.value_function().detach().clone()

        torch.testing.assert_close(logits_before, logits_after)
        assert not torch.allclose(value_before, value_after)
    finally:
        env.close()
