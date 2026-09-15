"""Tests for fixed-duration homogeneous drone skills."""

import torch

from src.skill_discovery.drone_skill_vae import DroneSkillVAE


def build_model(*, skill_duration: int = 4) -> DroneSkillVAE:
    return DroneSkillVAE(
        global_map_channels=7,
        local_map_channels=7,
        skill_duration=skill_duration,
        decoder_observation_conditioned=True,
    )


def test_skill_sequence_is_held_until_the_next_episode_aligned_boundary() -> None:
    model = build_model(skill_duration=4)
    values = torch.arange(10, dtype=torch.float32).view(1, 10, 1, 1)

    held, decisions = model.hold_skill_sequence(values)

    assert held.flatten().tolist() == [0, 0, 0, 0, 4, 4, 4, 4, 8, 8]
    assert decisions.flatten().tolist() == [
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
        False,
    ]


def test_cropped_sequence_uses_offset_for_future_skill_boundaries() -> None:
    model = build_model(skill_duration=4)
    values = torch.arange(8, dtype=torch.float32).view(1, 8, 1, 1)

    held, decisions = model.hold_skill_sequence(values, torch.tensor([2]))

    assert held.flatten().tolist() == [0, 0, 2, 2, 2, 2, 6, 6]
    assert decisions.flatten().tolist() == [
        True,
        False,
        True,
        False,
        False,
        False,
        True,
        False,
    ]


def test_decoder_conditions_skill_residual_on_observation_features() -> None:
    model = build_model()
    first_layer = model.skill_action_head[0]

    assert first_layer.in_features == model.observation_dim + model.latent_dim
    features = torch.randn(2, 3, model.observation_dim)
    skills = torch.randn(2, 3, model.latent_dim)
    assert model.skill_residual(features, skills).shape == (2, 3, model.action_dim)


def test_online_inference_selects_only_after_skill_expiration() -> None:
    model = build_model(skill_duration=3)
    feature_dim = model.observation_dim
    model.encode_observations = lambda observations: observations["features"]
    state = model.initial_inference_state(2, device=torch.device("cpu"))
    outputs = []

    with torch.inference_mode():
        for step in range(4):
            observation = {
                "features": torch.full((1, 2, feature_dim), float(step + 1))
            }
            output, state = model.inference_step(observation, state)
            outputs.append(output)

    assert [output["skill_switched"] for output in outputs] == [
        True,
        False,
        False,
        True,
    ]
    assert torch.equal(outputs[0]["skills"], outputs[1]["skills"])
    assert torch.equal(outputs[1]["skills"], outputs[2]["skills"])
    assert outputs[0]["skill_steps_remaining"] == 2
    assert outputs[2]["skill_steps_remaining"] == 0
