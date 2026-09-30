"""Focused tests for conservative online HiSSD PPO fine-tuning."""

import random
from types import SimpleNamespace

import numpy as np
import torch
from torch.distributions import Normal

from src.skill_discovery.finetune_hissd_online import (
    add_gae,
    observer_residual_action,
    sample_stage_difficulty,
    squashed_log_prob,
)
from src.skill_discovery.finetune_hissd_drone_online import (
    OnlineNoSkillResidualPolicy,
    add_per_drone_gae,
    apply_shared_terminal_crash_penalty,
    combine_per_drone_rewards,
    environment_action_distribution,
    final_fusion_linear,
    online_action_mean,
    per_drone_gaussian_log_prob,
    per_drone_squashed_log_prob,
    validation_regressed,
)


def test_squashed_log_prob_is_finite() -> None:
    raw_action = torch.tensor([[[0.0, 1.0, -1.0], [2.0, -2.0, 0.5]]])
    distribution = Normal(torch.zeros_like(raw_action), torch.ones_like(raw_action))

    log_prob = squashed_log_prob(distribution, raw_action)

    assert log_prob.shape == (1,)
    assert torch.isfinite(log_prob).all()


def test_drone_log_prob_keeps_shared_policy_agents_separate() -> None:
    raw_action = torch.zeros(2, 3, 3)
    distribution = Normal(torch.zeros_like(raw_action), torch.ones_like(raw_action))

    log_prob = per_drone_squashed_log_prob(distribution, raw_action)

    assert log_prob.shape == (2, 3)
    assert torch.isfinite(log_prob).all()


def test_environment_gaussian_log_prob_keeps_agents_separate() -> None:
    action = torch.zeros(2, 3, 3)
    distribution = Normal(torch.zeros_like(action), torch.ones(3))

    log_prob = per_drone_gaussian_log_prob(distribution, action)

    assert log_prob.shape == (2, 3)
    assert torch.isfinite(log_prob).all()


def test_environment_distribution_scales_normalized_exploration() -> None:
    mean = torch.zeros(2, 3)
    log_std = torch.full((3,), -2.0)

    distribution = environment_action_distribution(mean, log_std, 25.0)

    torch.testing.assert_close(
        distribution.scale,
        torch.full_like(mean, float(np.exp(-2.0) * 25.0)),
    )


def test_final_fusion_linear_selects_only_last_projection() -> None:
    encoder = torch.nn.Module()
    encoder.fusion = torch.nn.Sequential(
        torch.nn.Linear(8, 6),
        torch.nn.ReLU(),
        torch.nn.Linear(6, 4),
        torch.nn.ReLU(),
    )

    selected = final_fusion_linear(encoder)

    assert selected is encoder.fusion[2]


def test_no_skill_residual_preserves_bc_prior_at_initialization() -> None:
    policy = OnlineNoSkillResidualPolicy(96, 3, 64, skill_dim=16)

    observation = torch.randn(2, 3, 96)
    residual = policy(observation, torch.randn(2, 3, 16), torch.randn(2, 3, 16))

    torch.testing.assert_close(residual, torch.zeros_like(residual))


def test_skill_and_no_skill_start_from_identical_bc_action() -> None:
    class Decoder:
        base_action_head = staticmethod(lambda observation: observation[..., :3])

    class Model:
        action_decoder = Decoder()

        @staticmethod
        def decode_action_logits(observation, common_skill, task_skill):
            return observation[..., :3] + common_skill[..., :3] + task_skill[..., :3]

    policy = OnlineNoSkillResidualPolicy(8, 3, 16, skill_dim=4)
    observation = torch.randn(2, 3, 8)
    common_skill = torch.randn(2, 3, 4)
    task_skill = torch.randn(2, 3, 4)

    full, skill_delta = online_action_mean(
        Model(), policy, observation, common_skill, task_skill, "full", 25.0
    )
    no_skill, _ = online_action_mean(
        Model(), policy, observation, common_skill, task_skill, "no_skill", 25.0
    )

    torch.testing.assert_close(full, no_skill)
    torch.testing.assert_close(skill_delta, torch.zeros_like(skill_delta))


def test_nonzero_skill_prior_uses_pretrained_decoder() -> None:
    class Decoder:
        base_action_head = staticmethod(lambda observation: observation[..., :3])

    class Model:
        action_decoder = Decoder()

        @staticmethod
        def decode_action_logits(observation, common_skill, task_skill):
            return observation[..., :3] + common_skill[..., :3] + task_skill[..., :3]

    policy = OnlineNoSkillResidualPolicy(
        8, 3, 16, skill_dim=4, skill_prior_init=0.25
    )
    observation = torch.zeros(2, 3, 8)
    common_skill = torch.full((2, 3, 4), 0.2)
    task_skill = torch.full((2, 3, 4), 0.1)

    full, skill_delta = online_action_mean(
        Model(), policy, observation, common_skill, task_skill, "full", 25.0
    )
    pretrained = torch.tanh(common_skill[..., :3] + task_skill[..., :3]) * 25.0

    torch.testing.assert_close(full, 0.25 * pretrained)
    torch.testing.assert_close(skill_delta, full.abs().mean())


def test_gae_propagates_terminal_reward_backward() -> None:
    transitions = [
        {"reward": 0.0, "value": torch.tensor(0.0)},
        {"reward": 0.0, "value": torch.tensor(0.0)},
        {"reward": 1.0, "value": torch.tensor(0.0)},
    ]

    add_gae(transitions, gamma=1.0, gae_lambda=1.0)

    assert [item["advantage"] for item in transitions] == [1.0, 1.0, 1.0]
    assert [item["return"] for item in transitions] == [1.0, 1.0, 1.0]


def test_per_drone_gae_preserves_individual_credit() -> None:
    transitions = [
        {
            "reward": torch.tensor([1.0, -1.0, 0.0]),
            "value": torch.zeros(3),
        },
        {
            "reward": torch.tensor([0.0, 0.0, 3.0]),
            "value": torch.zeros(3),
        },
    ]

    add_per_drone_gae(transitions, gamma=1.0, gae_lambda=1.0)

    torch.testing.assert_close(
        transitions[0]["advantage"], torch.tensor([1.0, -1.0, 3.0])
    )
    torch.testing.assert_close(
        transitions[1]["advantage"], torch.tensor([0.0, 0.0, 3.0])
    )


def test_terminal_crash_is_shared_without_doubling_existing_penalty() -> None:
    rewards = np.array([-300.05, -0.05, 0.25], dtype=np.float32)

    shared = apply_shared_terminal_crash_penalty(
        rewards, fatal_crash=True, penalty=300.0
    )

    np.testing.assert_allclose(shared, [-300.05, -300.0, -300.0])


def test_agent_reward_uses_distributed_success_credit() -> None:
    local = np.array([1.0, -0.05, 0.25], dtype=np.float32)
    distributed_success = np.full(3, 75.0, dtype=np.float32)

    rewards = combine_per_drone_rewards(local, distributed_success)

    np.testing.assert_allclose(rewards, [76.0, 74.95, 75.25])


def test_validation_rollback_only_triggers_for_large_regression() -> None:
    assert validation_regressed(0.40, 0.64, 0.15)
    assert not validation_regressed(0.54, 0.64, 0.15)
    assert not validation_regressed(0.0, 1.0, 0.0)


def test_stage_six_mixture_includes_replay_difficulties() -> None:
    rng = random.Random(17)
    sampled = [sample_stage_difficulty(6, rng) for _ in range(1000)]

    assert set(sampled) == {4, 5, 6}
    assert sampled.count(6) > sampled.count(5) > sampled.count(4)


def test_observer_residual_action_preserves_baseline_at_zero() -> None:
    action_space = SimpleNamespace(
        low=np.full(3, -10.0, dtype=np.float32),
        high=np.full(3, 10.0, dtype=np.float32),
    )
    baseline = np.array([3.0, -4.0, 1.0], dtype=np.float32)

    action = observer_residual_action(
        baseline, torch.zeros(3), action_space, residual_scale=0.35
    )

    np.testing.assert_array_equal(action, baseline)
