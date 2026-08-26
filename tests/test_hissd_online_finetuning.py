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


def test_squashed_log_prob_is_finite() -> None:
    raw_action = torch.tensor([[[0.0, 1.0, -1.0], [2.0, -2.0, 0.5]]])
    distribution = Normal(torch.zeros_like(raw_action), torch.ones_like(raw_action))

    log_prob = squashed_log_prob(distribution, raw_action)

    assert log_prob.shape == (1,)
    assert torch.isfinite(log_prob).all()


def test_gae_propagates_terminal_reward_backward() -> None:
    transitions = [
        {"reward": 0.0, "value": torch.tensor(0.0)},
        {"reward": 0.0, "value": torch.tensor(0.0)},
        {"reward": 1.0, "value": torch.tensor(0.0)},
    ]

    add_gae(transitions, gamma=1.0, gae_lambda=1.0)

    assert [item["advantage"] for item in transitions] == [1.0, 1.0, 1.0]
    assert [item["return"] for item in transitions] == [1.0, 1.0, 1.0]


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
