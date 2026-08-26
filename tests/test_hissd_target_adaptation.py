"""Unit tests for conservative target-task HiSSD adaptation."""

from types import SimpleNamespace

import torch

from src.skill_discovery.adapt_hissd_target import (
    supervised_contrastive_loss,
    target_sample_weights,
    weighted_action_mse,
)


def test_weighted_action_mse_respects_transition_weights() -> None:
    prediction = torch.zeros(1, 2, 2, 3)
    target = torch.ones_like(prediction)
    target[:, 1] = 100.0
    valid = torch.ones(1, 2, 2, dtype=torch.bool)
    weights = torch.tensor([[[1.0, 1.0], [0.0, 0.0]]])

    assert torch.isclose(
        weighted_action_mse(prediction, target, valid, weights),
        torch.tensor(1.0),
    )


def test_target_weights_prefer_success_and_remove_crash_actions() -> None:
    batch = {
        "actions": torch.zeros(3, 1, 2, 3),
        "outcome_category": torch.tensor([0, 1, 2]),
        "individual_rewards": torch.tensor(
            [[[0.0, -300.0]], [[0.0, 0.0]], [[0.0, 0.0]]]
        ),
    }
    args = SimpleNamespace(
        success_weight=1.0,
        goal_found_failure_weight=0.2,
        goal_not_found_weight=0.02,
        crash_reward_threshold=-100.0,
        crash_lookback_steps=0,
    )

    weights = target_sample_weights(batch, args)

    assert torch.allclose(
        weights,
        torch.tensor([[[1.0, 0.0]], [[0.2, 0.2]], [[0.02, 0.02]]]),
    )


def test_target_weights_remove_actions_leading_into_collision() -> None:
    batch = {
        "actions": torch.zeros(1, 5, 1, 3),
        "outcome_category": torch.tensor([0]),
        "individual_rewards": torch.tensor(
            [[[0.0], [0.0], [0.0], [-300.0], [0.0]]]
        ),
    }
    args = SimpleNamespace(
        success_weight=1.0,
        goal_found_failure_weight=0.0,
        goal_not_found_weight=0.0,
        crash_reward_threshold=-100.0,
        crash_lookback_steps=2,
    )

    weights = target_sample_weights(batch, args)

    assert torch.equal(
        weights,
        torch.tensor([[[1.0], [0.0], [0.0], [0.0], [1.0]]]),
    )


def test_target_weights_focus_on_visible_warning_zones() -> None:
    local_map = torch.zeros(1, 2, 1, 7, 20, 20)
    local_map[:, 1, :, 3, 10, 10] = 0.75
    batch = {
        "actions": torch.zeros(1, 2, 1, 3),
        "observations": {"local_map": local_map},
        "outcome_category": torch.tensor([0]),
        "individual_rewards": torch.zeros(1, 2, 1),
    }
    args = SimpleNamespace(
        success_weight=1.0,
        goal_found_failure_weight=0.0,
        goal_not_found_weight=0.0,
        crash_reward_threshold=-100.0,
        crash_lookback_steps=8,
        warning_focus_weight=1.0,
    )

    weights = target_sample_weights(batch, args)

    assert torch.allclose(weights, torch.tensor([[[1.0], [1.75]]]))


def test_observer_collision_removes_all_drone_leadup_actions() -> None:
    batch = {
        "actions": torch.zeros(1, 5, 3, 3),
        "outcome_category": torch.tensor([1]),
        "individual_rewards": torch.zeros(1, 5, 3),
        "observer_rewards": torch.tensor(
            [[[0.0], [0.0], [0.0], [-300.0], [0.0]]]
        ),
    }
    args = SimpleNamespace(
        success_weight=1.0,
        goal_found_failure_weight=0.1,
        goal_not_found_weight=0.01,
        crash_reward_threshold=-100.0,
        crash_lookback_steps=2,
        warning_focus_weight=0.0,
        observer_warning_focus_weight=0.0,
    )

    weights = target_sample_weights(batch, args)

    assert torch.equal(weights[:, 1:4], torch.zeros(1, 3, 3))
    assert torch.allclose(weights[:, (0, 4)], torch.full((1, 2, 3), 0.1))


def test_success_weights_focus_on_observer_warning_overlap() -> None:
    global_map = torch.zeros(1, 1, 1, 7, 40, 40)
    global_map[..., 3, 10, 10] = 0.75
    global_map[..., 5, 10, 10] = 1.0
    batch = {
        "actions": torch.zeros(1, 1, 1, 3),
        "observations": {"global_map": global_map},
        "outcome_category": torch.tensor([0]),
        "individual_rewards": torch.zeros(1, 1, 1),
        "observer_rewards": torch.zeros(1, 1, 1),
    }
    args = SimpleNamespace(
        success_weight=1.0,
        goal_found_failure_weight=0.1,
        goal_not_found_weight=0.01,
        crash_reward_threshold=-100.0,
        crash_lookback_steps=12,
        warning_focus_weight=0.0,
        observer_warning_focus_weight=2.0,
    )

    weights = target_sample_weights(batch, args)

    assert torch.allclose(weights, torch.tensor([[[2.5]]]))


def test_supervised_contrastive_loss_rewards_task_separation() -> None:
    labels = torch.tensor([0, 0, 1, 1])
    collapsed = torch.ones(4, 2)
    separated = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]]
    )

    collapsed_loss, collapsed_margin = supervised_contrastive_loss(
        collapsed, labels, 0.1
    )
    separated_loss, separated_margin = supervised_contrastive_loss(
        separated, labels, 0.1
    )

    assert separated_loss < collapsed_loss
    assert separated_margin > collapsed_margin
