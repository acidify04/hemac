"""Tests for the drone-only skill discovery task definition."""

import torch

from src.skill_discovery.drone_task import (
    classify_drone_skill_outcome,
    drone_skill_outcome_from_payload,
)


def test_drone_skill_outcome_boundary() -> None:
    assert classify_drone_skill_outcome(True, 0.40) == "success"
    assert classify_drone_skill_outcome(True, 0.399) == "goal_found_failure"
    assert classify_drone_skill_outcome(False, 0.90) == "goal_not_found"
    assert (
        classify_drone_skill_outcome(True, 0.90, fatal_crash=True)
        == "goal_found_failure"
    )


def test_legacy_episode_is_relabelled_from_drone_fields() -> None:
    payload = {
        "outcome": {
            "success": torch.tensor(False),
            "drone_goal_found": torch.tensor(True),
        },
        "metadata": {"final_info": {"coverage_ratio": 0.45}},
    }

    category, goal_found, coverage = drone_skill_outcome_from_payload(payload)

    assert category == "success"
    assert goal_found
    assert coverage == 0.45


def test_crashed_episode_is_never_relabelled_as_success() -> None:
    payload = {
        "outcome": {
            "drone_goal_found": torch.tensor(True),
            "fatal_crash": torch.tensor(True),
        },
        "metadata": {
            "final_info": {
                "coverage_ratio": 0.90,
                "fatal_crash": True,
            }
        },
    }

    category, goal_found, coverage = drone_skill_outcome_from_payload(payload)

    assert category == "goal_found_failure"
    assert goal_found
    assert coverage == 0.90
