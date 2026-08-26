"""Shared outcome semantics for drone-only skill discovery."""

from __future__ import annotations

from typing import Any

import torch


DRONE_SKILL_SUCCESS_MIN_COVERAGE_RATIO = 0.4
DRONE_SKILL_SUCCESS_REWARD = 300.0


def scalar_bool(value: Any) -> bool:
    """Convert a persisted scalar tensor or Python value to bool."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Expected a scalar bool, got {tuple(value.shape)}.")
        return bool(value.item())
    return bool(value)


def classify_drone_skill_outcome(
    drone_goal_found: bool,
    coverage_ratio: float,
    min_coverage_ratio: float = DRONE_SKILL_SUCCESS_MIN_COVERAGE_RATIO,
) -> str:
    """Classify an episode using the drone exploration task definition."""
    if not 0.0 <= min_coverage_ratio <= 1.0:
        raise ValueError("min_coverage_ratio must be in [0, 1].")
    if bool(drone_goal_found) and float(coverage_ratio) >= min_coverage_ratio:
        return "success"
    if bool(drone_goal_found):
        return "goal_found_failure"
    return "goal_not_found"


def drone_skill_outcome_from_payload(
    payload: dict[str, Any],
    min_coverage_ratio: float = DRONE_SKILL_SUCCESS_MIN_COVERAGE_RATIO,
) -> tuple[str, bool, float]:
    """Derive the drone-task category from an existing offline episode."""
    outcome = payload.get("outcome", {})
    final_info = payload.get("metadata", {}).get("final_info", {})
    drone_goal_found = scalar_bool(
        outcome.get(
            "drone_goal_found",
            final_info.get("drone_goal_found", False),
        )
    )
    if "coverage_ratio" not in final_info:
        raise ValueError("Episode final_info has no coverage_ratio.")
    coverage_ratio = float(final_info["coverage_ratio"])
    return (
        classify_drone_skill_outcome(
            drone_goal_found,
            coverage_ratio,
            min_coverage_ratio,
        ),
        drone_goal_found,
        coverage_ratio,
    )
