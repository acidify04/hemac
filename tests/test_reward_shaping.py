"""Tests for reward shaping helpers."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hemac.helpers.reward_shaping import proximity_penalty_from_distances


def test_proximity_penalty_is_zero_when_no_hazard_is_in_range():
    penalty = proximity_penalty_from_distances([50.0, 50.0, 50.0, 50.0], sensing_range=50.0, penalty_scale=0.02)

    assert penalty == pytest.approx(0.0)


def test_proximity_penalty_grows_as_hazard_gets_closer():
    far_penalty = proximity_penalty_from_distances([40.0, 50.0, 50.0, 50.0], sensing_range=50.0, penalty_scale=0.02)
    near_penalty = proximity_penalty_from_distances([10.0, 50.0, 50.0, 50.0], sensing_range=50.0, penalty_scale=0.02)

    assert 0.0 < far_penalty < near_penalty < 0.02


def test_proximity_penalty_is_bounded_by_scale():
    penalty = proximity_penalty_from_distances([0.0, 5.0, 12.0, 50.0], sensing_range=50.0, penalty_scale=0.02)

    assert penalty == pytest.approx(0.02)
