import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hemac.environment.HeMAC import heading_alignment_reward


def test_heading_alignment_reward_is_max_when_facing_goal():
    reward = heading_alignment_reward(
        agent_x=0.0,
        agent_y=0.0,
        orientation=0.0,
        goal_x=10.0,
        goal_y=0.0,
        reward_scale=0.01,
    )

    assert reward == pytest.approx(0.01)


def test_heading_alignment_reward_is_zero_when_facing_away():
    reward = heading_alignment_reward(
        agent_x=0.0,
        agent_y=0.0,
        orientation=math.pi,
        goal_x=10.0,
        goal_y=0.0,
        reward_scale=0.01,
    )

    assert reward == pytest.approx(0.0)


def test_heading_alignment_reward_scales_with_alignment():
    reward = heading_alignment_reward(
        agent_x=0.0,
        agent_y=0.0,
        orientation=math.pi / 4,
        goal_x=10.0,
        goal_y=0.0,
        reward_scale=0.01,
    )

    assert reward == pytest.approx(math.cos(math.pi / 4) * 0.01)
