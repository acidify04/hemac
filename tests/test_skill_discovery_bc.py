"""Unit tests for the standalone drone behavior-cloning model."""

import torch

from src.skill_discovery.models import DroneBehaviorCloningPolicy
from src.skill_discovery.train_drone_bc import masked_mse


def test_drone_bc_preserves_batch_time_and_agent_dimensions() -> None:
    """The shared model should independently predict every drone action."""
    model = DroneBehaviorCloningPolicy(7, 7)
    prediction = model(
        torch.zeros(2, 4, 3, 7, 40, 40),
        torch.zeros(2, 4, 3, 7, 20, 20),
        torch.zeros(2, 4, 3, 5, 3),
    )

    assert prediction.shape == (2, 4, 3, 3)
    assert torch.all(prediction >= -1.0)
    assert torch.all(prediction <= 1.0)


def test_masked_mse_ignores_padding_and_inactive_agents() -> None:
    """Invalid sequence and agent positions must not contribute to BC loss."""
    prediction = torch.zeros(1, 2, 3, 3)
    target = torch.ones_like(prediction)
    target[0, 1] = 100.0
    mask = torch.tensor([[[True, False, True], [False, False, False]]])

    loss = masked_mse(prediction, target, mask)

    assert torch.isclose(loss, torch.tensor(1.0))
