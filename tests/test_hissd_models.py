"""Shape and objective tests for the HeMAC HiSSD adaptation."""

from types import SimpleNamespace

import torch

from src.skill_discovery.hissd_models import HeMACHISSD
from src.skill_discovery.train_hissd import (
    controller_objective,
    planner_objective,
    value_objective,
)


def make_batch() -> dict:
    """Create two source tasks so the MoCo objective has valid negatives."""
    batch_size, sequence_length, agent_count = 2, 3, 3

    def observations() -> dict[str, torch.Tensor]:
        return {
            "global_map": torch.rand(batch_size, sequence_length, agent_count, 7, 40, 40),
            "local_map": torch.rand(batch_size, sequence_length, agent_count, 7, 20, 20),
            "action_history": torch.rand(batch_size, sequence_length, agent_count, 5, 3) * 2 - 1,
        }

    return {
        "observations": observations(),
        "next_observations": observations(),
        "actions": torch.rand(batch_size, sequence_length, agent_count, 3) * 2 - 1,
        "central_map": torch.rand(batch_size, sequence_length, 7, 20, 20),
        "next_central_map": torch.rand(batch_size, sequence_length, 7, 20, 20),
        "team_reward": torch.rand(batch_size, sequence_length),
        "task_id": torch.tensor([0, 1]),
        "valid_agents": torch.ones(
            batch_size, sequence_length, agent_count, dtype=torch.bool
        ),
        "valid_steps": torch.ones(batch_size, sequence_length, dtype=torch.bool),
        "done": torch.zeros(batch_size, sequence_length, dtype=torch.bool),
    }


def objective_args() -> SimpleNamespace:
    """Return the official HiSSD defaults used by objective functions."""
    return SimpleNamespace(
        beta=0.05,
        gamma=0.99,
        expectile=0.9,
        alpha=10.0,
        reward_scale=100.0,
        contrastive_temperature=0.1,
        max_contrastive_samples=256,
    )


def test_hissd_objectives_are_finite_and_shape_preserving() -> None:
    """All official training phases should accept the HeMAC tensor schema."""
    model = HeMACHISSD(7, 7, 7)
    batch = make_batch()
    features = model.encode_observations(batch["observations"])
    common, task_skill, contrastive = model.infer_skills(
        features, batch["valid_agents"]
    )
    actions = model.decode_actions(features, common, task_skill)

    assert features.shape == (2, 3, 3, 96)
    assert common.shape == (2, 3, 3, 64)
    assert task_skill.shape == (2, 3, 3, 64)
    assert contrastive.shape == (2, 3, 3, 64)
    assert actions.shape == (2, 3, 3, 3)

    args = objective_args()
    controller_loss, controller_metrics = controller_objective(model, batch, args)
    value_loss, _ = value_objective(model, batch, args)
    planner_loss, _ = planner_objective(model, batch, args)

    assert torch.isfinite(controller_loss)
    assert torch.isfinite(value_loss)
    assert torch.isfinite(planner_loss)
    assert controller_metrics["contrastive_samples"] > 0
