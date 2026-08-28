"""Shape and objective tests for the HeMAC HiSSD adaptation."""

from types import SimpleNamespace

import torch

from src.skill_discovery.hissd_models import HeMACHISSD
from src.skill_discovery.models import ObserverTaskResidualPolicy
from src.skill_discovery.train_hissd import (
    controller_objective,
    planner_objective,
    value_objective,
)
from src.skill_discovery.task_descriptor import TASK_DESCRIPTOR_NAMES


def make_batch() -> dict:
    """Create two source tasks so the MoCo objective has valid negatives."""
    batch_size, sequence_length, agent_count = 4, 3, 3

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
        "task_id": torch.tensor([0, 0, 1, 1]),
        "episode_id": torch.tensor([0, 1, 2, 3]),
        "task_descriptor": torch.tensor(
            [
                [0.333, 0.444, 0.667, 0.143, 0.429, 0.475, 0.550],
                [0.333, 0.444, 0.667, 0.143, 0.429, 0.475, 0.550],
                [0.444, 0.556, 0.667, 0.286, 0.429, 0.550, 0.650],
                [0.444, 0.667, 1.000, 0.429, 0.714, 0.650, 0.750],
            ],
            dtype=torch.float32,
        ),
        "task_descriptor_available": torch.ones(batch_size, dtype=torch.bool),
        "task_supervision": torch.ones(batch_size, dtype=torch.bool),
        "valid_agents": torch.ones(
            batch_size, sequence_length, agent_count, dtype=torch.bool
        ),
        "valid_steps": torch.ones(batch_size, sequence_length, dtype=torch.bool),
        "done": torch.zeros(batch_size, sequence_length, dtype=torch.bool),
    }


def objective_args() -> SimpleNamespace:
    """Return the official HiSSD defaults used by objective functions."""
    return SimpleNamespace(
        descriptor_weight=1.0,
        descriptor_metric_weight=0.1,
        task_contrastive_weight=0.1,
        task_contrastive_temperature=0.1,
        task_contrastive_tail_steps=2,
        task_label_smoothing=0.1,
        gamma=0.99,
        expectile=0.9,
        alpha=10.0,
        reward_scale=100.0,
        contrastive_temperature=0.1,
        max_contrastive_samples=256,
    )


def test_hissd_objectives_are_finite_and_shape_preserving() -> None:
    """All official training phases should accept the HeMAC tensor schema."""
    model = HeMACHISSD(
        7,
        7,
        7,
        contrastive_from_action_skill=True,
        task_context_pooling=True,
        task_descriptor_dim=len(TASK_DESCRIPTOR_NAMES),
        task_prior_count=3,
    )
    batch = make_batch()
    features = model.encode_observations(batch["observations"])
    common, task_skill, contrastive = model.infer_skills(
        features, batch["valid_agents"]
    )
    actions = model.decode_actions(features, common, task_skill)

    assert features.shape == (4, 3, 3, 96)
    assert common.shape == (4, 3, 3, 64)
    assert task_skill.shape == (4, 3, 3, 64)
    assert contrastive.shape == (4, 3, 3, 64)
    assert actions.shape == (4, 3, 3, 3)

    args = objective_args()
    controller_loss, controller_metrics = controller_objective(model, batch, args)
    value_loss, _ = value_objective(model, batch, args)
    planner_loss, _ = planner_objective(model, batch, args)

    assert torch.isfinite(controller_loss)
    assert torch.isfinite(value_loss)
    assert torch.isfinite(planner_loss)
    assert controller_metrics["descriptor_samples"] == 4
    assert controller_metrics["descriptor_mae"] >= 0.0
    assert 0.0 <= controller_metrics["task_contrastive_accuracy"] <= 1.0
    controller_loss.backward()
    assert model.task_descriptor_head[-2].weight.grad is not None
    assert model.task_skill_encoder.context_gru.weight_hh.grad is not None


def test_same_task_batch_has_finite_controller_gradients() -> None:
    """A batch without task-distance variation must not corrupt the encoder."""
    model = HeMACHISSD(
        7,
        7,
        7,
        contrastive_from_action_skill=True,
        task_context_pooling=True,
        task_descriptor_dim=len(TASK_DESCRIPTOR_NAMES),
        task_prior_count=3,
    )
    batch = make_batch()
    batch["task_id"].zero_()
    batch["task_descriptor"][:] = batch["task_descriptor"][0]

    loss, metrics = controller_objective(model, batch, objective_args())
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["descriptor_metric_loss"] == 0.0
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_task_auxiliary_ablation_keeps_action_training_active() -> None:
    """Removing both auxiliary heads must retain finite action gradients."""
    model = HeMACHISSD(
        7,
        7,
        7,
        contrastive_from_action_skill=True,
        task_context_pooling=True,
        task_descriptor_dim=0,
        task_prior_count=0,
        learned_task_classifier=False,
    )

    loss, metrics = controller_objective(model, make_batch(), objective_args())
    loss.backward()

    assert torch.isfinite(loss)
    assert model.task_descriptor_head is None
    assert model.task_classifier_head is None
    assert metrics["descriptor_loss"] == 0.0
    assert metrics["task_contrastive_loss"] == 0.0
    assert model.task_skill_encoder.context_gru.weight_hh.grad is not None


def test_task_action_residual_upgrade_preserves_actions_and_serializes() -> None:
    """An old checkpoint can gain the task head without initial policy drift."""
    torch.manual_seed(11)
    model = HeMACHISSD(7, 7, 7).eval()
    features = torch.randn(2, 3, model.observation_encoder.output_dim)
    common = torch.randn(2, 3, model.skill_dim)
    task = torch.randn(2, 3, model.skill_dim)

    with torch.inference_mode():
        before = model.decode_actions(features, common, task)
        model.enable_task_action_residual()
        after = model.decode_actions(features, common, task)

    assert torch.equal(before, after)
    assert model.config()["task_action_residual"] is True
    restored = HeMACHISSD(**model.config())
    restored.load_state_dict(model.state_dict())
    with torch.inference_mode():
        assert torch.equal(after, restored.decode_actions(features, common, task))


def test_observer_task_residual_starts_at_zero_and_serializes() -> None:
    """Joint fine-tuning must initially preserve the MAPPO observer action."""
    policy = ObserverTaskResidualPolicy(6, 6, task_skill_dim=64).eval()
    global_map = torch.rand(2, 6, 40, 40)
    local_map = torch.rand(2, 6, 20, 20)
    action_history = torch.rand(2, 5, 3)
    task_skill = torch.rand(2, 64)

    with torch.inference_mode():
        residual, features = policy(
            global_map, local_map, action_history, task_skill
        )

    assert residual.shape == (2, 3)
    assert features.shape == (2, 96)
    assert torch.count_nonzero(residual) == 0
    restored = ObserverTaskResidualPolicy(**policy.config()).eval()
    restored.load_state_dict(policy.state_dict())
    with torch.inference_mode():
        restored_residual, _ = restored(
            global_map, local_map, action_history, task_skill
        )
    assert torch.equal(residual, restored_residual)


def test_online_inference_matches_batched_sequence() -> None:
    """Step-wise recurrent inference must preserve offline sequence behavior."""
    torch.manual_seed(7)
    model = HeMACHISSD(
        7,
        7,
        7,
        contrastive_from_action_skill=True,
        task_context_pooling=True,
        task_descriptor_dim=len(TASK_DESCRIPTOR_NAMES),
        task_prior_count=3,
        task_feature_deltas=True,
        separate_task_observation_encoder=True,
        learned_task_classifier=True,
        task_spatial_statistics=True,
        normalize_task_context=False,
        task_running_statistics=True,
        direct_task_summary=True,
    ).eval()
    batch = make_batch()
    observations = batch["observations"]
    valid = batch["valid_agents"]

    with torch.inference_mode():
        features = model.encode_observations(observations)
        task_features = model.encode_task_observations(observations)
        common, task, _ = model.infer_skills(features, valid, task_features)
        expected_actions = model.decode_actions(features, common, task)

        state = model.initial_inference_state(batch_size=features.shape[0])
        step_actions = []
        step_common = []
        step_task = []
        step_descriptors = []
        for time_index in range(features.shape[1]):
            step_observations = {
                name: value[:, time_index] for name, value in observations.items()
            }
            outputs, state = model.inference_step(
                step_observations,
                valid[:, time_index],
                state,
            )
            step_actions.append(outputs["actions"])
            step_common.append(outputs["common_skills"])
            step_task.append(outputs["task_skills"])
            step_descriptors.append(outputs["task_descriptor"])

    assert torch.allclose(torch.stack(step_common, dim=1), common, atol=5e-6)
    assert torch.allclose(torch.stack(step_task, dim=1), task, atol=2e-4)
    assert torch.allclose(
        torch.stack(step_actions, dim=1), expected_actions, atol=2e-4
    )
    assert torch.stack(step_descriptors, dim=1).shape == (
        4,
        3,
        len(TASK_DESCRIPTOR_NAMES),
    )
