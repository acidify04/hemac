"""Tests for the maintained MaMuJoCo transfer pipeline."""

import argparse
import importlib.util
import json

import numpy as np
import pytest
import torch

from mamujoco.dataset import MultiTaskTrajectoryDataset
from mamujoco.difficulty_protocol import (
    DifficultyProtocol,
    adaptation_evaluation_steps,
    adaptation_replay_ids,
    reward_auc,
    validate_adaptation_counts,
)
from mamujoco.env import (
    DEFAULT_ENVIRONMENT_VERSION,
    add_environment_version_argument,
    infer_space_dimensions,
    make_env,
    mask_disabled_action,
)
from mamujoco.evaluate_zero_shot import evaluate_episode
from mamujoco.happo import EpisodeBatch, HAPPOTrainer
from mamujoco.models import CentralCritic, IndependentActors
from mamujoco.offline_models import (
    HISSD_ALPHA,
    HISSD_BETA,
    HISSD_EXPECTILE,
    HISSD_SKILL_DIM,
    HiSSD,
    MultiAgentSkillVAE,
)
from mamujoco.pilot_difficulty import build_metric_matrix, cross_evaluation_pairs
from mamujoco.tasks import AGENT_IDS, build_difficulty_tasks, get_task, list_tasks


OBSERVATION_DIMS = {
    "agent_0": 9,
    "agent_1": 9,
    "agent_2": 8,
    "agent_3": 9,
    "agent_4": 9,
    "agent_5": 8,
}
ACTION_DIMS = {agent: 1 for agent in AGENT_IDS}


def test_task_suites_have_five_source_and_two_unseen_tasks():
    for suite in ("dynamics", "joint_disable"):
        assert len(list_tasks(suite, "source")) == 5
        assert len(list_tasks(suite, "target")) == 2
    assert get_task("joint_disable", "back_shin").disabled_agent == "agent_1"
    assert get_task("joint_disable", "front_foot").disabled_agent == "agent_5"


def test_paper_hyperparameters_and_baseline_skill_dimension_match():
    assert HISSD_ALPHA == 10.0
    assert HISSD_BETA == 2.0
    assert HISSD_EXPECTILE == 0.9
    model = MultiAgentSkillVAE(OBSERVATION_DIMS, ACTION_DIMS, state_dim=17)
    assert model.skill_dim == HISSD_SKILL_DIM == 256


def test_default_difficulty_protocol_and_independent_target_replay():
    tasks = build_difficulty_tasks()
    assert [(task.name, task.actuator_scale) for task in tasks] == [
        ("D1", 1.0),
        ("D2", 0.8),
        ("D3", 0.6),
        ("D4", 0.4),
    ]
    assert [task.name for task in tasks if task.split == "source"] == ["D1", "D2"]
    assert [task.name for task in tasks if task.split == "target"] == ["D3", "D4"]
    protocol = DifficultyProtocol()
    assert adaptation_replay_ids(protocol, "D3") == ("D3", ("D1", "D2"))
    assert adaptation_replay_ids(protocol, "D4") == ("D4", ("D1", "D2"))
    with pytest.raises(ValueError, match="two source and two target"):
        DifficultyProtocol(source_ids=("D1",), target_ids=("D2", "D3", "D4"))


def test_default_adaptation_counts_schedule_and_auc():
    validate_adaptation_counts(64, 32, 32, 128)
    points = adaptation_evaluation_steps(500_000, 10_000)
    assert len(points) == 51
    assert points[0] == 0
    assert points[-1] == 500_000
    raw, normalized = reward_auc([0, 10, 20], [0.0, 10.0, 20.0])
    assert raw == pytest.approx(200.0)
    assert normalized == pytest.approx(10.0)


def test_cross_evaluation_has_exactly_sixteen_unique_pairs():
    pairs = cross_evaluation_pairs(("D1", "D2", "D3", "D4"))
    assert len(pairs) == 16
    assert len(set(pairs)) == 16
    assert set(pairs) == {
        (training, evaluation)
        for training in ("D1", "D2", "D3", "D4")
        for evaluation in ("D1", "D2", "D3", "D4")
    }


def test_cross_evaluation_matrix_rows_are_training_and_columns_are_evaluation():
    difficulty_ids = ("D1", "D2", "D3", "D4")
    records = [
        {
            "training_difficulty": training,
            "evaluation_difficulty": evaluation,
            "episode_return_mean": float(row_index * 10 + column_index),
        }
        for row_index, training in enumerate(difficulty_ids)
        for column_index, evaluation in enumerate(difficulty_ids)
    ]
    matrix = build_metric_matrix(records, difficulty_ids, "episode_return")
    assert matrix["row_axis"] == "training_difficulty"
    assert matrix["column_axis"] == "evaluation_difficulty"
    assert matrix["rows"] == list(difficulty_ids)
    assert matrix["columns"] == list(difficulty_ids)
    assert matrix["values"][2][3] == 23.0


def test_canonical_environment_version_default_and_v5_override():
    parser = argparse.ArgumentParser()
    add_environment_version_argument(parser)
    assert DEFAULT_ENVIRONMENT_VERSION == "HalfCheetah-v2"
    assert parser.parse_args([]).environment_version == "HalfCheetah-v2"
    assert (
        parser.parse_args(["--environment-version", "HalfCheetah-v5"])
        .environment_version
        == "HalfCheetah-v5"
    )


def test_difficulty_label_does_not_change_policy_action():
    model = HiSSD(
        OBSERVATION_DIMS,
        ACTION_DIMS,
        state_dim=17,
        num_source_tasks=2,
        hidden_dim=32,
        skill_dim=16,
        history_length=4,
    ).eval()
    batch = _synthetic_batch(batch_size=2, sequence_length=1)
    batch["history_observations"] = {
        agent: value.unsqueeze(2).expand(-1, -1, 4, -1).clone()
        for agent, value in batch["observations"].items()
    }
    batch["history_valid"] = torch.ones(2, 1, 4, dtype=torch.bool)
    batch["task_id"] = torch.tensor([0, 0])
    first = model(batch)["actions"]
    batch["task_id"] = torch.tensor([1, 1])
    second = model(batch)["actions"]
    assert all(torch.equal(first[agent], second[agent]) for agent in AGENT_IDS)


def test_one_step_hissd_inference_resets_only_common_recurrent_state():
    model = HiSSD(
        OBSERVATION_DIMS,
        ACTION_DIMS,
        state_dim=17,
        num_source_tasks=2,
        hidden_dim=32,
        skill_dim=16,
        history_length=4,
        training_sequence_length=1,
    ).eval()
    observations = {
        agent: torch.randn(1, dim) for agent, dim in OBSERVATION_DIMS.items()
    }
    state = model.initial_inference_state(1, device=torch.device("cpu"))

    with torch.no_grad():
        _, state = model.act_step(observations, state)
        _, state = model.act_step(observations, state)

    assert all(torch.count_nonzero(value) == 0 for value in state["common"].values())
    assert state["task_history_valid"].sum().item() == 2


def test_multi_step_hissd_inference_carries_common_recurrent_state():
    model = HiSSD(
        OBSERVATION_DIMS,
        ACTION_DIMS,
        state_dim=17,
        num_source_tasks=2,
        hidden_dim=32,
        skill_dim=16,
        history_length=4,
        training_sequence_length=4,
    ).eval()
    observations = {
        agent: torch.randn(1, dim) for agent, dim in OBSERVATION_DIMS.items()
    }
    state = model.initial_inference_state(1, device=torch.device("cpu"))

    with torch.no_grad():
        _, state = model.act_step(observations, state)

    assert any(torch.count_nonzero(value) > 0 for value in state["common"].values())


def test_hissd_inference_masks_disabled_agent_like_offline_training():
    model = HiSSD(
        OBSERVATION_DIMS,
        ACTION_DIMS,
        state_dim=17,
        num_source_tasks=2,
        hidden_dim=32,
        skill_dim=16,
        history_length=4,
    ).eval()
    observations = {
        agent: torch.randn(1, dim) for agent, dim in OBSERVATION_DIMS.items()
    }
    active_agents = torch.ones(1, len(AGENT_IDS), dtype=torch.bool)
    active_agents[:, 3] = False
    state = model.initial_inference_state(1, device=torch.device("cpu"))

    with torch.no_grad():
        _, state = model.act_step(
            observations, state, active_agents=active_agents
        )

    assert torch.count_nonzero(state["common"][AGENT_IDS[3]]) == 0
    assert torch.count_nonzero(state["last_task_skill"][:, 3]) == 0


def test_disabled_action_is_zeroed_without_mutating_input():
    actions = {agent: np.ones((1,), dtype=np.float32) for agent in AGENT_IDS}
    masked = mask_disabled_action(actions, "agent_2")
    assert masked["agent_2"].item() == 0.0
    assert actions["agent_2"].item() == 1.0
    assert all(masked[agent].item() == 1.0 for agent in AGENT_IDS if agent != "agent_2")


def _synthetic_batch(batch_size=3, sequence_length=4):
    return {
        "observations": {
            agent: torch.randn(batch_size, sequence_length, dim)
            for agent, dim in OBSERVATION_DIMS.items()
        },
        "next_observations": {
            agent: torch.randn(batch_size, sequence_length, dim)
            for agent, dim in OBSERVATION_DIMS.items()
        },
        "actions": {
            agent: torch.tanh(torch.randn(batch_size, sequence_length, 1))
            for agent in AGENT_IDS
        },
        "states": torch.randn(batch_size, sequence_length, 17),
        "next_states": torch.randn(batch_size, sequence_length, 17),
        "rewards": torch.randn(batch_size, sequence_length),
        "terminations": torch.zeros(batch_size, sequence_length, dtype=torch.bool),
        "valid": torch.ones(batch_size, sequence_length, dtype=torch.bool),
        "active_agents": torch.ones(batch_size, 6, dtype=torch.bool),
        "task_id": torch.tensor([0, 1, 2]),
    }


@pytest.mark.parametrize("model_type", (HiSSD, MultiAgentSkillVAE))
def test_offline_models_compute_finite_loss(model_type):
    model = model_type(
        OBSERVATION_DIMS,
        ACTION_DIMS,
        state_dim=17,
        hidden_dim=32,
        skill_dim=16,
    )
    loss, metrics = model.loss(_synthetic_batch())
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in metrics.values())
    loss.backward()


def test_happo_skips_disabled_actor_and_updates_others():
    actors = IndependentActors(OBSERVATION_DIMS, ACTION_DIMS, hidden_dim=32)
    critic = CentralCritic(17, hidden_dim=32)
    count = 12
    observations = {
        agent: torch.randn(count, dim) for agent, dim in OBSERVATION_DIMS.items()
    }
    raw_actions = {agent: torch.randn(count, 1) for agent in AGENT_IDS}
    old_log_probs = {
        agent: actors.actors[agent]
        .distribution(observations[agent])
        .log_prob(raw_actions[agent])
        .sum(-1)
        .detach()
        for agent in AGENT_IDS
    }
    batch = EpisodeBatch(
        observations=observations,
        raw_actions=raw_actions,
        actions={agent: torch.tanh(raw_actions[agent]) for agent in AGENT_IDS},
        old_log_probs=old_log_probs,
        active_masks={
            agent: torch.full((count,), agent != "agent_1", dtype=torch.bool)
            for agent in AGENT_IDS
        },
        states=torch.randn(count, 17),
        rewards=torch.randn(count),
        advantages=torch.randn(count),
        returns=torch.randn(count),
        metrics={},
    )
    disabled_before = {
        name: value.clone() for name, value in actors.actors["agent_1"].state_dict().items()
    }
    trainer = HAPPOTrainer(
        actors,
        critic,
        ppo_epochs=1,
        minibatch_size=count,
        device=torch.device("cpu"),
    )
    metrics = trainer.update(batch)
    assert metrics["actor_loss/agent_1"] == 0.0
    assert all(
        torch.equal(disabled_before[name], value)
        for name, value in actors.actors["agent_1"].state_dict().items()
    )


def test_dataset_builds_fixed_windows_for_all_source_tasks(tmp_path):
    suite_root = tmp_path / "joint_disable"
    manifest = {
        "format_version": 1,
        "suite": "joint_disable",
        "tasks": {},
    }
    for task in list_tasks("joint_disable", "source"):
        task_dir = suite_root / task.name
        task_dir.mkdir(parents=True)
        relative_path = f"joint_disable/{task.name}/episode_0000.pt"
        length = 5
        payload = {
            "observations": {
                agent: torch.randn(length, OBSERVATION_DIMS[agent])
                for agent in AGENT_IDS
            },
            "next_observations": {
                agent: torch.randn(length, OBSERVATION_DIMS[agent])
                for agent in AGENT_IDS
            },
            "actions": {agent: torch.randn(length, 1) for agent in AGENT_IDS},
            "states": torch.randn(length, 17),
            "next_states": torch.randn(length, 17),
            "rewards": torch.randn(length),
            "terminations": torch.zeros(length, dtype=torch.bool),
            "truncations": torch.zeros(length, dtype=torch.bool),
        }
        torch.save(payload, tmp_path / relative_path)
        manifest["tasks"][task.name] = {
            "spec": task.to_dict(),
            "episodes": [{"path": relative_path, "transitions": length}],
        }
    manifest_path = suite_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    dataset = MultiTaskTrajectoryDataset(
        manifest_path, sequence_length=4, stride=4
    )
    assert len(dataset) == 10
    item = dataset[0]
    assert item["states"].shape == (4, 17)
    assert item["valid"].all()
    assert item["active_agents"].shape == (6,)
    assert item["quality"] == "unspecified"
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2)))
    assert batch["quality"] == ["unspecified", "unspecified"]


def test_difficulty_dataset_rejects_target_leakage(tmp_path):
    manifest = {
        "format_version": 1,
        "suite": "difficulty",
        "difficulty_protocol": DifficultyProtocol().to_dict(),
        "tasks": {
            name: {
                "spec": get_task("difficulty", name).to_dict(),
                "episodes": [],
            }
            for name in ("D1", "D2", "D3")
        },
    }
    manifest_path = tmp_path / "difficulty" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Target difficulty leakage"):
        MultiTaskTrajectoryDataset(manifest_path)


@pytest.mark.skipif(
    importlib.util.find_spec("gymnasium_robotics") is None,
    reason="MaMuJoCo integration dependency is not installed",
)
def test_latest_mamujoco_joint_disable_integration():
    env = make_env(get_task("joint_disable", "back_thigh"), max_cycles=2)
    try:
        assert env.environment_version == "HalfCheetah-v2"
        assert env.env.single_agent_env.spec.id == "HalfCheetah-v2"
        observations, _ = env.reset(seed=3)
        observation_dims, action_dims, state_dim = infer_space_dimensions(env)
        assert observation_dims == OBSERVATION_DIMS
        assert action_dims == ACTION_DIMS
        assert state_dim == 17
        actions = {agent: np.ones((1,), dtype=np.float32) for agent in AGENT_IDS}
        _, _, _, _, infos = env.step(actions)
        assert env.last_executed_actions["agent_0"].item() == 0.0
        assert infos["agent_0"]["disabled"] is True
        assert infos["agent_1"]["disabled"] is False
    finally:
        env.close()


@pytest.mark.skipif(
    importlib.util.find_spec("gymnasium_robotics") is None,
    reason="MaMuJoCo integration dependency is not installed",
)
def test_latest_mamujoco_dynamics_intervention():
    env = make_env(get_task("dynamics", "mass_075"), max_cycles=2)
    try:
        positive = env._nominal_body_mass > 0.0
        ratio = (
            env._physics_env.model.body_mass[positive]
            / env._nominal_body_mass[positive]
        )
        assert np.allclose(ratio, 0.75)
        env.reset(seed=4)
        assert np.allclose(
            env._physics_env.model.body_mass[positive]
            / env._nominal_body_mass[positive],
            0.75,
        )
    finally:
        env.close()


@pytest.mark.skipif(
    importlib.util.find_spec("gymnasium_robotics") is None,
    reason="MaMuJoCo integration dependency is not installed",
)
@pytest.mark.parametrize("difficulty,strength", (("D1", 1.0), ("D2", 0.8), ("D3", 0.6), ("D4", 0.4)))
def test_difficulty_actuator_scaling_is_exact_and_non_cumulative(
    difficulty, strength
):
    env = make_env(get_task("difficulty", difficulty), max_cycles=2)
    try:
        nominal = env._nominal_actuator_gear.copy()
        assert np.allclose(env._physics_env.model.actuator_gear, nominal * strength)
        env.reset(seed=10)
        assert np.allclose(env._physics_env.model.actuator_gear, nominal * strength)
        env.reset(seed=11)
        assert np.allclose(env._physics_env.model.actuator_gear, nominal * strength)
    finally:
        env.close()


@pytest.mark.skipif(
    importlib.util.find_spec("gymnasium_robotics") is None,
    reason="MaMuJoCo integration dependency is not installed",
)
def test_zero_shot_evaluation_performs_no_parameter_update():
    env = make_env(get_task("difficulty", "D3"), max_cycles=2)
    model = HiSSD(
        OBSERVATION_DIMS,
        ACTION_DIMS,
        state_dim=17,
        num_source_tasks=2,
        hidden_dim=32,
        skill_dim=16,
        history_length=4,
    ).eval()
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    try:
        evaluate_episode(
            env,
            model,
            seed=17,
            device=torch.device("cpu"),
            render=False,
        )
    finally:
        env.close()
    assert all(
        torch.equal(before[name], parameter)
        for name, parameter in model.named_parameters()
    )
