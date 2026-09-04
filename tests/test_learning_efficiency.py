import json

import pytest

from src.skill_discovery.analyze_learning_efficiency import (
    append_curve_points,
    load_curve,
    reference_curve_budgets,
    run_metrics,
    static_baseline_points,
)


def test_run_metrics_uses_trapezoidal_auc_and_first_observed_crossing():
    points = [
        {"joint_env_steps": 0, "success_rate": 0.1},
        {"joint_env_steps": 100, "success_rate": 0.5},
        {"joint_env_steps": 200, "success_rate": 0.9},
    ]

    metrics = run_metrics(points, budget=200, threshold=0.5)

    assert metrics["success_auc"] == pytest.approx(0.5)
    assert metrics["success_gain_auc"] == pytest.approx(0.4)
    assert metrics["initial_success_rate"] == pytest.approx(0.1)
    assert metrics["final_success_rate"] == pytest.approx(0.9)
    assert metrics["final_success_gain"] == pytest.approx(0.8)
    assert metrics["first_threshold_step"] == 100


def test_run_metrics_interpolates_at_budget():
    points = [
        {"joint_env_steps": 0, "success_rate": 0.1},
        {"joint_env_steps": 100, "success_rate": 0.5},
        {"joint_env_steps": 200, "success_rate": 0.9},
    ]

    metrics = run_metrics(points, budget=150, threshold=0.8)

    assert metrics["success_auc"] == pytest.approx(0.4)
    assert metrics["success_gain_auc"] == pytest.approx(0.3)
    assert metrics["final_success_rate"] == pytest.approx(0.7)
    assert metrics["first_threshold_step"] is None
    assert metrics["capped_threshold_step"] == 150


def test_append_curve_points_replaces_the_same_run_step(tmp_path):
    curve_path = tmp_path / "curve.json"
    base = {
        "method": "hissd",
        "seed": 2026,
        "difficulty": 4,
        "joint_env_steps": 0,
        "success_rate": 0.2,
    }
    append_curve_points(curve_path, [base])
    append_curve_points(curve_path, [{**base, "success_rate": 0.3}])

    payload = load_curve(curve_path)

    assert len(payload["points"]) == 1
    assert payload["points"][0]["success_rate"] == pytest.approx(0.3)
    assert json.loads(curve_path.read_text())["step_unit"] == (
        "joint_environment_cycle"
    )


def test_static_baselines_span_the_matching_reference_budget(tmp_path):
    reference_path = tmp_path / "reference.json"
    append_curve_points(
        reference_path,
        [
            {
                "method": "hissd",
                "seed": 2026,
                "difficulty": 4,
                "joint_env_steps": 0,
                "success_rate": 0.2,
            },
            {
                "method": "hissd",
                "seed": 2026,
                "difficulty": 4,
                "joint_env_steps": 200,
                "success_rate": 0.6,
            },
        ],
    )
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(
        json.dumps(
            {
                "summaries": {
                    "bc/difficulty_4": {"success_rate": 0.3},
                    "mappo/difficulty_4": {"success_rate": 0.4},
                }
            }
        )
    )

    budgets = reference_curve_budgets([reference_path], seed=2026)
    points = static_baseline_points(
        evaluation_path,
        {"bc": "bc_checkpoint", "mappo": "mappo_checkpoint"},
        seed=2026,
        budgets=budgets,
    )

    assert budgets == {4: 200}
    assert len(points) == 4
    assert {point["joint_env_steps"] for point in points} == {0, 200}
    assert {point["method"] for point in points} == {
        "bc_checkpoint",
        "mappo_checkpoint",
    }
