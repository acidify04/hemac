"""Tests for step-budgeted VAE skill-control orchestration."""

import json

from src.skill_discovery.run_vae_skill_control import curve_is_complete


def write_curve(path, *, iteration: int, joint_env_steps: int) -> None:
    path.write_text(
        json.dumps(
            {
                "points": [
                    {
                        "iteration": iteration,
                        "joint_env_steps": joint_env_steps,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_curve_completion_uses_joint_steps_when_budget_is_configured(tmp_path) -> None:
    curve = tmp_path / "curve.json"
    write_curve(curve, iteration=900, joint_env_steps=499_999)

    assert not curve_is_complete(curve, 40, 500_000)

    write_curve(curve, iteration=901, joint_env_steps=500_042)
    assert curve_is_complete(curve, 40, 500_000)


def test_curve_completion_remains_iteration_based_without_budget(tmp_path) -> None:
    curve = tmp_path / "curve.json"
    write_curve(curve, iteration=40, joint_env_steps=25_000)

    assert curve_is_complete(curve, 40)
