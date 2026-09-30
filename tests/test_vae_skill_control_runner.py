"""Tests for step-budgeted VAE skill-control orchestration."""

import json
from types import SimpleNamespace

from src.skill_discovery.run_vae_skill_control import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_QUICK_OUTPUT_ROOT,
    configure_quick_check,
    curve_is_complete,
)


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


def test_quick_check_uses_one_hard_task_and_short_step_budget() -> None:
    args = SimpleNamespace(
        quick_check=True,
        stages=("online",),
        modes=("full",),
        seeds=(2026, 2027),
        difficulties=(3, 4),
        joint_step_budget=None,
        eval_every_joint_steps=None,
        train_batch_joint_steps=1_000,
        eval_episodes=100,
        test_seed_base=200_000_000,
        test_episodes=200,
        ppo_epochs=5,
        minibatch_size=1_024,
        entropy_coeff=0.01,
        eval_seed_base=None,
        output_root=DEFAULT_OUTPUT_ROOT,
    )

    configure_quick_check(args)

    assert args.stages == ("online", "analyze")
    assert args.modes == ("full", "no_skill")
    assert args.seeds == (2026,)
    assert args.difficulties == (4,)
    assert args.joint_step_budget == 50_000
    assert args.eval_every_joint_steps == 10_000
    assert args.train_batch_joint_steps == 1_200
    assert args.eval_episodes == 50
    assert args.ppo_epochs == 8
    assert args.minibatch_size == 256
    assert args.entropy_coeff == 0.002
    assert args.output_root == DEFAULT_QUICK_OUTPUT_ROOT
