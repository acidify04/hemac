"""Configuration, validation, and reporting helpers for difficulty transfer."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .tasks import (
    DEFAULT_DIFFICULTY_STRENGTHS,
    DEFAULT_SOURCE_DIFFICULTIES,
    DEFAULT_TARGET_DIFFICULTIES,
    TaskSpec,
    build_difficulty_tasks,
)


QUALITY_FRACTIONS = {"high": 1.0, "mid": 2.0 / 3.0, "low": 1.0 / 3.0}
DEFAULT_EXPERIMENT_SEEDS = (1, 2, 3, 4, 5)


@dataclass(frozen=True)
class DifficultyProtocol:
    """Resolved user-configurable difficulty split."""

    strengths: tuple[float, ...] = DEFAULT_DIFFICULTY_STRENGTHS
    source_ids: tuple[str, ...] = DEFAULT_SOURCE_DIFFICULTIES
    target_ids: tuple[str, ...] = DEFAULT_TARGET_DIFFICULTIES

    def __post_init__(self) -> None:
        build_difficulty_tasks(self.strengths, self.source_ids, self.target_ids)

    @property
    def tasks(self) -> tuple[TaskSpec, ...]:
        return build_difficulty_tasks(
            self.strengths, self.source_ids, self.target_ids
        )

    def task(self, difficulty_id: str) -> TaskSpec:
        for task in self.tasks:
            if task.name == difficulty_id:
                return task
        raise ValueError(f"Unknown difficulty: {difficulty_id}")

    def to_dict(self) -> dict[str, object]:
        return {
            "strengths": {
                task.name: task.actuator_scale for task in self.tasks
            },
            "source_difficulties": list(self.source_ids),
            "target_difficulties": list(self.target_ids),
        }


def add_difficulty_arguments(parser: argparse.ArgumentParser) -> None:
    """Expose the fixed default protocol without hiding its configurable parts."""
    parser.add_argument(
        "--difficulty-strengths",
        nargs=4,
        type=float,
        default=DEFAULT_DIFFICULTY_STRENGTHS,
        metavar=("D1", "D2", "D3", "D4"),
    )
    parser.add_argument(
        "--source-difficulties",
        nargs="+",
        default=DEFAULT_SOURCE_DIFFICULTIES,
    )
    parser.add_argument(
        "--target-difficulties",
        nargs="+",
        default=DEFAULT_TARGET_DIFFICULTIES,
    )


def protocol_from_args(args: argparse.Namespace) -> DifficultyProtocol:
    return DifficultyProtocol(
        strengths=tuple(args.difficulty_strengths),
        source_ids=tuple(args.source_difficulties),
        target_ids=tuple(args.target_difficulties),
    )


def adaptation_evaluation_steps(budget: int, interval: int) -> list[int]:
    """Return step zero and every requested point through the exact budget."""
    budget = int(budget)
    interval = int(interval)
    if budget <= 0 or interval <= 0:
        raise ValueError("adaptation budget and evaluation interval must be positive")
    if budget % interval:
        raise ValueError("adaptation budget must be divisible by evaluation interval")
    return list(range(0, budget + 1, interval))


def reward_auc(env_steps: Sequence[int], returns: Sequence[float]) -> tuple[float, float]:
    """Compute raw and budget-normalized trapezoidal return AUC."""
    x = np.asarray(env_steps, dtype=np.float64)
    y = np.asarray(returns, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 2:
        raise ValueError("AUC requires equally sized one-dimensional curves")
    if x[0] != 0 or np.any(np.diff(x) <= 0):
        raise ValueError("AUC steps must start at zero and be strictly increasing")
    budget = float(x[-1])
    if budget <= 0:
        raise ValueError("AUC budget must be positive")
    raw = float(np.trapz(y, x))
    return raw, raw / budget


def validate_adaptation_counts(
    target_count: int,
    d1_count: int,
    d2_count: int,
    batch_size: int,
) -> None:
    if target_count + d1_count + d2_count != batch_size:
        raise ValueError("adaptation sample counts must sum to batch size")
    if min(target_count, d1_count, d2_count) <= 0:
        raise ValueError("every adaptation replay component must be positive")


def adaptation_replay_ids(
    protocol: DifficultyProtocol, target_id: str
) -> tuple[str, tuple[str, ...]]:
    """Return one target and source-only replay IDs, rejecting cross-target use."""
    if target_id not in protocol.target_ids:
        raise ValueError(f"{target_id} is not a configured target difficulty")
    replay_ids = tuple(protocol.source_ids)
    if set(replay_ids) & set(protocol.target_ids):
        raise ValueError("Target difficulty leaked into source replay IDs")
    return target_id, replay_ids
