"""Continuous task descriptors shared by collection, training, and analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


TASK_DESCRIPTOR_NAMES = (
    "min_moving_obstacles",
    "max_moving_obstacles",
    "static_obstacle_count",
    "obstacle_min_speed",
    "obstacle_max_speed",
    "goal_min_base_distance",
    "goal_max_base_distance",
)

# The current curriculum's maximum settings map every component near [0, 1].
TASK_DESCRIPTOR_SCALES = np.asarray(
    (9.0, 9.0, 3.0, 7.0, 7.0, 1000.0, 1000.0),
    dtype=np.float32,
)

REALIZED_TASK_DESCRIPTOR_NAMES = (
    "moving_obstacle_count",
    "static_obstacle_count",
    "obstacle_speed_mean",
    "obstacle_speed_std",
    "obstacle_speed_max",
    "goal_base_distance",
)
REALIZED_TASK_DESCRIPTOR_SCALES = np.asarray(
    (9.0, 3.0, 7.0, 7.0, 7.0, 1000.0),
    dtype=np.float32,
)


def normalize_task_descriptor(raw_values: np.ndarray) -> np.ndarray:
    """Normalize one raw descriptor using fixed curriculum-wide scales."""
    values = np.asarray(raw_values, dtype=np.float32)
    if values.shape != TASK_DESCRIPTOR_SCALES.shape:
        raise ValueError(
            f"Expected descriptor shape {TASK_DESCRIPTOR_SCALES.shape}, got {values.shape}."
        )
    return np.clip(values / TASK_DESCRIPTOR_SCALES, 0.0, 1.0).astype(
        np.float32,
        copy=False,
    )


def build_task_descriptor(environment_config: dict[str, Any]) -> dict[str, Any]:
    """Describe the MDP distribution rather than one episode's random draw."""
    required_names = (
        "min_obstacles",
        "max_obstacles",
        "n_static_obstacles",
        "obstacle_min_speed",
        "obstacle_max_speed",
        "goal_min_base_distance",
        "goal_max_base_distance",
    )
    missing = [name for name in required_names if name not in environment_config]
    if missing:
        raise KeyError(f"Environment config is missing task fields: {missing}")
    raw = np.asarray(
        [float(environment_config[name]) for name in required_names],
        dtype=np.float32,
    )
    return {
        "normalized": normalize_task_descriptor(raw),
        "raw": raw,
        "names": TASK_DESCRIPTOR_NAMES,
        "scales": TASK_DESCRIPTOR_SCALES.copy(),
    }


def normalize_realized_task_descriptor(raw_values: np.ndarray) -> np.ndarray:
    """Normalize analysis-only episode realizations."""
    values = np.asarray(raw_values, dtype=np.float32)
    if values.shape != REALIZED_TASK_DESCRIPTOR_SCALES.shape:
        raise ValueError(
            "Expected realized descriptor shape "
            f"{REALIZED_TASK_DESCRIPTOR_SCALES.shape}, got {values.shape}."
        )
    return np.clip(values / REALIZED_TASK_DESCRIPTOR_SCALES, 0.0, 1.0).astype(
        np.float32,
        copy=False,
    )


@dataclass
class TaskDescriptorRecorder:
    """Accumulate realized environment dynamics for one collected episode."""

    moving_obstacle_count: int
    static_obstacle_count: int
    goal_base_distance: float
    initial_moving_speeds: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.float32)
    )

    @classmethod
    def from_environment(cls, core_env) -> "TaskDescriptorRecorder":
        """Capture task properties fixed at the reset boundary."""
        world = core_env.world
        obstacle_count = len(world.obstacles)
        static_flags = np.asarray(
            getattr(world, "obstacle_is_static", np.zeros(obstacle_count, dtype=bool)),
            dtype=bool,
        )
        if static_flags.shape != (obstacle_count,):
            static_flags = np.zeros(obstacle_count, dtype=bool)

        base_x = float(world.base.centerx)
        base_y = float(core_env.area.height - world.base.centery)
        goal_distances = [
            float(np.hypot(float(goal.x) - base_x, float(goal.y) - base_y))
            for goal in core_env.goals
            if goal.x is not None and goal.y is not None
        ]
        goal_base_distance = min(goal_distances) if goal_distances else 0.0
        moving_obstacle_count = int((~static_flags).sum())
        return cls(
            moving_obstacle_count=moving_obstacle_count,
            static_obstacle_count=int(static_flags.sum()),
            goal_base_distance=goal_base_distance,
            initial_moving_speeds=np.full(
                moving_obstacle_count, np.nan, dtype=np.float32
            ),
        )

    def record_step(self, core_env) -> None:
        """Capture each moving obstacle's first realized nonzero speed once."""
        world = core_env.world
        speeds = np.asarray(
            getattr(world, "obstacle_move_speeds", ()),
            dtype=np.float32,
        )
        static_flags = np.asarray(
            getattr(world, "obstacle_is_static", ()),
            dtype=bool,
        )
        if speeds.ndim != 1 or speeds.shape != static_flags.shape:
            return
        moving_speeds = speeds[~static_flags]
        if moving_speeds.shape != self.initial_moving_speeds.shape:
            return
        unresolved = ~np.isfinite(self.initial_moving_speeds)
        newly_available = unresolved & (moving_speeds > 0.0)
        self.initial_moving_speeds[newly_available] = moving_speeds[newly_available]

    def finalize(self, core_env) -> dict[str, Any]:
        """Return normalized and raw descriptor values after the episode."""
        observed_speeds = self.initial_moving_speeds[
            np.isfinite(self.initial_moving_speeds)
        ]
        if observed_speeds.size:
            speeds = observed_speeds
            speed_sample_count = int(speeds.size)
        else:
            world = core_env.world
            fallback_speed = 0.5 * (
                float(world.obstacle_min_speed) + float(world.obstacle_max_speed)
            )
            speeds = np.asarray([fallback_speed], dtype=np.float32)
            speed_sample_count = 0
        raw = np.asarray(
            (
                self.moving_obstacle_count,
                self.static_obstacle_count,
                float(speeds.mean()),
                float(speeds.std()),
                float(speeds.max()),
                self.goal_base_distance,
            ),
            dtype=np.float32,
        )
        return {
            "normalized": normalize_realized_task_descriptor(raw),
            "raw": raw,
            "names": REALIZED_TASK_DESCRIPTOR_NAMES,
            "scales": REALIZED_TASK_DESCRIPTOR_SCALES.copy(),
            "speed_sample_count": speed_sample_count,
        }
