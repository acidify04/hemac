"""Episode metrics shared by collection and zero-shot evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass


@dataclass
class EpisodeMetrics:
    episode_return: float = 0.0
    forward_distance: float = 0.0
    forward_reward: float = 0.0
    forward_velocity: float = 0.0
    control_cost: float = 0.0
    episode_length: int = 0
    _initial_x: float | None = None
    _final_x: float | None = None
    _forward_velocity_sum: float = 0.0

    def begin(self, x_position: float) -> None:
        """Record the position immediately after reset, before the first action."""
        self._initial_x = float(x_position)
        self._final_x = float(x_position)

    def update(
        self,
        rewards: Mapping[str, float],
        infos: Mapping[str, Mapping[str, float]],
    ) -> None:
        if not rewards:
            return
        # MaMuJoCo broadcasts the same team reward/info to every agent.
        first_agent = next(iter(rewards))
        info = infos.get(first_agent, {})
        self.episode_return += float(rewards[first_agent])
        self.forward_reward += float(info.get("reward_forward", 0.0))
        self._forward_velocity_sum += float(info.get("x_velocity", 0.0))
        # Gymnasium exposes reward_ctrl as a negative reward component. Report
        # the diagnostic as a conventional non-negative control cost.
        self.control_cost += -float(info.get("reward_ctrl", 0.0))
        x_position = info.get("x_position")
        if x_position is not None:
            x_position = float(x_position)
            if self._initial_x is None:
                self._initial_x = x_position
            self._final_x = x_position
        self.episode_length += 1

    def finalize(self) -> dict[str, float | int]:
        if self._initial_x is not None and self._final_x is not None:
            self.forward_distance = self._final_x - self._initial_x
        if self.episode_length:
            self.forward_velocity = self._forward_velocity_sum / self.episode_length
        result = asdict(self)
        result.pop("_initial_x")
        result.pop("_final_x")
        result.pop("_forward_velocity_sum")
        return result
