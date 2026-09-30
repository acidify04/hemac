"""Maintained Gymnasium-Robotics MaMuJoCo adapter and task wrappers."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from typing import Any

import numpy as np

from .tasks import AGENT_IDS, TaskSpec


SCENARIO = "HalfCheetah"
DEFAULT_ENVIRONMENT_VERSION = "HalfCheetah-v2"
SUPPORTED_ENVIRONMENT_VERSIONS = ("HalfCheetah-v2", "HalfCheetah-v5")
DEFAULT_AGENT_CONF = "6x1"
DEFAULT_AGENT_OBSK = 1


def add_environment_version_argument(parser: argparse.ArgumentParser) -> None:
    """Add the canonical v2 default and maintained v5 override."""
    parser.add_argument(
        "--environment-version",
        choices=SUPPORTED_ENVIRONMENT_VERSIONS,
        default=DEFAULT_ENVIRONMENT_VERSION,
    )


def recorded_environment_version(metadata: Mapping[str, Any]) -> str | None:
    """Read new metadata while recognizing pre-change v5 backend records."""
    version = metadata.get("environment_version")
    if version in SUPPORTED_ENVIRONMENT_VERSIONS:
        return str(version)
    backend = str(metadata.get("backend", ""))
    for candidate in SUPPORTED_ENVIRONMENT_VERSIONS:
        if backend.endswith(candidate):
            return candidate
    return None


def _import_mamujoco():
    try:
        from gymnasium_robotics import mamujoco_v1
    except ImportError as exc:
        raise ImportError(
            "MaMuJoCo requires the dedicated dependencies. Activate the mappo "
            "environment and run `pip install -r src/mamujoco/requirements.txt`."
        ) from exc
    return mamujoco_v1


def mask_disabled_action(
    actions: Mapping[str, np.ndarray], disabled_agent: str | None
) -> dict[str, np.ndarray]:
    """Return copied actions with a disabled joint torque forced to zero."""
    masked = {
        agent: np.asarray(action, dtype=np.float32).copy()
        for agent, action in actions.items()
    }
    if disabled_agent is not None:
        if disabled_agent not in masked:
            raise KeyError(f"Missing action for disabled agent {disabled_agent}")
        masked[disabled_agent].fill(0.0)
    return masked


class TaskedHalfCheetah:
    """Fixed-interface 6-agent HalfCheetah with task-specific interventions."""

    metadata = {"name": "mamujoco_halfcheetah_transfer"}

    def __init__(
        self,
        task: TaskSpec,
        *,
        agent_conf: str = DEFAULT_AGENT_CONF,
        agent_obsk: int = DEFAULT_AGENT_OBSK,
        environment_version: str = DEFAULT_ENVIRONMENT_VERSION,
        render_mode: str | None = None,
        max_cycles: int = 1000,
    ) -> None:
        if agent_conf != DEFAULT_AGENT_CONF:
            raise ValueError(
                "The transfer protocol requires agent_conf='6x1'; other values "
                "would invalidate the fixed six-actor architecture."
            )
        if int(agent_obsk) != DEFAULT_AGENT_OBSK:
            raise ValueError("The transfer protocol requires agent_obsk=1.")
        self.task = task
        self.agent_conf = agent_conf
        self.agent_obsk = int(agent_obsk)
        if environment_version not in SUPPORTED_ENVIRONMENT_VERSIONS:
            raise ValueError(
                f"Unsupported environment version {environment_version!r}; choose from "
                f"{SUPPORTED_ENVIRONMENT_VERSIONS}"
            )
        self.environment_version = environment_version
        self.max_cycles = int(max_cycles)
        mamujoco_v1 = _import_mamujoco()
        environment_options = {
            "agent_obsk": self.agent_obsk,
            "render_mode": render_mode,
        }
        if self.environment_version == "HalfCheetah-v2":
            import gymnasium

            environment_options["gym_env"] = gymnasium.make(
                self.environment_version,
                render_mode=render_mode,
                max_episode_steps=self.max_cycles,
            )
        else:
            environment_options["max_episode_steps"] = self.max_cycles
        self.env = mamujoco_v1.parallel_env(
            SCENARIO,
            self.agent_conf,
            **environment_options,
        )
        if tuple(self.env.possible_agents) != AGENT_IDS:
            raise RuntimeError(
                f"Expected agents {AGENT_IDS}, got {tuple(self.env.possible_agents)}"
            )
        self.possible_agents = list(self.env.possible_agents)
        self.agents = list(self.env.agents)
        self.last_executed_actions: dict[str, np.ndarray] = {}
        self._physics_env = self.env.single_agent_env.unwrapped
        self._nominal_body_mass = self._physics_env.model.body_mass.copy()
        self._nominal_geom_friction = self._physics_env.model.geom_friction.copy()
        self._nominal_actuator_gear = self._physics_env.model.actuator_gear.copy()
        self._apply_task_dynamics()

    @property
    def observation_spaces(self):
        return self.env.observation_spaces

    @property
    def action_spaces(self):
        return self.env.action_spaces

    def observation_space(self, agent: str):
        return self.env.observation_space(agent)

    def action_space(self, agent: str):
        return self.env.action_space(agent)

    def state(self) -> np.ndarray:
        return np.asarray(self.env.state(), dtype=np.float32)

    @property
    def x_position(self) -> float:
        """Current root x coordinate used for exact episode displacement."""
        return float(self._physics_env.data.qpos[0])

    def _apply_task_dynamics(self) -> None:
        model = self._physics_env.model
        model.body_mass[:] = self._nominal_body_mass * self.task.mass_scale
        model.geom_friction[:] = (
            self._nominal_geom_friction * self.task.friction_scale
        )
        model.actuator_gear[:] = (
            self._nominal_actuator_gear * self.task.actuator_scale
        )

    def reset(self, seed: int | None = None, options: dict | None = None):
        self._apply_task_dynamics()
        observations, infos = self.env.reset(seed=seed, options=options)
        self.agents = list(self.env.agents)
        self.last_executed_actions = {
            agent: np.zeros(self.action_space(agent).shape, dtype=np.float32)
            for agent in self.agents
        }
        return observations, infos

    def step(self, actions: Mapping[str, np.ndarray]):
        previous_x = self.x_position
        executed = mask_disabled_action(actions, self.task.disabled_agent)
        self.last_executed_actions = executed
        result = self.env.step(executed)
        self.agents = list(self.env.agents)
        observations, rewards, terminations, truncations, raw_infos = result
        current_x = self.x_position
        infos = {agent: dict(info) for agent, info in raw_infos.items()}
        for agent, info in infos.items():
            forward_velocity = (current_x - previous_x) / float(self._physics_env.dt)
            info.setdefault("x_position", current_x)
            info.setdefault("x_velocity", forward_velocity)
            info.setdefault("reward_forward", info.get("reward_run", forward_velocity))
            info["task_suite"] = self.task.suite
            info["task_name"] = self.task.name
            info["disabled"] = agent == self.task.disabled_agent
            info["actuator_strength"] = self.task.actuator_scale
            info["environment_version"] = self.environment_version
        return observations, rewards, terminations, truncations, infos

    def render(self):
        return self.env.render()

    def close(self) -> None:
        self.env.close()


def make_env(
    task: TaskSpec,
    *,
    agent_conf: str = DEFAULT_AGENT_CONF,
    agent_obsk: int = DEFAULT_AGENT_OBSK,
    environment_version: str = DEFAULT_ENVIRONMENT_VERSION,
    render_mode: str | None = None,
    max_cycles: int = 1000,
) -> TaskedHalfCheetah:
    return TaskedHalfCheetah(
        task,
        agent_conf=agent_conf,
        agent_obsk=agent_obsk,
        environment_version=environment_version,
        render_mode=render_mode,
        max_cycles=max_cycles,
    )


def infer_space_dimensions(env: Any) -> tuple[dict[str, int], dict[str, int], int]:
    """Return agent observation/action sizes and global-state size."""
    observation_dims = {
        agent: int(np.prod(env.observation_space(agent).shape))
        for agent in env.possible_agents
    }
    action_dims = {
        agent: int(np.prod(env.action_space(agent).shape))
        for agent in env.possible_agents
    }
    state_dim = int(np.asarray(env.state()).size)
    return observation_dims, action_dims, state_dim
