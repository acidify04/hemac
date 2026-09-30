"""Neural networks shared by HAPPO and offline MaMuJoCo learners."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as functional
from torch import nn
from torch.distributions import Normal


LOG_STD_MIN = -5.0
LOG_STD_MAX = 1.0


def mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, output_dim),
    )


def squashed_log_prob(distribution: Normal, raw_action: torch.Tensor) -> torch.Tensor:
    """Log probability of tanh(raw_action), summed over action dimensions."""
    correction = 2.0 * (
        math.log(2.0) - raw_action - functional.softplus(-2.0 * raw_action)
    )
    return (distribution.log_prob(raw_action) - correction).sum(dim=-1)


class GaussianActor(nn.Module):
    """One independent continuous actor for one HalfCheetah joint."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int = 1,
        hidden_dim: int = 256,
        log_std_init: float = -0.5,
    ) -> None:
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.network = mlp(self.observation_dim, hidden_dim, self.action_dim)
        self.log_std = nn.Parameter(
            torch.full((self.action_dim,), float(log_std_init))
        )

    def distribution(self, observation: torch.Tensor) -> Normal:
        mean = self.network(observation)
        std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp()
        return Normal(mean, std.expand_as(mean))

    def sample(
        self, observation: torch.Tensor, *, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observation)
        raw_action = distribution.mean if deterministic else distribution.rsample()
        action = torch.tanh(raw_action)
        log_prob = squashed_log_prob(distribution, raw_action)
        return action, raw_action, log_prob


class IndependentActors(nn.Module):
    """Six non-sharing actors with agent-specific observation dimensions."""

    def __init__(
        self,
        observation_dims: Mapping[str, int],
        action_dims: Mapping[str, int],
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if tuple(observation_dims) != tuple(action_dims):
            raise ValueError("Observation and action agent order must match.")
        self.agent_ids = tuple(observation_dims)
        self.observation_dims = dict(observation_dims)
        self.action_dims = dict(action_dims)
        self.actors = nn.ModuleDict(
            {
                agent: GaussianActor(
                    observation_dims[agent], action_dims[agent], hidden_dim
                )
                for agent in self.agent_ids
            }
        )

    def act(
        self,
        observations: Mapping[str, torch.Tensor],
        *,
        deterministic: bool = False,
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        actions = {}
        raw_actions = {}
        log_probs = {}
        for agent in self.agent_ids:
            action, raw_action, log_prob = self.actors[agent].sample(
                observations[agent], deterministic=deterministic
            )
            actions[agent] = action
            raw_actions[agent] = raw_action
            log_probs[agent] = log_prob
        return actions, raw_actions, log_probs


class CentralCritic(nn.Module):
    """Centralized team value function over the 17-D global state."""

    def __init__(self, state_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.network = mlp(self.state_dim, hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state).squeeze(-1)


def checkpoint_payload(
    actors: IndependentActors,
    critic: CentralCritic,
    *,
    metadata: dict,
) -> dict:
    return {
        "format_version": 1,
        "algorithm": "happo",
        "observation_dims": actors.observation_dims,
        "action_dims": actors.action_dims,
        "state_dim": critic.state_dim,
        "actors": actors.state_dict(),
        "critic": critic.state_dict(),
        "metadata": metadata,
    }


def load_happo_checkpoint(
    path,
    *,
    device: torch.device,
) -> tuple[IndependentActors, CentralCritic, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("algorithm") != "happo":
        raise ValueError(f"Not a HAPPO checkpoint: {path}")
    actors = IndependentActors(
        payload["observation_dims"], payload["action_dims"]
    ).to(device)
    critic = CentralCritic(payload["state_dim"]).to(device)
    actors.load_state_dict(payload["actors"])
    critic.load_state_dict(payload["critic"])
    actors.eval()
    critic.eval()
    return actors, critic, payload
