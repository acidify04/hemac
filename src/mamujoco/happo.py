"""Independent-actor HAPPO with sequential policy updates."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .metrics import EpisodeMetrics
from .models import (
    CentralCritic,
    IndependentActors,
    squashed_log_prob,
)


@dataclass
class EpisodeBatch:
    observations: dict[str, torch.Tensor]
    raw_actions: dict[str, torch.Tensor]
    actions: dict[str, torch.Tensor]
    old_log_probs: dict[str, torch.Tensor]
    active_masks: dict[str, torch.Tensor]
    states: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    metrics: dict[str, float | int]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


def _tensor_observations(observations, device):
    return {
        agent: torch.as_tensor(value, dtype=torch.float32, device=device).reshape(
            1, -1
        )
        for agent, value in observations.items()
    }


def _episode_done(terminations, truncations) -> bool:
    keys = set(terminations) | set(truncations)
    return bool(keys) and all(
        bool(terminations.get(agent, False) or truncations.get(agent, False))
        for agent in keys
    )


@torch.no_grad()
def collect_episode(
    env,
    actors: IndependentActors,
    critic: CentralCritic,
    *,
    seed: int,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    deterministic: bool = False,
    device: torch.device,
) -> EpisodeBatch:
    """Collect one complete parallel-env episode and calculate team GAE."""
    observations, _ = env.reset(seed=seed)
    agent_ids = actors.agent_ids
    obs_rows = {agent: [] for agent in agent_ids}
    raw_action_rows = {agent: [] for agent in agent_ids}
    action_rows = {agent: [] for agent in agent_ids}
    log_prob_rows = {agent: [] for agent in agent_ids}
    active_rows = {agent: [] for agent in agent_ids}
    state_rows = []
    reward_rows = []
    value_rows = []
    metrics = EpisodeMetrics()
    if hasattr(env, "x_position"):
        metrics.begin(env.x_position)

    while observations:
        tensor_obs = _tensor_observations(observations, device)
        state = torch.as_tensor(env.state(), dtype=torch.float32, device=device)
        actions, raw_actions, log_probs = actors.act(
            tensor_obs, deterministic=deterministic
        )
        numpy_actions = {
            agent: actions[agent].squeeze(0).cpu().numpy().astype(np.float32)
            for agent in agent_ids
        }
        next_observations, rewards, terminations, truncations, infos = env.step(
            numpy_actions
        )
        executed_actions = env.last_executed_actions
        team_reward = float(rewards[agent_ids[0]])

        for agent in agent_ids:
            obs_rows[agent].append(tensor_obs[agent].squeeze(0).cpu())
            raw_action_rows[agent].append(raw_actions[agent].squeeze(0).cpu())
            action_rows[agent].append(
                torch.as_tensor(executed_actions[agent], dtype=torch.float32)
            )
            log_prob_rows[agent].append(log_probs[agent].squeeze(0).cpu())
            active_rows[agent].append(agent != env.task.disabled_agent)
        state_rows.append(state.cpu())
        reward_rows.append(team_reward)
        value_rows.append(float(critic(state.unsqueeze(0)).item()))
        metrics.update(rewards, infos)
        observations = next_observations
        if _episode_done(terminations, truncations):
            break

    rewards_tensor = torch.tensor(reward_rows, dtype=torch.float32)
    values = torch.tensor(value_rows + [0.0], dtype=torch.float32)
    advantages = torch.zeros_like(rewards_tensor)
    gae = torch.tensor(0.0)
    for index in range(len(reward_rows) - 1, -1, -1):
        delta = rewards_tensor[index] + gamma * values[index + 1] - values[index]
        gae = delta + gamma * gae_lambda * gae
        advantages[index] = gae
    returns = advantages + values[:-1]

    return EpisodeBatch(
        observations={agent: torch.stack(rows) for agent, rows in obs_rows.items()},
        raw_actions={
            agent: torch.stack(rows) for agent, rows in raw_action_rows.items()
        },
        actions={agent: torch.stack(rows) for agent, rows in action_rows.items()},
        old_log_probs={
            agent: torch.stack(rows) for agent, rows in log_prob_rows.items()
        },
        active_masks={
            agent: torch.tensor(rows, dtype=torch.bool)
            for agent, rows in active_rows.items()
        },
        states=torch.stack(state_rows),
        rewards=rewards_tensor,
        advantages=advantages,
        returns=returns,
        metrics=metrics.finalize(),
    )


def concatenate_episodes(episodes: list[EpisodeBatch]) -> EpisodeBatch:
    if not episodes:
        raise ValueError("At least one episode is required.")
    agent_ids = tuple(episodes[0].observations)

    def concatenate(name, agent):
        return torch.cat(
            [getattr(episode, name)[agent] for episode in episodes], dim=0
        )
    return EpisodeBatch(
        observations={a: concatenate("observations", a) for a in agent_ids},
        raw_actions={a: concatenate("raw_actions", a) for a in agent_ids},
        actions={a: concatenate("actions", a) for a in agent_ids},
        old_log_probs={a: concatenate("old_log_probs", a) for a in agent_ids},
        active_masks={a: concatenate("active_masks", a) for a in agent_ids},
        states=torch.cat([episode.states for episode in episodes]),
        rewards=torch.cat([episode.rewards for episode in episodes]),
        advantages=torch.cat([episode.advantages for episode in episodes]),
        returns=torch.cat([episode.returns for episode in episodes]),
        metrics={
            "episode_return": float(
                np.mean([episode.metrics["episode_return"] for episode in episodes])
            ),
            "forward_distance": float(
                np.mean([episode.metrics["forward_distance"] for episode in episodes])
            ),
            "forward_reward": float(
                np.mean([episode.metrics["forward_reward"] for episode in episodes])
            ),
            "forward_velocity": float(
                np.mean([episode.metrics["forward_velocity"] for episode in episodes])
            ),
            "control_cost": float(
                np.mean([episode.metrics["control_cost"] for episode in episodes])
            ),
            "episode_length": float(
                np.mean([episode.metrics["episode_length"] for episode in episodes])
            ),
        },
    )


class HAPPOTrainer:
    """Sequentially update independent actors, then update one central critic."""

    def __init__(
        self,
        actors: IndependentActors,
        critic: CentralCritic,
        *,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        clip_ratio: float = 0.2,
        entropy_coeff: float = 0.01,
        value_coeff: float = 0.5,
        max_grad_norm: float = 1.0,
        ppo_epochs: int = 5,
        minibatch_size: int = 1024,
        device: torch.device,
    ) -> None:
        self.actors = actors
        self.critic = critic
        self.clip_ratio = float(clip_ratio)
        self.entropy_coeff = float(entropy_coeff)
        self.value_coeff = float(value_coeff)
        self.max_grad_norm = float(max_grad_norm)
        self.ppo_epochs = int(ppo_epochs)
        self.minibatch_size = int(minibatch_size)
        self.device = device
        self.actor_optimizers = {
            agent: torch.optim.Adam(self.actors.actors[agent].parameters(), lr=actor_lr)
            for agent in self.actors.agent_ids
        }
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr
        )

    def update(self, batch: EpisodeBatch) -> dict[str, float]:
        count = batch.states.shape[0]
        advantages = batch.advantages.to(self.device)
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )
        factor = torch.ones(count, device=self.device)
        metrics: dict[str, float] = {}

        for agent in self.actors.agent_ids:
            actor = self.actors.actors[agent]
            observations = batch.observations[agent].to(self.device)
            raw_actions = batch.raw_actions[agent].to(self.device)
            old_log_probs = batch.old_log_probs[agent].to(self.device)
            active = batch.active_masks[agent].to(self.device)
            if not active.any():
                metrics[f"actor_loss/{agent}"] = 0.0
                continue
            optimizer = self.actor_optimizers[agent]
            final_loss = 0.0
            for _ in range(self.ppo_epochs):
                permutation = torch.randperm(count, device=self.device)
                for start in range(0, count, self.minibatch_size):
                    indices = permutation[start : start + self.minibatch_size]
                    mask = active[indices]
                    if not mask.any():
                        continue
                    distribution = actor.distribution(observations[indices])
                    new_log_prob = squashed_log_prob(
                        distribution, raw_actions[indices]
                    )
                    ratio = (new_log_prob - old_log_probs[indices]).exp()
                    weighted_advantage = factor[indices] * advantages[indices]
                    objective = ratio * weighted_advantage
                    clipped = ratio.clamp(
                        1.0 - self.clip_ratio, 1.0 + self.clip_ratio
                    ) * weighted_advantage
                    policy_loss = -torch.minimum(objective, clipped)[mask].mean()
                    entropy = distribution.entropy().sum(dim=-1)[mask].mean()
                    loss = policy_loss - self.entropy_coeff * entropy
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad_norm)
                    optimizer.step()
                    final_loss = float(policy_loss.detach().cpu())
            with torch.no_grad():
                updated_log_prob = squashed_log_prob(
                    actor.distribution(observations), raw_actions
                )
                agent_ratio = (updated_log_prob - old_log_probs).exp()
                factor = torch.where(active, factor * agent_ratio, factor).clamp(
                    0.05, 20.0
                )
            metrics[f"actor_loss/{agent}"] = final_loss

        states = batch.states.to(self.device)
        returns = batch.returns.to(self.device)
        critic_loss_value = 0.0
        for _ in range(self.ppo_epochs):
            permutation = torch.randperm(count, device=self.device)
            for start in range(0, count, self.minibatch_size):
                indices = permutation[start : start + self.minibatch_size]
                predictions = self.critic(states[indices])
                critic_loss = self.value_coeff * torch.mean(
                    (predictions - returns[indices]) ** 2
                )
                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                critic_loss_value = float(critic_loss.detach().cpu())
        metrics["critic_loss"] = critic_loss_value
        metrics["importance_factor_mean"] = float(factor.mean().detach().cpu())
        return metrics
