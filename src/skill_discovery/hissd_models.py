"""HiSSD modules adapted from the official implementation for HeMAC drones.

The original implementation uses decomposed SMAC entity observations and
discrete actions. This adaptation preserves its planner/value/discriminator/
decoder separation while replacing entity encoders with the validated HeMAC
CNN encoder and the action classifier with a continuous action decoder.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .models import (
    DRONE_HIDDEN_SIZES,
    GLOBAL_MAP_ENCODER_CHANNELS,
    DroneBehaviorCloningPolicy,
    DroneObservationEncoder,
    activation_module,
    build_map_encoder,
)


CENTRAL_MAP_ENCODER_CHANNELS = GLOBAL_MAP_ENCODER_CHANNELS


def _single_layer_transformer(hidden_dim: int, heads: int) -> nn.TransformerEncoder:
    """Build the one-layer transformer used throughout official HiSSD."""
    layer = nn.TransformerEncoderLayer(
        d_model=hidden_dim,
        nhead=heads,
        dim_feedforward=hidden_dim * 2,
        dropout=0.0,
        activation="relu",
        batch_first=True,
        norm_first=False,
    )
    return nn.TransformerEncoder(
        layer,
        num_layers=1,
        enable_nested_tensor=False,
    )


class HistoryTransformerEncoder(nn.Module):
    """Process one observation token together with recurrent history."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, heads: int = 1) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        self.transformer = _single_layer_transformer(self.hidden_dim, self.heads)
        self.output_norm = nn.LayerNorm(self.hidden_dim)

    def initial_state(
        self,
        batch_size: int,
        agent_count: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return the recurrent history used at the start of an episode."""
        return torch.zeros(
            int(batch_size),
            int(agent_count),
            self.hidden_dim,
            device=device,
            dtype=dtype,
        )

    def forward_step(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor,
        history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode one `[B,A,D]` step and return output plus next history."""
        if features.ndim != 3:
            raise ValueError(f"Expected [B,A,D] features, got {features.shape}.")
        if valid_mask.shape != features.shape[:2]:
            raise ValueError(
                f"Mask {valid_mask.shape} does not match {features.shape[:2]}."
            )
        expected_history = (*features.shape[:2], self.hidden_dim)
        if history.shape != expected_history:
            raise ValueError(
                f"History {history.shape} does not match {expected_history}."
            )

        batch_size, agent_count, _ = features.shape
        observation_token = self.input_projection(features).reshape(
            batch_size * agent_count,
            self.hidden_dim,
        )
        flat_history = history.reshape(batch_size * agent_count, self.hidden_dim)
        tokens = torch.stack((observation_token, flat_history), dim=1)
        encoded = self.transformer(tokens)
        current = self.output_norm(encoded[:, 0])
        next_history = encoded[:, 1]
        active = valid_mask.bool().reshape(-1, 1)
        current = torch.where(active, current, torch.zeros_like(current))
        next_history = torch.where(active, next_history, flat_history)
        output_shape = (batch_size, agent_count, self.hidden_dim)
        return current.reshape(output_shape), next_history.reshape(output_shape)

    def forward(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode `[B,T,A,D]` features into `[B,T,A,H]` hidden states."""
        if features.ndim != 4:
            raise ValueError(f"Expected [B,T,A,D] features, got {features.shape}.")
        if valid_mask.shape != features.shape[:3]:
            raise ValueError(
                f"Mask {valid_mask.shape} does not match {features.shape[:3]}."
            )
        batch_size, sequence_length, agent_count, _ = features.shape
        history = self.initial_state(
            batch_size,
            agent_count,
            device=features.device,
            dtype=features.dtype,
        )
        outputs = []
        for time_index in range(sequence_length):
            current, history = self.forward_step(
                features[:, time_index],
                valid_mask[:, time_index],
                history,
            )
            outputs.append(current)
        return torch.stack(outputs, dim=1)


class CommonSkillEncoder(nn.Module):
    """Official high-level planner analogue that infers common skills."""

    def __init__(
        self,
        observation_dim: int,
        hidden_dim: int = 64,
        skill_dim: int = 64,
        heads: int = 1,
    ) -> None:
        super().__init__()
        self.backbone = HistoryTransformerEncoder(observation_dim, hidden_dim, heads)
        self.skill_projection = nn.Sequential(
            nn.Linear(hidden_dim, skill_dim),
            nn.LayerNorm(skill_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        observation_features: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.backbone(observation_features, valid_mask)
        return self.skill_projection(hidden)

    def forward_step(
        self,
        observation_features: torch.Tensor,
        valid_mask: torch.Tensor,
        history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Infer common skills for one online environment step."""
        hidden, next_history = self.backbone.forward_step(
            observation_features,
            valid_mask,
            history,
        )
        return self.skill_projection(hidden), next_history


class TaskSpecificSkillEncoder(nn.Module):
    """Infer action-conditioned and shared causal task representations."""

    def __init__(
        self,
        observation_dim: int,
        hidden_dim: int = 64,
        skill_dim: int = 64,
        heads: int = 1,
        contrastive_from_action_skill: bool = False,
        task_context_pooling: bool = False,
        dropout: float = 0.0,
        use_feature_deltas: bool = False,
        normalize_context: bool = True,
        use_running_statistics: bool = False,
        direct_task_summary: bool = False,
    ) -> None:
        super().__init__()
        self.contrastive_from_action_skill = bool(contrastive_from_action_skill)
        self.task_context_pooling = bool(task_context_pooling)
        self.use_feature_deltas = bool(use_feature_deltas)
        self.normalize_context = bool(normalize_context)
        self.use_running_statistics = bool(use_running_statistics)
        self.direct_task_summary = bool(direct_task_summary)
        self.observation_dim = int(observation_dim)
        self.skill_dim = int(skill_dim)
        self.dropout_rate = float(dropout)
        self.context_dropout = nn.Dropout(self.dropout_rate)
        self.action_dropout = nn.Dropout(self.dropout_rate)
        self.contrastive_dropout = nn.Dropout(self.dropout_rate)
        if self.use_running_statistics and not self.use_feature_deltas:
            raise ValueError("Running task statistics require feature deltas.")
        input_multiplier = 5 if self.use_running_statistics else (
            2 if self.use_feature_deltas else 1
        )
        backbone_input_dim = self.observation_dim * input_multiplier
        self.backbone = HistoryTransformerEncoder(
            backbone_input_dim, hidden_dim, heads
        )
        self.summary_projection = None
        if self.direct_task_summary:
            self.summary_projection = nn.Sequential(
                nn.Linear(backbone_input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
            )
        self.action_projection = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, skill_dim),
            nn.LayerNorm(skill_dim),
            nn.Tanh(),
        )
        self.contrastive_projection = nn.Sequential(
            nn.Linear(
                skill_dim
                if self.contrastive_from_action_skill or self.task_context_pooling
                else hidden_dim,
                128,
            ),
            nn.ReLU(),
            nn.Linear(128, skill_dim),
        )
        if self.task_context_pooling:
            # The delta exposes obstacle motion and other task dynamics that
            # cannot be inferred reliably from a single map snapshot.
            self.context_gru = nn.GRUCell(skill_dim * 2, skill_dim)
            self.context_adapter = nn.Linear(skill_dim, skill_dim)
            nn.init.zeros_(self.context_adapter.weight)
            nn.init.zeros_(self.context_adapter.bias)

    def initial_context_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return the causal task context used at the start of a window."""
        return torch.zeros(
            int(batch_size),
            self.skill_dim,
            device=device,
            dtype=dtype,
        )

    def initial_observation_state(
        self,
        batch_size: int,
        agent_count: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return the previous encoded observation used for online deltas."""
        return torch.zeros(
            int(batch_size),
            int(agent_count),
            self.observation_dim,
            device=device,
            dtype=dtype,
        )

    def _with_feature_deltas(
        self,
        observation_features: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Expose causal map dynamics before recurrent task compression."""
        if not self.use_feature_deltas:
            return observation_features
        previous = torch.cat(
            (observation_features[:, :1], observation_features[:, :-1]), dim=1
        )
        previous_valid = torch.cat(
            (
                torch.zeros_like(valid_mask[:, :1], dtype=torch.bool),
                valid_mask[:, :-1].bool(),
            ),
            dim=1,
        )
        delta_valid = valid_mask.bool() & previous_valid
        delta = torch.where(
            delta_valid.unsqueeze(-1),
            observation_features - previous,
            torch.zeros_like(observation_features),
        )
        if not self.use_running_statistics:
            return torch.cat((observation_features, delta), dim=-1)

        numeric_mask = valid_mask.bool().unsqueeze(-1).to(
            observation_features.dtype
        )
        count = numeric_mask.cumsum(dim=1).clamp_min(1.0)
        running_mean = (
            observation_features * numeric_mask
        ).cumsum(dim=1) / count
        running_square_mean = (
            observation_features.square() * numeric_mask
        ).cumsum(dim=1) / count
        running_std = (
            running_square_mean - running_mean.square()
        ).clamp_min(0.0).add(1e-6).sqrt()
        numeric_delta_mask = delta_valid.unsqueeze(-1).to(
            observation_features.dtype
        )
        delta_count = numeric_delta_mask.cumsum(dim=1).clamp_min(1.0)
        running_abs_delta = (
            delta.abs() * numeric_delta_mask
        ).cumsum(dim=1) / delta_count
        return torch.cat(
            (
                observation_features,
                delta,
                running_mean,
                running_std,
                running_abs_delta,
            ),
            dim=-1,
        )

    def _contextualize_sequence(
        self,
        action_skill: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Aggregate all drones causally and broadcast context at each step."""
        batch_size, sequence_length, agent_count, skill_dim = action_skill.shape
        context = self.initial_context_state(
            batch_size,
            device=action_skill.device,
            dtype=action_skill.dtype,
        )
        previous_pooled = torch.zeros_like(context)
        contextual_skills = []
        contrastive_skills = []
        for time_index in range(sequence_length):
            active_agents = valid_mask[:, time_index].bool()
            numeric_mask = active_agents.unsqueeze(-1).to(action_skill.dtype)
            denominator = numeric_mask.sum(dim=1).clamp_min(1.0)
            pooled = (
                action_skill[:, time_index] * numeric_mask
            ).sum(dim=1) / denominator
            context_input = torch.cat((pooled, pooled - previous_pooled), dim=-1)
            if self.direct_task_summary:
                candidate_context = pooled
            else:
                candidate_context = self.context_gru(
                    self.context_dropout(context_input), context
                )
            active_step = active_agents.any(dim=1, keepdim=True)
            context = torch.where(active_step, candidate_context, context)
            previous_pooled = torch.where(active_step, pooled, previous_pooled)
            contextual = action_skill[:, time_index] + self.context_adapter(
                context
            ).unsqueeze(1)
            contextual = torch.where(
                active_agents.unsqueeze(-1),
                contextual,
                torch.zeros_like(contextual),
            )
            contrastive = self.contrastive_projection(
                self.contrastive_dropout(context)
            )
            if self.normalize_context:
                contrastive = F.normalize(contrastive, dim=-1)
            contrastive = contrastive.unsqueeze(1).expand(
                batch_size, agent_count, skill_dim
            )
            contrastive = torch.where(
                active_agents.unsqueeze(-1),
                contrastive,
                torch.zeros_like(contrastive),
            )
            contextual_skills.append(contextual)
            contrastive_skills.append(contrastive)
        return (
            torch.stack(contextual_skills, dim=1),
            torch.stack(contrastive_skills, dim=1),
        )

    def forward(
        self,
        observation_features: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        task_features = self._with_feature_deltas(
            observation_features, valid_mask
        )
        if self.summary_projection is None:
            hidden = self.backbone(task_features, valid_mask)
        else:
            hidden = self.summary_projection(task_features)
            hidden = torch.where(
                valid_mask.bool().unsqueeze(-1),
                hidden,
                torch.zeros_like(hidden),
            )
        action_skill = self.action_dropout(self.action_projection(hidden))
        if self.task_context_pooling:
            return self._contextualize_sequence(action_skill, valid_mask)
        contrastive_input = (
            action_skill if self.contrastive_from_action_skill else hidden
        )
        contrastive_skill = F.normalize(
            self.contrastive_projection(
                self.contrastive_dropout(contrastive_input)
            ),
            dim=-1,
        )
        return action_skill, contrastive_skill

    def forward_step(
        self,
        observation_features: torch.Tensor,
        valid_mask: torch.Tensor,
        history: torch.Tensor,
        context_state: torch.Tensor,
        previous_pooled: torch.Tensor,
        previous_observation: torch.Tensor,
        previous_observation_valid: torch.Tensor,
        running_sum: torch.Tensor,
        running_square_sum: torch.Tensor,
        running_abs_delta_sum: torch.Tensor,
        running_count: torch.Tensor,
        running_delta_count: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Infer task-specific skills for one online environment step."""
        if self.use_feature_deltas:
            delta_valid = valid_mask.bool() & previous_observation_valid.bool()
            feature_delta = torch.where(
                delta_valid.unsqueeze(-1),
                observation_features - previous_observation,
                torch.zeros_like(observation_features),
            )
            task_features = torch.cat(
                (observation_features, feature_delta), dim=-1
            )
        else:
            task_features = observation_features
            feature_delta = torch.zeros_like(observation_features)
            delta_valid = torch.zeros_like(valid_mask, dtype=torch.bool)

        numeric_mask = valid_mask.bool().unsqueeze(-1).to(
            observation_features.dtype
        )
        next_running_sum = running_sum + observation_features * numeric_mask
        next_running_square_sum = (
            running_square_sum + observation_features.square() * numeric_mask
        )
        next_running_count = running_count + numeric_mask
        safe_count = next_running_count.clamp_min(1.0)
        running_mean = next_running_sum / safe_count
        running_variance = (
            next_running_square_sum / safe_count - running_mean.square()
        ).clamp_min(0.0)
        numeric_delta_mask = delta_valid.unsqueeze(-1).to(
            observation_features.dtype
        )
        next_running_abs_delta_sum = (
            running_abs_delta_sum + feature_delta.abs() * numeric_delta_mask
        )
        next_running_delta_count = running_delta_count + numeric_delta_mask
        if self.use_running_statistics:
            task_features = torch.cat(
                (
                    observation_features,
                    feature_delta,
                    running_mean,
                    running_variance.add(1e-6).sqrt(),
                    next_running_abs_delta_sum
                    / next_running_delta_count.clamp_min(1.0),
                ),
                dim=-1,
            )
        if self.summary_projection is None:
            hidden, next_history = self.backbone.forward_step(
                task_features,
                valid_mask,
                history,
            )
        else:
            hidden = self.summary_projection(task_features)
            hidden = torch.where(
                valid_mask.bool().unsqueeze(-1),
                hidden,
                torch.zeros_like(hidden),
            )
            next_history = history
        next_observation = torch.where(
            valid_mask.bool().unsqueeze(-1),
            observation_features,
            previous_observation,
        )
        next_observation_valid = (
            previous_observation_valid.bool() | valid_mask.bool()
        )
        action_skill = self.action_dropout(self.action_projection(hidden))
        if self.task_context_pooling:
            numeric_mask = valid_mask.bool().unsqueeze(-1).to(action_skill.dtype)
            denominator = numeric_mask.sum(dim=1).clamp_min(1.0)
            pooled = (action_skill * numeric_mask).sum(dim=1) / denominator
            context_input = torch.cat(
                (pooled, pooled - previous_pooled), dim=-1
            )
            if self.direct_task_summary:
                candidate_context = pooled
            else:
                candidate_context = self.context_gru(
                    self.context_dropout(context_input), context_state
                )
            active_step = valid_mask.bool().any(dim=1, keepdim=True)
            next_context = torch.where(
                active_step, candidate_context, context_state
            )
            next_pooled = torch.where(active_step, pooled, previous_pooled)
            contextual_skill = action_skill + self.context_adapter(
                next_context
            ).unsqueeze(1)
            contextual_skill = torch.where(
                valid_mask.bool().unsqueeze(-1),
                contextual_skill,
                torch.zeros_like(contextual_skill),
            )
            contrastive_skill = self.contrastive_projection(
                self.contrastive_dropout(next_context)
            )
            if self.normalize_context:
                contrastive_skill = F.normalize(contrastive_skill, dim=-1)
            contrastive_skill = contrastive_skill.unsqueeze(1).expand_as(
                contextual_skill
            )
            contrastive_skill = torch.where(
                valid_mask.bool().unsqueeze(-1),
                contrastive_skill,
                torch.zeros_like(contrastive_skill),
            )
            return (
                contextual_skill,
                contrastive_skill,
                next_history,
                next_context,
                next_pooled,
                next_observation,
                next_observation_valid,
                next_running_sum,
                next_running_square_sum,
                next_running_abs_delta_sum,
                next_running_count,
                next_running_delta_count,
            )
        contrastive_input = (
            action_skill if self.contrastive_from_action_skill else hidden
        )
        contrastive_skill = F.normalize(
            self.contrastive_projection(
                self.contrastive_dropout(contrastive_input)
            ),
            dim=-1,
        )
        return (
            action_skill,
            contrastive_skill,
            next_history,
            context_state,
            previous_pooled,
            next_observation,
            next_observation_valid,
            next_running_sum,
            next_running_square_sum,
            next_running_abs_delta_sum,
            next_running_count,
            next_running_delta_count,
        )


class TaskObservationEncoder(nn.Module):
    """Preserve map statistics that the action-oriented BC encoder may discard."""

    def __init__(
        self,
        global_map_channels: int,
        local_map_channels: int,
        *,
        global_map_size: tuple[int, int],
        local_map_size: tuple[int, int],
        action_history_shape: tuple[int, int],
        hidden_sizes: Sequence[int],
        activation: str,
    ) -> None:
        super().__init__()
        self.base_encoder = DroneObservationEncoder(
            global_map_channels,
            local_map_channels,
            global_map_size=global_map_size,
            local_map_size=local_map_size,
            action_history_shape=action_history_shape,
            hidden_sizes=hidden_sizes,
            activation=activation,
        )
        self.global_map_channels = int(global_map_channels)
        self.local_map_channels = int(local_map_channels)
        self.output_dim = self.base_encoder.output_dim + 4 * (
            self.global_map_channels + self.local_map_channels
        )
        self.register_buffer(
            "global_x",
            torch.linspace(-1.0, 1.0, global_map_size[1]).view(1, 1, 1, -1),
            persistent=False,
        )
        self.register_buffer(
            "global_y",
            torch.linspace(-1.0, 1.0, global_map_size[0]).view(1, 1, -1, 1),
            persistent=False,
        )
        self.register_buffer(
            "local_x",
            torch.linspace(-1.0, 1.0, local_map_size[1]).view(1, 1, 1, -1),
            persistent=False,
        )
        self.register_buffer(
            "local_y",
            torch.linspace(-1.0, 1.0, local_map_size[0]).view(1, 1, -1, 1),
            persistent=False,
        )

    @staticmethod
    def _statistics(
        map_tensor: torch.Tensor,
        x_coordinates: torch.Tensor,
        y_coordinates: torch.Tensor,
    ) -> torch.Tensor:
        leading_shape = map_tensor.shape[:-3]
        channels = map_tensor.shape[-3]
        flat = map_tensor.reshape(-1, *map_tensor.shape[-3:])
        mean = flat.mean(dim=(-2, -1))
        maximum = flat.amax(dim=(-2, -1))
        mass = flat.sum(dim=(-2, -1)).clamp_min(1e-6)
        center_x = (flat * x_coordinates).sum(dim=(-2, -1)) / mass
        center_y = (flat * y_coordinates).sum(dim=(-2, -1)) / mass
        statistics = torch.cat((mean, maximum, center_x, center_y), dim=-1)
        return statistics.reshape(*leading_shape, channels * 4)

    def forward(
        self,
        global_map: torch.Tensor,
        local_map: torch.Tensor,
        action_history: torch.Tensor,
    ) -> torch.Tensor:
        base = self.base_encoder(global_map, local_map, action_history)
        global_statistics = self._statistics(
            global_map, self.global_x, self.global_y
        )
        local_statistics = self._statistics(local_map, self.local_x, self.local_y)
        return torch.cat((base, global_statistics, local_statistics), dim=-1)


class CentralStateEncoder(nn.Module):
    """Encode the training-only world-centered central map for value mixing."""

    def __init__(
        self,
        channels: int,
        map_size: tuple[int, int] = (20, 20),
        hidden_dim: int = 64,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.map_size = tuple(int(value) for value in map_size)
        self.hidden_dim = int(hidden_dim)
        self.map_encoder = build_map_encoder(
            self.channels,
            CENTRAL_MAP_ENCODER_CHANNELS,
            activation,
            final_stride=1,
        )
        with torch.no_grad():
            flat_dim = self.map_encoder(
                torch.zeros(1, self.channels, *self.map_size)
            ).shape[-1]
        self.projection = nn.Sequential(
            nn.Linear(flat_dim, self.hidden_dim),
            activation_module(activation),
            nn.LayerNorm(self.hidden_dim),
        )

    def forward(self, central_map: torch.Tensor) -> torch.Tensor:
        leading_shape = central_map.shape[:-3]
        if tuple(central_map.shape[-3:]) != (self.channels, *self.map_size):
            raise ValueError(f"Unexpected central map shape: {central_map.shape}.")
        flat_map = central_map.reshape(-1, *central_map.shape[-3:])
        encoded = self.projection(self.map_encoder(flat_map))
        return encoded.reshape(*leading_shape, self.hidden_dim)


class AgentValueNetwork(nn.Module):
    """Estimate individual values before centralized value mixing."""

    def __init__(
        self,
        observation_dim: int,
        hidden_dim: int = 64,
        heads: int = 1,
    ) -> None:
        super().__init__()
        self.backbone = HistoryTransformerEncoder(observation_dim, hidden_dim, heads)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        observation_features: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.value_head(self.backbone(observation_features, valid_mask))


class CentralValueMixer(nn.Module):
    """Mix individual drone values using the training-only central state."""

    def __init__(self, central_dim: int, agent_count: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.agent_count = int(agent_count)
        self.network = nn.Sequential(
            nn.Linear(central_dim + self.agent_count, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        individual_values: torch.Tensor,
        central_features: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        masked_values = individual_values.squeeze(-1) * valid_mask
        return self.network(torch.cat((masked_values, central_features), dim=-1))


class ForwardPredictor(nn.Module):
    """Predict next global state and next local information from common skills."""

    def __init__(
        self,
        skill_dim: int,
        observation_dim: int,
        central_channels: int,
        central_map_size: tuple[int, int],
        hidden_dim: int = 64,
        heads: int = 1,
    ) -> None:
        super().__init__()
        self.central_channels = int(central_channels)
        self.central_map_size = tuple(int(value) for value in central_map_size)
        self.skill_projection = nn.Linear(skill_dim, hidden_dim)
        self.transformer = _single_layer_transformer(hidden_dim, heads)
        self.local_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, observation_dim),
        )
        central_size = (
            self.central_channels
            * self.central_map_size[0]
            * self.central_map_size[1]
        )
        self.central_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, central_size),
        )

    def forward(
        self,
        common_skills: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, agent_count, _ = common_skills.shape
        tokens = self.skill_projection(common_skills).reshape(
            batch_size * sequence_length,
            agent_count,
            -1,
        )
        encoded = self.transformer(tokens).reshape(
            batch_size,
            sequence_length,
            agent_count,
            -1,
        )
        local_prediction = self.local_predictor(encoded)
        denominator = valid_mask.sum(dim=2, keepdim=True).clamp_min(1.0)
        pooled = (encoded * valid_mask.unsqueeze(-1)).sum(dim=2) / denominator
        central_logits = self.central_predictor(pooled)
        central_prediction = torch.sigmoid(
            central_logits.reshape(
                batch_size,
                sequence_length,
                self.central_channels,
                *self.central_map_size,
            )
        )
        return central_prediction, local_prediction


class ContinuousActionDecoder(nn.Module):
    """HiSSD low-level controller with an exact BC-compatible base path."""

    def __init__(
        self,
        observation_dim: int,
        skill_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        heads: int = 1,
        task_action_residual: bool = False,
    ) -> None:
        super().__init__()
        self.observation_projection = nn.Linear(observation_dim, hidden_dim)
        self.common_projection = nn.Linear(skill_dim, hidden_dim)
        self.specific_projection = nn.Linear(skill_dim, hidden_dim)
        self.transformer = _single_layer_transformer(hidden_dim, heads)
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )
        self.base_action_head = nn.Linear(observation_dim, action_dim)
        self.task_action_residual_head: nn.Module | None = None
        if task_action_residual:
            self.enable_task_action_residual(skill_dim, hidden_dim, action_dim)
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def enable_task_action_residual(
        self,
        skill_dim: int,
        hidden_dim: int,
        action_dim: int,
    ) -> None:
        """Add a zero-initialized direct task-to-action path without policy drift."""
        if self.task_action_residual_head is not None:
            return
        head = nn.Sequential(
            nn.Linear(skill_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim, bias=False),
        )
        nn.init.zeros_(head[-1].weight)
        reference = self.base_action_head.weight
        self.task_action_residual_head = head.to(
            device=reference.device,
            dtype=reference.dtype,
        )

    def forward_logits(
        self,
        observation_features: torch.Tensor,
        common_skills: torch.Tensor,
        task_skills: torch.Tensor,
    ) -> torch.Tensor:
        """Return pre-tanh action logits for deterministic or stochastic control."""
        leading_shape = observation_features.shape[:-1]
        tokens = torch.stack(
            (
                self.observation_projection(observation_features),
                self.common_projection(common_skills),
                self.specific_projection(task_skills),
            ),
            dim=-2,
        ).reshape(-1, 3, self.observation_projection.out_features)
        decoded = self.transformer(tokens).reshape(*leading_shape, -1)
        logits = self.base_action_head(observation_features) + self.residual_head(decoded)
        if self.task_action_residual_head is not None:
            logits = logits + self.task_action_residual_head(task_skills)
        return logits

    def forward(
        self,
        observation_features: torch.Tensor,
        common_skills: torch.Tensor,
        task_skills: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.forward_logits(
            observation_features,
            common_skills,
            task_skills,
        )
        return torch.tanh(logits)


class HeMACHISSD(nn.Module):
    """Parameter-shared drone HiSSD model with EMA target modules."""

    def __init__(
        self,
        global_map_channels: int,
        local_map_channels: int,
        central_map_channels: int,
        *,
        agent_count: int = 3,
        action_dim: int = 3, # 에이전트들의 action space 크기
        global_map_size: tuple[int, int] = (40, 40),
        local_map_size: tuple[int, int] = (20, 20),
        central_map_size: tuple[int, int] = (20, 20),
        action_history_shape: tuple[int, int] = (5, 3), # 이전 5 step action
        observation_hidden_sizes: Sequence[int] = DRONE_HIDDEN_SIZES,
        hidden_dim: int = 64,
        skill_dim: int = 64,
        transformer_heads: int = 1,
        activation: str = "relu",
        contrastive_from_action_skill: bool = False, # contrastive loss를 action skill에서 계산할지, task skill에서 계산할지
        task_context_pooling: bool = False,
        task_descriptor_dim: int = 0, # task descriptor의 dimension
        task_prior_count: int = 0, # task prior의 개수
        task_dropout: float = 0.0,
        task_feature_deltas: bool = False,
        separate_task_observation_encoder: bool = False,
        learned_task_classifier: bool = False,
        task_spatial_statistics: bool = False,
        normalize_task_context: bool = True,
        task_running_statistics: bool = False,
        direct_task_summary: bool = False,
        task_action_residual: bool = False,
    ) -> None:
        super().__init__()
        self.agent_count = int(agent_count)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.skill_dim = int(skill_dim)
        self.central_map_channels = int(central_map_channels)
        self.central_map_size = tuple(int(value) for value in central_map_size)
        self.task_descriptor_dim = int(task_descriptor_dim)
        self.task_prior_count = int(task_prior_count)
        self.task_dropout = float(task_dropout)
        self.task_feature_deltas = bool(task_feature_deltas)
        self.separate_task_observation_encoder = bool(
            separate_task_observation_encoder
        )
        self.learned_task_classifier = bool(learned_task_classifier)
        self.task_spatial_statistics = bool(task_spatial_statistics)
        self.normalize_task_context = bool(normalize_task_context)
        self.task_running_statistics = bool(task_running_statistics)
        self.direct_task_summary = bool(direct_task_summary)
        self.task_action_residual = bool(task_action_residual)
        self.model_config = {
            "global_map_channels": int(global_map_channels),
            "local_map_channels": int(local_map_channels),
            "central_map_channels": self.central_map_channels,
            "agent_count": self.agent_count,
            "action_dim": self.action_dim,
            "global_map_size": tuple(global_map_size),
            "local_map_size": tuple(local_map_size),
            "central_map_size": self.central_map_size,
            "action_history_shape": tuple(action_history_shape),
            "observation_hidden_sizes": tuple(observation_hidden_sizes),
            "hidden_dim": self.hidden_dim,
            "skill_dim": self.skill_dim,
            "transformer_heads": int(transformer_heads),
            "activation": activation,
            "contrastive_from_action_skill": bool(contrastive_from_action_skill),
            "task_context_pooling": bool(task_context_pooling),
            "task_descriptor_dim": self.task_descriptor_dim,
            "task_prior_count": self.task_prior_count,
            "task_dropout": self.task_dropout,
            "task_feature_deltas": self.task_feature_deltas,
            "separate_task_observation_encoder": (
                self.separate_task_observation_encoder
            ),
            "learned_task_classifier": self.learned_task_classifier,
            "task_spatial_statistics": self.task_spatial_statistics,
            "normalize_task_context": self.normalize_task_context,
            "task_running_statistics": self.task_running_statistics,
            "direct_task_summary": self.direct_task_summary,
            "task_action_residual": self.task_action_residual,
        }

        self.observation_encoder = DroneObservationEncoder( # drone observation encoder (drone observation을 feature로 encoding)
            global_map_channels,
            local_map_channels,
            global_map_size=global_map_size,
            local_map_size=local_map_size,
            action_history_shape=action_history_shape,
            hidden_sizes=observation_hidden_sizes,
            activation=activation,
        )
        self.task_observation_encoder = None
        if self.separate_task_observation_encoder:
            task_encoder_type = (
                TaskObservationEncoder # task observation encoder (drone observation을 feature로 encoding, task specific)
                if self.task_spatial_statistics
                else DroneObservationEncoder
            )
            self.task_observation_encoder = task_encoder_type(
                global_map_channels,
                local_map_channels,
                global_map_size=global_map_size,
                local_map_size=local_map_size,
                action_history_shape=action_history_shape,
                hidden_sizes=observation_hidden_sizes,
                activation=activation,
            )
        observation_dim = self.observation_encoder.output_dim # DroneObservationEncoder의 output
        task_observation_dim = ( # TaskObservationEncoder의 output
            self.task_observation_encoder.output_dim
            if self.task_observation_encoder is not None
            else observation_dim
        )
        self.common_skill_encoder = CommonSkillEncoder( # common skill encoder (observation feature를 skill로 encoding)
            observation_dim, hidden_dim, skill_dim, transformer_heads
        )
        self.task_skill_encoder = TaskSpecificSkillEncoder( # task specific skill encoder (task observation feature를 skill로 encoding)
            task_observation_dim,
            hidden_dim,
            skill_dim,
            transformer_heads,
            contrastive_from_action_skill,
            task_context_pooling,
            self.task_dropout,
            self.task_feature_deltas,
            self.normalize_task_context,
            self.task_running_statistics,
            self.direct_task_summary,
        )
        self.action_decoder = ContinuousActionDecoder( # action decoder (observation feature와 skill를 action으로 decoding)
            observation_dim,
            skill_dim,
            action_dim,
            hidden_dim,
            transformer_heads,
            self.task_action_residual,
        )
        self.value_network = AgentValueNetwork( # value network (observation feature에서 value 예측)
            observation_dim, hidden_dim, transformer_heads
        )
        self.central_state_encoder = CentralStateEncoder( # central state encoder (training-only central map를 feature로 encoding)
            central_map_channels,
            central_map_size,
            hidden_dim,
            activation,
        )
        self.value_mixer = CentralValueMixer(hidden_dim, agent_count) # value mixer (individual value와 central state를 mix하여 global value 예측)
        self.forward_predictor = ForwardPredictor( # forward predictor (common skill에서 next global state와 next local information 예측)
            skill_dim,
            observation_dim,
            central_map_channels,
            central_map_size,
            hidden_dim,
            transformer_heads,
        )
        self.task_descriptor_head = None
        self.task_descriptor_dropout = nn.Dropout(self.task_dropout)
        if self.task_descriptor_dim > 0:
            self.task_descriptor_head = nn.Sequential( # task descriptor head (skill에서 task descriptor 예측)
                nn.Linear(self.skill_dim, self.hidden_dim),
                activation_module(activation),
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.task_descriptor_dim),
                nn.Sigmoid(),
            )
        self.task_prior_embeddings = None
        self.task_classifier_head = None
        if self.task_prior_count > 0:
            if self.learned_task_classifier:
                self.task_classifier_head = nn.Linear( # task classifier head (skill에서 task prior 예측)
                    self.skill_dim, self.task_prior_count
                )
            else:
                if self.task_prior_count > self.skill_dim:
                    raise ValueError("task_prior_count cannot exceed skill_dim.")
                self.task_prior_embeddings = nn.Embedding(
                    self.task_prior_count, self.skill_dim
                )
                with torch.no_grad():
                    self.task_prior_embeddings.weight.zero_()
                    self.task_prior_embeddings.weight[
                        :, : self.task_prior_count
                    ].copy_(torch.eye(self.task_prior_count))
                self.task_prior_embeddings.weight.requires_grad_(False)

        self.target_observation_encoder = copy.deepcopy(self.observation_encoder)
        self.target_task_skill_encoder = copy.deepcopy(self.task_skill_encoder)
        self.target_value_network = copy.deepcopy(self.value_network)
        self.target_central_state_encoder = copy.deepcopy(self.central_state_encoder)
        self.target_value_mixer = copy.deepcopy(self.value_mixer)
        self._freeze_target_modules()

    def _freeze_target_modules(self) -> None:
        for module in self.target_modules():
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def target_modules(self) -> tuple[nn.Module, ...]:
        return (
            self.target_observation_encoder,
            self.target_task_skill_encoder,
            self.target_value_network,
            self.target_central_state_encoder,
            self.target_value_mixer,
        )

    def initialize_from_bc(self, checkpoint_path: str | Path) -> dict[str, Any]:
        """Load the validated CNN and base action head from a BC checkpoint."""
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if payload.get("model_type") != "drone_behavior_cloning":
            raise ValueError(f"Not a drone BC checkpoint: {checkpoint_path}")
        bc_policy = DroneBehaviorCloningPolicy(**payload["model_config"])
        bc_policy.load_state_dict(payload["model_state_dict"])
        self.observation_encoder.load_state_dict(bc_policy.encoder.state_dict())
        if self.task_observation_encoder is not None:
            task_base_encoder = getattr(
                self.task_observation_encoder,
                "base_encoder",
                self.task_observation_encoder,
            )
            task_base_encoder.load_state_dict(bc_policy.encoder.state_dict())
        self.action_decoder.base_action_head.load_state_dict(
            bc_policy.action_head.state_dict()
        )
        self.target_observation_encoder.load_state_dict(
            self.observation_encoder.state_dict()
        )
        return {
            "checkpoint": str(checkpoint_path),
            "epoch": payload.get("epoch"),
            "metrics": payload.get("metrics", {}),
        }

    def encode_observations(
        self,
        observations: dict[str, torch.Tensor],
        *,
        target: bool = False,
    ) -> torch.Tensor:
        encoder = (
            self.target_observation_encoder if target else self.observation_encoder
        )
        return encoder(
            observations["global_map"],
            observations["local_map"],
            observations["action_history"],
        )

    def encode_task_observations(
        self,
        observations: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Encode observations through the task-specialized visual pathway."""
        if self.task_observation_encoder is None:
            return self.encode_observations(observations)
        return self.task_observation_encoder(
            observations["global_map"],
            observations["local_map"],
            observations["action_history"],
        )

    def infer_skills(
        self,
        observation_features: torch.Tensor,
        valid_mask: torch.Tensor,
        task_observation_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        common = self.common_skill_encoder(observation_features, valid_mask)
        if task_observation_features is None:
            task_observation_features = observation_features
        task_skill, contrastive = self.task_skill_encoder(
            task_observation_features, valid_mask
        )
        return common, task_skill, contrastive

    def decode_actions(
        self,
        observation_features: torch.Tensor,
        common_skills: torch.Tensor,
        task_skills: torch.Tensor,
    ) -> torch.Tensor:
        return self.action_decoder(
            observation_features,
            common_skills,
            task_skills,
        )

    def enable_task_action_residual(self) -> None:
        """Upgrade an older checkpoint with a backward-compatible task action head."""
        self.action_decoder.enable_task_action_residual(
            self.skill_dim,
            self.hidden_dim,
            self.action_dim,
        )
        self.task_action_residual = True
        self.model_config["task_action_residual"] = True

    def pool_task_context(
        self,
        contrastive_skills: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool each sequence's last valid shared task context over agents."""
        if contrastive_skills.ndim != 4 or valid_mask.ndim != 3:
            raise ValueError(
                "Expected contrastive skills [B,T,A,D] and mask [B,T,A]."
            )
        step_valid = valid_mask.bool().any(dim=2)
        valid_windows = step_valid.any(dim=1)
        last_indices = step_valid.long().sum(dim=1).sub(1).clamp_min(0)
        batch_indices = torch.arange(
            contrastive_skills.shape[0], device=contrastive_skills.device
        )
        last_skills = contrastive_skills[batch_indices, last_indices]
        last_mask = valid_mask[batch_indices, last_indices].bool()
        numeric_mask = last_mask.unsqueeze(-1).to(contrastive_skills.dtype)
        context = (last_skills * numeric_mask).sum(dim=1) / numeric_mask.sum(
            dim=1
        ).clamp_min(1.0)
        return context, valid_windows

    def predict_task_descriptor(
        self,
        contrastive_skills: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict a normalized continuous descriptor from causal task context."""
        if self.task_descriptor_head is None:
            raise RuntimeError("This model has no task descriptor prediction head.")
        context, valid_windows = self.pool_task_context(
            contrastive_skills, valid_mask
        )
        return self.task_descriptor_head(
            self.task_descriptor_dropout(context)
        ), valid_windows

    def predict_task_descriptor_sequence(
        self,
        contrastive_skills: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict episode dynamics from every causal shared task context."""
        if self.task_descriptor_head is None:
            raise RuntimeError("This model has no task descriptor prediction head.")
        numeric_mask = valid_mask.bool().unsqueeze(-1).to(contrastive_skills.dtype)
        context = (contrastive_skills * numeric_mask).sum(dim=2) / numeric_mask.sum(
            dim=2
        ).clamp_min(1.0)
        predictions = self.task_descriptor_head(
            self.task_descriptor_dropout(context)
        )
        return predictions, valid_mask.bool().any(dim=2)

    def task_prior_logits(
        self,
        contrastive_skills: torch.Tensor,
        valid_mask: torch.Tensor,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compare the last inferred context with fixed source-task priors."""
        if self.task_prior_embeddings is None:
            if self.task_classifier_head is None:
                raise RuntimeError("This model has no source-task classifier.")
            context, valid_windows = self.pool_task_context(
                contrastive_skills, valid_mask
            )
            return self.task_classifier_head(context), valid_windows
        context, valid_windows = self.pool_task_context(
            contrastive_skills, valid_mask
        )
        context = F.normalize(context, dim=-1)
        priors = F.normalize(self.task_prior_embeddings.weight, dim=-1)
        return context @ priors.transpose(0, 1) / temperature, valid_windows

    def task_prior_sequence_logits(
        self,
        contrastive_skills: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """Classify every causal context so each valid step supplies supervision."""
        if self.task_prior_embeddings is None:
            if self.task_classifier_head is None:
                raise RuntimeError("This model has no source-task classifier.")
            return self.task_classifier_head(contrastive_skills)
        if contrastive_skills.ndim != 4:
            raise ValueError("Expected contrastive skills [B,T,A,D].")
        contexts = F.normalize(contrastive_skills, dim=-1)
        priors = F.normalize(self.task_prior_embeddings.weight, dim=-1)
        return contexts @ priors.transpose(0, 1) / temperature

    def initial_inference_state(
        self,
        batch_size: int = 1,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor]:
        """Create episode state for online drone skill inference."""
        parameter = next(self.parameters())
        state_device = parameter.device if device is None else device
        state_dtype = parameter.dtype if dtype is None else dtype
        shape = (int(batch_size), self.agent_count, self.hidden_dim)
        return {
            "common_history": torch.zeros(
                shape, device=state_device, dtype=state_dtype
            ),
            "task_history": torch.zeros(
                shape, device=state_device, dtype=state_dtype
            ),
            "task_context": self.task_skill_encoder.initial_context_state(
                batch_size,
                device=state_device,
                dtype=state_dtype,
            ),
            "task_previous_pooled": self.task_skill_encoder.initial_context_state(
                batch_size,
                device=state_device,
                dtype=state_dtype,
            ),
            "task_previous_observation": (
                self.task_skill_encoder.initial_observation_state(
                    batch_size,
                    self.agent_count,
                    device=state_device,
                    dtype=state_dtype,
                )
            ),
            "task_previous_observation_valid": torch.zeros(
                int(batch_size),
                self.agent_count,
                device=state_device,
                dtype=torch.bool,
            ),
            "task_running_sum": self.task_skill_encoder.initial_observation_state(
                batch_size,
                self.agent_count,
                device=state_device,
                dtype=state_dtype,
            ),
            "task_running_square_sum": (
                self.task_skill_encoder.initial_observation_state(
                    batch_size,
                    self.agent_count,
                    device=state_device,
                    dtype=state_dtype,
                )
            ),
            "task_running_abs_delta_sum": (
                self.task_skill_encoder.initial_observation_state(
                    batch_size,
                    self.agent_count,
                    device=state_device,
                    dtype=state_dtype,
                )
            ),
            "task_running_count": torch.zeros(
                int(batch_size),
                self.agent_count,
                1,
                device=state_device,
                dtype=state_dtype,
            ),
            "task_running_delta_count": torch.zeros(
                int(batch_size),
                self.agent_count,
                1,
                device=state_device,
                dtype=state_dtype,
            ),
        }

    def inference_step(
        self,
        observations: dict[str, torch.Tensor],
        valid_mask: torch.Tensor,
        state: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Run one recurrent actor step for all parameter-sharing drones."""
        features = self.encode_observations(observations)
        task_features = self.encode_task_observations(observations)
        if features.ndim != 3 or features.shape[1] != self.agent_count:
            raise ValueError(
                "Online observations must encode to "
                f"[B,{self.agent_count},D], got {features.shape}."
            )
        common, common_history = self.common_skill_encoder.forward_step(
            features,
            valid_mask,
            state["common_history"],
        )
        (
            task_skill,
            contrastive,
            task_history,
            task_context,
            task_previous_pooled,
            task_previous_observation,
            task_previous_observation_valid,
            task_running_sum,
            task_running_square_sum,
            task_running_abs_delta_sum,
            task_running_count,
            task_running_delta_count,
        ) = self.task_skill_encoder.forward_step(
            task_features,
            valid_mask,
            state["task_history"],
            state["task_context"],
            state["task_previous_pooled"],
            state["task_previous_observation"],
            state["task_previous_observation_valid"],
            state["task_running_sum"],
            state["task_running_square_sum"],
            state["task_running_abs_delta_sum"],
            state["task_running_count"],
            state["task_running_delta_count"],
        )
        action_logits = self.action_decoder.forward_logits(
            features,
            common,
            task_skill,
        )
        actions = torch.tanh(action_logits)
        outputs = {
            "actions": actions,
            "action_logits": action_logits,
            "observation_features": features,
            "common_skills": common,
            "task_skills": task_skill,
            "contrastive_skills": contrastive,
        }
        if self.task_descriptor_head is not None:
            numeric_mask = valid_mask.bool().unsqueeze(-1).to(contrastive.dtype)
            pooled_context = (contrastive * numeric_mask).sum(dim=1) / numeric_mask.sum(
                dim=1
            ).clamp_min(1.0)
            outputs["task_descriptor"] = self.task_descriptor_head(
                self.task_descriptor_dropout(pooled_context)
            )
        next_state = {
            "common_history": common_history,
            "task_history": task_history,
            "task_context": task_context,
            "task_previous_pooled": task_previous_pooled,
            "task_previous_observation": task_previous_observation,
            "task_previous_observation_valid": task_previous_observation_valid,
            "task_running_sum": task_running_sum,
            "task_running_square_sum": task_running_square_sum,
            "task_running_abs_delta_sum": task_running_abs_delta_sum,
            "task_running_count": task_running_count,
            "task_running_delta_count": task_running_delta_count,
        }
        return outputs, next_state

    def total_value(
        self,
        observation_features: torch.Tensor,
        central_map: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        target: bool = False,
    ) -> torch.Tensor:
        boolean_mask = valid_mask.bool()
        numeric_mask = valid_mask.to(observation_features.dtype)
        if target:
            individual = self.target_value_network(
                observation_features, boolean_mask
            )
            central = self.target_central_state_encoder(central_map)
            return self.target_value_mixer(individual, central, numeric_mask)
        individual = self.value_network(observation_features, boolean_mask)
        central = self.central_state_encoder(central_map)
        return self.value_mixer(individual, central, numeric_mask)

    @torch.no_grad()
    def update_targets(self, tau: float) -> None:
        """EMA-update the frozen target networks used by offline objectives."""
        pairs = (
            (self.observation_encoder, self.target_observation_encoder),
            (self.task_skill_encoder, self.target_task_skill_encoder),
            (self.value_network, self.target_value_network),
            (self.central_state_encoder, self.target_central_state_encoder),
            (self.value_mixer, self.target_value_mixer),
        )
        for online, target in pairs:
            for online_parameter, target_parameter in zip(
                online.parameters(), target.parameters(), strict=True
            ):
                target_parameter.lerp_(online_parameter, tau)

    def config(self) -> dict[str, Any]:
        return dict(self.model_config)
