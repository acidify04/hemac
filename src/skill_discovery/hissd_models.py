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
        projected = self.input_projection(features)
        history = projected.new_zeros(batch_size * agent_count, self.hidden_dim)
        outputs = []
        for time_index in range(sequence_length):
            observation_token = projected[:, time_index].reshape(
                batch_size * agent_count,
                self.hidden_dim,
            )
            tokens = torch.stack((observation_token, history), dim=1)
            encoded = self.transformer(tokens)
            current = self.output_norm(encoded[:, 0])
            next_history = encoded[:, 1]
            active = valid_mask[:, time_index].reshape(-1, 1)
            history = torch.where(active, next_history, history)
            outputs.append(
                torch.where(active, current, torch.zeros_like(current)).reshape(
                    batch_size,
                    agent_count,
                    self.hidden_dim,
                )
            )
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


class TaskSpecificSkillEncoder(nn.Module):
    """HiSSD discriminator with action and MoCo projection heads."""

    def __init__(
        self,
        observation_dim: int,
        hidden_dim: int = 64,
        skill_dim: int = 64,
        heads: int = 1,
    ) -> None:
        super().__init__()
        self.backbone = HistoryTransformerEncoder(observation_dim, hidden_dim, heads)
        self.action_projection = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, skill_dim),
            nn.LayerNorm(skill_dim),
            nn.Tanh(),
        )
        self.contrastive_projection = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, skill_dim),
        )

    def forward(
        self,
        observation_features: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(observation_features, valid_mask)
        action_skill = self.action_projection(hidden)
        contrastive_skill = F.normalize(
            self.contrastive_projection(hidden),
            dim=-1,
        )
        return action_skill, contrastive_skill


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
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def forward(
        self,
        observation_features: torch.Tensor,
        common_skills: torch.Tensor,
        task_skills: torch.Tensor,
    ) -> torch.Tensor:
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
        action_dim: int = 3,
        global_map_size: tuple[int, int] = (40, 40),
        local_map_size: tuple[int, int] = (20, 20),
        central_map_size: tuple[int, int] = (20, 20),
        action_history_shape: tuple[int, int] = (5, 3),
        observation_hidden_sizes: Sequence[int] = DRONE_HIDDEN_SIZES,
        hidden_dim: int = 64,
        skill_dim: int = 64,
        transformer_heads: int = 1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.agent_count = int(agent_count)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.skill_dim = int(skill_dim)
        self.central_map_channels = int(central_map_channels)
        self.central_map_size = tuple(int(value) for value in central_map_size)
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
        }

        self.observation_encoder = DroneObservationEncoder(
            global_map_channels,
            local_map_channels,
            global_map_size=global_map_size,
            local_map_size=local_map_size,
            action_history_shape=action_history_shape,
            hidden_sizes=observation_hidden_sizes,
            activation=activation,
        )
        observation_dim = self.observation_encoder.output_dim
        self.common_skill_encoder = CommonSkillEncoder(
            observation_dim, hidden_dim, skill_dim, transformer_heads
        )
        self.task_skill_encoder = TaskSpecificSkillEncoder(
            observation_dim, hidden_dim, skill_dim, transformer_heads
        )
        self.action_decoder = ContinuousActionDecoder(
            observation_dim, skill_dim, action_dim, hidden_dim, transformer_heads
        )
        self.value_network = AgentValueNetwork(
            observation_dim, hidden_dim, transformer_heads
        )
        self.central_state_encoder = CentralStateEncoder(
            central_map_channels,
            central_map_size,
            hidden_dim,
            activation,
        )
        self.value_mixer = CentralValueMixer(hidden_dim, agent_count)
        self.forward_predictor = ForwardPredictor(
            skill_dim,
            observation_dim,
            central_map_channels,
            central_map_size,
            hidden_dim,
            transformer_heads,
        )

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

    def infer_skills(
        self,
        observation_features: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        common = self.common_skill_encoder(observation_features, valid_mask)
        task_skill, contrastive = self.task_skill_encoder(
            observation_features, valid_mask
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
        """EMA-update the value and MoCo target networks."""
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
