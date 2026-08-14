"""Standalone PyTorch models used by offline drone skill discovery."""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt

import torch
from torch import nn


# Keep these aligned with the shared MAPPO drone actor in hemac.rllib_policy.
GLOBAL_MAP_ENCODER_CHANNELS = (8, 16, 16, 32)
LOCAL_MAP_ENCODER_CHANNELS = (16, 32, 32, 64)
DRONE_HIDDEN_SIZES = (96, 96)


def activation_module(name: str) -> nn.Module:
    """Return a fresh activation module for one network layer."""
    normalized = name.lower()
    if normalized == "tanh":
        return nn.Tanh()
    if normalized in {"silu", "swish"}:
        return nn.SiLU()
    if normalized == "linear":
        return nn.Identity()
    if normalized != "relu":
        raise ValueError(f"Unsupported activation: {name!r}")
    return nn.ReLU()


def build_map_encoder(
    input_channels: int,
    encoder_channels: Sequence[int],
    activation: str,
    *,
    final_stride: int,
) -> nn.Sequential:
    """Build the same compact map CNN used by the online MAPPO actor."""
    conv1, conv2, conv3, conv4 = (int(value) for value in encoder_channels)
    return nn.Sequential(
        nn.Conv2d(input_channels, conv1, kernel_size=5, stride=2, padding=2),
        activation_module(activation),
        nn.Conv2d(conv1, conv2, kernel_size=3, stride=2, padding=1),
        activation_module(activation),
        nn.Conv2d(conv2, conv3, kernel_size=3, stride=1, padding=1),
        activation_module(activation),
        nn.Conv2d(conv3, conv3, kernel_size=3, stride=1, padding=1),
        activation_module(activation),
        nn.MaxPool2d(2),
        nn.Conv2d(conv3, conv4, kernel_size=3, stride=final_stride, padding=1),
        activation_module(activation),
        nn.Flatten(),
    )


class DroneObservationEncoder(nn.Module):
    """Encode decentralized observations for any leading batch dimensions."""

    def __init__(
        self,
        global_map_channels: int,
        local_map_channels: int,
        *,
        global_map_size: tuple[int, int] = (40, 40),
        local_map_size: tuple[int, int] = (20, 20),
        action_history_shape: tuple[int, int] = (5, 3),
        hidden_sizes: Sequence[int] = DRONE_HIDDEN_SIZES,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.global_map_channels = int(global_map_channels)
        self.local_map_channels = int(local_map_channels)
        self.global_map_size = tuple(int(value) for value in global_map_size)
        self.local_map_size = tuple(int(value) for value in local_map_size)
        self.action_history_shape = tuple(int(value) for value in action_history_shape)
        self.hidden_sizes = tuple(int(value) for value in hidden_sizes)
        self.activation = activation

        self.global_map_encoder = build_map_encoder(
            self.global_map_channels,
            GLOBAL_MAP_ENCODER_CHANNELS,
            activation,
            final_stride=2,
        )
        self.local_map_encoder = build_map_encoder(
            self.local_map_channels,
            LOCAL_MAP_ENCODER_CHANNELS,
            activation,
            final_stride=1,
        )

        with torch.no_grad():
            global_feature_dim = self.global_map_encoder(
                torch.zeros(1, self.global_map_channels, *self.global_map_size)
            ).shape[-1]
            local_feature_dim = self.local_map_encoder(
                torch.zeros(1, self.local_map_channels, *self.local_map_size)
            ).shape[-1]

        history_dim = self.action_history_shape[0] * self.action_history_shape[1]
        input_dim = int(global_feature_dim + local_feature_dim + history_dim)
        layers: list[nn.Module] = []
        for hidden_size in self.hidden_sizes:
            linear = nn.Linear(input_dim, hidden_size)
            nn.init.orthogonal_(linear.weight, gain=sqrt(2.0))
            nn.init.zeros_(linear.bias)
            layers.extend((linear, activation_module(activation)))
            input_dim = hidden_size
        self.fusion = nn.Sequential(*layers) if layers else nn.Identity()
        self.output_dim = input_dim

    def forward(
        self,
        global_map: torch.Tensor,
        local_map: torch.Tensor,
        action_history: torch.Tensor,
    ) -> torch.Tensor:
        """Return one feature vector per input time and drone position."""
        if global_map.shape[:-3] != local_map.shape[:-3]:
            raise ValueError("Global and local maps have different leading dimensions.")
        if global_map.shape[:-3] != action_history.shape[:-2]:
            raise ValueError("Map and action-history leading dimensions do not match.")
        if tuple(global_map.shape[-3:]) != (
            self.global_map_channels,
            *self.global_map_size,
        ):
            raise ValueError(f"Unexpected global map shape: {tuple(global_map.shape)}")
        if tuple(local_map.shape[-3:]) != (
            self.local_map_channels,
            *self.local_map_size,
        ):
            raise ValueError(f"Unexpected local map shape: {tuple(local_map.shape)}")
        if tuple(action_history.shape[-2:]) != self.action_history_shape:
            raise ValueError(
                f"Unexpected action history shape: {tuple(action_history.shape)}"
            )

        leading_shape = global_map.shape[:-3]
        flat_count = global_map.numel() // (
            self.global_map_channels
            * self.global_map_size[0]
            * self.global_map_size[1]
        )
        global_features = self.global_map_encoder(
            global_map.reshape(flat_count, *global_map.shape[-3:])
        )
        local_features = self.local_map_encoder(
            local_map.reshape(flat_count, *local_map.shape[-3:])
        )
        history_features = action_history.reshape(flat_count, -1)
        features = self.fusion(
            torch.cat((global_features, local_features, history_features), dim=-1)
        )
        return features.reshape(*leading_shape, self.output_dim)

    def config(self) -> dict[str, object]:
        """Return serializable constructor settings for checkpoints."""
        return {
            "global_map_channels": self.global_map_channels,
            "local_map_channels": self.local_map_channels,
            "global_map_size": self.global_map_size,
            "local_map_size": self.local_map_size,
            "action_history_shape": self.action_history_shape,
            "hidden_sizes": self.hidden_sizes,
            "activation": self.activation,
        }


class DroneBehaviorCloningPolicy(nn.Module):
    """Shared deterministic policy for all drones in the offline dataset."""

    def __init__(
        self,
        global_map_channels: int,
        local_map_channels: int,
        *,
        action_dim: int = 3,
        global_map_size: tuple[int, int] = (40, 40),
        local_map_size: tuple[int, int] = (20, 20),
        action_history_shape: tuple[int, int] = (5, 3),
        hidden_sizes: Sequence[int] = DRONE_HIDDEN_SIZES,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.encoder = DroneObservationEncoder(
            global_map_channels,
            local_map_channels,
            global_map_size=global_map_size,
            local_map_size=local_map_size,
            action_history_shape=action_history_shape,
            hidden_sizes=hidden_sizes,
            activation=activation,
        )
        self.action_head = nn.Linear(self.encoder.output_dim, self.action_dim)
        nn.init.orthogonal_(self.action_head.weight, gain=0.01)
        nn.init.zeros_(self.action_head.bias)

    def forward(
        self,
        global_map: torch.Tensor,
        local_map: torch.Tensor,
        action_history: torch.Tensor,
    ) -> torch.Tensor:
        """Predict actions normalized to the dataset range [-1, 1]."""
        features = self.encoder(global_map, local_map, action_history)
        return torch.tanh(self.action_head(features))

    def config(self) -> dict[str, object]:
        """Return serializable constructor settings for checkpoints."""
        return {**self.encoder.config(), "action_dim": self.action_dim}

