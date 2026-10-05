"""Joint heterogeneous behavior-cloning baseline for HeMAC.

Role-specific CNN encoders map Drone/Observer observations to the same feature
width. A shared permutation-equivariant Transformer then mixes the current
features of all active agents. Role-specific heads decode normalized actions.

There is no skill encoder, value function, planner, task loss, adapter, agent ID,
or population-size embedding.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .models import DroneObservationEncoder

ROLE_ORDER = ("observer", "drone")


def _normalise_role_config(config: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "global_map_channels",
        "local_map_channels",
        "global_map_size",
        "local_map_size",
        "action_history_shape",
        "hidden_sizes",
        "activation",
        "action_dim",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Role config is missing keys: {missing}")
    return {
        "global_map_channels": int(config["global_map_channels"]),
        "local_map_channels": int(config["local_map_channels"]),
        "global_map_size": tuple(int(x) for x in config["global_map_size"]),
        "local_map_size": tuple(int(x) for x in config["local_map_size"]),
        "action_history_shape": tuple(int(x) for x in config["action_history_shape"]),
        "hidden_sizes": tuple(int(x) for x in config["hidden_sizes"]),
        "activation": str(config["activation"]),
        "action_dim": int(config["action_dim"]),
    }


class JointBehaviorCloningPolicy(nn.Module):
    """Joint BC with role-specific I/O and one shared agent-interaction block."""

    def __init__(
        self,
        role_configs: Mapping[str, Mapping[str, Any]],
        *,
        joint_heads: int = 4,
        joint_layers: int = 1,
        joint_ff_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.role_configs = {
            role: _normalise_role_config(role_configs[role])
            for role in ROLE_ORDER
            if role in role_configs
        }
        if set(self.role_configs) != set(ROLE_ORDER):
            raise ValueError("role_configs must contain observer and drone")

        dims = {
            role: cfg["hidden_sizes"][-1]
            for role, cfg in self.role_configs.items()
        }
        if len(set(dims.values())) != 1:
            raise ValueError(f"Role encoders must share output width, got {dims}")
        self.feature_dim = int(next(iter(dims.values())))
        self.joint_heads = int(joint_heads)
        self.joint_layers = int(joint_layers)
        self.joint_ff_dim = int(joint_ff_dim or (2 * self.feature_dim))
        self.dropout = float(dropout)

        if self.feature_dim % self.joint_heads != 0:
            raise ValueError(
                f"feature_dim={self.feature_dim} must be divisible by joint_heads={self.joint_heads}"
            )

        self.observation_encoder = nn.ModuleDict({
            role: DroneObservationEncoder(
                cfg["global_map_channels"],
                cfg["local_map_channels"],
                global_map_size=cfg["global_map_size"],
                local_map_size=cfg["local_map_size"],
                action_history_shape=cfg["action_history_shape"],
                hidden_sizes=cfg["hidden_sizes"],
                activation=cfg["activation"],
            )
            for role, cfg in self.role_configs.items()
        })

        layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=self.joint_heads,
            dim_feedforward=self.joint_ff_dim,
            dropout=self.dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.joint_encoder = nn.TransformerEncoder(
            layer,
            num_layers=self.joint_layers,
            norm=nn.LayerNorm(self.feature_dim),
        )

        self.action_head = nn.ModuleDict({
            role: nn.Linear(self.feature_dim, cfg["action_dim"])
            for role, cfg in self.role_configs.items()
        })
        for head in self.action_head.values():
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)

    def config(self) -> dict[str, Any]:
        return {
            "role_configs": copy.deepcopy(self.role_configs),
            "joint_heads": self.joint_heads,
            "joint_layers": self.joint_layers,
            "joint_ff_dim": self.joint_ff_dim,
            "dropout": self.dropout,
        }

    def encode_role(self, role: str, observations: Mapping[str, torch.Tensor]) -> torch.Tensor:
        enc = self.observation_encoder[role]
        return enc(
            observations["global_map"],
            observations["local_map"],
            observations["action_history"],
        )

    def encode_joint(
        self,
        observations_by_role: Mapping[str, Mapping[str, torch.Tensor]],
        valid_masks_by_role: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, slice]]:
        features = {role: self.encode_role(role, observations_by_role[role]) for role in ROLE_ORDER}
        prefix = features[ROLE_ORDER[0]].shape[:-2]
        width = features[ROLE_ORDER[0]].shape[-1]

        role_slices = {}
        xs, ms = [], []
        offset = 0
        for role in ROLE_ORDER:
            x = features[role]
            m = valid_masks_by_role[role].bool()
            if x.shape[:-2] != prefix or x.shape[-1] != width:
                raise ValueError("Role feature batch/time dimensions do not match")
            if x.shape[:-1] != m.shape:
                raise ValueError(f"{role} feature/mask mismatch: {x.shape} vs {m.shape}")
            count = int(x.shape[-2])
            role_slices[role] = slice(offset, offset + count)
            offset += count
            xs.append(x)
            ms.append(m)

        return torch.cat(xs, dim=-2), torch.cat(ms, dim=-1), role_slices

    def mix_agents(self, joint_features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Mix agents independently at each leading batch/time position."""
        if joint_features.shape[:-1] != valid_mask.shape:
            raise ValueError("joint feature/mask shape mismatch")
        leading = joint_features.shape[:-2]
        agent_count = joint_features.shape[-2]
        flat_x = joint_features.reshape(-1, agent_count, self.feature_dim)
        flat_valid = valid_mask.reshape(-1, agent_count).bool()

        # Transformer softmax is undefined if every key is masked. Padded time
        # rows are temporarily given one unmasked zero token, then zeroed again.
        padding_mask = ~flat_valid
        all_invalid = ~flat_valid.any(dim=1)
        if all_invalid.any():
            padding_mask = padding_mask.clone()
            padding_mask[all_invalid, 0] = False
            flat_x = flat_x.clone()
            flat_x[all_invalid, 0] = 0.0

        mixed = self.joint_encoder(flat_x, src_key_padding_mask=padding_mask)
        mixed = mixed * flat_valid.unsqueeze(-1).to(mixed.dtype)
        return mixed.reshape(*leading, agent_count, self.feature_dim)

    def forward_joint(
        self,
        observations_by_role: Mapping[str, Mapping[str, torch.Tensor]],
        valid_masks_by_role: Mapping[str, torch.Tensor],
    ) -> dict[str, Any]:
        features, valid_mask, role_slices = self.encode_joint(
            observations_by_role, valid_masks_by_role
        )
        mixed = self.mix_agents(features, valid_mask)
        actions = {
            role: torch.tanh(self.action_head[role](mixed[..., sl, :]))
            for role, sl in role_slices.items()
        }
        return {
            "actions": actions,
            "observation_features": features,
            "joint_features": mixed,
            "valid_mask": valid_mask,
            "role_slices": role_slices,
        }


def load_joint_bc_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[JointBehaviorCloningPolicy, dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload.get("model_type") != "hemac_joint_behavior_cloning":
        raise ValueError(
            f"Not a joint BC checkpoint: model_type={payload.get('model_type')!r}"
        )
    model = JointBehaviorCloningPolicy(**payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state_dict"])
    return model, payload
