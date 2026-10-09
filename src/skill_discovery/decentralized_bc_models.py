"""Decentralized heterogeneous behavior-cloning baseline for HeMAC.

Each agent action is computed only from that agent's own actor observation:
    h_i = E_role(o_i)
    a_i = tanh(H_role(h_i))

No cross-agent Transformer is used. Environment-provided teammate positions,
shared explored regions, and known enemy positions already contained in o_i
remain valid inputs.
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
        "global_map_channels", "local_map_channels",
        "global_map_size", "local_map_size",
        "action_history_shape", "hidden_sizes",
        "activation", "action_dim",
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


class DecentralizedBehaviorCloningPolicy(nn.Module):
    def __init__(self, role_configs: Mapping[str, Mapping[str, Any]]) -> None:
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
        self.action_head = nn.ModuleDict({
            role: nn.Linear(self.feature_dim, cfg["action_dim"])
            for role, cfg in self.role_configs.items()
        })
        for head in self.action_head.values():
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)

    def config(self) -> dict[str, Any]:
        return {"role_configs": copy.deepcopy(self.role_configs)}

    def encode_role(
        self,
        role: str,
        observations: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
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
        features = {
            role: self.encode_role(role, observations_by_role[role])
            for role in ROLE_ORDER
        }
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

        # Concatenation is bookkeeping only; there is no cross-agent operation.
        return torch.cat(xs, dim=-2), torch.cat(ms, dim=-1), role_slices

    def forward_joint(
        self,
        observations_by_role: Mapping[str, Mapping[str, torch.Tensor]],
        valid_masks_by_role: Mapping[str, torch.Tensor],
    ) -> dict[str, Any]:
        features, valid_mask, role_slices = self.encode_joint(
            observations_by_role, valid_masks_by_role
        )
        actions = {
            role: torch.tanh(self.action_head[role](features[..., sl, :]))
            for role, sl in role_slices.items()
        }
        return {
            "actions": actions,
            "observation_features": features,
            "joint_features": features,  # compatibility; not mixed
            "valid_mask": valid_mask,
            "role_slices": role_slices,
        }


def load_decentralized_bc_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[DecentralizedBehaviorCloningPolicy, dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    expected = "hemac_decentralized_behavior_cloning"
    if payload.get("model_type") != expected:
        raise ValueError(
            f"Not a decentralized BC checkpoint: "
            f"model_type={payload.get('model_type')!r}, expected={expected!r}"
        )
    model = DecentralizedBehaviorCloningPolicy(
        **payload["model_config"]
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    return model, payload
