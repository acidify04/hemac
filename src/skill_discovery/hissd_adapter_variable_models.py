"""Variable-population HiSSD with a residual common-skill adapter.

Only one architectural change is introduced:

    observation -> CommonSkillEncoder -> CommonSkillAdapter -> downstream HiSSD

Everything downstream sees the adapted common skill, including:
- controller/action decoder,
- planner/forward predictor,
- recurrent online inference.

The adapter takes only the common-skill vector. It does not receive the
observation feature, role ID, task ID, or population size.

A separate checkpoint is trained for each role (drone / observer), so each
checkpoint contains one role-specific adapter while retaining the original
role-specific observation/action spaces.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .hissd_variable_models import VariableAgentHeMACHISSD


class ResidualCommonSkillAdapter(nn.Module):
    """Residual MLP c' = c + A(c), identity-initialized."""

    def __init__(self, skill_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.skill_dim = int(skill_dim)
        self.hidden_dim = int(hidden_dim)

        self.network = nn.Sequential(
            nn.Linear(self.skill_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.skill_dim),
        )

        # Exact identity at initialization: A(c) == 0.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        common_skill: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if common_skill.shape[-1] != self.skill_dim:
            raise ValueError(
                f"Expected common skill dim {self.skill_dim}, "
                f"got {common_skill.shape[-1]}."
            )
        delta = self.network(common_skill)
        return common_skill + delta, delta


class AdaptedCommonSkillEncoder(nn.Module):
    """Wrap the original CommonSkillEncoder and adapt every emitted skill.

    This wrapper preserves the original encoder interface:
      forward(features, valid_mask) -> common_skill
      forward_step(features, valid_mask, history) -> common_skill, next_history

    Therefore existing HiSSD code does not need to change. Any existing path
    that calls model.common_skill_encoder(...) automatically receives the
    adapted common skill.
    """

    def __init__(
        self,
        base_encoder: nn.Module,
        skill_dim: int,
        adapter_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.base_encoder = base_encoder
        self.skill_dim = int(skill_dim)
        self.adapter_hidden_dim = int(adapter_hidden_dim)
        self.adapter = ResidualCommonSkillAdapter(
            skill_dim=self.skill_dim,
            hidden_dim=self.adapter_hidden_dim,
        )

    def forward(
        self,
        observation_features: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        raw_common = self.base_encoder(observation_features, valid_mask)
        adapted_common, _ = self.adapter(raw_common)
        return adapted_common

    def forward_step(
        self,
        observation_features: torch.Tensor,
        valid_mask: torch.Tensor,
        history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw_common, next_history = self.base_encoder.forward_step(
            observation_features,
            valid_mask,
            history,
        )
        adapted_common, _ = self.adapter(raw_common)
        return adapted_common, next_history

    def raw_forward(
        self,
        observation_features: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Expose the pre-adapter common skill for diagnostics only."""
        return self.base_encoder(observation_features, valid_mask)

    def adapt(
        self,
        raw_common: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.adapter(raw_common)


class VariableAgentAdaptedHeMACHISSD(VariableAgentHeMACHISSD):
    """Variable-agent HiSSD whose common skill is adapted before all use."""

    def __init__(
        self,
        global_map_channels: int,
        local_map_channels: int,
        central_map_channels: int,
        *,
        common_skill_adapter_hidden_dim: int = 128,
        common_skill_adapter_enabled: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            global_map_channels,
            local_map_channels,
            central_map_channels,
            **kwargs,
        )

        if self.skill_structure != "split":
            raise ValueError(
                "VariableAgentAdaptedHeMACHISSD expects "
                "skill_structure='split'."
            )

        self.common_skill_adapter_hidden_dim = int(
            common_skill_adapter_hidden_dim
        )
        self.common_skill_adapter_enabled = bool(
            common_skill_adapter_enabled
        )
        if not self.common_skill_adapter_enabled:
            raise ValueError(
                "This model class is specifically for the adapter condition; "
                "use VariableAgentHeMACHISSD for vanilla HiSSD."
            )

        original_encoder = self.common_skill_encoder
        self.common_skill_encoder = AdaptedCommonSkillEncoder(
            original_encoder,
            skill_dim=self.skill_dim,
            adapter_hidden_dim=self.common_skill_adapter_hidden_dim,
        )

        # Persist reconstruction metadata.
        self.model_config["common_skill_adapter_enabled"] = True
        self.model_config["common_skill_adapter_hidden_dim"] = (
            self.common_skill_adapter_hidden_dim
        )

    def config(self) -> dict[str, Any]:
        config = super().config()
        config["common_skill_adapter_enabled"] = True
        config["common_skill_adapter_hidden_dim"] = (
            self.common_skill_adapter_hidden_dim
        )
        return config

    @torch.no_grad()
    def adapter_identity_error(
        self,
        observation_features: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> float:
        """Max |adapted - raw|; exactly zero immediately after construction."""
        raw = self.common_skill_encoder.raw_forward(
            observation_features,
            valid_mask,
        )
        adapted, _ = self.common_skill_encoder.adapt(raw)
        return float((adapted - raw).abs().max().item())

    @torch.no_grad()
    def adapter_delta_statistics(
        self,
        observation_features: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, float]:
        """Diagnostic magnitude of learned role adaptation."""
        raw = self.common_skill_encoder.raw_forward(
            observation_features,
            valid_mask,
        )
        _, delta = self.common_skill_encoder.adapt(raw)

        mask = valid_mask.bool().unsqueeze(-1)
        numeric = mask.to(delta.dtype)
        denominator = (
            numeric.sum() * delta.shape[-1]
        ).clamp_min(1.0)

        delta_rms = torch.sqrt(
            (delta.square() * numeric).sum() / denominator
        )
        raw_rms = torch.sqrt(
            (raw.square() * numeric).sum() / denominator
        )
        ratio = delta_rms / raw_rms.clamp_min(1e-8)

        return {
            "adapter_delta_rms": float(delta_rms.item()),
            "raw_common_rms": float(raw_rms.item()),
            "adapter_to_common_ratio": float(ratio.item()),
        }


def load_adapted_hissd_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[VariableAgentAdaptedHeMACHISSD, dict[str, Any]]:
    """Load a native adapter checkpoint."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if "model_config" not in payload or "model_state_dict" not in payload:
        raise ValueError(
            f"Checkpoint lacks model_config/model_state_dict: {checkpoint_path}"
        )

    config = dict(payload["model_config"])
    if not bool(config.get("common_skill_adapter_enabled", False)):
        raise ValueError(
            f"Not an adapted HiSSD checkpoint: {checkpoint_path}"
        )

    model = VariableAgentAdaptedHeMACHISSD(**config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    return model, payload
