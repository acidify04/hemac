"""Minimal heterogeneous-agent extension for HeMAC HiSSD.

This module intentionally leaves hissd_models.py unchanged.
It inserts a residual, observation-conditioned adapter between the
existing common-skill planner and the existing low-level controller.

Intended first experiment:
    raw common skill c_i + encoded observation feature h_i
        -> role adapter -> delta c_i
        -> adapted common skill c'_i = c_i + delta c_i
        -> existing HiSSD action decoder

The original common skill remains available to HiSSD's forward predictor
and other planner-side objectives.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .hissd_models import HeMACHISSD


class RoleConditionedCommonSkillAdapter(nn.Module):
    """Adapt a common skill using each agent's encoded observation feature.

    The same adapter parameters are shared across all agents. Therefore the
    module cannot rely on a dedicated per-agent network; differences must be
    inferred from the per-agent observation feature.

    Shapes are arbitrary in the leading dimensions, e.g.
      offline: [B, T, A, D]
      online:  [B, A, D]
    """

    def __init__(
        self,
        observation_dim: int,
        skill_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.skill_dim = int(skill_dim)
        self.hidden_dim = int(hidden_dim)

        self.adapter = nn.Sequential(
            nn.Linear(self.observation_dim + self.skill_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.skill_dim),
        )

        # Exact identity at initialization:
        # adapted_common == raw_common before any adapter training.
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(
        self,
        observation_features: torch.Tensor,
        common_skills: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if observation_features.shape[:-1] != common_skills.shape[:-1]:
            raise ValueError(
                "Observation/common leading shapes must match: "
                f"{observation_features.shape} vs {common_skills.shape}."
            )
        if observation_features.shape[-1] != self.observation_dim:
            raise ValueError(
                f"Expected observation feature dim {self.observation_dim}, "
                f"got {observation_features.shape[-1]}."
            )
        if common_skills.shape[-1] != self.skill_dim:
            raise ValueError(
                f"Expected common skill dim {self.skill_dim}, "
                f"got {common_skills.shape[-1]}."
            )

        adapter_input = torch.cat((common_skills, observation_features), dim=-1)
        delta = self.adapter(adapter_input)
        adapted_common = common_skills + delta
        return adapted_common, delta


class HeterogeneousHeMACHISSD(HeMACHISSD):
    """HiSSD with a residual role-conditioned common-skill adapter.

    Important design choice:
    - CommonSkillEncoder is unchanged.
    - ForwardPredictor still receives the *raw* common skill.
    - Only the common skill passed into the low-level action decoder is adapted.

    This keeps the new module exactly between planner and controller.
    """

    def __init__(
        self,
        global_map_channels: int,
        local_map_channels: int,
        central_map_channels: int,
        *,
        role_adapter_hidden_dim: int = 128,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            global_map_channels,
            local_map_channels,
            central_map_channels,
            **kwargs,
        )

        # First version is intentionally defined for the normal HiSSD split:
        # common skill + task-specific skill.
        if self.skill_structure != "split":
            raise ValueError(
                "HeterogeneousHeMACHISSD first version expects "
                "skill_structure='split'."
            )

        observation_dim = int(self.observation_encoder.output_dim)
        self.role_adapter_hidden_dim = int(role_adapter_hidden_dim)
        self.common_skill_role_adapter = RoleConditionedCommonSkillAdapter(
            observation_dim=observation_dim,
            skill_dim=self.skill_dim,
            hidden_dim=self.role_adapter_hidden_dim,
        )

    def adapt_common_skills(
        self,
        observation_features: torch.Tensor,
        common_skills: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return adapted common skills and the residual delta."""
        return self.common_skill_role_adapter(
            observation_features,
            common_skills,
        )

    def decode_action_logits(
        self,
        observation_features: torch.Tensor,
        common_skills: torch.Tensor,
        task_skills: torch.Tensor,
    ) -> torch.Tensor:
        """Insert the adapter immediately before the existing controller."""
        adapted_common, _ = self.adapt_common_skills(
            observation_features,
            common_skills,
        )

        return self.action_decoder.forward_logits(
            observation_features,
            adapted_common,
            task_skills,
            direct_residual_skills=self.conditioning_skill(
                adapted_common,
                task_skills,
            ),
        )

    def inference_step(
        self,
        observations: dict[str, torch.Tensor],
        valid_mask: torch.Tensor,
        state: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Run base recurrent inference and expose adapter diagnostics.

        super().inference_step() calls self.decode_action_logits(), so the
        returned actions already use the adapted common skill.
        """
        outputs, next_state = super().inference_step(
            observations,
            valid_mask,
            state,
        )

        raw_common = outputs["common_skills"]
        adapted_common, delta = self.adapt_common_skills(
            outputs["observation_features"],
            raw_common,
        )

        # Keep the original key semantically identical to base HiSSD and add
        # explicit diagnostic tensors for analysis.
        outputs["raw_common_skills"] = raw_common
        outputs["adapted_common_skills"] = adapted_common
        outputs["common_skill_delta"] = delta

        return outputs, next_state

    @classmethod
    def from_base_model(
        cls,
        base_model: HeMACHISSD,
        *,
        role_adapter_hidden_dim: int = 128,
    ) -> "HeterogeneousHeMACHISSD":
        """Upgrade an already-loaded base HiSSD model without changing its file.

        All original parameters are copied exactly. The only missing parameters
        are the newly created role-adapter parameters, which start as identity.
        """
        model = cls(
            **base_model.config(),
            role_adapter_hidden_dim=role_adapter_hidden_dim,
        )
        incompatible = model.load_state_dict(base_model.state_dict(), strict=False)

        unexpected = list(incompatible.unexpected_keys)
        missing = list(incompatible.missing_keys)
        allowed_prefix = "common_skill_role_adapter."

        if unexpected:
            raise RuntimeError(f"Unexpected base-model keys: {unexpected}")
        if any(not key.startswith(allowed_prefix) for key in missing):
            raise RuntimeError(
                "Missing non-adapter parameters while upgrading base HiSSD: "
                f"{missing}"
            )

        return model

    def adapter_parameter_count(self) -> int:
        return sum(p.numel() for p in self.common_skill_role_adapter.parameters())
