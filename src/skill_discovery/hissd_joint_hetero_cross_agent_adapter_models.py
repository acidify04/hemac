"""Cross-agent contextual common-skill adapter for joint heterogeneous HiSSD.

This module intentionally leaves the existing local adapter implementation
(`hissd_joint_hetero_adapter_models.py`) unchanged so the two variants can be
tested independently.

Baseline local adapter
----------------------
    h_i = E_{r_i}(o_i)
    c_i = C(h_i, history_i)
    delta_i = A_local([c_i, h_i])
    c'_i = c_i + delta_i

Cross-agent adapter (this file)
-------------------------------
    x_i = [h_i, c_i]
    g_i = SelfAttention(x_1, ..., x_N)_i
    delta_i = A_cross([c_i, h_i, g_i])
    c'_i = c_i + delta_i

The self-attention is applied across the agent axis at each batch/time index.
It supports variable population sizes through `valid_mask`. No agent ID, role
embedding, or population-size embedding is introduced. Role information can
only be represented implicitly through the role-specific observation features
and the common skills.

The residual MLP's final layer is zero-initialized, so c'_i == c_i exactly at
construction even though the context-attention block is randomly initialized.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .hissd_joint_hetero_adapter_models import (
    JointHeterogeneousAdapterHiSSD,
)


class CrossAgentCommonSkillAdapter(nn.Module):
    """Shared residual adapter with explicit cross-agent context.

    Inputs
    ------
    common_skills:
        [..., A, skill_dim]
    observation_features:
        [..., A, observation_dim]
    valid_mask:
        [..., A]

    Computation
    -----------
        x_i = Proj([h_i, c_i])
        g_1:A = MHA(x_1:A, x_1:A, x_1:A)
        delta_i = MLP([c_i, h_i, g_i])
        c'_i = c_i + delta_i

    The attention includes the query agent itself as well as all other valid
    agents. Because [c_i, h_i] is also supplied directly to the residual MLP,
    g_i acts as an additional joint-team context rather than replacing the
    agent's own information.
    """

    def __init__(
        self,
        observation_dim: int,
        skill_dim: int,
        hidden_dim: int = 128,
        context_dim: int = 64,
        attention_heads: int = 4,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.skill_dim = int(skill_dim)
        self.hidden_dim = int(hidden_dim)
        self.context_dim = int(context_dim)
        self.attention_heads = int(attention_heads)
        self.attention_dropout = float(attention_dropout)

        if self.observation_dim <= 0:
            raise ValueError("observation_dim must be positive.")
        if self.skill_dim <= 0:
            raise ValueError("skill_dim must be positive.")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if self.context_dim <= 0:
            raise ValueError("context_dim must be positive.")
        if self.attention_heads <= 0:
            raise ValueError("attention_heads must be positive.")
        if self.context_dim % self.attention_heads != 0:
            raise ValueError(
                "context_dim must be divisible by attention_heads: "
                f"{self.context_dim} vs {self.attention_heads}"
            )
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0, 1).")

        token_dim = self.observation_dim + self.skill_dim
        self.context_projection = nn.Linear(token_dim, self.context_dim)
        self.context_input_norm = nn.LayerNorm(self.context_dim)
        self.agent_attention = nn.MultiheadAttention(
            embed_dim=self.context_dim,
            num_heads=self.attention_heads,
            dropout=self.attention_dropout,
            batch_first=True,
        )
        self.context_output_norm = nn.LayerNorm(self.context_dim)

        self.network = nn.Sequential(
            nn.Linear(
                self.skill_dim + self.observation_dim + self.context_dim,
                self.hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.skill_dim),
        )

        # Exact identity at initialization. The attention block may produce a
        # non-zero context, but the zero final residual layer guarantees:
        # adapted_common == raw_common before training.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def _validate_inputs(
        self,
        common_skills: torch.Tensor,
        observation_features: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if common_skills.ndim < 2 or observation_features.ndim < 2:
            raise ValueError(
                "common_skills and observation_features need an agent axis."
            )
        if common_skills.shape[:-1] != observation_features.shape[:-1]:
            raise ValueError(
                "Common-skill/feature leading shapes differ: "
                f"{tuple(common_skills.shape)} vs "
                f"{tuple(observation_features.shape)}"
            )
        if common_skills.shape[-1] != self.skill_dim:
            raise ValueError(
                f"Expected skill_dim={self.skill_dim}, "
                f"got {common_skills.shape[-1]}"
            )
        if observation_features.shape[-1] != self.observation_dim:
            raise ValueError(
                f"Expected observation_dim={self.observation_dim}, "
                f"got {observation_features.shape[-1]}"
            )

        if valid_mask is None:
            valid_mask = torch.ones(
                common_skills.shape[:-1],
                dtype=torch.bool,
                device=common_skills.device,
            )
        else:
            if valid_mask.shape != common_skills.shape[:-1]:
                raise ValueError(
                    "valid_mask does not match common-skill leading shape: "
                    f"{tuple(valid_mask.shape)} vs "
                    f"{tuple(common_skills.shape[:-1])}"
                )
            valid_mask = valid_mask.bool()

        # Offline sequence batches can contain fully padded timesteps.
        # compute_context() handles those safely.
        return valid_mask

    def compute_context(
        self,
        common_skills: torch.Tensor,
        observation_features: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return g_i with the same leading shape as the agent features."""
        valid_mask = self._validate_inputs(
            common_skills, observation_features, valid_mask
        )

        token_input = torch.cat(
            (observation_features, common_skills),
            dim=-1,
        )
        tokens = self.context_input_norm(
            self.context_projection(token_input)
        )

        agent_count = int(tokens.shape[-2])
        leading_shape = tokens.shape[:-2]
        flat_tokens = tokens.reshape(
            -1, agent_count, self.context_dim
        )
        flat_valid = valid_mask.reshape(-1, agent_count)

        # Fully padded timesteps would give MHA an all-masked key row.
        # Temporarily expose agent 0 only for the attention computation.
        safe_valid = flat_valid.clone()
        empty_rows = safe_valid.sum(dim=-1) == 0
        if bool(empty_rows.any()):
            safe_valid[empty_rows, 0] = True

        flat_context, _ = self.agent_attention(
            flat_tokens,
            flat_tokens,
            flat_tokens,
            key_padding_mask=~safe_valid,
            need_weights=False,
        )
        flat_context = self.context_output_norm(flat_context)

        # Use the ORIGINAL mask. Fully padded rows therefore become exactly 0.
        flat_context = (
            flat_context
            * flat_valid.unsqueeze(-1).to(flat_context.dtype)
        )

        return flat_context.reshape(
            *leading_shape,
            agent_count,
            self.context_dim,
        )

    def forward(
        self,
        common_skills: torch.Tensor,
        observation_features: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid_mask = self._validate_inputs(
            common_skills, observation_features, valid_mask
        )
        context = self.compute_context(
            common_skills,
            observation_features,
            valid_mask,
        )

        adapter_input = torch.cat(
            (common_skills, observation_features, context),
            dim=-1,
        )
        delta = self.network(adapter_input)

        numeric_mask = (
            valid_mask.unsqueeze(-1).to(delta.dtype)
        )
        delta = delta * numeric_mask
        return common_skills + delta, delta


class JointHeterogeneousCrossAgentAdapterHiSSD(
    JointHeterogeneousAdapterHiSSD
):
    """Joint heterogeneous HiSSD with explicit cross-agent skill adaptation.

    Everything except the common-skill adapter is inherited from the existing
    local-adapter model. Therefore this is an intentionally narrow ablation:

        Local:
            c'_i = c_i + A([c_i, h_i])

        Cross-agent:
            g_i = SelfAttention([h_1,c_1], ..., [h_N,c_N])_i
            c'_i = c_i + A([c_i, h_i, g_i])

    The inherited controller, planner, value path, role-specific observation
    encoders, role-specific action decoders, recurrent common-skill history,
    target updates, BC initialization support, and inference API remain the same.
    """

    def __init__(
        self,
        *args: Any,
        cross_agent_context_dim: int = 64,
        cross_agent_attention_heads: int = 4,
        cross_agent_attention_dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.cross_agent_context_dim = int(
            cross_agent_context_dim
        )
        self.cross_agent_attention_heads = int(
            cross_agent_attention_heads
        )
        self.cross_agent_attention_dropout = float(
            cross_agent_attention_dropout
        )

        # Replace ONLY the local residual adapter. All inherited call sites
        # already use the signature (common_skills, features, valid_mask), so
        # controller, planner, offline forward and online inference all obtain
        # the cross-agent contextualized common skill automatically.
        self.common_skill_adapter = CrossAgentCommonSkillAdapter(
            observation_dim=self.shared_observation_dim,
            skill_dim=self.skill_dim,
            hidden_dim=self.common_skill_adapter_hidden_dim,
            context_dim=self.cross_agent_context_dim,
            attention_heads=self.cross_agent_attention_heads,
            attention_dropout=self.cross_agent_attention_dropout,
        )

        # Keep checkpoint reconstruction self-contained.
        self.model_config.update(
            {
                "cross_agent_context_dim": self.cross_agent_context_dim,
                "cross_agent_attention_heads": (
                    self.cross_agent_attention_heads
                ),
                "cross_agent_attention_dropout": (
                    self.cross_agent_attention_dropout
                ),
            }
        )

    @torch.no_grad()
    def cross_agent_context_statistics(
        self,
        observation_features: torch.Tensor,
        raw_common_skills: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, float]:
        context = self.common_skill_adapter.compute_context(
            raw_common_skills,
            observation_features,
            valid_mask,
        )
        numeric = valid_mask.bool().unsqueeze(-1).to(context.dtype)
        denominator = (
            numeric.sum() * context.shape[-1]
        ).clamp_min(1.0)
        rms = torch.sqrt(
            (context.square() * numeric).sum() / denominator
        )
        return {
            "cross_agent_context_rms": float(rms.item()),
        }


def load_joint_heterogeneous_cross_agent_adapter_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[
    JointHeterogeneousCrossAgentAdapterHiSSD,
    dict[str, Any],
]:
    """Load a native cross-agent-adapter checkpoint."""
    checkpoint_path = Path(
        checkpoint_path
    ).expanduser().resolve()
    payload = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if (
        "model_config" not in payload
        or "model_state_dict" not in payload
    ):
        raise ValueError(
            "Checkpoint lacks model_config/model_state_dict: "
            f"{checkpoint_path}"
        )

    model = JointHeterogeneousCrossAgentAdapterHiSSD(
        **payload["model_config"]
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    return model, payload
