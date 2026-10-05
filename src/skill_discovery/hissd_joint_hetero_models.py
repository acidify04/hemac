"""Joint heterogeneous HiSSD baseline WITHOUT the common-skill adapter.

This is the direct ablation counterpart of
``JointHeterogeneousAdapterHiSSD``:

    role-specific observation encoder E_r(o_i) -> h_i
    shared CommonSkillEncoder C(h_i)            -> c_i
    shared task-skill/value/planner stack
    role-specific action decoder pi_r(h_i, c_i, z_i)

There is no adapter module and no adapter parameter in the state dict.
Drone and Observer are trained jointly in one synchronized episode batch.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import torch

from .hissd_joint_hetero_adapter_models import (
    JointHeterogeneousAdapterHiSSD,
    ROLE_ORDER,
)
from .hissd_variable_models import VariableAgentHeMACHISSD


class JointHeterogeneousHiSSD(JointHeterogeneousAdapterHiSSD):
    """Joint heterogeneous HiSSD with shared common skill and NO adapter."""

    def __init__(
        self,
        role_configs: Mapping[str, Mapping[str, Any]],
        central_map_channels: int,
        *,
        reference_observer_count: int = 1,
        reference_drone_count: int = 3,
        central_map_size: tuple[int, int] = (20, 20),
        hidden_dim: int = 64,
        skill_dim: int = 64,
        transformer_heads: int = 1,
        variable_mixer_hidden_dim: int = 128,
        contrastive_from_action_skill: bool = False,
        task_context_pooling: bool = False,
        task_descriptor_dim: int = 0,
        task_prior_count: int = 0,
        task_dropout: float = 0.0,
        task_feature_deltas: bool = False,
        learned_task_classifier: bool = False,
        normalize_task_context: bool = True,
        task_running_statistics: bool = False,
        direct_task_summary: bool = False,
        task_action_residual: bool = False,
        skill_structure: str = "split",
    ) -> None:
        # Reuse the already validated heterogeneous I/O construction, then
        # remove the adapter completely. A temporary one-parameter-width adapter
        # is created only during construction and is deleted before use/save.
        super().__init__(
            role_configs=role_configs,
            central_map_channels=central_map_channels,
            reference_observer_count=reference_observer_count,
            reference_drone_count=reference_drone_count,
            central_map_size=central_map_size,
            hidden_dim=hidden_dim,
            skill_dim=skill_dim,
            transformer_heads=transformer_heads,
            common_skill_adapter_hidden_dim=1,
            variable_mixer_hidden_dim=variable_mixer_hidden_dim,
            contrastive_from_action_skill=contrastive_from_action_skill,
            task_context_pooling=task_context_pooling,
            task_descriptor_dim=task_descriptor_dim,
            task_prior_count=task_prior_count,
            task_dropout=task_dropout,
            task_feature_deltas=task_feature_deltas,
            learned_task_classifier=learned_task_classifier,
            normalize_task_context=normalize_task_context,
            task_running_statistics=task_running_statistics,
            direct_task_summary=direct_task_summary,
            task_action_residual=task_action_residual,
            skill_structure=skill_structure,
        )
        del self.common_skill_adapter
        del self.common_skill_adapter_hidden_dim
        self.model_config.pop("common_skill_adapter_hidden_dim", None)

    def infer_joint_skills(
        self,
        observation_features: torch.Tensor,
        valid_mask: torch.Tensor,
        task_observation_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Infer the original HiSSD common/task skills with no adaptation."""
        common, task_skill, contrastive = VariableAgentHeMACHISSD.infer_skills(
            self,
            observation_features,
            valid_mask,
            task_observation_features=task_observation_features,
        )
        return {
            "common_skills": common,
            "task_skills": task_skill,
            "contrastive_skills": contrastive,
        }

    def infer_skills(
        self,
        observation_features: torch.Tensor,
        valid_mask: torch.Tensor,
        task_observation_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.infer_joint_skills(
            observation_features,
            valid_mask,
            task_observation_features=task_observation_features,
        )
        return (
            outputs["common_skills"],
            outputs["task_skills"],
            outputs["contrastive_skills"],
        )

    def forward_joint(
        self,
        observations_by_role: Mapping[str, Mapping[str, torch.Tensor]],
        valid_masks_by_role: Mapping[str, torch.Tensor],
    ) -> dict[str, Any]:
        features, valid_mask, role_slices = self.encode_joint_observations(
            observations_by_role,
            valid_masks_by_role,
        )
        skills = self.infer_joint_skills(features, valid_mask)
        actions = self.decode_joint_actions(
            features,
            skills["common_skills"],
            skills["task_skills"],
            role_slices,
        )
        return {
            "actions": actions,
            "observation_features": features,
            "valid_mask": valid_mask,
            "role_slices": role_slices,
            **skills,
        }

    def inference_step(
        self,
        observations_by_role: Mapping[str, Mapping[str, torch.Tensor]],
        valid_masks_by_role: Mapping[str, torch.Tensor],
        state: dict[str, torch.Tensor],
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        """One recurrent heterogeneous step using raw HiSSD common skill c_i."""
        features, valid_mask, role_slices = self.encode_joint_observations(
            observations_by_role,
            valid_masks_by_role,
        )
        if features.ndim != 3 or valid_mask.ndim != 2:
            raise ValueError(
                "Online joint observations must encode to [B,A,D] and [B,A]."
            )

        batch_size, total_count, _ = features.shape
        history = state.get("common_history")
        if (
            history is None
            or history.ndim != 3
            or history.shape[:2] != (batch_size, total_count)
        ):
            state = VariableAgentHeMACHISSD.initial_inference_state(
                self,
                batch_size=batch_size,
                agent_count=total_count,
                device=features.device,
                dtype=features.dtype,
            )

        common, next_common_history = self.common_skill_encoder.forward_step(
            features,
            valid_mask,
            state["common_history"],
        )

        (
            task_skill,
            contrastive,
            next_task_history,
            next_task_context,
            next_task_previous_pooled,
            next_task_previous_observation,
            next_task_previous_observation_valid,
            next_task_running_sum,
            next_task_running_square_sum,
            next_task_running_abs_delta_sum,
            next_task_running_count,
            next_task_running_delta_count,
        ) = self.task_skill_encoder.forward_step(
            features,
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

        actions = self.decode_joint_actions(
            features,
            common,
            task_skill,
            role_slices,
        )

        outputs: dict[str, Any] = {
            "actions": actions,
            "observation_features": features,
            "common_skills": common,
            "task_skills": task_skill,
            "contrastive_skills": contrastive,
            "conditioning_skills": self.conditioning_skill(common, task_skill),
            "valid_mask": valid_mask,
            "role_slices": role_slices,
        }

        if self.task_descriptor_head is not None:
            numeric_mask = valid_mask.bool().unsqueeze(-1).to(contrastive.dtype)
            pooled_context = (
                contrastive * numeric_mask
            ).sum(dim=1) / numeric_mask.sum(dim=1).clamp_min(1.0)
            outputs["task_descriptor"] = self.task_descriptor_head(
                self.task_descriptor_dropout(pooled_context)
            )

        next_state = {
            "common_history": next_common_history,
            "task_history": next_task_history,
            "task_context": next_task_context,
            "task_previous_pooled": next_task_previous_pooled,
            "task_previous_observation": next_task_previous_observation,
            "task_previous_observation_valid": next_task_previous_observation_valid,
            "task_running_sum": next_task_running_sum,
            "task_running_square_sum": next_task_running_square_sum,
            "task_running_abs_delta_sum": next_task_running_abs_delta_sum,
            "task_running_count": next_task_running_count,
            "task_running_delta_count": next_task_running_delta_count,
        }
        return outputs, next_state

    def config(self) -> dict[str, Any]:
        return copy.deepcopy(self.model_config)


def load_joint_heterogeneous_hissd_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[JointHeterogeneousHiSSD, dict[str, Any]]:
    """Load a native joint heterogeneous HiSSD checkpoint without adapter."""
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
    model = JointHeterogeneousHiSSD(**payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state_dict"])
    return model, payload
