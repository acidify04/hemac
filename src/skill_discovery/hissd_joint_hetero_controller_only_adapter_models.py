"""Controller-only adapter variant of joint heterogeneous HiSSD.

The original joint heterogeneous HiSSD+Adapter model applies the residual
common-skill adapter to the controller and the current trainer also applies it
explicitly in the planner objective.

This variant preserves the existing model architecture and parameterization,
but restores the *base HiSSD skill API* to raw common skills:

    h_i -> CommonSkillEncoder -> raw c_i

Planner-side code should consume raw c_i.

The existing controller path is inherited unchanged:

    raw c_i + h_i -> common_skill_adapter -> adapted c'_i
    (h_i, c'_i, z_i) -> role-specific action decoder -> action

Therefore no original HiSSD module is replaced and no new parameter is added.
Only the scope of the existing adapter is restricted to action execution.

IMPORTANT:
The matched trainer must NOT explicitly call common_skill_adapter inside the
planner objective.  It should feed raw_common directly to forward_predictor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .hissd_joint_hetero_adapter_models import (
    JointHeterogeneousAdapterHiSSD,
)
from .hissd_variable_models import VariableAgentHeMACHISSD


class JointHeterogeneousControllerOnlyAdapterHiSSD(
    JointHeterogeneousAdapterHiSSD
):
    """Same architecture as the existing Adapter model, controller-only scope.

    Inherited unchanged:
      - role-specific observation encoders
      - shared CommonSkillEncoder
      - shared task-skill/value/forward-predictor stack
      - ONE shared residual common-skill adapter
      - role-specific action decoders
      - offline forward_joint()
      - recurrent inference_step()
      - BC initialization utilities
      - adapter diagnostics

    Semantic difference:
      ``infer_skills()`` again returns the RAW common skill, matching base
      HiSSD.  The inherited controller path continues to call
      ``infer_skills_with_adapter()`` / ``common_skill_adapter`` and therefore
      acts using the adapted common skill.
    """

    adapter_scope = "controller_only"

    def infer_skills(
        self,
        observation_features: torch.Tensor,
        valid_mask: torch.Tensor,
        task_observation_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return RAW common skills for base/planner-side HiSSD consumers.

        This deliberately bypasses
        JointHeterogeneousAdapterHiSSD.infer_skills(), because that method
        changes the base API to return adapted common skills.
        """
        return VariableAgentHeMACHISSD.infer_skills(
            self,
            observation_features,
            valid_mask,
            task_observation_features=task_observation_features,
        )

    def raw_common_skills(
        self,
        observation_features: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Explicit helper for planner-side code."""
        return self.common_skill_encoder(
            observation_features,
            valid_mask,
        )


def load_joint_heterogeneous_controller_only_adapter_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[
    JointHeterogeneousControllerOnlyAdapterHiSSD,
    dict[str, Any],
]:
    """Load a checkpoint trained with the controller-only adapter trainer."""
    checkpoint_path = (
        Path(checkpoint_path).expanduser().resolve()
    )

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

    model = (
        JointHeterogeneousControllerOnlyAdapterHiSSD(
            **payload["model_config"]
        ).to(device)
    )
    model.load_state_dict(
        payload["model_state_dict"]
    )

    return model, payload
