"""Variable-population extension for HeMAC HiSSD.

This file intentionally leaves ``hissd_models.py`` unchanged.

Changes relative to HeMACHISSD:
1. Replace the fixed-width CentralValueMixer with a permutation-invariant
   set mixer whose parameters do not depend on the number of agents.
2. Allow recurrent inference state to be created for a runtime agent count.
3. Allow inference_step to accept a different agent count on each episode.

All actor/skill/planner modules remain parameter-shared and unchanged.
"""
from __future__ import annotations

import copy
import threading
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .hissd_models import HeMACHISSD


class VariableAgentValueMixer(nn.Module):
    """Permutation-invariant centralized value mixer for arbitrary team size.

    Each scalar individual value is embedded independently. The valid-agent
    embeddings are summarized by both a sum and a mean, and the active count is
    supplied explicitly. Parameter shapes therefore do not depend on A.

    Shapes:
        individual_values: [..., A, 1]
        central_features:  [..., C]
        valid_mask:        [..., A]
        return:            [..., 1]
    """

    def __init__(
        self,
        central_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.central_dim = int(central_dim)
        self.hidden_dim = int(hidden_dim)

        self.agent_encoder = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.network = nn.Sequential(
            nn.Linear(
                self.central_dim + 2 * self.hidden_dim + 1,
                self.hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self,
        individual_values: torch.Tensor,
        central_features: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if individual_values.ndim < 2 or individual_values.shape[-1] != 1:
            raise ValueError(
                "individual_values must end in [A,1], got "
                f"{tuple(individual_values.shape)}"
            )
        if valid_mask.shape != individual_values.shape[:-1]:
            raise ValueError(
                "valid_mask must match individual value leading shape: "
                f"{tuple(valid_mask.shape)} vs "
                f"{tuple(individual_values.shape[:-1])}"
            )
        if central_features.shape[:-1] != individual_values.shape[:-2]:
            raise ValueError(
                "central feature leading shape must match batch/time axes: "
                f"{tuple(central_features.shape)} vs "
                f"{tuple(individual_values.shape)}"
            )
        if central_features.shape[-1] != self.central_dim:
            raise ValueError(
                f"Expected central dim {self.central_dim}, "
                f"got {central_features.shape[-1]}"
            )

        mask = valid_mask.bool().unsqueeze(-1)
        numeric_mask = mask.to(individual_values.dtype)
        encoded = self.agent_encoder(individual_values) * numeric_mask

        summed = encoded.sum(dim=-2)
        count = numeric_mask.sum(dim=-2).clamp_min(1.0)
        mean = summed / count
        log_count = torch.log1p(count)

        mixed_input = torch.cat(
            (central_features, summed, mean, log_count),
            dim=-1,
        )
        return self.network(mixed_input)


class VariableAgentHeMACHISSD(HeMACHISSD):
    """HeMACHISSD whose train/test population size can change at runtime."""

    def __init__(
        self,
        global_map_channels: int,
        local_map_channels: int,
        central_map_channels: int,
        *,
        agent_count: int = 3,
        variable_mixer_hidden_dim: int = 128,
        variable_agent_model: bool = True,
        **kwargs: Any,
    ) -> None:
        # ``agent_count`` is retained only as a default/reference cardinality
        # for backward compatibility with old checkpoints and call sites.
        super().__init__(
            global_map_channels,
            local_map_channels,
            central_map_channels,
            agent_count=agent_count,
            **kwargs,
        )
        self.reference_agent_count = int(agent_count)
        self.variable_mixer_hidden_dim = int(variable_mixer_hidden_dim)
        self.variable_agent_model = bool(variable_agent_model)

        # CentralStateEncoder outputs self.hidden_dim features.
        self.value_mixer = VariableAgentValueMixer(
            self.hidden_dim,
            self.variable_mixer_hidden_dim,
        )
        self.target_value_mixer = copy.deepcopy(self.value_mixer)
        self._freeze_target_modules()

        # Keep enough metadata to reconstruct this subclass from a checkpoint.
        self.model_config["agent_count"] = self.reference_agent_count
        self.model_config["variable_agent_model"] = True
        self.model_config["variable_mixer_hidden_dim"] = (
            self.variable_mixer_hidden_dim
        )

        # Base inference checks self.agent_count. We preserve that implementation
        # exactly by setting the attribute only inside a protected call. This
        # avoids duplicating the long recurrent inference method.
        self._agent_count_lock = threading.RLock()

    def train(self, mode: bool = True):
        # Preserve EMA target semantics even when the outer trainer calls
        # model.train().
        super().train(mode)
        for module in self.target_modules():
            module.eval()
        return self

    def initial_inference_state(
        self,
        batch_size: int = 1,
        *,
        agent_count: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor]:
        runtime_count = (
            self.reference_agent_count
            if agent_count is None
            else int(agent_count)
        )
        if runtime_count <= 0:
            raise ValueError("agent_count must be positive")

        with self._agent_count_lock:
            original = self.agent_count
            self.agent_count = runtime_count
            try:
                return super().initial_inference_state(
                    batch_size,
                    device=device,
                    dtype=dtype,
                )
            finally:
                self.agent_count = original

    def inference_step(
        self,
        observations: dict[str, torch.Tensor],
        valid_mask: torch.Tensor,
        state: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if valid_mask.ndim != 2:
            raise ValueError(
                f"Online valid_mask must be [B,A], got {valid_mask.shape}"
            )
        runtime_count = int(valid_mask.shape[1])
        if runtime_count <= 0:
            raise ValueError("Runtime agent count must be positive")

        # Existing online runners often create the recurrent state before they
        # know the runtime population. Repair it automatically on the first step.
        common_history = state.get("common_history")
        if (
            common_history is None
            or common_history.ndim != 3
            or common_history.shape[1] != runtime_count
        ):
            state = self.initial_inference_state(
                batch_size=int(valid_mask.shape[0]),
                agent_count=runtime_count,
                device=valid_mask.device,
            )

        with self._agent_count_lock:
            original = self.agent_count
            self.agent_count = runtime_count
            try:
                return super().inference_step(
                    observations,
                    valid_mask,
                    state,
                )
            finally:
                self.agent_count = original

    @classmethod
    def from_base_model(
        cls,
        base_model: HeMACHISSD,
        *,
        variable_mixer_hidden_dim: int = 128,
    ) -> "VariableAgentHeMACHISSD":
        """Upgrade an existing HiSSD model without changing its actor policy.

        All compatible parameters are copied. The old fixed-width value mixer
        cannot be mapped exactly to a cardinality-invariant mixer, so both the
        online and target mixers are freshly initialized and should be retrained.
        """
        config = dict(base_model.config())
        config.pop("variable_agent_model", None)
        config.pop("variable_mixer_hidden_dim", None)

        model = cls(
            **config,
            variable_mixer_hidden_dim=variable_mixer_hidden_dim,
        )

        source_state = base_model.state_dict()
        copied_state = {
            key: value
            for key, value in source_state.items()
            if not key.startswith("value_mixer.")
            and not key.startswith("target_value_mixer.")
        }
        incompatible = model.load_state_dict(copied_state, strict=False)

        allowed_missing = (
            "value_mixer.",
            "target_value_mixer.",
        )
        bad_missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith(allowed_missing)
        ]
        if bad_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Unexpected checkpoint conversion mismatch: "
                f"missing={bad_missing}, "
                f"unexpected={list(incompatible.unexpected_keys)}"
            )

        reference = next(base_model.parameters())
        model = model.to(
            device=reference.device,
            dtype=reference.dtype,
        )
        model.train(base_model.training)
        model._freeze_target_modules()
        model.conversion_report = {
            "reference_agent_count": int(base_model.agent_count),
            "fresh_modules": ["value_mixer", "target_value_mixer"],
            "copied_parameter_tensors": len(copied_state),
        }
        return model

    def config(self) -> dict[str, Any]:
        config = super().config()
        config["agent_count"] = self.reference_agent_count
        config["variable_agent_model"] = True
        config["variable_mixer_hidden_dim"] = self.variable_mixer_hidden_dim
        return config


def load_variable_hissd_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[VariableAgentHeMACHISSD, dict[str, Any]]:
    """Load either a native variable checkpoint or upgrade an old HiSSD one."""
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
    is_variable = bool(config.get("variable_agent_model", False))
    if is_variable:
        model = VariableAgentHeMACHISSD(**config).to(device)
        model.load_state_dict(payload["model_state_dict"])
        return model, payload

    base = HeMACHISSD(**config).to(device)
    base.load_state_dict(payload["model_state_dict"])
    model = VariableAgentHeMACHISSD.from_base_model(base)
    return model, payload
