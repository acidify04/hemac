"""Joint heterogeneous-agent HiSSD with one agent-conditioned common-skill adapter.

Design goal
-----------
Handle Drone and Observer in ONE HiSSD model while keeping their heterogeneous
observation/action interfaces. The only new skill module is a shared residual
adapter:

    role-specific observation encoder E_r(o_i) -> h_i
    shared CommonSkillEncoder C(h_i)          -> raw common skill c_i
    shared adapter A([c_i, h_i])              -> delta_i
    adapted common skill c'_i = c_i + delta_i
    role-specific action decoder pi_r(h_i, c'_i, z_i)

The same adapter parameters are used by every agent and every role. Agent-specific
adaptation arises only from each agent's encoded observation feature h_i.

The adapted common skill is the common skill used downstream by the action
decoders and by any trainer that consumes ``infer_skills`` / ``forward_joint``
outputs (e.g. the forward predictor). Raw common skills are exposed only for
diagnostics.

This first joint prototype intentionally keeps the existing HiSSD task-skill,
value, central-state, and forward-predictor modules shared. Separate task
observation encoders are deliberately disabled so that Drone/Observer features
share one latent dimensionality before the shared HiSSD modules.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .hissd_models import ContinuousActionDecoder
from .hissd_variable_models import VariableAgentHeMACHISSD
from .models import DroneBehaviorCloningPolicy, DroneObservationEncoder


ROLE_ORDER = ("observer", "drone")
SUPPORTED_BC_TYPES = {
    "drone": {"drone_behavior_cloning"},
    "observer": {"observer_behavior_cloning"},
}


def _normalise_role_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one BC-style role config into serializable Python values."""
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
        "global_map_size": tuple(int(v) for v in config["global_map_size"]),
        "local_map_size": tuple(int(v) for v in config["local_map_size"]),
        "action_history_shape": tuple(
            int(v) for v in config["action_history_shape"]
        ),
        "hidden_sizes": tuple(int(v) for v in config["hidden_sizes"]),
        "activation": str(config["activation"]),
        "action_dim": int(config["action_dim"]),
    }


class AgentConditionedCommonSkillAdapter(nn.Module):
    """Shared residual adapter: c'_i = c_i + A([c_i, h_i]).

    Leading dimensions can be [B,T,A] for offline training or [B,A] for online
    inference. The final dimension is the only semantically constrained axis.
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

        self.network = nn.Sequential(
            nn.Linear(self.observation_dim + self.skill_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.skill_dim),
        )

        # Exact identity at construction. Before training, adapted == raw.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        common_skills: torch.Tensor,
        observation_features: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

        adapter_input = torch.cat((common_skills, observation_features), dim=-1)
        delta = self.network(adapter_input)

        if valid_mask is not None:
            if valid_mask.shape != common_skills.shape[:-1]:
                raise ValueError(
                    "valid_mask does not match common-skill leading shape: "
                    f"{tuple(valid_mask.shape)} vs "
                    f"{tuple(common_skills.shape[:-1])}"
                )
            numeric_mask = valid_mask.bool().unsqueeze(-1).to(delta.dtype)
            delta = delta * numeric_mask

        return common_skills + delta, delta


class JointHeterogeneousAdapterHiSSD(VariableAgentHeMACHISSD):
    """One variable-population HiSSD for Drone + Observer.

    Heterogeneity:
      * role-specific observation encoders,
      * role-specific action decoders,
      * one shared common-skill encoder,
      * one shared task-skill encoder,
      * one shared agent-conditioned common-skill adapter,
      * one shared variable-cardinality value pathway / forward predictor.

    Canonical joint agent order is ``observer`` then ``drone`` to match HeMAC's
    AEC ordering. No agent ID or population-size embedding is used.
    """

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
        common_skill_adapter_hidden_dim: int = 128,
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
        configs = {
            role: _normalise_role_config(role_configs[role])
            for role in ROLE_ORDER
            if role in role_configs
        }
        if set(configs) != set(ROLE_ORDER):
            raise ValueError(
                "role_configs must contain exactly 'observer' and 'drone'."
            )
        if skill_structure != "split":
            raise ValueError(
                "JointHeterogeneousAdapterHiSSD first prototype expects "
                "skill_structure='split'."
            )

        # Both heterogeneous role encoders can have different input spaces, but
        # must terminate in the same latent width before shared HiSSD modules.
        role_output_dims = {
            role: int(config["hidden_sizes"][-1])
            if config["hidden_sizes"]
            else None
            for role, config in configs.items()
        }
        if None in role_output_dims.values():
            raise ValueError(
                "This prototype requires non-empty role hidden_sizes so both "
                "roles have an explicit shared feature dimension."
            )
        if len(set(role_output_dims.values())) != 1:
            raise ValueError(
                "Drone/Observer encoders must emit the same feature dimension "
                f"for the shared skill encoder, got {role_output_dims}."
            )

        observer_cfg = configs["observer"]
        reference_total = int(reference_observer_count) + int(
            reference_drone_count
        )
        if reference_total <= 0:
            raise ValueError("Reference population must contain at least one agent.")

        # Build the shared HiSSD stack once using Observer dimensions only as a
        # temporary constructor interface. Role-specific I/O modules are replaced
        # immediately below; all shared skill/value/planner modules are retained.
        super().__init__(
            global_map_channels=observer_cfg["global_map_channels"],
            local_map_channels=observer_cfg["local_map_channels"],
            central_map_channels=int(central_map_channels),
            agent_count=reference_total,
            action_dim=observer_cfg["action_dim"],
            global_map_size=observer_cfg["global_map_size"],
            local_map_size=observer_cfg["local_map_size"],
            central_map_size=central_map_size,
            action_history_shape=observer_cfg["action_history_shape"],
            observation_hidden_sizes=observer_cfg["hidden_sizes"],
            hidden_dim=hidden_dim,
            skill_dim=skill_dim,
            transformer_heads=transformer_heads,
            activation=observer_cfg["activation"],
            contrastive_from_action_skill=contrastive_from_action_skill,
            task_context_pooling=task_context_pooling,
            task_descriptor_dim=task_descriptor_dim,
            task_prior_count=task_prior_count,
            task_dropout=task_dropout,
            task_feature_deltas=task_feature_deltas,
            separate_task_observation_encoder=False,
            learned_task_classifier=learned_task_classifier,
            task_spatial_statistics=False,
            normalize_task_context=normalize_task_context,
            task_running_statistics=task_running_statistics,
            direct_task_summary=direct_task_summary,
            task_action_residual=task_action_residual,
            skill_structure=skill_structure,
            variable_mixer_hidden_dim=variable_mixer_hidden_dim,
            variable_agent_model=True,
        )

        self.role_configs = configs
        self.reference_observer_count = int(reference_observer_count)
        self.reference_drone_count = int(reference_drone_count)
        self.common_skill_adapter_hidden_dim = int(
            common_skill_adapter_hidden_dim
        )
        self.shared_observation_dim = int(
            next(iter(role_output_dims.values()))
        )

        # Replace the homogeneous encoder by role-specific encoders.
        self.observation_encoder = nn.ModuleDict(
            {
                role: DroneObservationEncoder(
                    config["global_map_channels"],
                    config["local_map_channels"],
                    global_map_size=config["global_map_size"],
                    local_map_size=config["local_map_size"],
                    action_history_shape=config["action_history_shape"],
                    hidden_sizes=config["hidden_sizes"],
                    activation=config["activation"],
                )
                for role, config in self.role_configs.items()
            }
        )
        for role, encoder in self.observation_encoder.items():
            if int(encoder.output_dim) != self.shared_observation_dim:
                raise RuntimeError(
                    f"{role} encoder emitted {encoder.output_dim}, expected "
                    f"{self.shared_observation_dim}."
                )

        self.target_observation_encoder = copy.deepcopy(
            self.observation_encoder
        )

        # Replace the homogeneous action decoder by role-specific decoders.
        self.action_decoder = nn.ModuleDict(
            {
                role: ContinuousActionDecoder(
                    observation_dim=self.shared_observation_dim,
                    skill_dim=self.skill_dim,
                    action_dim=config["action_dim"],
                    hidden_dim=self.hidden_dim,
                    heads=transformer_heads,
                    task_action_residual=task_action_residual,
                )
                for role, config in self.role_configs.items()
            }
        )

        # The ONLY new coordination module.
        self.common_skill_adapter = AgentConditionedCommonSkillAdapter(
            observation_dim=self.shared_observation_dim,
            skill_dim=self.skill_dim,
            hidden_dim=self.common_skill_adapter_hidden_dim,
        )

        self.action_dims = {
            role: config["action_dim"]
            for role, config in self.role_configs.items()
        }

        # Re-freeze target modules after replacing target_observation_encoder.
        self._freeze_target_modules()

        # Replace ambiguous homogeneous reconstruction metadata.
        self.model_config = {
            "role_configs": copy.deepcopy(self.role_configs),
            "central_map_channels": int(central_map_channels),
            "reference_observer_count": self.reference_observer_count,
            "reference_drone_count": self.reference_drone_count,
            "central_map_size": tuple(int(v) for v in central_map_size),
            "hidden_dim": int(hidden_dim),
            "skill_dim": int(skill_dim),
            "transformer_heads": int(transformer_heads),
            "common_skill_adapter_hidden_dim": self.common_skill_adapter_hidden_dim,
            "variable_mixer_hidden_dim": int(variable_mixer_hidden_dim),
            "contrastive_from_action_skill": bool(
                contrastive_from_action_skill
            ),
            "task_context_pooling": bool(task_context_pooling),
            "task_descriptor_dim": int(task_descriptor_dim),
            "task_prior_count": int(task_prior_count),
            "task_dropout": float(task_dropout),
            "task_feature_deltas": bool(task_feature_deltas),
            "learned_task_classifier": bool(learned_task_classifier),
            "normalize_task_context": bool(normalize_task_context),
            "task_running_statistics": bool(task_running_statistics),
            "direct_task_summary": bool(direct_task_summary),
            "task_action_residual": bool(task_action_residual),
            "skill_structure": str(skill_structure),
        }

    # ------------------------------------------------------------------
    # Role-specific observation interface
    # ------------------------------------------------------------------
    def encode_role_observations(
        self,
        role: str,
        observations: Mapping[str, torch.Tensor],
        *,
        target: bool = False,
    ) -> torch.Tensor:
        if role not in ROLE_ORDER:
            raise ValueError(f"Unknown role: {role!r}")
        encoders = (
            self.target_observation_encoder
            if target
            else self.observation_encoder
        )
        encoder = encoders[role]
        return encoder(
            observations["global_map"],
            observations["local_map"],
            observations["action_history"],
        )

    @staticmethod
    def _validate_role_feature_shapes(
        features_by_role: Mapping[str, torch.Tensor],
        masks_by_role: Mapping[str, torch.Tensor],
    ) -> None:
        reference_prefix = None
        feature_dim = None
        for role in ROLE_ORDER:
            features = features_by_role[role]
            mask = masks_by_role[role]
            if features.shape[:-1] != mask.shape:
                raise ValueError(
                    f"{role} feature/mask shapes differ: "
                    f"{tuple(features.shape)} vs {tuple(mask.shape)}"
                )
            if reference_prefix is None:
                reference_prefix = features.shape[:-2]
                feature_dim = features.shape[-1]
            else:
                if features.shape[:-2] != reference_prefix:
                    raise ValueError(
                        "Drone/Observer batch/time dimensions differ: "
                        f"{features_by_role['observer'].shape} vs "
                        f"{features_by_role['drone'].shape}"
                    )
                if features.shape[-1] != feature_dim:
                    raise ValueError(
                        "Drone/Observer feature dimensions differ after encoding."
                    )

    def encode_joint_observations(
        self,
        observations_by_role: Mapping[str, Mapping[str, torch.Tensor]],
        valid_masks_by_role: Mapping[str, torch.Tensor],
        *,
        target: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, slice]]:
        """Encode roles separately, then concatenate along the agent axis.

        Offline:
            role features [B,T,A_role,D] -> joint [B,T,A_total,D]
        Online:
            role features [B,A_role,D]   -> joint [B,A_total,D]
        """
        features_by_role = {
            role: self.encode_role_observations(
                role, observations_by_role[role], target=target
            )
            for role in ROLE_ORDER
        }
        self._validate_role_feature_shapes(
            features_by_role, valid_masks_by_role
        )

        role_slices: dict[str, slice] = {}
        offset = 0
        ordered_features = []
        ordered_masks = []
        for role in ROLE_ORDER:
            features = features_by_role[role]
            mask = valid_masks_by_role[role]
            count = int(features.shape[-2])
            role_slices[role] = slice(offset, offset + count)
            offset += count
            ordered_features.append(features)
            ordered_masks.append(mask)

        return (
            torch.cat(ordered_features, dim=-2),
            torch.cat(ordered_masks, dim=-1),
            role_slices,
        )

    @staticmethod
    def split_joint_agents(
        tensor: torch.Tensor,
        role_slices: Mapping[str, slice],
    ) -> dict[str, torch.Tensor]:
        """Split a [...,A,D] joint tensor into role tensors."""
        return {
            role: tensor[..., role_slice, :]
            for role, role_slice in role_slices.items()
        }

    # ------------------------------------------------------------------
    # Shared skill encoder + one agent-conditioned adapter
    # ------------------------------------------------------------------
    def infer_skills_with_adapter(
        self,
        observation_features: torch.Tensor,
        valid_mask: torch.Tensor,
        task_observation_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Infer raw shared common skills and adapt them agent-by-agent."""
        raw_common, task_skill, contrastive = super().infer_skills(
            observation_features,
            valid_mask,
            task_observation_features=task_observation_features,
        )
        adapted_common, delta = self.common_skill_adapter(
            raw_common,
            observation_features,
            valid_mask,
        )
        return {
            "raw_common_skills": raw_common,
            "common_skills": adapted_common,
            "adapted_common_skills": adapted_common,
            "common_skill_delta": delta,
            "task_skills": task_skill,
            "contrastive_skills": contrastive,
        }

    def infer_skills(
        self,
        observation_features: torch.Tensor,
        valid_mask: torch.Tensor,
        task_observation_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Preserve the base API, but return ADAPTED common skills."""
        outputs = self.infer_skills_with_adapter(
            observation_features,
            valid_mask,
            task_observation_features=task_observation_features,
        )
        return (
            outputs["common_skills"],
            outputs["task_skills"],
            outputs["contrastive_skills"],
        )

    # ------------------------------------------------------------------
    # Role-specific heterogeneous action interface
    # ------------------------------------------------------------------
    def decode_role_action_logits(
        self,
        role: str,
        observation_features: torch.Tensor,
        common_skills: torch.Tensor,
        task_skills: torch.Tensor,
    ) -> torch.Tensor:
        if role not in ROLE_ORDER:
            raise ValueError(f"Unknown role: {role!r}")
        return self.action_decoder[role].forward_logits(
            observation_features,
            common_skills,
            task_skills,
            direct_residual_skills=self.conditioning_skill(
                common_skills, task_skills
            ),
        )

    def decode_joint_actions(
        self,
        observation_features: torch.Tensor,
        common_skills: torch.Tensor,
        task_skills: torch.Tensor,
        role_slices: Mapping[str, slice],
    ) -> dict[str, torch.Tensor]:
        """Decode each role with its own action space/head."""
        role_features = self.split_joint_agents(
            observation_features, role_slices
        )
        role_common = self.split_joint_agents(common_skills, role_slices)
        role_task = self.split_joint_agents(task_skills, role_slices)

        return {
            role: torch.tanh(
                self.decode_role_action_logits(
                    role,
                    role_features[role],
                    role_common[role],
                    role_task[role],
                )
            )
            for role in ROLE_ORDER
        }

    def forward_joint(
        self,
        observations_by_role: Mapping[str, Mapping[str, torch.Tensor]],
        valid_masks_by_role: Mapping[str, torch.Tensor],
    ) -> dict[str, Any]:
        """Convenience offline forward pass for one heterogeneous episode batch."""
        features, valid_mask, role_slices = self.encode_joint_observations(
            observations_by_role,
            valid_masks_by_role,
        )
        skills = self.infer_skills_with_adapter(features, valid_mask)
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

    # ------------------------------------------------------------------
    # BC initialization for BOTH heterogeneous roles
    # ------------------------------------------------------------------
    def initialize_role_from_bc(
        self,
        role: str,
        checkpoint_path: str | Path,
    ) -> dict[str, Any]:
        if role not in ROLE_ORDER:
            raise ValueError(f"Unknown role: {role!r}")
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        model_type = payload.get("model_type")
        expected = SUPPORTED_BC_TYPES[role]
        if model_type not in expected:
            raise ValueError(
                f"{role} expects BC type {sorted(expected)}, got "
                f"{model_type!r}: {checkpoint_path}"
            )

        bc_config = _normalise_role_config(payload["model_config"])
        if bc_config != self.role_configs[role]:
            raise ValueError(
                f"{role} BC config does not match joint-model role config.\n"
                f"checkpoint={bc_config}\nmodel={self.role_configs[role]}"
            )

        bc_policy = DroneBehaviorCloningPolicy(**payload["model_config"])
        bc_policy.load_state_dict(payload["model_state_dict"])

        self.observation_encoder[role].load_state_dict(
            bc_policy.encoder.state_dict()
        )
        self.target_observation_encoder[role].load_state_dict(
            bc_policy.encoder.state_dict()
        )
        self.action_decoder[role].base_action_head.load_state_dict(
            bc_policy.action_head.state_dict()
        )

        return {
            "role": role,
            "checkpoint": str(checkpoint_path),
            "epoch": payload.get("epoch"),
            "metrics": payload.get("metrics", {}),
        }

    def initialize_from_bc(
        self,
        drone_checkpoint: str | Path,
        observer_checkpoint: str | Path,
    ) -> dict[str, dict[str, Any]]:
        """Initialize both role-specific I/O paths from their BC checkpoints."""
        reports = {
            "drone": self.initialize_role_from_bc(
                "drone", drone_checkpoint
            ),
            "observer": self.initialize_role_from_bc(
                "observer", observer_checkpoint
            ),
        }
        self._freeze_target_modules()
        return reports

    @classmethod
    def from_bc_checkpoints(
        cls,
        drone_checkpoint: str | Path,
        observer_checkpoint: str | Path,
        central_map_channels: int,
        **kwargs: Any,
    ) -> tuple["JointHeterogeneousAdapterHiSSD", dict[str, Any]]:
        """Construct the joint model directly from two role-specific BC files."""
        checkpoints = {
            "drone": Path(drone_checkpoint).expanduser().resolve(),
            "observer": Path(observer_checkpoint).expanduser().resolve(),
        }
        payloads = {
            role: torch.load(path, map_location="cpu", weights_only=False)
            for role, path in checkpoints.items()
        }
        for role in ROLE_ORDER:
            model_type = payloads[role].get("model_type")
            if model_type not in SUPPORTED_BC_TYPES[role]:
                raise ValueError(
                    f"Unexpected {role} BC model_type={model_type!r}: "
                    f"{checkpoints[role]}"
                )

        role_configs = {
            role: _normalise_role_config(payloads[role]["model_config"])
            for role in ROLE_ORDER
        }
        model = cls(
            role_configs=role_configs,
            central_map_channels=central_map_channels,
            **kwargs,
        )
        report = model.initialize_from_bc(
            drone_checkpoint=checkpoints["drone"],
            observer_checkpoint=checkpoints["observer"],
        )
        return model, report

    # ------------------------------------------------------------------
    # Online heterogeneous recurrent inference
    # ------------------------------------------------------------------
    def initial_joint_inference_state(
        self,
        batch_size: int,
        *,
        observer_count: int,
        drone_count: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor]:
        total_count = int(observer_count) + int(drone_count)
        if observer_count < 0 or drone_count < 0 or total_count <= 0:
            raise ValueError(
                "observer_count/drone_count must be non-negative and total > 0."
            )
        return super().initial_inference_state(
            batch_size=batch_size,
            agent_count=total_count,
            device=device,
            dtype=dtype,
        )

    def inference_step(
        self,
        observations_by_role: Mapping[str, Mapping[str, torch.Tensor]],
        valid_masks_by_role: Mapping[str, torch.Tensor],
        state: dict[str, torch.Tensor],
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        """One recurrent heterogeneous step with runtime role cardinalities."""
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
            state = super().initial_inference_state(
                batch_size=batch_size,
                agent_count=total_count,
                device=features.device,
                dtype=features.dtype,
            )

        raw_common, next_common_history = (
            self.common_skill_encoder.forward_step(
                features,
                valid_mask,
                state["common_history"],
            )
        )
        adapted_common, delta = self.common_skill_adapter(
            raw_common,
            features,
            valid_mask,
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
            adapted_common,
            task_skill,
            role_slices,
        )

        outputs: dict[str, Any] = {
            "actions": actions,
            "observation_features": features,
            "raw_common_skills": raw_common,
            "common_skills": adapted_common,
            "adapted_common_skills": adapted_common,
            "common_skill_delta": delta,
            "task_skills": task_skill,
            "contrastive_skills": contrastive,
            "conditioning_skills": self.conditioning_skill(
                adapted_common, task_skill
            ),
            "valid_mask": valid_mask,
            "role_slices": role_slices,
        }

        if self.task_descriptor_head is not None:
            numeric_mask = valid_mask.bool().unsqueeze(-1).to(
                contrastive.dtype
            )
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
            "task_previous_observation_valid": (
                next_task_previous_observation_valid
            ),
            "task_running_sum": next_task_running_sum,
            "task_running_square_sum": next_task_running_square_sum,
            "task_running_abs_delta_sum": (
                next_task_running_abs_delta_sum
            ),
            "task_running_count": next_task_running_count,
            "task_running_delta_count": next_task_running_delta_count,
        }
        return outputs, next_state

    # ------------------------------------------------------------------
    # Diagnostics / persistence
    # ------------------------------------------------------------------
    @torch.no_grad()
    def adapter_identity_error(
        self,
        observation_features: torch.Tensor,
        raw_common_skills: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> float:
        adapted, _ = self.common_skill_adapter(
            raw_common_skills,
            observation_features,
            valid_mask,
        )
        return float(
            (adapted - raw_common_skills).abs().max().item()
        )

    @torch.no_grad()
    def adapter_delta_statistics(
        self,
        observation_features: torch.Tensor,
        raw_common_skills: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, float]:
        _, delta = self.common_skill_adapter(
            raw_common_skills,
            observation_features,
            valid_mask,
        )
        numeric = valid_mask.bool().unsqueeze(-1).to(delta.dtype)
        denominator = (
            numeric.sum() * delta.shape[-1]
        ).clamp_min(1.0)
        delta_rms = torch.sqrt(
            (delta.square() * numeric).sum() / denominator
        )
        raw_rms = torch.sqrt(
            (raw_common_skills.square() * numeric).sum() / denominator
        )
        return {
            "adapter_delta_rms": float(delta_rms.item()),
            "raw_common_rms": float(raw_rms.item()),
            "adapter_to_common_ratio": float(
                (delta_rms / raw_rms.clamp_min(1e-8)).item()
            ),
        }

    def adapter_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.common_skill_adapter.parameters()
        )

    def config(self) -> dict[str, Any]:
        return copy.deepcopy(self.model_config)


def load_joint_heterogeneous_adapter_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[JointHeterogeneousAdapterHiSSD, dict[str, Any]]:
    """Load a native joint heterogeneous-adapter checkpoint."""
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
    model = JointHeterogeneousAdapterHiSSD(
        **payload["model_config"]
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    return model, payload
