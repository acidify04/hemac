"""Paper-oriented HiSSD model for heterogeneous continuous-action HeMAC.

Compared with the previous ``hissd_joint_hetero_models.py``:
  * Does NOT inherit or instantiate an adapter model.
  * ONE observation encoder, ONE common-skill encoder, ONE task-skill
    encoder, and ONE low-level action decoder are shared by ALL agents.
  * Each agent still produces its OWN common/task skills and action.
  * Heterogeneity is confined to (a) normalized continuous actions,
    (b) agent-type indicators, and (c) environment-specific observations.
  * The shared decoder uses skill-conditioned Transformer tokens directly;
    there is NO BC-only base-action shortcut.
  * Value network, value mixer, and forward predictor remain shared.

This is a repository-ALIGNED HeMAC model, NOT a line-for-line port of
mooricAnna/HiSSD/src/modules/agents/multi_task/hissd_agent.py. The official
SMAC entity decomposer, MoCo negative-batch sampling / loss, and IQL/planner training updates
must be addressed in the TRAINER; changing the model alone cannot supply them.

Compatibility: the historical dataset represents Drone A under the internal
role key 'observer' and Drone B under 'drone'. This is ONLY a dataset alias.
Neither role represents a physical Observer in the drone-only environment.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .hissd_variable_models import VariableAgentHeMACHISSD

# The existing offline loader and objective functions use these internal keys.
ROLE_ORDER = ("observer", "drone")
ROLE_TO_DRONE_TYPE = {"observer": "A", "drone": "B"}


def _normalise_role_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "global_map_channels", "local_map_channels", "global_map_size",
        "local_map_size", "action_history_shape", "hidden_sizes",
        "activation", "action_dim",
    )
    absent = [k for k in required if k not in raw]
    if absent:
        raise ValueError(f"Missing role configuration keys: {absent}")
    return {
        "global_map_channels": int(raw["global_map_channels"]),
        "local_map_channels": int(raw["local_map_channels"]),
        "global_map_size": tuple(map(int, raw["global_map_size"])),
        "local_map_size": tuple(map(int, raw["local_map_size"])),
        "action_history_shape": tuple(map(int, raw["action_history_shape"])),
        "hidden_sizes": tuple(map(int, raw["hidden_sizes"])),
        "activation": str(raw["activation"]),
        "action_dim": int(raw["action_dim"]),
    }


class SharedSkillConditionedController(nn.Module):
    """One skill-conditioned continuous-action decoder for all agents.

    Adapts official HISSDAgent.Decoder's combination of observation, planner
    context, and discriminator/task context to continuous actions. Unlike the
    old HeMAC decoder it has no independently trained BC action shortcut, so
    action learning must flow through its skill-conditioned pathway.
    """

    def __init__(self, observation_dim: int, skill_dim: int,
                 action_dim: int, hidden_dim: int = 64,
                 heads: int = 1) -> None:
        super().__init__()
        self.observation_projection = nn.Linear(observation_dim, hidden_dim)
        self.common_projection = nn.Linear(skill_dim, hidden_dim)
        self.task_projection = nn.Linear(skill_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads, dim_feedforward=hidden_dim * 4,
            batch_first=True, activation="relu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward_logits(self, observation_features: torch.Tensor,
                       common_skills: torch.Tensor,
                       task_skills: torch.Tensor,
                       *, direct_residual_skills: torch.Tensor | None = None
                       ) -> torch.Tensor:
        # direct_residual_skills exists only for compatibility with the
        # previous trainer API; the official-style path has no residual head.
        del direct_residual_skills
        leading = observation_features.shape[:-1]
        tokens = torch.stack((
            self.observation_projection(observation_features),
            self.common_projection(common_skills),
            self.task_projection(task_skills),
        ), dim=-2)
        tokens = tokens.reshape(-1, 3, tokens.shape[-1])
        attended = self.transformer(tokens).reshape(*leading, -1)
        return self.action_head(attended)

    def forward(self, observation_features: torch.Tensor,
                common_skills: torch.Tensor,
                task_skills: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.forward_logits(
            observation_features, common_skills, task_skills
        ))


class JointHeterogeneousHiSSD(VariableAgentHeMACHISSD):
    """Shared-parameter HiSSD, returning separate latent skills per agent.

    ``role_configs`` have to describe the SAME normalized observation and
    action TENSOR SHAPES for both types. A/B dynamics and camera geometry may
    differ. Physical speed limits are restored by the environment action
    wrapper after the shared policy emits tanh-normalized actions.

    ``agent_type_embedding`` is an optional HeMAC interface adaptation: the
    official SMAC implementation passes agent-identity information into the
    shared model. Here the embedded input is the drone TYPE (A/B), not its
    fixed index, so the model remains valid under population changes.
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
        use_agent_type_embedding: bool = True,
    ) -> None:
        if set(role_configs) != set(ROLE_ORDER):
            raise ValueError(
                f"Expected role_configs keys {ROLE_ORDER}, got {tuple(role_configs)}"
            )
        if skill_structure != "split":
            raise ValueError("The paper-oriented model requires skill_structure='split'.")

        configs = {role: _normalise_role_config(role_configs[role])
                   for role in ROLE_ORDER}
        # The A/B environment currently supplies the same dimensionality for
        # both roles. Sharing the complete encoder/decoder is consequently
        # possible without modifying the original actor's I/O dimensions.
        a, b = (configs[r] for r in ROLE_ORDER)
        differences = {key: (a[key], b[key]) for key in a if a[key] != b[key]}
        if differences:
            raise ValueError(
                "Cannot share one exact observation encoder / action decoder "
                "across incompatible tensor schemas. Provide a common normalized "
                f"I/O representation first. Different fields: {differences}"
            )

        a_count, b_count = int(reference_observer_count), int(reference_drone_count)
        if a_count <= 0 or b_count <= 0:
            raise ValueError("A and B reference populations must both be positive.")
        super().__init__(
            global_map_channels=a["global_map_channels"],
            local_map_channels=a["local_map_channels"],
            central_map_channels=int(central_map_channels),
            agent_count=a_count + b_count,
            action_dim=a["action_dim"],
            global_map_size=a["global_map_size"],
            local_map_size=a["local_map_size"],
            central_map_size=central_map_size,
            action_history_shape=a["action_history_shape"],
            observation_hidden_sizes=a["hidden_sizes"],
            hidden_dim=hidden_dim,
            skill_dim=skill_dim,
            transformer_heads=transformer_heads,
            activation=a["activation"],
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
        self.reference_observer_count = a_count
        self.reference_drone_count = b_count
        self.use_agent_type_embedding = bool(use_agent_type_embedding)
        self.shared_observation_dim = int(self.observation_encoder.output_dim)
        self.action_dims = {role: cfg["action_dim"] for role, cfg in configs.items()}

        # Replace the legacy BC-compatible actor head by a fully
        # skill-conditioned ONE controller, analogous to official Decoder.
        self.action_decoder = SharedSkillConditionedController(
            observation_dim=self.shared_observation_dim,
            skill_dim=self.skill_dim,
            action_dim=a["action_dim"],
            hidden_dim=self.hidden_dim,
            heads=transformer_heads,
        )

        # Shared encoder, not ModuleDict. The same encoded feature basis is used
        # by both agent types. Optional type information is injected after the
        # shared CNN, like the ID features in the official SMAC controller.
        # A zero initialization gives neutral initial type semantics.
        self.agent_type_embedding = None
        self.target_agent_type_embedding = None
        if self.use_agent_type_embedding:
            self.agent_type_embedding = nn.Embedding(len(ROLE_ORDER),
                                                     self.shared_observation_dim)
            nn.init.zeros_(self.agent_type_embedding.weight)
            self.target_agent_type_embedding = copy.deepcopy(self.agent_type_embedding)
            self.target_agent_type_embedding.eval()
            self.target_agent_type_embedding.requires_grad_(False)

        self.model_config = {
            "role_configs": copy.deepcopy(configs),
            "central_map_channels": int(central_map_channels),
            "reference_observer_count": a_count,
            "reference_drone_count": b_count,
            "central_map_size": tuple(int(x) for x in central_map_size),
            "hidden_dim": int(hidden_dim),
            "skill_dim": int(skill_dim),
            "transformer_heads": int(transformer_heads),
            "variable_mixer_hidden_dim": int(variable_mixer_hidden_dim),
            "contrastive_from_action_skill": bool(contrastive_from_action_skill),
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
            "use_agent_type_embedding": self.use_agent_type_embedding,
        }

    def target_modules(self) -> tuple[nn.Module, ...]:
        base = super().target_modules()
        # Called by base constructor before agent_type_embedding exists.
        target = getattr(self, "target_agent_type_embedding", None)
        return base + ((target,) if target is not None else ())

    def train(self, mode: bool = True):
        super().train(mode)
        target = getattr(self, "target_agent_type_embedding", None)
        if target is not None:
            target.eval()
        return self

    @torch.no_grad()
    def update_targets(self, tau: float) -> None:
        super().update_targets(tau)
        if self.agent_type_embedding is not None:
            self.target_agent_type_embedding.weight.lerp_(
                self.agent_type_embedding.weight, float(tau)
            )

    def encode_role_observations(
        self,
        role: str,
        observations: Mapping[str, torch.Tensor],
        *,
        target: bool = False,
    ) -> torch.Tensor:
        if role not in ROLE_ORDER:
            raise ValueError(f"Unknown internal role {role!r}")
        encoder = (self.target_observation_encoder if target
                   else self.observation_encoder)
        features = encoder(
            observations["global_map"],
            observations["local_map"],
            observations["action_history"],
        )
        emb = (self.target_agent_type_embedding if target
               else self.agent_type_embedding)
        if emb is not None:
            features = features + emb.weight[ROLE_ORDER.index(role)]
        return features

    def encode_joint_observations(
        self,
        observations_by_role: Mapping[str, Mapping[str, torch.Tensor]],
        valid_masks_by_role: Mapping[str, torch.Tensor],
        *,
        target: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, slice]]:
        features, masks, role_slices = [], [], {}
        start = 0
        prefix = None
        for role in ROLE_ORDER:
            feat = self.encode_role_observations(
                role, observations_by_role[role], target=target
            )
            mask = valid_masks_by_role[role]
            if tuple(feat.shape[:-1]) != tuple(mask.shape):
                raise ValueError(f"{role}: observation/mask mismatch: {feat.shape}, {mask.shape}")
            if prefix is None:
                prefix = tuple(feat.shape[:-2])
            elif tuple(feat.shape[:-2]) != prefix:
                raise ValueError("A/B batch and time axes differ")
            count = feat.shape[-2]
            role_slices[role] = slice(start, start + count)
            start += count
            features.append(feat)
            masks.append(mask)
        return torch.cat(features, dim=-2), torch.cat(masks, dim=-1), role_slices

    @staticmethod
    def split_joint_agents(
        tensor: torch.Tensor, role_slices: Mapping[str, slice]
    ) -> dict[str, torch.Tensor]:
        return {role: tensor[..., s, :] for role, s in role_slices.items()}

    def infer_joint_skills(
        self,
        observation_features: torch.Tensor,
        valid_mask: torch.Tensor,
        task_observation_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # IMPORTANT: one COMMON encoder object, K independent per-agent outputs.
        common, task, contrastive = super().infer_skills(
            observation_features, valid_mask,
            task_observation_features=task_observation_features,
        )
        return {"common_skills": common, "task_skills": task,
                "contrastive_skills": contrastive}

    def decode_joint_actions(
        self,
        observation_features: torch.Tensor,
        common_skills: torch.Tensor,
        task_skills: torch.Tensor,
        role_slices: Mapping[str, slice],
    ) -> dict[str, torch.Tensor]:
        # No role-specific heads. Physical control bounds are applied ONLY by
        # the A/B environment action interface, after this normalized output.
        actions = self.decode_actions(
            observation_features, common_skills, task_skills
        )
        return self.split_joint_agents(actions, role_slices)

    def forward_joint(
        self,
        observations_by_role: Mapping[str, Mapping[str, torch.Tensor]],
        valid_masks_by_role: Mapping[str, torch.Tensor],
    ) -> dict[str, Any]:
        features, valid_mask, role_slices = self.encode_joint_observations(
            observations_by_role, valid_masks_by_role
        )
        skills = self.infer_joint_skills(features, valid_mask)
        actions = self.decode_joint_actions(
            features, skills["common_skills"], skills["task_skills"],
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
        features, valid_mask, role_slices = self.encode_joint_observations(
            observations_by_role, valid_masks_by_role
        )
        if features.ndim != 3 or valid_mask.ndim != 2:
            raise ValueError("Online observations must be [B,A,D] with mask [B,A].")
        batch_size, n_agents, _ = features.shape
        previous = state.get("common_history")
        if previous is None or previous.shape != (
            batch_size, n_agents, self.hidden_dim
        ):
            state = self.initial_inference_state(
                batch_size=batch_size, agent_count=n_agents,
                device=features.device, dtype=features.dtype,
            )
        common, common_history = self.common_skill_encoder.forward_step(
            features, valid_mask, state["common_history"]
        )
        (
            task, contrastive, task_history, task_context,
            task_previous_pooled, task_previous_observation,
            task_previous_observation_valid, task_running_sum,
            task_running_square_sum, task_running_abs_delta_sum,
            task_running_count, task_running_delta_count,
        ) = self.task_skill_encoder.forward_step(
            features, valid_mask, state["task_history"],
            state["task_context"], state["task_previous_pooled"],
            state["task_previous_observation"],
            state["task_previous_observation_valid"],
            state["task_running_sum"], state["task_running_square_sum"],
            state["task_running_abs_delta_sum"],
            state["task_running_count"], state["task_running_delta_count"],
        )
        actions = self.decode_joint_actions(features, common, task, role_slices)
        outputs = {
            "actions": actions,
            "observation_features": features,
            "common_skills": common,
            "task_skills": task,
            "contrastive_skills": contrastive,
            "conditioning_skills": self.conditioning_skill(common, task),
            "valid_mask": valid_mask,
            "role_slices": role_slices,
        }
        if self.task_descriptor_head is not None:
            numeric = valid_mask.unsqueeze(-1).to(contrastive.dtype)
            pooled = (contrastive * numeric).sum(dim=1) / numeric.sum(dim=1).clamp_min(1)
            outputs["task_descriptor"] = self.task_descriptor_head(
                self.task_descriptor_dropout(pooled)
            )
        next_state = {
            "common_history": common_history,
            "task_history": task_history,
            "task_context": task_context,
            "task_previous_pooled": task_previous_pooled,
            "task_previous_observation": task_previous_observation,
            "task_previous_observation_valid": task_previous_observation_valid,
            "task_running_sum": task_running_sum,
            "task_running_square_sum": task_running_square_sum,
            "task_running_abs_delta_sum": task_running_abs_delta_sum,
            "task_running_count": task_running_count,
            "task_running_delta_count": task_running_delta_count,
        }
        return outputs, next_state

    def config(self) -> dict[str, Any]:
        return copy.deepcopy(self.model_config)


def load_joint_heterogeneous_hissd_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[JointHeterogeneousHiSSD, dict[str, Any]]:
    """Load a new paper-oriented checkpoint. Old separate-heads are incompatible."""
    path = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(path, map_location=device, weights_only=False)
    if "model_config" not in payload or "model_state_dict" not in payload:
        raise ValueError(f"Missing model_config/state_dict in {path}")
    model = JointHeterogeneousHiSSD(**payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, payload
