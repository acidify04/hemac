"""Single-latent conditional VAE for homogeneous HeMAC drone policies.

All drones share every parameter.  Each drone nevertheless obtains its own
latent from its decentralized observation history, which keeps the policy
permutation equivariant without an agent-id or role embedding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from skill_discovery.models import DroneObservationEncoder


class DroneSkillVAE(nn.Module):
    """Infer one stochastic behavior skill per drone and reconstruct actions."""

    def __init__(
        self,
        global_map_channels: int,
        local_map_channels: int,
        *,
        action_dim: int = 3,
        global_map_size: tuple[int, int] = (40, 40),
        local_map_size: tuple[int, int] = (20, 20),
        action_history_shape: tuple[int, int] = (5, 3),
        observation_hidden_sizes: tuple[int, ...] = (96, 96),
        temporal_hidden_dim: int = 96,
        latent_dim: int = 8,
        decoder_hidden_dim: int = 64,
        residual_logit_scale: float = 0.2,
        skill_duration: int = 8,
        decoder_observation_conditioned: bool = True,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.temporal_hidden_dim = int(temporal_hidden_dim)
        self.latent_dim = int(latent_dim)
        self.decoder_hidden_dim = int(decoder_hidden_dim)
        self.residual_logit_scale = float(residual_logit_scale)
        self.skill_duration = int(skill_duration)
        self.decoder_observation_conditioned = bool(
            decoder_observation_conditioned
        )
        if self.residual_logit_scale <= 0.0:
            raise ValueError("residual_logit_scale must be positive.")
        if self.skill_duration <= 0:
            raise ValueError("skill_duration must be positive.")

        self.observation_encoder = DroneObservationEncoder(
            global_map_channels,
            local_map_channels,
            global_map_size=global_map_size,
            local_map_size=local_map_size,
            action_history_shape=action_history_shape,
            hidden_sizes=observation_hidden_sizes,
            activation=activation,
        )
        feature_dim = self.observation_encoder.output_dim
        self.temporal_encoder = nn.GRU(
            feature_dim,
            self.temporal_hidden_dim,
            batch_first=True,
        )
        self.posterior_mu = nn.Linear(self.temporal_hidden_dim, self.latent_dim)
        self.posterior_logvar = nn.Linear(
            self.temporal_hidden_dim, self.latent_dim
        )
        self.base_action_head = nn.Linear(feature_dim, self.action_dim)
        skill_decoder_input_dim = self.latent_dim
        if self.decoder_observation_conditioned:
            skill_decoder_input_dim += feature_dim
        self.skill_action_head = nn.Sequential(
            nn.Linear(skill_decoder_input_dim, self.decoder_hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(self.decoder_hidden_dim),
            nn.Linear(self.decoder_hidden_dim, self.action_dim, bias=False),
        )
        self.forward_predictor = nn.Sequential(
            nn.Linear(
                feature_dim + self.latent_dim,
                self.decoder_hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(self.decoder_hidden_dim, feature_dim),
        )
        nn.init.orthogonal_(self.skill_action_head[-1].weight, gain=0.01)

    @property
    def observation_dim(self) -> int:
        return int(self.observation_encoder.output_dim)

    def config(self) -> dict[str, Any]:
        encoder = self.observation_encoder.config()
        return {
            "global_map_channels": encoder["global_map_channels"],
            "local_map_channels": encoder["local_map_channels"],
            "action_dim": self.action_dim,
            "global_map_size": encoder["global_map_size"],
            "local_map_size": encoder["local_map_size"],
            "action_history_shape": encoder["action_history_shape"],
            "observation_hidden_sizes": encoder["hidden_sizes"],
            "temporal_hidden_dim": self.temporal_hidden_dim,
            "latent_dim": self.latent_dim,
            "decoder_hidden_dim": self.decoder_hidden_dim,
            "residual_logit_scale": self.residual_logit_scale,
            "skill_duration": self.skill_duration,
            "decoder_observation_conditioned": (
                self.decoder_observation_conditioned
            ),
            "activation": encoder["activation"],
        }

    def encode_observations(
        self, observations: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return self.observation_encoder(
            observations["global_map"],
            observations["local_map"],
            observations["action_history"],
        )

    def encode_sequence(
        self,
        features: torch.Tensor,
        initial_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode `[B,T,A,D]` using the same GRU for every drone."""
        if features.ndim != 4:
            raise ValueError(f"Expected [B,T,A,D], got {tuple(features.shape)}.")
        batch_size, time_steps, agent_count, feature_dim = features.shape
        sequence = features.permute(0, 2, 1, 3).reshape(
            batch_size * agent_count, time_steps, feature_dim
        )
        state = None
        if initial_state is not None:
            expected = (batch_size, agent_count, self.temporal_hidden_dim)
            if tuple(initial_state.shape) != expected:
                raise ValueError(
                    f"Initial state {tuple(initial_state.shape)} != {expected}."
                )
            state = initial_state.reshape(1, batch_size * agent_count, -1)
        encoded, final_state = self.temporal_encoder(sequence, state)
        contexts = encoded.reshape(
            batch_size, agent_count, time_steps, self.temporal_hidden_dim
        ).permute(0, 2, 1, 3)
        return contexts, final_state.reshape(
            batch_size, agent_count, self.temporal_hidden_dim
        )

    def posterior(
        self, contexts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.posterior_mu(contexts)
        logvar = self.posterior_logvar(contexts).clamp(-8.0, 4.0)
        return mu, logvar

    @staticmethod
    def reparameterize(
        mu: torch.Tensor,
        logvar: torch.Tensor,
        *,
        sample: bool,
    ) -> torch.Tensor:
        if not sample:
            return mu
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def hold_skill_sequence(
        self,
        values: torch.Tensor,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Hold each selected `[B,T,A,D]` value for `skill_duration` steps."""
        if values.ndim != 4:
            raise ValueError(f"Expected [B,T,A,D], got {tuple(values.shape)}.")
        batch_size, time_steps, agent_count, value_dim = values.shape
        if offsets is None:
            offsets = torch.zeros(
                batch_size, dtype=torch.long, device=values.device
            )
        offsets = offsets.to(device=values.device, dtype=torch.long).reshape(-1)
        if offsets.numel() != batch_size:
            raise ValueError(
                f"Expected {batch_size} skill offsets, got {offsets.numel()}."
            )

        relative_steps = torch.arange(time_steps, device=values.device).view(1, -1)
        phase = (offsets.view(-1, 1) + relative_steps) % self.skill_duration
        selection_indices = (relative_steps - phase).clamp_min(0)
        gather_indices = selection_indices.view(batch_size, time_steps, 1, 1)
        gather_indices = gather_indices.expand(-1, -1, agent_count, value_dim)
        held = torch.gather(values, dim=1, index=gather_indices)
        decision_mask = phase.eq(0)
        # A cropped window may begin in the middle of a skill. With no preceding
        # recurrent state available, its first value is the best causal choice.
        decision_mask[:, 0] = True
        return held, decision_mask

    def skill_residual(
        self,
        features: torch.Tensor,
        skills: torch.Tensor,
        *,
        head: nn.Module | None = None,
    ) -> torch.Tensor:
        """Decode a bounded residual while retaining current obstacle context."""
        if features.shape[:-1] != skills.shape[:-1]:
            raise ValueError("Observation features and skills must align.")
        decoder_input = skills
        if self.decoder_observation_conditioned:
            decoder_input = torch.cat((features, skills), dim=-1)
        decoder = self.skill_action_head if head is None else head
        return torch.tanh(decoder(decoder_input))

    def decode_logits(
        self, features: torch.Tensor, skills: torch.Tensor
    ) -> torch.Tensor:
        if features.shape[:-1] != skills.shape[:-1]:
            raise ValueError("Observation features and skills must align.")
        residual = self.skill_residual(features, skills)
        return self.base_action_head(features) + self.residual_logit_scale * residual

    def decode_actions(
        self, features: torch.Tensor, skills: torch.Tensor
    ) -> torch.Tensor:
        return torch.tanh(self.decode_logits(features, skills))

    def predict_next_features(
        self,
        features: torch.Tensor,
        skills: torch.Tensor,
    ) -> torch.Tensor:
        delta = self.forward_predictor(
            torch.cat((features, skills), dim=-1)
        )
        return features + delta

    def forward(
        self,
        observations: dict[str, torch.Tensor],
        *,
        sample_latent: bool = True,
        skill_offsets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        features = self.encode_observations(observations)
        contexts, _ = self.encode_sequence(features)
        candidate_mu, candidate_logvar = self.posterior(contexts)
        candidate_skills = self.reparameterize(
            candidate_mu, candidate_logvar, sample=sample_latent
        )
        mu, decision_mask = self.hold_skill_sequence(
            candidate_mu, skill_offsets
        )
        logvar, _ = self.hold_skill_sequence(candidate_logvar, skill_offsets)
        skills, _ = self.hold_skill_sequence(candidate_skills, skill_offsets)
        return {
            "observation_features": features,
            "skill_context": contexts,
            "skill_mu": mu,
            "skill_logvar": logvar,
            "candidate_skill_mu": candidate_mu,
            "candidate_skill_logvar": candidate_logvar,
            "skill_decision_mask": decision_mask,
            "skills": skills,
            "action_logits": self.decode_logits(features, skills),
            "actions": self.decode_actions(features, skills),
        }

    def initial_inference_state(
        self,
        agent_count: int,
        *,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor | int]:
        if dtype is None:
            dtype = next(self.parameters()).dtype
        agent_count = int(agent_count)
        return {
            "recurrent": torch.zeros(
                1,
                agent_count,
                self.temporal_hidden_dim,
                device=device,
                dtype=dtype,
            ),
            "active_skill": torch.zeros(
                1,
                agent_count,
                self.latent_dim,
                device=device,
                dtype=dtype,
            ),
            "active_logvar": torch.zeros(
                1,
                agent_count,
                self.latent_dim,
                device=device,
                dtype=dtype,
            ),
            "selection_context": torch.zeros(
                1,
                agent_count,
                self.temporal_hidden_dim,
                device=device,
                dtype=dtype,
            ),
            "steps_remaining": 0,
        }

    def inference_step(
        self,
        observations: dict[str, torch.Tensor],
        state: dict[str, torch.Tensor | int],
    ) -> tuple[dict[str, torch.Tensor | bool | int], dict[str, torch.Tensor | int]]:
        """Run one cycle, selecting a new skill only after the current one expires."""
        features = self.encode_observations(observations)
        if features.ndim != 3 or features.shape[0] != 1:
            raise ValueError(
                "Online observations must have shape [1,A,...], got "
                f"features={tuple(features.shape)}."
            )
        contexts, next_recurrent = self.encode_sequence(
            features.unsqueeze(1), state["recurrent"]
        )
        contexts = contexts[:, 0]
        candidate_mu, candidate_logvar = self.posterior(contexts)
        switched = int(state["steps_remaining"]) <= 0
        if switched:
            skills = candidate_mu
            logvar = candidate_logvar
            selection_context = contexts
            steps_remaining = self.skill_duration
        else:
            skills = state["active_skill"]
            logvar = state["active_logvar"]
            selection_context = state["selection_context"]
            steps_remaining = int(state["steps_remaining"])
        next_state: dict[str, torch.Tensor | int] = {
            "recurrent": next_recurrent,
            "active_skill": skills,
            "active_logvar": logvar,
            "selection_context": selection_context,
            "steps_remaining": steps_remaining - 1,
        }
        return {
            "observation_features": features,
            "skill_context": selection_context,
            "recurrent_context": contexts,
            "skill_mu": skills,
            "skill_logvar": logvar,
            "candidate_skill_mu": candidate_mu,
            "candidate_skill_logvar": candidate_logvar,
            "skills": skills,
            "skill_switched": switched,
            "skill_steps_remaining": steps_remaining - 1,
            "action_logits": self.decode_logits(features, skills),
        }, next_state

    def initialize_from_bc(self, checkpoint: str | Path) -> dict[str, Any]:
        """Initialize the shared observation and base action path from BC."""
        path = Path(checkpoint).expanduser().resolve()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("model_type") != "drone_behavior_cloning":
            raise ValueError(f"Not a drone BC checkpoint: {path}")
        state = payload["model_state_dict"]
        encoder_state = {
            key.removeprefix("encoder."): value
            for key, value in state.items()
            if key.startswith("encoder.")
        }
        self.observation_encoder.load_state_dict(encoder_state)
        self.base_action_head.load_state_dict(
            {
                key.removeprefix("action_head."): value
                for key, value in state.items()
                if key.startswith("action_head.")
            }
        )
        return payload


def load_drone_skill_vae(
    checkpoint: str | Path,
    device: torch.device,
) -> tuple[DroneSkillVAE, dict[str, Any]]:
    """Load an offline or online homogeneous skill-VAE checkpoint."""
    path = Path(checkpoint).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("model_type") != "homogeneous_drone_skill_vae":
        raise ValueError(f"Not a homogeneous drone skill-VAE checkpoint: {path}")
    format_version = int(payload.get("format_version", 0))
    if format_version < 4:
        raise ValueError(
            "This checkpoint predates the anti-collapse skill dynamics model. "
            "Retrain it with train_drone_skill_vae.py."
        )
    model_config = dict(payload["model_config"])
    if format_version < 5:
        model_config.setdefault("skill_duration", 1)
        model_config.setdefault("decoder_observation_conditioned", False)
    model = DroneSkillVAE(**model_config)
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device), payload
