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
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.temporal_hidden_dim = int(temporal_hidden_dim)
        self.latent_dim = int(latent_dim)
        self.decoder_hidden_dim = int(decoder_hidden_dim)
        self.residual_logit_scale = float(residual_logit_scale)
        if self.residual_logit_scale <= 0.0:
            raise ValueError("residual_logit_scale must be positive.")

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
        self.skill_action_head = nn.Sequential(
            nn.Linear(self.latent_dim, self.decoder_hidden_dim),
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

    def decode_logits(
        self, features: torch.Tensor, skills: torch.Tensor
    ) -> torch.Tensor:
        if features.shape[:-1] != skills.shape[:-1]:
            raise ValueError("Observation features and skills must align.")
        # The frozen BC path already models observation-conditioned behavior.
        # Keeping observations out of this residual prevents a second actor
        # from bypassing and ignoring the latent skill.
        residual = torch.tanh(self.skill_action_head(skills))
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
    ) -> dict[str, torch.Tensor]:
        features = self.encode_observations(observations)
        contexts, _ = self.encode_sequence(features)
        mu, logvar = self.posterior(contexts)
        skills = self.reparameterize(mu, logvar, sample=sample_latent)
        return {
            "observation_features": features,
            "skill_context": contexts,
            "skill_mu": mu,
            "skill_logvar": logvar,
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
    ) -> torch.Tensor:
        if dtype is None:
            dtype = next(self.parameters()).dtype
        return torch.zeros(
            1,
            int(agent_count),
            self.temporal_hidden_dim,
            device=device,
            dtype=dtype,
        )

    def inference_step(
        self,
        observations: dict[str, torch.Tensor],
        state: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Run deterministic posterior-mean inference for one world cycle."""
        features = self.encode_observations(observations)
        if features.ndim != 3 or features.shape[0] != 1:
            raise ValueError(
                "Online observations must have shape [1,A,...], got "
                f"features={tuple(features.shape)}."
            )
        contexts, next_state = self.encode_sequence(
            features.unsqueeze(1), state
        )
        contexts = contexts[:, 0]
        mu, logvar = self.posterior(contexts)
        return {
            "observation_features": features,
            "skill_context": contexts,
            "skill_mu": mu,
            "skill_logvar": logvar,
            "skills": mu,
            "action_logits": self.decode_logits(features, mu),
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
    if int(payload.get("format_version", 0)) < 4:
        raise ValueError(
            "This checkpoint predates the anti-collapse skill dynamics model. "
            "Retrain it with train_drone_skill_vae.py."
        )
    model = DroneSkillVAE(**payload["model_config"])
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device), payload
