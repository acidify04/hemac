"""Vector-observation Skill-VAE and HiSSD models for six MaMuJoCo agents."""

from __future__ import annotations

import copy
from collections.abc import Mapping

import torch
import torch.nn.functional as functional
from torch import nn

from .tasks import AGENT_IDS


HISSD_HIDDEN_DIM = 256
HISSD_MLP_HIDDEN_UNITS = 256
HISSD_ATTENTION_DIM = 256
HISSD_SKILL_DIM = 256
HISSD_ALPHA = 10.0
HISSD_BETA = 2.0
HISSD_EXPECTILE = 0.9


def _agent_mlp(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
    )


class CoordinatedSequenceEncoder(nn.Module):
    """Agent-specific input encoders, recurrent history, and team attention."""

    def __init__(
        self,
        observation_dims: Mapping[str, int],
        hidden_dim: int = 256,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.agent_ids = tuple(observation_dims)
        self.hidden_dim = int(hidden_dim)
        self.input_encoders = nn.ModuleDict(
            {agent: _agent_mlp(dim, hidden_dim) for agent, dim in observation_dims.items()}
        )
        self.recurrent = nn.ModuleDict(
            {agent: nn.GRUCell(hidden_dim, hidden_dim) for agent in self.agent_ids}
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            batch_first=True,
        )
        self.coordination = nn.TransformerEncoder(
            layer, num_layers=1, enable_nested_tensor=False
        )

    def initial_state(self, batch_size: int, *, device, dtype):
        return {
            agent: torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
            for agent in self.agent_ids
        }

    def step(self, observations, hidden, active_agents=None):
        tokens = []
        next_hidden = {}
        for agent_index, agent in enumerate(self.agent_ids):
            encoded = self.input_encoders[agent](observations[agent])
            current = self.recurrent[agent](encoded, hidden[agent])
            if active_agents is not None:
                mask = active_agents[:, agent_index].unsqueeze(-1)
                current = torch.where(mask, current, hidden[agent])
            next_hidden[agent] = current
            tokens.append(current)
        coordinated = self.coordination(torch.stack(tokens, dim=1))
        return coordinated, next_hidden

    def forward(self, observations, valid, active_agents):
        first = observations[self.agent_ids[0]]
        batch_size, sequence_length = first.shape[:2]
        hidden = self.initial_state(
            batch_size, device=first.device, dtype=first.dtype
        )
        outputs = []
        for time_index in range(sequence_length):
            step_obs = {
                agent: observations[agent][:, time_index]
                for agent in self.agent_ids
            }
            step_output, candidate_hidden = self.step(
                step_obs, hidden, active_agents=active_agents
            )
            time_mask = valid[:, time_index].reshape(batch_size, 1)
            hidden = {
                agent: torch.where(
                    time_mask,
                    candidate_hidden[agent],
                    hidden[agent],
                )
                for agent in self.agent_ids
            }
            outputs.append(step_output * time_mask.unsqueeze(-1))
        return torch.stack(outputs, dim=1)


class TaskObservationEncoder(nn.Module):
    """Encode each agent's local vector without a difficulty label."""

    def __init__(self, observation_dims: Mapping[str, int], hidden_dim: int) -> None:
        super().__init__()
        self.agent_ids = tuple(observation_dims)
        self.encoders = nn.ModuleDict(
            {
                agent: _agent_mlp(observation_dims[agent], hidden_dim)
                for agent in self.agent_ids
            }
        )

    def forward(self, observations: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.stack(
            [self.encoders[agent](observations[agent]) for agent in self.agent_ids],
            dim=-2,
        )


class DifficultyHistoryEncoder(nn.Module):
    """Infer latent actuator difficulty from causal local-observation history."""

    def __init__(
        self,
        observation_dims: Mapping[str, int],
        hidden_dim: int = HISSD_HIDDEN_DIM,
        skill_dim: int = HISSD_SKILL_DIM,
        heads: int = 4,
        history_length: int = 32,
    ) -> None:
        super().__init__()
        self.agent_ids = tuple(observation_dims)
        self.hidden_dim = int(hidden_dim)
        self.skill_dim = int(skill_dim)
        self.history_length = int(history_length)
        self.observation_encoder = TaskObservationEncoder(
            observation_dims, hidden_dim
        )
        self.statistics_projection = nn.Linear(hidden_dim * 5, hidden_dim)
        self.position_embedding = nn.Parameter(
            torch.zeros(self.history_length, hidden_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=HISSD_MLP_HIDDEN_UNITS,
            dropout=0.0,
            batch_first=True,
        )
        self.history_transformer = nn.TransformerEncoder(
            layer, num_layers=1, enable_nested_tensor=False
        )
        self.context_gru = nn.GRUCell(skill_dim * 2, skill_dim)
        self.agent_projection = nn.Sequential(
            nn.Linear(hidden_dim, skill_dim), nn.LayerNorm(skill_dim), nn.Tanh()
        )
        self.context_projection = nn.Linear(skill_dim, skill_dim)
        self.query_projection = nn.Sequential(
            nn.Linear(skill_dim, skill_dim), nn.ReLU(), nn.Linear(skill_dim, skill_dim)
        )

    @staticmethod
    def _running_statistics(
        features: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        numeric = valid.unsqueeze(-1).to(features.dtype)
        previous = torch.cat((features[:, :1], features[:, :-1]), dim=1)
        previous_valid = torch.cat(
            (torch.zeros_like(valid[:, :1]), valid[:, :-1]), dim=1
        )
        delta_valid = valid & previous_valid
        delta = torch.where(
            delta_valid.unsqueeze(-1), features - previous, torch.zeros_like(features)
        )
        count = numeric.cumsum(dim=1).clamp_min(1.0)
        running_mean = (features * numeric).cumsum(dim=1) / count
        running_square_mean = (features.square() * numeric).cumsum(dim=1) / count
        running_std = (
            running_square_mean - running_mean.square()
        ).clamp_min(0.0).add(1e-6).sqrt()
        delta_numeric = delta_valid.unsqueeze(-1).to(features.dtype)
        delta_count = delta_numeric.cumsum(dim=1).clamp_min(1.0)
        running_abs_delta = (delta.abs() * delta_numeric).cumsum(dim=1) / delta_count
        return torch.cat(
            (features, delta, running_mean, running_std, running_abs_delta), dim=-1
        )

    def forward(
        self,
        history_observations: Mapping[str, torch.Tensor],
        history_valid: torch.Tensor,
        active_agents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        first = history_observations[self.agent_ids[0]]
        batch_size, sequence_length, history_length = first.shape[:3]
        if history_length != self.history_length:
            raise ValueError(
                f"Expected history length {self.history_length}, got {history_length}"
            )
        flat_observations = {
            agent: value.reshape(
                batch_size * sequence_length, history_length, value.shape[-1]
            )
            for agent, value in history_observations.items()
        }
        encoded = self.observation_encoder(flat_observations)
        flat_valid = history_valid.reshape(
            batch_size * sequence_length, history_length
        ).bool()
        active = active_agents[:, None, :].expand(
            batch_size, sequence_length, len(self.agent_ids)
        ).reshape(batch_size * sequence_length, len(self.agent_ids))
        valid_agents = flat_valid.unsqueeze(-1) & active.unsqueeze(1)
        statistics = self._running_statistics(encoded, valid_agents)
        tokens = self.statistics_projection(statistics)
        tokens = tokens + self.position_embedding[None, :, None, :]
        flat_tokens = tokens.permute(0, 2, 1, 3).reshape(
            -1, history_length, self.hidden_dim
        )
        padding = ~valid_agents.permute(0, 2, 1).reshape(-1, history_length)
        all_padding = padding.all(dim=1)
        if all_padding.any():
            padding = padding.clone()
            padding[all_padding, 0] = False
        causal_mask = torch.triu(
            torch.ones(
                history_length,
                history_length,
                dtype=torch.bool,
                device=tokens.device,
            ),
            diagonal=1,
        )
        transformed = self.history_transformer(
            flat_tokens,
            mask=causal_mask,
            src_key_padding_mask=padding,
        ).reshape(
            batch_size * sequence_length,
            len(self.agent_ids),
            history_length,
            self.hidden_dim,
        ).permute(0, 2, 1, 3)
        transformed = transformed.masked_fill(~valid_agents.unsqueeze(-1), 0.0)
        agent_features = self.agent_projection(transformed)
        context = torch.zeros(
            batch_size * sequence_length,
            self.skill_dim,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        previous_pooled = torch.zeros_like(context)
        for time_index in range(history_length):
            mask = valid_agents[:, time_index]
            numeric = mask.unsqueeze(-1).to(tokens.dtype)
            pooled = (
                agent_features[:, time_index] * numeric
            ).sum(dim=1) / numeric.sum(dim=1).clamp_min(1.0)
            candidate = self.context_gru(
                torch.cat((pooled, pooled - previous_pooled), dim=-1), context
            )
            active_step = mask.any(dim=1, keepdim=True)
            context = torch.where(active_step, candidate, context)
            previous_pooled = torch.where(active_step, pooled, previous_pooled)
        positions = torch.arange(history_length, device=tokens.device)
        last_indices = torch.where(
            flat_valid,
            positions.unsqueeze(0),
            torch.zeros_like(positions).unsqueeze(0),
        ).max(dim=1).values
        row_indices = torch.arange(flat_valid.shape[0], device=tokens.device)
        last_agent_features = agent_features[row_indices, last_indices]
        task_skill = last_agent_features + self.context_projection(context).unsqueeze(1)
        task_skill = task_skill * active.unsqueeze(-1).to(task_skill.dtype)
        query = functional.normalize(self.query_projection(context), dim=-1)
        query = query.unsqueeze(1).expand(-1, len(self.agent_ids), -1)
        task_skill = task_skill.reshape(
            batch_size, sequence_length, len(self.agent_ids), self.skill_dim
        )
        query = query.reshape(
            batch_size, sequence_length, len(self.agent_ids), self.skill_dim
        )
        return task_skill, query


class AgentActionDecoders(nn.Module):
    def __init__(self, action_dims, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.agent_ids = tuple(action_dims)
        self.decoders = nn.ModuleDict(
            {
                agent: nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, action_dims[agent]),
                    nn.Tanh(),
                )
                for agent in self.agent_ids
            }
        )

    def forward(self, features: torch.Tensor):
        return {
            agent: self.decoders[agent](features[:, :, index])
            for index, agent in enumerate(self.agent_ids)
        }


def _masked_agent_action_loss(predicted, target, valid, active_agents):
    losses = []
    for index, agent in enumerate(AGENT_IDS):
        mask = valid & active_agents[:, index].unsqueeze(1)
        squared = (predicted[agent] - target[agent]).pow(2).mean(dim=-1)
        if mask.any():
            losses.append(squared[mask].mean())
    if not losses:
        return torch.tensor(0.0, device=valid.device)
    return torch.stack(losses).mean()


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Paper-style cosine contrast: same-task positives, other-task negatives."""
    embeddings = functional.normalize(embeddings, dim=-1)
    logits = embeddings @ embeddings.transpose(0, 1) / temperature
    diagonal = torch.eye(labels.shape[0], device=labels.device, dtype=torch.bool)
    positive = labels[:, None].eq(labels[None, :]) & ~diagonal
    available = positive.any(dim=1)
    if not available.any():
        return embeddings.sum() * 0.0
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits).masked_fill(diagonal, 0.0)
    log_probability = logits - torch.log(
        exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-8)
    )
    mean_positive = (
        log_probability.masked_fill(~positive, 0.0).sum(dim=1)
        / positive.sum(dim=1).clamp_min(1)
    )
    return -mean_positive[available].mean()


class MultiAgentSkillVAE(nn.Module):
    """Observation-conditioned recurrent skill VAE with separate action heads."""

    algorithm = "skill_vae"

    def __init__(
        self,
        observation_dims,
        action_dims,
        state_dim: int,
        hidden_dim: int = HISSD_HIDDEN_DIM,
        skill_dim: int = HISSD_SKILL_DIM,
    ) -> None:
        super().__init__()
        self.observation_dims = dict(observation_dims)
        self.action_dims = dict(action_dims)
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.skill_dim = int(skill_dim)
        self.encoder = CoordinatedSequenceEncoder(observation_dims, hidden_dim)
        self.mean = nn.Linear(hidden_dim, skill_dim)
        self.log_variance = nn.Linear(hidden_dim, skill_dim)
        self.decoders = AgentActionDecoders(action_dims, skill_dim, hidden_dim)
        self.forward_predictor = nn.Sequential(
            nn.Linear(skill_dim * len(AGENT_IDS), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, batch, *, sample: bool = True):
        hidden = self.encoder(
            batch["observations"], batch["valid"], batch["active_agents"]
        )
        mean = self.mean(hidden)
        log_variance = self.log_variance(hidden).clamp(-8.0, 4.0)
        if sample:
            skill = mean + torch.randn_like(mean) * (0.5 * log_variance).exp()
        else:
            skill = mean
        actions = self.decoders(skill)
        next_state = self.forward_predictor(skill.flatten(start_dim=2))
        return {
            "actions": actions,
            "next_state": next_state,
            "mean": mean,
            "log_variance": log_variance,
            "skill": skill,
        }

    def loss(self, batch, *, kl_coeff=1e-3, prediction_coeff=1.0):
        output = self(batch)
        action_loss = _masked_agent_action_loss(
            output["actions"],
            batch["actions"],
            batch["valid"],
            batch["active_agents"],
        )
        valid = batch["valid"].unsqueeze(-1).unsqueeze(-1)
        kl = -0.5 * (
            1.0
            + output["log_variance"]
            - output["mean"].pow(2)
            - output["log_variance"].exp()
        )
        kl_loss = kl.masked_select(valid).mean()
        state_error = (output["next_state"] - batch["next_states"]).pow(2).mean(-1)
        prediction_loss = state_error[batch["valid"]].mean()
        total = action_loss + kl_coeff * kl_loss + prediction_coeff * prediction_loss
        return total, {
            "loss": total.detach(),
            "action_loss": action_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "prediction_loss": prediction_loss.detach(),
        }

    def initial_inference_state(self, batch_size: int, *, device):
        return self.encoder.initial_state(batch_size, device=device, dtype=torch.float32)

    @torch.no_grad()
    def act_step(self, observations, hidden):
        coordinated, next_hidden = self.encoder.step(observations, hidden)
        skill = self.mean(coordinated)
        actions = {
            agent: self.decoders.decoders[agent](skill[:, index])
            for index, agent in enumerate(AGENT_IDS)
        }
        return actions, next_hidden


class HiSSD(nn.Module):
    """Continuous HiSSD with common/task skills and IQL value learning."""

    algorithm = "hissd"

    def __init__(
        self,
        observation_dims,
        action_dims,
        state_dim: int,
        num_source_tasks: int = 5,
        hidden_dim: int = HISSD_HIDDEN_DIM,
        skill_dim: int = HISSD_SKILL_DIM,
        history_length: int = 32,
        training_sequence_length: int = 1,
    ) -> None:
        super().__init__()
        self.observation_dims = dict(observation_dims)
        self.action_dims = dict(action_dims)
        self.state_dim = int(state_dim)
        self.num_source_tasks = int(num_source_tasks)
        self.hidden_dim = int(hidden_dim)
        self.skill_dim = int(skill_dim)
        self.history_length = int(history_length)
        self.training_sequence_length = int(training_sequence_length)
        if self.training_sequence_length <= 0:
            raise ValueError("training_sequence_length must be positive")
        self.model_config = {
            "num_source_tasks": self.num_source_tasks,
            "hidden_dim": self.hidden_dim,
            "skill_dim": self.skill_dim,
            "history_length": self.history_length,
            "training_sequence_length": self.training_sequence_length,
        }
        self.common_encoder = CoordinatedSequenceEncoder(observation_dims, hidden_dim)
        self.task_encoder = DifficultyHistoryEncoder(
            observation_dims,
            hidden_dim,
            skill_dim,
            history_length=self.history_length,
        )
        self.common_projection = nn.Sequential(nn.Linear(hidden_dim, skill_dim), nn.Tanh())
        self.task_classifier = nn.Linear(skill_dim, self.num_source_tasks)
        self.decoders = AgentActionDecoders(
            action_dims, skill_dim * 2, hidden_dim
        )
        self.forward_predictor = nn.Sequential(
            nn.Linear(skill_dim * len(AGENT_IDS), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.value_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.target_value_network = copy.deepcopy(self.value_network)
        for parameter in self.target_value_network.parameters():
            parameter.requires_grad_(False)

    def forward(self, batch):
        common_hidden = self.common_encoder(
            batch["observations"], batch["valid"], batch["active_agents"]
        )
        history_observations = batch.get("history_observations")
        history_valid = batch.get("history_valid")
        if history_observations is None:
            history_observations = {
                agent: functional.pad(
                    value.unsqueeze(2),
                    (0, 0, self.history_length - 1, 0),
                )
                for agent, value in batch["observations"].items()
            }
            history_valid = functional.pad(
                batch["valid"].unsqueeze(2),
                (self.history_length - 1, 0),
            )
        task_skill, task_query = self.task_encoder(
            history_observations, history_valid, batch["active_agents"]
        )
        common_skill = self.common_projection(common_hidden)
        decoder_input = torch.cat((common_skill, task_skill), dim=-1)
        actions = self.decoders(decoder_input)
        predicted_state = self.forward_predictor(
            common_skill.flatten(start_dim=2)
        )
        numeric_active = batch["active_agents"].to(task_query.dtype)
        pooled_task = (
            task_query.mean(dim=1) * numeric_active.unsqueeze(-1)
        ).sum(dim=1) / numeric_active.sum(dim=1, keepdim=True).clamp_min(1.0)
        return {
            "actions": actions,
            "common_skill": common_skill,
            "task_skill": task_skill,
            "task_query": task_query,
            "predicted_state": predicted_state,
            "pooled_task_skill": pooled_task,
        }

    def loss(
        self,
        batch,
        *,
        gamma=0.99,
        alpha=HISSD_ALPHA,
        beta=HISSD_BETA,
        expectile=HISSD_EXPECTILE,
    ):
        output = self(batch)
        valid = batch["valid"]
        values = self.value_network(batch["states"]).squeeze(-1)
        with torch.no_grad():
            next_values = self.target_value_network(batch["next_states"]).squeeze(-1)
            continuation = (~batch["terminations"]).float()
            target = batch["rewards"] + gamma * continuation * next_values
        td_residual = target - values
        expectile_weight = torch.abs(
            expectile - (td_residual.detach() < 0).to(values.dtype)
        )
        value_loss = (
            expectile_weight[valid] * td_residual[valid].pow(2)
        ).mean()
        planner_weight = torch.exp(td_residual.detach() / alpha).clamp(max=100.0)
        action_loss = _masked_agent_action_loss(
            output["actions"],
            batch["actions"],
            valid,
            batch["active_agents"],
        )
        state_error = (
            output["predicted_state"] - batch["next_states"]
        ).pow(2).mean(-1)
        prediction_loss = (planner_weight[valid] * state_error[valid]).mean()
        task_loss = supervised_contrastive_loss(
            output["pooled_task_skill"], batch["task_id"]
        )
        classification_loss = functional.cross_entropy(
            self.task_classifier(output["pooled_task_skill"]), batch["task_id"]
        )
        total = (
            value_loss
            + action_loss
            + prediction_loss
            + beta * (task_loss + classification_loss)
        )
        return total, {
            "loss": total.detach(),
            "value_loss": value_loss.detach(),
            "action_loss": action_loss.detach(),
            "prediction_loss": prediction_loss.detach(),
            "task_loss": task_loss.detach(),
            "classification_loss": classification_loss.detach(),
        }

    @torch.no_grad()
    def update_target(self, tau: float = 0.005):
        for target, source in zip(
            self.target_value_network.parameters(), self.value_network.parameters()
        ):
            target.mul_(1.0 - tau).add_(source, alpha=tau)

    def initial_inference_state(self, batch_size: int, *, device):
        return {
            "common": self.common_encoder.initial_state(
                batch_size, device=device, dtype=torch.float32
            ),
            "task_observation_history": {
                agent: torch.zeros(
                    batch_size,
                    self.history_length,
                    self.observation_dims[agent],
                    device=device,
                )
                for agent in AGENT_IDS
            },
            "task_history_valid": torch.zeros(
                batch_size, self.history_length, dtype=torch.bool, device=device
            ),
        }

    @torch.no_grad()
    def act_step(self, observations, hidden, active_agents=None):
        if active_agents is None:
            active_agents = torch.ones(
                observations[AGENT_IDS[0]].shape[0],
                len(AGENT_IDS),
                dtype=torch.bool,
                device=observations[AGENT_IDS[0]].device,
            )
        common_input_state = hidden["common"]
        if self.training_sequence_length == 1:
            common_input_state = self.common_encoder.initial_state(
                observations[AGENT_IDS[0]].shape[0],
                device=observations[AGENT_IDS[0]].device,
                dtype=observations[AGENT_IDS[0]].dtype,
            )
        common_hidden, next_common = self.common_encoder.step(
            observations, common_input_state, active_agents=active_agents
        )
        if self.training_sequence_length == 1:
            next_common = common_input_state
        next_observation_history = {
            agent: torch.cat(
                (
                    hidden["task_observation_history"][agent][:, 1:],
                    observations[agent].unsqueeze(1),
                ),
                dim=1,
            )
            for agent in AGENT_IDS
        }
        next_history_valid = torch.cat(
            (
                hidden["task_history_valid"][:, 1:],
                torch.ones(
                    observations[AGENT_IDS[0]].shape[0],
                    1,
                    dtype=torch.bool,
                    device=observations[AGENT_IDS[0]].device,
                ),
            ),
            dim=1,
        )
        history_batch = {
            agent: value.unsqueeze(1)
            for agent, value in next_observation_history.items()
        }
        task_skill, task_query = self.task_encoder(
            history_batch, next_history_valid.unsqueeze(1), active_agents
        )
        task_skill = task_skill[:, 0]
        task_query = task_query[:, 0]
        common_skill = self.common_projection(common_hidden)
        features = torch.cat((common_skill, task_skill), dim=-1)
        actions = {
            agent: self.decoders.decoders[agent](features[:, index])
            for index, agent in enumerate(AGENT_IDS)
        }
        return actions, {
            "common": next_common,
            "task_observation_history": next_observation_history,
            "task_history_valid": next_history_valid,
            "last_task_skill": task_skill,
            "last_task_query": task_query,
        }


def build_offline_model(
    algorithm: str,
    observation_dims,
    action_dims,
    state_dim: int,
    *,
    num_source_tasks: int = 5,
    history_length: int = 32,
    training_sequence_length: int = 1,
):
    if algorithm == "hissd":
        return HiSSD(
            observation_dims,
            action_dims,
            state_dim,
            num_source_tasks=num_source_tasks,
            history_length=history_length,
            training_sequence_length=training_sequence_length,
        )
    if algorithm == "skill_vae":
        return MultiAgentSkillVAE(observation_dims, action_dims, state_dim)
    raise ValueError(f"Unknown offline algorithm: {algorithm}")
