"""RLlib model helpers for HeMAC training and evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

DRONE_CUSTOM_MODEL_NAME = "hemac_clamped_gaussian_torch"
DRONE_LOG_STD_INIT = -1.35
DRONE_LOG_STD_MIN = -2.5
DRONE_LOG_STD_MAX = -0.35
DRONE_MODEL_HIDDEN_SIZES = [256, 256]

_MODEL_REGISTERED = False


def _activation_module(name: str) -> nn.Module:
    """Return a torch activation module from an RLlib-style name."""
    normalized = (name or "relu").lower()
    if normalized == "tanh":
        return nn.Tanh()
    if normalized in {"silu", "swish"}:
        return nn.SiLU()
    if normalized == "linear":
        return nn.Identity()
    return nn.ReLU()


class ClampedGaussianTorchModel(TorchModelV2, nn.Module):
    """Continuous-control policy head with explicit, clamped log std."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        if torch is None or nn is None:
            raise RuntimeError("Torch is required for ClampedGaussianTorchModel.")

        obs_dim = int(np.prod(obs_space.shape))
        action_dim = int(np.prod(action_space.shape))
        custom_config = model_config.get("custom_model_config", {})

        hidden_sizes = custom_config.get("hidden_sizes", model_config.get("fcnet_hiddens", DRONE_MODEL_HIDDEN_SIZES))
        activation_name = custom_config.get("activation", model_config.get("fcnet_activation", "relu"))
        self.log_std_min = float(custom_config.get("log_std_min", DRONE_LOG_STD_MIN))
        self.log_std_max = float(custom_config.get("log_std_max", DRONE_LOG_STD_MAX))
        log_std_init = float(custom_config.get("log_std_init", DRONE_LOG_STD_INIT))

        layers = []
        last_size = obs_dim
        for hidden_size in hidden_sizes:
            linear = nn.Linear(last_size, hidden_size)
            nn.init.orthogonal_(linear.weight, gain=np.sqrt(2.0))
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(_activation_module(activation_name))
            last_size = hidden_size

        self.encoder = nn.Sequential(*layers) if layers else nn.Identity()
        self.policy_head = nn.Linear(last_size, action_dim)
        self.value_head = nn.Linear(last_size, 1)

        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

        self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init, dtype=torch.float32))
        self._last_features = None
        self.num_outputs = action_dim * 2

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"].float()
        features = self.encoder(obs)
        self._last_features = features

        mean = torch.tanh(self.policy_head(features))
        log_std = torch.clamp(self.log_std, min=self.log_std_min, max=self.log_std_max)
        log_std = log_std.unsqueeze(0).expand_as(mean)
        logits = torch.cat([mean, log_std], dim=1)
        return logits, state

    def value_function(self):
        if self._last_features is None:
            raise ValueError("value_function() called before forward().")
        return self.value_head(self._last_features).squeeze(1)

    def get_log_std_stats(self) -> dict[str, float]:
        """Return current log-std summary stats for logging."""
        log_std = torch.clamp(self.log_std.detach(), min=self.log_std_min, max=self.log_std_max)
        return {
            "mean": float(log_std.mean().cpu().item()),
            "min": float(log_std.min().cpu().item()),
            "max": float(log_std.max().cpu().item()),
        }


def register_hemac_rllib_models() -> None:
    """Register all custom RLlib models needed by HeMAC scripts."""
    global _MODEL_REGISTERED
    if _MODEL_REGISTERED:
        return

    ModelCatalog.register_custom_model(DRONE_CUSTOM_MODEL_NAME, ClampedGaussianTorchModel)
    _MODEL_REGISTERED = True


def drone_policy_model_config() -> dict[str, Any]:
    """Return the RLlib model config used by the continuous drone policy."""
    return {
        "custom_model": DRONE_CUSTOM_MODEL_NAME,
        "vf_share_layers": False,
        "fcnet_hiddens": DRONE_MODEL_HIDDEN_SIZES,
        "fcnet_activation": "relu",
        "custom_model_config": {
            "hidden_sizes": DRONE_MODEL_HIDDEN_SIZES,
            "activation": "relu",
            "log_std_init": DRONE_LOG_STD_INIT,
            "log_std_min": DRONE_LOG_STD_MIN,
            "log_std_max": DRONE_LOG_STD_MAX,
        },
    }


def get_policy_log_std_stats(algo, policy_id: str) -> dict[str, float] | None:
    """Fetch log-std summary stats from a policy model if available."""
    policy = algo.get_policy(policy_id)
    model = getattr(policy, "model", None)
    if model is None or not hasattr(model, "get_log_std_stats"):
        return None
    return model.get_log_std_stats()
