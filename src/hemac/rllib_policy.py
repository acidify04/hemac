# CNN Layer / Channel size / Pooling 여부

"""RLlib model helpers for HeMAC training and evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

DRONE_CUSTOM_MODEL_NAME = "hemac_clamped_gaussian_torch"
DRONE_MAPPO_CUSTOM_MODEL_NAME = "hemac_mappo_centralized_critic_torch"
DRONE_MAPPO_MODEL_VERSION = 2
OBSERVER_CUSTOM_MODEL_NAME = "hemac_discrete_spatial_torch"
DRONE_LOG_STD_INIT = -1.8
DRONE_LOG_STD_MIN = -2.5
DRONE_LOG_STD_MAX = -0.35
DRONE_MODEL_HIDDEN_SIZES = [96, 96]
GLOBAL_MAP_ENCODER_CHANNELS = (8, 16, 16, 32)
LOCAL_MAP_ENCODER_CHANNELS = (16, 32, 32, 64)
CENTRAL_MAP_ENCODER_CHANNELS = (8, 16, 16, 32)
CENTRAL_CRITIC_HIDDEN_SIZES = [96, 96]

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


def _space_has_spatial_obs(space) -> bool:
    """Return whether an observation space exposes the multi-map observation schema."""
    return hasattr(space, "spaces") and {"vector", "global_map", "local_map"}.issubset(space.spaces.keys())


def _space_has_legacy_spatial_obs(space) -> bool:
    """Return whether an observation space exposes the legacy vector+relative_map schema."""
    return hasattr(space, "spaces") and {"vector", "relative_map"}.issubset(space.spaces.keys())


def _space_has_mappo_obs(space) -> bool:
    """Return whether a space includes drone centralized-critic inputs."""
    return (
        _space_has_spatial_obs(space)
        and {"central_map", "central_vector"}.issubset(space.spaces.keys())
    )


def _flatten_obs_tensor(obs, device=None):
    """Flatten dict/array observations into a batch-first torch tensor."""
    if isinstance(obs, dict):
        flat_parts = [_flatten_obs_tensor(value, device=device) for value in obs.values()]
        return torch.cat(flat_parts, dim=1)

    if not torch.is_tensor(obs):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    else:
        obs = obs.float()
        if device is not None:
            obs = obs.to(device)

    if obs.dim() == 0:
        obs = obs.reshape(1, 1)
    elif obs.dim() == 1:
        obs = obs.unsqueeze(0)
    elif obs.dim() == 2:
        pass
    elif obs.dim() == 3:
        obs = obs.unsqueeze(0)
    else:
        obs = obs.reshape(obs.shape[0], -1)
    return obs.reshape(obs.shape[0], -1)


def _to_float_tensor(obs, device=None):
    """Convert one observation component to a float torch tensor."""
    if not torch.is_tensor(obs):
        return torch.as_tensor(obs, dtype=torch.float32, device=device)
    obs = obs.float()
    if device is not None:
        obs = obs.to(device)
    return obs


class _SpatialObsEncoder(nn.Module):
    """Shared encoder for vector + 2D relative-map observations."""

    def __init__(self, obs_space, hidden_sizes, activation_name):
        super().__init__()
        original_space = getattr(obs_space, "original_space", obs_space)
        self._obs_schema = "flat"
        self._use_spatial_obs = False

        if _space_has_spatial_obs(original_space) or _space_has_spatial_obs(obs_space):
            self._obs_schema = "multi_map"
            self._use_spatial_obs = True
            source_space = original_space if _space_has_spatial_obs(original_space) else obs_space
            vector_space = source_space.spaces["vector"]
            global_map_space = source_space.spaces["global_map"]
            local_map_space = source_space.spaces["local_map"]
            self.vector_dim = int(np.prod(vector_space.shape))
            self.global_map_channels = int(global_map_space.shape[-1])
            self.local_map_channels = int(local_map_space.shape[-1])
            self.global_map_encoder = self._build_map_encoder(
                self.global_map_channels,
                activation_name,
                GLOBAL_MAP_ENCODER_CHANNELS,
            )
            self.local_map_encoder = self._build_map_encoder(
                self.local_map_channels,
                activation_name,
                LOCAL_MAP_ENCODER_CHANNELS,
                final_stride=1,
            )
            with torch.no_grad():
                dummy_global_map = torch.zeros(
                    1,
                    self.global_map_channels,
                    global_map_space.shape[0],
                    global_map_space.shape[1],
                    dtype=torch.float32,
                )
                dummy_local_map = torch.zeros(
                    1,
                    self.local_map_channels,
                    local_map_space.shape[0],
                    local_map_space.shape[1],
                    dtype=torch.float32,
                )
                global_map_feature_dim = int(self.global_map_encoder(dummy_global_map).shape[1])
                local_map_feature_dim = int(self.local_map_encoder(dummy_local_map).shape[1])
            encoder_input_dim = self.vector_dim + global_map_feature_dim + local_map_feature_dim
        elif _space_has_legacy_spatial_obs(original_space) or _space_has_legacy_spatial_obs(obs_space):
            self._obs_schema = "legacy_map"
            self._use_spatial_obs = True
            source_space = original_space if _space_has_legacy_spatial_obs(original_space) else obs_space
            vector_space = source_space.spaces["vector"]
            relative_map_space = source_space.spaces["relative_map"]
            self.vector_dim = int(np.prod(vector_space.shape))
            self.map_channels = int(relative_map_space.shape[-1])
            self.map_encoder = self._build_map_encoder(
                self.map_channels,
                activation_name,
                GLOBAL_MAP_ENCODER_CHANNELS,
            )
            with torch.no_grad():
                dummy_map = torch.zeros(
                    1,
                    self.map_channels,
                    relative_map_space.shape[0],
                    relative_map_space.shape[1],
                    dtype=torch.float32,
                )
                map_feature_dim = int(self.map_encoder(dummy_map).shape[1])
            encoder_input_dim = self.vector_dim + map_feature_dim
        else:
            encoder_input_dim = int(np.prod(obs_space.shape))
            self.vector_dim = encoder_input_dim
            self.global_map_channels = 0
            self.local_map_channels = 0
            self.map_channels = 0
            self.global_map_encoder = nn.Identity()
            self.local_map_encoder = nn.Identity()
            self.map_encoder = nn.Identity()

        layers = []
        last_size = encoder_input_dim
        for hidden_size in hidden_sizes:
            linear = nn.Linear(last_size, hidden_size)
            nn.init.orthogonal_(linear.weight, gain=np.sqrt(2.0))
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(_activation_module(activation_name))
            last_size = hidden_size

        self.encoder = nn.Sequential(*layers) if layers else nn.Identity()
        self.output_dim = last_size

    @staticmethod
    def _build_map_encoder(
        map_channels,
        activation_name,
        encoder_channels,
        final_stride=2,
    ):
        """Build a compact CNN encoder for one spatial map."""
        conv1_channels, conv2_channels, conv3_channels, conv4_channels = encoder_channels
        return nn.Sequential(
            nn.Conv2d(map_channels, conv1_channels, kernel_size=5, stride=2, padding=2),
            _activation_module(activation_name),
            nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, stride=2, padding=1),
            _activation_module(activation_name),
            nn.Conv2d(conv2_channels, conv3_channels, kernel_size=3, stride=1, padding=1),
            _activation_module(activation_name),
            nn.Conv2d(conv3_channels, conv3_channels, kernel_size=3, stride=1, padding=1),
            _activation_module(activation_name),
            nn.MaxPool2d(2),
            nn.Conv2d(conv3_channels, conv4_channels, kernel_size=3, stride=final_stride, padding=1),
            _activation_module(activation_name),
            nn.Flatten(),
        )

    def encode(self, input_dict):
        """Encode the current observation into a flat feature vector."""
        if self._obs_schema == "multi_map":
            obs_dict = input_dict["obs"]
            if not isinstance(obs_dict, dict) and isinstance(input_dict.get("obs_flat"), dict):
                obs_dict = input_dict["obs_flat"]
            vector_obs = _to_float_tensor(obs_dict["vector"])
            global_map = _to_float_tensor(obs_dict["global_map"])
            local_map = _to_float_tensor(obs_dict["local_map"])
            if vector_obs.dim() == 1:
                vector_obs = vector_obs.unsqueeze(0)
            if global_map.dim() == 3:
                global_map = global_map.unsqueeze(0)
            if local_map.dim() == 3:
                local_map = local_map.unsqueeze(0)
            if global_map.shape[-1] == self.global_map_channels:
                global_map = global_map.permute(0, 3, 1, 2)
            if local_map.shape[-1] == self.local_map_channels:
                local_map = local_map.permute(0, 3, 1, 2)
            global_map_features = self.global_map_encoder(global_map)
            local_map_features = self.local_map_encoder(local_map)
            encoder_input = torch.cat([vector_obs, global_map_features, local_map_features], dim=1)
        elif self._obs_schema == "legacy_map":
            obs_dict = input_dict["obs"]
            if not isinstance(obs_dict, dict) and isinstance(input_dict.get("obs_flat"), dict):
                obs_dict = input_dict["obs_flat"]
            vector_obs = _to_float_tensor(obs_dict["vector"])
            relative_map = _to_float_tensor(obs_dict["relative_map"])
            if vector_obs.dim() == 1:
                vector_obs = vector_obs.unsqueeze(0)
            if relative_map.dim() == 3:
                relative_map = relative_map.unsqueeze(0)
            if relative_map.shape[-1] == self.map_channels:
                relative_map = relative_map.permute(0, 3, 1, 2)
            map_features = self.map_encoder(relative_map)
            encoder_input = torch.cat([vector_obs, map_features], dim=1)
        else:
            encoder_input = _flatten_obs_tensor(input_dict["obs_flat"])
        return self.encoder(encoder_input)


class ClampedGaussianTorchModel(TorchModelV2, _SpatialObsEncoder):
    """Continuous-control policy head with explicit, clamped log std."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        if torch is None or nn is None:
            raise RuntimeError("Torch is required for ClampedGaussianTorchModel.")

        action_dim = int(np.prod(action_space.shape))
        custom_config = model_config.get("custom_model_config", {})
        hidden_sizes = custom_config.get("hidden_sizes", model_config.get("fcnet_hiddens", DRONE_MODEL_HIDDEN_SIZES))
        activation_name = custom_config.get("activation", model_config.get("fcnet_activation", "relu"))
        self.log_std_min = float(custom_config.get("log_std_min", DRONE_LOG_STD_MIN))
        self.log_std_max = float(custom_config.get("log_std_max", DRONE_LOG_STD_MAX))
        log_std_init = float(custom_config.get("log_std_init", DRONE_LOG_STD_INIT))
        _SpatialObsEncoder.__init__(self, obs_space, hidden_sizes, activation_name)

        self.policy_head = nn.Linear(self.output_dim, action_dim)
        self.value_head = nn.Linear(self.output_dim, 1)

        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

        self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init, dtype=torch.float32))
        self._last_features = None
        self.num_outputs = action_dim * 2

    def forward(self, input_dict, state, seq_lens):
        features = self.encode(input_dict)
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

    def reset_log_std(self, value: float | None = None) -> dict[str, float]:
        """Reset Gaussian exploration scale without changing policy weights."""
        reset_value = DRONE_LOG_STD_INIT if value is None else float(value)
        reset_value = float(np.clip(reset_value, self.log_std_min, self.log_std_max))
        with torch.no_grad():
            self.log_std.fill_(reset_value)
        return self.get_log_std_stats()


class MAPPOCentralizedCriticTorchModel(ClampedGaussianTorchModel):
    """Shared drone actor with a separate world-centered centralized critic."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        original_space = getattr(obs_space, "original_space", obs_space)
        source_space = original_space if _space_has_mappo_obs(original_space) else obs_space
        if not _space_has_mappo_obs(source_space):
            raise ValueError(
                "MAPPO drone observations must include central_map and central_vector."
            )

        custom_config = model_config.get("custom_model_config", {})
        activation_name = custom_config.get(
            "activation",
            model_config.get("fcnet_activation", "relu"),
        )
        central_hidden_sizes = custom_config.get(
            "central_critic_hidden_sizes",
            CENTRAL_CRITIC_HIDDEN_SIZES,
        )
        central_map_space = source_space.spaces["central_map"]
        central_vector_space = source_space.spaces["central_vector"]
        self.central_map_channels = int(central_map_space.shape[-1])
        self.central_map_encoder = self._build_map_encoder(
            self.central_map_channels,
            activation_name,
            CENTRAL_MAP_ENCODER_CHANNELS,
            final_stride=1,
        )

        with torch.no_grad():
            dummy_central_map = torch.zeros(
                1,
                self.central_map_channels,
                central_map_space.shape[0],
                central_map_space.shape[1],
                dtype=torch.float32,
            )
            central_map_feature_dim = int(
                self.central_map_encoder(dummy_central_map).shape[1]
            )

        central_vector_dim = int(np.prod(central_vector_space.shape))
        critic_layers = []
        critic_input_dim = central_map_feature_dim + central_vector_dim
        for hidden_size in central_hidden_sizes:
            linear = nn.Linear(critic_input_dim, int(hidden_size))
            nn.init.orthogonal_(linear.weight, gain=np.sqrt(2.0))
            nn.init.zeros_(linear.bias)
            critic_layers.extend([linear, _activation_module(activation_name)])
            critic_input_dim = int(hidden_size)

        self.central_critic_encoder = (
            nn.Sequential(*critic_layers) if critic_layers else nn.Identity()
        )
        self.central_value_head = nn.Linear(critic_input_dim, 1)
        nn.init.orthogonal_(self.central_value_head.weight, gain=1.0)
        nn.init.zeros_(self.central_value_head.bias)

        # The inherited local value head is intentionally unused in MAPPO.
        del self.value_head
        self._last_central_features = None
        self.central_critic_input_dim = central_map_feature_dim + central_vector_dim

    def forward(self, input_dict, state, seq_lens):
        actor_features = self.encode(input_dict)
        mean = torch.tanh(self.policy_head(actor_features))
        log_std = torch.clamp(
            self.log_std,
            min=self.log_std_min,
            max=self.log_std_max,
        )
        logits = torch.cat(
            [mean, log_std.unsqueeze(0).expand_as(mean)],
            dim=1,
        )

        obs_dict = input_dict["obs"]
        if not isinstance(obs_dict, dict) and isinstance(input_dict.get("obs_flat"), dict):
            obs_dict = input_dict["obs_flat"]
        if not isinstance(obs_dict, dict):
            raise TypeError("MAPPO model expected a dictionary observation.")

        central_map = _to_float_tensor(
            obs_dict["central_map"],
            device=actor_features.device,
        )
        central_vector = _to_float_tensor(
            obs_dict["central_vector"],
            device=actor_features.device,
        )
        if central_map.dim() == 3:
            central_map = central_map.unsqueeze(0)
        if central_vector.dim() == 1:
            central_vector = central_vector.unsqueeze(0)
        if central_map.shape[-1] == self.central_map_channels:
            central_map = central_map.permute(0, 3, 1, 2)

        central_map_features = self.central_map_encoder(central_map)
        central_input = torch.cat(
            [central_map_features, central_vector.reshape(central_vector.shape[0], -1)],
            dim=1,
        )
        self._last_central_features = self.central_critic_encoder(central_input)
        return logits, state

    def value_function(self):
        if self._last_central_features is None:
            raise ValueError("value_function() called before forward().")
        return self.central_value_head(self._last_central_features).squeeze(1)


class SpatialCategoricalTorchModel(TorchModelV2, _SpatialObsEncoder):
    """Discrete-action policy/value model for spatial observations."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        if torch is None or nn is None:
            raise RuntimeError("Torch is required for SpatialCategoricalTorchModel.")

        custom_config = model_config.get("custom_model_config", {})
        hidden_sizes = custom_config.get("hidden_sizes", model_config.get("fcnet_hiddens", DRONE_MODEL_HIDDEN_SIZES))
        activation_name = custom_config.get("activation", model_config.get("fcnet_activation", "relu"))
        _SpatialObsEncoder.__init__(self, obs_space, hidden_sizes, activation_name)

        self.policy_head = nn.Linear(self.output_dim, num_outputs)
        self.value_head = nn.Linear(self.output_dim, 1)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)
        self._last_features = None

    def forward(self, input_dict, state, seq_lens):
        features = self.encode(input_dict)
        self._last_features = features
        logits = self.policy_head(features)
        return logits, state

    def value_function(self):
        if self._last_features is None:
            raise ValueError("value_function() called before forward().")
        return self.value_head(self._last_features).squeeze(1)


def register_hemac_rllib_models() -> None:
    """Register all custom RLlib models needed by HeMAC scripts."""
    global _MODEL_REGISTERED
    if _MODEL_REGISTERED:
        return

    ModelCatalog.register_custom_model(DRONE_CUSTOM_MODEL_NAME, ClampedGaussianTorchModel)
    ModelCatalog.register_custom_model(
        DRONE_MAPPO_CUSTOM_MODEL_NAME,
        MAPPOCentralizedCriticTorchModel,
    )
    ModelCatalog.register_custom_model(OBSERVER_CUSTOM_MODEL_NAME, SpatialCategoricalTorchModel)
    _MODEL_REGISTERED = True


def drone_policy_model_config() -> dict[str, Any]:
    """Return the MAPPO actor/central-critic config used by the drone policy."""
    return {
        "custom_model": DRONE_MAPPO_CUSTOM_MODEL_NAME,
        "vf_share_layers": False,
        "fcnet_hiddens": DRONE_MODEL_HIDDEN_SIZES,
        "fcnet_activation": "relu",
        "custom_model_config": {
            "hidden_sizes": DRONE_MODEL_HIDDEN_SIZES,
            "activation": "relu",
            "log_std_init": DRONE_LOG_STD_INIT,
            "log_std_min": DRONE_LOG_STD_MIN,
            "log_std_max": DRONE_LOG_STD_MAX,
            "central_critic_hidden_sizes": CENTRAL_CRITIC_HIDDEN_SIZES,
            "mappo_model_version": DRONE_MAPPO_MODEL_VERSION,
        },
    }


def observer_policy_model_config() -> dict[str, Any]:
    """Return the RLlib model config used by the observer policy."""
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
