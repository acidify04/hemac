"""PyTorch Dataset and DataLoader utilities for joint HeMAC trajectories."""

from __future__ import annotations

import json
import math
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, WeightedRandomSampler

from skill_discovery.drone_task import DRONE_SKILL_SUCCESS_REWARD
from skill_discovery.task_descriptor import (
    REALIZED_TASK_DESCRIPTOR_NAMES,
    TASK_DESCRIPTOR_NAMES,
    build_task_descriptor,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "src/skill_discovery/offline_data"
MISSION_MANIFEST_PATH = DEFAULT_DATA_ROOT / "dataset_splits.json"
DRONE_TASK_MANIFEST_PATH = DEFAULT_DATA_ROOT / "drone_task_dataset_splits.json"
DEFAULT_MANIFEST_PATH = MISSION_MANIFEST_PATH
CATEGORY_IDS = {
    "success": 0,
    "goal_found_failure": 1,
    "goal_not_found": 2,
}


@dataclass(frozen=True)
class EpisodeWindow:
    """Identify one fixed-length sequence within a manifest episode."""

    entry_index: int
    start: int
    valid_length: int


def load_manifest(path: Path) -> dict[str, Any]:
    """Load a generated split manifest from JSON."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Split manifest not found: {path}. Run build_dataset_splits.py first."
        )
    with path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    if int(manifest.get("manifest_version", 0)) not in (1, 2):
        raise ValueError(f"Unsupported manifest version in {path}")
    if "splits" not in manifest:
        raise ValueError(f"Manifest has no splits: {path}")
    return manifest


def pad_time_slice(
    tensor: torch.Tensor,
    start: int,
    valid_length: int,
    output_length: int,
    offset: int = 0,
) -> torch.Tensor:
    """Copy one time slice into a fixed-length zero-padded tensor."""
    source = tensor[start + offset : start + offset + valid_length]
    output = torch.zeros(
        (output_length, *tensor.shape[1:]),
        dtype=tensor.dtype,
    )
    output[:valid_length].copy_(source)
    return output


def scalar_bool(value: Any) -> bool:
    """Convert a persisted scalar label to bool."""
    if isinstance(value, torch.Tensor):
        return bool(value.item())
    return bool(value)


class JointTrajectoryDataset(Dataset):
    """Lazily load padded multi-agent trajectory windows from PT episodes."""

    def __init__(
        self,
        manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
        split: str = "source_train",
        *,
        data_root: str | Path | None = None,
        sequence_length: int = 16,
        stride: int | None = None,
        drop_last_window: bool = False,
        normalize_actions: bool = True,
        include_observer: bool = True,
        include_labels: bool = False,
        cache_size: int = 4,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.manifest = load_manifest(self.manifest_path)
        if split not in self.manifest["splits"]:
            available = ", ".join(sorted(self.manifest["splits"]))
            raise ValueError(f"Unknown split {split!r}; available splits: {available}")
        self.split = split
        self.entries = list(self.manifest["splits"][split])
        if not self.entries:
            raise ValueError(f"Split {split!r} contains no episodes.")
        self._entry_transitions = {
            entry["path"]: int(entry["transitions"])
            for entry in self.entries
        }

        if data_root is None:
            manifest_root = Path(self.manifest.get("data_root", DEFAULT_DATA_ROOT))
            self.data_root = manifest_root.expanduser().resolve()
        else:
            self.data_root = Path(data_root).expanduser().resolve()
        self.sequence_length = int(sequence_length)
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")
        self.stride = self.sequence_length if stride is None else int(stride)
        if self.stride <= 0:
            raise ValueError("stride must be positive.")
        self.drop_last_window = bool(drop_last_window)
        self.normalize_actions = bool(normalize_actions)
        self.include_observer = bool(include_observer)
        self.include_labels = bool(include_labels)
        self.cache_size = max(int(cache_size), 1)
        self._episode_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.windows = self._build_windows()
        if not self.windows:
            raise ValueError(
                f"Split {split!r} produced no windows with sequence_length="
                f"{self.sequence_length} and drop_last_window={self.drop_last_window}."
            )

    def _build_windows(self) -> list[EpisodeWindow]:
        """Build deterministic window indices without loading trajectory tensors."""
        windows = []
        for entry_index, entry in enumerate(self.entries):
            transition_count = int(entry["transitions"])
            for start in range(0, transition_count, self.stride):
                valid_length = min(self.sequence_length, transition_count - start)
                if self.drop_last_window and valid_length < self.sequence_length:
                    continue
                windows.append(EpisodeWindow(entry_index, start, valid_length))
        return windows

    def __len__(self) -> int:
        """Return the number of fixed-length windows."""
        return len(self.windows)

    def __getstate__(self) -> dict[str, Any]:
        """Do not copy open memory maps into DataLoader worker processes."""
        state = dict(self.__dict__)
        state["_episode_cache"] = OrderedDict()
        return state

    def _load_episode(self, relative_path: str) -> dict[str, Any]:
        """Load one episode through a small per-worker memory-map cache."""
        cached = self._episode_cache.pop(relative_path, None)
        if cached is not None:
            self._episode_cache[relative_path] = cached
            return cached

        path = self.data_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Manifest episode does not exist: {path}")
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        expected_transitions = self._entry_transitions[relative_path]
        actual_transitions = int(payload["team_reward"].shape[0])
        if actual_transitions != expected_transitions:
            raise RuntimeError(
                "Dataset manifest is stale because an episode file changed: "
                f"{path} records {actual_transitions} transitions but the manifest "
                f"records {expected_transitions}. Re-run build_dataset_splits.py."
            )
        self._episode_cache[relative_path] = payload
        while len(self._episode_cache) > self.cache_size:
            self._episode_cache.popitem(last=False)
        return payload

    @staticmethod
    def _action_scale(payload: dict[str, Any], role: str) -> float:
        """Read the continuous action limit used by one role."""
        env_config = payload.get("metadata", {}).get("environment_config", {})
        if role == "observer":
            return max(float(env_config.get("observer_speed", 10.0)), 1e-6)
        drone_config = env_config.get("drone_config") or {}
        return max(float(drone_config.get("drone_max_speed", 25.0)), 1e-6)

    def _role_transition_data(
        self,
        payload: dict[str, Any],
        role: str,
        start: int,
        valid_length: int,
    ) -> dict[str, Any]:
        """Build current/next observations and actions for one agent role."""
        role_payload = payload[role]
        action_scale = self._action_scale(payload, role)
        observations = {}
        next_observations = {}
        for name, tensor in role_payload["observations"].items():
            current = pad_time_slice(
                tensor,
                start,
                valid_length,
                self.sequence_length,
            )
            following = pad_time_slice(
                tensor,
                start,
                valid_length,
                self.sequence_length,
                offset=1,
            )
            if self.normalize_actions and name == "action_history":
                current = current / action_scale
                following = following / action_scale
            observations[name] = current
            next_observations[name] = following

        actions = pad_time_slice(
            role_payload["actions"],
            start,
            valid_length,
            self.sequence_length,
        )
        if self.normalize_actions:
            actions = actions / action_scale
        return {
            "observations": observations,
            "next_observations": next_observations,
            "actions": actions,
            "action_scale": torch.tensor(action_scale, dtype=torch.float32),
        }

    def _transition_tensor(
        self,
        payload: dict[str, Any],
        name: str,
        start: int,
        valid_length: int,
    ) -> torch.Tensor:
        """Read and pad one top-level transition tensor."""
        return pad_time_slice(
            payload[name],
            start,
            valid_length,
            self.sequence_length,
        )

    @staticmethod
    def _task_distribution_descriptor(
        payload: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the curriculum-distribution descriptor for analysis."""
        expected_size = len(TASK_DESCRIPTOR_NAMES)
        environment_config = payload.get("metadata", {}).get(
            "environment_config", {}
        )
        try:
            descriptor = build_task_descriptor(environment_config)["normalized"]
        except (KeyError, TypeError, ValueError):
            return (
                torch.zeros(expected_size, dtype=torch.float32),
                torch.tensor(False, dtype=torch.bool),
            )
        descriptor = torch.as_tensor(descriptor, dtype=torch.float32).reshape(-1)
        if descriptor.numel() != expected_size:
            raise ValueError(
                f"Expected {expected_size} task descriptor values, "
                f"got {descriptor.numel()}."
            )
        return descriptor, torch.tensor(True, dtype=torch.bool)

    @staticmethod
    def _task_descriptor(payload: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Load the realized episode dynamics used as the learnable target."""
        expected_size = len(REALIZED_TASK_DESCRIPTOR_NAMES)
        descriptor = payload.get("realized_task_descriptor")
        if descriptor is None:
            legacy_descriptor = payload.get("task_descriptor")
            if (
                isinstance(legacy_descriptor, torch.Tensor)
                and legacy_descriptor.numel() == expected_size
            ):
                descriptor = legacy_descriptor
        if descriptor is None:
            return (
                torch.zeros(expected_size, dtype=torch.float32),
                torch.tensor(False, dtype=torch.bool),
            )
        descriptor = torch.as_tensor(descriptor, dtype=torch.float32).reshape(-1)
        if descriptor.numel() != expected_size or not torch.isfinite(descriptor).all():
            raise ValueError(
                f"Expected {expected_size} finite realized descriptor values, "
                f"got shape {tuple(descriptor.shape)}."
            )
        return descriptor, torch.tensor(True, dtype=torch.bool)

    def _labels(
        self,
        payload: dict[str, Any],
        entry: dict[str, Any],
        start: int,
        valid_length: int,
    ) -> dict[str, torch.Tensor]:
        """Return analysis-only outcome labels in an isolated namespace."""
        outcome = payload.get("outcome", {})
        task_success = bool(entry.get("task_success", entry["category"] == "success"))
        task_goal_found = bool(
            entry.get("task_goal_found", entry["category"] != "goal_not_found")
        )
        labels = {
            "outcome_category": torch.tensor(
                CATEGORY_IDS[entry["category"]],
                dtype=torch.long,
            ),
            "episode_success": torch.tensor(
                task_success,
                dtype=torch.bool,
            ),
            "episode_goal_found": torch.tensor(
                task_goal_found,
                dtype=torch.bool,
            ),
            "episode_drone_goal_found": torch.tensor(
                scalar_bool(outcome.get("drone_goal_found", False)),
                dtype=torch.bool,
            ),
            "episode_mission_success": torch.tensor(
                scalar_bool(
                    outcome.get("mission_success", outcome.get("success", False))
                ),
                dtype=torch.bool,
            ),
            "episode_coverage_ratio": torch.tensor(
                float(
                    entry.get(
                        "coverage_ratio",
                        payload.get("metadata", {})
                        .get("final_info", {})
                        .get("coverage_ratio", 0.0),
                    )
                ),
                dtype=torch.float32,
            ),
        }
        agent_count = int(payload["agent_mask"].shape[-1])
        transition_labels = {
            "success": (torch.bool, (1,)),
            "goal_found": (torch.bool, (1,)),
            "drone_goal_found": (torch.bool, (1,)),
            "coverage_ratio": (torch.float32, (1,)),
            "drone_task_success": (torch.bool, (1,)),
            "agent_goal_found": (torch.bool, (agent_count,)),
        }
        for name, (dtype, trailing_shape) in transition_labels.items():
            available = name in payload
            if available:
                labels[name] = self._transition_tensor(
                    payload,
                    name,
                    start,
                    valid_length,
                )
            else:
                labels[name] = torch.zeros(
                    (self.sequence_length, *trailing_shape),
                    dtype=dtype,
                )
            labels[f"{name}_available"] = torch.tensor(
                available,
                dtype=torch.bool,
            )
        return labels

    def _drone_task_reward(
        self,
        payload: dict[str, Any],
        entry: dict[str, Any],
        start: int,
        valid_length: int,
    ) -> torch.Tensor:
        """Build drone-local shaping plus the new terminal task reward."""
        individual = payload["individual_rewards"]
        agent_mask = payload["agent_mask"].bool()
        drone_count = int(payload["drone"]["actions"].shape[1])
        drone_rewards = individual[:, -drone_count:]
        drone_mask = agent_mask[:, -drone_count:]
        reward = (
            (drone_rewards * drone_mask).sum(dim=1)
            / drone_mask.sum(dim=1).clamp_min(1)
        )
        reward = pad_time_slice(
            reward.unsqueeze(-1),
            start,
            valid_length,
            self.sequence_length,
        )
        if bool(entry.get("task_success", entry["category"] == "success")):
            terminal_index = int(payload["team_reward"].shape[0]) - 1
            if start <= terminal_index < start + valid_length:
                reward[terminal_index - start, 0] += float(
                    entry.get("task_success_reward", DRONE_SKILL_SUCCESS_REWARD)
                )
        return reward

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load one padded joint trajectory window."""
        window = self.windows[index]
        entry = self.entries[window.entry_index]
        payload = self._load_episode(entry["path"])
        start = window.start
        valid_length = window.valid_length

        central_map = payload["global_state"]["central_map"]
        task_descriptor, task_descriptor_available = self._task_descriptor(payload)
        task_distribution, task_distribution_available = (
            self._task_distribution_descriptor(payload)
        )
        batch = {
            "task_id": torch.tensor(entry["task_id"], dtype=torch.long),
            "difficulty": torch.tensor(entry["difficulty"], dtype=torch.long),
            "episode_index": torch.tensor(entry["episode_index"], dtype=torch.long),
            "window_start": torch.tensor(start, dtype=torch.long),
            "valid_length": torch.tensor(valid_length, dtype=torch.long),
            "episode_path": entry["path"],
            "task_descriptor": task_descriptor,
            "task_descriptor_available": task_descriptor_available,
            "task_distribution_descriptor": task_distribution,
            "task_distribution_descriptor_available": task_distribution_available,
            "global_state": {
                "central_map": pad_time_slice(
                    central_map,
                    start,
                    valid_length,
                    self.sequence_length,
                ),
            },
            "next_global_state": {
                "central_map": pad_time_slice(
                    central_map,
                    start,
                    valid_length,
                    self.sequence_length,
                    offset=1,
                ),
            },
            "drone": self._role_transition_data(
                payload,
                "drone",
                start,
                valid_length,
            ),
            "individual_rewards": self._transition_tensor(
                payload,
                "individual_rewards",
                start,
                valid_length,
            ),
            "team_reward": self._transition_tensor(
                payload,
                "team_reward",
                start,
                valid_length,
            ),
            "drone_task_reward": self._drone_task_reward(
                payload,
                entry,
                start,
                valid_length,
            ),
            "shared_success_reward": self._transition_tensor(
                payload,
                "shared_success_reward",
                start,
                valid_length,
            ),
            "agent_mask": self._transition_tensor(
                payload,
                "agent_mask",
                start,
                valid_length,
            ),
            "terminated": self._transition_tensor(
                payload,
                "terminated",
                start,
                valid_length,
            ),
            "truncated": self._transition_tensor(
                payload,
                "truncated",
                start,
                valid_length,
            ),
            "filled": self._transition_tensor(
                payload,
                "filled",
                start,
                valid_length,
            ),
        }
        if self.include_observer:
            batch["observer"] = self._role_transition_data(
                payload,
                "observer",
                start,
                valid_length,
            )
        if self.include_labels:
            batch["labels"] = self._labels(
                payload,
                entry,
                start,
                valid_length,
            )
        return batch

    def balanced_sample_weights(self) -> torch.Tensor:
        """Weight windows so each task/category and episode has equal mass."""
        episode_window_counts = Counter(window.entry_index for window in self.windows)
        stratum_episode_counts = Counter(
            (entry["difficulty"], entry["category"])
            for entry in self.entries
        )
        weights = []
        for window in self.windows:
            entry = self.entries[window.entry_index]
            stratum = (entry["difficulty"], entry["category"])
            weight = 1.0 / (
                stratum_episode_counts[stratum]
                * episode_window_counts[window.entry_index]
            )
            weights.append(weight)
        return torch.tensor(weights, dtype=torch.double)


class TaskBalancedBatchSampler(BatchSampler):
    """Build replacement-sampled batches containing every source task."""

    def __init__(
        self,
        dataset: JointTrajectoryDataset,
        batch_size: int,
        *,
        seed: int,
        drop_last: bool,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.generator = torch.Generator().manual_seed(int(seed))
        task_windows: dict[int, list[int]] = {}
        for window_index, window in enumerate(dataset.windows):
            task_id = int(dataset.entries[window.entry_index]["task_id"])
            task_windows.setdefault(task_id, []).append(window_index)
        self.task_windows = {
            task_id: torch.tensor(indices, dtype=torch.long)
            for task_id, indices in sorted(task_windows.items())
        }
        if self.batch_size < len(self.task_windows):
            raise ValueError(
                "task-balanced batches require batch_size >= number of tasks "
                f"({self.batch_size} < {len(self.task_windows)})."
            )

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        task_ids = tuple(self.task_windows)
        task_count = len(task_ids)
        for batch_index in range(len(self)):
            current_size = min(
                self.batch_size,
                len(self.dataset) - batch_index * self.batch_size,
            )
            if self.drop_last:
                current_size = self.batch_size
            counts = [current_size // task_count] * task_count
            for offset in range(current_size % task_count):
                counts[(batch_index + offset) % task_count] += 1
            batch = []
            for task_id, count in zip(task_ids, counts):
                candidates = self.task_windows[task_id]
                positions = torch.randint(
                    len(candidates),
                    (count,),
                    generator=self.generator,
                )
                batch.extend(candidates[positions].tolist())
            order = torch.randperm(len(batch), generator=self.generator).tolist()
            yield [batch[index] for index in order]


def create_dataloader(
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    split: str = "source_train",
    *,
    data_root: str | Path | None = None,
    sequence_length: int = 16,
    stride: int | None = None,
    batch_size: int = 4,
    num_workers: int = 2,
    normalize_actions: bool = True,
    include_observer: bool = True,
    include_labels: bool = False,
    balanced_sampling: bool | None = None,
    task_balanced_batches: bool = False,
    seed: int = 2026,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    drop_last_batch: bool | None = None,
    cache_size: int = 4,
) -> tuple[JointTrajectoryDataset, DataLoader]:
    """Create a dataset and DataLoader with optional balanced source sampling."""
    dataset = JointTrajectoryDataset(
        manifest_path=manifest_path,
        split=split,
        data_root=data_root,
        sequence_length=sequence_length,
        stride=stride,
        normalize_actions=normalize_actions,
        include_observer=include_observer,
        include_labels=include_labels,
        cache_size=cache_size,
    )
    if balanced_sampling is None:
        balanced_sampling = split == "source_train"
    if drop_last_batch is None:
        drop_last_batch = split == "source_train"

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    sampler = None
    batch_sampler = None
    shuffle = split == "source_train"
    if task_balanced_batches:
        batch_sampler = TaskBalancedBatchSampler(
            dataset,
            batch_size,
            seed=seed,
            drop_last=bool(drop_last_batch),
        )
        shuffle = False
    elif balanced_sampling:
        sampler = WeightedRandomSampler(
            dataset.balanced_sample_weights(),
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )
        shuffle = False

    worker_count = max(int(num_workers), 0)
    loader_kwargs = {
        "dataset": dataset,
        "num_workers": worker_count,
        "pin_memory": bool(pin_memory),
        "generator": generator,
    }
    if batch_sampler is not None:
        loader_kwargs["batch_sampler"] = batch_sampler
    else:
        loader_kwargs.update(
            {
                "batch_size": int(batch_size),
                "shuffle": shuffle,
                "sampler": sampler,
                "drop_last": bool(drop_last_batch),
            }
        )
    if worker_count > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(**loader_kwargs)
    return dataset, loader
