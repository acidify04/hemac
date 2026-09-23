"""Lazy fixed-window dataset for MaMuJoCo multi-task trajectories."""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from .tasks import AGENT_IDS, TaskSpec, list_tasks


@dataclass(frozen=True)
class Window:
    episode_index: int
    start: int
    length: int


def _pad(tensor: torch.Tensor, start: int, length: int, output_length: int):
    output = torch.zeros((output_length, *tensor.shape[1:]), dtype=tensor.dtype)
    output[:length].copy_(tensor[start : start + length])
    return output


def _history_windows(
    tensor: torch.Tensor,
    start: int,
    length: int,
    output_length: int,
    history_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build causal history ending at each requested transition."""
    histories = torch.zeros(
        (output_length, history_length, *tensor.shape[1:]), dtype=tensor.dtype
    )
    valid = torch.zeros((output_length, history_length), dtype=torch.bool)
    for offset in range(length):
        end = start + offset + 1
        begin = max(0, end - history_length)
        count = end - begin
        histories[offset, history_length - count :].copy_(tensor[begin:end])
        valid[offset, history_length - count :] = True
    return histories, valid


class MultiTaskTrajectoryDataset(Dataset):
    """Sample balanced source-task windows without loading all episodes at once."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        sequence_length: int = 1,
        history_length: int = 32,
        stride: int | None = None,
        cache_size: int = 4,
        allowed_tasks: tuple[str, ...] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        with self.manifest_path.open("r", encoding="utf-8") as file:
            self.manifest = json.load(file)
        if self.manifest.get("format_version") != 1:
            raise ValueError("Unsupported MaMuJoCo manifest format.")
        self.root = self.manifest_path.parents[1]
        self.suite = self.manifest["suite"]
        if self.suite == "difficulty":
            protocol = self.manifest.get("difficulty_protocol")
            if not protocol:
                raise ValueError("Difficulty manifest is missing difficulty_protocol")
            source_names = tuple(protocol["source_difficulties"])
            target_names = set(protocol["target_difficulties"])
            leaked = target_names & set(self.manifest.get("tasks", {}))
            if leaked:
                raise ValueError(
                    f"Target difficulty leakage in source manifest: {sorted(leaked)}"
                )
            unexpected = set(self.manifest.get("tasks", {})) - set(source_names)
            if unexpected:
                raise ValueError(
                    f"Non-source tasks in source manifest: {sorted(unexpected)}"
                )
            task_fields = {
                "suite",
                "name",
                "split",
                "disabled_agent",
                "mass_scale",
                "friction_scale",
                "actuator_scale",
            }
            self.source_tasks = [
                TaskSpec(
                    **{
                        key: value
                        for key, value in self.manifest["tasks"][name]["spec"].items()
                        if key in task_fields
                    }
                )
                for name in source_names
            ]
        else:
            self.source_tasks = list(list_tasks(self.suite, "source"))
        if allowed_tasks is not None:
            allowed = set(allowed_tasks)
            unknown = allowed - {task.name for task in self.source_tasks}
            if unknown:
                raise ValueError(f"Requested non-source tasks: {sorted(unknown)}")
            self.source_tasks = [
                task for task in self.source_tasks if task.name in allowed
            ]
        self.task_to_id = {task.name: index for index, task in enumerate(self.source_tasks)}
        self.sequence_length = int(sequence_length)
        self.history_length = int(history_length)
        self.stride = int(stride or sequence_length)
        self.cache_size = max(int(cache_size), 1)
        if self.sequence_length <= 0 or self.history_length <= 0 or self.stride <= 0:
            raise ValueError("sequence_length, history_length, and stride must be positive")
        self.episodes = []
        for task in self.source_tasks:
            task_record = self.manifest["tasks"].get(task.name)
            if task_record is None:
                raise ValueError(f"Manifest is missing source task {task.name}")
            for entry in task_record["episodes"]:
                self.episodes.append((task, entry))
        self.windows = []
        for episode_index, (_, entry) in enumerate(self.episodes):
            transitions = int(entry["transitions"])
            for start in range(0, transitions, self.stride):
                length = min(self.sequence_length, transitions - start)
                self.windows.append(Window(episode_index, start, length))
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.windows)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_cache"] = OrderedDict()
        return state

    def _load(self, relative_path: str) -> dict[str, Any]:
        cached = self._cache.pop(relative_path, None)
        if cached is not None:
            self._cache[relative_path] = cached
            return cached
        path = self.root / relative_path
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        self._cache[relative_path] = payload
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return payload

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self.windows[index]
        task, entry = self.episodes[window.episode_index]
        payload = self._load(entry["path"])
        valid = torch.zeros(self.sequence_length, dtype=torch.bool)
        valid[: window.length] = True
        observations = {
            agent: _pad(
                payload["observations"][agent],
                window.start,
                window.length,
                self.sequence_length,
            ).float()
            for agent in AGENT_IDS
        }
        history_observations = {}
        history_valid = None
        for agent in AGENT_IDS:
            history, agent_history_valid = _history_windows(
                payload["observations"][agent],
                window.start,
                window.length,
                self.sequence_length,
                self.history_length,
            )
            history_observations[agent] = history.float()
            if history_valid is None:
                history_valid = agent_history_valid
        next_observations = {
            agent: _pad(
                payload["next_observations"][agent],
                window.start,
                window.length,
                self.sequence_length,
            ).float()
            for agent in AGENT_IDS
        }
        actions = {
            agent: _pad(
                payload["actions"][agent],
                window.start,
                window.length,
                self.sequence_length,
            ).float()
            for agent in AGENT_IDS
        }
        active_agents = torch.tensor(
            [agent != task.disabled_agent for agent in AGENT_IDS], dtype=torch.bool
        )
        return {
            "observations": observations,
            "history_observations": history_observations,
            "history_valid": history_valid,
            "next_observations": next_observations,
            "actions": actions,
            "states": _pad(
                payload["states"], window.start, window.length, self.sequence_length
            ).float(),
            "next_states": _pad(
                payload["next_states"],
                window.start,
                window.length,
                self.sequence_length,
            ).float(),
            "rewards": _pad(
                payload["rewards"], window.start, window.length, self.sequence_length
            ).float(),
            "terminations": _pad(
                payload["terminations"],
                window.start,
                window.length,
                self.sequence_length,
            ).bool(),
            "valid": valid,
            "active_agents": active_agents,
            "task_id": torch.tensor(self.task_to_id[task.name], dtype=torch.long),
            "task_name": task.name,
            # Legacy/paper-style joint-disable trajectories do not have the
            # optional difficulty-quality label. Keep the batch collatable
            # without inventing a supervised task label.
            "quality": entry.get("quality") or "unspecified",
        }


def build_dataloader(
    manifest_path: str | Path,
    *,
    batch_size: int = 128,
    sequence_length: int = 1,
    history_length: int = 32,
    stride: int | None = None,
    num_workers: int = 0,
    shuffle: bool = True,
    allowed_tasks: tuple[str, ...] | None = None,
) -> DataLoader:
    dataset = MultiTaskTrajectoryDataset(
        manifest_path,
        sequence_length=sequence_length,
        history_length=history_length,
        stride=stride,
        allowed_tasks=allowed_tasks,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
