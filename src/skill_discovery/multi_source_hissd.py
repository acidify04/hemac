"""Multi-source data utilities for variable-population HiSSD source training."""
from __future__ import annotations

import inspect
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch

from .dataset import create_dataloader


@dataclass(frozen=True)
class SourceSpec:
    name: str
    manifest: Path
    data_root: Path


def create_role_dataloader(
    source: SourceSpec,
    split: str,
    *,
    role: str,
    sequence_length: int,
    stride: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    pin_memory: bool,
    cache_size: int,
):
    if role not in {"drone", "observer"}:
        raise ValueError(f"Unsupported role: {role}")
    return create_dataloader(
        manifest_path=source.manifest,
        split=split,
        data_root=source.data_root,
        sequence_length=sequence_length,
        stride=stride,
        batch_size=batch_size,
        num_workers=num_workers,
        normalize_actions=True,
        include_observer=(role == "observer"),
        include_labels=False,
        balanced_sampling=True,
        task_balanced_batches=True,
        seed=seed,
        pin_memory=pin_memory,
        drop_last_batch=(split == "source_train"),
        cache_size=cache_size,
    )


def _move_observations(
    observations: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        name: observations[name].to(device, non_blocking=True)
        for name in ("global_map", "local_map", "action_history")
    }


def _fallback_role_mask(batch: dict[str, Any], role: str) -> torch.Tensor:
    """Extract role mask when the installed single-source trainer lacks role support.

    Current HeMAC collectors persist actions by role but agent_mask in environment
    order. If explicit role indices are not available, we use the common HeMAC
    ordering observer(s) followed by drone(s). The main trainer first tries the
    repository's own role-aware prepare_batch, so this is only a compatibility
    fallback for older checkouts.
    """
    full = batch["agent_mask"]
    count = int(batch[role]["actions"].shape[-2])
    if full.shape[-1] == count:
        return full

    observer_count = 0
    drone_count = 0
    if "observer" in batch and "actions" in batch["observer"]:
        observer_count = int(batch["observer"]["actions"].shape[-2])
    if "drone" in batch and "actions" in batch["drone"]:
        drone_count = int(batch["drone"]["actions"].shape[-2])

    if full.shape[-1] != observer_count + drone_count:
        raise RuntimeError(
            "Cannot infer role slice from agent_mask. Use the latest "
            "train_hissd_hetero_baseline.py with prepare_batch(..., role)."
        )
    if role == "observer":
        return full[..., :observer_count]
    return full[..., observer_count : observer_count + drone_count]


def prepare_role_batch(
    raw_batch: dict[str, Any],
    device: torch.device,
    role: str,
    *,
    base_trainer=None,
) -> dict[str, Any]:
    """Use the repo's role-aware converter when available, else a fallback."""
    if base_trainer is not None and hasattr(base_trainer, "prepare_batch"):
        fn = base_trainer.prepare_batch
        params = inspect.signature(fn).parameters
        if "role" in params:
            return fn(raw_batch, device, role)
        if role == "drone":
            return fn(raw_batch, device)

    actions = raw_batch[role]["actions"].to(device, non_blocking=True)
    role_mask = _fallback_role_mask(raw_batch, role).to(device, non_blocking=True)
    filled = raw_batch["filled"].squeeze(-1).to(device, non_blocking=True).bool()
    valid_agents = role_mask.bool() & filled.unsqueeze(-1)
    terminated = raw_batch["terminated"].squeeze(-1).to(device, non_blocking=True)
    truncated = raw_batch["truncated"].squeeze(-1).to(device, non_blocking=True)

    if "team_reward" in raw_batch:
        team_reward = raw_batch["team_reward"].squeeze(-1).to(
            device, non_blocking=True
        )
    elif "drone_task_reward" in raw_batch:
        team_reward = raw_batch["drone_task_reward"].squeeze(-1).to(
            device, non_blocking=True
        )
    else:
        raise KeyError("Dataset has neither team_reward nor drone_task_reward")

    return {
        "observations": _move_observations(raw_batch[role]["observations"], device),
        "next_observations": _move_observations(
            raw_batch[role]["next_observations"], device
        ),
        "actions": actions,
        "central_map": raw_batch["global_state"]["central_map"].to(
            device, non_blocking=True
        ),
        "next_central_map": raw_batch["next_global_state"]["central_map"].to(
            device, non_blocking=True
        ),
        "team_reward": team_reward,
        "task_descriptor": raw_batch["task_descriptor"].to(
            device, non_blocking=True
        ),
        "task_descriptor_available": raw_batch[
            "task_descriptor_available"
        ].to(device, non_blocking=True).bool(),
        "task_id": raw_batch["task_id"].to(device, non_blocking=True),
        "task_supervision": (
            raw_batch["window_start"].to(device, non_blocking=True) == 0
        ),
        "valid_agents": valid_agents,
        "valid_steps": filled,
        "done": terminated.bool() | truncated.bool(),
    }


def balanced_source_batches(
    loaders: dict[str, Any],
    *,
    seed: int,
    batches_per_source: int | None = None,
) -> Iterator[tuple[str, Any]]:
    """Yield exactly the same number of batches from every source configuration."""
    if not loaders:
        return
    names = tuple(loaders)
    if batches_per_source is None:
        batches_per_source = max(len(loader) for loader in loaders.values())
    batches_per_source = int(batches_per_source)
    if batches_per_source <= 0:
        return

    rng = random.Random(seed)
    iterators = {name: iter(loader) for name, loader in loaders.items()}

    for _ in range(batches_per_source):
        round_names = list(names)
        rng.shuffle(round_names)
        for name in round_names:
            try:
                batch = next(iterators[name])
            except StopIteration:
                iterators[name] = iter(loaders[name])
                batch = next(iterators[name])
            yield name, batch
