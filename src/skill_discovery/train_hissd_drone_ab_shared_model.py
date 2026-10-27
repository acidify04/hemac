#!/usr/bin/env python3
"""Train the shared-encoder/decoder HiSSD **architecture** for HeMAC A+B.

This script is intentionally separate from train_hissd_joint_hetero_scratch_resume.py.
It still reuses the previous HeMAC learning objectives (NOT the official
MoCo/IQL/planner update sequence), but
trains on synchronized A/B drone-only episodes collected by
collect_offline_data_drone_ab_vanilla.py.

IMPORTANT role aliases for the existing architecture:
    HiSSD 'observer' role = Drone A (not a physical observer)
    HiSSD 'drone'    role = Drone B

Requires the original project skill_discovery modules (not bundled here).
No BC weights, common-skill adapters, or expert-policy weights are loaded.
The expert RLlib checkpoint is only used by the separate collection script.

Example:
    python -u src/skill_discovery/train_hissd_drone_ab_vanilla.py \
      --data-root src/skill_discovery/offline_data_drone_ab_vanilla \
      --output-dir src/skill_discovery/checkpoints/hissd-drone-ab-vanilla \
      --epochs 120 --batch-size 4 --sequence-length 128
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from skill_discovery import train_hissd_joint_hetero_scratch_resume as vanilla
from skill_discovery import train_hissd_multisource as multi
from skill_discovery import train_hissd_joint_hetero_adapter as joint_utils
from skill_discovery import train_hissd_hetero_baseline as base
from skill_discovery.hissd_joint_hetero_models_paper import JointHeterogeneousHiSSD

ROLE_ALIAS = {"observer": "A", "drone": "B"}
ROLE_ORDER = ("observer", "drone")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--difficulty", type=int, default=None,
                   help="Optional source difficulty; default: all difficulty_XX folders")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--sequence-length", type=int, default=128)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--dataset-cache-size", type=int, default=12)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--expert-only", action="store_true",
                   help="Only train/validate on labeled success episodes")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--task-learning-rate-multiplier", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--task-weight-decay", type=float, default=None)
    p.add_argument("--task-warmup-epochs", type=int, default=None)
    p.add_argument("--task-contrastive-weight", type=float, default=None)
    p.add_argument("--task-action-contrastive-weight", type=float, default=None)
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-val-batches", type=int, default=None)
    p.add_argument("--early-stopping-patience", type=int, default=0)
    p.add_argument("--resume-checkpoint", type=Path, default=None)
    p.add_argument("--resume-lr-scale", type=float, default=1.0)
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def load_episode(path: Path):
    # mmap minimizes disk reads during upfront trajectory indexing.
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, RuntimeError):
        return torch.load(path, map_location="cpu", weights_only=False)


@dataclass(frozen=True)
class EpisodeSpec:
    path: Path
    length: int
    seed: int
    category: str
    action_scales: tuple[float, float]
    difficulty: int


def validate_and_describe(path: Path) -> EpisodeSpec:
    ep = load_episode(path)
    meta = ep.get("metadata", {})
    aliases = meta.get("role_aliases")
    if aliases != {"observer": "drone_A", "drone": "drone_B"}:
        raise ValueError(f"Not a drone A/B aliased episode: {path}: {aliases}")
    cfg = meta.get("environment_config", {})
    if cfg.get("n_observers") != 0 or cfg.get("drone_only_task_mode") != "maximize_coverage":
        raise ValueError(f"Dataset environment is not drone-only maximize_coverage: {path}")
    if ep["observer"]["actions"].shape[1] < 1 or ep["drone"]["actions"].shape[1] < 1:
        raise ValueError(f"Episode missing A/B agents: {path}")
    drone_cfg = cfg.get("drone_config", {})
    types = [str(x).upper() for x in drone_cfg.get("drone_types", [])]
    assert len(types) == ep["agent_mask"].shape[-1], (path, types)
    # Critical: on-disk agent ordering must match A-first then B for mask slicing.
    expected_order = ep["observer"]["agent_ids"] + ep["drone"]["agent_ids"]
    if expected_order != meta.get("agent_order"):
        raise ValueError(f"A/B agent_order is not concatenated A then B: {path}")
    type_cfgs = drone_cfg.get("drone_type_configs", {})
    def speed(t):
        return float(type_cfgs.get(t, type_cfgs.get(t.lower(), {})).get(
            "max_speed", drone_cfg.get("drone_max_speed", 16.0)))
    scales = (speed("A"), speed("B"))
    if min(scales) <= 0:
        raise ValueError(f"Invalid action scale: {scales}")
    T = ep["observer"]["actions"].shape[0]
    if ep["drone"]["actions"].shape[0] != T or ep["global_state"]["central_map"].shape[0] != T + 1:
        raise ValueError(f"Misaligned A/B joint time axes: {path}")
    if T < 1:
        raise ValueError(f"Empty episode: {path}")
    return EpisodeSpec(path, T, int(meta.get("seed", -1)), str(ep["outcome"]["category"]), scales,
                       int(meta.get("difficulty", -1)))


class DroneABWindows(Dataset):
    """Episode-disjoint, time-aligned batches preserving original HiSSD input keys."""
    def __init__(self, episodes: list[EpisodeSpec], seq_len: int, stride: int, cache_size: int,
                 task_ids: dict[int, int] | None = None):
        self.episodes = episodes
        self.task_ids = task_ids or {episodes[0].difficulty: 0}
        self.seq_len = seq_len
        self.index = [(i, start) for i, ep in enumerate(episodes) for start in range(0, ep.length, stride)]
        self._cache: OrderedDict[Path, dict[str, Any]] = OrderedDict()
        self.cache_size = max(cache_size, 1)
        if not self.index:
            raise ValueError("No sequence windows found")

    def __len__(self):
        return len(self.index)

    def _get(self, path):
        if path not in self._cache:
            self._cache[path] = load_episode(path)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        self._cache.move_to_end(path)
        return self._cache[path]

    def __getitem__(self, ix):
        episode_index, start = self.index[ix]
        ep_info = self.episodes[episode_index]
        ep = self._get(ep_info.path)
        end = min(start + self.seq_len, ep_info.length)
        valid = end - start

        def slice_pad(tensor: torch.Tensor, offset: int = 0, pad_value=0):
            x = tensor[start + offset:end + offset].to(dtype=(
                torch.float32 if tensor.is_floating_point() else tensor.dtype))
            if valid == self.seq_len:
                return x.contiguous()
            padding = torch.full((self.seq_len - valid, *x.shape[1:]), pad_value, dtype=x.dtype)
            return torch.cat([x, padding], dim=0).contiguous()

        result: dict[str, Any] = {}
        for role, scale in zip(ROLE_ORDER, ep_info.action_scales):
            role_data = ep[role]
            obs = role_data["observations"]
            current_obs = {k: slice_pad(v) for k, v in obs.items()
                           if k in ("global_map", "local_map", "action_history")}
            next_obs = {k: slice_pad(v, 1) for k, v in obs.items()
                        if k in ("global_map", "local_map", "action_history")}
            # The shared encoder expects physical action histories in the same
            # normalized [-1, 1] convention as its continuous target actions.
            # The original collection file stores history in environment units.
            for mapped_obs in (current_obs, next_obs):
                mapped_obs["action_history"] = (
                    mapped_obs["action_history"] / scale
                ).clamp(-1, 1)
            result[role] = {
                "observations": current_obs,
                "next_observations": next_obs,
                "actions": (slice_pad(role_data["actions"]) / scale).clamp(-1, 1),
            }
        result["global_state"] = {"central_map": slice_pad(ep["global_state"]["central_map"])}
        result["next_global_state"] = {"central_map": slice_pad(ep["global_state"]["central_map"], 1)}
        for key in ("team_reward", "agent_mask", "terminated", "truncated"):
            result[key] = slice_pad(ep[key])
        filled = torch.zeros((self.seq_len, 1), dtype=torch.bool)
        filled[:valid] = True
        result["filled"] = filled
        result["window_start"] = torch.tensor(start, dtype=torch.long)
        result["task_id"] = torch.tensor(self.task_ids[ep_info.difficulty], dtype=torch.long)
        result["task_descriptor"] = ep["task_descriptor"].to(torch.float32).clone()
        result["task_descriptor_available"] = torch.tensor(True, dtype=torch.bool)
        return result


def split_episodes(all_eps: list[EpisodeSpec], val_fraction: float, seed: int):
    if len(all_eps) < 2:
        raise ValueError("Need at least 2 episodes for episode-disjoint train/validation")
    train, val = [], []
    by_difficulty: dict[int, list[EpisodeSpec]] = {}
    for ep in all_eps:
        by_difficulty.setdefault(ep.difficulty, []).append(ep)
    for diff, items in sorted(by_difficulty.items()):
        if len(items) < 2:
            raise ValueError(f"Need >=2 episodes for difficulty {diff} (found {len(items)})")
        random.Random(seed + diff).shuffle(items)
        val_n = min(len(items) - 1, max(1, round(len(items) * val_fraction)))
        train.extend(items[val_n:])
        val.extend(items[:val_n])
    return train, val


def save_checkpoint(path, model, optimizer, epoch, args, cli, train_metrics, val_metrics, best):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": 1,
        "model_type": "hemac_joint_hissd_shared_ab_architecture",
        "model_config": model.config(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "skill_structure": model.skill_structure,
        "architecture": {
            "roles": list(ROLE_ORDER),
            "role_aliases": ROLE_ALIAS,
            "role_specific_observation_encoders": False,
            "shared_observation_encoder": True,
            "agent_type_embedding": bool(model.use_agent_type_embedding),
            "role_specific_action_decoders": False,
            "shared_action_decoder": True,
            "shared_common_skill_encoder": True,
            "shared_task_skill_encoder": True,
            "adapter_count": 0,
            "role_embedding": False,
            "agent_id_embedding": False,
            "population_embedding": False,
            "planner_uses_raw_common_skill": True,
            "controller_uses_raw_common_skill": True,
            "role_balanced_action_loss": True,  # inherited training objective, NOT official
            "role_balanced_planner_local_loss": True,  # inherited training objective, NOT official
        },
        "initialization": "scratch", "bc_initialization": None,
        "source_environment": "drone_only/maximize_coverage",
        "data_root": str(cli.data_root),
        "training_args": vars(args),
        "training_metrics": dict(train_metrics),
        "validation_metrics": dict(val_metrics),
        "best_validation_loss": best,
    }, path)


def main():
    cli = parse_args()
    if not (cli.epochs > 0 and cli.batch_size > 0 and cli.sequence_length > 0 and cli.stride > 0):
        raise ValueError("epochs/batch-size/sequence-length/stride must be positive")
    if not (0 < cli.val_fraction < 1):
        raise ValueError("--val-fraction must be in (0,1)")
    if cli.resume_lr_scale <= 0 or not math.isfinite(cli.resume_lr_scale):
        raise ValueError("--resume-lr-scale must be finite and positive")
    multi.seed_everything(cli.seed)
    device = multi.resolve_device(cli.device)
    if hasattr(base, "configure_gpu_backend"):
        base.configure_gpu_backend(device)

    root = cli.data_root.expanduser().resolve()
    if cli.difficulty is None:
        files = sorted(root.glob("difficulty_*/*/*.pt"))
    else:
        files = sorted((root / f"difficulty_{cli.difficulty:02d}").glob("*/*.pt"))
    if not files:
        raise FileNotFoundError(f"No episodes in {root}; run the A/B collector first")
    eps = [validate_and_describe(f) for f in files]
    if cli.expert_only:
        eps = [ep for ep in eps if ep.category == "success"]
    # One fixed A/B population schema per training run; prevent silent mistakes.
    speeds = {ep.action_scales for ep in eps}
    if len(speeds) != 1:
        raise ValueError(f"Multiple A/B speed configurations found: {speeds}")
    train_eps, val_eps = split_episodes(eps, cli.val_fraction, cli.seed)
    task_ids = {diff: i for i, diff in enumerate(sorted({ep.difficulty for ep in eps}))}
    train_ds = DroneABWindows(train_eps, cli.sequence_length, cli.stride, cli.dataset_cache_size, task_ids)
    val_ds = DroneABWindows(val_eps, cli.sequence_length, cli.stride, cli.dataset_cache_size, task_ids)
    if len(train_ds) < cli.batch_size:
        raise ValueError(f"Only {len(train_ds)} train windows; lower --batch-size or collect more episodes")
    loader_kwargs = dict(batch_size=cli.batch_size, num_workers=cli.num_workers,
                         pin_memory=device.type == "cuda", persistent_workers=cli.num_workers > 0)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)

    args = vanilla.configure_training_args(cli)
    sample = train_ds[0]
    role_configs = {role: vanilla.role_config_from_sample(sample, role) for role in ROLE_ORDER}
    central_shape = sample["global_state"]["central_map"].shape
    metadata = load_episode(train_eps[0].path)["metadata"]
    args.task_descriptor_names = metadata.get("task_descriptor_names", [])
    args.training_tasks = sorted(task_ids)
    args.held_out_tasks = []
    # Task descriptor supervision uses all training episodes only (avoids val leakage).
    descs = torch.stack([load_episode(ep.path)["task_descriptor"].float() for ep in train_eps])
    args.task_descriptor_mean = descs.mean(dim=0).tolist()
    args.task_descriptor_scale = descs.std(dim=0, unbiased=False).clamp_min(
        float(args.descriptor_scale_floor)).tolist()
    if len(task_ids) == 1 and args.task_contrastive_enabled:
        print("[info] Only one task ID; disabling cross-task contrastive classifier", flush=True)
        args.task_contrastive_enabled = False
        args.task_contrastive_weight = 0.0
        args.task_action_contrastive_weight = 0.0

    model = JointHeterogeneousHiSSD(
        role_configs=role_configs,
        central_map_channels=int(central_shape[-3]),
        central_map_size=tuple(int(x) for x in central_shape[-2:]),
        reference_observer_count=int(sample["observer"]["actions"].shape[-2]),
        reference_drone_count=int(sample["drone"]["actions"].shape[-2]),
        hidden_dim=args.hidden_dim, skill_dim=args.skill_dim,
        transformer_heads=args.transformer_heads,
        contrastive_from_action_skill=False, task_context_pooling=False,
        task_descriptor_dim=(int(sample["task_descriptor"].numel()) if args.descriptor_enabled else 0),
        task_prior_count=(len(task_ids) if args.task_contrastive_enabled else 0),
        task_dropout=args.task_dropout, task_feature_deltas=True,
        learned_task_classifier=args.task_contrastive_enabled,
        normalize_task_context=False, task_running_statistics=True,
        direct_task_summary=False, task_action_residual=False, skill_structure="split",
    ).to(device)
    joint_utils.validate_joint_source_compatibility({"AB": train_ds}, model)
    optimizer, _, _ = multi.optimizer_for_model(model, args)
    start_epoch = 1
    if cli.resume_checkpoint:
        state = torch.load(cli.resume_checkpoint, map_location=device, weights_only=False)
        if state.get("model_type") != "hemac_joint_hissd_shared_ab_architecture" or state.get("architecture", {}).get("role_aliases") != ROLE_ALIAS:
            raise ValueError("Resume checkpoint is not A/B vanilla HiSSD")
        if state.get("model_config") != model.config():
            raise ValueError("Resume model architecture differs from current A/B dataset")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] *= cli.resume_lr_scale
        start_epoch = int(state["epoch"]) + 1

    output_dir = cli.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_metadata = {
        "role_aliases": ROLE_ALIAS,
        "action_scales": {"A": eps[0].action_scales[0], "B": eps[0].action_scales[1]},
        "train_episodes": [str(e.path.relative_to(root)) for e in train_eps],
        "validation_episodes": [str(e.path.relative_to(root)) for e in val_eps],
        "seed": cli.seed,
        "task_ids": task_ids,
        "notes": "Episode-disjoint split; no BC initialization; no adapter; full horizon",
    }
    (output_dir / "data_split.json").write_text(json.dumps(split_metadata, indent=2))
    print(f"[HiSSD A/B] device={device} roles=observer->A({len(sample['observer']['actions'][0])}), "
          f"drone->B({len(sample['drone']['actions'][0])}), "
          f"train_ep={len(train_eps)} val_ep={len(val_eps)} "
          f"train_windows={len(train_ds)} val_windows={len(val_ds)}", flush=True)
    print("[HiSSD A/B] shared encoder / common skill / task skill / controller, no BC, no adapter", flush=True)
    print("[WARN] This trainer still uses CUSTOM HeMAC objectives, NOT original HiSSD MoCo/IQL/planner training. Do not report as official algorithm reproduction.", flush=True)
    print("[HiSSD A/B] legacy D3O1/D4O2 rollout evaluation DISABLED; requires A/B evaluator", flush=True)

    best = math.inf
    history_path = output_dir / "train_history.jsonl"
    iterator = range(start_epoch, cli.epochs + 1)
    if tqdm is not None and not cli.no_progress:
        iterator = tqdm(iterator, desc="Shared-architecture HiSSD drone A/B", unit="epoch", dynamic_ncols=True)
    for epoch in iterator:
        warmup = epoch <= args.task_warmup_epochs
        train_metrics, _ = vanilla.train_epoch(
            model, {"AB": train_loader}, optimizer, device, args,
            epoch=epoch, batches_per_source=None, task_only=warmup,
        )
        val_metrics, _ = vanilla.validate_all_sources(model, {"AB": val_loader}, device, args)
        combined_loss, task_loss = base.validation_losses(val_metrics, args)
        score = task_loss if (args.descriptor_enabled or args.task_contrastive_enabled) else combined_loss
        if not math.isfinite(score):
            raise RuntimeError(f"Non-finite validation score at epoch {epoch}: {score}")
        record = {"epoch": epoch, "train": train_metrics, "validation": val_metrics,
                  "validation_combined_loss": combined_loss, "validation_task_loss": task_loss}
        with history_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        save_checkpoint(output_dir / "hissd_drone_ab_last.pt", model, optimizer,
                        epoch, args, cli, train_metrics, val_metrics, min(best, score))
        if score < best:
            best = score
            save_checkpoint(output_dir / "hissd_drone_ab_best.pt", model, optimizer,
                            epoch, args, cli, train_metrics, val_metrics, best)
        log = f"epoch={epoch}/{cli.epochs} val={combined_loss:.6f} selected={score:.6f} best={best:.6f} " \
              f"train[{vanilla.compact(train_metrics)}] val[{vanilla.compact(val_metrics)}]"
        if tqdm is not None and not cli.no_progress:
            tqdm.write(log)
            iterator.set_postfix(train=f"{train_metrics['action_mse']:.4f}", val=f"{score:.4f}")
        else:
            print(log, flush=True)
    print(f"[done] best_validation={best:.6f}, output={output_dir}", flush=True)


if __name__ == "__main__":
    main()
