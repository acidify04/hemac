#!/usr/bin/env python3
"""
Audit why HiSSD task-specific learning fails on the 4-2 drone source dataset.

Checks:
  1) source_train/source_val task balance and task-id consistency
  2) realized task-descriptor separability by task
  3) frozen task-observation feature linear probe
  4) frozen learned task-context linear probe
  5) current checkpoint task-classifier accuracy
  6) tiny-set MLP overfit on task-observation features

This script does NOT modify any checkpoint or training file.

Expected repository layout:
  src/skill_discovery/dataset.py
  src/skill_discovery/hissd_models.py
  src/skill_discovery/train_hissd_hetero_baseline.py
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from skill_discovery.dataset import create_dataloader
from skill_discovery.hissd_models import HeMACHISSD


DEFAULT_MANIFEST = (
    PROJECT_ROOT / "src/skill_discovery/offline_data-4-2/dataset_splits.json"
)
DEFAULT_DATA_ROOT = PROJECT_ROOT / "src/skill_discovery/offline_data-4-2"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "src/skill_discovery/checkpoints/hissd-4-2/drone/hissd_best.pt"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--sequence-length", type=int, default=128)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=2026)

    p.add_argument("--probe-epochs", type=int, default=300)
    p.add_argument("--probe-lr", type=float, default=1e-2)
    p.add_argument("--probe-weight-decay", type=float, default=1e-4)

    p.add_argument("--tiny-per-task", type=int, default=16)
    p.add_argument("--tiny-epochs", type=int, default=500)
    p.add_argument("--tiny-lr", type=float, default=3e-3)

    p.add_argument(
        "--max-windows",
        type=int,
        default=0,
        help="0 = use all windows. Positive value caps each split for a faster smoke test.",
    )
    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable.")
    return torch.device(name)


def move_drone_observations(
    observations: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Move the decentralized drone observation tensors used by HeMAC HiSSD."""
    return {
        name: observations[name].to(device, non_blocking=True)
        for name in ("global_map", "local_map", "action_history")
    }


def prepare_drone_batch(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Self-contained drone batch conversion.

    This intentionally does not import train_hissd_hetero_baseline.prepare_batch
    because that helper's signature differs across repo revisions
    (some versions require an explicit role argument).
    """
    actions = batch["drone"]["actions"].to(device, non_blocking=True)
    drone_count = actions.shape[-2]

    agent_mask = batch["agent_mask"][..., -drone_count:].to(
        device, non_blocking=True
    )
    filled = batch["filled"].squeeze(-1).to(
        device, non_blocking=True
    ).bool()
    valid_agents = agent_mask.bool() & filled.unsqueeze(-1)

    terminated = batch["terminated"].squeeze(-1).to(
        device, non_blocking=True
    )
    truncated = batch["truncated"].squeeze(-1).to(
        device, non_blocking=True
    )

    return {
        "observations": move_drone_observations(
            batch["drone"]["observations"], device
        ),
        "next_observations": move_drone_observations(
            batch["drone"]["next_observations"], device
        ),
        "actions": actions,
        "central_map": batch["global_state"]["central_map"].to(
            device, non_blocking=True
        ),
        "next_central_map": batch["next_global_state"]["central_map"].to(
            device, non_blocking=True
        ),
        "team_reward": batch["drone_task_reward"].squeeze(-1).to(
            device, non_blocking=True
        ),
        "task_descriptor": batch["task_descriptor"].to(
            device, non_blocking=True
        ),
        "task_descriptor_available": batch[
            "task_descriptor_available"
        ].to(device, non_blocking=True).bool(),
        "task_id": batch["task_id"].to(device, non_blocking=True),
        "task_supervision": (
            batch["window_start"].to(device, non_blocking=True) == 0
        ),
        "valid_agents": valid_agents,
        "valid_steps": filled,
        "done": terminated.bool() | truncated.bool(),
    }


def make_loader(
    manifest: Path,
    data_root: Path,
    split: str,
    *,
    sequence_length: int,
    stride: int,
    batch_size: int,
    seed: int,
):
    # Match the source HiSSD runner as closely as possible, but keep num_workers=0
    # so this diagnostic is deterministic and easy to debug.
    return create_dataloader(
        manifest_path=manifest,
        split=split,
        data_root=data_root,
        sequence_length=sequence_length,
        stride=stride,
        batch_size=batch_size,
        num_workers=0,
        normalize_actions=True,
        include_observer=False,
        include_labels=False,
        balanced_sampling=True,
        task_balanced_batches=True,
        seed=seed,
        pin_memory=False,
        drop_last_batch=False,
        cache_size=16,
    )


def task_from_entry(entry: dict[str, Any]) -> int | str:
    for key in ("difficulty", "task_id", "task", "task_name"):
        if key in entry:
            value = entry[key]
            try:
                return int(value)
            except (TypeError, ValueError):
                return str(value)
    return "<unknown>"


def category_from_entry(entry: dict[str, Any]) -> str:
    for key in ("category", "outcome", "quality"):
        if key in entry and entry[key] is not None:
            return str(entry[key])
    path = str(entry.get("path", entry.get("file", "")))
    for candidate in ("success", "goal_found_failure", "goal_not_found"):
        if candidate in path:
            return candidate
    return "<unknown>"


def print_dataset_entry_audit(dataset, split: str) -> None:
    print(f"\n{'=' * 78}")
    print(f"[1] DATASET ENTRY AUDIT: {split}")
    print(f"{'=' * 78}")
    print(f"windows={len(dataset)}  entries={len(getattr(dataset, 'entries', []))}")

    entries = getattr(dataset, "entries", [])
    if entries:
        by_task = Counter(task_from_entry(x) for x in entries)
        by_category = Counter(category_from_entry(x) for x in entries)
        pair = Counter(
            (task_from_entry(x), category_from_entry(x))
            for x in entries
        )
        print("entry_count_by_task:", dict(sorted(by_task.items(), key=lambda x: str(x[0]))))
        print("entry_count_by_category:", dict(sorted(by_category.items())))
        print("entry_count_by_task_category:")
        for key, value in sorted(pair.items(), key=lambda x: (str(x[0][0]), x[0][1])):
            print(f"  {key}: {value}")


def collect_sample_metadata(dataset, max_windows: int = 0):
    task_counts = Counter()
    descriptors = defaultdict(list)
    descriptor_available = Counter()
    task_id_to_names = defaultdict(set)
    task_id_to_entry_difficulty = defaultdict(set)

    count = len(dataset) if max_windows <= 0 else min(len(dataset), max_windows)
    for index in range(count):
        sample = dataset[index]
        tid = int(torch.as_tensor(sample["task_id"]).item())
        task_counts[tid] += 1

        if "task_name" in sample:
            task_id_to_names[tid].add(str(sample["task_name"]))

        available = bool(torch.as_tensor(
            sample.get("task_descriptor_available", True)
        ).item())
        descriptor_available[(tid, available)] += 1

        if "task_descriptor" in sample:
            descriptors[tid].append(
                torch.as_tensor(sample["task_descriptor"], dtype=torch.float32)
                .reshape(-1)
                .cpu()
            )

        entries = getattr(dataset, "entries", None)
        windows = getattr(dataset, "windows", None)
        if entries is not None and windows is not None and index < len(windows):
            window = windows[index]
            entry_index = getattr(window, "entry_index", None)
            if entry_index is None:
                entry_index = getattr(window, "episode_index", None)
            if entry_index is not None and 0 <= int(entry_index) < len(entries):
                entry = entries[int(entry_index)]
                if "difficulty" in entry:
                    task_id_to_entry_difficulty[tid].add(int(entry["difficulty"]))

    return {
        "task_counts": task_counts,
        "descriptors": descriptors,
        "descriptor_available": descriptor_available,
        "task_id_to_names": task_id_to_names,
        "task_id_to_entry_difficulty": task_id_to_entry_difficulty,
        "count": count,
    }


def descriptor_audit(meta: dict[str, Any], split: str) -> dict[str, float]:
    print(f"\n{'=' * 78}")
    print(f"[2] TASK-ID / DESCRIPTOR AUDIT: {split}")
    print(f"{'=' * 78}")
    print("window_task_counts:", dict(sorted(meta["task_counts"].items())))
    print("descriptor_available:", dict(sorted(meta["descriptor_available"].items())))

    if meta["task_id_to_names"]:
        print("task_id_to_names:", {
            k: sorted(v) for k, v in sorted(meta["task_id_to_names"].items())
        })
    if meta["task_id_to_entry_difficulty"]:
        print("task_id_to_entry_difficulty:", {
            k: sorted(v) for k, v in sorted(meta["task_id_to_entry_difficulty"].items())
        })

    means = {}
    stds = {}
    for tid, values in sorted(meta["descriptors"].items()):
        if not values:
            continue
        x = torch.stack(values)
        means[tid] = x.mean(dim=0)
        stds[tid] = x.std(dim=0, unbiased=False)
        print(f"task={tid} descriptor_mean={means[tid].tolist()}")
        print(f"task={tid} descriptor_std ={stds[tid].tolist()}")

    result = {}
    tids = sorted(means)
    if len(tids) == 2:
        a, b = tids
        distance = torch.linalg.vector_norm(means[a] - means[b]).item()

        all_values = torch.cat(
            [torch.stack(meta["descriptors"][tid]) for tid in tids],
            dim=0,
        )
        global_scale = all_values.std(dim=0, unbiased=False).clamp_min(1e-8)
        normalized_distance = torch.linalg.vector_norm(
            (means[a] - means[b]) / global_scale
        ).item()

        max_abs_diff = (means[a] - means[b]).abs().max().item()
        exact_same = bool(torch.equal(means[a], means[b]))

        print(f"descriptor_centroid_l2={distance:.6f}")
        print(f"descriptor_centroid_standardized_l2={normalized_distance:.6f}")
        print(f"descriptor_centroid_max_abs_diff={max_abs_diff:.6f}")
        print(f"descriptor_centroids_exactly_same={exact_same}")

        result.update(
            descriptor_l2=distance,
            descriptor_standardized_l2=normalized_distance,
            descriptor_max_abs_diff=max_abs_diff,
        )
    return result


def load_model(checkpoint: Path, device: torch.device) -> tuple[HeMACHISSD, dict[str, Any]]:
    payload = torch.load(
        checkpoint.expanduser().resolve(),
        map_location=device,
        weights_only=False,
    )
    if "model_config" not in payload or "model_state_dict" not in payload:
        raise ValueError(
            f"Checkpoint does not contain model_config/model_state_dict: {checkpoint}"
        )
    model = HeMACHISSD(**payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def pool_last_agent_mean(
    features: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool [B,T,A,D] features at each sequence's last valid step over agents."""
    if features.ndim != 4 or valid_mask.ndim != 3:
        raise ValueError(
            f"Expected features [B,T,A,D], mask [B,T,A], got "
            f"{tuple(features.shape)}, {tuple(valid_mask.shape)}"
        )
    step_valid = valid_mask.bool().any(dim=2)
    valid_windows = step_valid.any(dim=1)
    last_idx = step_valid.long().sum(dim=1).sub(1).clamp_min(0)
    batch_idx = torch.arange(features.shape[0], device=features.device)
    last_feat = features[batch_idx, last_idx]
    last_mask = valid_mask[batch_idx, last_idx].bool()
    m = last_mask.unsqueeze(-1).to(last_feat.dtype)
    pooled = (last_feat * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
    return pooled, valid_windows


@torch.no_grad()
def extract_representations(
    model: HeMACHISSD,
    loader,
    device: torch.device,
    *,
    max_windows: int = 0,
):
    task_input_features = []
    task_latent_contexts = []
    labels = []
    current_logits = []

    seen = 0
    for raw_batch in loader:
        batch = prepare_drone_batch(raw_batch, device)

        task_features = model.encode_task_observations(batch["observations"])
        _, contrastive = model.task_skill_encoder(
            task_features,
            batch["valid_agents"],
        )

        input_context, input_valid = pool_last_agent_mean(
            task_features,
            batch["valid_agents"],
        )
        latent_context, latent_valid = model.pool_task_context(
            contrastive,
            batch["valid_agents"],
        )
        valid = input_valid & latent_valid

        y = batch["task_id"]
        task_input_features.append(input_context[valid].detach().cpu())
        task_latent_contexts.append(latent_context[valid].detach().cpu())
        labels.append(y[valid].detach().cpu())

        if model.task_classifier_head is not None:
            current_logits.append(
                model.task_classifier_head(latent_context[valid]).detach().cpu()
            )

        seen += int(valid.sum().item())
        if max_windows > 0 and seen >= max_windows:
            break

    x_input = torch.cat(task_input_features, dim=0)
    x_latent = torch.cat(task_latent_contexts, dim=0)
    y = torch.cat(labels, dim=0).long()

    if max_windows > 0:
        x_input = x_input[:max_windows]
        x_latent = x_latent[:max_windows]
        y = y[:max_windows]

    logits = None
    if current_logits:
        logits = torch.cat(current_logits, dim=0)
        if max_windows > 0:
            logits = logits[:max_windows]

    return x_input, x_latent, y, logits


def standardize_train_val(
    x_train: torch.Tensor,
    x_val: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return (x_train - mean) / std, (x_val - mean) / std


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == labels).float().mean().item())


def train_probe(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    hidden_dim: int | None,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
):
    seed_everything(seed)
    classes = int(max(y_train.max().item(), y_val.max().item()) + 1)

    x_train, x_val = standardize_train_val(x_train.float(), x_val.float())
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_val = x_val.to(device)
    y_val = y_val.to(device)

    if hidden_dim is None:
        model = nn.Linear(x_train.shape[-1], classes).to(device)
    else:
        model = nn.Sequential(
            nn.Linear(x_train.shape[-1], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, classes),
        ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_val = 0.0
    best_train = 0.0
    for _ in range(epochs):
        model.train()
        logits = model(x_train)
        loss = F.cross_entropy(logits, y_train)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            train_acc = accuracy(model(x_train), y_train)
            val_acc = accuracy(model(x_val), y_val)
            best_train = max(best_train, train_acc)
            best_val = max(best_val, val_acc)

    with torch.no_grad():
        model.eval()
        final_train = accuracy(model(x_train), y_train)
        final_val = accuracy(model(x_val), y_val)

    return {
        "best_train": best_train,
        "best_val": best_val,
        "final_train": final_train,
        "final_val": final_val,
    }


def balanced_tiny_subset(
    x: torch.Tensor,
    y: torch.Tensor,
    per_task: int,
    seed: int,
):
    generator = torch.Generator().manual_seed(seed)
    indices = []
    for tid in sorted(y.unique().tolist()):
        pool = torch.nonzero(y == tid, as_tuple=False).flatten()
        perm = pool[torch.randperm(len(pool), generator=generator)]
        take = min(per_task, len(perm))
        indices.append(perm[:take])
    idx = torch.cat(indices)
    idx = idx[torch.randperm(len(idx), generator=generator)]
    return x[idx], y[idx]


def report_current_classifier(
    logits: torch.Tensor | None,
    labels: torch.Tensor,
    split: str,
) -> float | None:
    if logits is None:
        print(f"checkpoint_task_classifier_{split}=N/A (no learned classifier head)")
        return None
    acc = accuracy(logits, labels)
    ce = float(F.cross_entropy(logits, labels).item())
    conf = float(logits.softmax(dim=-1).max(dim=-1).values.mean().item())
    print(
        f"checkpoint_task_classifier_{split}: "
        f"acc={acc:.4f} ce={ce:.4f} mean_conf={conf:.4f}"
    )
    return acc


def interpretation(
    descriptor_stats: dict[str, float],
    input_probe: dict[str, float],
    latent_probe: dict[str, float],
    tiny_probe: dict[str, float],
    current_val_acc: float | None,
) -> None:
    print(f"\n{'=' * 78}")
    print("[6] AUTOMATIC INTERPRETATION")
    print(f"{'=' * 78}")

    d = descriptor_stats.get("descriptor_standardized_l2", float("nan"))
    input_val = input_probe["best_val"]
    latent_val = latent_probe["best_val"]
    tiny_train = tiny_probe["best_train"]

    print(
        f"summary: descriptor_std_l2={d:.3f}, "
        f"input_linear_val={input_val:.3f}, "
        f"latent_linear_val={latent_val:.3f}, "
        f"tiny_mlp_train={tiny_train:.3f}, "
        f"checkpoint_classifier_val={current_val_acc if current_val_acc is not None else float('nan'):.3f}"
    )

    if not math.isnan(d) and d < 0.25:
        print(
            "DIAGNOSIS A: 두 source task의 descriptor centroid가 매우 가깝습니다. "
            "먼저 4-2 task definition / realized descriptor 생성이 실제로 다른지 확인하세요."
        )

    if tiny_train < 0.90:
        print(
            "DIAGNOSIS B: tiny MLP조차 task-observation feature를 잘 외우지 못합니다. "
            "task label 전달 오류, feature에 task 신호 부재, 또는 aggregation에서 신호 소실을 우선 의심하세요."
        )
    elif input_val < 0.65:
        print(
            "DIAGNOSIS C: tiny subset은 외우지만 validation input-feature probe가 낮습니다. "
            "source task 간 일반화 가능한 관측 차이가 약하거나 split별 분포가 다를 가능성이 큽니다."
        )
    elif latent_val + 0.10 < input_val:
        print(
            "DIAGNOSIS D: task encoder 입력에서는 task가 구분되지만 learned task latent에서 정보가 크게 사라집니다. "
            "task_skill_encoder / contrastive objective / pooling을 우선 수정해야 합니다."
        )
    elif current_val_acc is not None and latent_val >= 0.70 and current_val_acc < 0.60:
        print(
            "DIAGNOSIS E: latent 자체는 구분 가능하지만 기존 classifier head만 chance 수준입니다. "
            "classifier optimization, loss weight, label smoothing, classifier gradient 연결을 점검하세요."
        )
    elif input_val >= 0.70 and latent_val >= 0.70:
        print(
            "DIAGNOSIS F: 데이터와 representation에는 task 구분 신호가 있습니다. "
            "기존 4-2 학습 실패는 task auxiliary optimization 쪽일 가능성이 높습니다. "
            "warmup 연장과 task loss weight 조정을 다음 실험으로 권장합니다."
        )

    print("\nRecommended next action:")
    if tiny_train < 0.90 or input_val < 0.65:
        print("  1) 모델 재학습 전에 dataset/task definition과 observation aggregation부터 수정")
    elif latent_val + 0.10 < input_val:
        print("  1) task_skill_encoder/pooling 진단 후 수정")
        print("  2) 수정 전에는 단순 epoch 증가 금지")
    else:
        print("  1) task_warmup 10 -> 25")
        print("  2) task classification/contrastive weight를 우선 2x")
        print("  3) 별도 ablation으로 task_action_gradient=True")
        print("  4) 한 번에 여러 변경을 섞지 말 것")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)

    manifest = args.manifest.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()

    print("=" * 78)
    print("HiSSD 4-2 TASK-BRANCH AUDIT")
    print("=" * 78)
    print(f"manifest   : {manifest}")
    print(f"data_root  : {data_root}")
    print(f"checkpoint : {checkpoint}")
    print(f"device     : {device}")

    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if not data_root.exists():
        raise FileNotFoundError(data_root)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    print("manifest source_difficulties:", payload.get("source_difficulties"))
    print("manifest target_difficulties:", payload.get("target_difficulties"))
    if "task_descriptor" in payload:
        print("manifest task_descriptor schema:", payload["task_descriptor"])

    train_dataset, train_loader = make_loader(
        manifest,
        data_root,
        "source_train",
        sequence_length=args.sequence_length,
        stride=args.stride,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    val_dataset, val_loader = make_loader(
        manifest,
        data_root,
        "source_val",
        sequence_length=args.sequence_length,
        stride=args.stride,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    print_dataset_entry_audit(train_dataset, "source_train")
    print_dataset_entry_audit(val_dataset, "source_val")

    train_meta = collect_sample_metadata(train_dataset, args.max_windows)
    val_meta = collect_sample_metadata(val_dataset, args.max_windows)
    train_desc_stats = descriptor_audit(train_meta, "source_train")
    _ = descriptor_audit(val_meta, "source_val")

    if not checkpoint.is_file():
        print("\nCheckpoint not found; dataset audit completed, model probes skipped.")
        print(f"Missing: {checkpoint}")
        return

    print(f"\n{'=' * 78}")
    print("[3] LOAD CHECKPOINT AND EXTRACT REPRESENTATIONS")
    print(f"{'=' * 78}")
    model, ckpt_payload = load_model(checkpoint, device)
    print("checkpoint_epoch:", ckpt_payload.get("epoch"))
    print("skill_structure:", getattr(model, "skill_structure", "<unknown>"))
    print("model_agent_count:", getattr(model, "agent_count", "<unknown>"))
    print("skill_dim:", getattr(model, "skill_dim", "<unknown>"))

    train_input, train_latent, train_y, train_logits = extract_representations(
        model,
        train_loader,
        device,
        max_windows=args.max_windows,
    )
    val_input, val_latent, val_y, val_logits = extract_representations(
        model,
        val_loader,
        device,
        max_windows=args.max_windows,
    )

    print("train representations:", tuple(train_input.shape), tuple(train_latent.shape))
    print("val representations  :", tuple(val_input.shape), tuple(val_latent.shape))
    print("train labels:", dict(Counter(train_y.tolist())))
    print("val labels  :", dict(Counter(val_y.tolist())))

    train_current = report_current_classifier(train_logits, train_y, "train")
    val_current = report_current_classifier(val_logits, val_y, "val")

    print(f"\n{'=' * 78}")
    print("[4] LINEAR PROBES")
    print(f"{'=' * 78}")

    input_probe = train_probe(
        train_input,
        train_y,
        val_input,
        val_y,
        hidden_dim=None,
        epochs=args.probe_epochs,
        lr=args.probe_lr,
        weight_decay=args.probe_weight_decay,
        seed=args.seed,
        device=device,
    )
    print("task-observation INPUT feature linear probe:", input_probe)

    latent_probe = train_probe(
        train_latent,
        train_y,
        val_latent,
        val_y,
        hidden_dim=None,
        epochs=args.probe_epochs,
        lr=args.probe_lr,
        weight_decay=args.probe_weight_decay,
        seed=args.seed + 1,
        device=device,
    )
    print("learned TASK LATENT linear probe:", latent_probe)

    print(f"\n{'=' * 78}")
    print("[5] TINY-SET NONLINEAR OVERFIT")
    print(f"{'=' * 78}")
    tiny_x, tiny_y = balanced_tiny_subset(
        train_input,
        train_y,
        args.tiny_per_task,
        args.seed,
    )
    # Train and evaluate on the exact same tiny set. This is intentional:
    # failure to memorize is a strong sign that the pooled feature lacks task signal.
    tiny_probe = train_probe(
        tiny_x,
        tiny_y,
        tiny_x,
        tiny_y,
        hidden_dim=128,
        epochs=args.tiny_epochs,
        lr=args.tiny_lr,
        weight_decay=0.0,
        seed=args.seed + 2,
        device=device,
    )
    print(
        f"tiny samples={len(tiny_y)} "
        f"counts={dict(Counter(tiny_y.tolist()))}"
    )
    print("tiny MLP overfit:", tiny_probe)

    interpretation(
        train_desc_stats,
        input_probe,
        latent_probe,
        tiny_probe,
        val_current,
    )


if __name__ == "__main__":
    main()
