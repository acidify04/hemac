#!/usr/bin/env python3
"""Train a JOINT heterogeneous BC policy on synchronized D3O1 + D4O2 data.

Separate BC remains unchanged. This adds a joint baseline that processes Drone
and Observer from the same episode/time step in one model:

    Observer obs -> role encoder --\
                                  > shared agent Transformer -> role heads
    Drone obs    -> role encoder --/

Loss:
    L = 0.5 * L_observer + 0.5 * L_drone

D3O1 and D4O2 contribute equal minibatch counts. There is no HiSSD skill, value,
planner, task loss, adapter, agent-ID embedding, or population-size embedding.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from skill_discovery import train_hissd_joint_hetero_adapter as joint_utils
from skill_discovery.decentralized_bc_models import DecentralizedBehaviorCloningPolicy, ROLE_ORDER
from skill_discovery.multi_source_hissd import SourceSpec, balanced_source_batches


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest-31", type=Path, required=True)
    p.add_argument("--data-root-31", type=Path, required=True)
    p.add_argument("--manifest-42", type=Path, required=True)
    p.add_argument("--data-root-42", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)

    # Match the current separate-BC run unless explicitly changed.
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--sequence-length", type=int, default=16)
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--dataset-cache-size", type=int, default=16)
    p.add_argument("--batches-per-source", type=int, default=0)
    p.add_argument(
        "--balanced-sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Default=True to match the already-trained separate BC checkpoints.",
    )

    
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--max-train-batches", type=int)
    p.add_argument("--max-val-batches", type=int)
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def role_config(sample: dict[str, Any], role: str) -> dict[str, Any]:
    obs = sample[role]["observations"]
    actions = sample[role]["actions"]
    return {
        "global_map_channels": int(obs["global_map"].shape[-3]),
        "local_map_channels": int(obs["local_map"].shape[-3]),
        "global_map_size": tuple(int(x) for x in obs["global_map"].shape[-2:]),
        "local_map_size": tuple(int(x) for x in obs["local_map"].shape[-2:]),
        "action_history_shape": tuple(int(x) for x in obs["action_history"].shape[-2:]),
        "hidden_sizes": (96, 96),
        "activation": "relu",
        "action_dim": int(actions.shape[-1]),
    }


def validate_source_schemas(datasets: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
    reference = None
    counts: dict[str, dict[str, int]] = {}
    for source_name, dataset in datasets.items():
        sample = dataset[0]
        cfg = {role: role_config(sample, role) for role in ROLE_ORDER}
        counts[source_name] = {
            role: int(sample[role]["actions"].shape[-2])
            for role in ROLE_ORDER
        }
        if reference is None:
            reference = cfg
        elif cfg != reference:
            raise ValueError(
                "D3O1/D4O2 role observation/action schemas differ.\n"
                f"reference={json.dumps(reference, default=list, sort_keys=True)}\n"
                f"{source_name}={json.dumps(cfg, default=list, sort_keys=True)}"
            )
    assert reference is not None
    return reference, counts


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    numeric = mask.to(prediction.dtype).unsqueeze(-1)
    denominator = (numeric.sum() * target.shape[-1]).clamp_min(1.0)
    return ((prediction - target).square() * numeric).sum() / denominator


def masked_mae(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    numeric = mask.to(prediction.dtype).unsqueeze(-1)
    denominator = (numeric.sum() * target.shape[-1]).clamp_min(1.0)
    return ((prediction - target).abs() * numeric).sum() / denominator


def batch_objective(model, batch):
    outputs = model.forward_joint(
        batch["observations_by_role"],
        batch["valid_agents_by_role"],
    )
    role_mse = {}
    role_mae = {}
    for role in ROLE_ORDER:
        role_mse[role] = masked_mse(
            outputs["actions"][role],
            batch["actions_by_role"][role],
            batch["valid_agents_by_role"][role],
        )
        role_mae[role] = masked_mae(
            outputs["actions"][role],
            batch["actions_by_role"][role],
            batch["valid_agents_by_role"][role],
        )
    loss = 0.5 * (role_mse["observer"] + role_mse["drone"])
    mae = 0.5 * (role_mae["observer"] + role_mae["drone"])
    return loss, {
        "mse": float(loss.detach()),
        "mae": float(mae.detach()),
        "observer_mse": float(role_mse["observer"].detach()),
        "drone_mse": float(role_mse["drone"].detach()),
        "observer_mae": float(role_mae["observer"].detach()),
        "drone_mae": float(role_mae["drone"].detach()),
    }


def accumulate(acc, metrics):
    for k, v in metrics.items():
        acc[k] += float(v)


def average(acc, count):
    if count <= 0:
        raise RuntimeError("No batches processed")
    return {k: v / count for k, v in acc.items()}


def source_average(by_source):
    keys = set.intersection(*(set(v) for v in by_source.values()))
    return {
        k: sum(v[k] for v in by_source.values()) / len(by_source)
        for k in keys
    }


def train_epoch(model, loaders, optimizer, device, args, epoch):
    model.train()
    total = defaultdict(float)
    per_source = {name: defaultdict(float) for name in loaders}
    counts = defaultdict(int)
    processed = 0

    batches_per_source = (
        None if args.batches_per_source <= 0 else args.batches_per_source
    )
    iterator = balanced_source_batches(
        dict(loaders),
        seed=args.seed + epoch,
        batches_per_source=batches_per_source,
    )

    for source_name, raw_batch in iterator:
        if args.max_train_batches is not None and processed >= args.max_train_batches:
            break
        batch = joint_utils.prepare_joint_batch(raw_batch, device)
        loss, metrics = batch_objective(model, batch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        metrics["grad_norm"] = float(grad_norm)

        accumulate(total, metrics)
        accumulate(per_source[source_name], metrics)
        counts[source_name] += 1
        processed += 1

    return average(total, processed), {
        name: average(per_source[name], counts[name])
        for name in loaders if counts[name] > 0
    }


@torch.inference_mode()
def validate(model, loaders, device, args):
    model.eval()
    by_source = {}
    for source_name, loader in loaders.items():
        acc = defaultdict(float)
        count = 0
        for raw_batch in loader:
            if args.max_val_batches is not None and count >= args.max_val_batches:
                break
            batch = joint_utils.prepare_joint_batch(raw_batch, device)
            _, metrics = batch_objective(model, batch)
            accumulate(acc, metrics)
            count += 1
        by_source[source_name] = average(acc, count)
    return source_average(by_source), by_source


def save_checkpoint(path, model, optimizer, epoch, train_metrics, train_by_source, val_metrics, val_by_source, population_counts, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized_args = vars(args).copy()
    torch.save(
        {
            "format_version": 1,
            "model_type": "hemac_decentralized_behavior_cloning",
            "model_config": model.config(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch),
            "metrics": {
                "train": dict(train_metrics),
                "validation": dict(val_metrics),
                "train_by_source": {k: dict(v) for k, v in train_by_source.items()},
                "validation_by_source": {k: dict(v) for k, v in val_by_source.items()},
            },
            "architecture": {
                "role_specific_observation_encoders": True,
                "shared_agent_transformer": False,
                "decentralized_action_path": True,
                "other_agent_private_observation": False,
                "role_specific_action_heads": True,
                "skills": False,
                "planner": False,
                "adapter": False,
                "agent_id_embedding": False,
                "population_embedding": False,
                "role_balanced_loss": True,
            },
            "action_normalization": {
                "drone": "divide by per-episode drone_max_speed",
                "observer": "divide by per-episode observer_speed",
            },
            "multisource": {
                "configuration_balancing": "equal_batches_per_source",
                "outcome_balanced_sampling": bool(args.balanced_sampling),
                "population_counts": population_counts,
                "sources": {
                    "3-1": {"manifest": str(args.manifest_31), "data_root": str(args.data_root_31)},
                    "4-2": {"manifest": str(args.manifest_42), "data_root": str(args.data_root_42)},
                },
            },
            "args": serialized_args,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    for name in ("epochs", "batch_size", "sequence_length"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_','-')} must be positive")
    if args.batches_per_source < 0:
        raise ValueError("--batches-per-source cannot be negative")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    pin_memory = device.type == "cuda"

    sources = {
        "3-1": SourceSpec("3-1", args.manifest_31, args.data_root_31),
        "4-2": SourceSpec("4-2", args.manifest_42, args.data_root_42),
    }
    train_datasets, train_loaders = {}, {}
    val_datasets, val_loaders = {}, {}
    for index, (name, source) in enumerate(sources.items()):
        train_datasets[name], train_loaders[name] = joint_utils.create_joint_dataloader(
            source,
            "source_train",
            args,
            seed=args.seed + index * 1000,
            pin_memory=pin_memory,
        )
        val_datasets[name], val_loaders[name] = joint_utils.create_joint_dataloader(
            source,
            "source_val",
            args,
            seed=args.seed + 10_000 + index * 1000,
            pin_memory=pin_memory,
        )

    model_config, population_counts = validate_source_schemas(train_datasets)
    val_config, _ = validate_source_schemas(val_datasets)
    if val_config != model_config:
        raise ValueError("Train/validation joint schemas differ")

    model = DecentralizedBehaviorCloningPolicy(
        role_configs=model_config,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    parameter_count = sum(p.numel() for p in model.parameters())
    effective_bps = (
        max(len(loader) for loader in train_loaders.values())
        if args.batches_per_source <= 0
        else args.batches_per_source
    )

    print(f"device={device} parameters={parameter_count:,}", flush=True)
    print(f"population_counts={population_counts}", flush=True)
    print(
        f"configuration_balancing=equal_batches_per_source "
        f"batches_per_source={effective_bps} "
        f"outcome_balanced_sampling={args.balanced_sampling}",
        flush=True,
    )
    print(
        "architecture=role_specific_encoder + NO_cross_agent_mixing + role_specific_head",
        flush=True,
    )
    print("loss=0.5*observer_mse + 0.5*drone_mse", flush=True)

    best_val = float("inf")
    best_epoch = 0
    epoch_iterator = range(1, args.epochs + 1)
    if tqdm is not None and not args.no_progress:
        epoch_iterator = tqdm(
            epoch_iterator,
            total=args.epochs,
            desc="decentralized BC multisource",
            unit="epoch",
            dynamic_ncols=True,
        )

    for epoch in epoch_iterator:
        train_metrics, train_by_source = train_epoch(
            model, train_loaders, optimizer, device, args, epoch
        )
        val_metrics, val_by_source = validate(model, val_loaders, device, args)

        line = (
            f"epoch={epoch:03d} train_mse={train_metrics['mse']:.6f} "
            f"val_mse={val_metrics['mse']:.6f} "
            f"obs={val_metrics['observer_mse']:.6f} "
            f"drone={val_metrics['drone_mse']:.6f} "
            f"val_3-1={val_by_source['3-1']['mse']:.6f} "
            f"val_4-2={val_by_source['4-2']['mse']:.6f}"
        )
        if tqdm is not None and not args.no_progress:
            tqdm.write(line)
            epoch_iterator.set_postfix(
                train=f"{train_metrics['mse']:.4f}",
                val=f"{val_metrics['mse']:.4f}",
            )
        else:
            print(line, flush=True)

        save_checkpoint(
            args.output_dir / "decentralized_bc_last.pt",
            model,
            optimizer,
            epoch,
            train_metrics,
            train_by_source,
            val_metrics,
            val_by_source,
            population_counts,
            args,
        )
        if val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]
            best_epoch = epoch
            save_checkpoint(
                args.output_dir / "decentralized_bc_best.pt",
                model,
                optimizer,
                epoch,
                train_metrics,
                train_by_source,
                val_metrics,
                val_by_source,
                population_counts,
                args,
            )

    print(
        f"Best equal-source decentralized validation MSE: {best_val:.6f} at epoch {best_epoch} "
        f"({args.output_dir / 'joint_bc_best.pt'})",
        flush=True,
    )


if __name__ == "__main__":
    main()
