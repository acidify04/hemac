"""Train HiSSD or Skill-VAE on fixed MaMuJoCo source-task datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .dataset import build_dataloader
from .env import DEFAULT_ENVIRONMENT_VERSION, recorded_environment_version
from .happo import resolve_device, seed_everything
from .offline_models import (
    HISSD_ALPHA,
    HISSD_BETA,
    HISSD_EXPECTILE,
    build_offline_model,
)
from .tasks import AGENT_IDS, SUPPORTED_SUITES


DEFAULT_DATA_ROOT = Path("src/mamujoco/offline_data")
DEFAULT_OUTPUT_ROOT = Path("src/mamujoco/checkpoints/offline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("hissd", "skill_vae"), required=True)
    parser.add_argument("--suite", choices=SUPPORTED_SUITES, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--training-steps", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=1)
    parser.add_argument("--history-length", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target-update-rate", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=HISSD_ALPHA)
    parser.add_argument("--beta", type=float, default=HISSD_BETA)
    parser.add_argument("--expectile", type=float, default=HISSD_EXPECTILE)
    parser.add_argument("--gradient-norm-clip", type=float, default=10.0)
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def _to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def infer_dimensions(manifest_path: Path):
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    first_task = next(iter(manifest["tasks"].values()))
    first_entry = first_task["episodes"][0]
    data_root = manifest_path.parents[1]
    payload = torch.load(
        data_root / first_entry["path"], map_location="cpu", weights_only=False
    )
    observation_dims = {
        agent: int(payload["observations"][agent].shape[-1]) for agent in AGENT_IDS
    }
    action_dims = {
        agent: int(payload["actions"][agent].shape[-1]) for agent in AGENT_IDS
    }
    state_dim = int(payload["states"].shape[-1])
    return observation_dims, action_dims, state_dim


def save_checkpoint(path: Path, model, args, step: int, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "algorithm": args.algorithm,
            "suite": args.suite,
            "step": step,
            "seed": args.seed,
            "observation_dims": model.observation_dims,
            "action_dims": model.action_dims,
            "state_dim": model.state_dim,
            "model_config": getattr(model, "model_config", {}),
            "model": model.state_dict(),
            "metrics": metrics,
            "config": vars(args),
            "environment_version": args.environment_version,
            "backend": f"gymnasium_robotics.mamujoco_v1/{args.environment_version}",
        },
        path,
    )


def main() -> None:
    args = parse_args()
    for name in (
        "training_steps",
        "batch_size",
        "sequence_length",
        "history_length",
        "checkpoint_every",
        "alpha",
        "beta",
        "expectile",
        "gradient_norm_clip",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.expectile >= 1:
        raise ValueError("--expectile must be strictly between 0 and 1")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    manifest_path = args.manifest or args.data_root / args.suite / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    args.environment_version = (
        recorded_environment_version(manifest) or DEFAULT_ENVIRONMENT_VERSION
    )
    if args.suite == "difficulty":
        protocol = manifest.get("difficulty_protocol", {})
        target_names = set(protocol.get("target_difficulties", ()))
        leaked = target_names & set(manifest.get("tasks", {}))
        if leaked:
            raise ValueError(
                f"Target difficulty leakage in offline data: {sorted(leaked)}"
            )
    num_source_tasks = len(manifest["tasks"])
    observation_dims, action_dims, state_dim = infer_dimensions(manifest_path)
    model = build_offline_model(
        args.algorithm,
        observation_dims,
        action_dims,
        state_dim,
        num_source_tasks=num_source_tasks,
        history_length=args.history_length,
        training_sequence_length=args.sequence_length,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    dataloader = build_dataloader(
        manifest_path,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        history_length=args.history_length,
        num_workers=args.num_workers,
    )
    iterator = iter(dataloader)
    output_dir = args.output_root / args.suite / args.algorithm / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "training_metrics.jsonl"

    for step in range(1, args.training_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)
        batch = _to_device(batch, device)
        if args.algorithm == "hissd":
            loss, metrics = model.loss(
                batch,
                gamma=args.gamma,
                alpha=args.alpha,
                beta=args.beta,
                expectile=args.expectile,
            )
        else:
            loss, metrics = model.loss(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.gradient_norm_clip
        )
        optimizer.step()
        if args.algorithm == "hissd":
            model.update_target(args.target_update_rate)

        if step == 1 or step % 100 == 0:
            record = {
                "step": step,
                **{key: float(value.cpu()) for key, value in metrics.items()},
            }
            with metrics_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, sort_keys=True) + "\n")
        if step % args.checkpoint_every == 0 or step == args.training_steps:
            scalar_metrics = {
                key: float(value.cpu()) for key, value in metrics.items()
            }
            save_checkpoint(output_dir / "latest.pt", model, args, step, scalar_metrics)
            save_checkpoint(
                output_dir / f"checkpoint_{step:07d}.pt",
                model,
                args,
                step,
                scalar_metrics,
            )
            print(
                f"algorithm={args.algorithm} suite={args.suite} "
                f"step={step} loss={scalar_metrics['loss']:.6f}"
            )


if __name__ == "__main__":
    main()
