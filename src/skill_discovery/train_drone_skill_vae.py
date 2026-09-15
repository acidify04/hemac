"""Train a single-latent skill VAE on homogeneous drone-only trajectories."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
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
from skill_discovery.drone_skill_vae import DroneSkillVAE


DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "src/skill_discovery/offline_data_drone_d12_t34_cp7800/"
    "drone_task_dataset_splits.json"
)
DEFAULT_BC_CHECKPOINT = (
    PROJECT_ROOT
    / "src/skill_discovery/checkpoints/bc_drone_d12_t34_cp7800/"
    "drone_bc_best.pt"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "src/skill_discovery/checkpoints/drone_skill_vae"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--bc-checkpoint", type=Path, default=DEFAULT_BC_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--dataset-cache-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--temporal-hidden-dim", type=int, default=96)
    parser.add_argument("--decoder-hidden-dim", type=int, default=64)
    parser.add_argument("--residual-logit-scale", type=float, default=0.2)
    parser.add_argument("--skill-duration", type=int, default=8)
    parser.add_argument(
        "--decoder-observation-conditioned",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--kl-beta", type=float, default=0.002)
    parser.add_argument("--kl-warmup-epochs", type=int, default=20)
    parser.add_argument("--kl-free-bits", type=float, default=0.005)
    parser.add_argument("--dynamics-weight", type=float, default=0.10)
    parser.add_argument("--usage-weight", type=float, default=0.2)
    parser.add_argument("--usage-margin", type=float, default=0.02)
    parser.add_argument("--action-anchor-weight", type=float, default=0.10)
    parser.add_argument("--smoothness-weight", type=float, default=0.02)
    parser.add_argument("--latent-variance-weight", type=float, default=0.10)
    parser.add_argument("--latent-std-target", type=float, default=0.20)
    parser.add_argument("--decorrelation-weight", type=float, default=0.01)
    parser.add_argument(
        "--bc-warmup-epochs",
        type=int,
        default=5,
        help="Initially freeze the BC observation/base-action path.",
    )
    parser.add_argument(
        "--freeze-bc-path",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep the pretrained observation encoder and base action head fixed "
            "so all reconstruction improvement must use the latent residual."
        ),
    )
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--no-tensorboard", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "epochs",
        "batch_size",
        "sequence_length",
        "stride",
        "learning_rate",
        "latent_dim",
        "temporal_hidden_dim",
        "decoder_hidden_dim",
        "residual_logit_scale",
        "skill_duration",
        "grad_clip",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.num_workers < 0 or args.dataset_cache_size <= 0:
        raise ValueError("Worker count cannot be negative and cache size must be positive.")
    for name in (
        "weight_decay",
        "kl_beta",
        "kl_free_bits",
        "dynamics_weight",
        "usage_weight",
        "usage_margin",
        "action_anchor_weight",
        "smoothness_weight",
        "latent_variance_weight",
        "latent_std_target",
        "decorrelation_weight",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative.")
    if args.kl_warmup_epochs < 0 or args.bc_warmup_epochs < 0:
        raise ValueError("Warm-up epochs cannot be negative.")


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_backend(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def build_model(sample: dict[str, Any], args: argparse.Namespace) -> DroneSkillVAE:
    observations = sample["drone"]["observations"]
    global_map = observations["global_map"]
    local_map = observations["local_map"]
    action_history = observations["action_history"]
    actions = sample["drone"]["actions"]
    return DroneSkillVAE(
        global_map_channels=global_map.shape[-3],
        local_map_channels=local_map.shape[-3],
        action_dim=actions.shape[-1],
        global_map_size=tuple(global_map.shape[-2:]),
        local_map_size=tuple(local_map.shape[-2:]),
        action_history_shape=tuple(action_history.shape[-2:]),
        temporal_hidden_dim=args.temporal_hidden_dim,
        latent_dim=args.latent_dim,
        decoder_hidden_dim=args.decoder_hidden_dim,
        residual_logit_scale=args.residual_logit_scale,
        skill_duration=args.skill_duration,
        decoder_observation_conditioned=args.decoder_observation_conditioned,
    )


def to_device_tree(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: to_device_tree(item, device) for key, item in value.items()}
    return value


def valid_agent_mask(batch: dict[str, Any]) -> torch.Tensor:
    return batch["agent_mask"].bool() & batch["filled"].bool()


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    expanded = mask.unsqueeze(-1).expand_as(prediction)
    return (prediction - target).square()[expanded].mean()


def shuffled_skills(skills: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    result = skills.clone()
    valid = skills[mask]
    if valid.shape[0] > 1:
        result[mask] = valid[torch.randperm(valid.shape[0], device=valid.device)]
    return result


def objective(
    model: DroneSkillVAE,
    batch: dict[str, Any],
    args: argparse.Namespace,
    *,
    beta: float,
    sample_latent: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    observations = batch["drone"]["observations"]
    next_observations = batch["drone"]["next_observations"]
    target_actions = batch["drone"]["actions"]
    mask = valid_agent_mask(batch)
    outputs = model(
        observations,
        sample_latent=sample_latent,
        skill_offsets=batch.get("window_start"),
    )

    reconstruction = masked_mse(outputs["actions"], target_actions, mask)
    base_actions = torch.tanh(
        model.base_action_head(outputs["observation_features"])
    )
    base_mse = masked_mse(base_actions, target_actions, mask)

    mu = outputs["skill_mu"]
    logvar = outputs["skill_logvar"]
    kl_by_dimension = -0.5 * (1.0 + logvar - mu.square() - logvar.exp())
    decision_mask = mask & outputs["skill_decision_mask"].unsqueeze(-1)
    valid_kl = kl_by_dimension[decision_mask]
    raw_kl = valid_kl.sum(dim=-1).mean()
    free_bits_kl = valid_kl.mean(dim=0).clamp_min(args.kl_free_bits).sum()

    with torch.no_grad():
        next_features = model.encode_observations(next_observations)
    predicted_next = model.predict_next_features(
        outputs["observation_features"], outputs["skills"]
    )
    transition_mask = mask & ~batch["terminated"].bool().expand_as(mask)
    if transition_mask.any():
        dynamics = masked_mse(predicted_next, next_features, transition_mask)
    else:
        dynamics = reconstruction.new_zeros(())

    # Measure skill usage on posterior means because online execution is
    # deterministic and also uses mu rather than a noisy posterior sample.
    mean_actions = model.decode_actions(outputs["observation_features"], mu)
    alternative_skills = shuffled_skills(mu, mask)
    alternative_actions = model.decode_actions(
        outputs["observation_features"], alternative_skills
    )
    sensitivity_by_agent = (
        mean_actions - alternative_actions
    ).abs().mean(dim=-1)
    sensitivity = sensitivity_by_agent[mask].mean()
    usage = F.relu(args.usage_margin - sensitivity_by_agent[mask]).mean()
    action_anchor = masked_mse(mean_actions, base_actions, mask)

    skill_changes = (
        decision_mask[:, 1:] & mask[:, :-1] & mask[:, 1:]
    )
    if skill_changes.any():
        skill_delta = (mu[:, 1:] - mu[:, :-1]).square().mean(dim=-1)
        smoothness = skill_delta[skill_changes].mean()
    else:
        smoothness = reconstruction.new_zeros(())

    loss = (
        reconstruction
        + beta * free_bits_kl
        + args.dynamics_weight * dynamics
        + args.usage_weight * usage
        + args.action_anchor_weight * action_anchor
        + args.smoothness_weight * smoothness
    )
    valid_mu = mu[decision_mask]
    latent_std_by_dimension = valid_mu.std(dim=0, unbiased=False)
    latent_variance_loss = F.relu(
        args.latent_std_target - latent_std_by_dimension
    ).mean()
    centered_mu = valid_mu - valid_mu.mean(dim=0, keepdim=True)
    denominator = max(valid_mu.shape[0] - 1, 1)
    covariance = centered_mu.transpose(0, 1) @ centered_mu / denominator
    off_diagonal = covariance - torch.diag_embed(torch.diagonal(covariance))
    decorrelation = off_diagonal.square().sum() / max(args.latent_dim, 1)
    latent_variance = latent_std_by_dimension.square()
    loss = (
        loss
        + args.latent_variance_weight * latent_variance_loss
        + args.decorrelation_weight * decorrelation
    )
    metrics = {
        "loss": float(loss.detach()),
        "reconstruction": float(reconstruction.detach()),
        "base_mse": float(base_mse.detach()),
        "reconstruction_gain": float((base_mse - reconstruction).detach()),
        "kl_raw": float(raw_kl.detach()),
        "kl_free_bits": float(free_bits_kl.detach()),
        "dynamics": float(dynamics.detach()),
        "usage": float(usage.detach()),
        "action_sensitivity": float(sensitivity.detach()),
        "action_anchor": float(action_anchor.detach()),
        "smoothness": float(smoothness.detach()),
        "latent_variance_loss": float(latent_variance_loss.detach()),
        "decorrelation": float(decorrelation.detach()),
        "latent_std": float(valid_mu.std(unbiased=False).detach()),
        "posterior_std": float(
            torch.exp(0.5 * logvar[mask]).mean().detach()
        ),
        "active_units": float((latent_variance > 1e-2).sum().detach()),
        "skill_decision_fraction": float(
            (
                decision_mask.sum(dtype=torch.float32)
                / mask.sum(dtype=torch.float32).clamp_min(1.0)
            ).detach()
        ),
        "beta": float(beta),
    }
    return loss, metrics


def set_bc_path_trainable(model: DroneSkillVAE, trainable: bool) -> None:
    for parameter in model.observation_encoder.parameters():
        parameter.requires_grad_(trainable)
    for parameter in model.base_action_head.parameters():
        parameter.requires_grad_(trainable)


def optimizer_to_device(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    """Move optimizer moments restored from a CPU checkpoint to the model device."""
    for state in optimizer.state.values():
        for name, value in state.items():
            if isinstance(value, torch.Tensor):
                state[name] = value.to(device)


def run_epoch(
    model: DroneSkillVAE,
    loader,
    device: torch.device,
    args: argparse.Namespace,
    *,
    beta: float,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
) -> dict[str, float]:
    model.train(optimizer is not None)
    totals: dict[str, float] = defaultdict(float)
    batch_count = 0
    for batch_index, raw_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = to_device_tree(raw_batch, device)
        loss, metrics = objective(
            model,
            batch,
            args,
            beta=beta,
            sample_latent=optimizer is not None,
        )
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        for name, value in metrics.items():
            totals[name] += value
        batch_count += 1
    if batch_count == 0:
        raise RuntimeError("No batches were processed.")
    return {name: value / batch_count for name, value in totals.items()}


def save_checkpoint(
    path: Path,
    model: DroneSkillVAE,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    train_metrics: dict[str, float],
    validation_metrics: dict[str, float],
    args: argparse.Namespace,
    bc_payload: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 5,
            "model_type": "homogeneous_drone_skill_vae",
            "model_config": model.config(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "bc_checkpoint_epoch": bc_payload.get("epoch"),
            "objective": {
                "reconstruction": "masked_action_mse",
                "regularization": "standard_normal_kl_with_free_bits",
                "dynamics": "next_observation_feature_mse",
                "usage": "shuffled_latent_action_sensitivity_margin",
                "action_anchor": "skill_policy_to_frozen_bc_action_mse",
                "latent_variance": "posterior_mean_std_floor",
                "decorrelation": "posterior_mean_off_diagonal_covariance",
                "dynamics_inputs": "observation_feature_and_skill_without_action",
                "smoothness": "consecutive_posterior_mean_mse",
                "decoder": "frozen_bc_action_plus_observation_skill_residual",
                "skill_schedule": "episode_aligned_fixed_duration",
            },
            "hyperparameters": vars(args),
        },
        path,
    )


def create_writer(args: argparse.Namespace):
    if args.no_tensorboard:
        return None
    from torch.utils.tensorboard import SummaryWriter

    return SummaryWriter(log_dir=args.output_dir / "tensorboard")


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    configure_backend(device)
    pin_memory = device.type == "cuda"
    train_dataset, train_loader = create_dataloader(
        manifest_path=args.manifest,
        split="source_train",
        data_root=args.data_root,
        sequence_length=args.sequence_length,
        stride=args.stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        include_observer=False,
        include_labels=False,
        balanced_sampling=True,
        seed=args.seed,
        pin_memory=pin_memory,
        cache_size=args.dataset_cache_size,
    )
    val_dataset, val_loader = create_dataloader(
        manifest_path=args.manifest,
        split="source_val",
        data_root=args.data_root,
        sequence_length=args.sequence_length,
        stride=args.stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        include_observer=False,
        include_labels=False,
        balanced_sampling=False,
        seed=args.seed,
        pin_memory=pin_memory,
        drop_last_batch=False,
        cache_size=args.dataset_cache_size,
    )
    model = build_model(train_dataset[0], args).to(device)
    bc_payload = model.initialize_from_bc(args.bc_checkpoint)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    start_epoch = 1
    if args.resume_checkpoint is not None:
        payload = torch.load(
            args.resume_checkpoint.expanduser().resolve(),
            map_location="cpu",
            weights_only=False,
        )
        if int(payload.get("format_version", 0)) < 5:
            raise ValueError(
                "Cannot resume a pre-chunked skill-VAE checkpoint. Start a new "
                "run so the observation-conditioned decoder is trained from "
                "its BC initialization."
            )
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        optimizer_to_device(optimizer, device)
        start_epoch = int(payload["epoch"]) + 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    writer = create_writer(args)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"device={device}, train_windows={len(train_dataset)}, "
        f"val_windows={len(val_dataset)}, parameters={parameter_count:,}"
    )
    print(f"model_config={json.dumps(model.config())}")
    best_score = float("inf")
    stale_epochs = 0
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            set_bc_path_trainable(
                model,
                not args.freeze_bc_path and epoch > args.bc_warmup_epochs,
            )
            if args.kl_warmup_epochs == 0:
                beta = args.kl_beta
            else:
                beta = args.kl_beta * min(epoch / args.kl_warmup_epochs, 1.0)
            train_metrics = run_epoch(
                model,
                train_loader,
                device,
                args,
                beta=beta,
                optimizer=optimizer,
                max_batches=args.max_train_batches,
            )
            with torch.inference_mode():
                validation_metrics = run_epoch(
                    model,
                    val_loader,
                    device,
                    args,
                    beta=beta,
                    optimizer=None,
                    max_batches=args.max_val_batches,
                )
            print(
                f"epoch={epoch:03d} "
                f"loss={train_metrics['loss']:.5f}/{validation_metrics['loss']:.5f} "
                f"recon={train_metrics['reconstruction']:.5f}/"
                f"{validation_metrics['reconstruction']:.5f} "
                f"base={validation_metrics['base_mse']:.5f} "
                f"gain={validation_metrics['reconstruction_gain']:+.5f} "
                f"kl={validation_metrics['kl_raw']:.4f} beta={beta:.5f} "
                f"dyn={validation_metrics['dynamics']:.5f} "
                f"usage={validation_metrics['action_sensitivity']:.4f} "
                f"anchor={validation_metrics['action_anchor']:.5f} "
                f"z_std={validation_metrics['latent_std']:.4f} "
                f"z_var={validation_metrics['latent_variance_loss']:.4f} "
                f"select_rate={validation_metrics['skill_decision_fraction']:.3f} "
                f"active={validation_metrics['active_units']:.1f}/{args.latent_dim}"
            )
            if writer is not None:
                for prefix, values in (
                    ("train", train_metrics),
                    ("validation", validation_metrics),
                ):
                    for name, value in values.items():
                        writer.add_scalar(f"{prefix}/{name}", value, epoch)
            save_checkpoint(
                args.output_dir / "drone_skill_vae_last.pt",
                model,
                optimizer,
                epoch=epoch,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                args=args,
                bc_payload=bc_payload,
            )
            # Select for held-out action quality first. Dynamics is retained as
            # a small tie-breaker, while KL/usage remain training regularizers.
            score = (
                validation_metrics["reconstruction"]
                + 0.05 * validation_metrics["dynamics"]
                + 0.10 * validation_metrics["action_anchor"]
                + 0.02 * validation_metrics["latent_variance_loss"]
            )
            if score < best_score:
                best_score = score
                stale_epochs = 0
                save_checkpoint(
                    args.output_dir / "drone_skill_vae_best.pt",
                    model,
                    optimizer,
                    epoch=epoch,
                    train_metrics=train_metrics,
                    validation_metrics=validation_metrics,
                    args=args,
                    bc_payload=bc_payload,
                )
            else:
                stale_epochs += 1
            if (
                args.early_stopping_patience > 0
                and stale_epochs >= args.early_stopping_patience
            ):
                print(f"Early stopping after {stale_epochs} stale epochs.")
                break
    finally:
        if writer is not None:
            writer.close()
    print(
        f"Best validation objective: {best_score:.6f} "
        f"({args.output_dir / 'drone_skill_vae_best.pt'})"
    )


if __name__ == "__main__":
    main()
