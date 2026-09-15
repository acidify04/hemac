"""Visualize homogeneous drone VAE skills across environment difficulties."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from skill_discovery.dataset import create_dataloader
from skill_discovery.drone_skill_vae import load_drone_skill_vae


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "src/skill_discovery/checkpoints/drone_skill_vae_regularized/"
    "drone_skill_vae_best.pt"
)
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "src/skill_discovery/offline_data_drone_d12_t34_cp7800/"
    "drone_task_dataset_splits.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "src/skill_discovery/outputs/drone_skill_vae_difficulty_visualization.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("source_test", "target_test"),
        help="Manifest splits whose difficulties will be compared.",
    )
    parser.add_argument("--sequence-length", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-points-per-difficulty", type=int, default=1500)
    parser.add_argument("--max-points-per-episode", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    for name in (
        "sequence_length",
        "batch_size",
        "max_points_per_difficulty",
        "max_points_per_episode",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    return args


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def move_observations(
    observations: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        name: observations[name].to(device, non_blocking=True)
        for name in ("global_map", "local_map", "action_history")
    }


def evenly_spaced_indices(indices: torch.Tensor, count: int) -> torch.Tensor:
    """Select across the full episode instead of overrepresenting early cycles."""
    if indices.numel() <= count:
        return indices
    positions = torch.linspace(
        0, indices.numel() - 1, steps=count, device=indices.device
    ).round().long()
    return indices[positions]


@torch.inference_mode()
def collect_embeddings(model, args: argparse.Namespace, device: torch.device):
    chunks: dict[str, list[np.ndarray]] = {
        "skill": [],
        "posterior_std": [],
        "action_residual": [],
        "difficulty": [],
        "episode_group": [],
        "agent": [],
        "split": [],
    }
    difficulty_counts: dict[int, int] = {}
    episode_counts: dict[tuple[int, int, int], int] = {}

    for split_index, split in enumerate(args.splits):
        dataset, loader = create_dataloader(
            manifest_path=args.manifest,
            split=split,
            data_root=args.data_root,
            sequence_length=args.sequence_length,
            stride=args.sequence_length,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            normalize_actions=True,
            include_observer=False,
            include_labels=False,
            balanced_sampling=False,
            seed=args.seed,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
            drop_last_batch=False,
        )
        split_difficulties = sorted(
            {int(entry["difficulty"]) for entry in dataset.entries}
        )
        for difficulty in split_difficulties:
            difficulty_counts.setdefault(difficulty, 0)

        for batch in loader:
            observations = move_observations(batch["drone"]["observations"], device)
            drone_count = observations["global_map"].shape[2]
            valid = (
                batch["agent_mask"][..., -drone_count:].bool()
                & batch["filled"].squeeze(-1).bool().unsqueeze(-1)
            ).to(device)
            outputs = model(
                observations,
                sample_latent=False,
                skill_offsets=batch["window_start"].to(device),
            )
            skills = outputs["skill_mu"]
            posterior_std = (0.5 * outputs["skill_logvar"]).exp()
            action_residual = model.residual_logit_scale * model.skill_residual(
                outputs["observation_features"], skills
            )
            _, sequence_length, agent_count = skills.shape[:3]
            flat_agents = (
                torch.arange(agent_count, device=device)
                .view(1, agent_count)
                .expand(sequence_length, agent_count)
                .reshape(-1)
            )

            for batch_index in range(skills.shape[0]):
                difficulty = int(batch["difficulty"][batch_index])
                if difficulty_counts[difficulty] >= args.max_points_per_difficulty:
                    continue
                episode_index = int(batch["episode_index"][batch_index])
                episode_key = (split_index, difficulty, episode_index)
                episode_remaining = (
                    args.max_points_per_episode
                    - episode_counts.get(episode_key, 0)
                )
                difficulty_remaining = (
                    args.max_points_per_difficulty - difficulty_counts[difficulty]
                )
                take = min(episode_remaining, difficulty_remaining)
                if take <= 0:
                    continue
                flat_indices = valid[batch_index].reshape(-1).nonzero().squeeze(-1)
                if flat_indices.numel() == 0:
                    continue
                flat_indices = evenly_spaced_indices(flat_indices, take)
                actual = int(flat_indices.numel())
                episode_group = (
                    split_index * 10_000_000
                    + difficulty * 1_000_000
                    + episode_index
                )
                chunks["skill"].append(
                    skills[batch_index].reshape(-1, model.latent_dim)[flat_indices]
                    .cpu()
                    .numpy()
                )
                chunks["posterior_std"].append(
                    posterior_std[batch_index]
                    .reshape(-1, model.latent_dim)[flat_indices]
                    .cpu()
                    .numpy()
                )
                chunks["action_residual"].append(
                    action_residual[batch_index]
                    .reshape(-1, model.action_dim)[flat_indices]
                    .cpu()
                    .numpy()
                )
                chunks["difficulty"].append(
                    np.full(actual, difficulty, dtype=np.int64)
                )
                chunks["episode_group"].append(
                    np.full(actual, episode_group, dtype=np.int64)
                )
                chunks["agent"].append(flat_agents[flat_indices].cpu().numpy())
                chunks["split"].append(
                    np.full(actual, split_index, dtype=np.int64)
                )
                difficulty_counts[difficulty] += actual
                episode_counts[episode_key] = (
                    episode_counts.get(episode_key, 0) + actual
                )

    if not chunks["skill"]:
        raise RuntimeError("No valid VAE skill vectors were extracted.")
    arrays = {name: np.concatenate(values) for name, values in chunks.items()}
    arrays["split_names"] = np.asarray(args.splits)
    for difficulty in sorted(difficulty_counts):
        print(f"difficulty=D{difficulty}: points={difficulty_counts[difficulty]}")
    return arrays


def fit_pca_2d(values: np.ndarray):
    values64 = values.astype(np.float64)
    mean = values64.mean(axis=0, keepdims=True)
    centered = values64 - mean
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    components = right_vectors[:2]
    coordinates = centered @ components.T
    variance = np.square(singular_values)
    explained = variance[:2] / max(float(variance.sum()), np.finfo(float).eps)
    return coordinates.astype(np.float32), explained.astype(np.float32), mean, components


def episode_means(arrays: dict[str, np.ndarray]):
    groups = np.unique(arrays["episode_group"])
    means = []
    difficulties = []
    for group in groups:
        selected = arrays["episode_group"] == group
        means.append(arrays["skill"][selected].mean(axis=0))
        difficulties.append(int(arrays["difficulty"][selected][0]))
    return np.asarray(means), np.asarray(difficulties), groups


def grouped_linear_probe(
    values: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    train_groups: list[int] = []
    test_groups: list[int] = []
    for class_id in classes:
        class_groups = np.unique(groups[labels == class_id])
        rng.shuffle(class_groups)
        if class_groups.size < 2:
            raise ValueError(f"D{class_id} needs at least two episodes.")
        count = min(max(round(class_groups.size * 0.7), 1), class_groups.size - 1)
        train_groups.extend(class_groups[:count].tolist())
        test_groups.extend(class_groups[count:].tolist())
    train = np.isin(groups, train_groups)
    test = np.isin(groups, test_groups)
    train_x = values[train].astype(np.float64)
    test_x = values[test].astype(np.float64)
    mean = train_x.mean(axis=0, keepdims=True)
    scale = train_x.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    train_x = np.concatenate(((train_x - mean) / scale, np.ones((train.sum(), 1))), axis=1)
    test_x = np.concatenate(((test_x - mean) / scale, np.ones((test.sum(), 1))), axis=1)
    class_to_index = {int(value): index for index, value in enumerate(classes)}
    target = np.zeros((train.sum(), classes.size), dtype=np.float64)
    target[np.arange(train.sum()), [class_to_index[int(x)] for x in labels[train]]] = 1.0
    regularizer = np.eye(train_x.shape[1]) * 1e-2
    regularizer[-1, -1] = 0.0
    weights = np.linalg.solve(
        train_x.T @ train_x + regularizer, train_x.T @ target
    )
    prediction = classes[np.argmax(test_x @ weights, axis=1)]
    per_class = {
        str(int(class_id)): float((prediction[labels[test] == class_id] == class_id).mean())
        for class_id in classes
    }
    return {
        "accuracy": float((prediction == labels[test]).mean()),
        "balanced_accuracy": float(np.mean(list(per_class.values()))),
        "chance_accuracy": float(1.0 / classes.size),
        "per_difficulty_accuracy": per_class,
        "train_episodes": len(train_groups),
        "test_episodes": len(test_groups),
    }


def probe_results(arrays: dict[str, np.ndarray], seed: int) -> dict[str, Any]:
    results = {
        "all_difficulties": grouped_linear_probe(
            arrays["skill"], arrays["difficulty"], arrays["episode_group"], seed=seed
        )
    }
    for split_index, split_name in enumerate(arrays["split_names"]):
        selected = arrays["split"] == split_index
        if np.unique(arrays["difficulty"][selected]).size < 2:
            continue
        results[str(split_name)] = grouped_linear_probe(
            arrays["skill"][selected],
            arrays["difficulty"][selected],
            arrays["episode_group"][selected],
            seed=seed + split_index + 1,
        )
    return results


def calculate_statistics(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    difficulties = sorted(int(x) for x in np.unique(arrays["difficulty"]))
    centroids = []
    per_difficulty = {}
    for difficulty in difficulties:
        selected = arrays["difficulty"] == difficulty
        skill = arrays["skill"][selected]
        posterior_std = arrays["posterior_std"][selected]
        residual = arrays["action_residual"][selected]
        centroid = skill.mean(axis=0)
        centroids.append(centroid)
        per_difficulty[str(difficulty)] = {
            "points": int(selected.sum()),
            "episodes": int(np.unique(arrays["episode_group"][selected]).size),
            "skill_mean": centroid.tolist(),
            "skill_std": skill.std(axis=0).tolist(),
            "posterior_std_mean": posterior_std.mean(axis=0).tolist(),
            "action_residual_mean": residual.mean(axis=0).tolist(),
            "action_residual_abs_mean": np.abs(residual).mean(axis=0).tolist(),
        }
    centroids_array = np.asarray(centroids)
    centroid_distances = np.linalg.norm(
        centroids_array[:, None] - centroids_array[None, :], axis=-1
    )
    within = []
    for index, difficulty in enumerate(difficulties):
        selected = arrays["difficulty"] == difficulty
        within.extend(
            np.linalg.norm(arrays["skill"][selected] - centroids_array[index], axis=1)
        )
    off_diagonal = centroid_distances[~np.eye(len(difficulties), dtype=bool)]
    return {
        "difficulties": difficulties,
        "per_difficulty": per_difficulty,
        "centroid_distance_matrix": centroid_distances.tolist(),
        "mean_between_centroid_distance": float(off_diagonal.mean()),
        "mean_within_difficulty_distance": float(np.mean(within)),
        "between_within_ratio": float(off_diagonal.mean() / max(np.mean(within), 1e-8)),
    }


def plot(arrays: dict[str, np.ndarray], statistics: dict[str, Any], output: Path, show: bool):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/hemac_matplotlib")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    skills = arrays["skill"]
    difficulty = arrays["difficulty"]
    difficulties = statistics["difficulties"]
    point_pca, point_explained, pca_mean, pca_components = fit_pca_2d(skills)
    ep_skill, ep_difficulty, _ = episode_means(arrays)
    episode_pca = (ep_skill.astype(np.float64) - pca_mean) @ pca_components.T
    colors = plt.get_cmap("tab10")
    figure, axes = plt.subplots(2, 3, figsize=(20, 12), constrained_layout=True)

    for diff in difficulties:
        selected = difficulty == diff
        axes[0, 0].scatter(
            point_pca[selected, 0], point_pca[selected, 1], s=9, alpha=0.25,
            color=colors((diff - 1) % 10), edgecolors="none", label=f"D{diff}",
        )
        centroid = point_pca[selected].mean(axis=0)
        axes[0, 0].text(*centroid, f"D{diff}", fontweight="bold", fontsize=11)
        ep_selected = ep_difficulty == diff
        axes[0, 1].scatter(
            episode_pca[ep_selected, 0], episode_pca[ep_selected, 1], s=28,
            alpha=0.65, color=colors((diff - 1) % 10), edgecolors="none",
            label=f"D{diff}",
        )
    axes[0, 0].set_title("Posterior mean skill z (timestep samples)")
    axes[0, 1].set_title("Episode-mean skill z")
    for axis in axes[0, :2]:
        axis.set_xlabel(f"PC1 ({point_explained[0] * 100:.1f}%)")
        axis.set_ylabel(f"PC2 ({point_explained[1] * 100:.1f}%)")
        axis.grid(alpha=0.18)
        axis.legend()

    means = np.asarray([
        statistics["per_difficulty"][str(diff)]["skill_mean"]
        for diff in difficulties
    ])
    stds = np.asarray([
        statistics["per_difficulty"][str(diff)]["skill_std"]
        for diff in difficulties
    ])
    residuals = np.asarray([
        statistics["per_difficulty"][str(diff)]["action_residual_abs_mean"]
        for diff in difficulties
    ])
    distance = np.asarray(statistics["centroid_distance_matrix"])

    images = [
        axes[0, 2].imshow(means, aspect="auto", cmap="coolwarm"),
        axes[1, 0].imshow(stds, aspect="auto", cmap="YlGnBu", vmin=0),
        axes[1, 1].imshow(distance, aspect="equal", cmap="magma", vmin=0),
        axes[1, 2].imshow(residuals, aspect="auto", cmap="YlOrRd", vmin=0),
    ]
    titles = (
        "Mean latent value by difficulty",
        "Latent standard deviation by difficulty",
        "Distance between difficulty centroids",
        "Mean absolute skill action residual",
    )
    for axis, image, title in zip((axes[0, 2], *axes[1]), images, titles):
        axis.set_title(title)
        axis.set_yticks(range(len(difficulties)), [f"D{x}" for x in difficulties])
        figure.colorbar(image, ax=axis, fraction=0.046)
    axes[0, 2].set_xticks(range(model_latent_dim := means.shape[1]), [f"z{x}" for x in range(model_latent_dim)])
    axes[1, 0].set_xticks(range(stds.shape[1]), [f"z{x}" for x in range(stds.shape[1])])
    axes[1, 1].set_xticks(range(len(difficulties)), [f"D{x}" for x in difficulties])
    axes[1, 2].set_xticks(range(residuals.shape[1]), [f"a{x}" for x in range(residuals.shape[1])])
    figure.suptitle("Homogeneous drone VAE skills across D1-D4", fontsize=16)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)
    arrays["skill_pca"] = point_pca
    arrays["episode_skill"] = ep_skill
    arrays["episode_difficulty"] = ep_difficulty
    arrays["episode_skill_pca"] = episode_pca.astype(np.float32)
    return point_explained


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = args.checkpoint.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    model, payload = load_drone_skill_vae(checkpoint, device)
    model.eval()
    print(
        f"Loaded VAE checkpoint={checkpoint}, latent_dim={model.latent_dim}, "
        f"device={device}"
    )
    arrays = collect_embeddings(model, args, device)
    statistics = calculate_statistics(arrays)
    probes = probe_results(arrays, args.seed)
    for name, result in probes.items():
        print(
            f"LINEAR_PROBE {name}: balanced_accuracy="
            f"{result['balanced_accuracy']:.3f}, chance={result['chance_accuracy']:.3f}"
        )
    output = args.output.expanduser().resolve()
    explained = plot(arrays, statistics, output, args.show)
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": payload.get("epoch"),
        "manifest": str(manifest),
        "splits": list(args.splits),
        "pca_explained_variance": explained.tolist(),
        "linear_probes": probes,
        "statistics": statistics,
    }
    report_path = output.with_name(f"{output.stem}_metrics.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    arrays_path = output.with_suffix(".npz")
    np.savez_compressed(arrays_path, **arrays)
    print(f"Saved visualization: {output}")
    print(f"Saved metrics: {report_path}")
    print(f"Saved embeddings: {arrays_path}")


if __name__ == "__main__":
    main()
