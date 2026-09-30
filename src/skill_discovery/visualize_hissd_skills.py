"""Visualize common and task-specific skills from a trained HiSSD checkpoint."""

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

from skill_discovery.dataset import DEFAULT_MANIFEST_PATH, create_dataloader
from skill_discovery.hissd_models import HeMACHISSD
from skill_discovery.task_descriptor import (
    REALIZED_TASK_DESCRIPTOR_NAMES,
    TASK_DESCRIPTOR_NAMES,
)


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "src/skill_discovery/checkpoints/hissd_joint_online_v23/hissd_online_best.pt"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "src/skill_discovery/hissd_skill_visualization_v23.png"


def parse_args() -> argparse.Namespace:
    """Parse checkpoint, data, and plotting settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("source_train", "source_val", "source_test", "target_test"),
        default=("source_test", "target_test"),
    )
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers; 0 is portable to restricted environments.",
    )
    parser.add_argument("--max-points-per-difficulty", type=int, default=1500)
    parser.add_argument(
        "--max-points-per-episode",
        type=int,
        default=50,
        help="Prevent a few long episodes from dominating the latent plots.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--within-difficulty",
        type=int,
        help="Difficulty whose episode-level skill variation is shown in detail.",
    )
    parser.add_argument("--max-within-episodes", type=int, default=12)
    parser.add_argument(
        "--comparison-output",
        type=Path,
        help="Optional path for the within/across difficulty comparison figure.",
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    """Choose an available inference device."""
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def load_hissd_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[HeMACHISSD, dict[str, Any]]:
    """Restore a complete HiSSD model from its standalone checkpoint."""
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"HiSSD checkpoint not found: {checkpoint_path}")
    payload = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if payload.get("model_type") != "hemac_drone_hissd":
        raise ValueError(f"Not a HeMAC HiSSD checkpoint: {checkpoint_path}")
    model = HeMACHISSD(**payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def resolve_descriptor_schema(
    model: HeMACHISSD,
    payload: dict[str, Any],
) -> tuple[tuple[str, ...], str, str]:
    """Select realized descriptors for new models while supporting old checkpoints."""
    saved_names = tuple(payload.get("task_descriptor_names", ()))
    if saved_names:
        descriptor_names = saved_names
    elif model.task_descriptor_dim == len(TASK_DESCRIPTOR_NAMES):
        descriptor_names = TASK_DESCRIPTOR_NAMES
    else:
        descriptor_names = REALIZED_TASK_DESCRIPTOR_NAMES

    if model.task_descriptor_dim not in (0, len(descriptor_names)):
        raise ValueError(
            "Checkpoint descriptor metadata does not match its model: "
            f"head={model.task_descriptor_dim}, names={len(descriptor_names)}."
        )
    if descriptor_names == tuple(TASK_DESCRIPTOR_NAMES):
        return (
            descriptor_names,
            "task_distribution_descriptor",
            "task_distribution_descriptor_available",
        )
    return descriptor_names, "task_descriptor", "task_descriptor_available"


def _move_observations(
    observations: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        name: observations[name].to(device, non_blocking=True)
        for name in ("global_map", "local_map", "action_history")
    }


@torch.inference_mode()
def collect_embeddings(
    model: HeMACHISSD,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Extract a balanced number of valid drone skill vectors per difficulty."""
    collected: dict[int, dict[str, list[np.ndarray]]] = {}
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
            drop_last_batch=False,
        )
        split_difficulties = sorted(
            {int(entry["difficulty"]) for entry in dataset.entries}
        )
        for difficulty in split_difficulties:
            collected.setdefault(
                difficulty,
                {
                    "common": [],
                    "task": [],
                    "contrastive": [],
                    "agent": [],
                    "split": [],
                    "episode_group": [],
                    "descriptor": [],
                    "descriptor_available": [],
                    "descriptor_prediction": [],
                },
            )

        counts = {difficulty: 0 for difficulty in split_difficulties}
        for batch in loader:
            observations = _move_observations(
                batch["drone"]["observations"], device
            )
            drone_count = observations["global_map"].shape[2]
            valid = (
                batch["agent_mask"][..., -drone_count:].bool()
                & batch["filled"].squeeze(-1).bool().unsqueeze(-1)
            ).to(device)
            features = model.encode_observations(observations)
            task_features = model.encode_task_observations(observations)
            common, task, contrastive = model.infer_skills(
                features, valid, task_features
            )
            descriptor_prediction = None
            if model.task_descriptor_head is not None:
                descriptor_prediction, _ = model.predict_task_descriptor(
                    contrastive, valid
                )

            batch_size, sequence_length, agent_count = valid.shape
            agent_indices = torch.arange(agent_count).view(1, agent_count).expand(
                sequence_length, agent_count
            ).reshape(-1)
            for batch_index in range(batch_size):
                difficulty = int(batch["difficulty"][batch_index])
                episode_index = int(batch["episode_index"][batch_index])
                remaining = args.max_points_per_difficulty - counts[difficulty]
                if remaining <= 0:
                    continue
                episode_key = (split_index, difficulty, episode_index)
                episode_remaining = (
                    args.max_points_per_episode - episode_counts.get(episode_key, 0)
                )
                if episode_remaining <= 0:
                    continue
                flat_indices = (
                    valid[batch_index]
                    .reshape(-1)
                    .nonzero(as_tuple=False)
                    .squeeze(-1)
                )
                if flat_indices.numel() == 0:
                    continue
                if model.task_skill_encoder.task_context_pooling:
                    valid_steps = valid[batch_index].any(dim=1)
                    last_step = int(
                        valid_steps.nonzero(as_tuple=False)[-1].item()
                    )
                    step_start = last_step * agent_count
                    flat_indices = flat_indices[
                        (flat_indices >= step_start)
                        & (flat_indices < step_start + agent_count)
                    ]
                take = min(remaining, episode_remaining, int(flat_indices.numel()))
                flat_indices = flat_indices[:take]
                flat_indices_cpu = flat_indices.cpu()
                target = collected[difficulty]
                target["common"].append(
                    common[batch_index].reshape(-1, model.skill_dim)[flat_indices]
                    .cpu()
                    .numpy()
                )
                target["task"].append(
                    task[batch_index].reshape(-1, model.skill_dim)[flat_indices]
                    .cpu()
                    .numpy()
                )
                target["contrastive"].append(
                    contrastive[batch_index]
                    .reshape(-1, model.skill_dim)[flat_indices]
                    .cpu()
                    .numpy()
                )
                target["agent"].append(
                    agent_indices[flat_indices_cpu].numpy()
                )
                target["split"].append(
                    np.full(flat_indices.numel(), split_index, dtype=np.int64)
                )
                episode_group = (
                    split_index * 10_000_000
                    + difficulty * 1_000_000
                    + episode_index
                )
                target["episode_group"].append(
                    np.full(flat_indices.numel(), episode_group, dtype=np.int64)
                )
                target["descriptor"].append(
                    np.repeat(
                        batch[args.descriptor_field][batch_index]
                        .reshape(1, -1)
                        .numpy(),
                        flat_indices.numel(),
                        axis=0,
                    )
                )
                target["descriptor_available"].append(
                    np.full(
                        flat_indices.numel(),
                        bool(
                            batch[args.descriptor_available_field][batch_index]
                        ),
                        dtype=np.bool_,
                    )
                )
                if descriptor_prediction is None:
                    prediction = np.zeros(
                        (1, len(args.descriptor_names)), dtype=np.float32
                    )
                else:
                    prediction = (
                        descriptor_prediction[batch_index]
                        .reshape(1, -1)
                        .cpu()
                        .numpy()
                    )
                target["descriptor_prediction"].append(
                    np.repeat(prediction, flat_indices.numel(), axis=0)
                )
                counts[difficulty] += take
                episode_counts[episode_key] = episode_counts.get(episode_key, 0) + take
            if all(
                count >= args.max_points_per_difficulty for count in counts.values()
            ):
                break

    arrays: dict[str, list[np.ndarray]] = {
        "common": [],
        "task": [],
        "contrastive": [],
        "difficulty": [],
        "agent": [],
        "split": [],
        "episode_group": [],
        "descriptor": [],
        "descriptor_available": [],
        "descriptor_prediction": [],
    }
    for difficulty in sorted(collected):
        values = collected[difficulty]
        if not values["common"]:
            continue
        count = sum(chunk.shape[0] for chunk in values["common"])
        arrays["common"].append(np.concatenate(values["common"]))
        arrays["task"].append(np.concatenate(values["task"]))
        arrays["contrastive"].append(np.concatenate(values["contrastive"]))
        arrays["difficulty"].append(np.full(count, difficulty, dtype=np.int64))
        arrays["agent"].append(np.concatenate(values["agent"]))
        arrays["split"].append(np.concatenate(values["split"]))
        arrays["episode_group"].append(np.concatenate(values["episode_group"]))
        arrays["descriptor"].append(np.concatenate(values["descriptor"]))
        arrays["descriptor_available"].append(
            np.concatenate(values["descriptor_available"])
        )
        arrays["descriptor_prediction"].append(
            np.concatenate(values["descriptor_prediction"])
        )
        print(f"difficulty={difficulty}: extracted {count} valid skill vectors")
    if not arrays["common"]:
        raise RuntimeError("No valid skill vectors were found in the requested splits.")
    output = {name: np.concatenate(chunks) for name, chunks in arrays.items()}
    output["split_names"] = np.asarray(args.splits)
    return output


def pca_2d(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project vectors to two dimensions using a dependency-free PCA."""
    centered = values.astype(np.float64) - values.mean(axis=0, keepdims=True)
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    coordinates = centered @ right_vectors[:2].T
    variance = np.square(singular_values)
    explained = variance[:2] / max(float(variance.sum()), np.finfo(float).eps)
    return coordinates.astype(np.float32), explained.astype(np.float32)


def fit_pca_2d(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a reusable two-dimensional PCA projection."""
    values64 = values.astype(np.float64)
    mean = values64.mean(axis=0, keepdims=True)
    centered = values64 - mean
    _, singular_values, right_vectors = np.linalg.svd(
        centered, full_matrices=False
    )
    components = right_vectors[:2]
    coordinates = centered @ components.T
    variance = np.square(singular_values)
    explained = variance[:2] / max(
        float(variance.sum()), np.finfo(float).eps
    )
    return (
        coordinates.astype(np.float32),
        explained.astype(np.float32),
        mean,
        components,
    )


def episode_means(
    values: np.ndarray,
    difficulties: np.ndarray,
    episode_groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate timestep/agent skills into one vector per episode."""
    groups = np.unique(episode_groups)
    means = []
    labels = []
    for group in groups:
        selected = episode_groups == group
        means.append(values[selected].mean(axis=0))
        labels.append(int(difficulties[selected][0]))
    return np.asarray(means), np.asarray(labels), groups


def skill_comparison_statistics(
    embeddings: dict[str, np.ndarray],
    within_difficulty: int,
) -> dict[str, Any]:
    """Quantify episode variation within tasks and separation across tasks."""
    difficulties = sorted(int(x) for x in np.unique(embeddings["difficulty"]))
    results: dict[str, Any] = {
        "within_difficulty": within_difficulty,
        "difficulties": difficulties,
        "representations": {},
    }
    labels = embeddings["difficulty"]
    groups = embeddings["episode_group"]
    for name in ("common", "task", "contrastive"):
        values = embeddings[name]
        centroids = np.asarray(
            [values[labels == difficulty].mean(axis=0) for difficulty in difficulties]
        )
        distance_matrix = np.linalg.norm(
            centroids[:, None] - centroids[None, :], axis=-1
        )
        per_difficulty = {}
        all_within = []
        for index, difficulty in enumerate(difficulties):
            selected = labels == difficulty
            distances = np.linalg.norm(
                values[selected] - centroids[index], axis=1
            )
            all_within.extend(distances.tolist())
            per_difficulty[str(difficulty)] = {
                "points": int(selected.sum()),
                "episodes": int(np.unique(groups[selected]).size),
                "mean_distance_to_difficulty_centroid": float(distances.mean()),
            }
        selected = labels == within_difficulty
        within_episode_means, _, _ = episode_means(
            values[selected], labels[selected], groups[selected]
        )
        within_episode_centroid = within_episode_means.mean(axis=0)
        episode_spread = np.linalg.norm(
            within_episode_means - within_episode_centroid, axis=1
        )
        off_diagonal = distance_matrix[
            ~np.eye(len(difficulties), dtype=bool)
        ]
        mean_within = float(np.mean(all_within))
        mean_between = float(off_diagonal.mean()) if off_diagonal.size else 0.0
        results["representations"][name] = {
            "per_difficulty": per_difficulty,
            "centroid_distance_matrix": distance_matrix.tolist(),
            "mean_within_difficulty_distance": mean_within,
            "mean_between_difficulty_centroid_distance": mean_between,
            "between_within_ratio": mean_between / max(mean_within, 1e-8),
            "within_selected_episode_centroid_spread": float(
                episode_spread.mean()
            ),
        }
    return results


def plot_skill_comparisons(
    embeddings: dict[str, np.ndarray],
    output_path: Path,
    *,
    within_difficulty: int,
    max_within_episodes: int,
    show: bool,
) -> None:
    """Plot episode-level and cross-difficulty views of all skill branches."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/hemac_matplotlib")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = embeddings["difficulty"]
    episode_groups = embeddings["episode_group"]
    difficulties = sorted(int(x) for x in np.unique(labels))
    if within_difficulty not in difficulties:
        raise ValueError(
            f"D{within_difficulty} is unavailable; found {difficulties}."
        )
    colors = plt.get_cmap("tab10")
    representation_titles = {
        "common": "Common skill c",
        "task": "Task skill z",
        "contrastive": "Contrastive embedding q",
    }
    figure, axes = plt.subplots(
        3, 4, figsize=(24, 17), constrained_layout=True
    )

    for row, name in enumerate(("common", "task", "contrastive")):
        values = embeddings[name]
        global_coordinates, explained, mean, components = fit_pca_2d(values)
        selected_within = labels == within_difficulty
        within_coordinates, within_explained, _, _ = fit_pca_2d(
            values[selected_within]
        )
        within_groups = episode_groups[selected_within]
        unique_groups, counts = np.unique(within_groups, return_counts=True)
        order = np.lexsort((unique_groups, -counts))
        shown_groups = unique_groups[order[:max_within_episodes]]
        episode_colors = plt.get_cmap("turbo")
        for episode_index, group in enumerate(shown_groups):
            selected = within_groups == group
            color = episode_colors(
                episode_index / max(len(shown_groups) - 1, 1)
            )
            axes[row, 0].scatter(
                within_coordinates[selected, 0],
                within_coordinates[selected, 1],
                s=18,
                alpha=0.58,
                color=color,
                edgecolors="none",
                label=f"E{episode_index + 1}",
            )
            center = within_coordinates[selected].mean(axis=0)
            axes[row, 0].scatter(
                center[0], center[1], marker="x", s=55, color=color
            )

        for difficulty in difficulties:
            selected = labels == difficulty
            color = colors((difficulty - 1) % 10)
            axes[row, 1].scatter(
                global_coordinates[selected, 0],
                global_coordinates[selected, 1],
                s=9,
                alpha=0.23,
                color=color,
                edgecolors="none",
                label=f"D{difficulty}",
            )
            center = global_coordinates[selected].mean(axis=0)
            axes[row, 1].text(
                center[0], center[1], f"D{difficulty}",
                ha="center", va="center", fontweight="bold",
            )

        episode_values, episode_difficulties, _ = episode_means(
            values, labels, episode_groups
        )
        episode_coordinates = (
            episode_values.astype(np.float64) - mean
        ) @ components.T
        for difficulty in difficulties:
            selected = episode_difficulties == difficulty
            axes[row, 2].scatter(
                episode_coordinates[selected, 0],
                episode_coordinates[selected, 1],
                s=30,
                alpha=0.62,
                color=colors((difficulty - 1) % 10),
                edgecolors="none",
                label=f"D{difficulty}",
            )

        centroids = np.asarray(
            [values[labels == difficulty].mean(axis=0) for difficulty in difficulties]
        )
        distances = np.linalg.norm(
            centroids[:, None] - centroids[None, :], axis=-1
        )
        image = axes[row, 3].imshow(distances, cmap="magma", vmin=0)
        axes[row, 3].set_xticks(
            range(len(difficulties)), [f"D{x}" for x in difficulties]
        )
        axes[row, 3].set_yticks(
            range(len(difficulties)), [f"D{x}" for x in difficulties]
        )
        if len(difficulties) <= 8:
            for y in range(len(difficulties)):
                for x in range(len(difficulties)):
                    axes[row, 3].text(
                        x, y, f"{distances[y, x]:.2f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if distances[y, x] > distances.max() * 0.5 else "black",
                    )
        figure.colorbar(image, ax=axes[row, 3], fraction=0.046)

        axes[row, 0].set_title(
            f"{representation_titles[name]}: within D{within_difficulty} by episode"
        )
        axes[row, 1].set_title(
            f"{representation_titles[name]}: samples across difficulties"
        )
        axes[row, 2].set_title(
            f"{representation_titles[name]}: episode means"
        )
        axes[row, 3].set_title("Difficulty centroid distance")
        axes[row, 0].set_xlabel(
            f"within PC1 ({within_explained[0] * 100:.1f}%)"
        )
        axes[row, 0].set_ylabel(
            f"within PC2 ({within_explained[1] * 100:.1f}%)"
        )
        for column in (1, 2):
            axes[row, column].set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
            axes[row, column].set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
        for column in range(3):
            axes[row, column].grid(alpha=0.16)
        axes[row, 0].legend(fontsize=7, ncol=3)
        axes[row, 1].legend(fontsize=8, ncol=2)
        axes[row, 2].legend(fontsize=8, ncol=2)

    figure.suptitle(
        f"HiSSD skill comparison: within D{within_difficulty} and across tasks",
        fontsize=17,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved within/across skill comparison: {output_path}")
    if show:
        plt.show()
    plt.close(figure)


def grouped_linear_probe(
    values: np.ndarray,
    labels: np.ndarray,
    episode_groups: np.ndarray,
    *,
    seed: int,
    train_fraction: float = 0.7,
    ridge: float = 1e-2,
) -> dict[str, Any]:
    """Measure task information without leaking steps from the same episode."""
    rng = np.random.default_rng(seed)
    train_groups: list[int] = []
    test_groups: list[int] = []
    classes = np.unique(labels)
    for class_id in classes:
        class_groups = np.unique(episode_groups[labels == class_id])
        rng.shuffle(class_groups)
        if class_groups.size < 2:
            raise ValueError(
                f"Difficulty {class_id} needs at least two episodes for a probe."
            )
        train_count = int(round(class_groups.size * train_fraction))
        train_count = min(max(train_count, 1), class_groups.size - 1)
        train_groups.extend(class_groups[:train_count].tolist())
        test_groups.extend(class_groups[train_count:].tolist())

    train_mask = np.isin(episode_groups, train_groups)
    test_mask = np.isin(episode_groups, test_groups)
    train_x = values[train_mask].astype(np.float64)
    test_x = values[test_mask].astype(np.float64)
    train_y = labels[train_mask]
    test_y = labels[test_mask]
    mean = train_x.mean(axis=0, keepdims=True)
    scale = train_x.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    train_x = (train_x - mean) / scale
    test_x = (test_x - mean) / scale
    train_x = np.concatenate((train_x, np.ones((train_x.shape[0], 1))), axis=1)
    test_x = np.concatenate((test_x, np.ones((test_x.shape[0], 1))), axis=1)

    class_to_index = {int(class_id): index for index, class_id in enumerate(classes)}
    targets = np.zeros((train_y.size, classes.size), dtype=np.float64)
    targets[
        np.arange(train_y.size),
        np.asarray([class_to_index[int(value)] for value in train_y]),
    ] = 1.0
    regularizer = np.eye(train_x.shape[1], dtype=np.float64) * ridge
    regularizer[-1, -1] = 0.0
    weights = np.linalg.solve(
        train_x.T @ train_x + regularizer,
        train_x.T @ targets,
    )
    prediction = classes[np.argmax(test_x @ weights, axis=1)]
    per_class = {
        str(int(class_id)): float((prediction[test_y == class_id] == class_id).mean())
        for class_id in classes
    }
    return {
        "accuracy": float((prediction == test_y).mean()),
        "balanced_accuracy": float(np.mean(list(per_class.values()))),
        "chance_accuracy": float(1.0 / classes.size),
        "per_difficulty_accuracy": per_class,
        "train_episodes": len(train_groups),
        "test_episodes": len(test_groups),
        "train_points": int(train_mask.sum()),
        "test_points": int(test_mask.sum()),
    }


def evaluate_linear_probes(
    embeddings: dict[str, np.ndarray],
    *,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Evaluate common, decoder-task, and contrastive task information."""
    results: dict[str, dict[str, Any]] = {}
    for split_index, split_name in enumerate(embeddings["split_names"]):
        selected = embeddings["split"] == split_index
        split_labels = embeddings["difficulty"][selected]
        if np.unique(split_labels).size < 2:
            continue
        split_results = {}
        for name in ("common", "task", "contrastive"):
            split_results[name] = grouped_linear_probe(
                embeddings[name][selected],
                split_labels,
                embeddings["episode_group"][selected],
                seed=seed + split_index,
            )
            metrics = split_results[name]
            print(
                f"LINEAR_PROBE split={split_name} representation={name} "
                f"accuracy={metrics['accuracy']:.3f} "
                f"balanced={metrics['balanced_accuracy']:.3f} "
                f"chance={metrics['chance_accuracy']:.3f} "
                f"train/test episodes={metrics['train_episodes']}/"
                f"{metrics['test_episodes']}"
            )
        results[str(split_name)] = split_results
    return results


def grouped_descriptor_probe(
    values: np.ndarray,
    descriptors: np.ndarray,
    difficulty: np.ndarray,
    episode_groups: np.ndarray,
    *,
    descriptor_names: tuple[str, ...],
    seed: int,
    train_fraction: float = 0.7,
    ridge: float = 1e-2,
) -> dict[str, Any]:
    """Measure how linearly recoverable episode task parameters are."""
    rng = np.random.default_rng(seed)
    train_groups: list[int] = []
    test_groups: list[int] = []
    for difficulty_id in np.unique(difficulty):
        groups = np.unique(episode_groups[difficulty == difficulty_id])
        rng.shuffle(groups)
        if groups.size < 2:
            continue
        train_count = min(max(int(round(groups.size * train_fraction)), 1), groups.size - 1)
        train_groups.extend(groups[:train_count].tolist())
        test_groups.extend(groups[train_count:].tolist())
    if not train_groups or not test_groups:
        raise ValueError("Descriptor probe needs at least two episodes per difficulty.")

    train_mask = np.isin(episode_groups, train_groups)
    test_mask = np.isin(episode_groups, test_groups)
    train_x = values[train_mask].astype(np.float64)
    test_x = values[test_mask].astype(np.float64)
    train_y = descriptors[train_mask].astype(np.float64)
    test_y = descriptors[test_mask].astype(np.float64)
    mean = train_x.mean(axis=0, keepdims=True)
    scale = train_x.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    train_x = (train_x - mean) / scale
    test_x = (test_x - mean) / scale
    train_x = np.concatenate((train_x, np.ones((train_x.shape[0], 1))), axis=1)
    test_x = np.concatenate((test_x, np.ones((test_x.shape[0], 1))), axis=1)
    regularizer = np.eye(train_x.shape[1], dtype=np.float64) * ridge
    regularizer[-1, -1] = 0.0
    weights = np.linalg.solve(
        train_x.T @ train_x + regularizer,
        train_x.T @ train_y,
    )
    prediction = np.clip(test_x @ weights, 0.0, 1.0)
    absolute_error = np.abs(prediction - test_y)
    residual = np.square(prediction - test_y).sum(axis=0)
    centered = np.square(test_y - test_y.mean(axis=0, keepdims=True)).sum(axis=0)
    r2 = 1.0 - residual / np.maximum(centered, np.finfo(float).eps)
    return {
        "normalized_mae": float(absolute_error.mean()),
        "mean_r2": float(r2.mean()),
        "per_component_mae": {
            name: float(absolute_error[:, index].mean())
            for index, name in enumerate(descriptor_names)
        },
        "per_component_r2": {
            name: float(r2[index])
            for index, name in enumerate(descriptor_names)
        },
        "train_episodes": len(train_groups),
        "test_episodes": len(test_groups),
    }


def evaluate_descriptor_probes(
    embeddings: dict[str, np.ndarray],
    *,
    descriptor_names: tuple[str, ...],
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Evaluate continuous task information in each learned representation."""
    results: dict[str, dict[str, Any]] = {}
    for split_index, split_name in enumerate(embeddings["split_names"]):
        selected = (
            (embeddings["split"] == split_index)
            & embeddings["descriptor_available"]
        )
        if not selected.any():
            print(f"DESCRIPTOR_PROBE split={split_name}: skipped legacy data")
            continue
        split_results = {}
        direct_error = np.abs(
            embeddings["descriptor_prediction"][selected]
            - embeddings["descriptor"][selected]
        )
        split_results["prediction_head"] = {
            "normalized_mae": float(direct_error.mean()),
            "per_component_mae": {
                name: float(direct_error[:, index].mean())
                for index, name in enumerate(descriptor_names)
            },
        }
        print(
            f"DESCRIPTOR_HEAD split={split_name} "
            f"mae={direct_error.mean():.3f}"
        )
        for name in ("common", "task", "contrastive"):
            metrics = grouped_descriptor_probe(
                embeddings[name][selected],
                embeddings["descriptor"][selected],
                embeddings["difficulty"][selected],
                embeddings["episode_group"][selected],
                descriptor_names=descriptor_names,
                seed=seed + split_index,
            )
            split_results[name] = metrics
            print(
                f"DESCRIPTOR_PROBE split={split_name} representation={name} "
                f"mae={metrics['normalized_mae']:.3f} "
                f"r2={metrics['mean_r2']:.3f}"
            )
        results[str(split_name)] = split_results
    return results


def plot_embeddings(
    embeddings: dict[str, np.ndarray],
    output_path: Path,
    *,
    show: bool,
) -> None:
    """Render source/target common and task-specific skill PCA plots."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/hemac_matplotlib")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    common_2d, common_variance = pca_2d(embeddings["common"])
    task_2d, task_variance = pca_2d(embeddings["task"])
    contrastive_2d, contrastive_variance = pca_2d(embeddings["contrastive"])
    embeddings["common_pca"] = common_2d
    embeddings["task_pca"] = task_2d
    embeddings["contrastive_pca"] = contrastive_2d

    colors = plt.get_cmap("tab10")
    markers = ("o", "^")
    figure, axes = plt.subplots(1, 3, figsize=(22, 6), constrained_layout=True)
    panels = (
        (axes[0], common_2d, common_variance, "Common skill c"),
        (axes[1], task_2d, task_variance, "Decoder task skill z"),
        (
            axes[2],
            contrastive_2d,
            contrastive_variance,
            "Contrastive task embedding q",
        ),
    )
    for axis, coordinates, explained, title in panels:
        for difficulty in sorted(np.unique(embeddings["difficulty"])):
            for split_index, split_name in enumerate(embeddings["split_names"]):
                selected = (
                    (embeddings["difficulty"] == difficulty)
                    & (embeddings["split"] == split_index)
                )
                if not selected.any():
                    continue
                axis.scatter(
                    coordinates[selected, 0],
                    coordinates[selected, 1],
                    s=10,
                    alpha=0.28,
                    color=colors((int(difficulty) - 1) % 10),
                    marker=markers[split_index % len(markers)],
                    label=f"D{difficulty} ({split_name})",
                    edgecolors="none",
                )
                centroid = coordinates[selected].mean(axis=0)
                axis.text(
                    centroid[0],
                    centroid[1],
                    str(int(difficulty)),
                    fontsize=10,
                    fontweight="bold",
                    ha="center",
                    va="center",
                )
        axis.set_title(title)
        axis.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
        axis.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
        axis.grid(alpha=0.18)
        axis.legend(fontsize=8, ncol=2)
    figure.suptitle("HeMAC HiSSD skills: source tasks vs held-out target tasks")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    print(f"Saved skill visualization: {output_path}")
    if show:
        plt.show()
    plt.close(figure)


def main() -> None:
    """Load the best checkpoint, extract skills, and create PCA plots."""
    args = parse_args()
    if args.max_points_per_difficulty <= 0:
        raise ValueError("--max-points-per-difficulty must be positive.")
    if args.max_points_per_episode <= 0:
        raise ValueError("--max-points-per-episode must be positive.")
    if args.max_within_episodes <= 0:
        raise ValueError("--max-within-episodes must be positive.")
    device = resolve_device(args.device)
    model, payload = load_hissd_model(args.checkpoint, device)
    (
        args.descriptor_names,
        args.descriptor_field,
        args.descriptor_available_field,
    ) = resolve_descriptor_schema(model, payload)
    print(
        f"Loaded HiSSD epoch={payload.get('epoch')} on {device}; "
        f"training_tasks={payload.get('training_tasks')}, "
        f"held_out_tasks={payload.get('held_out_tasks')}, "
        f"descriptor={args.descriptor_field}[{len(args.descriptor_names)}]"
    )
    embeddings = collect_embeddings(model, args, device)
    probe_results = {
        "difficulty_classification": evaluate_linear_probes(
            embeddings, seed=args.seed
        ),
        "continuous_descriptor_regression": evaluate_descriptor_probes(
            embeddings,
            descriptor_names=args.descriptor_names,
            seed=args.seed,
        ),
    }
    output_path = args.output.expanduser().resolve()
    plot_embeddings(embeddings, output_path, show=args.show)
    available_difficulties = sorted(
        int(value) for value in np.unique(embeddings["difficulty"])
    )
    within_difficulty = (
        available_difficulties[0]
        if args.within_difficulty is None
        else args.within_difficulty
    )
    if within_difficulty not in available_difficulties:
        raise ValueError(
            f"--within-difficulty={within_difficulty} is unavailable; "
            f"loaded difficulties are {available_difficulties}."
        )
    comparison_output = (
        args.comparison_output.expanduser().resolve()
        if args.comparison_output is not None
        else output_path.with_name(
            f"{output_path.stem}_within_d{within_difficulty}_comparison.png"
        )
    )
    comparison_statistics = skill_comparison_statistics(
        embeddings, within_difficulty
    )
    plot_skill_comparisons(
        embeddings,
        comparison_output,
        within_difficulty=within_difficulty,
        max_within_episodes=args.max_within_episodes,
        show=args.show,
    )
    comparison_metrics_path = comparison_output.with_name(
        f"{comparison_output.stem}_metrics.json"
    )
    comparison_metrics_path.write_text(
        json.dumps(comparison_statistics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved skill comparison metrics: {comparison_metrics_path}")
    array_path = output_path.with_suffix(".npz")
    np.savez_compressed(array_path, **embeddings)
    print(f"Saved projected skill arrays: {array_path}")
    probe_path = output_path.with_name(f"{output_path.stem}_probe.json")
    probe_path.write_text(json.dumps(probe_results, indent=2), encoding="utf-8")
    print(f"Saved probe diagnostics: {probe_path}")


if __name__ == "__main__":
    main()
