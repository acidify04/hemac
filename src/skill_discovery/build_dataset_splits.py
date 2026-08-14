"""Build deterministic episode-level source/target splits for HiSSD."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "src/skill_discovery/offline_data"
DEFAULT_MANIFEST_PATH = DEFAULT_DATA_ROOT / "dataset_splits.json"
DEFAULT_SOURCE_DIFFICULTIES = (1, 2, 3)
DEFAULT_TARGET_DIFFICULTIES = (4, 5, 6)
OUTCOME_CATEGORIES = (
    "success",
    "goal_found_failure",
    "goal_not_found",
)
MINIMUM_FORMAT_VERSION = 4


def parse_args() -> argparse.Namespace:
    """Parse split locations, ratios, and deterministic seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--source-difficulties",
        nargs="+",
        type=int,
        default=DEFAULT_SOURCE_DIFFICULTIES,
    )
    parser.add_argument(
        "--target-difficulties",
        nargs="+",
        type=int,
        default=DEFAULT_TARGET_DIFFICULTIES,
    )
    parser.add_argument(
        "--minimum-format-version",
        type=int,
        default=MINIMUM_FORMAT_VERSION,
    )
    return parser.parse_args()


def scalar_bool(value: Any) -> bool:
    """Convert a scalar tensor or Python value to bool."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Expected scalar outcome label, got {tuple(value.shape)}.")
        return bool(value.item())
    return bool(value)


def validate_outcome(category: str, outcome: dict[str, Any], path: Path) -> None:
    """Verify directory category and persisted final labels agree."""
    success = scalar_bool(outcome.get("success", False))
    goal_found = scalar_bool(outcome.get("goal_found", False))
    expected = {
        "success": success,
        "goal_found_failure": (not success and goal_found),
        "goal_not_found": (not success and not goal_found),
    }
    if category not in expected or not expected[category]:
        raise ValueError(
            f"Outcome labels in {path} do not match category {category!r}: "
            f"success={success}, goal_found={goal_found}."
        )
    stored_category = outcome.get("category")
    if stored_category is not None and stored_category != category:
        raise ValueError(
            f"Stored category {stored_category!r} does not match {category!r}: {path}"
        )


def inspect_episode(
    path: Path,
    data_root: Path,
    difficulty: int,
    category: str,
    minimum_format_version: int,
) -> dict[str, Any]:
    """Read lightweight episode metadata and validate trajectory alignment."""
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    metadata = payload.get("metadata", {})
    format_version = int(metadata.get("format_version", 0))
    if format_version < minimum_format_version:
        raise ValueError(
            f"{path} uses format v{format_version}; expected v{minimum_format_version}+"
        )
    stored_difficulty = int(metadata.get("difficulty", difficulty))
    if stored_difficulty != difficulty:
        raise ValueError(
            f"Stored difficulty {stored_difficulty} does not match directory "
            f"difficulty {difficulty}: {path}"
        )

    outcome = payload.get("outcome", {})
    validate_outcome(category, outcome, path)
    transition_count = int(payload["team_reward"].shape[0])
    if transition_count <= 0:
        raise ValueError(f"Episode has no transitions: {path}")
    state_count = int(payload["global_state"]["central_map"].shape[0])
    if state_count != transition_count + 1:
        raise ValueError(
            f"Expected T+1 central states in {path}, got {state_count} for T={transition_count}."
        )
    for role in ("observer", "drone"):
        observations = payload[role]["observations"]
        if int(observations["global_map"].shape[0]) != state_count:
            raise ValueError(f"Misaligned {role} observations: {path}")
        if int(payload[role]["actions"].shape[0]) != transition_count:
            raise ValueError(f"Misaligned {role} actions: {path}")

    return {
        "path": path.relative_to(data_root).as_posix(),
        "difficulty": difficulty,
        "task_id": difficulty - 1,
        "category": category,
        "episode_index": int(metadata.get("episode_index", path.stem.split("_")[-1])),
        "collection_attempt": int(metadata.get("collection_attempt", -1)),
        "seed": int(metadata.get("seed", -1)),
        "format_version": format_version,
        "transitions": transition_count,
    }


def discover_episodes(
    data_root: Path,
    difficulties: tuple[int, ...],
    minimum_format_version: int,
) -> dict[tuple[int, str], list[dict[str, Any]]]:
    """Discover and validate every requested difficulty/category stratum."""
    strata = {}
    for difficulty in difficulties:
        difficulty_dir = data_root / f"difficulty_{difficulty:02d}"
        if not difficulty_dir.is_dir():
            raise FileNotFoundError(f"Missing difficulty directory: {difficulty_dir}")
        for category in OUTCOME_CATEGORIES:
            category_dir = difficulty_dir / category
            paths = sorted(category_dir.glob("*.pt"))
            if not paths:
                raise FileNotFoundError(f"No PT episodes found in {category_dir}")
            strata[(difficulty, category)] = [
                inspect_episode(
                    path,
                    data_root,
                    difficulty,
                    category,
                    minimum_format_version,
                )
                for path in paths
            ]
    return strata


def split_source_stratum(
    entries: list[dict[str, Any]],
    difficulty: int,
    category: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministically split one source task/outcome stratum by episode."""
    shuffled = list(entries)
    random.Random(f"{seed}:{difficulty}:{category}").shuffle(shuffled)
    count = len(shuffled)
    train_count = int(count * train_ratio)
    val_count = int(count * val_ratio)
    if train_count <= 0 or val_count <= 0 or train_count + val_count >= count:
        raise ValueError(
            f"Stratum difficulty={difficulty}, category={category} is too small "
            f"for ratios train={train_ratio}, val={val_ratio}: {count} episodes."
        )
    return (
        shuffled[:train_count],
        shuffled[train_count : train_count + val_count],
        shuffled[train_count + val_count :],
    )


def summarize_splits(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Build count and transition summaries for audit and logging."""
    summary = {}
    for split_name, entries in splits.items():
        by_difficulty = Counter(entry["difficulty"] for entry in entries)
        by_category = Counter(entry["category"] for entry in entries)
        summary[split_name] = {
            "episodes": len(entries),
            "transitions": sum(entry["transitions"] for entry in entries),
            "by_difficulty": {str(key): value for key, value in sorted(by_difficulty.items())},
            "by_category": dict(sorted(by_category.items())),
        }
    return summary


def build_manifest(
    data_root: Path,
    source_difficulties: tuple[int, ...],
    target_difficulties: tuple[int, ...],
    train_ratio: float,
    val_ratio: float,
    seed: int,
    minimum_format_version: int,
) -> dict[str, Any]:
    """Build source train/val/test splits and a held-out target test split."""
    overlap = set(source_difficulties).intersection(target_difficulties)
    if overlap:
        raise ValueError(f"Source and target difficulties overlap: {sorted(overlap)}")
    if train_ratio <= 0.0 or val_ratio <= 0.0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("Split ratios must satisfy train > 0, val > 0, train + val < 1.")

    all_difficulties = tuple(source_difficulties) + tuple(target_difficulties)
    strata = discover_episodes(
        data_root,
        all_difficulties,
        minimum_format_version,
    )
    splits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for difficulty in source_difficulties:
        for category in OUTCOME_CATEGORIES:
            train, val, test = split_source_stratum(
                strata[(difficulty, category)],
                difficulty,
                category,
                train_ratio,
                val_ratio,
                seed,
            )
            splits["source_train"].extend(train)
            splits["source_val"].extend(val)
            splits["source_test"].extend(test)
    for difficulty in target_difficulties:
        for category in OUTCOME_CATEGORIES:
            splits["target_test"].extend(strata[(difficulty, category)])

    result = {
        "manifest_version": 1,
        "data_root": str(data_root),
        "seed": seed,
        "source_difficulties": list(source_difficulties),
        "target_difficulties": list(target_difficulties),
        "split_ratios": {
            "train": train_ratio,
            "validation": val_ratio,
            "test": 1.0 - train_ratio - val_ratio,
        },
        "outcome_categories": list(OUTCOME_CATEGORIES),
        "minimum_format_version": minimum_format_version,
        "splits": dict(splits),
    }
    result["summary"] = summarize_splits(result["splits"])
    return result


def save_manifest(manifest: dict[str, Any], output_path: Path) -> None:
    """Atomically write the generated JSON manifest."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(temporary_path, output_path)


def print_summary(manifest: dict[str, Any], output_path: Path) -> None:
    """Print compact split counts after generation."""
    print(f"Saved split manifest: {output_path}")
    for split_name, values in manifest["summary"].items():
        print(
            f"{split_name}: episodes={values['episodes']}, "
            f"transitions={values['transitions']}, "
            f"difficulties={values['by_difficulty']}, "
            f"categories={values['by_category']}"
        )


def main() -> None:
    """Generate and save deterministic source/target episode splits."""
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    manifest = build_manifest(
        data_root=data_root,
        source_difficulties=tuple(args.source_difficulties),
        target_difficulties=tuple(args.target_difficulties),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
        minimum_format_version=int(args.minimum_format_version),
    )
    save_manifest(manifest, output_path)
    print_summary(manifest, output_path)


if __name__ == "__main__":
    main()
