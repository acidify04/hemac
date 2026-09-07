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

from skill_discovery.drone_task import (
    DRONE_SKILL_SUCCESS_MIN_COVERAGE_RATIO,
    DRONE_SKILL_SUCCESS_REWARD,
    drone_skill_outcome_from_payload,
)
from skill_discovery.task_descriptor import (
    REALIZED_TASK_DESCRIPTOR_NAMES,
    TASK_DESCRIPTOR_NAMES,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "src/skill_discovery/offline_data"
MISSION_MANIFEST_PATH = DEFAULT_DATA_ROOT / "dataset_splits.json"
DRONE_TASK_MANIFEST_PATH = DEFAULT_DATA_ROOT / "drone_task_dataset_splits.json"
DEFAULT_MANIFEST_PATH = MISSION_MANIFEST_PATH
DEFAULT_SOURCE_DIFFICULTIES = (1, 2, 3)
DEFAULT_TARGET_DIFFICULTIES = (4, 5, 6)
OUTCOME_CATEGORIES = (
    "success",
    "goal_found_failure",
    "goal_not_found",
)
MINIMUM_FORMAT_VERSION = 5


def parse_args() -> argparse.Namespace:
    """Parse split locations, ratios, and deterministic seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--task-definition",
        choices=("mission", "drone"),
        default="mission",
        help="Use observer goal arrival (mission) or drone exploration labels.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Defaults to the manifest matching --task-definition.",
    )
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
    parser.add_argument(
        "--success-min-coverage-ratio",
        type=float,
        default=DRONE_SKILL_SUCCESS_MIN_COVERAGE_RATIO,
    )
    return parser.parse_args()


def scalar_bool(value: Any) -> bool:
    """Convert a persisted scalar tensor or Python value to bool."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Expected a scalar bool, got {tuple(value.shape)}.")
        return bool(value.item())
    return bool(value)


def mission_outcome_from_payload(
    payload: dict[str, Any],
) -> tuple[str, bool, float]:
    """Derive the original observer-goal task outcome from any saved format."""
    outcome = payload.get("outcome", {})
    final_info = payload.get("metadata", {}).get("final_info", {})
    success = scalar_bool(
        outcome.get(
            "mission_success",
            outcome.get("success", final_info.get("success", False)),
        )
    )
    goal_found = scalar_bool(
        outcome.get(
            "mission_goal_found",
            outcome.get("goal_found", final_info.get("goal_found", False)),
        )
    )
    if success:
        category = "success"
    elif goal_found:
        category = "goal_found_failure"
    else:
        category = "goal_not_found"
    coverage_ratio = float(final_info.get("coverage_ratio", 0.0))
    return category, goal_found, coverage_ratio


def inspect_episode(
    path: Path,
    data_root: Path,
    difficulty: int,
    minimum_format_version: int,
    success_min_coverage_ratio: float,
    task_definition: str,
) -> dict[str, Any]:
    """Read lightweight episode metadata and validate trajectory alignment."""
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    metadata = payload.get("metadata", {})
    format_version = int(metadata.get("format_version", 0))
    if format_version < minimum_format_version:
        raise ValueError(
            f"{path} uses format v{format_version}; expected v{minimum_format_version}+"
        )
    task_descriptor = payload.get("task_descriptor")
    if format_version >= 6 and (
        not isinstance(task_descriptor, torch.Tensor)
        or tuple(task_descriptor.shape) != (len(TASK_DESCRIPTOR_NAMES),)
    ):
        raise ValueError(f"Invalid v6 task descriptor in {path}.")
    realized_descriptor = payload.get("realized_task_descriptor")
    if format_version >= 7 and (
        not isinstance(realized_descriptor, torch.Tensor)
        or tuple(realized_descriptor.shape)
        != (len(REALIZED_TASK_DESCRIPTOR_NAMES),)
    ):
        raise ValueError(f"Invalid v7 realized task descriptor in {path}.")
    stored_difficulty = int(metadata.get("difficulty", difficulty))
    if stored_difficulty != difficulty:
        raise ValueError(
            f"Stored difficulty {stored_difficulty} does not match directory "
            f"difficulty {difficulty}: {path}"
        )

    if task_definition == "mission":
        category, task_goal_found, coverage_ratio = mission_outcome_from_payload(
            payload
        )
    else:
        category, task_goal_found, coverage_ratio = drone_skill_outcome_from_payload(
            payload,
            success_min_coverage_ratio,
        )
    transition_count = int(payload["team_reward"].shape[0])
    if transition_count <= 0:
        raise ValueError(f"Episode has no transitions: {path}")
    state_count = int(payload["global_state"]["central_map"].shape[0])
    if state_count != transition_count + 1:
        raise ValueError(
            f"Expected T+1 central states in {path}, got {state_count} for T={transition_count}."
        )
    for role in ("observer", "drone"):
        role_payload = payload[role]
        actions = role_payload["actions"]
        if int(actions.shape[0]) != transition_count:
            raise ValueError(f"Misaligned {role} actions: {path}")
        agent_ids = role_payload.get("agent_ids", [])
        observations = role_payload["observations"]
        if not agent_ids:
            if int(actions.shape[1]) != 0 or observations:
                raise ValueError(f"Invalid empty {role} group: {path}")
            continue
        if int(observations["global_map"].shape[0]) != state_count:
            raise ValueError(f"Misaligned {role} observations: {path}")

    return {
        "path": path.relative_to(data_root).as_posix(),
        "difficulty": difficulty,
        "task_id": difficulty - 1,
        "category": category,
        "task_success": category == "success",
        "task_goal_found": task_goal_found,
        "coverage_ratio": coverage_ratio,
        "task_success_reward": (
            DRONE_SKILL_SUCCESS_REWARD if task_definition == "drone" else 0.0
        ),
        "original_category": str(
            payload.get("outcome", {}).get("category", path.parent.name)
        ),
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
    success_min_coverage_ratio: float,
    task_definition: str,
) -> dict[tuple[int, str], list[dict[str, Any]]]:
    """Discover episodes and re-label them without moving legacy PT files."""
    strata: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for difficulty in difficulties:
        difficulty_dir = data_root / f"difficulty_{difficulty:02d}"
        if not difficulty_dir.is_dir():
            raise FileNotFoundError(f"Missing difficulty directory: {difficulty_dir}")
        paths = sorted(difficulty_dir.glob("*/*.pt"))
        if not paths:
            raise FileNotFoundError(f"No PT episodes found in {difficulty_dir}")
        for path in paths:
            entry = inspect_episode(
                path,
                data_root,
                difficulty,
                minimum_format_version,
                success_min_coverage_ratio,
                task_definition,
            )
            strata[(difficulty, entry["category"])].append(entry)
    return dict(strata)


def split_stratum(
    entries: list[dict[str, Any]],
    difficulty: int,
    category: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministically split one task/outcome stratum by episode."""
    shuffled = list(entries)
    random.Random(f"{seed}:{difficulty}:{category}").shuffle(shuffled)
    count = len(shuffled)
    if count == 1:
        return shuffled, [], []
    if count == 2:
        return shuffled[:1], [], shuffled[1:]
    train_count = max(int(count * train_ratio), 1)
    val_count = max(int(count * val_ratio), 1)
    if train_count + val_count >= count:
        train_count = count - 2
        val_count = 1
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
    success_min_coverage_ratio: float,
    task_definition: str = "mission",
) -> dict[str, Any]:
    """Build disjoint source and target train/val/test episode splits."""
    overlap = set(source_difficulties).intersection(target_difficulties)
    if overlap:
        raise ValueError(f"Source and target difficulties overlap: {sorted(overlap)}")
    if train_ratio <= 0.0 or val_ratio <= 0.0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("Split ratios must satisfy train > 0, val > 0, train + val < 1.")
    if task_definition not in ("mission", "drone"):
        raise ValueError(f"Unsupported task definition: {task_definition!r}")

    all_difficulties = tuple(source_difficulties) + tuple(target_difficulties)
    strata = discover_episodes(
        data_root,
        all_difficulties,
        minimum_format_version,
        success_min_coverage_ratio,
        task_definition,
    )
    splits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for difficulty in source_difficulties:
        for category in OUTCOME_CATEGORIES:
            if (difficulty, category) not in strata:
                continue
            train, val, test = split_stratum(
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
            if (difficulty, category) not in strata:
                continue
            train, val, test = split_stratum(
                strata[(difficulty, category)],
                difficulty,
                category,
                train_ratio,
                val_ratio,
                seed + 1,
            )
            splits["target_train"].extend(train)
            splits["target_val"].extend(val)
            splits["target_test"].extend(test)

    result = {
        "manifest_version": 2,
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
        "task_descriptor": {
            "kind": "realized_episode",
            "scope": "environment",
            "shared_across_agents": True,
            "names": list(REALIZED_TASK_DESCRIPTOR_NAMES),
        },
        "success_definition": (
            {
                "name": "observer_goal_arrival",
                "condition": "observer reaches goal",
            }
            if task_definition == "mission"
            else {
                "name": "drone_exploration",
                "drone_goal_found": True,
                "coverage_ratio": "full_map",
                "minimum_coverage_ratio": success_min_coverage_ratio,
                "fatal_crash": False,
                "terminal_reward": DRONE_SKILL_SUCCESS_REWARD,
            }
        ),
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
    default_output_name = (
        MISSION_MANIFEST_PATH.name
        if args.task_definition == "mission"
        else DRONE_TASK_MANIFEST_PATH.name
    )
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else data_root / default_output_name
    )
    if not 0.0 <= args.success_min_coverage_ratio <= 1.0:
        raise ValueError("--success-min-coverage-ratio must be in [0, 1].")
    manifest = build_manifest(
        data_root=data_root,
        source_difficulties=tuple(args.source_difficulties),
        target_difficulties=tuple(args.target_difficulties),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
        minimum_format_version=int(args.minimum_format_version),
        success_min_coverage_ratio=float(args.success_min_coverage_ratio),
        task_definition=args.task_definition,
    )
    save_manifest(manifest, output_path)
    print_summary(manifest, output_path)


if __name__ == "__main__":
    main()
