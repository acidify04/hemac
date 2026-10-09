#!/usr/bin/env python3
"""Canonical zero-shot evaluation for decentralized BC using the HiSSD protocol.

This file intentionally mirrors `evaluate_hissd_joint_hetero_zero_shot.py`.

The environment construction, start positions, seed formula, action scaling,
AEC stepping order, metrics, and JSON schema are kept aligned with the HiSSD
evaluator. The only model-specific difference is action inference:

    HiSSD: model.inference_step(...)
    BC-DE: model.forward_joint(...)

BC-DE action_i depends only on agent i's own actor observation o_i. Another
agent's separate observation is never mixed into the action path.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from hemac import HeMAC_v0
from hemac.curriculum_config import OBSTACLE_CURRICULUM_LEVELS
from skill_discovery.collect_offline_data import (
    agent_found_goal,
    build_collection_env_config,
    convert_observation,
    get_core_env,
)
from skill_discovery.decentralized_bc_models import (
    load_decentralized_bc_checkpoint,
)


DRONE_START_POSITIONS = {
    2: [[130.0, 850.0, 5.0], [170.0, 850.0, 5.0]],
    3: [
        [130.0, 870.0, 5.0],
        [170.0, 870.0, 5.0],
        [150.0, 830.0, 5.0],
    ],
    4: [
        [130.0, 870.0, 5.0],
        [130.0, 830.0, 5.0],
        [170.0, 870.0, 5.0],
        [170.0, 830.0, 5.0],
    ],
    5: [
        [130.0, 870.0, 5.0],
        [170.0, 870.0, 5.0],
        [110.0, 830.0, 5.0],
        [150.0, 830.0, 5.0],
        [190.0, 830.0, 5.0],
    ],
}

DIFFICULTY_1_CONFIG = {
    "min_obstacles": 3,
    "max_obstacles": 4,
    "obstacle_min_speed": 1,
    "obstacle_max_speed": 3,
    "n_static_obstacles": 2,
    "goal_min_base_distance": 475.0,
    "goal_max_base_distance": 600.0,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--env-template-data-root", type=Path, required=True)
    p.add_argument("--n-drones", type=int, required=True)
    p.add_argument("--n-observers", type=int, required=True)
    p.add_argument("--difficulty", type=int, default=1)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed-base", type=int, default=100_000_000)
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def find_template_episode(root: Path) -> Path:
    root = root.expanduser().resolve()
    paths = sorted(root.glob("difficulty_*/*/*.pt")) or sorted(root.rglob("*.pt"))
    for path in paths:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        config = payload.get("metadata", {}).get("environment_config")
        if isinstance(config, dict) and config:
            return path
    raise RuntimeError(
        f"No episode with environment_config under {root}"
    )


def load_env_template(root: Path) -> tuple[Path, dict]:
    path = find_template_episode(root)
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    return path, dict(payload["metadata"]["environment_config"])


def build_target_config(
    template: dict,
    difficulty: int,
    n_drones: int,
    n_observers: int,
) -> dict:
    if n_drones not in DRONE_START_POSITIONS:
        raise ValueError(f"unsupported n_drones={n_drones}")

    config = copy.deepcopy(
        build_collection_env_config(
            copy.deepcopy(template),
            difficulty,
        )
    )

    # Exactly the same D1 canonical override as the HiSSD evaluator.
    if int(difficulty) == 1:
        config.update(DIFFICULTY_1_CONFIG)

    config.update(
        n_drones=int(n_drones),
        n_observers=int(n_observers),
        n_provisioners=0,
        render_mode=None,
        log_step_rewards=False,
    )

    drone_config = copy.deepcopy(
        config.get("drone_config") or {}
    )
    drone_config["drones_starting_pos"] = copy.deepcopy(
        DRONE_START_POSITIONS[int(n_drones)]
    )
    config["drone_config"] = drone_config
    return config


def drone_scale(config: dict) -> float:
    return float(
        (config.get("drone_config") or {}).get(
            "drone_max_speed",
            25.0,
        )
    )


def observer_scale(config: dict) -> float:
    return float(config.get("observer_speed", 10.0))


def observation_batch(
    env,
    ids: list[str],
    role: str,
    scale: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    xs = [
        convert_observation(env.observe(agent_id), role)
        for agent_id in ids
    ]
    return {
        "global_map": torch.from_numpy(
            np.stack([x["global_map"] for x in xs])
        )
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32),
        "local_map": torch.from_numpy(
            np.stack([x["local_map"] for x in xs])
        )
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32),
        "action_history": (
            torch.from_numpy(
                np.stack([x["action_history"] for x in xs])
            )
            .unsqueeze(0)
            .to(device=device, dtype=torch.float32)
            / float(scale)
        ),
    }


def check_schema(
    obs: dict[str, torch.Tensor],
    model,
    role: str,
) -> None:
    config = model.role_configs[role]
    expected = {
        "global_map": (
            int(config["global_map_channels"]),
            *tuple(config["global_map_size"]),
        ),
        "local_map": (
            int(config["local_map_channels"]),
            *tuple(config["local_map_size"]),
        ),
        "action_history": tuple(
            config["action_history_shape"]
        ),
    }
    for key, tail in expected.items():
        actual = tuple(obs[key].shape[-len(tail):])
        if actual != tuple(tail):
            raise ValueError(
                f"{role} {key} mismatch: "
                f"{actual} vs {tail}"
            )


def to_env_actions(
    normalized: torch.Tensor,
    ids: list[str],
    env,
    scale: float,
) -> dict[str, np.ndarray]:
    out = {}
    for index, agent_id in enumerate(ids):
        space = env.action_space(agent_id)
        action = (
            normalized[index].detach().cpu().numpy()
            * float(scale)
        )
        out[agent_id] = np.ascontiguousarray(
            np.clip(action, space.low, space.high),
            dtype=np.float32,
        )
    return out


@torch.inference_mode()
def run_episode(
    config: dict,
    model,
    seed: int,
    difficulty: int,
    device: torch.device,
) -> dict[str, float]:
    env = HeMAC_v0.env(**config)
    try:
        env.reset(seed=seed)
        core = get_core_env(env)
        order = list(env.possible_agents)

        observers = [
            agent_id
            for agent_id in order
            if agent_id.startswith("observer_")
        ]
        drones = [
            agent_id
            for agent_id in order
            if agent_id.startswith("drone_")
        ]

        drone_action_scale = drone_scale(config)
        observer_action_scale = observer_scale(config)

        masks = {
            "observer": torch.ones(
                1,
                len(observers),
                dtype=torch.bool,
                device=device,
            ),
            "drone": torch.ones(
                1,
                len(drones),
                dtype=torch.bool,
                device=device,
            ),
        }

        cached_actions: dict[str, np.ndarray] = {}
        cycles = 0
        reward_sum = 0.0
        final_info: dict = {}
        checked_schema = False
        last_agent = order[-1]

        for agent_id in env.agent_iter():
            _, reward, termination, truncation, info = env.last()
            reward_sum += float(reward)

            if info:
                final_info.update(info)

            if termination or truncation:
                env.step(None)
                continue

            if not cached_actions:
                observations = {
                    "observer": observation_batch(
                        env,
                        observers,
                        "observer",
                        observer_action_scale,
                        device,
                    ),
                    "drone": observation_batch(
                        env,
                        drones,
                        "drone",
                        drone_action_scale,
                        device,
                    ),
                }

                if not checked_schema:
                    check_schema(
                        observations["observer"],
                        model,
                        "observer",
                    )
                    check_schema(
                        observations["drone"],
                        model,
                        "drone",
                    )
                    checked_schema = True

                # BC-specific line.
                # No recurrent/joint HiSSD state and no cross-agent mixing:
                # each output action is produced from that agent's o_i only.
                outputs = model.forward_joint(
                    observations,
                    masks,
                )

                cached_actions.update(
                    to_env_actions(
                        outputs["actions"]["observer"][0],
                        observers,
                        env,
                        observer_action_scale,
                    )
                )
                cached_actions.update(
                    to_env_actions(
                        outputs["actions"]["drone"][0],
                        drones,
                        env,
                        drone_action_scale,
                    )
                )

            env.step(cached_actions.pop(agent_id))

            if (
                agent_id == last_agent
                or bool(core.terminate)
                or bool(core.truncate)
            ):
                cycles += 1
                cached_actions.clear()

        if hasattr(core, "build_episode_info"):
            final_info.update(core.build_episode_info())

        observer_goal = any(
            agent_found_goal(core, agent_id)
            for agent_id in observers
        )
        drone_goal = any(
            agent_found_goal(core, agent_id)
            for agent_id in drones
        )

        return {
            "seed": int(seed),
            "difficulty": int(difficulty),
            "n_drones": len(drones),
            "n_observers": len(observers),
            "success": float(
                bool(
                    final_info.get(
                        "success",
                        core.mission_success,
                    )
                )
            ),
            "goal_found": float(
                bool(
                    final_info.get(
                        "goal_found",
                        observer_goal,
                    )
                )
            ),
            "observer_goal_found": float(observer_goal),
            "drone_goal_found": float(drone_goal),
            "fatal_crash": float(
                bool(
                    final_info.get(
                        "fatal_crash",
                        core.collided,
                    )
                )
            ),
            "drone_crash": float(
                bool(
                    final_info.get(
                        "drone_crash",
                        getattr(core, "drone_crash", False),
                    )
                )
            ),
            "observer_crash": float(
                bool(
                    final_info.get(
                        "observer_crash",
                        getattr(core, "observer_crash", False),
                    )
                )
            ),
            "coverage": float(
                core.current_coverage_ratio()
            ),
            "cycles": float(cycles),
            "aec_reward_sum": float(reward_sum),
        }
    finally:
        env.close()


def aggregate(
    records: list[dict[str, float]],
) -> dict[str, float]:
    keys = (
        "success",
        "goal_found",
        "observer_goal_found",
        "drone_goal_found",
        "fatal_crash",
        "drone_crash",
        "observer_crash",
        "coverage",
        "cycles",
        "aec_reward_sum",
    )
    out = {}
    for key in keys:
        values = np.asarray(
            [record[key] for record in records],
            dtype=np.float64,
        )
        out[key] = float(values.mean())
        out[key + "_std"] = float(
            values.std(ddof=0)
        )
        out[key + "_sem"] = float(
            values.std(ddof=1) / np.sqrt(len(values))
            if len(values) > 1
            else 0.0
        )
    return out


def main() -> None:
    args = parse_args()

    if (
        args.n_drones not in DRONE_START_POSITIONS
        or args.n_observers not in (1, 2)
    ):
        raise ValueError("unsupported population")

    if not 1 <= args.difficulty <= len(
        OBSTACLE_CURRICULUM_LEVELS
    ):
        raise ValueError("invalid difficulty")

    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")

    device = resolve_device(args.device)

    template_path, template = load_env_template(
        args.env_template_data_root
    )
    config = build_target_config(
        template,
        args.difficulty,
        args.n_drones,
        args.n_observers,
    )

    model, payload = load_decentralized_bc_checkpoint(
        args.checkpoint,
        device,
    )

    expected_type = "hemac_decentralized_behavior_cloning"
    if payload.get("model_type") != expected_type:
        raise ValueError(
            "checkpoint type mismatch: "
            f"{payload.get('model_type')!r}"
        )

    model.eval()
    records = []

    iterator = range(args.episodes)

    print(
        f"device={device} "
        f"target=D{args.difficulty} "
        f"D{args.n_drones}O{args.n_observers} "
        f"checkpoint_epoch={payload.get('epoch')}"
    )
    print(
        f"seed_base={args.seed_base} "
        f"first_seed="
        f"{args.seed_base + args.difficulty * 100000} "
        f"last_seed="
        f"{args.seed_base + args.difficulty * 100000 + args.episodes - 1}"
    )

    if tqdm is not None and not args.quiet:
        iterator = tqdm(
            iterator,
            desc=(
                f"decentralized BC "
                f"D{args.n_drones}O{args.n_observers}"
            ),
            unit="ep",
            dynamic_ncols=True,
        )

    for episode_index in iterator:
        seed = (
            args.seed_base
            + args.difficulty * 100_000
            + episode_index
        )

        records.append(
            run_episode(
                config,
                model,
                seed,
                args.difficulty,
                device,
            )
        )

        if tqdm is not None and not args.quiet:
            n = len(records)
            iterator.set_postfix(
                success=(
                    f"{sum(r['success'] for r in records) / n:.3f}"
                ),
                crash=(
                    f"{sum(r['fatal_crash'] for r in records) / n:.3f}"
                ),
                coverage=(
                    f"{sum(r['coverage'] for r in records) / n:.3f}"
                ),
            )

    summary = aggregate(records)

    result = {
        "evaluation_type": (
            "decentralized_bc_"
            "hissd_protocol_"
            "zero_shot_population_generalization"
        ),
        "deterministic": True,
        "gradient_updates": 0,
        "target": {
            "difficulty": args.difficulty,
            "n_drones": args.n_drones,
            "n_observers": args.n_observers,
            "drone_start_positions": (
                config["drone_config"]["drones_starting_pos"]
            ),
        },
        "canonical_difficulty_config": (
            DIFFICULTY_1_CONFIG
            if args.difficulty == 1
            else None
        ),
        "episodes": args.episodes,
        "seed_base": args.seed_base,
        "seed_formula": (
            "seed_base + difficulty*100000 + episode_index"
        ),
        "template_episode": str(template_path),
        "environment_config": config,
        "checkpoint": {
            "path": str(args.checkpoint),
            "epoch": payload.get("epoch"),
            "model_type": payload.get("model_type"),
        },
        "information_constraint": {
            "action_uses_only_own_actor_observation": True,
            "cross_agent_feature_mixing": False,
            "other_agent_private_observation": False,
        },
        "summary": summary,
        "episode_records": records,
    }

    args.output_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output_json.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(
        "\n===== DECENTRALIZED BC "
        "(HISSD PROTOCOL) ZERO-SHOT RESULT ====="
    )
    for key in (
        "success",
        "goal_found",
        "observer_goal_found",
        "drone_goal_found",
        "fatal_crash",
        "drone_crash",
        "observer_crash",
        "coverage",
        "cycles",
    ):
        print(f"{key:28s}= {summary[key]:.4f}")

    print(f"\noutput = {args.output_json}")


if __name__ == "__main__":
    main()
