#!/usr/bin/env python3
"""Run one canonical HeMAC episode from a joint HiSSD checkpoint and save MP4.

Supported checkpoint types
--------------------------
- hemac_joint_heterogeneous_hissd
- hemac_joint_heterogeneous_hissd_agent_conditioned_adapter

The rollout path matches the joint zero-shot evaluators:
- canonical population-specific drone starts,
- one joint policy inference per AEC cycle,
- identical action scaling/clipping,
- deterministic seed formula when --seed is omitted.

Example
-------
python -u src/skill_discovery/visualize_hissd_episode.py \
  --checkpoint src/skill_discovery/checkpoints/hissd-joint-hetero/hissd_joint_best.pt \
  --env-template-data-root src/skill_discovery/offline_data-4-2 \
  --n-drones 3 --n-observers 1 --difficulty 1 \
  --seed 100100000 \
  --output-mp4 outputs/hissd_D3O1_seed100100000.mp4 \
  --device cuda
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

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
from skill_discovery.hissd_joint_hetero_models import (
    load_joint_heterogeneous_hissd_checkpoint,
)
from skill_discovery.hissd_joint_hetero_adapter_models import (
    load_joint_heterogeneous_adapter_checkpoint,
)


DRONE_START_POSITIONS = {
    2: [[130.0, 850.0, 5.0], [170.0, 850.0, 5.0]],
    3: [[130.0, 870.0, 5.0], [170.0, 870.0, 5.0], [150.0, 830.0, 5.0]],
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

SUPPORTED_MODEL_TYPES = {
    "hemac_joint_heterogeneous_hissd": "no_adapter",
    "hemac_joint_heterogeneous_hissd_agent_conditioned_adapter": "adapter",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--env-template-data-root", type=Path, required=True)
    p.add_argument("--n-drones", type=int, default=3)
    p.add_argument("--n-observers", type=int, default=1)
    p.add_argument("--difficulty", type=int, default=1)

    # Either give --seed directly, or use the same formula as zero-shot eval.
    p.add_argument("--seed", type=int)
    p.add_argument("--seed-base", type=int, default=100_000_000)
    p.add_argument("--episode-index", type=int, default=0)

    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument(
        "--capture-every-cycles",
        type=int,
        default=1,
        help="Capture one frame every N completed joint cycles.",
    )
    p.add_argument(
        "--hold-last-seconds",
        type=float,
        default=1.5,
        help="Repeat the final frame for this many seconds.",
    )
    p.add_argument("--output-mp4", type=Path, required=True)
    p.add_argument(
        "--output-json",
        type=Path,
        help="Optional episode summary JSON. Defaults next to the MP4.",
    )
    return p.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return torch.device(name)


def find_template_episode(root: Path) -> Path:
    root = root.expanduser().resolve()
    paths = sorted(root.glob("difficulty_*/*/*.pt")) or sorted(root.rglob("*.pt"))
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        cfg = payload.get("metadata", {}).get("environment_config")
        if isinstance(cfg, dict) and cfg:
            return path
    raise RuntimeError(f"No episode with metadata.environment_config under {root}")


def load_env_template(root: Path) -> tuple[Path, dict[str, Any]]:
    path = find_template_episode(root)
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    return path, dict(payload["metadata"]["environment_config"])


def build_target_config(
    template: dict[str, Any],
    difficulty: int,
    n_drones: int,
    n_observers: int,
) -> dict[str, Any]:
    if n_drones not in DRONE_START_POSITIONS:
        raise ValueError(f"--n-drones must be one of {sorted(DRONE_START_POSITIONS)}")
    if n_observers not in (1, 2):
        raise ValueError("--n-observers must be 1 or 2")
    if not 1 <= difficulty <= len(OBSTACLE_CURRICULUM_LEVELS):
        raise ValueError("Invalid --difficulty")

    config = copy.deepcopy(
        build_collection_env_config(copy.deepcopy(template), difficulty)
    )
    if difficulty == 1:
        config.update(DIFFICULTY_1_CONFIG)

    config["n_drones"] = int(n_drones)
    config["n_observers"] = int(n_observers)
    config["n_provisioners"] = 0
    config["render_mode"] = "rgb_array"
    config["log_step_rewards"] = False

    dc = copy.deepcopy(config.get("drone_config") or {})
    dc["drones_starting_pos"] = copy.deepcopy(DRONE_START_POSITIONS[n_drones])
    config["drone_config"] = dc
    return config


def drone_scale(config: dict[str, Any]) -> float:
    value = float((config.get("drone_config") or {}).get("drone_max_speed", 25.0))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid drone action scale: {value}")
    return value


def observer_scale(config: dict[str, Any]) -> float:
    value = float(config.get("observer_speed", 10.0))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid observer action scale: {value}")
    return value


def observation_batch(env, ids, role: str, scale: float, device: torch.device):
    converted = [convert_observation(env.observe(aid), role) for aid in ids]
    if not converted:
        raise RuntimeError(f"No {role} agents found")
    return {
        "global_map": torch.from_numpy(
            np.stack([x["global_map"] for x in converted])
        ).unsqueeze(0).to(device=device, dtype=torch.float32),
        "local_map": torch.from_numpy(
            np.stack([x["local_map"] for x in converted])
        ).unsqueeze(0).to(device=device, dtype=torch.float32),
        "action_history": (
            torch.from_numpy(np.stack([x["action_history"] for x in converted]))
            .unsqueeze(0)
            .to(device=device, dtype=torch.float32)
            / float(scale)
        ),
    }


def check_role_schema(obs, model, role: str) -> None:
    cfg = model.role_configs[role]
    expected = {
        "global_map": (
            int(cfg["global_map_channels"]),
            *tuple(cfg["global_map_size"]),
        ),
        "local_map": (
            int(cfg["local_map_channels"]),
            *tuple(cfg["local_map_size"]),
        ),
        "action_history": tuple(cfg["action_history_shape"]),
    }
    for key, tail in expected.items():
        actual = tuple(obs[key].shape[-len(tail):])
        if actual != tuple(tail):
            raise ValueError(
                f"{role} {key} mismatch: environment={actual}, checkpoint={tail}"
            )


def to_env_actions(normalized, ids, env, scale: float):
    if normalized.shape[0] != len(ids):
        raise RuntimeError(
            f"Action count mismatch: output={normalized.shape[0]}, agents={len(ids)}"
        )
    out = {}
    for i, aid in enumerate(ids):
        space = env.action_space(aid)
        action = normalized[i].detach().cpu().numpy() * float(scale)
        out[aid] = np.ascontiguousarray(
            np.clip(action, space.low, space.high), dtype=np.float32
        )
    return out


def load_joint_model(checkpoint: Path, device: torch.device):
    checkpoint = checkpoint.expanduser().resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_type = payload.get("model_type")
    mode = SUPPORTED_MODEL_TYPES.get(model_type)
    if mode is None:
        raise ValueError(
            "Unsupported checkpoint model_type. This visualizer expects one joint "
            f"HiSSD checkpoint, got {model_type!r}. Supported: {sorted(SUPPORTED_MODEL_TYPES)}"
        )

    if mode == "adapter":
        model, payload = load_joint_heterogeneous_adapter_checkpoint(checkpoint, device)
    else:
        model, payload = load_joint_heterogeneous_hissd_checkpoint(checkpoint, device)
    model.eval()
    return model, payload, mode


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
        raise ValueError(f"Unexpected render frame shape: {frame.shape}")
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.floating):
            if frame.max(initial=0.0) <= 1.0:
                frame = frame * 255.0
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        else:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def open_video_writer(path: Path, fps: int):
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError(
            "imageio is required to save MP4. Install imageio and imageio-ffmpeg "
            "in the hemac environment."
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        return imageio.get_writer(
            str(path),
            fps=int(fps),
            codec="libx264",
            quality=8,
            macro_block_size=1,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not open MP4 writer. Usually `pip install imageio imageio-ffmpeg` "
            "inside the hemac environment fixes this."
        ) from exc


@torch.inference_mode()
def run_episode_and_record(
    config: dict[str, Any],
    model,
    seed: int,
    difficulty: int,
    device: torch.device,
    output_mp4: Path,
    fps: int,
    capture_every_cycles: int,
    hold_last_seconds: float,
) -> dict[str, Any]:
    env = HeMAC_v0.env(**config)
    writer = None
    last_frame = None
    frame_count = 0

    try:
        env.reset(seed=int(seed))
        core = get_core_env(env)
        order = list(env.possible_agents)
        observers = [aid for aid in order if aid.startswith("observer_")]
        drones = [aid for aid in order if aid.startswith("drone_")]
        unsupported = [aid for aid in order if aid not in set(observers + drones)]
        if unsupported:
            raise RuntimeError(f"No policy for agents: {unsupported}")

        ds = drone_scale(config)
        os = observer_scale(config)
        masks = {
            "observer": torch.ones(1, len(observers), dtype=torch.bool, device=device),
            "drone": torch.ones(1, len(drones), dtype=torch.bool, device=device),
        }
        state = model.initial_joint_inference_state(
            batch_size=1,
            observer_count=len(observers),
            drone_count=len(drones),
            device=device,
        )

        writer = open_video_writer(output_mp4, fps)

        def capture() -> None:
            nonlocal last_frame, frame_count
            frame = env.render()
            if frame is None:
                raise RuntimeError(
                    "env.render() returned None even though render_mode='rgb_array'."
                )
            last_frame = normalize_frame(frame)
            writer.append_data(last_frame)
            frame_count += 1

        # Initial state.
        capture()

        cached = {}
        cycles = 0
        reward_sum = 0.0
        final_info: dict[str, Any] = {}
        checked = False
        last_agent = order[-1]

        for aid in env.agent_iter():
            _, reward, termination, truncation, info = env.last()
            reward_sum += float(reward)
            if info:
                final_info.update(info)

            if termination or truncation:
                env.step(None)
                continue

            # Compute actions for every agent from the same cycle-start snapshot.
            if not cached:
                obs = {
                    "observer": observation_batch(env, observers, "observer", os, device),
                    "drone": observation_batch(env, drones, "drone", ds, device),
                }
                if not checked:
                    check_role_schema(obs["observer"], model, "observer")
                    check_role_schema(obs["drone"], model, "drone")
                    checked = True

                outputs, state = model.inference_step(obs, masks, state)
                cached.update(
                    to_env_actions(outputs["actions"]["observer"][0], observers, env, os)
                )
                cached.update(
                    to_env_actions(outputs["actions"]["drone"][0], drones, env, ds)
                )

            env.step(cached.pop(aid))

            cycle_finished = (
                aid == last_agent
                or bool(getattr(core, "terminate", False))
                or bool(getattr(core, "truncate", False))
            )
            if cycle_finished:
                cycles += 1
                cached.clear()
                if cycles % capture_every_cycles == 0 or bool(core.terminate) or bool(core.truncate):
                    capture()

        if hasattr(core, "build_episode_info"):
            final_info.update(core.build_episode_info())

        observer_goal = any(agent_found_goal(core, aid) for aid in observers)
        drone_goal = any(agent_found_goal(core, aid) for aid in drones)

        # Hold the terminal state long enough to inspect the ending.
        if last_frame is not None and hold_last_seconds > 0:
            extra = max(0, int(round(float(hold_last_seconds) * fps)))
            for _ in range(extra):
                writer.append_data(last_frame)
                frame_count += 1

        return {
            "seed": int(seed),
            "difficulty": int(difficulty),
            "n_drones": len(drones),
            "n_observers": len(observers),
            "success": bool(final_info.get("success", core.mission_success)),
            "goal_found": bool(final_info.get("goal_found", observer_goal)),
            "observer_goal_found": bool(observer_goal),
            "drone_goal_found": bool(drone_goal),
            "fatal_crash": bool(final_info.get("fatal_crash", core.collided)),
            "drone_crash": bool(final_info.get("drone_crash", getattr(core, "drone_crash", False))),
            "observer_crash": bool(final_info.get("observer_crash", getattr(core, "observer_crash", False))),
            "coverage": float(core.current_coverage_ratio()),
            "cycles": int(cycles),
            "aec_reward_sum": float(reward_sum),
            "video_frames": int(frame_count),
            "video_fps": int(fps),
            "video_seconds": float(frame_count / fps),
        }
    finally:
        if writer is not None:
            writer.close()
        env.close()


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.capture_every_cycles <= 0:
        raise ValueError("--capture-every-cycles must be positive")
    if args.hold_last_seconds < 0:
        raise ValueError("--hold-last-seconds cannot be negative")
    if args.episode_index < 0:
        raise ValueError("--episode-index cannot be negative")

    seed = (
        int(args.seed)
        if args.seed is not None
        else int(args.seed_base + args.difficulty * 100_000 + args.episode_index)
    )

    device = resolve_device(args.device)
    template_path, template = load_env_template(args.env_template_data_root)
    config = build_target_config(
        template,
        args.difficulty,
        args.n_drones,
        args.n_observers,
    )
    model, payload, model_variant = load_joint_model(args.checkpoint, device)

    print(f"device={device}")
    print(f"checkpoint={args.checkpoint}")
    print(f"checkpoint_epoch={payload.get('epoch')}")
    print(f"model_type={payload.get('model_type')} ({model_variant})")
    print(f"target=D{args.difficulty} D{args.n_drones}O{args.n_observers}")
    print(f"seed={seed}")
    print(f"template_episode={template_path}")
    print(f"output_mp4={args.output_mp4}")

    result = run_episode_and_record(
        config=config,
        model=model,
        seed=seed,
        difficulty=args.difficulty,
        device=device,
        output_mp4=args.output_mp4,
        fps=args.fps,
        capture_every_cycles=args.capture_every_cycles,
        hold_last_seconds=args.hold_last_seconds,
    )

    output_json = args.output_json
    if output_json is None:
        output_json = args.output_mp4.with_suffix(".json")
    output_json.parent.mkdir(parents=True, exist_ok=True)

    payload_out = {
        "checkpoint": {
            "path": str(args.checkpoint),
            "epoch": payload.get("epoch"),
            "model_type": payload.get("model_type"),
            "initialization": payload.get("initialization"),
            "bc_initialization": payload.get("bc_initialization"),
        },
        "target": {
            "difficulty": args.difficulty,
            "n_drones": args.n_drones,
            "n_observers": args.n_observers,
            "drone_start_positions": config["drone_config"]["drones_starting_pos"],
        },
        "seed": seed,
        "seed_formula": (
            "explicit --seed"
            if args.seed is not None
            else "seed_base + difficulty*100000 + episode_index"
        ),
        "template_episode": str(template_path),
        "video": str(args.output_mp4),
        "episode": result,
    }
    output_json.write_text(json.dumps(payload_out, indent=2), encoding="utf-8")

    print("\n===== EPISODE RESULT =====")
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
        "video_frames",
        "video_seconds",
    ):
        print(f"{key:24s}= {result[key]}")
    print(f"mp4                     = {args.output_mp4}")
    print(f"json                    = {output_json}")


if __name__ == "__main__":
    main()
