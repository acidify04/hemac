#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
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
from skill_discovery.hissd_joint_hetero_adapter_models import (
    load_joint_heterogeneous_adapter_checkpoint,
)

DRONE_START_POSITIONS = {
    2: [[130.0, 850.0, 5.0], [170.0, 850.0, 5.0]],
    3: [[130.0, 870.0, 5.0], [170.0, 870.0, 5.0], [150.0, 830.0, 5.0]],
    4: [[130.0, 870.0, 5.0], [130.0, 830.0, 5.0], [170.0, 870.0, 5.0], [170.0, 830.0, 5.0]],
    5: [[130.0, 870.0, 5.0], [170.0, 870.0, 5.0], [110.0, 830.0, 5.0], [150.0, 830.0, 5.0], [190.0, 830.0, 5.0]],
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


def parse_args():
    p = argparse.ArgumentParser(description="Zero-shot eval for one joint heterogeneous HiSSD+adapter checkpoint.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--env-template-data-root", type=Path, required=True)
    p.add_argument("--n-drones", type=int, required=True)
    p.add_argument("--n-observers", type=int, required=True)
    p.add_argument("--difficulty", type=int, default=1)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed-base", type=int, default=100_000_000)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def find_template_episode(root: Path) -> Path:
    root = root.expanduser().resolve()
    paths = sorted(root.glob("difficulty_*/*/*.pt")) or sorted(root.rglob("*.pt"))
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        cfg = payload.get("metadata", {}).get("environment_config")
        if isinstance(cfg, dict) and cfg:
            return path
    raise RuntimeError(f"No episode with environment_config under {root}")


def load_env_template(root: Path):
    path = find_template_episode(root)
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    return path, dict(payload["metadata"]["environment_config"])


def build_target_config(template, difficulty, n_drones, n_observers):
    if n_drones not in DRONE_START_POSITIONS:
        raise ValueError(f"--n-drones must be one of {sorted(DRONE_START_POSITIONS)}")
    config = copy.deepcopy(build_collection_env_config(copy.deepcopy(template), difficulty))
    if int(difficulty) == 1:
        config.update(DIFFICULTY_1_CONFIG)
    config["n_drones"] = int(n_drones)
    config["n_observers"] = int(n_observers)
    config["n_provisioners"] = 0
    config["render_mode"] = None
    config["log_step_rewards"] = False
    dc = copy.deepcopy(config.get("drone_config") or {})
    dc["drones_starting_pos"] = copy.deepcopy(DRONE_START_POSITIONS[int(n_drones)])
    config["drone_config"] = dc
    return config


def drone_scale(config):
    x = float((config.get("drone_config") or {}).get("drone_max_speed", 25.0))
    if not math.isfinite(x) or x <= 0:
        raise ValueError(f"Invalid drone scale: {x}")
    return x


def observer_scale(config):
    x = float(config.get("observer_speed", 10.0))
    if not math.isfinite(x) or x <= 0:
        raise ValueError(f"Invalid observer scale: {x}")
    return x


def observation_batch(env, ids, role, scale, device):
    converted = [convert_observation(env.observe(a), role) for a in ids]
    return {
        "global_map": torch.from_numpy(np.stack([x["global_map"] for x in converted])).unsqueeze(0).to(device=device, dtype=torch.float32),
        "local_map": torch.from_numpy(np.stack([x["local_map"] for x in converted])).unsqueeze(0).to(device=device, dtype=torch.float32),
        "action_history": torch.from_numpy(np.stack([x["action_history"] for x in converted])).unsqueeze(0).to(device=device, dtype=torch.float32) / float(scale),
    }


def check_role_schema(obs, model, role):
    cfg = model.role_configs[role]
    expected = {
        "global_map": (int(cfg["global_map_channels"]), *tuple(cfg["global_map_size"])),
        "local_map": (int(cfg["local_map_channels"]), *tuple(cfg["local_map_size"])),
        "action_history": tuple(cfg["action_history_shape"]),
    }
    for key, tail in expected.items():
        actual = tuple(obs[key].shape[-len(tail):])
        if actual != tuple(tail):
            raise ValueError(f"{role} {key} mismatch: environment={actual}, checkpoint={tail}")


def to_env_actions(normalized, ids, env, scale):
    if normalized.shape[0] != len(ids):
        raise RuntimeError(f"Action count mismatch: output={normalized.shape[0]}, agents={len(ids)}")
    out = {}
    for i, aid in enumerate(ids):
        space = env.action_space(aid)
        action = normalized[i].detach().cpu().numpy() * float(scale)
        out[aid] = np.ascontiguousarray(np.clip(action, space.low, space.high), dtype=np.float32)
    return out


def masked_rms(x, mask):
    m = mask.bool().unsqueeze(-1).to(x.dtype)
    denom = (m.sum() * x.shape[-1]).clamp_min(1.0)
    return float(torch.sqrt((x.square() * m).sum() / denom).item())


def adapter_stats(outputs):
    delta = outputs["common_skill_delta"]
    raw = outputs["raw_common_skills"]
    mask = outputs["valid_mask"]
    slices = outputs["role_slices"]
    d = masked_rms(delta, mask)
    r = masked_rms(raw, mask)
    result = {
        "adapter_delta_rms": d,
        "adapter_to_common_ratio": d / max(r, 1e-8),
    }
    for role in ("observer", "drone"):
        sl = slices[role]
        rd = masked_rms(delta[..., sl, :], mask[..., sl])
        rr = masked_rms(raw[..., sl, :], mask[..., sl])
        result[f"{role}_adapter_delta_rms"] = rd
        result[f"{role}_adapter_to_common_ratio"] = rd / max(rr, 1e-8)
    return result


@torch.inference_mode()
def run_episode(config, model, seed, difficulty, device):
    env = HeMAC_v0.env(**config)
    try:
        env.reset(seed=seed)
        core = get_core_env(env)
        order = list(env.possible_agents)
        observers = [a for a in order if a.startswith("observer_")]
        drones = [a for a in order if a.startswith("drone_")]
        unsupported = [a for a in order if a not in set(observers + drones)]
        if unsupported:
            raise RuntimeError(f"No policy for agents: {unsupported}")
        if len(drones) != config["n_drones"] or len(observers) != config["n_observers"]:
            raise RuntimeError(f"Population mismatch: got D{len(drones)}O{len(observers)}")

        ds, os = drone_scale(config), observer_scale(config)
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

        cached = {}
        cycles = 0
        reward_sum = 0.0
        final_info = {}
        checked = False
        last_agent = order[-1]
        diag_sum = {k: 0.0 for k in (
            "adapter_delta_rms", "adapter_to_common_ratio",
            "observer_adapter_delta_rms", "observer_adapter_to_common_ratio",
            "drone_adapter_delta_rms", "drone_adapter_to_common_ratio",
        )}
        diag_n = 0

        for aid in env.agent_iter():
            _, reward, termination, truncation, info = env.last()
            reward_sum += float(reward)
            if info:
                final_info.update(info)
            if termination or truncation:
                env.step(None)
                continue

            if not cached:
                obs_by_role = {
                    "observer": observation_batch(env, observers, "observer", os, device),
                    "drone": observation_batch(env, drones, "drone", ds, device),
                }
                if not checked:
                    check_role_schema(obs_by_role["observer"], model, "observer")
                    check_role_schema(obs_by_role["drone"], model, "drone")
                    checked = True

                outputs, state = model.inference_step(obs_by_role, masks, state)
                cached.update(to_env_actions(outputs["actions"]["observer"][0], observers, env, os))
                cached.update(to_env_actions(outputs["actions"]["drone"][0], drones, env, ds))
                s = adapter_stats(outputs)
                for k, v in s.items():
                    diag_sum[k] += float(v)
                diag_n += 1

            env.step(cached.pop(aid))
            if aid == last_agent or bool(core.terminate) or bool(core.truncate):
                cycles += 1
                cached.clear()

        if hasattr(core, "build_episode_info"):
            final_info.update(core.build_episode_info())

        observer_goal = any(agent_found_goal(core, x) for x in observers)
        drone_goal = any(agent_found_goal(core, x) for x in drones)
        diagnostics = {k: (v / diag_n if diag_n else 0.0) for k, v in diag_sum.items()}

        return {
            "seed": int(seed),
            "difficulty": int(difficulty),
            "n_drones": len(drones),
            "n_observers": len(observers),
            "success": float(bool(final_info.get("success", core.mission_success))),
            "goal_found": float(bool(final_info.get("goal_found", observer_goal))),
            "observer_goal_found": float(observer_goal),
            "drone_goal_found": float(drone_goal),
            "fatal_crash": float(bool(final_info.get("fatal_crash", core.collided))),
            "drone_crash": float(bool(final_info.get("drone_crash", getattr(core, "drone_crash", False)))),
            "observer_crash": float(bool(final_info.get("observer_crash", getattr(core, "observer_crash", False)))),
            "coverage": float(core.current_coverage_ratio()),
            "cycles": float(cycles),
            "aec_reward_sum": float(reward_sum),
            **diagnostics,
        }
    finally:
        env.close()


def aggregate(records):
    keys = (
        "success", "goal_found", "observer_goal_found", "drone_goal_found",
        "fatal_crash", "drone_crash", "observer_crash", "coverage", "cycles",
        "aec_reward_sum", "adapter_delta_rms", "adapter_to_common_ratio",
        "observer_adapter_delta_rms", "observer_adapter_to_common_ratio",
        "drone_adapter_delta_rms", "drone_adapter_to_common_ratio",
    )
    out = {}
    for key in keys:
        x = np.asarray([r[key] for r in records], dtype=np.float64)
        out[key] = float(x.mean())
        out[key + "_std"] = float(x.std(ddof=0))
        out[key + "_sem"] = float(x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0)
    return out


def main():
    args = parse_args()
    if args.n_drones not in DRONE_START_POSITIONS:
        raise ValueError(f"--n-drones must be one of {sorted(DRONE_START_POSITIONS)}")
    if args.n_observers not in (1, 2):
        raise ValueError("--n-observers must be 1 or 2")
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if not 1 <= args.difficulty <= len(OBSTACLE_CURRICULUM_LEVELS):
        raise ValueError("Invalid difficulty")

    device = resolve_device(args.device)
    template_path, template = load_env_template(args.env_template_data_root)
    config = build_target_config(template, args.difficulty, args.n_drones, args.n_observers)

    model, payload = load_joint_heterogeneous_adapter_checkpoint(args.checkpoint, device)
    expected_type = "hemac_joint_heterogeneous_hissd_agent_conditioned_adapter"
    if payload.get("model_type") != expected_type:
        raise ValueError(f"Checkpoint model_type mismatch: {payload.get('model_type')!r}")
    model.eval()

    print(f"device={device} target=D{args.difficulty} D{args.n_drones}O{args.n_observers}")
    print(f"checkpoint={args.checkpoint}")
    print(f"checkpoint_epoch={payload.get('epoch')}")
    print(f"template_episode={template_path}")
    print("drone_starts=" + json.dumps(config["drone_config"]["drones_starting_pos"]))

    records = []
    iterator = range(args.episodes)
    if tqdm is not None and not args.quiet:
        iterator = tqdm(iterator, desc=f"joint zero-shot D{args.n_drones}O{args.n_observers}", unit="ep", dynamic_ncols=True)

    for episode_index in iterator:
        seed = args.seed_base + args.difficulty * 100_000 + episode_index
        record = run_episode(config, model, seed, args.difficulty, device)
        records.append(record)
        if tqdm is not None and not args.quiet:
            n = len(records)
            iterator.set_postfix(
                success=f"{sum(r['success'] for r in records)/n:.3f}",
                crash=f"{sum(r['fatal_crash'] for r in records)/n:.3f}",
                coverage=f"{sum(r['coverage'] for r in records)/n:.3f}",
            )

    summary = aggregate(records)
    result = {
        "evaluation_type": "joint_heterogeneous_hissd_adapter_zero_shot_population_generalization",
        "deterministic": True,
        "gradient_updates": 0,
        "target": {
            "difficulty": args.difficulty,
            "n_drones": args.n_drones,
            "n_observers": args.n_observers,
            "drone_start_positions": config["drone_config"]["drones_starting_pos"],
        },
        "canonical_difficulty_config": DIFFICULTY_1_CONFIG if args.difficulty == 1 else None,
        "episodes": args.episodes,
        "seed_base": args.seed_base,
        "template_episode": str(template_path),
        "environment_config": config,
        "checkpoint": {
            "path": str(args.checkpoint),
            "epoch": payload.get("epoch"),
            "model_type": payload.get("model_type"),
            "bc_initialization": payload.get("bc_initialization"),
        },
        "summary": summary,
        "episode_records": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n===== JOINT ZERO-SHOT RESULT =====")
    for key in ("success", "goal_found", "observer_goal_found", "drone_goal_found", "fatal_crash", "drone_crash", "observer_crash", "coverage", "cycles"):
        print(f"{key:32s}= {summary[key]:.4f}")

    print("\n===== ADAPTER ROLLOUT DIAGNOSTICS =====")
    for key in ("adapter_delta_rms", "adapter_to_common_ratio", "observer_adapter_delta_rms", "observer_adapter_to_common_ratio", "drone_adapter_delta_rms", "drone_adapter_to_common_ratio"):
        print(f"{key:32s}= {summary[key]:.6f}")

    print(f"\noutput                          = {args.output_json}")


if __name__ == "__main__":
    main()
