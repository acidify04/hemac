"""Pure read-only evaluation of an offline HeMAC HiSSD checkpoint.

Policy path:
    observations
      -> model.inference_step(...)
      -> outputs["actions"]
      -> environment action scaling
      -> HeMAC

No online residual policy, skill gate, PPO update, critic, or fine-tuning.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
from skill_discovery.collect_offline_data import ENV_NAME, env_creator

from skill_discovery.hissd_models import HeMACHISSD
import skill_discovery.finetune_hissd_drone_online as online


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hissd-checkpoint", type=Path, required=True)
    parser.add_argument("--mappo-checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed-base", type=int, default=100_000_000)
    parser.add_argument("--success-min-coverage-ratio", type=float, default=0.6)
    parser.add_argument(
        "--difficulties",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "src/skill_discovery/outputs/"
            "zero_shot_B_pure/pure_hissd_d1_d4.json"
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cuda",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm difficulty and episode progress bars.",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return torch.device(name)


def load_hissd(checkpoint_path: Path, device: torch.device):
    payload = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if "model_config" not in payload:
        raise KeyError("HiSSD checkpoint has no model_config.")
    if "model_state_dict" not in payload:
        raise KeyError("HiSSD checkpoint has no model_state_dict.")

    # Keep only actual HeMACHISSD constructor arguments.
    signature = inspect.signature(HeMACHISSD.__init__)
    allowed = {
        name
        for name in signature.parameters
        if name != "self"
    }

    model_config = {
        key: value
        for key, value in payload["model_config"].items()
        if key in allowed
    }

    model = HeMACHISSD(**model_config).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()

    return model, payload


def load_env_config(checkpoint_path: Path):
    """Read env_config from an RLlib checkpoint without restoring the Algorithm."""
    from ray import cloudpickle

    checkpoint_path = checkpoint_path.expanduser().resolve()

    candidates = [
        checkpoint_path / "algorithm_state.pkl",
        checkpoint_path / "checkpoint.pkl",
    ]

    # Fallback for slightly different RLlib checkpoint layouts.
    candidates.extend(
        path
        for path in checkpoint_path.glob("*.pkl")
        if path not in candidates
    )

    state_path = next(
        (path for path in candidates if path.is_file()),
        None,
    )

    if state_path is None:
        raise FileNotFoundError(
            "Could not find RLlib algorithm state pickle in "
            f"{checkpoint_path}. Files: "
            f"{[p.name for p in checkpoint_path.iterdir()]}"
        )

    print(f"Reading env_config only from: {state_path}")

    with state_path.open("rb") as file:
        state = cloudpickle.load(file)

    if not isinstance(state, dict):
        raise TypeError(
            f"Unexpected RLlib state type: {type(state)}"
        )

    config = state.get("config")

    # Some checkpoint layouts may nest the algorithm state.
    if config is None:
        nested = state.get("algorithm_state")
        if isinstance(nested, dict):
            config = nested.get("config")

    if config is None:
        raise KeyError(
            "RLlib checkpoint state contains no config. "
            f"Top-level keys: {list(state.keys())}"
        )

    if hasattr(config, "env_config"):
        env_config = dict(config.env_config)

    elif isinstance(config, dict):
        env_config = dict(config.get("env_config", {}))

    elif hasattr(config, "to_dict"):
        env_config = dict(
            config.to_dict().get("env_config", {})
        )

    else:
        raise TypeError(
            f"Unsupported RLlib config type: {type(config)}"
        )

    if not env_config:
        raise RuntimeError(
            "Checkpoint config contains no env_config."
        )

    print(
        "Loaded env_config without restoring MAPPO:",
        sorted(env_config.keys()),
    )

    return env_config

def run_episode(
    *,
    model,
    checkpoint_env_config,
    difficulty: int,
    success_min_coverage_ratio: float,
    seed: int,
    device: torch.device,
):
    config = online.build_env_config(
        checkpoint_env_config,
        difficulty,
        success_min_coverage_ratio,
    )
    action_scale = online.drone_action_scale(config)

    env = online.HeMAC_v0.env(**config)

    try:
        env.reset(seed=seed)
        core_env = online.get_core_env(env)

        agent_order = list(env.possible_agents)
        drone_ids = [
            agent_id
            for agent_id in agent_order
            if agent_id.startswith("drone_")
        ]

        if len(drone_ids) != len(agent_order):
            raise RuntimeError(
                "Pure HiSSD evaluator expects a drone-only HeMAC environment."
            )

        valid_mask = torch.ones(
            1,
            len(drone_ids),
            dtype=torch.bool,
            device=device,
        )

        state = model.initial_inference_state(
            batch_size=1,
            device=device,
        )

        # Actions for all drones are computed from the SAME
        # cycle-start observation, matching offline collection semantics.
        cached_actions: dict[str, np.ndarray] = {}

        cycle_count = 0
        last_agent_id = agent_order[-1]

        for agent_id in env.agent_iter():
            _, _, termination, truncation, _ = env.last()

            if termination or truncation:
                env.step(None)
                continue

            if not cached_actions:
                observations = online.drone_observation_batch(
                    env,
                    drone_ids,
                    action_scale,
                    device,
                )

                outputs, state = model.inference_step(
                    observations,
                    valid_mask,
                    state,
                )

                # PURE OFFLINE HISSD POLICY.
                # No OnlineNoSkillResidualPolicy.
                # No skill_prior gate.
                # No PPO residual.
                normalized_actions = outputs["actions"][0]

                if normalized_actions.shape[0] != len(drone_ids):
                    raise RuntimeError(
                        "HiSSD action/agent count mismatch: "
                        f"{tuple(normalized_actions.shape)} vs "
                        f"{len(drone_ids)} drones"
                    )

                # outputs["actions"] is tanh-normalized.
                # This is only environment-unit conversion.
                env_actions = normalized_actions * action_scale

                for index, drone_id in enumerate(drone_ids):
                    action_space = env.action_space(drone_id)
                    action = env_actions[index].detach().cpu().numpy()

                    cached_actions[drone_id] = np.ascontiguousarray(
                        np.clip(
                            action,
                            action_space.low,
                            action_space.high,
                        ),
                        dtype=np.float32,
                    )

            env.step(cached_actions.pop(agent_id))

            cycle_completed = (
                agent_id == last_agent_id
                or bool(core_env.terminate)
                or bool(core_env.truncate)
            )

            if cycle_completed:
                cycle_count += 1
                cached_actions.clear()

        goal_found = any(
            online.agent_found_goal(core_env, drone_id)
            for drone_id in drone_ids
        )

        return {
            "seed": int(seed),
            "difficulty": int(difficulty),
            "success": float(bool(core_env.mission_success)),
            "goal_found": float(goal_found),
            "fatal_crash": float(bool(core_env.collided)),
            "drone_crash": float(bool(core_env.drone_crash)),
            "coverage": float(core_env.current_coverage_ratio()),
            "reward_coverage": float(
                core_env.current_drone_reward_coverage_ratio()
            ),
            "cycles": float(cycle_count),
        }

    finally:
        env.close()


def summarize(records):
    names = (
        "success",
        "goal_found",
        "fatal_crash",
        "drone_crash",
        "coverage",
        "reward_coverage",
        "cycles",
    )

    return {
        name: float(np.mean([row[name] for row in records]))
        for name in names
    }


def main():
    args = parse_args()

    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")

    device = resolve_device(args.device)

    model, checkpoint_payload = load_hissd(
        args.hissd_checkpoint,
        device,
    )

    print(
        "Loaded pure HiSSD:",
        args.hissd_checkpoint,
    )
    print(
        "model_config:",
        "skill_structure=",
        checkpoint_payload["model_config"].get("skill_structure"),
        "contrastive_from_action_skill=",
        checkpoint_payload["model_config"].get(
            "contrastive_from_action_skill"
        ),
        "task_context_pooling=",
        checkpoint_payload["model_config"].get(
            "task_context_pooling"
        ),
    )

    checkpoint_env_config = load_env_config(
        args.mappo_checkpoint
    )

    result = {
        "checkpoint": str(args.hissd_checkpoint.resolve()),
        "environment_checkpoint": str(
            args.mappo_checkpoint.resolve()
        ),
        "policy": 'model.inference_step -> outputs["actions"]',
        "online_residual": False,
        "skill_gate": False,
        "ppo_updates": 0,
        "deterministic": True,
        "episodes_per_difficulty": args.episodes,
        "seed_base": args.seed_base,
        "difficulties": {},
    }

    print()
    print(
        "Difficulty  Split   Success   Goal    Crash   "
        "Coverage  RewardCov"
    )
    print("-" * 70)

    difficulty_progress = tqdm(
        args.difficulties,
        desc="pure HiSSD difficulties",
        unit="difficulty",
        position=0,
        dynamic_ncols=True,
        disable=args.no_progress,
    )

    for difficulty in difficulty_progress:
        records = []

        difficulty_progress.set_postfix(difficulty=f"D{difficulty}")

        progress = tqdm(
            range(args.episodes),
            desc=f"pure HiSSD D{difficulty}",
            unit="episode",
            position=1,
            leave=False,
            dynamic_ncols=True,
            disable=args.no_progress,
        )

        successes = 0.0
        goals = 0.0
        crashes = 0.0
        coverage_sum = 0.0

        for episode_index in progress:
            # Same seed convention used by the online evaluator,
            # allowing paired comparison.
            seed = (
                args.seed_base
                + difficulty * 100_000
                + episode_index
            )

            record = run_episode(
                model=model,
                checkpoint_env_config=checkpoint_env_config,
                difficulty=difficulty,
                success_min_coverage_ratio=(
                    args.success_min_coverage_ratio
                ),
                seed=seed,
                device=device,
            )

            records.append(record)
            successes += record["success"]
            goals += record["goal_found"]
            crashes += record["fatal_crash"]
            coverage_sum += record["coverage"]

            done = episode_index + 1
            progress.set_postfix(
                success=f"{successes / done:.3f}",
                goal=f"{goals / done:.3f}",
                crash=f"{crashes / done:.3f}",
                cov=f"{coverage_sum / done:.3f}",
            )

        progress.close()

        summary = summarize(records)
        split = "source" if difficulty in (1, 2) else "target"

        result["difficulties"][f"D{difficulty}"] = {
            "split": split,
            "summary": summary,
            "episodes": records,
        }

        tqdm.write(
            f"D{difficulty:<10}"
            f"{split:<8}"
            f"{summary['success']:<10.3f}"
            f"{summary['goal_found']:<8.3f}"
            f"{summary['fatal_crash']:<8.3f}"
            f"{summary['coverage']:<10.3f}"
            f"{summary['reward_coverage']:.3f}"
        )

    difficulty_progress.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
