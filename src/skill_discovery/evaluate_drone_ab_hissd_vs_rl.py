#!/usr/bin/env python3
"""Paired online rollout evaluation: HiSSD A/B vs RLlib expert PPO A/B.

Identical difficulty, environment settings and episode seeds; each method
runs on an independent freshly-reset environment, not on recorded trajectories.
Metric timestep = one full joint A/B action cycle (A1, B1, B2), including
partial final cycles if there is an early crash.

Install under src/skill_discovery/ along with the earlier collector and model.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Reuse the SAME RLlib checkpoint/environment construction and obs conversion
# as the expert offline-data collector, to avoid silent evaluation differences.
from skill_discovery import collect_offline_data_drone_ab_vanilla as collector
from skill_discovery.hissd_drone_ab_paper_models import load_paper_drone_ab_checkpoint

FIELDNAMES = [
    "policy", "episode", "seed", "goal_found", "goal_step",
    "coverage60_reached", "coverage60_step", "full_coverage60_reached",
    "full_coverage60_step", "final_coverage", "final_reward_coverage",
    "fatal_crash", "goal_and_coverage60_no_crash", "env_success",
    "episode_steps", "individual_actions", "max_cycles", "runtime_sec",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rl-checkpoint", type=Path, default=Path(
        "src/train/hetero_drone_checkpoint/checkpoint_08100"))
    p.add_argument("--hissd-checkpoint", type=Path, default=Path(
        "src/skill_discovery/checkpoints/hissd-drone-ab-repository-d1/hissd_drone_ab_paper_best.pt"))
    p.add_argument("--output-dir", type=Path, default=Path(
        "src/skill_discovery/eval_drone_ab_d1"))
    p.add_argument("--difficulty", type=int, default=1)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed-base", type=int, default=300_000_000,
                   help="Use evaluation seeds disjoint from offline training data (not 100000000).")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                   help="Device for HiSSD. RLlib expert loaded for CPU inference.")
    p.add_argument("--hissd-variant", choices=("best", "last"), default="best",
                   help="Informational only: checkpoint path is authoritative.")
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--no-progress", action="store_true",
                   help="Disable the paired-episode tqdm progress bar.")
    p.add_argument("--coverage-threshold", type=float, default=0.6)
    p.add_argument("--allow-training-seed-overlap", action="store_true",
                   help="Override guard against typical offline training seed bases.")
    p.add_argument("--no-plots", action="store_true", help="Skip optional matplotlib bar charts")
    return p.parse_args()


def _relative(path: Path):
    return path.expanduser().resolve() if path.is_absolute() else (ROOT / path).resolve()


def _policy_id(agent_id: str, drone_types: tuple[str, ...]) -> str:
    ix = int(agent_id.rsplit("_", 1)[-1])
    return f"drone_{drone_types[ix].lower()}_policy"


def _snapshots(env, role_ids):
    """All agent observations at the SAME cycle-start instant."""
    raw = {}
    encoded = {}
    for role, names in role_ids.items():
        observations = []
        for agent_id in names:
            obs = env.observe(agent_id)
            raw[agent_id] = obs
            observations.append(collector.convert_observation(obs, "drone"))
        encoded[role] = {
            key: np.stack([o[key] for o in observations], axis=0)
            for key in ("global_map", "local_map", "action_history")
        }
    return raw, encoded


def _predict_rl(algo, raw, role_ids, drone_types, env):
    result = {}
    for role in ("observer", "drone"):
        for agent_id in role_ids[role]:
            out = algo.compute_single_action(raw[agent_id],
                                             policy_id=_policy_id(agent_id, drone_types),
                                             explore=False)
            # Keep RL policy's exact action output, clipping only tiny numeric
            # excursions to conform to Gym Box bounds, as with the HiSSD actor.
            arr = np.asarray(out, dtype=np.float32).reshape(-1)
            space = env.action_space(agent_id)
            if arr.shape != space.shape:
                raise ValueError(f"RL {agent_id} output {arr.shape} != {space.shape}")
            result[agent_id] = np.clip(arr, space.low, space.high).astype(np.float32)
    return result


@torch.inference_mode()
def _predict_hissd(model, encoded, role_ids, recurrent_state, device, env):
    tensors = {}
    masks = {}
    for role in ("observer", "drone"):
        obs = encoded[role]
        count = len(role_ids[role])
        if count < 1:
            raise ValueError(f"Missing role {role} in A/B episode")
        speed = float(np.asarray(env.action_space(role_ids[role][0]).high)[0])
        if speed <= 0:
            raise ValueError(f"Invalid speed for {role}: {speed}")
        tensors[role] = {
            "global_map": torch.from_numpy(obs["global_map"][None]).to(device),
            "local_map": torch.from_numpy(obs["local_map"][None]).to(device),
            # Training dataset normalizes physical action histories to [-1,1].
            "action_history": torch.from_numpy(
                np.clip(obs["action_history"] / speed, -1.0, 1.0)[None]
            ).to(device),
        }
        masks[role] = torch.ones((1, count), dtype=torch.bool, device=device)
    outputs, next_state = model.inference_step(tensors, masks, recurrent_state)
    actions = {}
    for role in ("observer", "drone"):
        predicted = outputs["actions"][role].squeeze(0).detach().float().cpu().numpy()
        if len(predicted) != len(role_ids[role]):
            raise ValueError(f"{role} action count mismatch: {predicted.shape}")
        for ix, agent_id in enumerate(role_ids[role]):
            space = env.action_space(agent_id)
            if predicted[ix].shape != space.shape:
                raise ValueError(f"HiSSD {agent_id} output {predicted[ix].shape} != {space.shape}")
            # Shared tanh decoder yields normalized actions. A/B have different
            # Box physical limits, so this is the inverse of training scaling.
            high = np.asarray(space.high, dtype=np.float32)
            arr = np.clip(predicted[ix], -1.0, 1.0) * high
            if not np.isfinite(arr).all():
                raise FloatingPointError(f"Nonfinite HiSSD actions on {agent_id}")
            actions[agent_id] = np.clip(arr, space.low, space.high).astype(np.float32)
    return actions, next_state


def _get_core(env):
    return collector.get_core_env(env)


def run_episode(*, kind, seed, config, drone_types, algo, model, device, threshold, ep_number):
    """Return first-passage steps and terminal metrics; NEVER stop at goal/60%."""
    # Make any global RNG effects reproducible across paired rollouts.
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    # Rollout uses the raw PettingZoo AEC API, exactly like the dataset collector.
    # collector.env_creator() is the RLlib wrapper and has different step semantics.
    env = collector.HeMAC_v0.env(**config)
    start = time.perf_counter()
    try:
        env.reset(seed=seed)
        core = _get_core(env)
        order = tuple(env.possible_agents)
        if not order or any(not x.startswith("drone_") for x in order):
            raise ValueError(f"Expected drone-only A/B env, found: {order}")
        role_ids = {
            "observer": [x for x in order if drone_types[int(x.rsplit("_",1)[-1])] == "A"],
            "drone": [x for x in order if drone_types[int(x.rsplit("_",1)[-1])] == "B"],
        }
        n_agents = sum(map(len, role_ids.values()))
        if min(map(len, role_ids.values())) < 1 or n_agents != len(order):
            raise RuntimeError("Invalid A/B population")
        if kind == "hissd":
            assert model is not None
            if n_agents != int(model.reference_observer_count + model.reference_drone_count):
                raise ValueError("Population does not match trained HiSSD reference counts")
            state = model.initial_inference_state(batch_size=1, agent_count=n_agents,
                                                  device=device)
        else:
            state = None
        cycle = 0
        actions_taken = 0
        planned_actions = None
        # Some environments initialize exploration at reset; count as time 0.
        goal_step = 0 if bool(core.found_goal) else None
        coverage60_step = 0 if core.current_drone_reward_coverage_ratio() >= threshold else None
        full_coverage60_step = 0 if core.current_coverage_ratio() >= threshold else None

        for agent_id in env.agent_iter():
            _, _, terminated, truncated, _info = env.last()
            if terminated or truncated:
                env.step(None)
                continue
            if planned_actions is None:
                cycle += 1
                raw, encoded = _snapshots(env, role_ids)
                if kind == "rl":
                    planned_actions = _predict_rl(algo, raw, role_ids, drone_types, env)
                else:
                    planned_actions, state = _predict_hissd(
                        model, encoded, role_ids, state, device, env)
            if agent_id not in planned_actions:
                raise RuntimeError(f"Missing action for {agent_id}")
            action = planned_actions[agent_id]
            env.step(action)
            actions_taken += 1
            # Observe first crossing even if it happens in the middle of the cycle.
            if goal_step is None and core.found_goal:
                goal_step = cycle
            if coverage60_step is None and core.current_drone_reward_coverage_ratio() >= threshold:
                coverage60_step = cycle
            if full_coverage60_step is None and core.current_coverage_ratio() >= threshold:
                full_coverage60_step = cycle
            if agent_id == order[-1] or core.terminate or core.truncate:
                planned_actions = None
        full_cov = float(core.current_coverage_ratio())
        reward_cov = float(core.current_drone_reward_coverage_ratio())
        found_goal = bool(core.found_goal)
        crash = bool(core.collided)
        # Avoid confusing goal discovery (user's requested success rate)
        # with HeMAC's mission_success, which depends on timeout and no crash.
        return {
            "policy": kind, "episode": ep_number, "seed": seed,
            "goal_found": int(found_goal), "goal_step": goal_step,
            "coverage60_reached": int(coverage60_step is not None),
            "coverage60_step": coverage60_step,
            "full_coverage60_reached": int(full_coverage60_step is not None),
            "full_coverage60_step": full_coverage60_step,
            "final_coverage": full_cov, "final_reward_coverage": reward_cov,
            "fatal_crash": int(crash),
            "goal_and_coverage60_no_crash": int(found_goal and reward_cov >= threshold and not crash),
            "env_success": int(core.mission_success),
            "episode_steps": cycle, "individual_actions": actions_taken,
            "max_cycles": int(core.max_cycles),
            "runtime_sec": time.perf_counter() - start,
        }
    finally:
        env.close()


def _mean(xs):
    return float(np.mean(xs)) if xs else None


def _mean_or_none(rows, k, condition=None):
    return _mean([float(r[k]) for r in rows
                  if r[k] is not None and (condition is None or condition(r))])


def summarize(rows, threshold):
    out = {}
    for kind in ("rl", "hissd"):
        r = [x for x in rows if x["policy"] == kind]
        n = len(r)
        if not n:
            continue
        out[kind] = {
            "episodes": n,
            "goal_found_rate": _mean_or_none(r, "goal_found"),
            "goal_step_reached_mean": _mean_or_none(r, "goal_step"),
            "goal_step_reached_median": (
                float(np.median([x["goal_step"] for x in r if x["goal_step"] is not None]))
                if any(x["goal_step"] is not None for x in r) else None
            ),
            "final_coverage_mean": _mean_or_none(r, "final_coverage"),
            "final_reward_coverage_mean": _mean_or_none(r, "final_reward_coverage"),
            "coverage60_rate": _mean_or_none(r, "coverage60_reached"),
            "coverage60_step_reached_mean": _mean_or_none(r, "coverage60_step"),
            "full_coverage60_rate": _mean_or_none(r, "full_coverage60_reached"),
            "full_coverage60_step_reached_mean": _mean_or_none(r, "full_coverage60_step"),
            "fatal_crash_rate": _mean_or_none(r, "fatal_crash"),
            "goal_and_coverage60_no_crash_rate": _mean_or_none(r, "goal_and_coverage60_no_crash"),
            "env_success_rate": _mean_or_none(r, "env_success"),
            "episode_steps_mean": _mean_or_none(r, "episode_steps"),
            "runtime_sec_mean": _mean_or_none(r, "runtime_sec"),
        }
    # Pairwise comparisons, one row per seed per model. First-passage times
    # undefined when the event never occurred: do NOT fill with episode length.
    rl = {x["seed"]: x for x in rows if x["policy"] == "rl"}
    hi = {x["seed"]: x for x in rows if x["policy"] == "hissd"}
    common = sorted(set(rl) & set(hi))
    out["paired"] = {"seeds": len(common), "mean_hissd_minus_rl": {}}
    for key in ("goal_found", "final_coverage", "final_reward_coverage",
                "coverage60_reached", "fatal_crash", "goal_and_coverage60_no_crash"):
        out["paired"]["mean_hissd_minus_rl"][key] = _mean(
            [float(hi[s][key]) - float(rl[s][key]) for s in common])
    for key in ("goal_step", "coverage60_step", "full_coverage60_step"):
        both = [s for s in common if hi[s][key] is not None and rl[s][key] is not None]
        out["paired"]["mean_hissd_minus_rl"][key] = (
            _mean([hi[s][key] - rl[s][key] for s in both]))
        out["paired"][f"both_reached_{key}"] = len(both)
    out["metric_definitions"] = {
        "goal_found_rate": "Number of episodes where goal ever detected / N; crash after detection does not remove goal detection",
        "goal_step": "First goal detection joint timestep (one A1+B1+B2 cycle), null if not detected",
        "coverage60_step": f"First joint timestep reward-eligible coverage >= {threshold:.3f}, null if not reached",
        "full_coverage60_step": f"First joint timestep entire-search-area coverage >= {threshold:.3f}, null if not reached",
        "final_coverage": "Terminal core.current_coverage_ratio() over entire search area",
        "final_reward_coverage": "Terminal core.current_drone_reward_coverage_ratio(), excludes reward-ineligible base sectors",
        "missing_steps": "Time means condition on reaching the goal/coverage; null failures are excluded. Reach rates are always reported.",
        "paired": "Same environment and seed; method-specific trajectories with deterministic policy inference",
    }
    return out


def _write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_csv(path, summary):
    data = []
    for policy in ("rl", "hissd"):
        if policy in summary:
            data.append({"policy": policy, **summary[policy]})
    if data:
        columns = list(data[0])
        _write_csv(path, data, columns)


def maybe_plot(path, summary, no_plots):
    if no_plots:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[info] matplotlib missing; skipping plots", flush=True)
        return
    columns = [
        ("goal_found_rate", "Goal discovery rate", 100),
        ("goal_step_reached_mean", "Steps to goal (when found)", 1),
        ("final_coverage_mean", "Final full-map coverage", 100),
        ("coverage60_step_reached_mean", "Steps to 60% reward coverage (when reached)", 1),
    ]
    for key, title, factor in columns:
        values = [summary.get(k,{}).get(key) for k in ("rl", "hissd")]
        if any(x is None for x in values):
            continue
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.bar(["RL expert", "HiSSD"], [x * factor for x in values])
        ax.set_title(title)
        ax.set_ylabel("%" if factor==100 else "Joint timesteps")
        fig.tight_layout()
        fig.savefig(path / f"{key}.png", dpi=160)
        plt.close(fig)


def main():
    a = parse_args()
    if a.episodes <= 0 or a.print_every <= 0 or not (0.0 < a.coverage_threshold <= 1.0):
        raise ValueError(
            "episodes and print-every must be positive, and coverage threshold "
            "must be in (0,1]"
        )
    if a.seed_base in (0, 100_000_000, 200_000_000) and not a.allow_training_seed_overlap:
        raise ValueError("Choose an evaluation-only seed-base, e.g. 300000000")
    device = torch.device("cuda" if (a.device=="auto" and torch.cuda.is_available()) else
                          "cpu" if a.device=="auto" else a.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    out = _relative(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rl_path = _relative(a.rl_checkpoint)
    hi_path = _relative(a.hissd_checkpoint)
    if not hi_path.is_file():
        raise FileNotFoundError(hi_path)
    collector.ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=1, log_to_driver=False)
    algo = None
    try:
        collector.register_hemac_rllib_models()
        collector.register_env(collector.ENV_NAME, collector.env_creator)
        algo = collector.load_inference_algorithm(rl_path)
        env_cfg = collector.build_collection_env_config(
            getattr(algo.config, "env_config", {}) or {}, a.difficulty)
        if env_cfg.get("n_observers") != 0:
            raise ValueError("Expert checkpoint env must be drone-only")
        if env_cfg.get("drone_only_task_mode") != "maximize_coverage":
            raise ValueError("Need maximize_coverage to evaluate full-length exploration")
        types = tuple(str(t).upper() for t in env_cfg["drone_config"]["drone_types"])
        if len(types) != int(env_cfg["n_drones"]) or not ({"A", "B"} <= set(types)):
            raise ValueError(f"Unexpected A/B population: {types}")
        for p in ("drone_a_policy", "drone_b_policy"):
            if algo.get_policy(p) is None:
                raise RuntimeError(f"RLlib checkpoint lacks policy {p}")
        model, hi_payload = load_paper_drone_ab_checkpoint(hi_path, device=device)
        model.eval()
        expected_a = int(model.reference_observer_count)
        expected_b = int(model.reference_drone_count)
        if types.count("A") != expected_a or types.count("B") != expected_b:
            raise ValueError(f"HiSSD trained on A{expected_a}B{expected_b}, RL env {types}")
        split = hi_payload.get("split", {})
        trained_scales = split.get("action_scales", None)
        config_scales = tuple(float(env_cfg["drone_config"].get("drone_type_configs", {}).get(t,{}).get(
            "max_speed",env_cfg["drone_config"].get("drone_max_speed",16))) for t in ("A", "B"))
        if trained_scales is not None and not np.allclose(trained_scales, config_scales):
            raise ValueError(f"HiSSD training action scales {trained_scales} != evaluation {config_scales}")
        print(f"[EVAL] A/B={types} difficulty={a.difficulty} max_cycles={env_cfg['max_cycles']}", flush=True)
        print(f"[EVAL] RL={rl_path}", flush=True)
        print(f"[EVAL] HiSSD={hi_path} trained_epoch={hi_payload.get('epoch')} device={device}", flush=True)
        print(f"[EVAL] {a.episodes} paired seeds={a.seed_base}..{a.seed_base+a.episodes-1}", flush=True)
        print("[EVAL] No early stop at goal or coverage 60%; episodes continue to crash/time limit", flush=True)
        rows = []
        episode_iterator = range(a.episodes)
        progress = None
        if tqdm is not None and not a.no_progress:
            progress = tqdm(
                episode_iterator,
                total=a.episodes,
                desc=f"Paired RL vs HiSSD D{a.difficulty}",
                unit="pair",
                dynamic_ncols=True,
                smoothing=0.05,
            )
            episode_iterator = progress
        for idx in episode_iterator:
            seed = a.seed_base + idx
            latest = {}
            for kind in ("rl", "hissd"):
                row = run_episode(kind=kind,seed=seed,config=env_cfg,
                                  drone_types=types,algo=algo,model=model,
                                  device=device,threshold=a.coverage_threshold,
                                  ep_number=idx)
                rows.append(row)
                latest[kind] = row
            # Save as we go, retaining partial results if the evaluation is interrupted.
            _write_csv(out / "episodes.csv", rows, FIELDNAMES)
            if progress is not None:
                progress.set_postfix(
                    rl_goal=latest["rl"]["goal_found"],
                    hi_goal=latest["hissd"]["goal_found"],
                    rl_cov=f"{latest['rl']['final_coverage']:.3f}",
                    hi_cov=f"{latest['hissd']['final_coverage']:.3f}",
                    refresh=False,
                )
            if (idx+1)%a.print_every == 0 or idx==0 or idx+1==a.episodes:
                s=summarize(rows,a.coverage_threshold)
                status = (f"[{idx+1:4d}/{a.episodes}] "
                          f"goal RL={s['rl']['goal_found_rate']:.3f} HiSSD={s['hissd']['goal_found_rate']:.3f} | "
                          f"coverage RL={s['rl']['final_coverage_mean']:.3f} HiSSD={s['hissd']['final_coverage_mean']:.3f} | "
                          f"reached60 RL={s['rl']['coverage60_rate']:.3f} HiSSD={s['hissd']['coverage60_rate']:.3f}")
                if progress is not None:
                    tqdm.write(status)
                else:
                    print(status, flush=True)
        summary=summarize(rows,a.coverage_threshold)
        (out / "summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False))
        _write_summary_csv(out / "summary.csv",summary)
        (out / "evaluation_config.json").write_text(json.dumps({
            "rl_checkpoint":str(rl_path),"hissd_checkpoint":str(hi_path),
            "hissd_epoch":hi_payload.get("epoch"),"difficulty":a.difficulty,
            "episodes":a.episodes,"seed_base":a.seed_base,
            "seed_formula":"seed_base + episode_index", "drone_types":types,
            "action_scales":config_scales,"max_cycles":env_cfg["max_cycles"],
            "coverage_threshold":a.coverage_threshold,
            "coverage60_definition":"drone_reward_coverage_ratio",
            "same_environment_config":True,
            "deterministic_inference":True,
        },indent=2,ensure_ascii=False))
        maybe_plot(out,summary,a.no_plots)
        print("\n==== FINAL PAIRED EVALUATION ====")
        for p in ("rl", "hissd"):
            s=summary[p]
            print(f"{p:5s} goal={100*s['goal_found_rate']:.2f}% "
                  f"goal_time={s['goal_step_reached_mean']} "
                  f"final_coverage={100*s['final_coverage_mean']:.2f}% "
                  f"reward_coverage={100*s['final_reward_coverage_mean']:.2f}% "
                  f"coverage60_rate={100*s['coverage60_rate']:.2f}% "
                  f"coverage60_step={s['coverage60_step_reached_mean']} "
                  f"crash={100*s['fatal_crash_rate']:.2f}%")
        print(f"Saved: {out}",flush=True)
    finally:
        if algo is not None:
            algo.stop()
        collector.ray.shutdown()

if __name__ == "__main__":
    main()
