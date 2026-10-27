#!/usr/bin/env python3
"""Same-seed HeMAC D1 online rollout: A/B RLlib expert vs dynamic frontier A*.

Metrics:
  - goal discovery rate and FIRST goal detection joint timestep
  - end-of-episode full map and reward-eligible coverage
  - first time to 60% coverage, and probability of achieving 60%
  - crash rate, no-crash goal discovery, runtime

Never uses true goal location, true obstacle coordinates, or future enemy state.
A* reads HeMAC shared observed coverage and obstacle/warning belief maps, plus
self/teammate pose/velocity; these are explicit classical-baseline assumptions.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skill_discovery import collect_offline_data_drone_ab_vanilla as collector
from skill_discovery.drone_ab_dynamic_astar import AStarConfig, DynamicFrontierAStar

FIELDNAMES = [
    'policy', 'episode', 'seed', 'goal_found', 'goal_step',
    'coverage60_reached', 'coverage60_step', 'full_coverage60_reached',
    'full_coverage60_step', 'final_coverage', 'final_reward_coverage',
    'fatal_crash', 'goal_and_coverage60_no_crash', 'env_success',
    'episode_steps', 'individual_actions', 'max_cycles', 'runtime_sec',
    'runtime_ms_per_step', 'decision_sec', 'decision_ms_per_step',
    'env_step_sec', 'env_step_ms_per_step',
    'astar_replans', 'astar_failed_plans',
]
MEASURES = [
    'goal_found_rate', 'goal_step_reached_mean', 'goal_step_reached_median',
    'goal_step_penalized_mean', 'final_coverage_mean', 'final_reward_coverage_mean',
    'coverage60_rate', 'coverage60_step_reached_mean',
    'coverage60_step_penalized_mean', 'full_coverage60_rate',
    'full_coverage60_step_reached_mean', 'fatal_crash_rate',
    'goal_and_coverage60_no_crash_rate', 'env_success_rate',
    'episode_steps_mean', 'runtime_sec_mean', 'runtime_sec_total',
    'episode_steps_total', 'runtime_ms_per_step',
    'decision_sec_total', 'decision_ms_per_step',
    'env_step_sec_total', 'env_step_ms_per_step', 'joint_steps_per_second',
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rl-checkpoint', type=Path, default=Path('src/train/hetero_drone_checkpoint/checkpoint_08100'))
    p.add_argument('--output-dir', type=Path, default=Path('src/skill_discovery/eval_drone_ab_astar_vs_rl_d1'))
    p.add_argument('--difficulty', type=int, default=1)
    p.add_argument('--episodes', type=int, default=200)
    p.add_argument('--seed-base', type=int, default=300000000)
    p.add_argument('--coverage-threshold', type=float, default=.6)
    p.add_argument('--print-every', type=int, default=10)
    p.add_argument('--no-progress', action='store_true',
                   help='Disable the paired-episode tqdm progress bar.')
    p.add_argument('--inflation-cells', type=int, default=0,
                   help='Extra cells added around detected warning zones; original warning map already includes hazard radius.')
    p.add_argument('--obstacle-threshold', type=float, default=.35)
    p.add_argument('--warning-threshold', type=float, default=.25)
    p.add_argument('--max-candidates', type=int, default=6)
    p.add_argument('--separation', type=float, default=68.0)
    p.add_argument('--no-plots', action='store_true')
    p.add_argument('--sync-cuda-timing', action='store_true',
                   help='Synchronize CUDA before and after RL inference when available (more precise GPU timing).')
    return p.parse_args()


def _relative(path):
    return path.expanduser().resolve() if path.is_absolute() else (ROOT / path).resolve()


def _predict_rl(algo, env, drone_types, order):
    outputs = {}
    for agent_name in order:
        drone_idx = int(agent_name.rsplit('_', 1)[-1])
        role = drone_types[drone_idx].lower()
        action = algo.compute_single_action(env.observe(agent_name),
                                            policy_id=f'drone_{role}_policy', explore=False)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        box = env.action_space(agent_name)
        if action.shape != box.shape or not np.isfinite(action).all():
            raise ValueError(f'RL action invalid {agent_name}: shape={action.shape}')
        outputs[agent_name] = np.clip(action, box.low, box.high).astype(np.float32)
    return outputs


def rollout(*, kind, episode, seed, env_config, drone_types, algo, cfg, threshold, sync_cuda=False):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    # Recreate environment per method with SAME seed; policy trajectories diverge.
    env = collector.HeMAC_v0.env(**env_config)
    start_time = time.perf_counter()
    planner = DynamicFrontierAStar(cfg) if kind == 'astar' else None
    try:
        env.reset(seed=seed)
        core = collector.get_core_env(env)
        order = tuple(env.possible_agents)
        if not order or len(order) != len(drone_types) or any(not n.startswith('drone_') for n in order):
            raise RuntimeError(f'Expected exactly A/B drones; got {order}')
        cycle = 0
        actions_taken = 0
        decision_sec = 0.0
        env_step_sec = 0.0
        planned_actions = None
        goal_step = 0 if bool(core.found_goal) else None
        reward60 = 0 if core.current_drone_reward_coverage_ratio() >= threshold else None
        full60 = 0 if core.current_coverage_ratio() >= threshold else None
        for name in env.agent_iter():
            _obs, _reward, terminated, truncated, _info = env.last()
            if terminated or truncated:
                env.step(None)
                continue
            if planned_actions is None:
                cycle += 1
                # A single decision plans all three drones for this joint timestep.
                cuda_sync = bool(sync_cuda and kind == 'rl' and collector.torch.cuda.is_available())
                if cuda_sync:
                    collector.torch.cuda.synchronize()
                decision_start = time.perf_counter()
                if kind == 'rl':
                    planned_actions = _predict_rl(algo, env, drone_types, order)
                else:
                    planned_actions = planner.act(core, order)
                if cuda_sync:
                    collector.torch.cuda.synchronize()
                decision_sec += time.perf_counter() - decision_start
                for n, action in list(planned_actions.items()):
                    box = env.action_space(n)
                    if action.shape != box.shape or not np.isfinite(action).all():
                        raise ValueError(f'Invalid {kind} action for {n}: {action}')
                    planned_actions[n] = np.clip(action, box.low, box.high).astype(np.float32)
            env_start = time.perf_counter()
            env.step(planned_actions[name])
            env_step_sec += time.perf_counter() - env_start
            actions_taken += 1
            # Record first detection immediately, including mid-cycle crash.
            if goal_step is None and core.found_goal:
                goal_step = cycle
            if reward60 is None and core.current_drone_reward_coverage_ratio() >= threshold:
                reward60 = cycle
            if full60 is None and core.current_coverage_ratio() >= threshold:
                full60 = cycle
            if name == order[-1] or core.terminate or core.truncate:
                planned_actions = None
        full_cov = float(core.current_coverage_ratio())
        reward_cov = float(core.current_drone_reward_coverage_ratio())
        found = bool(core.found_goal)
        crashed = bool(core.collided)
        runtime_sec = time.perf_counter() - start_time
        # Cycle is a JOINT timestep: one action per living drone, not per-agent AEC step.
        denom = max(cycle, 1)
        return {
            'policy': kind, 'episode': episode, 'seed': seed,
            'goal_found': int(found), 'goal_step': goal_step,
            'coverage60_reached': int(reward60 is not None), 'coverage60_step': reward60,
            'full_coverage60_reached': int(full60 is not None), 'full_coverage60_step': full60,
            'final_coverage': full_cov, 'final_reward_coverage': reward_cov,
            'fatal_crash': int(crashed),
            'goal_and_coverage60_no_crash': int(found and reward_cov >= threshold and not crashed),
            'env_success': int(core.mission_success), 'episode_steps': cycle,
            'individual_actions': actions_taken, 'max_cycles': int(core.max_cycles),
            'runtime_sec': runtime_sec,
            'runtime_ms_per_step': runtime_sec * 1000.0 / denom if cycle else None,
            'decision_sec': decision_sec,
            'decision_ms_per_step': decision_sec * 1000.0 / denom if cycle else None,
            'env_step_sec': env_step_sec,
            'env_step_ms_per_step': env_step_sec * 1000.0 / denom if cycle else None,
            'astar_replans': planner.replans if planner else 0,
            'astar_failed_plans': planner.failures if planner else 0,
        }
    finally:
        env.close()


def mean(values):
    return float(np.mean(values)) if values else None


def summarize(rows):
    out = {}
    for kind in ('rl', 'astar'):
        r = [d for d in rows if d['policy'] == kind]
        if not r:
            continue
        def numeric(k):
            return [float(x[k]) for x in r if x[k] is not None]
        goal_steps = numeric('goal_step')
        c60_steps = numeric('coverage60_step')
        full60_steps = numeric('full_coverage60_step')
        total_steps = sum(int(x['episode_steps']) for x in r)
        total_runtime = sum(float(x['runtime_sec']) for x in r)
        total_decision = sum(float(x.get('decision_sec', 0.0)) for x in r)
        total_env_step = sum(float(x.get('env_step_sec', 0.0)) for x in r)
        out[kind] = {
            'episodes': len(r),
            'goal_found_rate': mean(numeric('goal_found')),
            'goal_step_reached_mean': mean(goal_steps),
            'goal_step_reached_median': float(np.median(goal_steps)) if goal_steps else None,
            'goal_step_penalized_mean': mean([float(x['goal_step']) if x['goal_step'] is not None else float(x['max_cycles']) + 1 for x in r]),
            'final_coverage_mean': mean(numeric('final_coverage')),
            'final_reward_coverage_mean': mean(numeric('final_reward_coverage')),
            'coverage60_rate': mean(numeric('coverage60_reached')),
            'coverage60_step_reached_mean': mean(c60_steps),
            'coverage60_step_penalized_mean': mean([float(x['coverage60_step']) if x['coverage60_step'] is not None else float(x['max_cycles']) + 1 for x in r]),
            'full_coverage60_rate': mean(numeric('full_coverage60_reached')),
            'full_coverage60_step_reached_mean': mean(full60_steps),
            'fatal_crash_rate': mean(numeric('fatal_crash')),
            'goal_and_coverage60_no_crash_rate': mean(numeric('goal_and_coverage60_no_crash')),
            'env_success_rate': mean(numeric('env_success')),
            'episode_steps_mean': mean(numeric('episode_steps')),
            'runtime_sec_mean': mean(numeric('runtime_sec')),
            'runtime_sec_total': total_runtime,
            'episode_steps_total': total_steps,
            'runtime_ms_per_step': 1000.0 * total_runtime / total_steps if total_steps else None,
            'decision_sec_total': total_decision,
            'decision_ms_per_step': 1000.0 * total_decision / total_steps if total_steps else None,
            'env_step_sec_total': total_env_step,
            'env_step_ms_per_step': 1000.0 * total_env_step / total_steps if total_steps else None,
            'joint_steps_per_second': total_steps / total_runtime if total_runtime > 0 else None,
            'astar_replans_mean': mean(numeric('astar_replans')),
            'astar_failed_plans_mean': mean(numeric('astar_failed_plans')),
        }
    if set(out) == {'rl', 'astar'}:
        x = {r['seed']: r for r in rows if r['policy'] == 'rl'}
        y = {r['seed']: r for r in rows if r['policy'] == 'astar'}
        both = sorted(set(x) & set(y))
        paired = {}
        for metric in ('goal_found', 'final_coverage', 'final_reward_coverage',
                       'coverage60_reached', 'fatal_crash', 'goal_and_coverage60_no_crash'):
            paired[metric] = mean([float(y[s][metric]) - float(x[s][metric]) for s in both])
        for metric in ('runtime_ms_per_step', 'decision_ms_per_step'):
            valid = [s for s in both if x[s].get(metric) is not None and y[s].get(metric) is not None]
            paired[metric] = {'mean_astar_minus_rl_paired': mean([float(y[s][metric]) - float(x[s][metric]) for s in valid]),
                              'paired_count': len(valid)}
        for metric in ('goal_step', 'coverage60_step', 'full_coverage60_step'):
            valid = [s for s in both if x[s][metric] is not None and y[s][metric] is not None]
            paired[metric] = {'mean_astar_minus_rl_both_reached': mean([float(y[s][metric]) - float(x[s][metric]) for s in valid]),
                              'both_reached_count': len(valid)}
        out['paired'] = {'seeds': len(both), 'astar_minus_rl': paired}
    return out


def save_csv(path, rows, fieldnames):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plots(output_dir, summary):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    for name, title, scale in (
        ('goal_found_rate', 'Goal found (%)', 100),
        ('goal_step_reached_mean', 'First goal timestep | success only', 1),
        ('final_coverage_mean', 'Final full-map coverage (%)', 100),
        ('coverage60_rate', 'Reached 60% reward coverage (%)', 100),
        ('coverage60_step_reached_mean', 'First 60% coverage timestep | reached only', 1),
        ('fatal_crash_rate', 'Collision rate (%)', 100),
        ('runtime_ms_per_step', 'Episode elapsed time per joint step (ms)', 1),
        ('decision_ms_per_step', 'Policy decision time per joint step (ms)', 1),
    ):
        values = [summary[kind].get(name) for kind in ('rl', 'astar')]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(('RL', 'A*'), [0 if v is None else v*scale for v in values])
        for i, v in enumerate(values):
            ax.text(i, (v or 0)*scale, 'N/A' if v is None else f'{v*scale:.2f}', ha='center', va='bottom')
        ax.set_ylabel(title)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(output_dir / (name + '.png'), dpi=140)
        plt.close(fig)


def _fmt(value, factor=1.0):
    return 'N/A' if value is None else f'{factor*value:.2f}'


def main():
    args = parse_args()
    if args.episodes < 1 or args.max_candidates < 1 or args.print_every < 1 or not 0 < args.coverage_threshold <= 1:
        raise ValueError('episodes/max_candidates positive; coverage-threshold in (0,1]')
    if args.seed_base in (0, 100000000, 200000000):
        raise ValueError('Use a held-out seed base, e.g. 300000000')
    checkpoint = _relative(args.rl_checkpoint)
    output_dir = _relative(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = AStarConfig(inflation_cells=args.inflation_cells,
                      obstacle_threshold=args.obstacle_threshold,
                      warning_threshold=args.warning_threshold,
                      max_candidates=args.max_candidates,
                      separation=args.separation)
    collector.ray.init(ignore_reinit_error=True, include_dashboard=False,
                       num_cpus=1, log_to_driver=False)
    algo = None
    try:
        collector.register_hemac_rllib_models()
        collector.register_env(collector.ENV_NAME, collector.env_creator)
        algo = collector.load_inference_algorithm(checkpoint)
        env_cfg = collector.build_collection_env_config(
            getattr(algo.config, 'env_config', {}) or {}, args.difficulty)
        if env_cfg.get('n_observers') != 0 or env_cfg.get('drone_only_task_mode') != 'maximize_coverage':
            raise ValueError('Checkpoint task must be drone-only maximize_coverage')
        types = tuple(str(t).upper() for t in env_cfg['drone_config']['drone_types'])
        if len(types) != int(env_cfg['n_drones']) or not {'A', 'B'} <= set(types):
            raise ValueError(f'Unexpected A/B population: {types}')
        for pid in ('drone_a_policy', 'drone_b_policy'):
            if algo.get_policy(pid) is None:
                raise RuntimeError(f'Missing RL policy: {pid}')
        print(f'[EVAL] RL vs online dynamic frontier A*; D{args.difficulty}; types={types}; '
              f'max_cycles={env_cfg["max_cycles"]}; episodes={args.episodes}; seed_base={args.seed_base}', flush=True)
        print('[A*] Replan EVERY cycle with shared observed coverage & obstacle/warning beliefs; '
              'no hidden goal and no real obstacle trajectories.', flush=True)
        print('[METRIC] reached60=FRACTION, step60=MEAN JOINT TIMESTEPS conditional on reaching.', flush=True)
        print('[TIMING] runtime_ms_per_step = SUM(episode wall seconds) / SUM(joint episode steps) * 1000; '
              'decision_ms_per_step = SUM(RL inference or A* planning seconds) / SUM(joint steps) * 1000.', flush=True)
        rows = []
        episode_iterator = range(args.episodes)
        progress = None
        if tqdm is not None and not args.no_progress:
            progress = tqdm(
                episode_iterator,
                total=args.episodes,
                desc=f'Paired RL vs A* D{args.difficulty}',
                unit='pair',
                dynamic_ncols=True,
                smoothing=0.05,
            )
            episode_iterator = progress
        for i in episode_iterator:
            seed = args.seed_base + i
            latest = {}
            for kind in ('rl', 'astar'):
                row = rollout(kind=kind, episode=i, seed=seed, env_config=env_cfg,
                              drone_types=types, algo=algo, cfg=cfg,
                              threshold=args.coverage_threshold,
                              sync_cuda=args.sync_cuda_timing)
                rows.append(row)
                latest[kind] = row
            save_csv(output_dir/'episodes.csv', rows, FIELDNAMES)
            if progress is not None:
                progress.set_postfix(
                    rl_goal=latest['rl']['goal_found'],
                    astar_goal=latest['astar']['goal_found'],
                    rl_cov=f"{latest['rl']['final_coverage']:.3f}",
                    astar_cov=f"{latest['astar']['final_coverage']:.3f}",
                    refresh=False,
                )
            if (i+1) % args.print_every == 0 or i == 0 or i+1 == args.episodes:
                summary = summarize(rows)
                rl, ab = summary['rl'], summary['astar']
                status = (f'[{i+1:4d}/{args.episodes}] '
                          f'goal: RL={_fmt(rl["goal_found_rate"],100)}% A*={_fmt(ab["goal_found_rate"],100)}% | '
                          f'goal_steps: RL={_fmt(rl["goal_step_reached_mean"])} A*={_fmt(ab["goal_step_reached_mean"])} | '
                          f'coverage: RL={_fmt(rl["final_coverage_mean"],100)}% A*={_fmt(ab["final_coverage_mean"],100)}% | '
                          f'reached60: RL={_fmt(rl["coverage60_rate"],100)}% A*={_fmt(ab["coverage60_rate"],100)}% | '
                          f'step60: RL={_fmt(rl["coverage60_step_reached_mean"])} A*={_fmt(ab["coverage60_step_reached_mean"])} | '
                          f'wall_ms/step: RL={_fmt(rl["runtime_ms_per_step"])} A*={_fmt(ab["runtime_ms_per_step"])} | '
                          f'policy_ms/step: RL={_fmt(rl["decision_ms_per_step"])} A*={_fmt(ab["decision_ms_per_step"])}')
                if progress is not None:
                    tqdm.write(status)
                else:
                    print(status, flush=True)
        summary = summarize(rows)
        (output_dir/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        save_csv(output_dir/'summary.csv',
                 [{'policy': k, **summary[k]} for k in ('rl', 'astar')],
                 ['policy'] + list(summary['rl'].keys()))
        (output_dir/'evaluation_config.json').write_text(json.dumps({
            'rl_checkpoint': str(checkpoint), 'difficulty': args.difficulty,
            'episodes': args.episodes, 'seed_base': args.seed_base,
            'seed_formula': 'seed_base+episode_index', 'max_cycles': env_cfg['max_cycles'],
            'drone_types': types, 'coverage_threshold': args.coverage_threshold,
            'astar_config': vars(cfg),
            'astar_observation_access': 'shared observed world maps + perfect self and teammate pose/velocity; NO hidden goal/true obstacles',
            'evaluation': 'same environment config, seed, 300-step max, independent online rollout; no early stopping at goal or coverage60',
            'time_reached': 'conditional on reaching; also report censored max_cycles+1',
            'timing': {
                'joint_step': 'one complete A/B team action round; a partially completed terminal round counts as one',
                'runtime_sec': 'wall clock from start of env.reset through episode completion and final metric reads; excludes env construction, Ray/algorithm startup, plotting and file writes',
                'runtime_ms_per_step': '1000 * sum(runtime_sec) / sum(episode_steps), not mean of episode ratios',
                'decision_sec': 'sum of RL action inference for all drones in a round, or A* planning for all drones in a round',
                'decision_ms_per_step': '1000 * sum(decision_sec) / sum(episode_steps)',
                'env_step_sec': 'sum of env.step(...) wall times (excludes env.last and other bookkeeping)',
                'sync_cuda_timing': args.sync_cuda_timing,
            },
        }, indent=2, ensure_ascii=False))
        if not args.no_plots:
            plots(output_dir, summary)
        print('\n===== FINAL RL VS DYNAMIC A* =====')
        for kind in ('rl', 'astar'):
            s=summary[kind]
            print(f'{kind:6s} goal={_fmt(s["goal_found_rate"],100)}% '
                  f'goal_step={_fmt(s["goal_step_reached_mean"])} '
                  f'final_cov={_fmt(s["final_coverage_mean"],100)}% '
                  f'cov60_reached={_fmt(s["coverage60_rate"],100)}% '
                  f'cov60_step={_fmt(s["coverage60_step_reached_mean"])} '
                  f'crash={_fmt(s["fatal_crash_rate"],100)}% '
                  f'wall={_fmt(s["runtime_sec_total"])}s/{s["episode_steps_total"]}steps '
                  f'= {_fmt(s["runtime_ms_per_step"])}ms/step '
                  f'policy={_fmt(s["decision_ms_per_step"])}ms/step', flush=True)
        print(f'Saved results: {output_dir}', flush=True)
    finally:
        if algo is not None:
            algo.stop()
        collector.ray.shutdown()


if __name__ == '__main__':
    main()
