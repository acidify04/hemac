# Cross-agent HiSSD Adapter

This bundle **does not replace** the existing local adapter implementation.

## A/B separation

Existing local adapter (keep as-is):

- `src/skill_discovery/hissd_joint_hetero_adapter_models.py`
- `src/skill_discovery/train_hissd_joint_hetero_adapter_scratch_with_rollout.py`
- `src/skill_discovery/evaluate_hissd_joint_hetero_adapter_zero_shot.py`

New cross-agent adapter:

- `hissd_joint_hetero_cross_agent_adapter_models.py`
- `train_hissd_joint_hetero_cross_agent_adapter_scratch_with_rollout.py`
- `evaluate_hissd_joint_hetero_cross_agent_adapter_zero_shot.py`
- `check_cross_agent_adapter.py`

Copy the four new `.py` files into `src/skill_discovery/`.

## Architecture

Local baseline:

```text
h_i = E_role(o_i)
c_i = CommonSkillEncoder(h_i, history_i)
delta_i = A_local([c_i, h_i])
c'_i = c_i + delta_i
```

Cross-agent variant:

```text
x_i = [h_i, c_i]
g_i = SelfAttention(x_1, ..., x_N)_i
delta_i = A_cross([c_i, h_i, g_i])
c'_i = c_i + delta_i
```

The attention operates across the agent axis and uses `valid_mask`, so D2O1,
D3O2, D5O2, etc. do not require a fixed population size.

The same contextual adapter parameters are shared across Drone and Observer.
There is no explicit role ID, agent ID, or population-size embedding.

The final residual layer is zero initialized, so the new adapter is an exact
identity at construction.

## Smoke test

```bash
cd ~/Documents/hemac
conda activate hemac

python -u src/skill_discovery/check_cross_agent_adapter.py
```

Expected:

```text
PASS
identity_error=0.000e+00
other_agent_effect_on_agent0=<positive number>
adapter_params=<number>
```

## Train: existing local adapter baseline

Keep using the existing trainer. Example:

```bash
python -u src/skill_discovery/train_hissd_joint_hetero_adapter_scratch_with_rollout.py \
  --manifest-31 src/skill_discovery/offline_data-3-1/dataset_splits_d1.json \
  --data-root-31 src/skill_discovery/offline_data-3-1 \
  --manifest-42 src/skill_discovery/offline_data-4-2/dataset_splits_d1.json \
  --data-root-42 src/skill_discovery/offline_data-4-2 \
  --output-dir src/skill_discovery/checkpoints/hissd-joint-local-adapter-scratch \
  --epochs 120 \
  --balanced-sampling \
  --early-stopping-patience 0 \
  --rollout-eval-every 5 \
  --rollout-eval-episodes 50 \
  --device cuda
```

## Train: cross-agent adapter

```bash
python -u src/skill_discovery/train_hissd_joint_hetero_cross_agent_adapter_scratch_with_rollout.py \
  --manifest-31 src/skill_discovery/offline_data-3-1/dataset_splits_d1.json \
  --data-root-31 src/skill_discovery/offline_data-3-1 \
  --manifest-42 src/skill_discovery/offline_data-4-2/dataset_splits_d1.json \
  --data-root-42 src/skill_discovery/offline_data-4-2 \
  --output-dir src/skill_discovery/checkpoints/hissd-joint-cross-agent-adapter-scratch \
  --epochs 120 \
  --adapter-hidden-dim 128 \
  --cross-agent-context-dim 64 \
  --cross-agent-attention-heads 4 \
  --cross-agent-attention-dropout 0 \
  --balanced-sampling \
  --early-stopping-patience 0 \
  --rollout-eval-every 5 \
  --rollout-eval-episodes 50 \
  --device cuda
```

Checkpoint names:

```text
hissd_joint_cross_agent_adapter_scratch_last.pt
hissd_joint_cross_agent_adapter_scratch_best.pt
hissd_joint_cross_agent_adapter_scratch_best_task.pt
```

`best.pt` uses the same combined source-validation loss as the existing
HiSSD+local-adapter trainer. Source rollout diagnostics remain
`checkpoint_selection=OFF`.

## Zero-shot evaluation

Example D2O1:

```bash
python -u src/skill_discovery/evaluate_hissd_joint_hetero_cross_agent_adapter_zero_shot.py \
  --checkpoint src/skill_discovery/checkpoints/hissd-joint-cross-agent-adapter-scratch/hissd_joint_cross_agent_adapter_scratch_best.pt \
  --env-template-data-root src/skill_discovery/offline_data-4-2 \
  --n-drones 2 \
  --n-observers 1 \
  --difficulty 2 \
  --episodes 50 \
  --seed-base 100000000 \
  --device cuda \
  --output-json /tmp/cross_adapter_D2O1.json
```

For D2 with `--seed-base 100000000`, the actual seeds are
`100200000..100200049`, matching the recent BC / HiSSD / local-adapter D2
comparison.

## Recommended ablation

Use the same training data, seed, budget, sampling and checkpoint criterion for:

1. Scratch Joint HiSSD — no adapter.
2. Scratch Joint HiSSD + local adapter: `A([c_i,h_i])`.
3. Scratch Joint HiSSD + cross-agent adapter:
   `A([c_i,h_i,g_i])`, where `g_i=SelfAttention([h_j,c_j]_{j=1..N})_i`.

This isolates the additional effect of explicit current cross-agent context.
