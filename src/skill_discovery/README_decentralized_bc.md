# Decentralized BC for fair HiSSD comparison

This implementation preserves the existing synchronized D3O1/D4O2 training
pipeline, equal source minibatch balancing, role-balanced BC loss, action
normalization, seed handling, and source-validation MSE checkpoint selection.

The only architecture change from the previous Joint BC is:

```text
OLD Joint BC:
o_i -> E_role(o_i) -> h_i
[h_1,...,h_N] -> cross-agent Transformer -> action_i

NEW Decentralized BC:
o_i -> E_role(o_i) -> h_i -> role action head -> action_i
```

Thus another agent's private `local_map`, `global_map`, or `action_history`
cannot enter agent i's action path. Information already present in `o_i`
(teammate positions, shared explored region, known enemies) remains available.

Copy these files into `src/skill_discovery/`:

- `decentralized_bc_models.py`
- `train_decentralized_bc_multisource.py`
- `evaluate_decentralized_bc_zero_shot.py`
- `check_decentralized_bc.py`

## Leakage check

```bash
cd ~/Documents/hemac
conda activate hemac
python -u src/skill_discovery/check_decentralized_bc.py
```

Expected:

```text
PASS
other_agent_private_obs_effect=0.000000e+00
own_obs_effect=<positive>
```

## Train

Use the same source data and protocol as the previous Joint BC:

```bash
python -u src/skill_discovery/train_decentralized_bc_multisource.py \
  --manifest-31 src/skill_discovery/offline_data-3-1/dataset_splits_d1.json \
  --data-root-31 src/skill_discovery/offline_data-3-1 \
  --manifest-42 src/skill_discovery/offline_data-4-2/dataset_splits_d1.json \
  --data-root-42 src/skill_discovery/offline_data-4-2 \
  --output-dir src/skill_discovery/checkpoints/decentralized-bc-source-combined \
  --epochs 50 \
  --balanced-sampling \
  --seed 2026 \
  --device cuda
```

Outputs:

```text
decentralized_bc_last.pt
decentralized_bc_best.pt
```

`best.pt` is selected by the same equal-source validation action MSE criterion
used by the previous Joint BC.

## D1 evaluation

```bash
CKPT=src/skill_discovery/checkpoints/decentralized-bc-source-combined/decentralized_bc_best.pt
EVAL=src/skill_discovery/evaluate_decentralized_bc_zero_shot.py
OUT=src/skill_discovery/checkpoints/decentralized-bc-source-combined/eval-d1
mkdir -p "$OUT"

for SPEC in "2 1" "2 2" "3 1" "3 2" "4 1" "4 2" "5 1" "5 2"; do
  read D O <<< "$SPEC"
  python -u "$EVAL" \
    --checkpoint "$CKPT" \
    --env-template-data-root src/skill_discovery/offline_data-4-2 \
    --n-drones "$D" \
    --n-observers "$O" \
    --difficulty 1 \
    --episodes 50 \
    --seed-base 100000000 \
    --device cuda \
    --output-json "$OUT/D${D}O${O}.json"
done
```

## D2 evaluation

Change only:

```text
--difficulty 2
OUT=.../eval-d2
```

With seed base `100000000`, D2 uses exactly
`100200000..100200049`.
