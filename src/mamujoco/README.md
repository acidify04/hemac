# MaMuJoCo offline multi-task transfer

This package ports the repository's independent-policy HAPPO, Skill-VAE, and
HiSSD pipeline to the maintained Gymnasium Robotics MaMuJoCo environment.

## Fixed environment protocol

- Backend: `gymnasium_robotics.mamujoco_v1` with canonical `HalfCheetah-v2`
- Scenario: `HalfCheetah`
- Factorization: `6x1`
- Observation radius: `agent_obsk=1`
- Six independent actors, one for each joint
- Sequential HAPPO actor updates followed by one centralized-critic update
- Episode horizon: 1,000

`HalfCheetah-v5` remains available through
`--environment-version HalfCheetah-v5`. Environment versions are written to
checkpoints, manifests, evaluations, and pilot results; incompatible
checkpoints are rejected rather than silently mixed.

## Transfer suites

### HiSSD-based difficulty-generalization adaptation

The `difficulty` suite keeps the forward-running task fixed and changes only
the common actuator-strength multiplier. Its default protocol is:

| ID | Strength | Split |
| --- | ---: | --- |
| D1 | 1.0 | source |
| D2 | 0.8 | source |
| D3 | 0.6 | unseen target |
| D4 | 0.4 | unseen target |

Each reset restores the nominal MuJoCo actuator gear and then applies the
requested multiplier, so scaling cannot accumulate. Difficulty IDs, scalar
values, one-hot encodings, and source/target flags are never policy inputs.
The HiSSD-based controller instead infers a context from causal observation
history using per-agent observation encoders, current/delta/running statistics,
a history transformer, active-agent mean pooling, pooled deltas, and a GRU.
Difficulty labels supervise only the source classification/contrastive losses
and target supervised-contrastive loss.

Only D1/D2 are stored in the source manifest. For each of them, High, Mid, and
Low HAPPO checkpoints are selected nearest to `Rmax`, `2/3 Rmax`, and `1/3
Rmax`; 100 episodes are then collected from each checkpoint (600 total).
Zero-shot evaluation uses 20 episodes per target and seed with zero target
gradient updates. D3 and D4 adaptation are independent: each uses only its own
online transitions plus immutable D1/D2 replay. The default 128-sample batch is
64 target + 32 D1 + 32 D2.

Each target receives 500,000 joint environment steps. One simultaneous action
from all six agents followed by one environment transition counts as one step,
not six. Evaluation occurs at step 0 and every 10,000 steps through 500,000
(51 points, 20 episodes per point). Results contain raw episodes, per-point
mean/std return, forward velocity and control cost, plus trapezoidal raw and
budget-normalized reward AUC.

This experiment is an extension of the repository's Drone/HiSSD design. It is
not presented as a reproduction of the HiSSD MAMuJoCo result.

### Earlier transfer suites

`joint_disable` follows the paper protocol. Source tasks are `complete`,
`back_thigh`, `back_foot`, `front_thigh`, and `front_shin`. The unseen target
tasks are `back_shin` and `front_foot`. A disabled agent remains present but its
joint torque is forced to zero and its actor is excluded from the HAPPO loss.

`dynamics` uses `nominal`, mass scales 0.75/1.25, and friction scales 0.75/1.25
as source tasks. Actuator-strength scales 0.75/1.25 are held out as unseen
targets. Agent and tensor shapes are identical in every task.

## Environment

The dependencies are intentionally isolated from the HeMAC RLlib environment.

```bash
conda activate mappo
pip install -r src/mamujoco/requirements.txt
export PYTHONPATH="$PWD/src"
```

## Difficulty protocol: execution order

Inspect the exact commands first:

```bash
python -m mamujoco.run_protocol \
  --suite difficulty --algorithms hissd --stage all --device auto --dry-run
```

Run the complete default pipeline:

```bash
python -m mamujoco.run_protocol \
  --suite difficulty --algorithms hissd --stage all --device cuda
```

`all` executes these stages in order:

1. `happo`: train D1 and D2 behavior policies for 2M environment steps each.
2. `pilot`: independently train D1-D4 HAPPO policies for all five experiment
   seeds in an isolated checkpoint root, then evaluate every policy on D1-D4.
3. `select`: select High/Mid/Low checkpoints from each HAPPO evaluation curve.
4. `collect`: collect 100 episodes per source difficulty and quality.
5. `offline`: train the source-only HiSSD-based model for 1M gradient steps for
   each of five seeds.
6. `zero_shot`: evaluate D1-D4; D3/D4 are the held-out results.
7. `adapt`: independently adapt D3 and D4 for 500k steps per seed.
8. `evaluate`: aggregate the five-seed zero-shot and adaptation outputs.

Each stage can be run independently by replacing `--stage all`. The `select`
stage is explicit because collection needs the selected checkpoint metadata.
The main controls are `--difficulty-strengths`, `--source-difficulties`,
`--target-difficulties`, `--environment-version`,
`--happo-total-env-steps`,
`--happo-episodes-per-update`, `--happo-eval-every-updates`,
`--happo-eval-episodes`, `--max-cycles`,
`--episodes-per-quality`, `--offline-training-steps`,
`--experiment-seeds`, `--difficulty-eval-episodes`, `--adaptation-budget`,
`--adaptation-eval-interval`, `--adaptation-batch-size`,
`--target-replay-ratio`, and
`--source-replay-ratios`. The source/target lists must remain a two-by-two
partition of D1-D4 because the adaptation protocol has two source replay
components.

Default artifacts are written below:

```text
src/mamujoco/checkpoints/happo/HalfCheetah-v2/difficulty/{D1,D2}/seed_1/
src/mamujoco/checkpoints/pilot_happo/HalfCheetah-v2/difficulty/D*/seed_*/
src/mamujoco/checkpoints/happo/HalfCheetah-v2/difficulty/quality_selection_seed_1.json
src/mamujoco/offline_data/difficulty/{D1,D2}/{high,mid,low}/
src/mamujoco/offline_data/difficulty/manifest.json
src/mamujoco/checkpoints/offline/difficulty/hissd/seed_<seed>/
src/mamujoco/checkpoints/adaptation/{D3,D4}/seed_<seed>/
src/mamujoco/outputs/difficulty/hissd/zero_shot_100_runs.json
src/mamujoco/outputs/difficulty/adaptation/{adaptation_summary.json,adaptation_summary.csv}
src/mamujoco/outputs/difficulty/pilot_cross/HalfCheetah-v2/
```

The pilot output directory contains `cross_evaluation.json`, per-seed
`cross_evaluation.csv`, raw episode CSV, and labeled 4x4 CSV matrices for
episode return, forward velocity, and control cost. JSON also separates the
four diagonal and twelve off-diagonal pair summaries. Pilot D3/D4 checkpoints
live only under `pilot_happo`; source collection reads only the regular D1/D2
`checkpoints/happo` tree.

For a quick wiring check, reduce all budgets without changing the production
defaults, for example:

```bash
python -m mamujoco.run_protocol \
  --suite difficulty --algorithms hissd --stage all --device cpu \
  --max-cycles 2 --happo-total-env-steps 4 \
  --happo-episodes-per-update 1 --happo-eval-every-updates 1 \
  --happo-eval-episodes 1 --episodes-per-quality 1 \
  --offline-training-steps 1 --experiment-seeds 1 \
  --difficulty-eval-episodes 1 --adaptation-budget 4 \
  --adaptation-eval-interval 2 --adaptation-batch-size 4
```

The 4-step smoke command uses a batch of four with counts 2/1/1. Omitting that
override retains the production batch of 128 with counts 64/32/32.

## Earlier full protocol

Print the commands without running them:

```bash
python -m mamujoco.run_protocol --suite both --dry-run
```

Run both suites, both offline algorithms, and four training seeds:

```bash
python -m mamujoco.run_protocol --suite both --device cuda
```

The paper-scale defaults are 100 trajectories per source task, batch size 128,
1,000,000 offline gradient steps, learning rate 5e-4, discount 0.99, target
update rate 0.005, alpha 10.0, beta 2.0, expectile 0.9, four training seeds,
and eight evaluation episodes per seed. Hidden, MLP, attention, and per-token
skill dimensions are all 256. The four evaluation files are pooled into 32
runs.

The paper samples batches of 128 one-step transitions, so the default
`--sequence-length` is 1. The implementation accepts longer windows only as an
explicit non-paper ablation.

See [PARAMETER_AUDIT.md](PARAMETER_AUDIT.md) for the exact-match table and the
remaining reproducibility limitations in the paper and released code.

Stages can also be run separately:

```bash
python -m mamujoco.train_happo \
  --suite joint_disable --task complete --seed 1 --device cuda

python -m mamujoco.collect_offline_data \
  --suite joint_disable --task all --trajectories-per-task 100 --device cuda

python -m mamujoco.train_offline \
  --algorithm hissd --suite joint_disable --seed 1 --device cuda

python -m mamujoco.evaluate_zero_shot \
  --algorithm hissd --suite joint_disable \
  --checkpoint src/mamujoco/checkpoints/offline/joint_disable/hissd/seed_1/latest.pt \
  --episodes 8 --device cuda
```

Target-task trajectories are never read by collection or offline training.
The primary metric is mean episode return. Forward distance, forward velocity,
cumulative forward reward, and cumulative control cost are diagnostics.
