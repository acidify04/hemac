# HiSSD MAMuJoCo parameter audit

## HiSSD-based difficulty-generalization adaptation on MAMuJoCo

This is not a HiSSD reproduction. It applies the repository's Drone/HiSSD
difficulty-generalization structure to the maintained six-agent HalfCheetah
environment: the locomotion task is unchanged, while a continuous common
actuator-strength multiplier defines D1=1.0 and D2=0.8 source difficulties and
D3=0.6 and D4=0.4 unseen targets. Difficulty is inferred implicitly from
observation history and is never passed to the policy as an ID or scalar.

The source dataset contains only D1/D2 (High/Mid/Low, 100 episodes each). D3
and D4 are evaluated zero-shot and then adapted in separate runs using the
current target plus immutable D1/D2 replay; the other target is never visible.
The default adaptation batch is 64 target, 32 D1, and 32 D2, with 500,000 joint
environment steps and evaluation at 0, 10k, ..., 500k. The primary adaptation
summary is normalized reward AUC, calculated over the actual configured budget
and including step zero.

The observation-history representation follows the existing Drone design at
the architectural level: local observation encoding; current feature, delta,
running mean, running standard deviation, and running absolute delta; a causal
history transformer; per-agent features; active-agent mean pooling; pooled
delta; GRU context; and task-skill/query heads. MAMuJoCo uses a vector-specific
implementation so the existing Drone execution path remains unchanged.

The source loss retains expectile TD value learning, `exp(TD / alpha)` planner
weighting, and the beta-weighted task objective. Source difficulty labels are
used for auxiliary classification/contrastive supervision only. Target
adaptation uses supervised contrastive learning together with source anchors
and a zero-initialized agent-specific residual actor; no observer is used.

### Fixed and configurable protocol values

| Item | Default | Configuration |
| --- | ---: | --- |
| Agent factorization / observation radius | 6x1 / 1 | HAPPO CLI |
| Environment | HalfCheetah-v2 | `--environment-version` (v5 optional) |
| D1/D2/D3/D4 strengths | 1.0/0.8/0.6/0.4 | `--difficulty-strengths` |
| Source / target | D1,D2 / D3,D4 | `--source-difficulties`, `--target-difficulties` |
| HAPPO budget | 2,000,000 steps/source | `--happo-total-env-steps` |
| Source trajectories | 100/quality | `--episodes-per-quality` |
| Experiment seeds | 1,2,3,4,5 | `--experiment-seeds` |
| Zero-shot/adaptation evaluation | 20 episodes | `--difficulty-eval-episodes` |
| Adaptation budget / interval | 500,000 / 10,000 | corresponding runner options |
| Adaptation batch size | 128 | `--adaptation-batch-size` |
| Replay ratio | 50%/25%/25% | target/source replay ratio options |

The canonical default injects Gymnasium `HalfCheetah-v2` into the maintained
Gymnasium Robotics MaMuJoCo factorization wrapper. Version 5 remains an
explicit optional override. Results from the two physics backends must not be
pooled; the selected version is stored in metadata and checked on load.

## Published HiSSD values retained by the source learner

The authoritative values below come from Table 10 in the HiSSD paper's
MAMuJoCo appendix. These are the defaults used by `train_offline.py`.

| Parameter | HiSSD paper | This package |
| --- | ---: | ---: |
| Hidden layer dimension | 256 | 256 |
| Hidden units in MLP | 256 | 256 |
| Attention dimension | 256 | 256 |
| Skill dimension per token | 256 | 256 |
| Discount factor | 0.99 | 0.99 |
| Target update rate | 0.005 | 0.005 |
| Alpha | 10.0 | 10.0 |
| Beta | 2.0 | 2.0 |
| Expectile epsilon | 0.9 | 0.9 |
| Batch size | 128 | 128 |
| Training steps | 1,000,000 | 1,000,000 |
| Optimizer | Adam | Adam |
| Learning rate | 0.0005 | 0.0005 |

The paper pseudocode samples 128 one-step transitions, so the protocol uses a
sequence length of 1. Both HiSSD and the Skill-VAE comparison use 256-dimensional
hidden and skill representations and the same optimizer, batch size, learning
rate, training budget, data, and seeds. Algorithm-specific losses necessarily
remain different.

## Reproducibility limitations

- The released HiSSD repository contains the SMAC implementation but not the
  continuous-action MAMuJoCo implementation. Transformer heads, activations,
  gradient clipping, contrastive temperature, and several module-level details
  are therefore not recoverable from the MAMuJoCo release. Gradient clipping 10
  follows the released HiSSD configuration; the remaining choices are recorded
  in checkpoints.
- The appendix prose says the MAMuJoCo transformers have 64-unit hidden layers,
  while its dedicated hyperparameter table says 256. This package follows the
  dedicated table and uses 256.
- The paper reports that HAPPO generated the data, but does not report the
  HAPPO training budget or its optimization hyperparameters. The local HAPPO
  defaults are data-generation choices, not paper-exact settings.
- Canonical HalfCheetah-v2 depends on the deprecated, unmaintained `mujoco-py`
  backend. HalfCheetah-v5 is supported as an explicit operational fallback,
  but v2 and v5 returns are not numerically interchangeable.
- The paper's contrastive loss uses two agents in the same task as a positive
  pair and a momentum encoder for negatives from other tasks. The current
  continuous adaptation uses supervised same-task contrast over a mixed batch
  and has no momentum task encoder. This is the largest remaining algorithmic
  difference and prevents claiming exact HiSSD reproduction.
- The paper pseudocode samples one source task per update and obtains negatives
  from other-task data through the momentum encoder. The current loader samples
  mixed-task batches directly; it matches the numerical batch size but not that
  sampling procedure.
- The paper's forward predictor produces both the next global state and next
  local-information embeddings. The current predictor reconstructs only the
  next global state.
- The current coordination transformer mixes all six local observations before
  producing actions. It is therefore centralized at execution, whereas the
  paper claims local-information-only decentralized execution. This must be
  redesigned before the implementation can be presented as a faithful HiSSD
  reproduction.
