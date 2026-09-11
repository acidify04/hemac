# Skill Discovery Directory

The directory intentionally keeps the established HiSSD experiments and the
new homogeneous skill-VAE experiment separate. Generated data, checkpoints,
and outputs are ignored by Git and can be archived without changing source.

## Homogeneous skill VAE

- `drone_skill_vae.py`: shared drone observation encoder, recurrent Gaussian
  posterior, reparameterization, skill-only residual decoder, and forward
  predictor. The frozen BC action path cannot bypass the bounded latent
  residual, and action-anchor plus latent variance/decorrelation terms prevent
  harmful source-data overfitting and posterior collapse.
- `train_drone_skill_vae.py`: offline VAE training with action reconstruction,
  KL regularization, dynamics prediction, latent usage, and temporal smoothness.
- `finetune_drone_skill_vae_online.py`: drone-only CTDE PPO. The actor uses only
  each drone's observation/history; the central map is used only by the critic.
  Its `full` and `no_skill` modes share the same PPO implementation for a
  controlled latent-skill ablation.

## Data pipeline

- `collect_offline_data.py`: collect episode-separated MAPPO trajectories.
- `build_dataset_splits.py`: build source/target train/validation/test manifests.
- `dataset.py`: lazy mmap-backed trajectory windows and DataLoaders.
- `drone_task.py`: drone goal-found plus coverage success definition.
- `task_descriptor.py`: curriculum and realized-dynamics descriptors for HiSSD.
- `visualize_offline_episode.py`: inspect one collected `.pt` episode.

## BC baseline

- `models.py`: shared CNN observation encoder, BC policy, and observer residual.
- `train_drone_bc.py`: train the drone behavior-cloning baseline.
- `evaluate_drone_bc.py`: evaluate BC and MAPPO policies in the environment.

## HiSSD pipeline

- `hissd_models.py`: common/task skill encoders, planner, decoder, and value nets.
- `train_hissd.py`: source-task offline HiSSD training.
- `adapt_hissd_target.py`: target offline adaptation with source anchors.
- `finetune_hissd_drone_online.py`: homogeneous drone-only online PPO.
- `finetune_hissd_online.py`: heterogeneous drone-observer joint online PPO.
- `evaluate_hissd.py`: rollout evaluation and BC/MAPPO comparisons.
- `visualize_hissd_skills.py`: skill embeddings and linear-probe visualization.

## Experiment orchestration and analysis

- `analyze_learning_efficiency.py`: normalized success AUC, gain AUC, threshold
  steps, paired statistics, CSV, and plots.
- `analyze_hissd_evaluation.py`: analyze static rollout evaluation JSON files.
- `run_skill_structure_ablation.py`: common-only/task-only/split/shared study.
- `run_hissd_ablation_multiseed.py`: descriptor/task-classifier HiSSD ablations.
- `run_learning_efficiency_ablation.py`: learning-efficiency ablation orchestration.
- `run_vae_skill_control.py`: paired multi-seed full-latent versus no-skill
  experiment using identical BC/PPO/critic settings and automatic AUC analysis.
- `finetune_drone_baseline_online.py`: scratch/MAPPO/BC online PPO baselines.

## Cleanup guidance

Safe to delete at any time:

- `__pycache__/`: Python bytecode cache.
- TensorBoard event files: generated logs that do not affect checkpoints.
- `.npz` visualization caches: regenerable from their model checkpoints.

Delete or archive only after preserving final research results:

- `outputs/`: evaluation JSON/CSV/PNG and learning curves.
- `checkpoints/`, `bc_checkpoints*/`, `hissd_checkpoints*/`: trained weights.
- `offline_data*/`: collected episodes; needed to retrain offline models.

No Python source file is currently safe to delete purely as dead code. Older
experiment runners overlap in purpose, but retained result metadata and command
history may still depend on their method names and checkpoint formats. Archive
old versioned outputs/checkpoints first; consolidate source only after the final
experimental protocol is fixed.
