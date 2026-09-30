"""Train independent HalfCheetah actors with sequential HAPPO updates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .env import add_environment_version_argument, infer_space_dimensions, make_env
from .happo import (
    HAPPOTrainer,
    collect_episode,
    concatenate_episodes,
    resolve_device,
    seed_everything,
)
from .models import CentralCritic, IndependentActors, checkpoint_payload
from .tasks import SUPPORTED_SUITES, get_task, task_names
from .difficulty_protocol import add_difficulty_arguments, protocol_from_args


DEFAULT_OUTPUT_ROOT = Path("src/mamujoco/checkpoints/happo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUPPORTED_SUITES, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--allow-target-task",
        action="store_true",
        help="Permit target-task HAPPO only for the isolated cross-evaluation pilot.",
    )
    parser.add_argument("--agent-conf", default="6x1")
    parser.add_argument("--agent-obsk", type=int, default=1)
    parser.add_argument("--max-cycles", type=int, default=1000)
    parser.add_argument("--total-env-steps", type=int, default=2_000_000)
    parser.add_argument("--episodes-per-update", type=int, default=8)
    parser.add_argument("--eval-every-updates", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--ppo-epochs", type=int, default=5)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    add_environment_version_argument(parser)
    add_difficulty_arguments(parser)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    task_options = {}
    if args.suite == "difficulty":
        protocol = protocol_from_args(args)
        task_options = {
            "difficulty_strengths": protocol.strengths,
            "source_difficulties": protocol.source_ids,
            "target_difficulties": protocol.target_ids,
        }
    if args.task not in task_names(args.suite, **task_options):
        raise ValueError(
            f"Unknown task {args.task!r}; choose from "
            f"{task_names(args.suite, **task_options)}"
        )
    if (
        get_task(args.suite, args.task, **task_options).split != "source"
        and not args.allow_target_task
    ):
        raise ValueError("HAPPO behavior policies may only be trained on source tasks.")
    for name in (
        "max_cycles",
        "total_env_steps",
        "episodes_per_update",
        "eval_every_updates",
        "eval_episodes",
        "ppo_epochs",
        "minibatch_size",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def evaluate(env, actors, critic, *, episodes: int, seed: int, device):
    records = []
    for index in range(episodes):
        episode = collect_episode(
            env,
            actors,
            critic,
            seed=seed + index,
            deterministic=True,
            device=device,
        )
        records.append(episode.metrics)
    return {
        key: float(np.mean([record[key] for record in records]))
        for key in (
            "episode_return",
            "forward_distance",
            "forward_reward",
            "forward_velocity",
            "control_cost",
            "episode_length",
        )
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    task_options = {}
    if args.suite == "difficulty":
        protocol = protocol_from_args(args)
        task_options = {
            "difficulty_strengths": protocol.strengths,
            "source_difficulties": protocol.source_ids,
            "target_difficulties": protocol.target_ids,
        }
    task = get_task(args.suite, args.task, **task_options)
    env = make_env(
        task,
        agent_conf=args.agent_conf,
        agent_obsk=args.agent_obsk,
        environment_version=args.environment_version,
        max_cycles=args.max_cycles,
    )
    env.reset(seed=args.seed)
    observation_dims, action_dims, state_dim = infer_space_dimensions(env)
    actors = IndependentActors(observation_dims, action_dims).to(device)
    critic = CentralCritic(state_dim).to(device)
    trainer = HAPPOTrainer(
        actors,
        critic,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        clip_ratio=args.clip_ratio,
        device=device,
    )

    output_dir = (
        args.output_root
        / args.environment_version
        / args.suite
        / args.task
        / f"seed_{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "training_metrics.jsonl"
    total_steps = 0
    update = 0
    best_return = -float("inf")
    try:
        while total_steps < args.total_env_steps:
            episodes = [
                collect_episode(
                    env,
                    actors,
                    critic,
                    seed=args.seed * 1_000_000 + update * args.episodes_per_update + i,
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                    device=device,
                )
                for i in range(args.episodes_per_update)
            ]
            batch = concatenate_episodes(episodes)
            update_metrics = trainer.update(batch)
            total_steps += int(batch.states.shape[0])
            update += 1
            record = {
                "update": update,
                "environment_steps": total_steps,
                **batch.metrics,
                **update_metrics,
            }
            if update % args.eval_every_updates == 0:
                evaluation = evaluate(
                    env,
                    actors,
                    critic,
                    episodes=args.eval_episodes,
                    seed=args.seed * 10_000_000 + update * args.eval_episodes,
                    device=device,
                )
                record.update({f"eval/{key}": value for key, value in evaluation.items()})
                metadata = {
                    "suite": args.suite,
                    "task": task.to_dict(),
                    "seed": args.seed,
                    "environment_steps": total_steps,
                    "update": update,
                    "evaluation": evaluation,
                    "agent_conf": args.agent_conf,
                    "agent_obsk": args.agent_obsk,
                    "environment_version": args.environment_version,
                    "backend": (
                        "gymnasium_robotics.mamujoco_v1/"
                        f"{args.environment_version}"
                    ),
                }
                payload = checkpoint_payload(actors, critic, metadata=metadata)
                torch.save(payload, output_dir / "latest.pt")
                torch.save(
                    payload,
                    output_dir / f"checkpoint_{total_steps:09d}.pt",
                )
                if evaluation["episode_return"] > best_return:
                    best_return = evaluation["episode_return"]
                    torch.save(payload, output_dir / "best.pt")
                print(
                    f"update={update} steps={total_steps} "
                    f"return={evaluation['episode_return']:.2f} "
                    f"distance={evaluation['forward_distance']:.2f}"
                )
            with history_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, sort_keys=True) + "\n")
    finally:
        env.close()


if __name__ == "__main__":
    main()
