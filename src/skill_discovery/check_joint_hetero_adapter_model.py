"""Smoke-test JointHeterogeneousAdapterHiSSD with real BC checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from skill_discovery.hissd_joint_hetero_adapter_models import (
    JointHeterogeneousAdapterHiSSD,
)
from skill_discovery.models import DroneBehaviorCloningPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drone-bc",
        type=Path,
        default=Path(
            "src/skill_discovery/checkpoints/"
            "bc-source-combined/drone/drone_bc_best.pt"
        ),
    )
    parser.add_argument(
        "--observer-bc",
        type=Path,
        default=Path(
            "src/skill_discovery/checkpoints/"
            "bc-source-combined/observer/observer_bc_best.pt"
        ),
    )
    parser.add_argument("--central-map-channels", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def make_role_observations(
    config: dict,
    *,
    batch_size: int,
    time_steps: int | None,
    agent_count: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    prefix = (
        (batch_size, agent_count)
        if time_steps is None
        else (batch_size, time_steps, agent_count)
    )
    return {
        "global_map": torch.randn(
            *prefix,
            int(config["global_map_channels"]),
            *tuple(config["global_map_size"]),
            device=device,
        ),
        "local_map": torch.randn(
            *prefix,
            int(config["local_map_channels"]),
            *tuple(config["local_map_size"]),
            device=device,
        ),
        "action_history": torch.randn(
            *prefix,
            *tuple(config["action_history_shape"]),
            device=device,
        ),
    }


@torch.no_grad()
def bc_action(
    checkpoint_path: Path,
    observations: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    payload = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    policy = DroneBehaviorCloningPolicy(
        **payload["model_config"]
    ).to(device)
    policy.load_state_dict(payload["model_state_dict"])
    policy.eval()
    return (
        policy(
            observations["global_map"],
            observations["local_map"],
            observations["action_history"],
        ),
        payload,
    )


@torch.no_grad()
def check_population(
    model: JointHeterogeneousAdapterHiSSD,
    *,
    observer_count: int,
    drone_count: int,
    device: torch.device,
) -> None:
    B, T = 2, 4
    observations = {
        "observer": make_role_observations(
            model.role_configs["observer"],
            batch_size=B,
            time_steps=T,
            agent_count=observer_count,
            device=device,
        ),
        "drone": make_role_observations(
            model.role_configs["drone"],
            batch_size=B,
            time_steps=T,
            agent_count=drone_count,
            device=device,
        ),
    }
    masks = {
        "observer": torch.ones(
            B, T, observer_count, dtype=torch.bool, device=device
        ),
        "drone": torch.ones(
            B, T, drone_count, dtype=torch.bool, device=device
        ),
    }

    outputs = model.forward_joint(observations, masks)
    raw = outputs["raw_common_skills"]
    adapted = outputs["adapted_common_skills"]
    identity_error = float((raw - adapted).abs().max().item())

    expected_total = observer_count + drone_count
    assert outputs["observation_features"].shape == (
        B, T, expected_total, model.shared_observation_dim
    )
    assert outputs["common_skills"].shape == (
        B, T, expected_total, model.skill_dim
    )
    assert outputs["actions"]["observer"].shape[-2:] == (
        observer_count,
        model.action_dims["observer"],
    )
    assert outputs["actions"]["drone"].shape[-2:] == (
        drone_count,
        model.action_dims["drone"],
    )
    assert identity_error == 0.0

    # Online recurrent path must accept the same runtime population.
    online_obs = {
        role: {key: value[:, 0] for key, value in role_obs.items()}
        for role, role_obs in observations.items()
    }
    online_masks = {
        role: mask[:, 0] for role, mask in masks.items()
    }
    state = model.initial_joint_inference_state(
        B,
        observer_count=observer_count,
        drone_count=drone_count,
        device=device,
    )
    online, _ = model.inference_step(
        online_obs, online_masks, state
    )
    assert online["common_skills"].shape == (
        B, expected_total, model.skill_dim
    )

    print(
        f"D{drone_count}O{observer_count}: PASS "
        f"joint={tuple(outputs['common_skills'].shape)} "
        f"observer_action={tuple(outputs['actions']['observer'].shape)} "
        f"drone_action={tuple(outputs['actions']['drone'].shape)} "
        f"adapter_identity={identity_error:.3e}"
    )


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    model, report = JointHeterogeneousAdapterHiSSD.from_bc_checkpoints(
        drone_checkpoint=args.drone_bc,
        observer_checkpoint=args.observer_bc,
        central_map_channels=args.central_map_channels,
    )
    model = model.to(device).eval()

    print("===== JOINT HETEROGENEOUS HiSSD + ADAPTER =====")
    print(
        "BC epochs:",
        {
            role: values.get("epoch")
            for role, values in report.items()
        },
    )
    print("role action dims:", model.action_dims)
    print("shared observation dim:", model.shared_observation_dim)
    print("skill dim:", model.skill_dim)
    print("adapter params:", model.adapter_parameter_count())

    # Exact BC-policy identity at initialization.
    #
    # forward_joint() is the OFFLINE sequence path. The shared
    # CommonSkillEncoder therefore expects [B,T,A,D], not [B,A,D].
    # Use a one-step sequence here (T=1). The dedicated online path is
    # tested separately by check_population() via inference_step().
    test_observations = {
        role: make_role_observations(
            model.role_configs[role],
            batch_size=3,
            time_steps=1,
            agent_count=(2 if role == "observer" else 4),
            device=device,
        )
        for role in ("observer", "drone")
    }
    test_masks = {
        role: torch.ones(
            obs["global_map"].shape[:-3],
            dtype=torch.bool,
            device=device,
        )
        for role, obs in test_observations.items()
    }
    outputs = model.forward_joint(test_observations, test_masks)

    bc_paths = {
        "observer": args.observer_bc,
        "drone": args.drone_bc,
    }
    max_diffs = {}
    for role in ("observer", "drone"):
        expected, _ = bc_action(
            bc_paths[role], test_observations[role], device
        )
        actual = outputs["actions"][role]
        max_diffs[role] = float(
            (expected - actual).abs().max().item()
        )
        if max_diffs[role] > 1e-6:
            raise RuntimeError(
                f"{role} BC identity failed: "
                f"max_action_diff={max_diffs[role]:.3e}"
            )

    adapter_identity = float(
        (
            outputs["raw_common_skills"]
            - outputs["adapted_common_skills"]
        )
        .abs()
        .max()
        .item()
    )
    if adapter_identity != 0.0:
        raise RuntimeError(
            "Adapter is not exact identity at initialization: "
            f"{adapter_identity:.3e}"
        )

    print(
        "BC POLICY IDENTITY: PASS",
        {
            role: f"{value:.3e}"
            for role, value in max_diffs.items()
        },
    )
    print(
        f"ADAPTER IDENTITY: PASS max_diff={adapter_identity:.3e}"
    )

    # Source population shapes.
    check_population(
        model, observer_count=1, drone_count=3, device=device
    )
    check_population(
        model, observer_count=2, drone_count=4, device=device
    )

    # Unseen target population shapes.
    for drone_count, observer_count in (
        (2, 1),
        (2, 2),
        (3, 2),
        (4, 1),
        (5, 1),
        (5, 2),
    ):
        check_population(
            model,
            observer_count=observer_count,
            drone_count=drone_count,
            device=device,
        )

    print("ALL CHECKS: PASS")


if __name__ == "__main__":
    main()
