"""Smoke-test the zero-initialized HiSSD heterogeneity adapter.

Run from the repository root after placing hissd_hetero_models.py under
src/skill_discovery/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from skill_discovery.hissd_hetero_models import HeterogeneousHeMACHISSD
from skill_discovery.visualize_hissd_skills import load_hissd_model, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hissd-checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--role-adapter-hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    base, _ = load_hissd_model(args.hissd_checkpoint, device)
    base.eval()

    hetero = HeterogeneousHeMACHISSD.from_base_model(
        base,
        role_adapter_hidden_dim=args.role_adapter_hidden_dim,
    ).to(device)
    hetero.eval()

    # Freeze everything except the new module so the gradient test is unambiguous.
    for parameter in hetero.parameters():
        parameter.requires_grad_(False)
    for parameter in hetero.common_skill_role_adapter.parameters():
        parameter.requires_grad_(True)

    batch = int(args.batch_size)
    agents = int(base.agent_count)
    obs_dim = int(base.observation_encoder.output_dim)
    skill_dim = int(base.skill_dim)

    generator = torch.Generator(device=device)
    generator.manual_seed(12345)
    observation_features = torch.randn(
        batch, agents, obs_dim, generator=generator, device=device
    )
    common_skills = torch.randn(
        batch, agents, skill_dim, generator=generator, device=device
    )
    task_skills = torch.randn(
        batch, agents, skill_dim, generator=generator, device=device
    )

    with torch.no_grad():
        adapted, delta = hetero.adapt_common_skills(
            observation_features, common_skills
        )
        base_logits = base.decode_action_logits(
            observation_features, common_skills, task_skills
        )
        hetero_logits = hetero.decode_action_logits(
            observation_features, common_skills, task_skills
        )

    delta_max = float(delta.abs().max())
    skill_identity_max = float((adapted - common_skills).abs().max())
    logit_diff_max = float((hetero_logits - base_logits).abs().max())

    print(f"device={device}")
    print(f"adapter_parameters={hetero.adapter_parameter_count()}")
    print(f"delta_max={delta_max:.9g}")
    print(f"adapted_minus_raw_max={skill_identity_max:.9g}")
    print(f"hetero_minus_base_logit_max={logit_diff_max:.9g}")

    tolerance = 1e-6
    if delta_max > tolerance or skill_identity_max > tolerance or logit_diff_max > tolerance:
        raise RuntimeError("Identity initialization check FAILED.")
    print("IDENTITY CHECK: PASS")

    # Confirm PPO can send a gradient into the new adapter.
    hetero.zero_grad(set_to_none=True)
    logits = hetero.decode_action_logits(
        observation_features, common_skills, task_skills
    )
    synthetic_loss = logits.square().mean()
    synthetic_loss.backward()

    grad_sq = torch.zeros((), device=device)
    nonzero_grad_tensors = 0
    for parameter in hetero.common_skill_role_adapter.parameters():
        if parameter.grad is not None:
            grad_sq = grad_sq + parameter.grad.detach().square().sum()
            if bool(parameter.grad.detach().abs().max() > 0):
                nonzero_grad_tensors += 1
    grad_norm = float(grad_sq.sqrt())

    print(f"adapter_grad_norm={grad_norm:.9g}")
    print(f"nonzero_adapter_grad_tensors={nonzero_grad_tensors}")
    if not grad_norm > 0.0:
        raise RuntimeError("Gradient-flow check FAILED: adapter gradient is zero.")
    print("GRADIENT CHECK: PASS")


if __name__ == "__main__":
    main()
