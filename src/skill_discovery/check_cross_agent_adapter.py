#!/usr/bin/env python3
from __future__ import annotations

import torch

from skill_discovery.hissd_joint_hetero_cross_agent_adapter_models import (
    CrossAgentCommonSkillAdapter,
)


def main() -> None:
    torch.manual_seed(2026)

    adapter = CrossAgentCommonSkillAdapter(
        observation_dim=96,
        skill_dim=64,
        hidden_dim=128,
        context_dim=64,
        attention_heads=4,
        attention_dropout=0.0,
    )

    # Offline-style batch with one fully padded timestep.
    h = torch.randn(2, 3, 6, 96)
    c = torch.randn(2, 3, 6, 64)
    valid = torch.ones(2, 3, 6, dtype=torch.bool)
    valid[:, :, -1] = False
    valid[0, 2, :] = False  # fully padded timestep

    adapted, delta = adapter(c, h, valid)

    assert torch.isfinite(adapted).all()
    assert torch.isfinite(delta).all()
    assert torch.equal(adapted[0, 2], c[0, 2])
    assert torch.equal(delta[0, 2], torch.zeros_like(delta[0, 2]))
    assert torch.equal(adapted, c), "Adapter must be identity at init."

    # Activate residual path, then verify another agent changes agent 0.
    with torch.no_grad():
        torch.nn.init.normal_(adapter.network[-1].weight, std=1e-2)
        torch.nn.init.zeros_(adapter.network[-1].bias)

    h1 = torch.randn(1, 4, 96)
    c1 = torch.randn(1, 4, 64)
    mask1 = torch.ones(1, 4, dtype=torch.bool)

    y1, _ = adapter(c1, h1, mask1)
    h2 = h1.clone()
    c2 = c1.clone()
    h2[:, 1] += 5.0
    c2[:, 1] -= 3.0
    y2, _ = adapter(c2, h2, mask1)

    effect = (y2[:, 0] - y1[:, 0]).abs().max().item()
    assert effect > 0.0

    print("PASS")
    print("all_invalid_timestep_handled=True")
    print(f"identity_error={(adapted-c).abs().max().item():.3e}")
    print(f"other_agent_effect_on_agent0={effect:.6e}")


if __name__ == "__main__":
    main()
