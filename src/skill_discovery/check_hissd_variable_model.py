#!/usr/bin/env python3
"""Convert/check a fixed-cardinality HiSSD checkpoint as a variable-agent model."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from skill_discovery.hissd_models import HeMACHISSD
from skill_discovery.hissd_variable_models import VariableAgentHeMACHISSD


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-checkpoint", type=Path)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--test-agent-counts", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--variable-mixer-hidden-dim", type=int, default=128)
    return p.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def random_observations(model, batch: int, agents: int, device: torch.device):
    cfg = model.config()
    g_channels = int(cfg["global_map_channels"])
    l_channels = int(cfg["local_map_channels"])
    gh, gw = tuple(cfg["global_map_size"])
    lh, lw = tuple(cfg["local_map_size"])
    hist, action_dim = tuple(cfg["action_history_shape"])
    gen = torch.Generator(device=device).manual_seed(1000 + agents)
    return {
        "global_map": torch.rand(batch, agents, g_channels, gh, gw, generator=gen, device=device),
        "local_map": torch.rand(batch, agents, l_channels, lh, lw, generator=gen, device=device),
        "action_history": torch.rand(batch, agents, hist, action_dim, generator=gen, device=device),
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    base = HeMACHISSD(**payload["model_config"]).to(device)
    base.load_state_dict(payload["model_state_dict"])
    base.eval()

    variable = VariableAgentHeMACHISSD.from_base_model(
        base,
        variable_mixer_hidden_dim=args.variable_mixer_hidden_dim,
    ).to(device)
    variable.eval()

    original_agents = int(base.agent_count)
    obs = random_observations(base, 1, original_agents, device)
    mask = torch.ones(1, original_agents, dtype=torch.bool, device=device)

    base_state = base.initial_inference_state(batch_size=1, device=device)
    var_state = variable.initial_inference_state(
        batch_size=1,
        agent_count=original_agents,
        device=device,
    )
    with torch.no_grad():
        base_out, _ = base.inference_step(obs, mask, base_state)
        var_out, _ = variable.inference_step(obs, mask, var_state)

    action_diff = float((base_out["actions"] - var_out["actions"]).abs().max())
    common_diff = float((base_out["common_skills"] - var_out["common_skills"]).abs().max())
    task_diff = float((base_out["task_skills"] - var_out["task_skills"]).abs().max())

    print(f"device={device}")
    print(f"reference_agent_count={original_agents}")
    print(f"policy_action_max_diff={action_diff:.9g}")
    print(f"common_skill_max_diff={common_diff:.9g}")
    print(f"task_skill_max_diff={task_diff:.9g}")

    tol = 1e-6
    if max(action_diff, common_diff, task_diff) > tol:
        raise RuntimeError("POLICY IDENTITY CHECK FAILED")
    print("POLICY IDENTITY CHECK: PASS")

    # Smoke-test online recurrent inference and centralized value mixing for A=1..5.
    for agents in args.test_agent_counts:
        obs = random_observations(variable, 1, int(agents), device)
        mask = torch.ones(1, int(agents), dtype=torch.bool, device=device)
        # Deliberately omit agent_count here: inference_step must repair the state.
        state = variable.initial_inference_state(batch_size=1, device=device)
        with torch.no_grad():
            out, _ = variable.inference_step(obs, mask, state)

            features = out["observation_features"].unsqueeze(1)
            valid = mask.unsqueeze(1)
            cfg = variable.config()
            ch = int(cfg["central_map_channels"])
            h, w = tuple(cfg["central_map_size"])
            central = torch.rand(1, 1, ch, h, w, device=device)
            value = variable.total_value(features, central, valid)

        print(
            f"A={agents}: actions={tuple(out['actions'].shape)} "
            f"value={tuple(value.shape)} PASS"
        )

    if args.output_checkpoint is not None:
        output = args.output_checkpoint.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        converted = dict(payload)
        converted.update(
            {
                "format_version": max(2, int(payload.get("format_version", 1))),
                "model_type": "hemac_variable_agent_hissd",
                "model_config": variable.config(),
                "model_state_dict": variable.state_dict(),
                "variable_agent_conversion": variable.conversion_report,
            }
        )
        # The old optimizer has fixed-mixer parameter groups and is not reusable.
        converted.pop("optimizer_state_dict", None)
        torch.save(converted, output)
        print(f"saved={output}")


if __name__ == "__main__":
    main()
