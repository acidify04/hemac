#!/usr/bin/env python3
"""HiSSD repository-aligned offline trainer for heterogeneous HeMAC Drone A/B.

Faithfulness boundary:
- Official HISSDLearner update order: Controller imitation+cross-task MoCo,
  Expectile TD Value, TD-weighted world-model Planner; hard target sync.
- Official MoCo logits use a learned bilinear W, same-task different-agent
  positives, and key features sampled from a stored different-task batch.
- Environment adaptations: shared HeMAC CNN/Transformer network; continuous
  normalized actions and MSE imitation instead of SMAC discrete CE; pixel-map
  next-state prediction instead of decomposed SMAC entity reconstruction.
- Official repository contains an unusual expectile SIGN convention. We keep
  its literal V-target convention by default for audit reproducibility.
- This is NOT a byte-for-byte port of HISSDAgent: the HeMAC architecture,
  per-agent state decomposition and continuous output necessarily differ.
- This script never imports historical Adapter trainers or adapter models.

Place beside hissd_joint_hetero_models_paper.py in src/skill_discovery/.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm=None

PROJECT_ROOT=Path(__file__).resolve().parents[2]
PROJECT_SRC=PROJECT_ROOT / 'src'
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0,str(PROJECT_SRC))

from skill_discovery.hissd_drone_ab_paper_models import PaperDroneABHiSSD

ROLE_ORDER=('observer','drone')
ROLE_ALIAS={'observer':'A','drone':'B'}

def _move_observations(
    observations: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        name: observations[name].to(device, non_blocking=True)
        for name in ("global_map", "local_map", "action_history")
    }

def prepare_joint_batch(
    raw_batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Convert one collated DnOm batch into joint heterogeneous tensors.

    HeMAC ordering is Observer(s) first, then Drone(s), which matches the joint
    model's canonical ROLE_ORDER.
    """
    for role in ROLE_ORDER:
        if role not in raw_batch:
            raise KeyError(
                f"Joint batch is missing role {role!r}. "
                "Use include_observer=True when creating the loader."
            )

    observer_actions = raw_batch["observer"]["actions"].to(
        device, non_blocking=True
    )
    drone_actions = raw_batch["drone"]["actions"].to(
        device, non_blocking=True
    )
    observer_count = int(observer_actions.shape[-2])
    drone_count = int(drone_actions.shape[-2])

    full_mask = raw_batch["agent_mask"].to(
        device, non_blocking=True
    ).bool()
    if int(full_mask.shape[-1]) != observer_count + drone_count:
        raise RuntimeError(
            "Joint agent_mask does not match Observer+Drone counts: "
            f"mask={full_mask.shape[-1]}, O={observer_count}, D={drone_count}."
        )

    filled = raw_batch["filled"].to(
        device, non_blocking=True
    ).bool()
    if filled.ndim == full_mask.ndim:
        if filled.shape[-1] != 1:
            raise ValueError(
                f"Expected filled[...,1], got {tuple(filled.shape)}"
            )
        filled = filled.squeeze(-1)
    if filled.shape != full_mask.shape[:-1]:
        raise ValueError(
            f"filled shape {tuple(filled.shape)} does not match "
            f"agent_mask prefix {tuple(full_mask.shape[:-1])}."
        )

    observer_mask = (
        full_mask[..., :observer_count] & filled.unsqueeze(-1)
    )
    drone_mask = (
        full_mask[
            ..., observer_count : observer_count + drone_count
        ]
        & filled.unsqueeze(-1)
    )
    joint_mask = torch.cat(
        (observer_mask, drone_mask),
        dim=-1,
    )

    terminated = raw_batch["terminated"].to(
        device, non_blocking=True
    ).bool()
    truncated = raw_batch["truncated"].to(
        device, non_blocking=True
    ).bool()
    if terminated.ndim == filled.ndim + 1:
        terminated = terminated.squeeze(-1)
    if truncated.ndim == filled.ndim + 1:
        truncated = truncated.squeeze(-1)

    if "team_reward" in raw_batch:
        team_reward = raw_batch["team_reward"].to(
            device, non_blocking=True
        )
    elif "drone_task_reward" in raw_batch:
        team_reward = raw_batch["drone_task_reward"].to(
            device, non_blocking=True
        )
    else:
        raise KeyError(
            "Dataset contains neither team_reward nor drone_task_reward."
        )
    if team_reward.ndim == filled.ndim + 1:
        team_reward = team_reward.squeeze(-1)

    batch = {
        "observations_by_role": {
            role: _move_observations(
                raw_batch[role]["observations"], device
            )
            for role in ROLE_ORDER
        },
        "next_observations_by_role": {
            role: _move_observations(
                raw_batch[role]["next_observations"], device
            )
            for role in ROLE_ORDER
        },
        "actions_by_role": {
            "observer": observer_actions,
            "drone": drone_actions,
        },
        "valid_agents_by_role": {
            "observer": observer_mask,
            "drone": drone_mask,
        },
        "valid_agents": joint_mask,
        "valid_steps": filled,
        "central_map": raw_batch["global_state"]["central_map"].to(
            device, non_blocking=True
        ),
        "next_central_map": raw_batch["next_global_state"][
            "central_map"
        ].to(device, non_blocking=True),
        "team_reward": team_reward,
        "task_descriptor": raw_batch["task_descriptor"].to(
            device, non_blocking=True
        ),
        "task_descriptor_available": raw_batch[
            "task_descriptor_available"
        ].to(device, non_blocking=True).bool(),
        "task_id": raw_batch["task_id"].to(
            device, non_blocking=True
        ),
        "task_supervision": (
            raw_batch["window_start"].to(
                device, non_blocking=True
            )
            == 0
        ),
        "done": terminated | truncated,
        "role_counts": {
            "observer": observer_count,
            "drone": drone_count,
        },
    }
    return batch

def load_episode(path: Path):
    # mmap minimizes disk reads during upfront trajectory indexing.
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, RuntimeError):
        return torch.load(path, map_location="cpu", weights_only=False)

@dataclass(frozen=True)
class EpisodeSpec:
    path: Path
    length: int
    seed: int
    category: str
    action_scales: tuple[float, float]
    difficulty: int

def validate_and_describe(path: Path) -> EpisodeSpec:
    ep = load_episode(path)
    meta = ep.get("metadata", {})
    aliases = meta.get("role_aliases")
    if aliases != {"observer": "drone_A", "drone": "drone_B"}:
        raise ValueError(f"Not a drone A/B aliased episode: {path}: {aliases}")
    cfg = meta.get("environment_config", {})
    if cfg.get("n_observers") != 0 or cfg.get("drone_only_task_mode") != "maximize_coverage":
        raise ValueError(f"Dataset environment is not drone-only maximize_coverage: {path}")
    if ep["observer"]["actions"].shape[1] < 1 or ep["drone"]["actions"].shape[1] < 1:
        raise ValueError(f"Episode missing A/B agents: {path}")
    drone_cfg = cfg.get("drone_config", {})
    types = [str(x).upper() for x in drone_cfg.get("drone_types", [])]
    assert len(types) == ep["agent_mask"].shape[-1], (path, types)
    # Critical: on-disk agent ordering must match A-first then B for mask slicing.
    expected_order = ep["observer"]["agent_ids"] + ep["drone"]["agent_ids"]
    if expected_order != meta.get("agent_order"):
        raise ValueError(f"A/B agent_order is not concatenated A then B: {path}")
    type_cfgs = drone_cfg.get("drone_type_configs", {})
    def speed(t):
        return float(type_cfgs.get(t, type_cfgs.get(t.lower(), {})).get(
            "max_speed", drone_cfg.get("drone_max_speed", 16.0)))
    scales = (speed("A"), speed("B"))
    if min(scales) <= 0:
        raise ValueError(f"Invalid action scale: {scales}")
    T = ep["observer"]["actions"].shape[0]
    if ep["drone"]["actions"].shape[0] != T or ep["global_state"]["central_map"].shape[0] != T + 1:
        raise ValueError(f"Misaligned A/B joint time axes: {path}")
    if T < 1:
        raise ValueError(f"Empty episode: {path}")
    return EpisodeSpec(path, T, int(meta.get("seed", -1)), str(ep["outcome"]["category"]), scales,
                       int(meta.get("difficulty", -1)))

class DroneABWindows(Dataset):
    """Episode-disjoint, time-aligned batches preserving original HiSSD input keys."""
    def __init__(self, episodes: list[EpisodeSpec], seq_len: int, stride: int, cache_size: int,
                 task_ids: dict[int, int] | None = None):
        self.episodes = episodes
        self.task_ids = task_ids or {episodes[0].difficulty: 0}
        self.seq_len = seq_len
        self.index = [(i, start) for i, ep in enumerate(episodes) for start in range(0, ep.length, stride)]
        self._cache: OrderedDict[Path, dict[str, Any]] = OrderedDict()
        self.cache_size = max(cache_size, 1)
        if not self.index:
            raise ValueError("No sequence windows found")

    def __len__(self):
        return len(self.index)

    def _get(self, path):
        if path not in self._cache:
            self._cache[path] = load_episode(path)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        self._cache.move_to_end(path)
        return self._cache[path]

    def __getitem__(self, ix):
        episode_index, start = self.index[ix]
        ep_info = self.episodes[episode_index]
        ep = self._get(ep_info.path)
        end = min(start + self.seq_len, ep_info.length)
        valid = end - start

        def slice_pad(tensor: torch.Tensor, offset: int = 0, pad_value=0):
            x = tensor[start + offset:end + offset].to(dtype=(
                torch.float32 if tensor.is_floating_point() else tensor.dtype))
            if valid == self.seq_len:
                return x.contiguous()
            padding = torch.full((self.seq_len - valid, *x.shape[1:]), pad_value, dtype=x.dtype)
            return torch.cat([x, padding], dim=0).contiguous()

        result: dict[str, Any] = {}
        for role, scale in zip(ROLE_ORDER, ep_info.action_scales):
            role_data = ep[role]
            obs = role_data["observations"]
            current_obs = {k: slice_pad(v) for k, v in obs.items()
                           if k in ("global_map", "local_map", "action_history")}
            next_obs = {k: slice_pad(v, 1) for k, v in obs.items()
                        if k in ("global_map", "local_map", "action_history")}
            # The shared encoder expects physical action histories in the same
            # normalized [-1, 1] convention as its continuous target actions.
            # The original collection file stores history in environment units.
            for mapped_obs in (current_obs, next_obs):
                mapped_obs["action_history"] = (
                    mapped_obs["action_history"] / scale
                ).clamp(-1, 1)
            result[role] = {
                "observations": current_obs,
                "next_observations": next_obs,
                "actions": (slice_pad(role_data["actions"]) / scale).clamp(-1, 1),
            }
        result["global_state"] = {"central_map": slice_pad(ep["global_state"]["central_map"])}
        result["next_global_state"] = {"central_map": slice_pad(ep["global_state"]["central_map"], 1)}
        for key in ("team_reward", "agent_mask", "terminated", "truncated"):
            result[key] = slice_pad(ep[key])
        filled = torch.zeros((self.seq_len, 1), dtype=torch.bool)
        filled[:valid] = True
        result["filled"] = filled
        result["window_start"] = torch.tensor(start, dtype=torch.long)
        result["task_id"] = torch.tensor(self.task_ids[ep_info.difficulty], dtype=torch.long)
        result["task_descriptor"] = ep["task_descriptor"].to(torch.float32).clone()
        result["task_descriptor_available"] = torch.tensor(True, dtype=torch.bool)
        return result

def split_episodes(all_eps: list[EpisodeSpec], val_fraction: float, seed: int):
    if len(all_eps) < 2:
        raise ValueError("Need at least 2 episodes for episode-disjoint train/validation")
    train, val = [], []
    by_difficulty: dict[int, list[EpisodeSpec]] = {}
    for ep in all_eps:
        by_difficulty.setdefault(ep.difficulty, []).append(ep)
    for diff, items in sorted(by_difficulty.items()):
        if len(items) < 2:
            raise ValueError(f"Need >=2 episodes for difficulty {diff} (found {len(items)})")
        random.Random(seed + diff).shuffle(items)
        val_n = min(len(items) - 1, max(1, round(len(items) * val_fraction)))
        train.extend(items[val_n:])
        val.extend(items[:val_n])
    return train, val

def role_config_from_sample(sample: Mapping[str, Any], role: str) -> dict[str, Any]:
    """Build the same role I/O architecture used by the existing BC-initialized run.

    The current multisource BC policies use two 96-unit hidden layers with ReLU.
    All observation/action dimensions are inferred from the synchronized dataset,
    so no BC checkpoint is consulted.
    """
    payload = sample[role]
    observations = payload["observations"]
    actions = payload["actions"]
    return {
        "global_map_channels": int(observations["global_map"].shape[-3]),
        "local_map_channels": int(observations["local_map"].shape[-3]),
        "global_map_size": tuple(int(x) for x in observations["global_map"].shape[-2:]),
        "local_map_size": tuple(int(x) for x in observations["local_map"].shape[-2:]),
        "action_history_shape": tuple(int(x) for x in observations["action_history"].shape[-2:]),
        "hidden_sizes": (96, 96),
        "activation": "relu",
        "action_dim": int(actions.shape[-1]),
    }


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--difficulty',type=int,default=None,help='Optional single difficulty; use multiple for MoCo')
    p.add_argument('--allow-single-task-no-moco',action='store_true',help='Ablation only; skip cross-task SSL')
    p.add_argument('--epochs',type=int,default=120)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--sequence-length',type=int,default=128)
    p.add_argument('--stride',type=int,default=128)
    p.add_argument('--val-fraction',type=float,default=0.1)
    p.add_argument('--dataset-cache-size',type=int,default=12)
    p.add_argument('--num-workers',type=int,default=0)
    p.add_argument('--expert-only',action='store_true')
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--learning-rate',type=float,default=1e-4)
    p.add_argument('--weight-decay',type=float,default=1e-4)
    p.add_argument('--hidden-dim',type=int,default=64)
    p.add_argument('--skill-dim',type=int,default=64)
    p.add_argument('--transformer-heads',type=int,default=1)
    p.add_argument('--gamma',type=float,default=0.99)
    p.add_argument('--expectile',type=float,default=0.9)
    p.add_argument('--expectile-sign',choices=['repository','standard'],default='repository')
    p.add_argument('--beta',type=float,default=0.05,help='SSL weight as in official SMAC yaml')
    p.add_argument('--td-weight',type=float,default=10.0)
    p.add_argument('--planner-mask-terminal-bootstrap',action='store_true',
      help='NOT repository default: suppress planner bootstrap on done transitions')
    p.add_argument('--reward-scale',type=float,default=1.0,help='Environment adapter; official implementation uses raw rewards')
    p.add_argument('--target-update-interval',type=int,default=80)
    p.add_argument('--gradient-clip',type=float,default=10.0)
    p.add_argument('--moco-negatives',type=int,default=128)
    p.add_argument('--batch-count-per-task',type=int,default=None,help='Optional cap on train batches from each task per epoch')
    p.add_argument('--max-train-batches',type=int,default=None,help='Total batch cap per epoch for smoke tests')
    p.add_argument('--max-val-batches',type=int,default=None)
    p.add_argument('--resume-checkpoint',type=Path)
    p.add_argument('--no-progress',action='store_true')
    return p.parse_args()


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def make_model(sample,args):
    configs={r:role_config_from_sample(sample,r) for r in ROLE_ORDER}
    cen=sample['global_state']['central_map']
    return PaperDroneABHiSSD(
        role_configs=configs,central_map_channels=int(cen.shape[-3]),
        central_map_size=tuple(map(int,cen.shape[-2:])),
        reference_observer_count=int(sample['observer']['actions'].shape[-2]),
        reference_drone_count=int(sample['drone']['actions'].shape[-2]),
        hidden_dim=args.hidden_dim,skill_dim=args.skill_dim,
        transformer_heads=args.transformer_heads, task_descriptor_dim=0,
        task_prior_count=0,task_dropout=0.0,
        task_feature_deltas=False,normalize_task_context=False,
        task_running_statistics=False,task_action_residual=False,
        skill_structure='split',
    )


def encode_online(model,batch):
    return model.encode_joint_observations(batch['observations_by_role'],batch['valid_agents_by_role'])


@torch.no_grad()
def encode_target(model,batch):
    features,mask,_=model.encode_joint_observations(batch['observations_by_role'],batch['valid_agents_by_role'],target=True)
    _,keys=model.target_task_skill_encoder(features,mask)
    keys=model.target_moco_projection(keys)
    return keys.detach(),mask


def choose_pair(query,valid_mask,valid_steps):
    """Pairs from distinct agents in the SAME task at the SAME valid time."""
    choices=torch.nonzero(valid_mask & valid_steps.unsqueeze(-1),as_tuple=False)
    if choices.numel()==0:return None
    # select sample/time with >=2 valid drones; mixed sensor types allowed
    ok=(valid_mask & valid_steps.unsqueeze(-1)).sum(-1)>=2
    times=torch.nonzero(ok,as_tuple=False)
    if times.numel()==0:return None
    b,t=times[torch.randint(len(times),(1,),device=times.device).item()].tolist()
    agents=torch.nonzero(valid_mask[b,t],as_tuple=False).flatten()
    idx=torch.randperm(agents.numel(),device=agents.device)[:2]
    i,j=agents[idx[0]],agents[idx[1]]
    return query[b,t,i].unsqueeze(0),query[b,t,j].unsqueeze(0)


@torch.no_grad()
def get_negative_keys(model,stored_batches,current_task,max_keys):
    negatives=[]
    for other_task,other in stored_batches.items():
        if other_task==current_task:continue
        key,mask=encode_target(model,other)
        v=other['valid_steps'].unsqueeze(-1)&mask
        if bool(v.any()):negatives.append(key[v])
    if not negatives:return None
    z=torch.cat(negatives,dim=0)
    if z.shape[0]>max_keys:
        z=z[torch.randperm(z.shape[0],device=z.device)[:max_keys]]
    return z.detach()


def moco_repo_loss(model,query,positive,negatives):
    """Literal HISSDAgent.Discriminator.compute_logits + HISSDLearner.contrastive_loss.

    Includes auxiliary detached negative rows and zero-column labels as in
    official HISSDLearner; this is not the more usual 1-vs-N InfoNCE formula.
    """
    source=torch.cat((query,negatives.detach()),dim=0)
    targets=torch.cat((positive.detach(),negatives.detach()),dim=0)
    logits=source@(model.moco_W@targets.T)
    logits=logits-logits.max(dim=1,keepdim=True).values.detach()
    labels=torch.zeros(logits.shape[0],device=logits.device,dtype=torch.long)
    return F.cross_entropy(logits,labels)


def controller_objective(model,batch,args,stored_batches,task_id):
    feats,mask,slices=encode_online(model,batch)
    outs=model.infer_joint_skills(feats,mask)
    pred=model.decode_joint_actions(feats,outs['common_skills'],outs['task_skills'],slices)
    total_error=feats.new_zeros(())
    total_count=feats.new_zeros(())
    type_mse={}
    for role in ROLE_ORDER:
        acts=batch['actions_by_role'][role]
        diff=(pred[role]-acts).square().mean(-1)
        m=batch['valid_agents_by_role'][role].float()
        total_error=total_error+(diff*m).sum()
        total_count=total_count+m.sum()
        type_mse[ROLE_ALIAS[role]]=((diff*m).sum()/m.sum().clamp_min(1)).detach()
    imitation=total_error/total_count.clamp_min(1)
    ssl=imitation.new_zeros(())
    used_neg=0
    if args.beta>0:
        projected=model.moco_projection(outs['contrastive_skills'])
        pair=choose_pair(projected,mask,batch['valid_steps'])
        neg=get_negative_keys(model,stored_batches,task_id,args.moco_negatives)
        if pair is not None and neg is not None:
            ssl=moco_repo_loss(model,pair[0],pair[1],neg)
            used_neg=int(neg.shape[0])
    loss=imitation+args.beta*ssl
    return loss,dict(action_mse=float(imitation.detach()),moco_loss=float(ssl.detach()),
        ssl_negatives=used_neg,action_A=float(type_mse['A']),action_B=float(type_mse['B']),
        controller_total=float(loss.detach()))


def value_objective(model,batch,args):
    f,mask,_=encode_online(model,batch)
    current=model.total_value(f,batch['central_map'],mask.float()).squeeze(-1)
    with torch.no_grad():
        next_f,next_mask,_=model.encode_joint_observations(batch['next_observations_by_role'],batch['valid_agents_by_role'],target=True)
        next_v=model.total_value(next_f,batch['next_central_map'],next_mask.float(),target=True).squeeze(-1)
        target=batch['team_reward']/args.reward_scale +args.gamma*(~batch['done']).float()*next_v
    # Official implementation uses error=V-target and weight abs(eps - (error<0)),
    # even though this is the REVERSE sign of the conventional optimistic IQL expectile.
    error=current-target
    if args.expectile_sign=='repository':
        weight=(args.expectile-(error.detach()<0).float()).abs()
    else:
        weight=(args.expectile-((target-current).detach()<0).float()).abs()
    m=batch['valid_steps'].float()
    loss=(weight*error.square()*m).sum()/m.sum().clamp_min(1)
    return loss,dict(value_loss=float(loss.detach()),value_td=float(((target-current).detach()*m).sum()/m.sum().clamp_min(1)))


def planner_objective(model,batch,args):
    # Official planner fits next observations by forward dynamics and weights
    # the entire batch prediction by exp(td_weight * mean(TD residual)).
    # HeMAC next-map/latent reconstruction substitutes for SMAC entities.
    with torch.no_grad():
        feats,mask,_=encode_online(model,batch)
        target_next,_,_=model.encode_joint_observations(batch['next_observations_by_role'],batch['valid_agents_by_role'],target=True)
    common=model.common_skill_encoder(feats.detach(),mask)
    prediction_map,prediction_agents=model.forward_predictor(common,mask.float())
    map_error=(prediction_map-batch['next_central_map']).square().mean(dim=(-1,-2,-3))
    local_squared=(prediction_agents-target_next.detach()).square().mean(dim=-1)
    local_error=(local_squared*mask.float()).sum(-1)/mask.float().sum(-1).clamp_min(1)
    step_mask=batch['valid_steps'].float()
    prediction_loss=((map_error+local_error)*step_mask).sum()/step_mask.sum().clamp_min(1)
    with torch.no_grad():
        current_val=model.total_value(feats,batch['central_map'],mask.float()).squeeze(-1)
        # Frozen target mixed value on PREDICTED features, as in official forward_value_skill
        pred_val=model.total_value(prediction_agents.detach(),prediction_map.detach(),mask.float(),target=True).squeeze(-1)
        # Official train_planner uses bootstrap without a terminal factor.
        # Optional flag provides the standard terminal-aware modification.
        bootstrap=(~batch['done']).float() if args.planner_mask_terminal_bootstrap else 1.0
        td=batch['team_reward']/args.reward_scale + args.gamma*bootstrap*pred_val-current_val
        mean_td=(td*step_mask).sum()/step_mask.sum().clamp_min(1)
        # Preserve repository: upper clamp only; lower clamp for stability of exponential
        weight=mean_td.mul(args.td_weight).clamp(max=math.log(100.0)).exp().detach()
    loss=weight*prediction_loss
    return loss,dict(planner_loss=float(loss.detach()),prediction_loss=float(prediction_loss.detach()),
        planner_td=float(mean_td.detach()),planner_weight=float(weight.detach()),
        planner_map_mse=float(((map_error*step_mask).sum()/step_mask.sum().clamp_min(1)).detach()),
        planner_local_mse=float(((local_error*step_mask).sum()/step_mask.sum().clamp_min(1)).detach()))


def opt_step(model,optimizer,loss,grad_clip):
    if not torch.isfinite(loss.detach()).all():
        raise FloatingPointError(f'Nonfinite objective: {loss.detach().item()}')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad=torch.nn.utils.clip_grad_norm_(model.parameters(),grad_clip)
    if not torch.isfinite(grad):raise FloatingPointError(f'Nonfinite grad norm: {grad}')
    optimizer.step()
    return float(grad.detach())


def clone_offline(batch):
    # Preserve all tensors used by the momentum key-encoder; detach to avoid
    # holding computation graphs across batches (one per task).
    return {'observations_by_role':{role:{key:x.detach().clone() for key,x in obs.items()}
              for role,obs in batch['observations_by_role'].items()},
            'valid_agents_by_role':{role:m.detach().clone() for role,m in batch['valid_agents_by_role'].items()},
            'valid_steps':batch['valid_steps'].detach().clone()}


def iter_task_batches(loaders,epoch,seed,limit_per_task):
    order=list(loaders)
    r=random.Random(seed+epoch)
    r.shuffle(order)
    counts={k:0 for k in order}
    its={k:iter(loaders[k]) for k in order}
    # Equal minibatch counts per source task, without reusing other task's samples.
    shortest=min(len(loader) for loader in loaders.values())
    sizes={k:(shortest if limit_per_task is None else min(shortest,limit_per_task)) for k in order}
    while any(counts[k]<sizes[k] for k in order):
        for task in order:
            if counts[task]>=sizes[task]:continue
            try:b=next(its[task])
            except StopIteration:continue
            counts[task]+=1
            yield task,b


def run_train_epoch(model,optimizer,train_loaders,args,device,epoch,global_step,last_batches):
    model.train()
    totals=defaultdict(float); n=0
    iterator=iter_task_batches(train_loaders,epoch,args.seed,args.batch_count_per_task)
    for task,raw in iterator:
        if args.max_train_batches is not None and n>=args.max_train_batches:break
        batch=prepare_joint_batch(raw,device)
        tid=int(batch['task_id'][0].item())
        # Official sequence: controller(+MoCo), value, planner; each gets optimizer step.
        controller,cc=controller_objective(model,batch,args,last_batches,tid)
        cc['controller_grad_norm']=opt_step(model,optimizer,controller,args.gradient_clip)
        # last batch from current task enters negative memory AFTER the controller step.
        last_batches[tid]=clone_offline(batch)
        value,vc=value_objective(model,batch,args)
        vc['value_grad_norm']=opt_step(model,optimizer,value,args.gradient_clip)
        planner,pc=planner_objective(model,batch,args)
        pc['planner_grad_norm']=opt_step(model,optimizer,planner,args.gradient_clip)
        global_step+=1
        if global_step%args.target_update_interval==0:
            model.update_targets(1.0)  # hard sync as official target_mac.load_state
        for k,v in {**cc,**vc,**pc}.items():totals[k]+=float(v)
        n+=1
    if n==0:raise RuntimeError('No training batches')
    return {k:v/n for k,v in totals.items()},global_step


@torch.no_grad()
def validate(model,loaders,args,device):
    model.eval()
    acc=defaultdict(float);n=0
    for task,loader in loaders.items():
        for raw in loader:
            if args.max_val_batches is not None and n>=args.max_val_batches:break
            batch=prepare_joint_batch(raw,device)
            # No SSL at validation: distinguish BC generalization from negative-memory state.
            cc_args=copy.copy(args);cc_args.beta=0.
            _,cm=controller_objective(model,batch,cc_args,{},int(batch['task_id'][0]))
            _,vm=value_objective(model,batch,args)
            _,pm=planner_objective(model,batch,args)
            for k,v in {**cm,**vm,**pm}.items():acc[k]+=float(v)
            n+=1
    if not n:raise RuntimeError('No validation batches')
    return {k:v/n for k,v in acc.items()}


def save(path,model,optimizer,args,epoch,step,score,split,train,val):
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'format_version':2,'model_type':'hemac_hissd_drone_ab_repository_objectives',
        'model_config':model.config(),'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),'epoch':epoch,'global_step':step,
        'task_configuration':split['task_ids'],
        'training_args':vars(args),'train_metrics':train,'validation_metrics':val,
        'best_validation_mse':score,'architecture':{'shared_encoder':True,
        'per_agent_common_skill':True,'shared_controller':True,'adapter_count':0},
        'algorithm':{'moco':'repository_bilinear_cross_task','controller':'continuous_mse_plus_beta_moco',
        'expectile_sign':args.expectile_sign,'planner':'mean_td_exponential_batch_weight',
        'target_update':'hard_every_n_updates','world_model':'HeMAC_map_and_feature',
        'planner_terminal_bootstrap_mask':args.planner_mask_terminal_bootstrap},
        'split':split,'initialization':'scratch','bc_initialization':None},path)


def main():
    args=arguments()
    if not 0<args.val_fraction<1:raise ValueError('--val-fraction must be in (0,1)')
    if args.beta<0 or args.target_update_interval<=0 or args.moco_negatives<=0:raise ValueError('invalid SSL/target parameters')
    if args.reward_scale<=0 or args.learning_rate<=0 or not 0<args.expectile<1:raise ValueError('invalid scale/lr/expectile')
    if min(args.epochs,args.batch_size,args.sequence_length,args.stride) <= 0:
        raise ValueError('epochs,batch_size,sequence_length,stride must be positive')
    seed_everything(args.seed)
    device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device))
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA is not available')
    root=args.data_root.resolve()
    if args.difficulty is None:paths=sorted(root.glob('difficulty_*/*/*.pt'))
    else:paths=sorted((root/f'difficulty_{args.difficulty:02d}').glob('*/*.pt'))
    if not paths:raise FileNotFoundError(f'No dataset episodes under {root}')
    eps=[validate_and_describe(p) for p in paths]
    if args.expert_only:eps=[e for e in eps if e.category=='success']
    if not eps:raise ValueError('No episodes after selection')
    scales={e.action_scales for e in eps}
    if len(scales)!=1:raise ValueError(f'Inconsistent A/B action scales: {scales}')
    tasks=sorted({e.difficulty for e in eps})
    if len(tasks)<2 and args.beta>0:
        if not args.allow_single_task_no_moco:
            raise ValueError('Repository MoCo needs >=2 source tasks/difficulties for cross-task negatives. Collect D1+D2 or pass --allow-single-task-no-moco for a non-MoCo ablation.')
        print('[WARN] Single task: disabling MoCo. This is NOT full HiSSD training.',flush=True)
        args.beta=0.0
    train_eps,val_eps=split_episodes(eps,args.val_fraction,args.seed)
    task_ids={d:i for i,d in enumerate(tasks)}
    kw=dict(batch_size=args.batch_size,num_workers=args.num_workers,
        pin_memory=device.type=='cuda',persistent_workers=args.num_workers>0)
    train_loaders={};val_loaders={}
    for difficulty in tasks:
        tr=DroneABWindows([e for e in train_eps if e.difficulty==difficulty],args.sequence_length,args.stride,args.dataset_cache_size,task_ids)
        va=DroneABWindows([e for e in val_eps if e.difficulty==difficulty],args.sequence_length,args.stride,args.dataset_cache_size,task_ids)
        if len(tr)<args.batch_size:raise ValueError(f'Difficulty {difficulty} has only {len(tr)} windows for batch size {args.batch_size}')
        train_loaders[difficulty]=DataLoader(tr,shuffle=True,drop_last=True,**kw)
        val_loaders[difficulty]=DataLoader(va,shuffle=False,drop_last=False,**kw)
    sample=next(iter(train_loaders.values())).dataset[0]
    model=make_model(sample,args).to(device)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                 lr=args.learning_rate,weight_decay=args.weight_decay)
    start_epoch=1;global_step=0;best=math.inf
    if args.resume_checkpoint:
        cp=torch.load(args.resume_checkpoint,map_location=device,weights_only=False)
        if cp.get('model_type')!='hemac_hissd_drone_ab_repository_objectives':
            raise ValueError('Checkpoint is NOT from the repository-objective HiSSD trainer')
        if cp['model_config']!=model.config():raise ValueError('Checkpoint model configuration differs')
        model.load_state_dict(cp['model_state_dict'])
        optimizer.load_state_dict(cp['optimizer_state_dict'])
        start_epoch=int(cp['epoch'])+1;global_step=int(cp['global_step']);best=float(cp['best_validation_mse'])
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    split={'train_episodes':[str(e.path.relative_to(root)) for e in train_eps],
           'validation_episodes':[str(e.path.relative_to(root)) for e in val_eps],
           'task_ids':task_ids,'seed':args.seed,'action_scales':list(next(iter(scales))),
           'note':'Episode-disjoint split, no adapter, no BC init'}
    splitpath=output/'data_split.json'
    if args.resume_checkpoint and splitpath.exists():
        saved=json.loads(splitpath.read_text())
        if set(saved['train_episodes'])!=set(split['train_episodes']) or set(saved['validation_episodes'])!=set(split['validation_episodes']):
            raise ValueError('Dataset split changed since checkpoint; refusing resume')
    splitpath.write_text(json.dumps(split,indent=2))
    print(f'[HiSSD paper objectives] device={device}, tasks={tasks}, train={len(train_eps)} val={len(val_eps)}, MoCo={args.beta>0}',flush=True)
    print(f'[Repository-pattern stages: Controller+MoCo -> Expectile Value -> TD-weighted Planner -> target hard sync/{args.target_update_interval}]',flush=True)
    print('[ADAPTATIONS] Continuous MSE actions; HeMAC map/feature predictor & variable-agent mixer; not a byte-identical SMAC model',flush=True)
    history=output/'train_history.jsonl'
    epochs=range(start_epoch,args.epochs+1)
    if tqdm is not None and not args.no_progress:
        epochs=tqdm(epochs,desc='HiSSD A/B repository losses',unit='epoch',dynamic_ncols=True)
    last_batches={}
    for ep in epochs:
        train,global_step=run_train_epoch(model,optimizer,train_loaders,args,device,ep,global_step,last_batches)
        val=validate(model,val_loaders,args,device)
        score=val['action_mse']
        if not math.isfinite(score):raise FloatingPointError('Non-finite validation action MSE')
        record={'epoch':ep,'global_step':global_step,'train':train,'validation':val}
        with history.open('a') as f:f.write(json.dumps(record)+'\n')
        if score<best:best=score;save(output/'hissd_drone_ab_paper_best.pt',model,optimizer,args,ep,global_step,best,split,train,val)
        save(output/'hissd_drone_ab_paper_last.pt',model,optimizer,args,ep,global_step,best,split,train,val)
        line=f"epoch={ep}/{args.epochs} steps={global_step} bc={train['action_mse']:.5f} ssl={train['moco_loss']:.5f} neg={train['ssl_negatives']:.1f} V={train['value_loss']:.5f} planner={train['planner_loss']:.5f} td_weight={train['planner_weight']:.3f} val_bc={score:.5f}"
        if tqdm is not None and not args.no_progress:tqdm.write(line);epochs.set_postfix(bc=f"{train['action_mse']:.4f}",val=f'{score:.4f}')
        else:print(line,flush=True)
    print(f'Done. Best validation BC={best:.6f}; directory={output}',flush=True)

if __name__=='__main__':main()
