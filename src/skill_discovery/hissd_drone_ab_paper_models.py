"""Shared A/B HiSSD and repository-style MoCo discriminator projection.

HeMAC observation CNN/continuous control replace the original SMAC
entity-decomposer and discrete-action head. The official Discriminator's
2-layer ssl_proj, bilinear W logits, and hard-updated target counterpart
are preserved at the interface level.
"""
from __future__ import annotations
import copy
import torch
from torch import nn
from .hissd_joint_hetero_models_paper import JointHeterogeneousHiSSD


class PaperDroneABHiSSD(JointHeterogeneousHiSSD):
    """Per-agent skills; common shared encoder, task encoder and controller."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # HiSSD HISSDAgent.Discriminator.ssl_proj: Linear -> ReLU -> Linear
        self.moco_projection=nn.Sequential(
            nn.Linear(self.skill_dim,128),nn.ReLU(),
            nn.Linear(128,self.skill_dim))
        self.target_moco_projection=copy.deepcopy(self.moco_projection)
        self.target_moco_projection.requires_grad_(False)
        self.target_moco_projection.eval()
        # HiSSD HISSDAgent.Discriminator.W initialized from uniform [0,1).
        self.moco_W=nn.Parameter(torch.rand(self.skill_dim,self.skill_dim))

    def train(self, mode=True):
        super().train(mode)
        self.target_moco_projection.eval()
        return self

    @torch.no_grad()
    def update_targets(self,tau:float)->None:
        # The repository uses periodic hard sync: tau=1.0 every 80 steps.
        super().update_targets(tau)
        for src,dst in zip(self.moco_projection.parameters(),
                           self.target_moco_projection.parameters(),strict=True):
            dst.lerp_(src, float(tau))


def load_paper_drone_ab_checkpoint(path, device='cpu'):
    """Load an A/B repository-objective checkpoint for later inference/evaluation.

    Older A/B model checkpoint loaders do NOT recognize new MoCo parameters.
    """
    from pathlib import Path
    payload=torch.load(Path(path),map_location=device,weights_only=False)
    if payload.get('model_type')!='hemac_hissd_drone_ab_repository_objectives':
        raise ValueError('Unexpected checkpoint model_type')
    model=PaperDroneABHiSSD(**payload['model_config']).to(device)
    model.load_state_dict(payload['model_state_dict'],strict=True)
    model.eval()
    return model,payload
