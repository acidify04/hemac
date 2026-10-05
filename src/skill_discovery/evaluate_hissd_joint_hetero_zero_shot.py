#!/usr/bin/env python3
"""Canonical zero-shot eval for joint heterogeneous HiSSD WITHOUT adapter."""
from __future__ import annotations
import argparse, copy, json, math, sys
from pathlib import Path
import numpy as np
import torch
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from hemac import HeMAC_v0
from hemac.curriculum_config import OBSTACLE_CURRICULUM_LEVELS
from skill_discovery.collect_offline_data import agent_found_goal, build_collection_env_config, convert_observation, get_core_env
from skill_discovery.hissd_joint_hetero_models import load_joint_heterogeneous_hissd_checkpoint

DRONE_START_POSITIONS = {
    2: [[130.,850.,5.],[170.,850.,5.]],
    3: [[130.,870.,5.],[170.,870.,5.],[150.,830.,5.]],
    4: [[130.,870.,5.],[130.,830.,5.],[170.,870.,5.],[170.,830.,5.]],
    5: [[130.,870.,5.],[170.,870.,5.],[110.,830.,5.],[150.,830.,5.],[190.,830.,5.]],
}
DIFFICULTY_1_CONFIG = {
    "min_obstacles":3,"max_obstacles":4,"obstacle_min_speed":1,"obstacle_max_speed":3,
    "n_static_obstacles":2,"goal_min_base_distance":475.0,"goal_max_base_distance":600.0,
}

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--env-template-data-root",type=Path,required=True)
    p.add_argument("--n-drones",type=int,required=True); p.add_argument("--n-observers",type=int,required=True)
    p.add_argument("--difficulty",type=int,default=1); p.add_argument("--episodes",type=int,default=200)
    p.add_argument("--seed-base",type=int,default=100_000_000)
    p.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    p.add_argument("--output-json",type=Path,required=True); p.add_argument("--quiet",action="store_true")
    return p.parse_args()

def resolve_device(name):
    if name=="auto": name="cuda" if torch.cuda.is_available() else "cpu"
    if name=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)

def find_template_episode(root):
    root=root.expanduser().resolve(); paths=sorted(root.glob("difficulty_*/*/*.pt")) or sorted(root.rglob("*.pt"))
    for path in paths:
        payload=torch.load(path,map_location="cpu",weights_only=False,mmap=True)
        cfg=payload.get("metadata",{}).get("environment_config")
        if isinstance(cfg,dict) and cfg: return path
    raise RuntimeError(f"No episode with environment_config under {root}")

def load_env_template(root):
    path=find_template_episode(root); payload=torch.load(path,map_location="cpu",weights_only=False,mmap=True)
    return path,dict(payload["metadata"]["environment_config"])

def build_target_config(template,difficulty,n_drones,n_observers):
    if n_drones not in DRONE_START_POSITIONS: raise ValueError(f"unsupported n_drones={n_drones}")
    config=copy.deepcopy(build_collection_env_config(copy.deepcopy(template),difficulty))
    if int(difficulty)==1: config.update(DIFFICULTY_1_CONFIG)
    config.update(n_drones=int(n_drones),n_observers=int(n_observers),n_provisioners=0,render_mode=None,log_step_rewards=False)
    dc=copy.deepcopy(config.get("drone_config") or {}); dc["drones_starting_pos"]=copy.deepcopy(DRONE_START_POSITIONS[int(n_drones)])
    config["drone_config"]=dc; return config

def drone_scale(config): return float((config.get("drone_config") or {}).get("drone_max_speed",25.0))
def observer_scale(config): return float(config.get("observer_speed",10.0))
def observation_batch(env,ids,role,scale,device):
    xs=[convert_observation(env.observe(a),role) for a in ids]
    return {
        "global_map":torch.from_numpy(np.stack([x["global_map"] for x in xs])).unsqueeze(0).to(device=device,dtype=torch.float32),
        "local_map":torch.from_numpy(np.stack([x["local_map"] for x in xs])).unsqueeze(0).to(device=device,dtype=torch.float32),
        "action_history":torch.from_numpy(np.stack([x["action_history"] for x in xs])).unsqueeze(0).to(device=device,dtype=torch.float32)/float(scale),
    }
def check_schema(obs,model,role):
    cfg=model.role_configs[role]
    expected={"global_map":(int(cfg["global_map_channels"]),*tuple(cfg["global_map_size"])),"local_map":(int(cfg["local_map_channels"]),*tuple(cfg["local_map_size"])),"action_history":tuple(cfg["action_history_shape"])}
    for k,tail in expected.items():
        actual=tuple(obs[k].shape[-len(tail):])
        if actual!=tuple(tail): raise ValueError(f"{role} {k} mismatch: {actual} vs {tail}")
def to_env_actions(normalized,ids,env,scale):
    out={}
    for i,aid in enumerate(ids):
        space=env.action_space(aid); a=normalized[i].detach().cpu().numpy()*float(scale)
        out[aid]=np.ascontiguousarray(np.clip(a,space.low,space.high),dtype=np.float32)
    return out

@torch.inference_mode()
def run_episode(config,model,seed,difficulty,device):
    env=HeMAC_v0.env(**config)
    try:
        env.reset(seed=seed); core=get_core_env(env); order=list(env.possible_agents)
        observers=[a for a in order if a.startswith("observer_")]; drones=[a for a in order if a.startswith("drone_")]
        ds,os=drone_scale(config),observer_scale(config)
        masks={"observer":torch.ones(1,len(observers),dtype=torch.bool,device=device),"drone":torch.ones(1,len(drones),dtype=torch.bool,device=device)}
        state=model.initial_joint_inference_state(batch_size=1,observer_count=len(observers),drone_count=len(drones),device=device)
        cached={}; cycles=0; reward_sum=0.; final_info={}; checked=False; last_agent=order[-1]
        for aid in env.agent_iter():
            _,reward,termination,truncation,info=env.last(); reward_sum+=float(reward)
            if info: final_info.update(info)
            if termination or truncation: env.step(None); continue
            if not cached:
                obs={"observer":observation_batch(env,observers,"observer",os,device),"drone":observation_batch(env,drones,"drone",ds,device)}
                if not checked: check_schema(obs["observer"],model,"observer"); check_schema(obs["drone"],model,"drone"); checked=True
                outputs,state=model.inference_step(obs,masks,state)
                cached.update(to_env_actions(outputs["actions"]["observer"][0],observers,env,os)); cached.update(to_env_actions(outputs["actions"]["drone"][0],drones,env,ds))
            env.step(cached.pop(aid))
            if aid==last_agent or bool(core.terminate) or bool(core.truncate): cycles+=1; cached.clear()
        if hasattr(core,"build_episode_info"): final_info.update(core.build_episode_info())
        og=any(agent_found_goal(core,x) for x in observers); dg=any(agent_found_goal(core,x) for x in drones)
        return {"seed":int(seed),"difficulty":int(difficulty),"n_drones":len(drones),"n_observers":len(observers),"success":float(bool(final_info.get("success",core.mission_success))),"goal_found":float(bool(final_info.get("goal_found",og))),"observer_goal_found":float(og),"drone_goal_found":float(dg),"fatal_crash":float(bool(final_info.get("fatal_crash",core.collided))),"drone_crash":float(bool(final_info.get("drone_crash",getattr(core,"drone_crash",False)))),"observer_crash":float(bool(final_info.get("observer_crash",getattr(core,"observer_crash",False)))),"coverage":float(core.current_coverage_ratio()),"cycles":float(cycles),"aec_reward_sum":float(reward_sum)}
    finally: env.close()

def aggregate(records):
    keys=("success","goal_found","observer_goal_found","drone_goal_found","fatal_crash","drone_crash","observer_crash","coverage","cycles","aec_reward_sum"); out={}
    for k in keys:
        x=np.asarray([r[k] for r in records],dtype=np.float64); out[k]=float(x.mean()); out[k+"_std"]=float(x.std(ddof=0)); out[k+"_sem"]=float(x.std(ddof=1)/np.sqrt(len(x)) if len(x)>1 else 0.)
    return out

def main():
    args=parse_args()
    if args.n_drones not in DRONE_START_POSITIONS or args.n_observers not in (1,2): raise ValueError("unsupported population")
    if not 1<=args.difficulty<=len(OBSTACLE_CURRICULUM_LEVELS): raise ValueError("invalid difficulty")
    device=resolve_device(args.device); template_path,template=load_env_template(args.env_template_data_root); config=build_target_config(template,args.difficulty,args.n_drones,args.n_observers)
    model,payload=load_joint_heterogeneous_hissd_checkpoint(args.checkpoint,device)
    if payload.get("model_type")!="hemac_joint_heterogeneous_hissd": raise ValueError(f"checkpoint type mismatch: {payload.get('model_type')!r}")
    model.eval(); records=[]; iterator=range(args.episodes)
    print(f"device={device} target=D{args.difficulty} D{args.n_drones}O{args.n_observers} checkpoint_epoch={payload.get('epoch')}")
    print(f"seed_base={args.seed_base} first_seed={args.seed_base+args.difficulty*100000} last_seed={args.seed_base+args.difficulty*100000+args.episodes-1}")
    if tqdm is not None and not args.quiet: iterator=tqdm(iterator,desc=f"joint HiSSD D{args.n_drones}O{args.n_observers}",unit="ep",dynamic_ncols=True)
    for i in iterator:
        seed=args.seed_base+args.difficulty*100_000+i; records.append(run_episode(config,model,seed,args.difficulty,device))
        if tqdm is not None and not args.quiet:
            n=len(records); iterator.set_postfix(success=f"{sum(r['success'] for r in records)/n:.3f}",crash=f"{sum(r['fatal_crash'] for r in records)/n:.3f}",coverage=f"{sum(r['coverage'] for r in records)/n:.3f}")
    summary=aggregate(records); result={"evaluation_type":"joint_heterogeneous_hissd_zero_shot_population_generalization","deterministic":True,"gradient_updates":0,"target":{"difficulty":args.difficulty,"n_drones":args.n_drones,"n_observers":args.n_observers,"drone_start_positions":config["drone_config"]["drones_starting_pos"]},"canonical_difficulty_config":DIFFICULTY_1_CONFIG if args.difficulty==1 else None,"episodes":args.episodes,"seed_base":args.seed_base,"seed_formula":"seed_base + difficulty*100000 + episode_index","template_episode":str(template_path),"environment_config":config,"checkpoint":{"path":str(args.checkpoint),"epoch":payload.get("epoch"),"model_type":payload.get("model_type"),"bc_initialization":payload.get("bc_initialization")},"summary":summary,"episode_records":records}
    args.output_json.parent.mkdir(parents=True,exist_ok=True); args.output_json.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print("\n===== JOINT HISSD ZERO-SHOT RESULT =====")
    for k in ("success","goal_found","observer_goal_found","drone_goal_found","fatal_crash","drone_crash","observer_crash","coverage","cycles"): print(f"{k:28s}= {summary[k]:.4f}")
    print(f"\noutput = {args.output_json}")
if __name__=="__main__": main()
