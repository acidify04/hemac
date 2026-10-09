#!/usr/bin/env python3
"""Canonical zero-shot eval for one joint heterogeneous BC checkpoint."""
from __future__ import annotations
import argparse,copy,json,sys
from pathlib import Path
import numpy as np
import torch
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm=None
PROJECT_ROOT=Path(__file__).resolve().parents[2]; PROJECT_SRC=PROJECT_ROOT/"src"
if str(PROJECT_SRC) not in sys.path: sys.path.insert(0,str(PROJECT_SRC))
from hemac import HeMAC_v0
from hemac.curriculum_config import OBSTACLE_CURRICULUM_LEVELS
from skill_discovery.collect_offline_data import agent_found_goal,build_collection_env_config,convert_observation,get_core_env
from skill_discovery.decentralized_bc_models import load_decentralized_bc_checkpoint
DRONE_START_POSITIONS={2:[[130.,850.,5.],[170.,850.,5.]],3:[[130.,870.,5.],[170.,870.,5.],[150.,830.,5.]],4:[[130.,870.,5.],[130.,830.,5.],[170.,870.,5.],[170.,830.,5.]],5:[[130.,870.,5.],[170.,870.,5.],[110.,830.,5.],[150.,830.,5.],[190.,830.,5.]]}
DIFFICULTY_1_CONFIG={"min_obstacles":3,"max_obstacles":4,"obstacle_min_speed":1,"obstacle_max_speed":3,"n_static_obstacles":2,"goal_min_base_distance":475.0,"goal_max_base_distance":600.0}
def parse_args():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--env-template-data-root",type=Path,required=True); p.add_argument("--n-drones",type=int,required=True); p.add_argument("--n-observers",type=int,required=True); p.add_argument("--difficulty",type=int,default=1); p.add_argument("--episodes",type=int,default=200); p.add_argument("--seed-base",type=int,default=100_000_000); p.add_argument("--device",choices=("auto","cpu","cuda"),default="auto"); p.add_argument("--output-json",type=Path,required=True); p.add_argument("--quiet",action="store_true"); return p.parse_args()
def device_of(x):
 if x=="auto": x="cuda" if torch.cuda.is_available() else "cpu"
 if x=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
 return torch.device(x)
def template(root):
 paths=sorted(root.expanduser().resolve().glob("difficulty_*/*/*.pt")) or sorted(root.expanduser().resolve().rglob("*.pt"))
 for p in paths:
  x=torch.load(p,map_location="cpu",weights_only=False,mmap=True); c=x.get("metadata",{}).get("environment_config")
  if isinstance(c,dict) and c:return p,dict(c)
 raise RuntimeError("no environment template")
def cfg(base,d,n,o):
 c=copy.deepcopy(build_collection_env_config(copy.deepcopy(base),d));
 if int(d)==1:c.update(DIFFICULTY_1_CONFIG)
 c.update(n_drones=int(n),n_observers=int(o),n_provisioners=0,render_mode=None,log_step_rewards=False); dc=copy.deepcopy(c.get("drone_config") or {}); dc["drones_starting_pos"]=copy.deepcopy(DRONE_START_POSITIONS[int(n)]); c["drone_config"]=dc; return c
def obs(env,ids,role,scale,device):
 xs=[convert_observation(env.observe(a),role) for a in ids]; return {"global_map":torch.from_numpy(np.stack([x["global_map"] for x in xs])).unsqueeze(0).float().to(device),"local_map":torch.from_numpy(np.stack([x["local_map"] for x in xs])).unsqueeze(0).float().to(device),"action_history":torch.from_numpy(np.stack([x["action_history"] for x in xs])).unsqueeze(0).float().to(device)/float(scale)}
def actions(x,ids,env,scale):
 out={}
 for i,a in enumerate(ids):
  sp=env.action_space(a); v=x[i].detach().cpu().numpy()*float(scale); out[a]=np.ascontiguousarray(np.clip(v,sp.low,sp.high),dtype=np.float32)
 return out
@torch.inference_mode()
def episode(c,model,seed,d,device):
 env=HeMAC_v0.env(**c)
 try:
  env.reset(seed=seed); core=get_core_env(env); order=list(env.possible_agents); O=[a for a in order if a.startswith("observer_")]; D=[a for a in order if a.startswith("drone_")]; ds=float((c.get("drone_config") or {}).get("drone_max_speed",25.)); os=float(c.get("observer_speed",10.)); masks={"observer":torch.ones(1,len(O),dtype=torch.bool,device=device),"drone":torch.ones(1,len(D),dtype=torch.bool,device=device)}; cached={}; cycles=0; rew=0.; infof={}; last=order[-1]
  for aid in env.agent_iter():
   _,r,t,tr,info=env.last(); rew+=float(r); infof.update(info or {})
   if t or tr: env.step(None); continue
   if not cached:
    ob={"observer":obs(env,O,"observer",os,device),"drone":obs(env,D,"drone",ds,device)}; out=model.forward_joint(ob,masks)["actions"]; cached.update(actions(out["observer"][0],O,env,os)); cached.update(actions(out["drone"][0],D,env,ds))
   env.step(cached.pop(aid))
   if aid==last or bool(core.terminate) or bool(core.truncate):cycles+=1;cached.clear()
  if hasattr(core,"build_episode_info"):infof.update(core.build_episode_info())
  og=any(agent_found_goal(core,x) for x in O); dg=any(agent_found_goal(core,x) for x in D)
  return {"seed":int(seed),"success":float(bool(infof.get("success",core.mission_success))),"goal_found":float(bool(infof.get("goal_found",og))),"observer_goal_found":float(og),"drone_goal_found":float(dg),"fatal_crash":float(bool(infof.get("fatal_crash",core.collided))),"drone_crash":float(bool(infof.get("drone_crash",getattr(core,"drone_crash",False)))),"observer_crash":float(bool(infof.get("observer_crash",getattr(core,"observer_crash",False)))),"coverage":float(core.current_coverage_ratio()),"cycles":float(cycles),"aec_reward_sum":float(rew)}
 finally:env.close()
def aggregate(rs):
 out={}
 for k in ("success","goal_found","observer_goal_found","drone_goal_found","fatal_crash","drone_crash","observer_crash","coverage","cycles","aec_reward_sum"):
  x=np.asarray([r[k] for r in rs],dtype=float);out[k]=float(x.mean());out[k+"_std"]=float(x.std());out[k+"_sem"]=float(x.std(ddof=1)/np.sqrt(len(x)) if len(x)>1 else 0.)
 return out
def main():
 a=parse_args();
 if a.n_drones not in DRONE_START_POSITIONS or a.n_observers not in (1,2):raise ValueError("unsupported population")
 if not 1<=a.difficulty<=len(OBSTACLE_CURRICULUM_LEVELS):raise ValueError("invalid difficulty")
 device=device_of(a.device); tp,t=template(a.env_template_data_root); c=cfg(t,a.difficulty,a.n_drones,a.n_observers); model,payload=load_decentralized_bc_checkpoint(a.checkpoint,device);model.eval();rs=[];it=range(a.episodes)
 print(f"device={device} target=D{a.difficulty} D{a.n_drones}O{a.n_observers} checkpoint_epoch={payload.get('epoch')}");print(f"seed_base={a.seed_base} first_seed={a.seed_base+a.difficulty*100000} last_seed={a.seed_base+a.difficulty*100000+a.episodes-1}")
 if tqdm is not None and not a.quiet:it=tqdm(it,desc=f"decentralized BC D{a.n_drones}O{a.n_observers}",unit="ep",dynamic_ncols=True)
 for i in it:
  rs.append(episode(c,model,a.seed_base+a.difficulty*100_000+i,a.difficulty,device))
  if tqdm is not None and not a.quiet:
   n=len(rs);it.set_postfix(success=f"{sum(r['success'] for r in rs)/n:.3f}",crash=f"{sum(r['fatal_crash'] for r in rs)/n:.3f}",coverage=f"{sum(r['coverage'] for r in rs)/n:.3f}")
 s=aggregate(rs);result={"evaluation_type":"decentralized_bc_zero_shot_population_generalization","deterministic":True,"gradient_updates":0,"target":{"difficulty":a.difficulty,"n_drones":a.n_drones,"n_observers":a.n_observers,"drone_start_positions":c["drone_config"]["drones_starting_pos"]},"canonical_difficulty_config":DIFFICULTY_1_CONFIG if a.difficulty==1 else None,"episodes":a.episodes,"seed_base":a.seed_base,"seed_formula":"seed_base + difficulty*100000 + episode_index","template_episode":str(tp),"environment_config":c,"checkpoint":{"path":str(a.checkpoint),"epoch":payload.get("epoch"),"model_type":payload.get("model_type")},"summary":s,"episode_records":rs};a.output_json.parent.mkdir(parents=True,exist_ok=True);a.output_json.write_text(json.dumps(result,indent=2),encoding="utf-8")
 print("\n===== DECENTRALIZED BC ZERO-SHOT RESULT =====");[print(f"{k:28s}= {s[k]:.4f}") for k in ("success","goal_found","observer_goal_found","drone_goal_found","fatal_crash","drone_crash","observer_crash","coverage","cycles")];print(f"\noutput = {a.output_json}")
if __name__=="__main__":main()
