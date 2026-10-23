"""Re-screen every selected native RSI frame at the configured physical box.

Never broadens the source pool, edits source poses, or overwrites source caches.
Outputs an independently hashed cache and full per-frame PD screen results.
"""
from isaacgym import gymapi, gymtorch
import hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from utils.config import get_args,load_cfg,parse_sim_params,set_seed
from utils.stage1_push import box_size_profiles,sample_boxes_and_goals
from utils.stage1_v2 import pose_geometry_valid
from utils.size_rsi import SCREEN,screen_states
from utils.door_reference import rotate
from env.tasks.multi_agent.humanoid_ma_stage1_push import HumanoidMAStage1Push

BATCH=16
args=get_args();cfg,train,_=load_cfg(args);cfg['args']=args;cfg['env']['motion_file']=args.motion_file
source=Path(cfg['env']['pushTask']['rsi_cache_file'])
with np.load(source,allow_pickle=False) as data:payload={key:data[key].copy() for key in data.files}
shapes=box_size_profiles(cfg['env']);assert len(shapes)==len(payload['sizes'])
for old,new in zip(payload['sizes'],shapes):assert np.allclose(old[[0,2]],np.array(new)[[0,2]])
class Probe(HumanoidMAStage1Push):
    _building_push_rsi_cache=True
    def _sample_push_box_sizes(self):
        sizes=torch.tensor(shapes,device=self.device).repeat_interleave(BATCH,0)
        return sizes[:,None].expand(-1,4,3).clone()
args.num_envs=cfg['env']['numEnvs']=len(shapes)*BATCH
set_seed(42);torch.set_num_threads(1)
t=Probe(cfg,parse_sim_params(args,cfg,train),args.physics_engine,args.device,args.device_id,True)
ids=torch.arange(t.num_envs,device=t.device);rows=ids.repeat_interleave(2);owners=torch.arange(2,device=t.device).repeat(len(ids))
lib=t._motion_lib['push'];geometry=t._push_reset_geometry();out=Path(args.output_path);out.mkdir(parents=True,exist_ok=True)
assert not (out/'verified_cache.npz').exists()
source_valid=payload['valid'].copy();accepted=np.zeros_like(source_valid);offsets=payload['clip_offsets']
frames=np.flatnonzero(source_valid.any(0));steps=math.ceil(SCREEN['seconds']/t.sim_params.dt)
records=[];max_dof_error=max_vel_error=0.;invalid_reset=0;max_hand_height=0.;max_sole_shift=0.;min_head_gap=float('inf')
try:
    masses=[];frictions=[]
    for i,actor in enumerate(t._box_handles):
        masses.append(sum(x.mass for x in t.gym.get_actor_rigid_body_properties(t.envs[i//4],actor)))
        frictions.extend(x.friction for x in t.gym.get_actor_rigid_shape_properties(t.envs[i//4],actor))
    assert all(abs(x-cfg['env']['pushBox']['mass'])<1e-4 for x in masses)
    assert all(abs(x-cfg['env']['pushBox']['friction'])<1e-6 for x in frictions)
    for start in range(0,len(frames),BATCH):
        subset=frames[start:start+BATCH];width=len(subset)
        chosen=np.pad(subset,(0,BATCH-width),mode='edge')
        clips=np.searchsorted(offsets[1:],chosen,side='right');local=chosen-offsets[clips]
        c=torch.tensor(clips,device=t.device).repeat(len(shapes)).repeat_interleave(2)
        f=torch.tensor(local,device=t.device).repeat(len(shapes)).repeat_interleave(2)
        times=f*lib._motion_dt[c]
        states,goals,_,_=sample_boxes_and_goals(t.num_envs,t.device,t._env_origins,cfg['env'],t._box_size)
        t._box_states.copy_(states);t._tar_pos.copy_(goals)
        t._reset_ref_slots={'loco':(rows,owners)};t._reset_ref_motion_ids={'loco':torch.zeros_like(rows)};t._reset_ref_motion_times={'loco':torch.zeros(len(rows),device=t.device)}
        t._apply_push_rsi(rows,owners,dict(motion_id=c,time=times,box_progress=torch.zeros_like(times)),states,ids,goals)
        pose=t._kinematic_humanoid_rigid_body_states[rows,owners];box=t._box_states[rows,owners];size=t._box_size[rows,owners]
        bad=geometry.invalid(pose[...,:3],pose[...,3:7],box,size);invalid_reset+=int(bad.sum())
        ref=lib.get_motion_state(c,times)
        max_dof_error=max(max_dof_error,float((t._dof_pos[rows,owners]-ref[2]).abs().max()))
        max_vel_error=max(max_vel_error,float((t._dof_vel[rows,owners]-ref[5]).abs().max()))
        max_hand_height=max(max_hand_height,float(pose[:,t._key_body_ids[:2],2].mean(-1).max()))
        # Independent height and rigid floor-shift checks.
        body,rot,_,_=lib.get_motion_state_max(c,times)
        dz=pose[...,2]-body[...,2]
        torch.testing.assert_close(dz,dz[:,:1].expand_as(dz),atol=2e-6,rtol=0)
        max_sole_shift=max(max_sole_shift,float(dz.max()))
        initial_root=t._humanoid_root_states[rows,owners].clone();initial_box=box[:,None].clone()
        finite=torch.ones(len(rows),device=t.device,dtype=torch.bool);fall=torch.zeros_like(finite);ok=~bad
        ok &= pose_geometry_valid(pose[...,:3],pose[:,t._key_body_ids[-2:],:3],box[:,None],size[:,None])
        t.gym.set_actor_root_state_tensor(t.sim,gymtorch.unwrap_tensor(t._root_states));t.gym.set_dof_state_tensor(t.sim,gymtorch.unwrap_tensor(t._dof_state))
        t.gym.set_dof_position_target_tensor(t.sim,gymtorch.unwrap_tensor(t._dof_pos.contiguous().clone()))
        for _ in range(steps):
            t.gym.simulate(t.sim);t.gym.fetch_results(t.sim,True);t._refresh_sim_tensors()
            body=t._rigid_body_pos[rows,owners];box=t._box_states[rows,owners][:,None]
            ok &= screen_states(t._humanoid_root_states[rows,owners],box,initial_root,initial_box)
            finite &= torch.isfinite(t._dof_state.view(t.num_envs,2,-1)[rows,owners]).all(-1)
            ok &= pose_geometry_valid(body,body[:,t._key_body_ids[-2:]],box,size[:,None])
            low=body[...,2]<t._termination_heights;low[:,t._contact_body_ids]=False
            forces=t._contact_forces[rows,owners].clone();forces[:,t._contact_body_ids]=0
            fall |= low.any(-1)&(forces.abs().sum(-1)>.1).any(-1)
        passed=(ok&finite&~fall).reshape(len(shapes),BATCH,2)
        passed_pair=passed.all(-1)[:,:width].cpu().numpy()
        accepted[:,subset]=passed_pair & source_valid[:,subset]
        records.append(dict(start=start,width=width,passed_agent_cases=int(passed[:,:width].sum()),agent_cases=len(shapes)*width*2,falls=int(fall.reshape(len(shapes),BATCH,2)[:,:width].sum()),nonfinite=int((~finite).reshape(len(shapes),BATCH,2)[:,:width].sum())))
        if start%(BATCH*10)==0:print('RECHECK',start,len(frames),flush=True)
    payload['valid']=accepted;payload['sizes']=np.array(shapes)
    meta=json.loads(str(payload['metadata']));meta.update(box=cfg['env']['pushBox'],rescreen_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),rescreen=dict(seconds=SCREEN['seconds'],steps=steps,selected_frames_only=True,original_dof_and_velocity=True))
    payload['metadata']=np.array(json.dumps(meta,sort_keys=True))
    assert accepted.any(-1).all();np.savez_compressed(out/'verified_cache.npz',**payload)
    report=dict(box=cfg['env']['pushBox'],profiles=len(shapes),source_candidates=int(source_valid.sum()),accepted_size_frames=int(accepted.sum()),rejected_size_frames=int((source_valid&~accepted).sum()),screened_agent_cases=sum(x['agent_cases'] for x in records),passed_agent_cases=sum(x['passed_agent_cases'] for x in records),falls=sum(x['falls'] for x in records),nonfinite=sum(x['nonfinite'] for x in records),invalid_reset=invalid_reset,max_dof_position_error=max_dof_error,max_dof_velocity_error=max_vel_error,max_hand_midpoint_z_m=max_hand_height,max_uniform_floor_shift_m=max_sole_shift,actual_masses_kg=sorted(set(masses)),actual_shape_frictions=sorted(set(frictions)),seconds=SCREEN['seconds'],steps=steps,cache_sha256=hashlib.sha256((out/'verified_cache.npz').read_bytes()).hexdigest(),batches=records,limitation='Initial original-pose PD hold; not learned policy performance or long-term stability')
    (out/'rescreen.json').write_text(json.dumps(report,indent=2)+'\n');print('RESCREEN_DONE',json.dumps({k:v for k,v in report.items() if k!='batches'}),flush=True)
finally:t.gym.destroy_sim(t.sim)
