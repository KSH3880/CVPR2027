"""Build PUSH size/frame pools with the imported SIT/CARRY screening rules.

Original motions and DOFs remain unchanged. Eight reference frames per size are
screened in parallel. This script writes a candidate cache to its check output;
it never overwrites the active config/cache automatically.
"""
from isaacgym import gymapi,gymtorch
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from utils.config import get_args,load_cfg,parse_sim_params,set_seed
from utils.size_rsi import SCREEN,screen_states
from utils.stage1_v2 import pose_geometry_valid
from utils.stage1_push import sample_boxes_and_goals, box_size_profiles, hand_box_overlap
from utils.push_reset_geometry import reset_geometry_contract
from env.tasks.multi_agent.humanoid_ma_stage1_push import HumanoidMAStage1Push

BATCH=8


class Builder(HumanoidMAStage1Push):
    _building_push_rsi_cache=True

    def _sample_push_box_sizes(self):
        sizes=torch.tensor(SHAPES,device=self.device).repeat_interleave(BATCH,0)
        return sizes[:,None].expand(-1,4,3).clone()


args=get_args();assert args.test and args.eval and args.headless
cfg,train,_=load_cfg(args);cfg['args']=args;cfg['env']['motion_file']=args.motion_file;cfg['env']['enableTrackInitState']=True
SHAPES=box_size_profiles(cfg['env'])
args.num_envs=cfg['env']['numEnvs']=len(SHAPES)*BATCH
sim=parse_sim_params(args,cfg,train);set_seed(42);torch.set_num_threads(1)
task=Builder(cfg,sim,args.physics_engine,args.device,args.device_id,True)
ids=torch.arange(task.num_envs,device=task.device);rows=ids.repeat_interleave(2);owners=torch.arange(2,device=task.device).repeat(len(ids))
lib=task._motion_lib['push'];counts=lib._motion_num_frames.cpu().numpy();offsets=np.r_[0,np.cumsum(counts)]
valid=np.zeros((len(SHAPES),int(offsets[-1])),dtype=np.bool_)
steps=math.ceil(SCREEN['seconds']/task.sim_params.dt)
out=Path(args.output_path);out.mkdir(parents=True,exist_ok=True)
hand_audit={stage:dict(hand_samples=0,center_inside=0,sphere_overlap=0,max_overlap_m=0.) for stage in ('reset','physics')}


def audit_hands(stage, body, boxes, sizes, width):
    depth, inside=hand_box_overlap(body[:,task._key_body_ids[:2]],boxes[:,0],sizes[:,0])
    depth=depth.reshape(len(SHAPES),BATCH,2,2)[:,:width]
    inside=inside.reshape(len(SHAPES),BATCH,2,2)[:,:width]
    stat=hand_audit[stage]
    stat['hand_samples']+=depth.numel()
    stat['center_inside']+=int(inside.sum())
    stat['sphere_overlap']+=int((depth>1e-5).sum())
    stat['max_overlap_m']=max(stat['max_overlap_m'],depth.max().item())


try:
    states,goals,_,_=sample_boxes_and_goals(len(ids),task.device,task._env_origins,cfg['env'],task._box_size)
    summaries=[]
    for clip in range(lib.num_motions()):
        frames=int(counts[clip])
        for start in range(0,frames,BATCH):
            frame=torch.arange(start,min(start+BATCH,frames),device=task.device)
            width=len(frame)
            if width<BATCH:frame=torch.cat((frame,frame[-1:].repeat(BATCH-width)))
            frame=frame.repeat(len(SHAPES)).repeat_interleave(2)
            sample=dict(motion_id=torch.full_like(frame,clip),time=frame*lib._motion_dt[clip],box_progress=torch.zeros_like(frame,dtype=torch.float))
            task._box_states.copy_(states);task._tar_pos.copy_(goals)
            empty=torch.empty(0,device=task.device,dtype=torch.long)
            task._reset_ref_slots={'loco':(empty,empty)};task._reset_ref_motion_ids={'loco':empty};task._reset_ref_motion_times={'loco':empty.float()}
            task._apply_push_rsi(rows,owners,sample,states,ids,goals)
            body=task._kinematic_humanoid_rigid_body_states[rows,owners,:,:3]
            boxes=task._box_states[rows,owners][:,None];sizes=task._box_size[rows,owners][:,None]
            audit_hands('reset',body,boxes,sizes,width)
            ok=pose_geometry_valid(body,body[:,task._key_body_ids[-2:]],boxes,sizes)
            initial_root=task._humanoid_root_states[rows,owners].clone();initial_box=boxes.clone()
            task.gym.set_actor_root_state_tensor(task.sim,gymtorch.unwrap_tensor(task._root_states))
            task.gym.set_dof_state_tensor(task.sim,gymtorch.unwrap_tensor(task._dof_state))
            targets=task._dof_pos.contiguous().clone()
            task.gym.set_dof_position_target_tensor(task.sim,gymtorch.unwrap_tensor(targets))
            for _ in range(steps):
                task.gym.simulate(task.sim);task.gym.fetch_results(task.sim,True);task._refresh_sim_tensors()
                body=task._rigid_body_pos[rows,owners];boxes=task._box_states[rows,owners][:,None]
                audit_hands('physics',body,boxes,sizes,width)
                ok &= screen_states(task._humanoid_root_states[rows,owners],boxes,initial_root,initial_box)
                ok &= torch.isfinite(task._dof_state.view(task.num_envs,2,-1)[rows,owners]).all(-1)
                ok &= pose_geometry_valid(body,body[:,task._key_body_ids[-2:]],boxes,sizes)
                low=body[...,2]<task._termination_heights;low[:,task._contact_body_ids]=False
                force=task._contact_forces[rows,owners].clone();force[:,task._contact_body_ids]=0
                ok &= ~(low.any(-1)&(force.abs().sum(-1)>.1).any(-1))
            accepted=ok.reshape(len(SHAPES),BATCH,2).all(-1)[:,:width].cpu().numpy()
            valid[:,offsets[clip]+start:offsets[clip]+start+width]=accepted
            if start%240==0:print('NATIVE_RSI_SCREEN',clip,start,frames,flush=True)
        summaries.append(dict(motion_id=clip,total_frames=frames,accepted_size_frames=int(valid[:,offsets[clip]:offsets[clip+1]].sum())))
        print('NATIVE_RSI_CLIP_DONE',json.dumps(summaries[-1]),flush=True)
        np.save(out/'valid.partial.npy',valid)
    meta=dict(version=1,motion_hashes=cfg['env']['pushTask']['motion_hashes'],box=cfg['env']['pushBox'],
        sim=cfg['sim'],screen=SCREEN,dt=task.sim_params.dt,steps=steps,owners=2,retarget=False,
        phase='all frames, as CARRY',late='utils.size_rsi.late_frames(valid, all_true)',
        asset_sha256=hashlib.sha256((Path(cfg['env']['asset']['assetRoot'])/cfg['env']['asset']['assetFileName']).read_bytes()).hexdigest())
    if reset_geometry_contract(cfg['env']) is not None:
        meta['reset_geometry']=reset_geometry_contract(cfg['env'])
    path=out/'stage1_push_rect_native_rsi_cache.npz'
    np.savez_compressed(path,valid=valid,sizes=np.array(SHAPES),clip_offsets=offsets,metadata=np.array(json.dumps(meta,sort_keys=True)))
    report=dict(shapes=len(SHAPES),source_frames=int(offsets[-1]),size_frame_pairs=int(valid.size),
        accepted_size_frame_pairs=int(valid.sum()),empty_size_profiles=int((~valid.any(-1)).sum()),
        clips=summaries,cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),metadata=meta)
    report['hand_box_audit']=hand_audit
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print('NATIVE_RSI_DONE',json.dumps({k:v for k,v in report.items() if k not in ('metadata','clips')}),flush=True)
finally:task.gym.destroy_sim(task.sim)
