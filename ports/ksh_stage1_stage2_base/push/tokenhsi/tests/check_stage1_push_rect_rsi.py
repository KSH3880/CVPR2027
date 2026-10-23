"""Screen each source RSI candidate against every physical rectangular size.

Two independently yawed owner stations per environment; new bank contains only
candidates passing ALL 343 sizes. This short zero-action PD audit is not a
learned-policy stability guarantee.
"""
from isaacgym import gymapi
import csv
import hashlib
import itertools
import json
from pathlib import Path
import torch
from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from utils.door_reference import rotate
from utils.push_point import point_geometry
from utils.stage1_push import box_tilt_degrees
from env.tasks.multi_agent.humanoid_ma_stage1_push import HumanoidMAStage1Push

source_path=Path('tokenhsi/data/stage1_push_bank.json')
source_hash=hashlib.sha256(source_path.read_bytes()).hexdigest()
source=json.loads(source_path.read_text())
shapes=list(itertools.product([1.2+.05*i for i in range(7)],repeat=3))
cases=list(itertools.product(shapes,range(len(source['candidates']))))


class Screen(HumanoidMAStage1Push):
    def _sample_push_box_sizes(self):
        return torch.tensor([c[0] for c in cases],device=self.device)[:,None].expand(-1,4,3).clone()


args=get_args();assert args.headless and args.test and args.eval
args.num_envs=len(cases)
cfg,train,_=load_cfg(args);cfg['args']=args;cfg['env']['motion_file']=args.motion_file
cfg['env']['enableTrackInitState']=True
sim=parse_sim_params(args,cfg,train);set_seed(42);torch.set_num_threads(1)
task=Screen(cfg,sim,args.physics_engine,args.device,args.device_id,True)
ids=torch.arange(task.num_envs,device=task.device);n=len(ids)
try:
    task._is_eval=True
    task._reset_envs(ids)
    rows=ids.repeat_interleave(2);owners=torch.arange(2,device=task.device).repeat(n)
    selected=[source['candidates'][i] for _,i in cases for _ in range(2)]
    sample={key:torch.tensor([r[key] for r in selected],device=task.device) for key in ('motion_id','time','phase','box_progress')}
    task._apply_push_rsi(rows,owners,sample,task._box_states.clone(),ids,task._tar_pos.clone())
    task._reset_env_tensors(ids)
    reference=task._kinematic_humanoid_rigid_body_states[:,:,:,:3]
    hands=task._key_body_ids[:2]
    _,delta,face,_=point_geometry(reference[:,:,hands],task._box_states[:,:2],task._box_size[:,:2],task._push_side,task._push_local)
    initial_z=delta[...,2].abs();initial_xy=delta[...,:2].norm(dim=-1);initial_face=face.amax(-1)
    props=task.gym.get_actor_dof_properties(task.envs[0],task.humanoid_handles_all[0][0])
    lower=torch.tensor(props['lower'],device=task.device);upper=torch.tensor(props['upper'],device=task.device)
    bad_joint=((task._dof_pos<lower-.01)|(task._dof_pos>upper+.01)).any(-1)
    inv=task._box_states[:,:2,3:7].clone();inv[...,:3]*=-1
    local=rotate(inv[:,:,None].expand(-1,-1,task.num_bodies,-1),reference-task._box_states[:,:2,None,:3])
    mask=torch.ones(task.num_bodies,dtype=torch.bool,device=task.device);mask[hands]=False
    penetration=(local[:,:,mask].abs()<task._box_size[:,:2,None]/2-.02).all(-1).any(-1)
    start_root=task._humanoid_root_states[:,:,:3].clone();start_box=task._box_states[:,:2,:3].clone()
    zeros=torch.zeros(n,2,device=task.device)
    min_root=torch.full_like(zeros,float('inf'));max_drift=zeros.clone();max_box=zeros.clone();max_tilt=zeros.clone()
    supported=zeros.clone();reset_seen=torch.zeros(n,device=task.device,dtype=torch.bool)
    first_z=first_xy=first_face=None
    for tick in range(12):
        task.pre_physics_step(torch.zeros(n*2,task.num_actions,device=task.device))
        task._physics_step();task.post_physics_step()
        assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task.rew_buf).all()
        min_root=torch.minimum(min_root,task._humanoid_root_states[:,:,2])
        max_drift=torch.maximum(max_drift,(task._humanoid_root_states[:,:,:3]-start_root).norm(dim=-1))
        max_box=torch.maximum(max_box,(task._box_states[:,:2,:3]-start_box).norm(dim=-1))
        max_tilt=torch.maximum(max_tilt,box_tilt_degrees(task._box_states[:,:2,3:7]))
        supported+=(task._contact_forces[:,:,task._key_body_ids[2:],2].clamp_min(0).sum(-1)>5)
        reset_seen|=task.reset_buf.bool()
        if tick==0:
            _,d,f,_=point_geometry(task._rigid_body_pos[:,:,hands],task._box_states[:,:2],task._box_size[:,:2],task._push_side,task._push_local)
            first_z=d[...,2].abs();first_xy=d[...,:2].norm(dim=-1);first_face=f.amax(-1)
        print('RECT_RSI_SCREEN',tick+1,12,flush=True)
    passed=((initial_z<=.030001)&(initial_xy<=.18)&(initial_face<=.18)&~bad_joint&~penetration
        &(first_z<=.030001)&(first_xy<=.18)&(first_face<=.18)&(min_root>=.65)&(max_drift<=.45)
        &(max_box<=.3)&(max_tilt<=2)&(supported>=1)&~reset_seen[:,None])
    metrics=dict(initial_z_error_m=initial_z,initial_xy_error_m=initial_xy,initial_face_distance_m=initial_face,
        initial_joint_violation=bad_joint,initial_nonhand_penetration=penetration,first_z_error_m=first_z,
        first_xy_error_m=first_xy,first_face_distance_m=first_face,min_root_height_m=min_root,
        max_root_drift_m=max_drift,max_box_displacement_m=max_box,max_tilt_deg=max_tilt,supported_steps=supported,passed=passed)
    data={k:v.cpu().tolist() for k,v in metrics.items()}
    records=[]
    for e,(size,candidate) in enumerate(cases):
        for owner in range(2):
            records.append(dict(size=list(size),candidate=candidate,owner=owner,**{k:v[e][owner] for k,v in data.items()}))
    survivors=[i for i in range(len(source['candidates'])) if all(r['passed'] for r in records if r['candidate']==i)]
    out=Path(args.output_path);out.mkdir(parents=True,exist_ok=True)
    report=dict(source_sha256=source_hash,shapes=len(shapes),owners=2,control_steps=12,seconds=12*task.dt,
        survivors=survivors,passed=sum(r['passed'] for r in records),tested=len(records),cases=records)
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    with (out/'cases.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    assert hashlib.sha256(source_path.read_bytes()).hexdigest()==source_hash
    if survivors:
        bank=json.loads(Path(cfg['env']['pushTask']['bank_file']).read_text())
        bank.update(candidates=[source['candidates'][i] for i in survivors],screen_status='passed_all_rectangular_sizes',
            source_candidate_indices=survivors,screen=dict(shapes=len(shapes),owners=2,control_steps=12,
            tilt_limit_deg=2.,height_m=.65,height_tolerance_m=.03,report=str(out/'audit.json'),
            audit_sha256=hashlib.sha256((out/'audit.json').read_bytes()).hexdigest()),
            reuse_limitations='Passed finite 343-size/two-yaw short-PD RSI audit; not a long-rollout or learned-policy guarantee.')
        (out/'stage1_push_rect_point_bank.json').write_text(json.dumps(bank,indent=2)+'\n')
    print('RECT_RSI_SCREEN_DONE',json.dumps({k:v for k,v in report.items() if k!='cases'}),flush=True)
finally:task.gym.destroy_sim(task.sim)
