"""No-checkpoint whole-body PD load search against a heavy box.

A deterministic Jacobian-transpose force drive is added to reference PD targets.
It measures physical feasible loads under searched poses/targets, not a global
maximum and not trained-policy performance. No external body force is applied.
"""
from isaacgym import gymapi,gymtorch
import csv
import itertools
import json
import os
from pathlib import Path
import torch
from utils.config import get_args,load_cfg,parse_sim_params,set_seed
from env.tasks.multi_agent.humanoid_ma_open_push import HumanoidMAOpenPush

ANKLES=[-.3,-.15,0.,.15]
TORSOS=[0.,.25,.5]
HIPS=[-.2,0.,.2]
LOADS=[100.,200.,300.,400.,500.,600.]
POSES=[0,5,10,15]
CASES=list(itertools.product(POSES,ANKLES,TORSOS,HIPS,LOADS))
old_build=HumanoidMAOpenPush._build_box

def build(self,e,b,ptr):
    old_build(self,e,b,ptr)
    if b>=2:
        handle=self._box_handles[-1]
        props=self.gym.get_actor_rigid_body_properties(ptr,handle);props[0].mass=1000.
        self.gym.set_actor_rigid_body_properties(ptr,handle,props,True)

HumanoidMAOpenPush._build_box=build
torch.set_num_threads(1)
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
args=get_args();assert args.num_envs==len(CASES) and args.test and args.eval and args.headless
cfg,train,_=load_cfg(args);cfg['args']=args;cfg['env']['motion_file']=args.motion_file
cfg['env']['enableTrackInitState']=True
sim=parse_sim_params(args,cfg,train);sim.physx.num_threads=1;set_seed(42)
task=HumanoidMAOpenPush(cfg,sim,args.physics_engine,args.device,args.device_id,True)
out=Path(args.output_path);out.mkdir(parents=True,exist_ok=True)
try:
    n=len(CASES);ids=torch.arange(n,device=task.device);task._task_graph_preset='push';task._reset_envs(ids)
    task._agent_lanes[:]=torch.tensor([0,1],device=task.device);task._push_binding[:]=task._agent_lanes
    task._box_states[:,2:]=0.;task._box_states[:,2:,2]=.65;task._box_states[:,2:,6]=1.
    task._box_states[:,2:,0]=-4.;task._box_states[:,2,1]=-5.;task._box_states[:,3,0]=-10.;task._box_states[:,3,1]=5.
    task._push_reset_states[:]=task._box_states[:,2:]
    task._push_directions[:]=task._push_directions.new_tensor([0.,1.,0.]);task._push_goal_offsets[:]=task._push_directions*2
    task._sample_box_assignments(ids)
    task.relation_runtime.reset(ids,torch.ones(n,2,device=task.device,dtype=torch.long),task._owner_push_boxes())
    bank=task._joint_rsi_bank.rows['push'];selected={}
    for mid in POSES:
        rows=[r for r in bank if r['motion_id']==mid];assert rows
        selected[mid]=rows[len(rows)//2]
    sample={k:torch.tensor([selected[mid][k] for mid,*_ in CASES],device=task.device) for k in bank[0]}
    sample['box_progress']=torch.zeros(n,device=task.device)
    task._apply_joint_rsi(ids,torch.zeros_like(ids),'push',sample)
    task._humanoid_root_states[:,1,:2]+=100.;task._kinematic_humanoid_rigid_body_states[:,1,:,:2]+=100.
    task._humanoid_root_states[...,7:]=0.;task._dof_vel[:]=0.;task._kinematic_humanoid_rigid_body_states[...,7:]=0.
    task._reset_task(ids);task.relation_runtime.reset_push_credit(ids,task._owner_push_boxes(),task._tar_pos)
    task._reset_env_tensors(ids)
    rest=task._dof_pos.clone()
    jac=gymtorch.wrap_tensor(task.gym.acquire_jacobian_tensor(task.sim,'humanoid_0'))
    task.gym.refresh_jacobian_tensors(task.sim)
    names=task.gym.get_actor_dof_names(task.envs[0],task.humanoid_handles_all[0][0])
    props=task.gym.get_actor_dof_properties(task.envs[0],task.humanoid_handles_all[0][0])
    kp=torch.tensor(props['stiffness'],device=task.device)
    offset=task._pd_action_offset;scale=task._pd_action_scale
    lo=offset-scale;hi=offset+scale
    settings=torch.tensor([row[1:] for row in CASES],device=task.device)
    knees=[names.index('right_knee_y'),names.index('left_knee_y')]
    ankles=[names.index('right_ankle_y'),names.index('left_ankle_y')]
    hips=[names.index('right_hip_y'),names.index('left_hip_y')]
    abdomen=names.index('abdomen_y')
    hands=task._key_body_ids[:2];feet=task._posture_body_ids[2:]
    valid_count=torch.zeros(n,device=task.device);pair_count=torch.zeros_like(valid_count)
    fy_sum=torch.zeros_like(valid_count);fy_min=torch.full_like(valid_count,float('inf'));fy_max=torch.zeros_like(valid_count)
    fz_sum=torch.zeros_like(valid_count);foot_z_sum=torch.zeros_like(valid_count);height_sum=torch.zeros_like(valid_count)
    pelvis_min=torch.full_like(valid_count,float('inf'));feet_slip_max=torch.zeros_like(valid_count)
    actual_tau_max=torch.zeros(n,32,device=task.device);target_clip_max=torch.zeros_like(valid_count)
    initial_feet=None;box_start=task._box_states[:,2,:3].clone();obs_steps=60;all_steps=150
    for step in range(all_steps):
        task.gym.refresh_jacobian_tensors(task.sim)
        ramp=min(1.,step/30.)
        push_ramp=max(0.,min(1.,(step-30)/30.))
        target=rest.clone()
        target[:,0,ankles]+=settings[:,0,None]*ramp
        target[:,0,abdomen]+=settings[:,1]*ramp
        target[:,0,hips]+=settings[:,2,None]*ramp
        desired=settings[:,3]*push_ramp/2
        tau=(jac[:,hands,1,6:]*desired[:,None,None]).sum(1)
        target[:,0]+=tau/kp.clamp_min(1.)
        clipped=target.clamp(lo,hi)
        target_clip_max=torch.maximum(target_clip_max,(target[:,0]-clipped[:,0]).abs().amax(-1))
        task._every_env_init_dof_pos[:]=clipped
        task.pre_physics_step(torch.zeros(n*2,32,device=task.device));task._physics_step()
        body=task._rigid_body_pos[:,0];root=task._humanoid_root_states[:,0]
        hand_force=task._contact_forces[:,0,hands]
        force=(-hand_force[...,1]).clamp_min(0).sum(-1)
        vertical=hand_force[...,2].sum(-1)
        foot_force=task._contact_forces[:,0,feet]
        contact=(foot_force[...,2]>1.).all(-1)
        # Normal distance to rear face and low enough body position for this box.
        near=((body[:,hands,1]-(task._box_states[:,2,1]-.65)[:,None]).abs()<.12).all(-1)
        heights=(body[:,hands,2]>.1).all(-1)&(body[:,hands,2]<1.3).all(-1)
        good=(root[:,2]>.65)&contact&near&heights
        if step==60:initial_feet=body[:,feet,:2].clone()
        if step>=all_steps-obs_steps:
            slip=(body[:,feet,:2]-initial_feet).norm(dim=-1).amax(-1)
            feet_slip_max=torch.maximum(feet_slip_max,slip)
            good &= slip<.1
            valid_count+=good
            pair_count+=near&heights
            fy_sum+=torch.where(good,force,0.)
            fy_min=torch.minimum(fy_min,torch.where(good,force,float('inf')))
            fy_max=torch.maximum(fy_max,torch.where(good,force,0.))
            fz_sum+=torch.where(good,vertical,0.)
            foot_z_sum+=torch.where(good,foot_force[...,2].sum(-1),0.)
            height_sum+=torch.where(good,body[:,hands,2].mean(-1),0.)
            pelvis_min=torch.minimum(pelvis_min,root[:,2])
        actual_tau_max=torch.maximum(actual_tau_max,task.dof_force_tensor[:,0].abs())
        if step%30==0:print('ACTUATOR_LOAD_PROGRESS',step,all_steps,flush=True)
    rows=[]
    for e,(mid,ankle,torso,hip,load) in enumerate(CASES):
        count=float(valid_count[e]);mean=float(fy_sum[e])/max(1,count)
        rows.append(dict(motion_id=mid,ankle_offset_rad=ankle,torso_offset_rad=torso,hip_offset_rad=hip,command_horizontal_load_N=load,
            stable_support_contact_fraction=count/obs_steps,paired_near_fraction=float(pair_count[e])/obs_steps,
            supported_mean_horizontal_hand_reaction_N=mean,supported_min_horizontal_hand_reaction_N=float(fy_min[e]) if count else None,
            supported_peak_horizontal_hand_reaction_N=float(fy_max[e]),supported_mean_vertical_hand_reaction_N=float(fz_sum[e])/max(1,count),
            supported_mean_feet_normal_N=float(foot_z_sum[e])/max(1,count),supported_mean_hand_height_m=float(height_sum[e])/max(1,count),
            max_feet_drift_m=float(feet_slip_max[e]),min_pelvis_height_m=float(pelvis_min[e]),max_pd_target_clip_rad=float(target_clip_max[e]),
            max_observed_dof_force_Nm=float(actual_tau_max[e].max()),box_displacement_m=float((task._box_states[e,2,:3]-box_start[e]).norm())))
    supported=[r for r in rows if r['stable_support_contact_fraction']>=.9]
    horizontal=[r for r in supported if abs(r['supported_mean_vertical_hand_reaction_N'])<=50.]
    report=dict(policy_used=False,checkpoint_used=False,external_body_force=False,gpu=os.environ['TOKENHSI_GPU'],num_cases=n,probe_seconds=all_steps*task.dt,steady_window_seconds=obs_steps*task.dt,
        method='Native PD with target-space Jacobian-transpose force offset, targets clipped to standard action range; ankle/torso/hip offsets grid. Source motion poses initialize mechanics; no neural policy or optimizer is loaded. Heavy1000kg box prevents substantial sliding.',
        limits='Feasible loads of a deterministic controller/pose search, not a proved global physical maximum. Valid supports require both feet contact, rootheight>.65m, hands near rear face, and feet drift<.1m in steady window. Net hand reactions include collisions; quasi-horizontal subset limits vertical hand reaction to50N.',
        supported_cases=len(supported),quasi_horizontal_supported_cases=len(horizontal),best_supported=max(supported,key=lambda r:r['supported_mean_horizontal_hand_reaction_N']) if supported else None,
        best_quasi_horizontal_supported=max(horizontal,key=lambda r:r['supported_mean_horizontal_hand_reaction_N']) if horizontal else None,cases=rows)
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    with (out/'cases.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2),flush=True)
finally:task.gym.destroy_sim(task.sim)
