"""Actual humanoid/box contacts with AMP initialization and midpoint-only IK.

This is a controlled physics feasibility sweep, not a trained-policy evaluation.
Only analysis-local assets/controllers are overridden; no training config or
checkpoint is modified. No force is applied directly to any box/root body.
"""
from isaacgym import gymapi, gymtorch
import csv
import hashlib
import itertools
import json
import os
import subprocess
from pathlib import Path
import torch
from utils.config import get_args,load_cfg,parse_sim_params,set_seed
from utils.door_reference import rotate,multiply
from utils.open_push_rsi import lower_push_arms,align_push_pose
from env.tasks.multi_agent.humanoid_ma_stage1_push import HumanoidMAStage1Push
from push_stability_common import SHAPES,MASSES,tilt_degrees

gpu_uuid=subprocess.check_output(['nvidia-smi','-i','6','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
assert os.environ.get('TOKENHSI_GPU') in ('6',gpu_uuid)
torch.set_num_threads(1);os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
MOTION_IDS=list(range(21))
HEIGHT_FRACTIONS=[.45,.50,.55,.60]
TRACK_REFERENCE=os.environ.get('PUSH_RECT_TRACK_REFERENCE')=='1'
if os.environ.get('PUSH_RECT_QUICK')=='1':
    SHAPES=[(1.2,1.2,1.5),(1.35,1.35,1.35),(1.5,1.5,1.2)]
    MASSES=[15.,20.];MOTION_IDS=[0,5,19];HEIGHT_FRACTIONS=[.5]
if os.environ.get('PUSH_RECT_DEPTH180')=='1':
    SHAPES=[(1.2+i*.05,y,1.3) for i in range(7) for y in (1.8,1.2+i*.05)]
    MASSES=[float(os.environ.get('PUSH_RECT_DEPTH_MASS','20'))]
    assert 0 < MASSES[0] < float('inf')
CASES=list(itertools.product(SHAPES,MASSES,MOTION_IDS))


class RectAnalysis(HumanoidMAStage1Push):
    def _load_box_asset(self):
        n,o=self.num_envs,self.num_objects
        self._box_size=torch.tensor([case[0] for case in CASES],device=self.device)[:,None].expand(n,o,3).clone()
        self._box_scale=self._box_size/1.3
        self._box_density=torch.tensor([case[1]/(case[0][0]*case[0][1]*case[0][2]) for case in CASES],device=self.device)[:,None].expand(n,o).reshape(-1)
        assets={};self._box_assets=[]
        for size,mass,clip in CASES:
            key=tuple(size)+(mass,)
            if key not in assets:
                options=gymapi.AssetOptions();options.density=mass/(size[0]*size[1]*size[2])
                options.angular_damping=options.linear_damping=.01
                options.default_dof_drive_mode=gymapi.DOF_MODE_NONE
                asset=self.gym.create_box(self.sim,*size,options)
                props=self.gym.get_asset_rigid_shape_properties(asset)
                for prop in props:prop.friction=.6;prop.restitution=0.
                self.gym.set_asset_rigid_shape_properties(asset,props);assets[key]=asset
            self._box_assets.extend([assets[key]]*o)


args=get_args();assert args.test and args.eval and args.headless
args.num_envs=len(CASES)
cfg,train,_=load_cfg(args);cfg['args']=args;cfg['env']['motion_file']=args.motion_file
cfg['env']['enableTrackInitState']=True
sim=parse_sim_params(args,cfg,train);sim.physx.num_threads=1;set_seed(42)
task=RectAnalysis(cfg,sim,args.physics_engine,args.device,args.device_id,True)
out=Path(args.output_path);out.mkdir(parents=True,exist_ok=True)
donor_path=Path(__file__).resolve().parents[6]/'tokenhsi/data/push_fixed130_fixed_face_midpoint_goal3to5_bank.json'
donor_raw=donor_path.read_bytes();donor=json.loads(donor_raw)['candidates']['push']
donor_hash=hashlib.sha256(donor_raw).hexdigest()
height_bank_path=Path(__file__).resolve().parents[6]/'tokenhsi/data/push_fixed130_point_zband_heightmatch_goal2to5_bank.json'
height_bank=json.loads(height_bank_path.read_text())['candidates']['push']
point_bank_path=Path(__file__).resolve().parents[6]/'tokenhsi/data/push_fixed130_point_goal3to5_bank.json'
point_bank=json.loads(point_bank_path.read_text())['candidates']['push']
try:
    n=len(CASES);device=task.device;ids=torch.arange(n,device=device)
    task._sample_episode_graph(ids)
    sizes=torch.tensor([case[0] for case in CASES],device=device)
    masses=torch.tensor([case[1] for case in CASES],device=device)
    clips=torch.tensor([case[2] for case in CASES],device=device,dtype=torch.long)
    selected={}
    for mid in MOTION_IDS:
        available=([r for r in height_bank if r['motion_id']==mid]
                   or [r for r in point_bank if r['motion_id']==mid]
                   or [r for r in donor if r['motion_id']==mid])
        selected[mid]=min(available,key=lambda r:abs(r['phase']-.65))
    times=torch.tensor([selected[case[2]]['time'] for case in CASES],device=device)
    lib=task._motion_lib['push'];hands=task._key_body_ids[:2];feet=task._key_body_ids[2:]
    joints=task.gym.get_actor_dof_names(task.envs[0],task.humanoid_handles_all[0][0])
    props=task.gym.get_actor_dof_properties(task.envs[0],task.humanoid_handles_all[0][0])
    kp=torch.tensor(props['stiffness'],device=device)
    arm_ids=torch.tensor([i for i,name in enumerate(joints) if 'shoulder' in name or 'elbow' in name],device=device)
    ankles=[joints.index(side+'_ankle_y') for side in ('right','left')]
    hips=[joints.index(side+'_hip_y') for side in ('right','left')]
    lower=torch.tensor(props['lower'],device=device);upper=torch.tensor(props['upper'],device=device)
    jac=gymtorch.wrap_tensor(task.gym.acquire_jacobian_tensor(task.sim,'humanoid_0'))
    actual_masses=[]
    for e,case in enumerate(CASES):
        actor=task._box_handles[4*e]
        physical=task.gym.get_actor_rigid_body_properties(task.envs[e],actor)[0]
        assert abs(physical.mass-case[1])<.001
        assert all(abs(p.friction-.6)<1e-5 for p in task.gym.get_actor_rigid_shape_properties(task.envs[e],actor))
        actual_masses.append(physical.mass)
    reset_geometry=None
    if os.environ.get('PUSH_RECT_DEPTH180')=='1':
        import yaml
        from utils.push_reset_geometry import PushResetGeometry
        safe_env=yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/stage1_push_depth180_safe.yaml').read_text())['env']
        names=task.gym.get_actor_rigid_body_names(task.envs[0],task.humanoid_handles_all[0][0])
        reset_geometry=PushResetGeometry(safe_env,names,device)
    rows=[];actions=torch.zeros(n*2,task.num_actions,device=device)
    absolute=os.environ.get('PUSH_RECT_ABSOLUTE_HEIGHTS')
    settings=[float(v) for v in absolute.split(',')] if absolute else HEIGHT_FRACTIONS
    if os.environ.get('PUSH_RECT_INCLUDE_NATIVE')=='1':settings=settings+['amp']
    for fraction,load_ratio in itertools.product(settings,[1.,2.]):
        native_height=fraction=='amp'
        root,rot,dof,vel,ang,dof_vel,_=lib.get_motion_state(clips,times)
        body,rotation,velocity,angular=lib.get_motion_state_max(clips,times)
        height=body[:,hands,2].mean(-1) if native_height else (torch.full_like(sizes[:,2],fraction) if absolute else sizes[:,2]*fraction)
        source_delta_z=body[:,hands[0],2]-body[:,hands[1],2]
        shift=height-body[:,hands,2].mean(-1)
        desired_heights=body[:,hands,2]+shift[:,None]
        # Preserve the source hand-height difference; only the requested mean changes.
        for hand_index,hand in ([] if native_height else enumerate(hands.cpu().tolist())):
            body,rotation,velocity,angular,dof,dof_vel=lower_push_arms(body,rotation,velocity,angular,dof,dof_vel,
                lib._motions[0].skeleton_tree.parent_indices.tolist(),[hand],task._dof_body_ids,task._dof_offsets,desired_heights[:,hand_index])
        task._box_states[:]=0;task._box_states[...,6]=1
        task._box_states[...,:2]=task._env_origins[:,:2,None].transpose(1,2)
        task._box_states[:,:,2]=sizes[:,None,2]/2
        for box in range(1,4):task._box_states[:,box,0]+=50+box*5
        point=task._box_states[:,0,:3].clone();point[:,1]-=sizes[:,1]/2;point[:,2]=height
        delta=body[:,hands[0]]-body[:,hands[1]]
        gap=delta[:,:2].norm(dim=-1)
        offset=torch.stack((gap/2,torch.zeros_like(gap),delta[:,2]/2),-1)
        targets=point[:,None]+torch.stack((offset,-offset),1)
        body,rotation,velocity,angular,_=align_push_pose(body,rotation,velocity,angular,targets,hands.cpu().tolist(),.06)
        if reset_geometry is not None:
            body=body.clone()
            floor_shift=reset_geometry.floor_shift(body,rotation)
            body[:,:,2]+=floor_shift[:,None]
            if native_height:
                height=body[:,hands,2].mean(-1)
                point[:,2]=height
            shift=reset_geometry.rear_shift(body,rotation,task._box_states[:,0],sizes)
            body[:, :, 1]-=shift[:,None]
            assert not reset_geometry.invalid(body,rotation,task._box_states[:,0],sizes).any()
        init_midpoint=body[:,hands].mean(1)
        init_point_error=(init_midpoint-point).norm(dim=-1)
        init_z_error=(init_midpoint[:,2]-height).abs()
        init_joint_bad=((dof<lower-.01)|(dof>upper+.01)).any(-1)
        task._set_pose(ids,torch.zeros_like(ids),body,rotation,velocity*0,angular*0,dof,dof_vel*0)
        # Unused actor stays physically separate and receives no task-dependent force.
        task._humanoid_root_states[:,1,:2]=task._env_origins[:,:2]+100
        task._humanoid_root_states[:,1,7:]=0
        task._dof_vel[:]=0;task._push_hits.zero_()
        task._every_env_init_dof_pos[:,0]=dof.clamp(lower,upper)
        task._every_env_init_dof_pos[:,1]=0
        task._tar_pos[:]=task._box_states[:,:2,:3];task._tar_pos[:,:,1]+=2
        task._reset_env_tensors(ids)
        # All initial states are indexed into the simulator before this settling tick.
        start=task._box_states[:,0,:3].clone()
        task.pre_physics_step(actions);task.gym.simulate(task.sim);task.gym.fetch_results(task.sim,True);task._refresh_sim_tensors()
        zeros=torch.zeros(n,device=device)
        stats={name:zeros.clone() for name in ('max_tilt_deg','max_forward_m','paired_contact_ticks','foot_supported_ticks','nonhand_contact_ticks',
            'peak_horizontal_hand_n','peak_vertical_hand_n','point_error_sum','height_sum','height_difference_peak_m','midpoint_error_peak_m')}
        stats['max_tilt_deg']=tilt_degrees(task._box_states[:,0,3:7])
        stats['min_root_height_m']=torch.full_like(zeros,float('inf'))
        traces=[];rest=dof.clone();reference=dof.clone();ticks=300
        other=torch.ones(task.num_bodies,dtype=torch.bool,device=device);other[hands]=False;other[feet]=False
        for tick in range(ticks):
            if TRACK_REFERENCE and tick%4==0:
                # Actual AMP leg/torso joint trajectory; root remains a free dynamic body.
                reference=lib.get_motion_state(clips,(times+tick/60).clamp_max(lib._motion_lengths[clips]))[2]
            task.gym.refresh_jacobian_tensors(task.sim)
            body_now=task._rigid_body_pos[:,0];box=task._box_states[:,0]
            local=box.new_zeros(n,3);local[:,1]=-sizes[:,1]/2;local[:,2]=height-sizes[:,2]/2
            normal=rotate(box[:,3:7],box.new_tensor([0.,1.,0.]).expand(n,3))
            wanted=box[:,:3]+rotate(box[:,3:7],local)+normal*.025
            midpoint=body_now[:,hands].mean(1)
            error=(wanted-midpoint).clamp(-.08,.08)
            jmean=jac[:,hands,:3,6:].mean(1)
            j=jmean[:,:,arm_ids]
            correction=(j.transpose(1,2)@torch.linalg.solve(j@j.transpose(1,2)+torch.eye(3,device=device)*.01,error[:,:,None])).squeeze(-1)
            target=reference.clone()
            target[:,arm_ids]=task._dof_pos[:,0,arm_ids]+correction.clamp(-.1,.1)+.03*(rest[:,arm_ids]-task._dof_pos[:,0,arm_ids])
            ramp=min(1.,tick/30.)
            target[:,ankles]+=.15*ramp;target[:,hips]+=.2*ramp
            load_gain=max(0.,min(1.,(tick-30)/60.)) if tick<180 else max(0.,1-(tick-180)/30.)
            torque=jmean[:,1]*masses[:,None]*9.81*load_ratio*load_gain
            target+=torque/kp.clamp_min(1.)
            task._every_env_init_dof_pos[:,0]=target.clamp(lower,upper)
            task.pre_physics_step(actions)
            task.gym.simulate(task.sim);task.gym.fetch_results(task.sim,True);task._refresh_sim_tensors()
            body_now=task._rigid_body_pos[:,0];box=task._box_states[:,0]
            tilt=tilt_degrees(box[:,3:7]);forward=box[:,1]-start[:,1]
            inv=box[:,3:7].clone();inv[:,:3]*=-1
            hand_local=rotate(inv[:,None].expand(n,2,4),body_now[:,hands]-box[:,None,:3])
            normal=rotate(box[:,3:7],box.new_tensor([0.,1.,0.]).expand(n,3))
            reaction=-(task._contact_forces[:,0,hands]*normal[:,None]).sum(-1)
            near=(hand_local[:,:,1]+sizes[:,None,1]/2).abs()<.15
            near&=(hand_local[:,:,0].abs()<sizes[:,None,0]/2+.05)&(hand_local[:,:,2].abs()<sizes[:,None,2]/2+.05)
            paired=(near&(reaction>.5)).all(-1)
            midpoint=body_now[:,hands].mean(1);actual_point=box[:,:3]+rotate(box[:,3:7],local)
            point_error=(midpoint-actual_point).norm(dim=-1)
            support=(task._contact_forces[:,0,feet,2]>1.).all(-1)
            local_other=rotate(inv[:,None].expand(-1,int(other.sum()),-1),body_now[:,other]-box[:,None,:3])
            near_other=(local_other.abs()<sizes[:,None]/2+.10).all(-1)
            nonhand=(near_other&(task._contact_forces[:,0,other,:2].norm(dim=-1)>1.)).any(-1)
            stats['max_tilt_deg']=torch.maximum(stats['max_tilt_deg'],tilt)
            stats['max_forward_m']=torch.maximum(stats['max_forward_m'],forward)
            stats['paired_contact_ticks']+=paired;stats['foot_supported_ticks']+=support&paired
            stats['nonhand_contact_ticks']+=nonhand
            stats['min_root_height_m']=torch.minimum(stats['min_root_height_m'],task._humanoid_root_states[:,0,2])
            stats['peak_horizontal_hand_n']=torch.maximum(stats['peak_horizontal_hand_n'],reaction.clamp_min(0).sum(-1))
            stats['peak_vertical_hand_n']=torch.maximum(stats['peak_vertical_hand_n'],task._contact_forces[:,0,hands,2].sum(-1).abs())
            stats['point_error_sum']+=point_error*paired;stats['height_sum']+=midpoint[:,2]*paired
            stats['height_difference_peak_m']=torch.maximum(stats['height_difference_peak_m'],(body_now[:,hands[0],2]-body_now[:,hands[1],2]).abs()*paired)
            stats['midpoint_error_peak_m']=torch.maximum(stats['midpoint_error_peak_m'],point_error*paired)
            assert torch.isfinite(task._box_states).all() and torch.isfinite(task._dof_pos).all()
            traces.append(torch.cat((box[:,:7],midpoint,point_error[:,None],paired[:,None],support[:,None],task._humanoid_root_states[:,0,2,None],nonhand[:,None]),-1).cpu())
            if tick%60==0:print('RECT_HUMANOID',fraction,load_ratio,tick,ticks,'cases',n,flush=True)
        values={key:value.cpu().tolist() for key,value in stats.items()}
        final=(task._box_states[:,0,1]-start[:,1]).cpu().tolist()
        initial_error=init_point_error.cpu().tolist();initial_z=init_z_error.cpu().tolist();joint_bad=init_joint_bad.cpu().tolist()
        for e,(size,mass,clip) in enumerate(CASES):
            row=dict(size=list(size),mass=mass,actual_mass_kg=actual_masses[e],motion_id=clip,reference_time_s=float(times[e]),height_fraction=float(height[e])/size[2],
                height_mode='amp_reference' if native_height else ('absolute' if absolute else 'fraction'),height_setting=fraction,initial_collider_clear=reset_geometry is not None,
                command_midpoint_z_m=float(height[e]),load_command_over_mg=load_ratio,final_forward_m=final[e],initial_midpoint_error_m=initial_error[e],
                initial_midpoint_z_error_m=initial_z[e],initial_joint_limit_violation=joint_bad[e])
            row.update({key:value[e] for key,value in values.items()})
            row['contact_mean_midpoint_error_m']=row['point_error_sum']/max(1,row['paired_contact_ticks'])
            row['contact_mean_midpoint_z_m']=row['height_sum']/max(1,row['paired_contact_ticks'])
            valid=(not joint_bad[e] and initial_z[e]<=.03 and row['min_root_height_m']>=.65 and row['paired_contact_ticks']>=15
                   and row['foot_supported_ticks']>=6 and row['contact_mean_midpoint_error_m']<=.10)
            row['valid_supported_hand_push']=bool(valid and row['nonhand_contact_ticks']==0)
            for bound in (2,5):row[f'stable_{bound}deg_move_10cm']=bool(row['valid_supported_hand_push'] and row['max_tilt_deg']<=bound and row['final_forward_m']>=.10)
            rows.append(row)
        import numpy as np
        height_label='amp' if native_height else f'{fraction:.2f}'
        np.savez_compressed(out/f'trace_h{height_label}_load{load_ratio:.1f}.npz',trace=torch.stack(traces).numpy(),start_box_xyz=start.cpu().numpy(),
            columns=np.array(['box_x','box_y','box_z','qx','qy','qz','qw','mid_x','mid_y','mid_z','point_error','paired_contact','feet_supported','root_z','nonhand_near_box']),dt_s=1/60)
        (out/'audit.partial.json').write_text(json.dumps(dict(cases=rows),allow_nan=False)+'\n')
        print('RECT_HUMANOID_GROUP_DONE',fraction,load_ratio,'strict2',sum(r['stable_2deg_move_10cm'] for r in rows[-n:]),flush=True)
    assert hashlib.sha256(donor_path.read_bytes()).hexdigest()==donor_hash
    report=dict(gpu=6,training=False,policy_used=False,humanoid_used=True,external_box_force=False,
        reference_motion_tracking='AMP leg/torso joint replay' if TRACK_REFERENCE else 'AMP screened initial posture hold',arm_control='Mean point only; no per-hand position or spacing target',
        mass_mode='same candidate mass across all XYZ sizes',box_friction=.6,floor_friction=1.,dt_s=1/60,substeps=2,duration_s=5,
        motion_count=len(MOTION_IDS),source_bank=str(donor_path),source_bank_sha256=donor_hash,source_bank_unchanged=True,
        limits='Finite shape/mass/height/load grid and one selected initial frame per21 clip. Whole-body PD and midpoint Jacobian force bias are experimental controls, not a trained policy or arbitrary-force guarantee. Net hand reactions plus geometry are not pair-resolved sensors; nonhand exclusions are conservative. Internal substep peaks are not sampled.',cases=rows)
    (out/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (out/'cases.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('RECT_HUMANOID_DONE',len(rows),flush=True)
finally:task.gym.destroy_sim(task.sim)
