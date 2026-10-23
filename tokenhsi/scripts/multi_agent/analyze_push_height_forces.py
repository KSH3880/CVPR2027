"""Known two-point wrench probes complement actual humanoid contact trials.

These probes isolate box mechanics; they are not real humanoid hand contacts.
No policy, RSI bank, learned weights or training configuration is modified.
"""
from isaacgym import gymapi, gymtorch
import argparse
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import torch
import yaml
from utils.door_reference import rotate

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
args = parser.parse_args()
physical_uuid = subprocess.check_output(['nvidia-smi','-i','6','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
assert os.environ.get('TOKENHSI_GPU') in ('6',physical_uuid)
torch.set_num_threads(1)
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
sizes = [round(1+i*.05,2) for i in range(11)]
heights = [round(.1+i*.025,3) for i in range(33)]
ratios = [0., .8, 1., 1.5, 2., 3., 5.]
profiles = ['step_equal','ramp_equal','pulse_equal','ramp_high_hand75_up20','ramp_high_hand75_down20']
cases = list(itertools.product(sizes,heights,ratios,profiles))
config = yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/push_fixed130_fixed_face_midpoint.yaml').read_text())
gym = gymapi.acquire_gym()
params = gymapi.SimParams()
params.dt=1/60;params.substeps=config['sim']['substeps'];params.up_axis=gymapi.UP_AXIS_Z
params.gravity=gymapi.Vec3(0,0,-9.81)
params.use_gpu_pipeline=True;params.physx.use_gpu=True
for key in ('solver_type','num_position_iterations','num_velocity_iterations','contact_offset','rest_offset','bounce_threshold_velocity','max_depenetration_velocity','default_buffer_size_multiplier'):
    setattr(params.physx,key,config['sim']['physx'][key])
params.physx.num_threads=1
sim=gym.create_sim(0,-1,gymapi.SIM_PHYSX,params)
assert sim is not None
plane=gymapi.PlaneParams();plane.normal=gymapi.Vec3(0,0,1)
plane.static_friction=plane.dynamic_friction=1.;plane.restitution=0
gym.add_ground(sim,plane)
assets={}
for side in sizes:
    options=gymapi.AssetOptions();options.density=20/1.3**3
    options.angular_damping=options.linear_damping=.01
    assets[side]=gym.create_box(sim,side,side,side,options)
actual_mass={};actual_com={};actual_inertia={}
for e,(side,height,ratio,profile) in enumerate(cases):
    env=gym.create_env(sim,gymapi.Vec3(-10,-10,0),gymapi.Vec3(10,10,10),128)
    pose=gymapi.Transform();pose.p.z=side/2
    actor=gym.create_actor(env,assets[side],pose,'box',e,0)
    shapes=gym.get_actor_rigid_shape_properties(env,actor)
    for shape in shapes:shape.friction=.6;shape.restitution=0.
    gym.set_actor_rigid_shape_properties(env,actor,shapes)
    if side not in actual_mass:
        p=gym.get_actor_rigid_body_properties(env,actor)[0]
        actual_mass[side]=p.mass
        actual_com[side]=[p.com.x,p.com.y,p.com.z]
        actual_inertia[side]=[p.inertia.x.x,p.inertia.y.y,p.inertia.z.z]
        assert abs(p.mass-20*(side/1.3)**3)<.001
gym.prepare_sim(sim)
state=gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim)).view(-1,13)
try:
    device=state.device;n=len(cases)
    for _ in range(60):gym.simulate(sim);gym.fetch_results(sim,True)
    gym.refresh_actor_root_state_tensor(sim)
    start=state[:,:3].clone()
    data=torch.tensor([[s,h,r,profiles.index(p)] for s,h,r,p in cases],device=device)
    side,fraction,ratio,profile=data.unbind(-1)
    mass=20*(side/1.3)**3
    force_total=ratio*mass*9.81
    asymmetric=profile>=3
    split=torch.where(asymmetric,.75,.5)
    vertical=torch.where(profile==3,.2,torch.where(profile==4,-.2,0.))
    local=torch.zeros(n,2,3,device=device)
    local[:,:,0]=side[:,None]*torch.tensor([-.15,.15],device=device)
    local[:,:,1]=-side[:,None]/2
    local[:,:,2]=(fraction-.5)[:,None]*side[:,None]
    local[:,0,2]+=torch.where(asymmetric,.1*side,0.)
    local[:,1,2]-=torch.where(asymmetric,.1*side,0.)
    max_tilt=torch.zeros(n,device=device);max_tilt_during=torch.zeros_like(max_tilt)
    contact_force=gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(sim)).view(n,3)
    initial_roll=torch.zeros_like(max_tilt)
    for tick in range(300):
        t=tick/60
        # 0--3s forcing, followed by 2s release; step/ramp/pulsed force histories.
        active=t<3.
        gain=torch.where(profile==0,torch.ones_like(profile),torch.full_like(profile,min(1.,t/.5)))
        gain=torch.where(profile==2,torch.full_like(gain,1. if tick%30<15 else 0.),gain)*active
        force=torch.zeros(n,2,3,device=device)
        force[:,0,1]=force_total*split*gain
        force[:,1,1]=force_total*(1-split)*gain
        force[:,:,2]=force[:,:,1]*vertical[:,None]
        r=rotate(state[:,None,3:7].expand(-1,2,-1),local)
        torques=torch.cross(r,force,dim=-1).sum(1).contiguous()
        result=force.sum(1).contiguous()
        gym.apply_rigid_body_force_tensors(sim,gymtorch.unwrap_tensor(result),gymtorch.unwrap_tensor(torques),gymapi.ENV_SPACE)
        gym.simulate(sim);gym.fetch_results(sim,True)
        gym.refresh_actor_root_state_tensor(sim)
        tilt=torch.acos((1-2*(state[:,3].square()+state[:,4].square())).clamp(-1,1))*180/torch.pi
        max_tilt=torch.maximum(max_tilt,tilt)
        if active:max_tilt_during=torch.maximum(max_tilt_during,tilt)
        assert torch.isfinite(state).all()
        if tick%60==0:print('FORCE_HEIGHT_PROGRESS',tick,300,'cases',n,flush=True)
    rows=[]
    final=(state[:,:3]-start).cpu().tolist()
    angles=max_tilt.cpu().tolist();during=max_tilt_during.cpu().tolist()
    for i,(s,h,r,p) in enumerate(cases):
        rows.append(dict(side_m=s,mass_kg=actual_mass[s],midpoint_z_over_side=h,midpoint_z_m=s*h,
            force_over_mg=r,total_horizontal_force_N=r*actual_mass[s]*9.81,profile=p,
            final_forward_m=final[i][1],final_vertical_m=final[i][2],max_tilt_deg=angles[i],max_forcing_tilt_deg=during[i],
            moved_10cm_max_tilt_le5=bool(final[i][1]>=.1 and angles[i]<=5)))
    # A no-force control exists for every size/height/profile.
    controls=[r for r in rows if r['force_over_mg']==0]
    assert all(abs(r['final_forward_m'])<.001 and r['max_tilt_deg']<.1 for r in controls)
    summaries=[]
    for s in sizes:
        stable=[]
        for h in heights:
            group=[r for r in rows if r['side_m']==s and r['midpoint_z_over_side']==h and r['force_over_mg']>=1]
            if all(r['moved_10cm_max_tilt_le5'] for r in group):stable.append(h)
        summaries.append(dict(side_m=s,mass_kg=actual_mass[s],all_profiles_forces_1to5mg_tested_heights_over_side=stable,
            all_profiles_forces_1to5mg_tested_heights_m=[round(x*s,4) for x in stable]))
    report=dict(gpu=6,num_cases=n,training=False,humanoid_used=False,known_external_wrenches=True,
        duration_s=5,force_duration_s=3,release_duration_s=2,dt=1/60,substeps=params.substeps,
        size_range_m=[1.,1.5],box_friction=.6,plane_friction=1.,density=20/1.3**3,
        actual_mass=actual_mass,actual_com=actual_com,actual_inertia=actual_inertia,
        profiles=profiles,force_over_mg=ratios,
        limits='Known world-horizontal forces and +/-20% vertical components at material-attached points; asymmetric cases use 75/25 force split and hand heights +/-0.1*side. These are external wrenches, not humanoid contacts. <=5deg sampled at1/60s with2substeps; finite grid/duration, not an absolute guarantee.',
        zero_force_controls_passed=True,summaries=summaries,cases=rows)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    import csv
    with (args.output/'cases.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(summaries,indent=2),flush=True)
finally:gym.destroy_sim(sim)
