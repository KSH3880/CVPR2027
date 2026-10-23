"""Rectangular equal-mass box sweep with two real contact proxies.

The mean hand position is the designated rear-face point. Individual height,
spacing and force sharing vary while preserving the commanded midpoint.
"""
from isaacgym import gymapi, gymtorch
import argparse
import csv
import itertools
import json
import os
import subprocess
from pathlib import Path
import torch
import yaml
from push_stability_common import SHAPES, MASSES, HEIGHT_FRACTIONS, tilt_degrees, static_envelope

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--dense-mass', type=float)
parser.add_argument('--dense-fraction', type=float)
parser.add_argument('--height-fractions', type=float, nargs='+')
parser.add_argument('--absolute-height', type=float)
parser.add_argument('--depth180-sweep', action='store_true', help='Seven X sizes, Y1.8m and matched X=Y controls, Z1.3m')
parser.add_argument('--absolute-heights', type=float, nargs='+')
parser.add_argument('--profile-mass', type=float, default=20.)
args=parser.parse_args()
gpu_uuid=subprocess.check_output(['nvidia-smi','-i','6','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
assert os.environ.get('TOKENHSI_GPU') in ('6',gpu_uuid)
torch.set_num_threads(1); os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
shapes=SHAPES; masses=MASSES; heights=HEIGHT_FRACTIONS
if args.height_fractions: heights=args.height_fractions
if args.dense_mass is not None:
    shapes=list(itertools.product([1.2+i*.05 for i in range(7)],repeat=3))
    masses=[args.dense_mass]
    assert args.dense_fraction is not None or args.absolute_height is not None
    heights=[args.dense_fraction if args.dense_fraction is not None else .4]
if args.depth180_sweep:
    assert args.dense_mass is None and args.absolute_height is None
    shapes=[(1.2+i*.05,y,1.3) for i in range(7) for y in (1.8,1.2+i*.05)]
    assert args.profile_mass > 0 and args.profile_mass < float('inf')
    masses=[args.profile_mass]
    heights=[.5]
profiles=['symmetric','unequal_capacity','unequal_height_and_width']
cases=[dict(size=list(size),mass=mass,height_fraction=height,cap_over_mg=cap,profile=profile)
       for size,mass,height,cap,profile in itertools.product(shapes,masses,heights,[1.,2.],profiles)]
if args.absolute_height is not None:
    assert args.absolute_height > .1
    for case in cases:case['height_fraction']=args.absolute_height/case['size'][2]
if args.absolute_heights:
    assert args.absolute_height is None and all(.1 < h < min(size[2] for size in shapes) for h in args.absolute_heights)
    cases=[dict(case,height_fraction=h/case['size'][2]) for case in cases for h in args.absolute_heights]
cases += [dict(size=list(size),mass=mass,height_fraction=.4,cap_over_mg=0.,profile='symmetric')
          for size,mass in itertools.product(shapes,masses)]
n=len(cases); cfg=yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/stage1_push_no_force.yaml').read_text())
gym=gymapi.acquire_gym(); params=gymapi.SimParams()
params.dt=1/60;params.substeps=2;params.up_axis=gymapi.UP_AXIS_Z;params.gravity=gymapi.Vec3(0,0,-9.81)
params.use_gpu_pipeline=True;params.physx.use_gpu=True;params.physx.num_threads=1
for key,value in cfg['sim']['physx'].items():
    if hasattr(params.physx,key) and key!='num_threads':setattr(params.physx,key,value)
sim=gym.create_sim(0,-1,gymapi.SIM_PHYSX,params);assert sim is not None
plane=gymapi.PlaneParams();plane.normal=gymapi.Vec3(0,0,1);plane.static_friction=plane.dynamic_friction=1.
gym.add_ground(sim,plane)
assets={}; actual={}
for size,mass in itertools.product(shapes,masses):
    options=gymapi.AssetOptions();options.density=mass/(size[0]*size[1]*size[2])
    options.angular_damping=options.linear_damping=.01
    asset=gym.create_box(sim,*size,options)
    props=gym.get_asset_rigid_shape_properties(asset)
    for prop in props:prop.friction=.6;prop.restitution=0.
    gym.set_asset_rigid_shape_properties(asset,props)
    assets[tuple(size),mass]=asset
options=gymapi.AssetOptions();options.disable_gravity=True;options.density=1/(4/3*3.141592653589793*.05**3)
pad=gym.create_sphere(sim,.05,options)
for e,case in enumerate(cases):
    x,y,z=case['size'];mass=case['mass'];h=case['height_fraction']*z
    env=gym.create_env(sim,gymapi.Vec3(-4,-4,0),gymapi.Vec3(4,4,4),80)
    pose=gymapi.Transform();pose.p.z=z/2
    actor=gym.create_actor(env,assets[tuple(case['size']),mass],pose,'box',e,0)
    key=tuple(case['size'])+(mass,)
    if key not in actual:
        p=gym.get_actor_rigid_body_properties(env,actor)[0]
        assert abs(p.mass-mass)<.001
        assert max(abs(p.com.x),abs(p.com.y),abs(p.com.z))<1e-6
        actual[key]=p.mass
    for hand in range(2):
        sign=1 if hand==0 else -1
        wide=case['profile']=='unequal_height_and_width'
        pose=gymapi.Transform();pose.p=gymapi.Vec3(sign*(.3 if wide else .2),-y/2-.08,h+sign*(.1 if wide else 0.))
        actor=gym.create_actor(env,pad,pose,'hand'+str(hand),e,0)
        props=gym.get_actor_rigid_shape_properties(env,actor)
        for p in props:p.friction=1.;p.restitution=0.
        gym.set_actor_rigid_shape_properties(env,actor,props)
gym.prepare_sim(sim)
root=gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim)).view(n,3,13)
contacts=gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(sim)).view(n,3,3)
try:
    gym.simulate(sim);gym.fetch_results(sim,True)
    gym.refresh_actor_root_state_tensor(sim);gym.refresh_net_contact_force_tensor(sim)
    device=root.device
    size=torch.tensor([c['size'] for c in cases],device=device)
    mass=torch.tensor([c['mass'] for c in cases],device=device)
    height=torch.tensor([c['height_fraction'] for c in cases],device=device)*size[:,2]
    torch.testing.assert_close(root[:,1:,2].mean(-1),height,atol=1e-5,rtol=0)
    start=root.clone();target=start[:,1:,:3].clone()
    cap=torch.tensor([c['cap_over_mg'] for c in cases],device=device)*mass*9.81
    split=torch.tensor([.75 if c['profile']=='unequal_capacity' else .5 for c in cases],device=device)
    caps=torch.stack((cap*split,cap*(1-split)),-1)
    zeros=torch.zeros(n,device=device)
    max_tilt=zeros.clone();touch_ticks=zeros.clone();peak_fy=zeros.clone();peak_fz=zeros.clone()
    height_sum=zeros.clone();height_error=zeros.clone();peak_lateral=zeros.clone();max_forward=zeros.clone()
    force=torch.zeros(n,3,3,device=device);traces=[]
    for tick in range(300):
        target[:,:,1]=start[:,1:,1]+.3*min((tick+1)/60,3.)
        desired=2000*(target-root[:,1:,:3])-60*root[:,1:,7:10]
        force[:,1:,0]=desired[:,:,0].clamp(-caps,caps)
        force[:,1:,1]=desired[:,:,1].clamp(-caps,caps)
        force[:,1:,2]=desired[:,:,2].clamp(-.2*caps,.2*caps)
        force[:,0]=0
        gym.apply_rigid_body_force_tensors(sim,gymtorch.unwrap_tensor(force.contiguous()),None,gymapi.ENV_SPACE)
        gym.simulate(sim);gym.fetch_results(sim,True)
        gym.refresh_actor_root_state_tensor(sim);gym.refresh_net_contact_force_tensor(sim)
        tilt=tilt_degrees(root[:,0,3:7]);forward=root[:,0,1]-start[:,0,1]
        touching=((-contacts[:,1:,1])>.5).all(-1)
        midpoint=root[:,1:,:3].mean(1)
        max_tilt=torch.maximum(max_tilt,tilt);max_forward=torch.maximum(max_forward,forward)
        touch_ticks+=touching;peak_fy=torch.maximum(peak_fy,(-contacts[:,1:,1]).clamp_min(0).sum(-1))
        peak_fz=torch.maximum(peak_fz,contacts[:,1:,2].sum(-1).abs())
        height_sum+=midpoint[:,2]*touching
        height_error=torch.maximum(height_error,(midpoint[:,2]-height).abs()*touching)
        peak_lateral=torch.maximum(peak_lateral,(root[:,0,0]-start[:,0,0]).abs())
        assert torch.isfinite(root).all()
        traces.append(torch.stack((tilt,forward,midpoint[:,2],touching.float()),-1).cpu())
        if tick%60==0:print('RECT_CONTACT',tick,300,'cases',n,flush=True)
    values={name:value.cpu().tolist() for name,value in dict(max_tilt_deg=max_tilt,final_forward_m=root[:,0,1]-start[:,0,1],max_forward_m=max_forward,
        paired_contact_ticks=touch_ticks,peak_hand_horizontal_n=peak_fy,peak_hand_vertical_n=peak_fz,
        mean_midpoint_z=height_sum/touch_ticks.clamp_min(1),max_midpoint_z_error=height_error,max_lateral_m=peak_lateral).items()}
    rows=[]
    for i,case in enumerate(cases):
        row=dict(case);row.update({key:value[i] for key,value in values.items()})
        row['actual_mass_kg']=actual[tuple(case['size'])+(case['mass'],)]
        row['command_midpoint_z_m']=case['height_fraction']*case['size'][2]
        row['stable_2deg_move_50cm']=row['max_tilt_deg']<=2 and row['final_forward_m']>=.5 and row['paired_contact_ticks']>=30
        row['stable_5deg_move_50cm']=row['max_tilt_deg']<=5 and row['final_forward_m']>=.5 and row['paired_contact_ticks']>=30
        row.update(static_envelope(case['size'],case['mass'],case['height_fraction']))
        rows.append(row)
    controls=[r for r in rows if r['cap_over_mg']==0]
    assert all(abs(r['final_forward_m'])<.002 and r['max_tilt_deg']<.1 for r in controls)
    args.output.mkdir(parents=True,exist_ok=True)
    report=dict(training=False,humanoid_used=False,policy_used=False,external_box_force=False,gpu=6,dt=1/60,substeps=2,
        duration_s=5,drive_s=3,command_travel_m=.9,command_speed_m_s=.3,box_friction=.6,floor_friction=1.,mass_mode='same candidate mass for every XYZ size',
        zero_force_controls_passed=True,cases=rows,
        limitations='Two1kg spheres with finite Cartesian force limits isolate contact physics, not humanoid balance or trained-policy success. All profiles preserve the commanded mean point; individual hand height/spacing differ. Internal substep peaks and untested continuous dimensions are not certified.')
    (args.output/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (args.output/'cases.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import numpy as np
    np.savez_compressed(args.output/'trace.npz',trace=torch.stack(traces).numpy(),columns=np.array(['tilt_deg','forward_m','midpoint_z_m','both_contact']),dt_s=1/60)
    print('RECT_CONTACT_DONE',n,sum(r['stable_2deg_move_50cm'] for r in rows),flush=True)
finally:gym.destroy_sim(sim)
