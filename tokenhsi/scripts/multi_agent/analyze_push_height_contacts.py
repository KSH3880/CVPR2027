"""Two physical spherical hand proxies push cubes via collision, not box force.

The proxies have controlled Cartesian motion and finite drive-force limits.
This isolates hand/box mechanics; it does not establish humanoid reach/balance.
"""
from isaacgym import gymapi, gymtorch
import argparse,csv,itertools,json,os,subprocess
from pathlib import Path
import torch,yaml

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
uuid=subprocess.check_output(['nvidia-smi','-i','6','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
assert os.environ.get('TOKENHSI_GPU') in ('6',uuid)
torch.set_num_threads(1);os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
sizes=[round(1+i*.05,2) for i in range(11)]
heights=[round(.2+i*.025,3) for i in range(25)]
profiles=['equal','unequal_capacity','unequal_height_and_capacity']
cases=list(itertools.product(sizes,heights,[0.,1.,1.5,2.,3.,5.],profiles));n=len(cases)
cfg=yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/push_fixed130_fixed_face_midpoint.yaml').read_text())
gym=gymapi.acquire_gym();params=gymapi.SimParams()
params.dt=1/60;params.substeps=2;params.up_axis=gymapi.UP_AXIS_Z;params.gravity=gymapi.Vec3(0,0,-9.81)
params.use_gpu_pipeline=True;params.physx.use_gpu=True;params.physx.num_threads=1
for k in ('solver_type','num_position_iterations','num_velocity_iterations','contact_offset','rest_offset','bounce_threshold_velocity','max_depenetration_velocity','default_buffer_size_multiplier'):
    setattr(params.physx,k,cfg['sim']['physx'][k])
sim=gym.create_sim(0,-1,gymapi.SIM_PHYSX,params);assert sim is not None
ground=gymapi.PlaneParams();ground.normal=gymapi.Vec3(0,0,1);ground.static_friction=ground.dynamic_friction=1.;gym.add_ground(sim,ground)
assets={}
for s in sizes:
    o=gymapi.AssetOptions();o.density=20/1.3**3;o.angular_damping=o.linear_damping=.01
    assets[s]=gym.create_box(sim,s,s,s,o)
o=gymapi.AssetOptions();o.disable_gravity=True;o.density=1/(4/3*3.141592653589793*.05**3)
hand=gym.create_sphere(sim,.05,o)
for e,(s,h,f,profile) in enumerate(cases):
    env=gym.create_env(sim,gymapi.Vec3(-5,-5,0),gymapi.Vec3(5,5,5),80)
    pose=gymapi.Transform();pose.p.z=s/2
    actor=gym.create_actor(env,assets[s],pose,'box',e,0)
    props=gym.get_actor_rigid_shape_properties(env,actor)
    for v in props:v.friction=.6;v.restitution=0.
    gym.set_actor_rigid_shape_properties(env,actor,props)
    for j in range(2):
        pose=gymapi.Transform();pose.p=gymapi.Vec3((1 if j==0 else -1)*.15*s,-s/2-.08,h*s+(.1*s*(1 if j==0 else -1) if profile=='unequal_height_and_capacity' else 0))
        actor=gym.create_actor(env,hand,pose,'hand'+str(j),e,0)
        props=gym.get_actor_rigid_shape_properties(env,actor)
        for v in props:v.friction=1.;v.restitution=0.
        gym.set_actor_rigid_shape_properties(env,actor,props)
gym.prepare_sim(sim)
root=gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim)).view(n,3,13)
contact=gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(sim)).view(n,3,3)
try:
    # Isaac Gym root tensors are not populated immediately by prepare_sim.
    # Refresh after one physics tick before using poses as Cartesian targets.
    gym.simulate(sim);gym.fetch_results(sim,True)
    gym.refresh_actor_root_state_tensor(sim);gym.refresh_net_contact_force_tensor(sim)
    device=root.device;d=torch.tensor([[s,h,f,profiles.index(pr)] for s,h,f,pr in cases],device=device)
    side,height,ratio,profile=d.unbind(-1);mass=20*(side/1.3)**3
    torch.testing.assert_close(root[:,1:,2].mean(-1),side*height,rtol=0,atol=1e-5)
    torch.testing.assert_close(root[:,0,6],torch.ones(n,device=device),rtol=0,atol=1e-5)
    start=root.clone();target=start[:,1:,:3].clone()
    cap=ratio*mass*9.81
    split=torch.where(profile>0,.75,.5)
    caps=torch.stack((cap*split,cap*(1-split)),-1)
    max_tilt=torch.zeros(n,device=device);peak_force=torch.zeros_like(max_tilt);peak_vertical=torch.zeros_like(max_tilt)
    contacts=torch.zeros_like(max_tilt);force_sum=torch.zeros_like(max_tilt);height_sum=torch.zeros_like(max_tilt)
    min_z=torch.full_like(max_tilt,float('inf'));max_z=torch.zeros_like(max_tilt)
    max_gap=torch.zeros_like(max_tilt);peak_drive=torch.zeros_like(max_tilt)
    force=torch.zeros(n,3,3,device=device)
    for tick in range(300):
        # Move 0.9m over3s, then hold another2s to catch residual rocking.
        target[:,:,1]=start[:,1:,1]+.3*min((tick+1)/60,3.)
        desired=2000*(target-root[:,1:,:3])-60*root[:,1:,7:10]
        force[:,1:,0]=desired[:,:,0].clamp(-caps,caps)
        force[:,1:,1]=desired[:,:,1].clamp(-caps,caps)
        force[:,1:,2]=desired[:,:,2].clamp(-.2*caps,.2*caps)
        force[:,0]=0
        gym.apply_rigid_body_force_tensors(sim,gymtorch.unwrap_tensor(force.contiguous()),None,gymapi.ENV_SPACE)
        gym.simulate(sim);gym.fetch_results(sim,True)
        gym.refresh_actor_root_state_tensor(sim);gym.refresh_net_contact_force_tensor(sim)
        q=root[:,0,3:7]
        tilt=torch.acos((1-2*(q[:,0].square()+q[:,1].square())).clamp(-1,1))*180/torch.pi
        max_tilt=torch.maximum(max_tilt,tilt)
        reaction=(-contact[:,1:,1]).clamp_min(0)
        touching=(reaction>.5).all(-1)
        fy=reaction.sum(-1);fz=contact[:,1:,2].sum(-1).abs()
        z=root[:,1:,2].mean(-1)
        contacts+=touching;force_sum+=fy*touching;height_sum+=z*touching
        min_z=torch.minimum(min_z,torch.where(touching,z,float('inf')))
        max_z=torch.maximum(max_z,z*touching)
        peak_force=torch.maximum(peak_force,fy);peak_vertical=torch.maximum(peak_vertical,fz)
        peak_drive=torch.maximum(peak_drive,force[:,1:,1].sum(-1))
        max_gap=torch.maximum(max_gap,(root[:,1,2]-root[:,2,2]).abs())
        assert torch.isfinite(root).all()
        if tick%60==0:print('HAND_CONTACT_PROGRESS',tick,300,'cases',n,flush=True)
    dy=(root[:,0,1]-start[:,0,1]).cpu().tolist()
    values={k:v.cpu().tolist() for k,v in dict(tilt=max_tilt,peak=peak_force,vertical=peak_vertical,ticks=contacts,force=force_sum,zsum=height_sum,zmin=min_z,zmax=max_z,gap=max_gap,drive=peak_drive).items()}
    rows=[]
    for i,(s,h,f,pr) in enumerate(cases):
        count=values['ticks'][i]
        rows.append(dict(side_m=s,mass_kg=20*(s/1.3)**3,command_midpoint_z_m=s*h,command_z_over_side=h,
            drive_limit_over_mg=f,drive_limit_N=f*20*(s/1.3)**3*9.81,profile=pr,
            final_forward_m=dy[i],max_tilt_deg=values['tilt'][i],paired_contact_ticks=count,
            mean_horizontal_contact_force_N=values['force'][i]/max(1,count),peak_horizontal_contact_force_N=values['peak'][i],
            peak_vertical_contact_force_N=values['vertical'][i],peak_horizontal_drive_N=values['drive'][i],
            mean_contact_midpoint_z_m=values['zsum'][i]/max(1,count),
            min_contact_midpoint_z_m=values['zmin'][i] if count else None,max_contact_midpoint_z_m=values['zmax'][i],
            max_hand_height_difference_m=values['gap'][i],
            moved_10cm_max_tilt_le5=bool(dy[i]>=.1 and values['tilt'][i]<=5 and count>=6)))
    controls=[r for r in rows if r['drive_limit_over_mg']==0]
    assert all(abs(r['final_forward_m'])<.001 and r['max_tilt_deg']<.1 for r in controls)
    a.output.mkdir(parents=True,exist_ok=True)
    report=dict(gpu=6,num_cases=n,real_rigid_contact=True,humanoid_used=False,external_box_force=False,
        duration_s=5,drive_duration_s=3,command_travel_m=.9,command_speed_m_s=.3,dt=1/60,substeps=2,
        hand_proxy='Two1kg gravity-free spheres radius0.05m, friction1.0, finite2000N/m Cartesian drive and60Ns/m damping; per-hand horizontal cap and vertical cap20percent. Box receives contact only.',
        box_friction=.6,floor_friction=1.,density=20/1.3**3,
        limits='Physical hand proxies isolate contact mechanics, not humanoid reach/balance or neural-policy success. Finite tested drive/profile/time grid; no absolute guarantee. Net pad contact force is box-only for valid-height upright trials; 1/60s sampling may miss internal substep peaks.',
        cases=rows,zero_force_controls_passed=True)
    (a.output/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (a.output/'cases.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('HAND_CONTACT_RESULT',n,sum(r['moved_10cm_max_tilt_le5'] for r in rows),flush=True)
finally:gym.destroy_sim(sim)
