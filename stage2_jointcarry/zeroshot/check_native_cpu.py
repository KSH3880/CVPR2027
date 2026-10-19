"""Validate original-scale assets on CPU only, including rigid topology and mass."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json
from pathlib import Path
import numpy as np
from isaacgym import gymapi

base=Path(__file__).resolve().parents[2]/'runs/jointcarry_zeroshot/native_assets'
metadata=json.loads((base/'catalog.json').read_text())
gym=gymapi.acquire_gym();p=gymapi.SimParams();p.use_gpu_pipeline=False;p.physx.use_gpu=False
p.up_axis=gymapi.UP_AXIS_Z;p.gravity=gymapi.Vec3(0,0,-9.81)
sim=gym.create_sim(0,-1,gymapi.SIM_PHYSX,p)
assert sim is not None
report=[]
try:
    ground=gymapi.PlaneParams();ground.normal=gymapi.Vec3(0,0,1);gym.add_ground(sim,ground)
    actors=[]
    for meta in metadata:
        folder=base/meta['name'];opt=gymapi.AssetOptions()
        opt.vhacd_enabled=True;opt.vhacd_params.max_convex_hulls=16
        opt.vhacd_params.max_num_vertices_per_ch=64;opt.vhacd_params.resolution=100000
        opt.mesh_normal_mode=gymapi.COMPUTE_PER_VERTEX
        asset=gym.load_asset(sim,str(folder),'asset.urdf',opt)
        assert gym.get_asset_rigid_body_count(asset)==1 and gym.get_asset_dof_count(asset)==0
        env=gym.create_env(sim,gymapi.Vec3(-4,-4,0),gymapi.Vec3(4,4,4),4)
        pose=gymapi.Transform();pose.p.z=meta['size'][2]/2+.2
        actor=gym.create_actor(env,asset,pose,meta['name'],len(actors),0)
        props=gym.get_actor_rigid_body_properties(env,actor)
        assert abs(props[0].mass-meta['mass'])<1e-3
        actors.append((env,actor,meta,gym.get_asset_rigid_shape_count(asset)))
    gym.prepare_sim(sim)
    for _ in range(180):gym.simulate(sim);gym.fetch_results(sim,True)
    for env,actor,meta,count in actors:
        state=gym.get_actor_rigid_body_states(env,actor,gymapi.STATE_ALL)
        z=float(state['pose']['p']['z'][0]);assert np.isfinite(z) and z>0
        report.append(dict(name=meta['name'],size=meta['size'],mass=meta['mass'],bodies=1,dofs=0,shapes=count,final_z=z,steps=180,device='cpu'))
finally:gym.destroy_sim(sim)
(base/'cpu_physics_check.json').write_text(json.dumps(report,indent=2)+'\n')
print('NATIVE_CPU_CHECK',json.dumps(report),flush=True)
