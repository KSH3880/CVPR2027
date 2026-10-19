"""No humanoids or GPU tensors: validate a local asset using CPU PhysX."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json
from pathlib import Path
from isaacgym import gymapi

root=Path(__file__).resolve().parents[1]
gym=gymapi.acquire_gym()
p=gymapi.SimParams(); p.use_gpu_pipeline=False; p.physx.use_gpu=False
p.up_axis=gymapi.UP_AXIS_Z; p.gravity=gymapi.Vec3(0,0,-9.81)
sim=gym.create_sim(0,-1,gymapi.SIM_PHYSX,p)
if sim is None: raise RuntimeError('CPU simulation creation failed')
try:
    plane=gymapi.PlaneParams(); plane.normal=gymapi.Vec3(0,0,1); gym.add_ground(sim,plane)
    asset_dir=root/'zeroshot/assets/round_table'
    options=gymapi.AssetOptions(); options.vhacd_enabled=True
    options.vhacd_params.max_convex_hulls=16; options.vhacd_params.max_num_vertices_per_ch=64
    options.vhacd_params.resolution=100000
    options.override_com=True; options.override_inertia=True
    options.mesh_normal_mode=gymapi.COMPUTE_PER_VERTEX
    asset=gym.load_asset(sim,str(asset_dir),'asset.urdf',options)
    assert gym.get_asset_rigid_body_count(asset)==1 and gym.get_asset_dof_count(asset)==0
    env=gym.create_env(sim,gymapi.Vec3(-2,-2,0),gymapi.Vec3(2,2,2),1)
    pose=gymapi.Transform(); pose.p.z=.6
    actor=gym.create_actor(env,asset,pose,'payload',0,0)
    gym.prepare_sim(sim)
    for _ in range(120): gym.simulate(sim); gym.fetch_results(sim,True)
    states=gym.get_actor_rigid_body_states(env,actor,gymapi.STATE_ALL)
    props=gym.get_actor_rigid_body_properties(env,actor)
    result=dict(device='cpu',steps=120,bodies=1,dofs=0,shapes=gym.get_asset_rigid_shape_count(asset),mass=float(props[0].mass),final_z=float(states['pose']['p']['z'][0]))
    assert .15<result['final_z']<.3, result
    print('CPU_ASSET_CHECK',json.dumps(result),flush=True)
finally: gym.destroy_sim(sim)
