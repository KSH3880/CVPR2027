"""Show actual training reset snapshots: Box/Table/Chair SIT and CLIMB."""
from pathlib import Path
import argparse,json,time
import isaacgym
from isaacgym import gymapi,gymutil
import numpy as np


def main():
 p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--frames',type=int,default=0);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 d=dict(np.load(a.snapshot/'rsi_reset_snapshot.npz'));meta=json.loads((a.snapshot/'rsi_reset_validation.json').read_text());catalog=json.loads(Path('chair_assets/catalog.json').read_text());assets={Path(e['path']).name:e['path'] for key in ['train','tables'] for e in catalog[key]}
 for i in range(len(meta['scenes'])):
  shift=np.r_[d['objects'][i,0,:2],0.]
  for key in ['root','objects','humans','goal']:d[key][i,...,:3]-=shift
 gym=gymapi.acquire_gym();params=gymapi.SimParams();params.up_axis=gymapi.UP_AXIS_Z;params.gravity=gymapi.Vec3(0,0,-9.81);params.dt=1/60;params.substeps=2;params.physx.solver_type=1;params.physx.num_position_iterations=4;params.physx.contact_offset=.02
 sim=gym.create_sim(0,0,gymapi.SIM_PHYSX,params);assert sim is not None;plane=gymapi.PlaneParams();plane.normal=gymapi.Vec3(0,0,1);plane.static_friction=1;plane.dynamic_friction=1;gym.add_ground(sim,plane)
 opts=gymapi.AssetOptions();opts.default_dof_drive_mode=gymapi.DOF_MODE_POS;human=gym.load_asset(sim,'tokenhsi/data/assets','mjcf/phys_humanoid_v3.xml',opts);envs=[];actors=[];cache={}
 def transform(v):
  pose=gymapi.Transform();pose.p=gymapi.Vec3(*v[:3]);pose.r=gymapi.Quat(*v[3:7]);return pose
 for i,scene in enumerate(meta['scenes']):
  env=gym.create_env(sim,gymapi.Vec3(-8,-8,0),gymapi.Vec3(8,8,4),3);envs.append(env);group=[]
  for j in range(4):
   key=scene['asset_keys'][j];options=gymapi.AssetOptions();options.fix_base_link=bool(key);options.density=100
   if key:
    if key not in cache:
     options.vhacd_enabled=True;options.vhacd_params.resolution=100000;options.vhacd_params.max_convex_hulls=128;options.vhacd_params.max_num_vertices_per_ch=64;options.use_mesh_materials=True;cache[key]=gym.load_asset(sim,assets[key],'asset.urdf',options)
    asset=cache[key]
   else:asset=gym.create_box(sim,*d['sizes'][i,j].tolist(),options)
   gym.create_actor(env,asset,transform(d['objects'][i,j]),'object'+str(j),i,0)
  for j in range(2):
   actor=gym.create_actor(env,human,transform(d['root'][i,j]),'human'+str(j),i,0);group.append(actor);props=gym.get_actor_dof_properties(env,actor);props['driveMode'].fill(gymapi.DOF_MODE_POS);gym.set_actor_dof_properties(env,actor,props)
  actors.append(group)
 gym.prepare_sim(sim)
 def reset():
  for i,(env,group) in enumerate(zip(envs,actors)):
   for j,actor in enumerate(group):
    state=gym.get_actor_rigid_body_states(env,actor,gymapi.STATE_ALL);body=d['humans'][i,j]
    for k,f in enumerate(['x','y','z']):state['pose']['p'][f]=body[:,k];state['vel']['linear'][f]=body[:,7+k];state['vel']['angular'][f]=body[:,10+k]
    for k,f in enumerate(['x','y','z','w']):state['pose']['r'][f]=body[:,3+k]
    gym.set_actor_rigid_body_states(env,actor,state,gymapi.STATE_ALL);joint=gym.get_actor_dof_states(env,actor,gymapi.STATE_ALL);joint['pos']=d['dof'][i,j];joint['vel']=d['dof_vel'][i,j];gym.set_actor_dof_states(env,actor,joint,gymapi.STATE_ALL);gym.set_actor_dof_position_targets(env,actor,np.asarray(d['dof'][i,j],dtype=np.float32))
  gym.simulate(sim);gym.fetch_results(sim,True)
 viewer=gym.create_viewer(sim,gymapi.CameraProperties());assert viewer is not None
 for key,name in [(gymapi.KEY_N,'next'),(gymapi.KEY_P,'prev'),(gymapi.KEY_R,'reset'),(gymapi.KEY_SPACE,'physics')]:gym.subscribe_viewer_keyboard_event(viewer,key,name)
 index=0;paused=True
 def camera():
  s=meta['scenes'][index];owner=s['owner'];center=d['root'][index,owner,:3];gym.viewer_camera_look_at(viewer,envs[index],gymapi.Vec3(*(center+np.array([2,-2.5,1]))),gymapi.Vec3(*center));print('RSI_SCENE',index,json.dumps(s),flush=True)
 reset();camera();small=gymutil.WireframeSphereGeometry(.03,8,8,None,color=(1,0,0));sphere=gymutil.WireframeSphereGeometry(.2,16,16,None,color=(1,1,0));frame=0;print('VIEWER_READY N/P=Box/Table/Chair SIT/CLIMB; SPACE=physics R=reset',flush=True)
 try:
  while not gym.query_viewer_has_closed(viewer):
   for event in gym.query_viewer_action_events(viewer):
    if event.value<=0:continue
    if event.action in ['next','prev']:index=(index+(1 if event.action=='next' else -1))%len(envs);camera()
    elif event.action=='physics':paused=not paused
    elif event.action=='reset':reset();paused=True
   if not paused:gym.simulate(sim);gym.fetch_results(sim,True)
   gym.clear_lines(viewer)
   for i,env in enumerate(envs):
    for goal in d['goal'][i]:
     pose=gymapi.Transform();pose.p=gymapi.Vec3(*goal);gymutil.draw_lines(small,gym,viewer,env,pose);gymutil.draw_lines(sphere,gym,viewer,env,pose)
   gym.step_graphics(sim);gym.draw_viewer(viewer,sim,True);frame+=1
   if frame==30:gym.write_viewer_image_to_file(viewer,str(a.output/'viewer.png'))
   if a.frames and frame>=a.frames:break
   time.sleep(1/60)
 finally:gym.destroy_viewer(viewer);gym.destroy_sim(sim)
if __name__=='__main__':main()
