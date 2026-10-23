"""Local native Table SIT RSI viewer; no policy or optimizer."""
from pathlib import Path
import argparse,json,time
import isaacgym
from isaacgym import gymapi,gymutil
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,default=Path('output_etc/check/native_rsi_physics_20261023_v8'));p.add_argument('--output',type=Path,required=True);p.add_argument('--frames',type=int,default=0);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((a.audit/'manifest.json').read_text());entries=[]
    for e in manifest['assets']:
        if e['category']!='Table':continue
        d=np.load(a.audit/'audited_bank'/e['id']/'sit_accepted.npz')
        if not len(d['root_state']):print('TABLE_SIT_NO_ACCEPTED_POSE',e['id'],flush=True);continue
        goal=np.array([0.,0.,e['bbox'][2]+.1]);cost=np.linalg.norm(d['root_state'][:,:3]-goal,axis=-1);k=int(cost.argmin());entries.append((e,dict((key,d[key][k]) for key in d.files),goal))
    gym=gymapi.acquire_gym();params=gymapi.SimParams();params.up_axis=gymapi.UP_AXIS_Z;params.gravity=gymapi.Vec3(0,0,-9.81);params.dt=1/60;params.substeps=2;params.physx.solver_type=1;params.physx.num_position_iterations=4;params.physx.num_velocity_iterations=0;params.physx.contact_offset=.02;params.physx.rest_offset=0
    sim=gym.create_sim(0,0,gymapi.SIM_PHYSX,params);assert sim is not None
    plane=gymapi.PlaneParams();plane.normal=gymapi.Vec3(0,0,1);plane.static_friction=1;plane.dynamic_friction=1;gym.add_ground(sim,plane)
    options=gymapi.AssetOptions();options.default_dof_drive_mode=gymapi.DOF_MODE_POS
    human=gym.load_asset(sim,'tokenhsi/data/assets','mjcf/phys_humanoid_v3.xml',options)
    envs=[];actors=[];records=[]
    for i,(e,d,goal) in enumerate(entries):
        opts=gymapi.AssetOptions();opts.fix_base_link=True;opts.vhacd_enabled=True;opts.vhacd_params.resolution=100000;opts.vhacd_params.max_convex_hulls=128;opts.vhacd_params.max_num_vertices_per_ch=64;opts.use_mesh_materials=True
        obj=gym.load_asset(sim,e['path'],'asset.urdf',opts);env=gym.create_env(sim,gymapi.Vec3(-1.5,-1.5,0),gymapi.Vec3(1.5,1.5,3),3)
        pose=gymapi.Transform();pose.p.z=e['bbox'][2]/2;gym.create_actor(env,obj,pose,'table',i,0)
        actor=gym.create_actor(env,human,gymapi.Transform(),'human',i,0)
        props=gym.get_actor_dof_properties(env,actor);props['driveMode'].fill(gymapi.DOF_MODE_POS);gym.set_actor_dof_properties(env,actor,props)
        envs.append(env);actors.append(actor);records.append(dict(asset=e['id'],table_top=float(e['bbox'][2]),goal=goal.tolist(),initial_root=d['root_state'][:3].tolist(),initial_goal_error=float(np.linalg.norm(d['root_state'][:3]-goal))))
    gym.prepare_sim(sim)
    def reset():
        for env,actor,(e,d,goal) in zip(envs,actors,entries):
            state=gym.get_actor_rigid_body_states(env,actor,gymapi.STATE_ALL)
            for field,values in [('x',d['body_state'][:,0]),('y',d['body_state'][:,1]),('z',d['body_state'][:,2])]:state['pose']['p'][field]=values
            for j,field in enumerate(['x','y','z','w']):state['pose']['r'][field]=d['body_state'][:,3+j]
            for j,field in enumerate(['x','y','z']):state['vel']['linear'][field]=d['body_state'][:,7+j];state['vel']['angular'][field]=d['body_state'][:,10+j]
            gym.set_actor_rigid_body_states(env,actor,state,gymapi.STATE_ALL)
            dof=gym.get_actor_dof_states(env,actor,gymapi.STATE_ALL);dof['pos']=d['dof'];dof['vel']=d['dof_vel'];gym.set_actor_dof_states(env,actor,dof,gymapi.STATE_ALL);gym.set_actor_dof_position_targets(env,actor,np.asarray(d['dof'],dtype=np.float32))
        gym.simulate(sim);gym.fetch_results(sim,True)
    viewer=gym.create_viewer(sim,gymapi.CameraProperties());assert viewer is not None
    for key,name in [(gymapi.KEY_N,'next'),(gymapi.KEY_P,'previous'),(gymapi.KEY_R,'reset'),(gymapi.KEY_SPACE,'physics')]:gym.subscribe_viewer_keyboard_event(viewer,key,name)
    index=0;paused=True
    def camera():
        gym.viewer_camera_look_at(viewer,envs[index],gymapi.Vec3(2.,-2.5,1.6),gymapi.Vec3(0,0,.7));print('TABLE_SIT_VIEW',index+1,len(entries),json.dumps(records[index]),flush=True)
    reset();camera();(a.output/'scene.json').write_text(json.dumps(dict(poses=records,mode='paused audited RSI pose, not learned-policy output',goal='red dot = bbox top + .10m; yellow wire sphere = 20cm root distance; cyan dot = pelvis',controls='N/P next/previous table; SPACE fixed-PD physics; R reset'),indent=2))
    dot=gymutil.WireframeSphereGeometry(.025,8,8,None,color=(1,0,0));threshold=gymutil.WireframeSphereGeometry(.2,16,16,None,color=(1,1,0));pelvis=gymutil.WireframeSphereGeometry(.025,8,8,None,color=(0,1,1));frame=0
    print('VIEWER_READY N/P=table SPACE=physics R=reset; RED=goal YELLOW=20cm CYAN=root',flush=True)
    try:
        while not gym.query_viewer_has_closed(viewer):
            for event in gym.query_viewer_action_events(viewer):
                if event.value<=0:continue
                if event.action in ['next','previous']:index=(index+(1 if event.action=='next' else -1))%len(entries);camera()
                elif event.action=='physics':paused=not paused;print('PHYSICS',not paused,flush=True)
                elif event.action=='reset':reset();paused=True
            if not paused:gym.simulate(sim);gym.fetch_results(sim,True)
            gym.clear_lines(viewer)
            for env,actor,(_,d,goal) in zip(envs,actors,entries):
                pose=gymapi.Transform();pose.p=gymapi.Vec3(*goal);gymutil.draw_lines(dot,gym,viewer,env,pose);gymutil.draw_lines(threshold,gym,viewer,env,pose)
                root=gym.get_actor_rigid_body_states(env,actor,gymapi.STATE_POS)['pose']['p'][0];pose.p=gymapi.Vec3(float(root['x']),float(root['y']),float(root['z']));gymutil.draw_lines(pelvis,gym,viewer,env,pose)
            gym.step_graphics(sim);gym.draw_viewer(viewer,sim,True);frame+=1
            if frame==30:gym.write_viewer_image_to_file(viewer,str(a.output/'viewer.png'))
            if a.frames and frame>=a.frames:break
            time.sleep(1/60)
    finally:gym.destroy_viewer(viewer);gym.destroy_sim(sim)
if __name__=='__main__':main()
