"""Frozen actual lift reset scene, without a policy/checkpoint."""
import os
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'tokenhsi')]
from isaacgym import gymapi, gymutil
import torch
from utils.config import get_args,load_cfg,parse_sim_params,set_seed
from utils.parse_task import parse_task

args=get_args();cfg,train,_=load_cfg(args)
set_seed(42,False)
if args.motion_file:cfg['env']['motion_file']=args.motion_file
params=parse_sim_params(args,cfg,train)
task,_=parse_task(args,cfg,train,params)
output=Path(args.output_path);output.mkdir(parents=True,exist_ok=True)
count=0;selected=-1
for env_id,env in enumerate(task.envs):
    for a in range(2):
        colour=task._agent_color(a)
        for body in range(task.num_bodies):
            task.gym.set_rigid_body_color(env,task.humanoid_handles_all[env_id][a],body,gymapi.MESH_VISUAL_AND_COLLISION,colour)
        task.gym.set_rigid_body_color(env,task._box_handles[env_id*2+a],0,gymapi.MESH_VISUAL_AND_COLLISION,colour)

def draw_goals():
    task.gym.clear_lines(task.viewer)
    for a in range(2):
        colour=task._agent_color(a)
        sphere=gymutil.WireframeSphereGeometry(.08,12,12,None,color=(colour.x,colour.y,colour.z))
        goal=task._tar_pos[0,a].cpu().tolist()
        pose=gymapi.Transform();pose.p=gymapi.Vec3(*goal)
        gymutil.draw_lines(sphere,task.gym,task.viewer,task.envs[0],pose)

def camera():
    if selected<0:
        eye=gymapi.Vec3(-4.5,-6.,3.2);target=gymapi.Vec3(-.6,0.,1.)
    else:
        y=(selected-.5)*task._lift_cfg['lane_spacing']
        eye=gymapi.Vec3(-3.3,y-2.8,2.4);target=gymapi.Vec3(-.6,y,.9)
    task.gym.viewer_camera_look_at(task.viewer,task.envs[0],eye,target)

def screenshot():
    task.gym.write_viewer_image_to_file(task.viewer,str(output/f'reset_{count:03d}.png'))

def reset_scene():
    global count
    ids=torch.arange(task.num_envs,device=task.device,dtype=torch.long)
    task.reset(ids)
    # Commit the indexed RSI setters, then leave physics frozen.
    task.gym.simulate(task.sim);task.gym.fetch_results(task.sim,True)
    task._refresh_sim_tensors()
    camera()
    draw_goals()
    task.gym.step_graphics(task.sim);task.gym.draw_viewer(task.viewer,task.sim,True)
    screenshot()
    print('[Lift frozen] sample={} human={} box={} goal={} | R reset, N camera, S screenshot'.format(
        count,task._humanoid_root_states[0,:,:3].cpu().tolist(),
        task._box_states[0,:,:3].cpu().tolist(),task._tar_pos[0].cpu().tolist()),flush=True)
    count+=1

try:
    if task.viewer is None:raise RuntimeError('Frozen preview requires a viewer')
    for key,action in [(gymapi.KEY_R,'RESAMPLE'),(gymapi.KEY_N,'CAMERA'),(gymapi.KEY_S,'SAVE')]:
        task.gym.subscribe_viewer_keyboard_event(task.viewer,key,action)
    reset_scene()
    frames=0;limit=int(os.environ.get('PREVIEW_FRAMES','0'))
    while not task.gym.query_viewer_has_closed(task.viewer):
        for event in task.gym.query_viewer_action_events(task.viewer):
            if event.value<=0:continue
            if event.action=='QUIT':raise KeyboardInterrupt
            if event.action=='RESAMPLE':reset_scene()
            elif event.action=='CAMERA':selected=(selected+2)%3-1;camera()
            elif event.action=='SAVE':screenshot()
        draw_goals()
        task.gym.step_graphics(task.sim);task.gym.draw_viewer(task.viewer,task.sim,True)
        frames+=1
        if limit and frames>=limit:break
        time.sleep(1/30)
except KeyboardInterrupt:
    pass
finally:
    if task.viewer is not None:task.gym.destroy_viewer(task.viewer)
    task.gym.destroy_sim(task.sim)
