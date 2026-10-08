"""Frozen reset preview, without a policy or checkpoint."""
import sys,time,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tokenhsi'))
from isaacgym import gymapi, gymtorch
import torch
from utils.config import get_args,load_cfg,parse_sim_params,set_seed
from utils.parse_task import parse_task
args=get_args()
cfg,train,_=load_cfg(args)
if 'task_rsi' in cfg['env']['interaction']:
    rsi=cfg['env']['interaction']['task_rsi']
    if 'RSI_PREVIEW_PROBABILITY' in os.environ:
        probability=float(os.environ['RSI_PREVIEW_PROBABILITY'])
        rsi['probability']=rsi['door_rsi_probability']=probability
    print('[preview] RSI probabilities PUSH:',rsi['probability'],'DOOR:',rsi.get('door_rsi_probability',rsi['probability']),flush=True)
set_seed(42,False)
if args.motion_file:cfg['env']['motion_file']=args.motion_file
sim_params=parse_sim_params(args,cfg,train)
task,env=parse_task(args,cfg,train,sim_params)
task.gym.subscribe_viewer_keyboard_event(task.viewer,gymapi.KEY_R,'RESAMPLE')
out=Path(args.output_path);out.mkdir(parents=True,exist_ok=True)
count=0

def reset_scene():
    global count
    ids=torch.arange(task.num_envs,device=task.device,dtype=torch.long)
    task.reset(ids)
    task.gym.simulate(task.sim);task.gym.fetch_results(task.sim,True)
    task._refresh_sim_tensors()
    # LOCAL preview only: park inactive objects outside the camera view.
    # Training/evaluation task code and allocations remain unchanged.
    inactive_boxes=task._box_actor_ids[task._tasks.bool()].long()
    door_ids=task.doors.actor_indices.view(task.num_envs,task.num_agents)
    inactive_doors=door_ids[~task._tasks.bool()].long()
    inactive=torch.cat([inactive_boxes,inactive_doors]).contiguous()
    task._root_states[inactive,0:2]=10000.
    task.gym.set_actor_root_state_tensor_indexed(task.sim,gymtorch.unwrap_tensor(task._root_states),gymtorch.unwrap_tensor(inactive.to(torch.int32)),len(inactive))
    # One physics step commits indexed root/DOF setters to the graphics scene.
    # After this step the simulation is frozen until the next reset.
    task.gym.simulate(task.sim);task.gym.fetch_results(task.sim,True)
    task._refresh_sim_tensors()
    task.gym.viewer_camera_look_at(task.viewer,task.envs[0],gymapi.Vec3(-4.5,-5.,2.7),gymapi.Vec3(-.7,0.,1.))
    task.gym.step_graphics(task.sim)
    task.gym.draw_viewer(task.viewer,task.sim,True)
    task.gym.write_viewer_image_to_file(task.viewer,str(out/('reset_%03d.png'%count)))
    print('[RSI frozen] sample=%d tasks=%s | R: resample, Esc: quit'%(count,task._tasks.cpu().tolist()),flush=True)
    if hasattr(task,'_last_rsi_selected'):
        print('[RSI hand] -1=non-RSI, 0=right, 1=left:',task._last_rsi_contact_hand.cpu().tolist(),flush=True)
        print('[RSI pose] selected=',task._last_rsi_selected.cpu().tolist(),'phase=',task._last_rsi_phase.cpu().tolist(),'door_angles_deg=',torch.rad2deg(task._door_state_view[...,0]).cpu().tolist(),flush=True)
    count+=1

try:
    reset_scene()
    while not task.gym.query_viewer_has_closed(task.viewer):
        for event in task.gym.query_viewer_action_events(task.viewer):
            if event.value>0 and event.action=='QUIT':raise KeyboardInterrupt
            if event.value>0 and event.action=='RESAMPLE':reset_scene()
        task.gym.step_graphics(task.sim)
        task.gym.draw_viewer(task.viewer,task.sim,True)
        time.sleep(1/30)
except KeyboardInterrupt:
    pass
finally:
    task.gym.destroy_viewer(task.viewer)
    task.gym.destroy_sim(task.sim)
