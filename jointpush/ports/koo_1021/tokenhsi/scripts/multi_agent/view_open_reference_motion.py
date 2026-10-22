"""Replay AMP joint targets against a physical door; no policy or root assistance."""
from isaacgym import gymapi
import os
import time
import json
import math
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from env.tasks.multi_agent.humanoid_ma_open_push import HumanoidMAOpenPush

torch.set_num_threads(1)
args=get_args()
assert args.num_envs==1 and args.test and args.eval
cfg,train,_=load_cfg(args)
cfg['args']=args
cfg['env']['motion_file']=args.motion_file
cfg['env']['enableTrackInitState']=True
set_seed(42)
task=HumanoidMAOpenPush(cfg,parse_sim_params(args,cfg,train),args.physics_engine,args.device,args.device_id,args.headless)
task.render=lambda *a,**k:None
task._task_graph_preset='open'
ids=torch.arange(1,device=task.device)
actions=torch.zeros(2,task.num_dof,device=task.device)
lib=task._motion_lib['doorOpen']
rows=task._joint_rsi_bank.rows['doorOpen']
selected=[rows[len(rows)//2],rows[-1],rows[0]]
cycles=int(os.environ.get('REFERENCE_VIEW_CYCLES','3' if args.headless else '0'))
mode=os.environ.get('REFERENCE_START_MODE','rsi')
assert mode in ('rsi','closed')
out=Path(args.output_path)
out.mkdir(parents=True,exist_ok=True)
forward_axis=None

def hud(lines):
    """Camera-facing line glyphs, using Gym's native debug-line rendering."""
    canvas=Image.new('L',(920,100),0)
    try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',15)
    except OSError:font=ImageFont.load_default()
    draw=ImageDraw.Draw(canvas)
    for i,line in enumerate(lines):draw.text((0,i*19),line,font=font,fill=255)
    pixels=np.asarray(canvas)>100;segments=[]
    for y,row in enumerate(pixels):
        edges=np.diff(np.pad(row.astype(np.int8),(1,1)))
        for x0,x1 in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
            segments.append([[-1.4+x0*.0035,.92-y*.0035,-2.],[-1.4+x1*.0035,.92-y*.0035,-2.]])
    return np.asarray(segments,dtype=np.float32)


def draw_hud(vertices):
    global forward_axis
    transform=task.gym.get_viewer_camera_transform(task.viewer,None)
    q=np.array([transform.r.x,transform.r.y,transform.r.z],dtype=np.float32)
    w=transform.r.w
    axes=np.eye(3,dtype=np.float32)
    uv=np.cross(np.broadcast_to(q,axes.shape),axes)
    axes=axes+2*(w*uv+np.cross(np.broadcast_to(q,axes.shape),uv))
    pos=np.array([transform.p.x,transform.p.y,transform.p.z])
    if forward_axis is None:
        expected=np.array([.2,-2.5,1.])-pos
        dots=axes@(expected/np.linalg.norm(expected))
        index=int(np.argmax(np.abs(dots)));forward_axis=(index,np.sign(dots[index]))
    forward=axes[forward_axis[0]]*forward_axis[1]
    right=np.cross(forward,np.array([0.,0.,1.]));right/=np.linalg.norm(right)
    up=np.cross(right,forward)
    flat=vertices.reshape(-1,3)
    rotated=pos+2*forward+flat[:,0,None]*right+flat[:,1,None]*up
    colors=np.tile(np.array([[1.,.9,.3]],dtype=np.float32),(len(vertices),1))
    task.gym.clear_lines(task.viewer)
    task.gym.add_lines(task.viewer,task.envs[0],len(vertices),
        np.ascontiguousarray(rotated.reshape(-1,2,3),dtype=np.float32),colors)


def reset(row):
    task._reset_envs(ids)
    source=dict(row,angle=row['angle'] if mode=='rsi' else 0.)
    sample={k:torch.tensor([v],device=task.device) for k,v in source.items()}
    task._apply_joint_rsi(ids,torch.zeros_like(ids),'doorOpen',sample)
    task._humanoid_root_states[:,1,0]-=10.
    task._reset_env_tensors(ids)
    task._sync_door_reset_states(ids)
    task._refresh_sim_tensors()
    task.relation_runtime.reset(ids,task._sample_tasks(ids),task._box_states[:,2:])
    task._reset_task(ids)
    task.relation_runtime.initialize_manipulation(ids,task._box_states[:,2:],task._tar_pos,task.door_hinge_angle)
    task._compute_observations(ids)
    task._init_amp_obs(ids)

if task.viewer is not None:
    for key,name in ((gymapi.KEY_N,'next'),(gymapi.KEY_SPACE,'pause'),(gymapi.KEY_O,'closed'),(gymapi.KEY_L,'rsi')):
        task.gym.subscribe_viewer_keyboard_event(task.viewer,key,name)
    task.gym.viewer_camera_look_at(task.viewer,None,gymapi.Vec3(4.,-7.,3.2),gymapi.Vec3(.2,-2.5,1.))
cycle=0
need_reset=True
paused=False
ended=False
vertices=None
try:
    with (out/'trace.jsonl').open('w') as log:
        while True:
            tick=time.monotonic()
            if task.viewer is not None:
                if task.gym.query_viewer_has_closed(task.viewer):break
                for event in task.gym.query_viewer_action_events(task.viewer):
                    if event.value<=0:continue
                    if event.action=='QUIT':raise KeyboardInterrupt
                    if event.action=='pause':paused=not paused
                    elif event.action=='next':need_reset=True
                    elif event.action in ('closed','rsi'):mode=event.action;need_reset=True
            if need_reset:
                if cycles and cycle>=cycles:break
                row=selected[cycle%len(selected)]
                reset(row)
                mid=int(row['motion_id'])
                length=float(lib._motion_lengths[mid])
                step=0
                limit=math.ceil((length-row['time']+1.)/task.dt)
                ended=False
                need_reset=False
                cycle+=1
                vertices=None
                print('REFERENCE_START',cycle,mode,row,flush=True)
            if not paused and not ended:
                step+=1
                ref_time=min(length,row['time']+step*task.dt)
                state=lib.get_motion_state(torch.tensor([mid],device=task.device),torch.tensor([ref_time],device=task.device))
                task._every_env_init_dof_pos[:,0]=state[2]
                task.pre_physics_step(actions)
                task._physics_step()
                task.post_physics_step()
                assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task._dof_pos).all()
                angle=math.degrees(float(task.door_hinge_angle[0,0]))
                hand_force=float(task._contact_forces[0,0,task._key_body_ids[:2]].norm(dim=-1).max())
                fallen=bool(task._terminate_buf[0])
                log.write(json.dumps(dict(cycle=cycle,start=mode,reference_time=ref_time,angle_deg=angle,
                    hand_net_force_N=hand_force,root_z=float(task._humanoid_root_states[0,0,2]),fallen=fallen))+'\n')
                if fallen or step>=limit:
                    ended=True
                    end_at=time.monotonic()
                    log.flush()
                    print('REFERENCE_END',cycle,'angle_deg',angle,'fallen',fallen,flush=True)
                if step%5==1 or ended:
                    vertices=hud(['AMP joint PD / free root / no policy',
                        f'Start: {mode} | clip {ref_time:.2f}/{length:.2f}s | door {angle:.1f} deg',
                        f'Hand NET force: {hand_force:.1f} N (all contacts) | fallen: {fallen}',
                        'N: next | SPACE: pause | O: closed door | L: RSI angle',
                        'Trial ends at fall; automatically repeats after 2 seconds'])
            if task.viewer is not None:
                task.gym.step_graphics(task.sim)
                if vertices is not None:draw_hud(vertices)
                task.gym.draw_viewer(task.viewer,task.sim,True)
            if ended and not paused and (args.headless or time.monotonic()-end_at>=2.):need_reset=True
            if not args.headless:time.sleep(max(0.,task.dt-(time.monotonic()-tick)))
finally:
    if task.viewer is not None:task.gym.destroy_viewer(task.viewer)
    task.gym.destroy_sim(task.sim)
