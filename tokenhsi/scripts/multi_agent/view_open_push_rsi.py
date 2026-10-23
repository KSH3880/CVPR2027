"""Interactive reset/short-PD inspection of RSI, without a trained policy."""
from isaacgym import gymapi
import os
import time
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from env.tasks.multi_agent.humanoid_ma_open_push import HumanoidMAOpenPush
from env.tasks.multi_agent.open_push_reward import push_hand_targets

torch.set_num_threads(1)
args=get_args();assert args.num_envs==1 and args.test and args.eval
cfg,train,_=load_cfg(args);cfg['args']=args;cfg['env']['motion_file']=args.motion_file
cfg['env']['enableTrackInitState']=True
set_seed(args.seed if args.seed>=0 else 42)
task=HumanoidMAOpenPush(cfg,parse_sim_params(args,cfg,train),args.physics_engine,args.device,args.device_id,args.headless)
# Restore the actual training skill list/weights for reference history.
task._is_eval=False;task._skill=list(cfg['env']['skill'])
task._skill_init_prob=torch.tensor(cfg['env']['skillInitProb'],device=task.device)
# Viewer events are handled here; physics ticks must not consume them.
task.render=lambda *a,**k:None
ids=torch.arange(1,device=task.device)
out=Path(args.output_path);out.mkdir(parents=True,exist_ok=True)
cycles=int(os.environ.get('RSI_VIEW_CYCLES','3' if args.headless else '0'))
mode=os.environ.get('RSI_VIEW_MODE','late');preset=os.environ.get('TASK_GRAPH','open_push')
focus=int(os.environ.get('RSI_VIEW_FOCUS','0'));assert focus in (0,1)
cycle=0;pd_steps=0;running=False;need_reset=True;status={};text=[];forward_axis=None;camera_target=None


def camera():
    if task.viewer is None:return
    # Alternate close views of each person's task; target includes the object.
    global camera_target
    root=task._humanoid_root_states[0,focus,:3]
    target=(task.door_handle_position[0,focus] if int(task.relation_runtime.tasks[0,focus])==0 else task._box_states[0,focus+2,:3])
    camera_target=((root+target)/2).cpu().numpy();camera_target[2]=1.
    task.gym.viewer_camera_look_at(task.viewer,None,
        gymapi.Vec3(float(camera_target[0]+4),float(camera_target[1]-4.5),3.2),
        gymapi.Vec3(*[float(v) for v in camera_target]))


def step():
    task.pre_physics_step(torch.zeros(2,32,device=task.device))
    task._physics_step();task.post_physics_step()
    assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task.rew_buf).all()


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
        expected=camera_target-pos
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
    targets=push_hand_targets(task._box_states[:,2:],cfg['env']['pushBox']['size'],
        cfg['env']['relationReward']['push'])[0].cpu().numpy()
    segments=[];point_colors=[]
    for agent in range(2):
        if int(task.relation_runtime.tasks[0,agent])!=1:continue
        for hand in range(2):
            for axis in np.eye(3,dtype=np.float32)*.06:
                segments.append([targets[agent,hand]-axis,targets[agent,hand]+axis])
                point_colors.append([1.,.2,.2] if hand==0 else [.2,1.,1.])
    if segments:
        task.gym.add_lines(task.viewer,task.envs[0],len(segments),
            np.asarray(segments,dtype=np.float32),np.asarray(point_colors,dtype=np.float32))


if task.viewer is not None:
    for key,name in ((gymapi.KEY_N,'next'),(gymapi.KEY_SPACE,'pd'),(gymapi.KEY_C,'camera'),
                     (gymapi.KEY_L,'late'),(gymapi.KEY_M,'mixed'),(gymapi.KEY_O,'open'),
                     (gymapi.KEY_P,'push'),(gymapi.KEY_B,'both')):
        task.gym.subscribe_viewer_keyboard_event(task.viewer,key,name)
    camera()
vertices=None
try:
    while True:
        if task.viewer is not None:
            if task.gym.query_viewer_has_closed(task.viewer):break
            for event in task.gym.query_viewer_action_events(task.viewer):
                if event.value<=0:continue
                if event.action=='QUIT':raise KeyboardInterrupt
                if event.action=='next':need_reset=True
                elif event.action=='pd':running=not running
                elif event.action=='camera':focus=1-focus;camera()
                elif event.action in ('late','mixed'):mode=event.action;need_reset=True
                elif event.action in ('open','push','both'):
                    preset={'open':'open','push':'push','both':'open_push'}[event.action];need_reset=True
        if need_reset:
            if cycles and cycle>=cycles:break
            task._joint_rsi_spec['interaction_probability']=1. if mode=='late' else .5
            task._joint_rsi_spec['late_probability']=1. if mode=='late' else .7
            task._task_graph_preset=preset;task._reset_envs(ids)
            cycle+=1;pd_steps=0;running=args.headless;need_reset=False
            agents=[]
            for a in range(2):
                record=dict(agent=a,task='OPEN' if task.relation_runtime.tasks[0,a]==0 else 'PUSH',
                    interaction=bool(task._rsi_interaction[0,a]),late_requested=bool(task._rsi_late[0,a]),
                    phase=float(task._rsi_phase[0,a]),root=task._humanoid_root_states[0,a,:3].tolist())
                for skill,(env,agent) in task._reset_ref_slots.items():
                    selected=((env==0)&(agent==a)).nonzero().flatten()
                    if len(selected):
                        i=int(selected[0]);record.update(skill=skill,motion_id=int(task._reset_ref_motion_ids[skill][i]),time=float(task._reset_ref_motion_times[skill][i]))
                agents.append(record)
            status=dict(cycle=cycle,mode=mode,agents=agents,door_angles=task.door_hinge_angle.tolist(),
                        boxes=task._box_states[:,2:,:3].tolist(),goals=task._tar_pos.tolist(),policy=False,
                        push_hand_targets=push_hand_targets(task._box_states[:,2:],cfg['env']['pushBox']['size'],
                            cfg['env']['relationReward']['push']).tolist())
            (out/'latest_sample.json').write_text(json.dumps(status,indent=2)+'\n')
            with (out/'samples.jsonl').open('a') as f:f.write(json.dumps(status)+'\n')
            print('RSI_VIEW',json.dumps(status),flush=True)
            text=['RSI RESET / PD INSPECTION (no policy)  mode='+mode]
            for r in agents:
                text.append('H%d %s %s clip%d t=%.2fs phase=%.1f%% late=%s'%(r['agent'],r['task'],r['skill'],r['motion_id'],r['time'],100*r['phase'],r['late_requested']))
            text+=['N next | SPACE 12-step PD | C other person', 'L forced late | M train mixture | O open | P push | B both']
            vertices=hud(text)
            # One real physics frame makes indexed reset states visible.
            step();pd_steps=1;camera()
        if running and pd_steps<12:
            step();pd_steps+=1
        if pd_steps>=12:
            running=False
            if args.headless:need_reset=True
        if task.viewer is not None:
            task.gym.step_graphics(task.sim);draw_hud(vertices)
            task.gym.draw_viewer(task.viewer,task.sim,True)
            time.sleep(1/30)
except KeyboardInterrupt:
    pass
finally:
    if task.viewer is not None:task.gym.destroy_viewer(task.viewer)
    task.gym.destroy_sim(task.sim)
