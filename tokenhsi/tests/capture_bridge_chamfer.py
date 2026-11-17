import json
import hashlib
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import isaacgym
from isaacgym import gymapi
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch import nn

from env.tasks.base_task import BaseTask
from env.tasks.multi_agent import square_sensor_scene as scene
from learning.multi_agent.sensor_stage2_network import SensorStage2Builder
from learning.multi_agent.scene_normalizer import SceneRunningMeanStd
from learning.multi_agent.formal_eval import freeze_fallen
from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from utils.parse_task import parse_task


base_init = BaseTask.__init__


def camera_init(self, cfg, enable_camera_sensors=False):
    base_init(self, cfg, enable_camera_sensors=True)


def camera(task, env, target, offset, width, height):
    props = gymapi.CameraProperties()
    props.width, props.height, props.horizontal_fov = width, height, 65.
    handle = task.gym.create_camera_sensor(task.envs[env], props)
    if handle < 0:
        raise RuntimeError('Camera creation failed')
    target = np.asarray(target)
    task.gym.set_camera_location(handle, task.envs[env], gymapi.Vec3(*(target+offset)), gymapi.Vec3(*target))
    return handle


def pixels(task, env, handle, width, height):
    data = task.gym.get_camera_image(task.sim, task.envs[env], handle, gymapi.IMAGE_COLOR)
    return Image.fromarray(np.asarray(data).reshape(height,width,4)[...,:3])


@torch.no_grad()
def main():
    args = get_args()
    args.test = args.eval = args.headless = True
    args.eval_protocol = 'general'
    cfg, train, _ = load_cfg(args)
    cfg['env']['motion_file'], cfg['args'] = args.motion_file, args
    torch.set_num_threads(4)
    set_seed(args.seed)
    BaseTask.__init__ = camera_init
    task, env = parse_task(args,cfg,train['params']['config'],parse_sim_params(args,cfg,train))
    output = Path(args.output_path)
    output.mkdir(parents=True,exist_ok=True)
    builder = SensorStage2Builder()
    builder.load(train['params']['network'])
    net = builder.build('amp',actions_num=32,input_shape=(task.get_obs_size(),),
        amp_input_shape=(task.get_num_amp_obs(),),value_size=1,num_agents=4,num_objects=8,
        observation_mode='clean_scene',scene_entity_sizes=task.get_scene_entity_sizes(),
        scene_kinematic_size=7,scene_arena_scale=task.get_scene_arena_scale(),
        relation_reward_mode=task._relation_cfg['mode'],relation_graph_spec=task._relation_graph_spec,
        device=task.device).to(task.device)
    model = nn.Module()
    model.a2c_network = net
    checkpoint = torch.load(args.checkpoint,map_location=task.device,weights_only=False)
    model.load_state_dict(checkpoint['model'],strict=True)
    model.eval()
    original_weights = {key:value.clone() for key,value in model.state_dict().items()}
    rms = SceneRunningMeanStd([223,30,1],[4,8,4],normalized_sizes=[223,30,0],
        kinematic_size=7,extra_passthrough_size=task.get_relation_suffix_size()).to(task.device).eval()
    rms.load_state_dict(checkpoint['running_mean_std'],strict=True)
    obs = env.reset()
    entry_check = None
    if task._bridge_entry_insert:
        ids = torch.arange(task.num_envs,device=task.device)
        boxes = task._box_states[ids,task._bridge_slot]
        local = scene.rotate_xy(boxes[:,:2]-task._env_origins[:,:2]-task._structure_shift[:,1],
                                -task._structure_yaw[:,1])
        size = task._box_size[ids,task._bridge_slot]
        depths = task._bridge_half_a[:,1]-(local[:,1]-size[:,1]/2)
        extent_error = (task._bridge_far[:,1]+task._bridge_half_b[:,1]-task._bridge_half_a[:,1]).abs()
        center_error = (local[:,0]-task._bridge_half_a[:,0]-task._bridge_gap/2).abs()
        torch.testing.assert_close(depths,torch.full_like(depths,task._bridge_entry_insert),atol=.001,rtol=0)
        assert extent_error.max() < 1e-5 and center_error.max() < .001
        assert not task._bridge_entry_chamfer and task._bridge_side.all()
        records = task._terrain_actor_records
        active = (records[:,1] == 1) & (records[:,2] == task._bridge_count[records[:,0]]-1) & (records[:,3] < 2)
        records = records[active]
        physical = task._root_states[records[:,4],:3]
        expected = task._static_centers[records[:,0],1,records[:,3]]
        torch.testing.assert_close(physical,expected,atol=.0005,rtol=0)
        entry_check = dict(checked_envs=task.num_envs,min_insert_m=float(depths.min()),
            max_insert_m=float(depths.max()),max_extent_error_m=float(extent_error.max()),
            max_center_error_m=float(center_error.max()),
            max_physical_position_error_m=float((physical-expected).abs().max()),
            original_rectangular_platforms=True)
        (output/'entry_alignment_verification.json').write_text(json.dumps(entry_check,indent=2))
    per_count = int(os.environ.get('CAPTURE_PER_COUNT','3'))
    picked = []
    for count in (2,3):
        choices = ((task._bridge_count == count) & ~task._bridge_near).nonzero().flatten().tolist()
        if len(choices) < per_count:
            raise RuntimeError(f'Not enough original-start scenes for {count} pushers: {choices}')
        picked.extend(choices[:per_count])
    torch.save(dict(roots=task._humanoid_root_states.cpu(),boxes=task._box_states.cpu(),
        dof_pos=task._dof_pos.cpu(),dof_vel=task._dof_vel.cpu(),groups=task._groups.cpu(),
        regions=task._regions.cpu(),roles=task._roles.cpu(),templates=task._templates.cpu(),
        bridge_count=task._bridge_count.cpu(),selected=picked),output/'initial.pt')
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',21)
    cameras, encoders, files, rows = {}, {}, {}, {}
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    for number,e in enumerate(picked,1):
        box = task._box_states[e,task._bridge_slot[e],:3].cpu().numpy()
        landing = task._bridge_landing_world[e].cpu().numpy()
        yaw = float(task._structure_yaw[e,1])
        sign = 1. if bool(task._bridge_side[e]) else -1.
        radius = max(float(task._box_size[e,task._bridge_slot[e],0])+5.,abs(float(box[1]-landing[1]))+5.)
        target = .6*box+.4*landing
        offset_xy = scene.rotate_xy(torch.tensor([[.25*radius,sign*1.4*radius]]),torch.tensor([yaw]))[0].numpy()
        main_cam = camera(task,e,target,np.array([*offset_xy,1.4*radius]),1280,720)
        ahead = scene.rotate_xy(torch.tensor([[0.,-sign*(float(task._box_size[e,task._bridge_slot[e],1])/2+.4)]]),
                               torch.tensor([yaw]))[0].numpy()
        inset_target = box+np.array([*ahead,0.])
        inset_cam = camera(task,e,inset_target,np.array([0.,-.01,float(task._bridge_gap[e])+3.]),480,360)
        cameras[e] = (main_cam,inset_cam)
        name = f'bridge_{int(task._bridge_count[e])}push_{number:02d}'
        files[e] = output/(name+'.mp4')
        log = (output/(name+'_encoder.log')).open('w')
        encoders[e] = subprocess.Popen([ffmpeg,'-y','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720',
            '-r','30','-i','-','-an','-c:v','libx264','-preset','veryfast','-crf','22','-threads','1',
            '-pix_fmt','yuv420p','-movflags','+faststart',str(files[e])],stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,stderr=log)
        rows[e] = dict(file=files[e].name,env=e,pushers=int(task._bridge_count[e]),
            start='aligned_entry' if task._bridge_entry_insert else 'original',
            bridge_size=task._box_size[e,task._bridge_slot[e]].tolist(),gap=float(task._bridge_gap[e]),
            templates=task._templates[e].tolist(),roles=task._roles[e].tolist(),
            look_ahead=[float(ahead[0]),float(ahead[1]),0.])
    alive = torch.ones(task.num_envs,4,dtype=torch.bool,device=task.device)
    steps = int(os.environ.get('CAPTURE_STEPS','900'))
    labels = 'CHAMFER 20cm x 40cm' if task._bridge_entry_chamfer else 'ORIGINAL ENTRY'
    if task._bridge_entry_insert:
        labels = f'ALIGNED ENTRY +{task._bridge_entry_insert*100:.1f}cm | ORIGINAL SHAPE'
    for step in range(steps):
        mu = net.eval_actor(rms(obs))[0].clamp(-1,1)
        obs,_,_,_ = env.step((mu.reshape(task.num_envs,4,-1)*alive[...,None]).flatten(0,1))
        fallen = task._terminate_buf.reshape(task.num_envs,4).bool() & alive
        freeze_fallen(task,fallen)
        alive &= ~fallen
        task.gym.fetch_results(task.sim,True)
        for e in picked:
            point = task._box_states[e,task._bridge_slot[e],:3].cpu().numpy()+np.array(rows[e]['look_ahead'])
            eye = point+np.array([0.,-.01,rows[e]['gap']+3.])
            task.gym.set_camera_location(cameras[e][1],task.envs[e],gymapi.Vec3(*eye),gymapi.Vec3(*point))
        task.gym.step_graphics(task.sim)
        task.gym.render_all_camera_sensors(task.sim)
        for e in picked:
            frame = pixels(task,e,cameras[e][0],1280,720)
            inset = pixels(task,e,cameras[e][1],480,360)
            if step == 0 and (np.asarray(frame).std() < 2 or np.asarray(inset).std() < 2):
                raise RuntimeError('Camera did not capture the scene')
            frame.paste(inset,(800,100))
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0,0,1280,64),fill=(20,23,28))
            draw.text((16,8),f'{labels} | {rows[e]["pushers"]} PUSHERS | scene {e} | t={step*task.dt:.1f}s',font=font,fill='white')
            draw.text((16,35),'Current checkpoint | deterministic first episode | no resets',font=font,fill='white')
            draw.rectangle((800,71,1280,100),fill=(20,23,28))
            draw.text((810,74),'ENTRY - FOLLOW TOP VIEW',font=font,fill='white')
            encoders[e].stdin.write(frame.tobytes())
            if step == 0:
                frame.save(output/(files[e].stem+'_start.png'))
            if step == steps-1:
                frame.save(output/(files[e].stem+'_end.png'))
        if step % 100 == 0:
            print(f'capture {step}/{steps} | alive={alive.sum().item()}',flush=True)
        if not torch.isfinite(obs).all() or not torch.isfinite(mu).all():
            raise RuntimeError('Non-finite inference state')
    for e in picked:
        encoders[e].stdin.close()
        if encoders[e].wait() != 0:
            raise RuntimeError(f'Video encoding failed: {files[e]}')
        rows[e].update(frames=steps,fps=30,arrived=bool(task._bridge_arrived[e]),
            crossed=bool(task._bridge_crossed[e]),alive=alive[e].tolist())
    manifest = dict(checkpoint=args.checkpoint,epoch=checkpoint['epoch'],
        checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        seed=args.seed,chamfer=task._bridge_entry_chamfer,
        width=.2 if task._bridge_entry_chamfer else 0.,depth=.4 if task._bridge_entry_chamfer else 0.,
        entry_insert_m=task._bridge_entry_insert,entry_verification=entry_check,
        simulated_steps=steps,control_dt=task.dt,clips=list(rows.values()))
    manifest['weights_unchanged'] = all(torch.equal(value,model.state_dict()[key])
                                        for key,value in original_weights.items())
    if not manifest['weights_unchanged']:
        raise RuntimeError('Inference modified model weights')
    (output/'clips.json').write_text(json.dumps(manifest,indent=2))
    task.gym.destroy_sim(task.sim)
    print('CAPTURE COMPLETE',output,flush=True)


if __name__ == '__main__':
    main()
