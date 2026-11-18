import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).resolve().parents[1])]
import isaacgym
from isaacgym import gymapi
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from env.tasks.base_task import BaseTask
from env.tasks.multi_agent import square_sensor_scene as scene
from env.tasks.multi_agent.scene_validation import scene_infeasible
from env.tasks.multi_agent.stage2_layout import invalid_before_layout
from env.tasks.multi_agent.warehouse_coop import opened
from utils.sensor_stage2_spec import WAREHOUSE
from utils.warehouse_coop import DIRECTIONS
from check_sensor_stage2 import check_sensor


def inspect(task, obs):
    ids = (task._scene_ids == WAREHOUSE).nonzero().flatten()
    push = task._templates[ids] == 5
    counts = task._bridge_count[ids]
    if not torch.equal(push.sum(-1), counts) or not torch.equal(task._members[ids][push], counts[:,None].expand_as(push)[push]):
        raise RuntimeError('Wrong cooperative members')
    if not (task._agent_box_assignment[ids][push] == task._bridge_slot[ids,None].expand_as(push)[push]).all():
        raise RuntimeError('PUSH does not share the physical box')
    boxes = task._box_states[ids,task._bridge_slot[ids]]
    sizes = task._box_size[ids,task._bridge_slot[ids]]
    expected = sizes.new_tensor(task._warehouse_coop['box_size']).expand_as(sizes).clone()
    expected[:,0] *= counts
    torch.testing.assert_close(sizes, expected)
    from utils.coupled_push_spec import inverse_rotate
    relative = task._humanoid_root_states[ids,:,:3]-boxes[:,None,:3]
    local = inverse_rotate(boxes[:,None,3:7].expand(-1,4,-1),relative)
    region_x = (task._regions[ids]+.5)*task._warehouse_coop['box_size'][0]-sizes[:,:1]/2
    torch.testing.assert_close(local[...,0][push],region_x[push],atol=.003,rtol=0)
    torch.testing.assert_close(local[...,1][push],torch.full_like(local[...,1][push],-1.7),atol=.003,rtol=0)
    yaw = task._structure_yaw[ids,WAREHOUSE]
    xy = scene.rotate_xy(boxes[:,:2]-task._env_origins[ids,:2]-task._structure_shift[ids,WAREHOUSE],-yaw)
    is_open = opened(task,ids,xy,boxes,sizes)
    mode = task._warehouse_start_mode[ids]
    if is_open[mode == 0].any() or not is_open[mode == 2].all():
        raise RuntimeError('Closed/open RSI geometry differs')
    roots = scene.rotate_xy(task._humanoid_root_states[ids,:,:2]-task._env_origins[ids,None,:2]
        -task._structure_shift[ids,WAREHOUSE,None],-yaw[:,None])
    roots[...,0] += task._surface_half[ids,WAREHOUSE,0,0,None]
    if not (roots[...,0][~push] < -.75).all():
        raise RuntimeError('Worker must start outside')
    inside = (task._warehouse_direction[ids] == 3) & (mode == 0)
    if not (roots[...,0][inside][push[inside]] > 0).all():
        raise RuntimeError('Outward PUSH must start inside')
    if scene_infeasible(task,ids).any() or invalid_before_layout(task,ids).any():
        raise RuntimeError('Physical/BEFORE layout invalid')
    floor = task._env_origins[ids,2]+task._warehouse['floor_height']
    torch.testing.assert_close(boxes[:,2]-sizes[:,2]/2, floor, atol=.002, rtol=0)
    variant = (task._warehouse_direction[ids] >= 2).long()*3+counts-1
    records = task._terrain_actor_records
    actual = task._root_states.reshape(-1,13)
    for row,e in enumerate(ids.tolist()):
        selected = records[(records[:,0] == e) & (records[:,1] == WAREHOUSE) & (records[:,2] == variant[row])]
        torch.testing.assert_close(actual[selected[:,-1],:3],task._static_centers[e,WAREHOUSE,selected[:,3]],atol=.002,rtol=0)
    stats = {}
    check_sensor(task,obs,stats)
    key = ((counts-1)*4+task._warehouse_direction[ids])*3+mode
    stats.update(warehouse=len(ids),scenes=torch.bincount(task._scene_ids,minlength=4).tolist(),
        combinations=torch.bincount(key,minlength=36).reshape(3,4,3).tolist(),
        shared_box=True,terrain_matches_sensor=True,closed_open=True,physical_and_before_valid=True)
    return stats


def capture(task, output):
    cameras = []
    for count in (1,2,3):
        for direction,name in enumerate(DIRECTIONS):
            selected = ((task._scene_ids == WAREHOUSE) & (task._bridge_count == count)
                & (task._warehouse_direction == direction) & (task._warehouse_start_mode == 0)).nonzero().flatten()
            if not len(selected):
                raise RuntimeError('Missing closed warehouse combination')
            e = int(selected[0])
            box_handle = task._box_handles[e*task.num_physical_objects+int(task._bridge_slot[e])]
            task.gym.set_rigid_body_color(task.envs[e],box_handle,0,gymapi.MESH_VISUAL,gymapi.Vec3(.9,.08,.06))
            center = task._env_origins[e]+torch.cat((task._structure_shift[e,WAREHOUSE],task._env_origins.new_zeros(1)))
            box = task._box_states[e,task._bridge_slot[e],:3]
            focus = (center+box)/2
            focus[2] = .6
            offset = scene.rotate_xy(focus.new_tensor([[-8.,-9.]]),task._structure_yaw[e,WAREHOUSE,None])[0]
            props = gymapi.CameraProperties()
            props.width,props.height,props.horizontal_fov = 1280,900,62.
            handle = task.gym.create_camera_sensor(task.envs[e],props)
            task.gym.set_camera_location(handle,task.envs[e],gymapi.Vec3(*(focus+torch.cat((offset,offset.new_tensor([15.])))).tolist()),gymapi.Vec3(*focus.tolist()))
            cameras.append((e,count,name,handle))
    task.gym.simulate(task.sim)
    task.gym.fetch_results(task.sim,True)
    task._refresh_sim_tensors()
    task.gym.step_graphics(task.sim)
    task.gym.render_all_camera_sensors(task.sim)
    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',27)
    sheet = Image.new('RGB',(1920,1920),(23,28,34))
    names = {'left':'좌우 · 왼쪽','right':'좌우 · 오른쪽','in':'앞뒤 · 안쪽','out':'앞뒤 · 바깥쪽'}
    for e,count,name,handle in cameras:
        pixels = task.gym.get_camera_image(task.sim,task.envs[e],handle,gymapi.IMAGE_COLOR)
        raw = Image.fromarray(np.asarray(pixels).reshape(900,1280,4)[...,:3])
        raw.save(output/'{}_{}p_raw.png'.format(name,count))
        panel = Image.new('RGB',(1280,960),(23,28,34))
        panel.paste(raw,(0,60))
        ImageDraw.Draw(panel).text((18,12),'실제 학습 reset | {} | PUSH {}명 + 작업자 {}명'.format(names[name],count,4-count),font=font,fill='white')
        panel.save(output/'{}_{}p.png'.format(name,count))
        sheet.paste(panel.resize((640,480)),((count-1)*640,DIRECTIONS.index(name)*480))
        task.gym.destroy_camera_sensor(task.sim,task.envs[e],handle)
    sheet.save(output/'training_resets.png')


def main():
    from utils.config import get_args,load_cfg,parse_sim_params,set_seed
    from utils.parse_task import parse_task
    args = get_args()
    torch.set_num_threads(4)
    cfg,train,_ = load_cfg(args)
    cfg['env']['motion_file'],cfg['args'] = args.motion_file,args
    set_seed(42)
    original = BaseTask.__init__
    BaseTask.__init__ = lambda self,cfg,enable_camera_sensors=False: original(self,cfg,enable_camera_sensors=True)
    task,env = parse_task(args,cfg,train['params']['config'],parse_sim_params(args,cfg,train))
    output = Path(args.output_path)
    output.mkdir(parents=True,exist_ok=True)
    rounds = []
    for index in range(4):
        obs = env.reset()
        rounds.append(inspect(task,obs))
        print('COOP RESET',index+1,rounds[-1]['warehouse'],flush=True)
        if index == 0:
            capture(task,output)
        for _ in range(8):
            obs,reward,_,_ = env.step(torch.zeros(task.num_envs*4,task.get_action_size(),device=task.device))
            if not torch.isfinite(obs).all() or not torch.isfinite(reward).all():
                raise RuntimeError('Nonfinite physics')
    previous = task._root_states.clone()
    env.reset(torch.tensor([0],device=task.device))
    if not torch.equal(previous[task.get_num_actors_per_env():],task._root_states[task.get_num_actors_per_env():]):
        raise RuntimeError('Partial reset changed other environments')
    (output/'placement.json').write_text(json.dumps(dict(envs=task.num_envs,rounds=rounds,partial_reset=True,physics_steps=32),indent=2)+'\n')
    task.gym.destroy_sim(task.sim)
    print('COOP CHECK COMPLETE',flush=True)


if __name__ == '__main__':
    main()
