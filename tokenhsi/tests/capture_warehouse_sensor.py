import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import isaacgym
from isaacgym import gymapi, gymtorch
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch

from env.tasks.multi_agent.humanoid_ma_sensor_stage2 import HumanoidMASensorStage2
from env.tasks.multi_agent import square_sensor_scene as scene
from utils.config import get_args,load_cfg,parse_sim_params,set_seed
from utils.parse_task import parse_task
from utils.sensor_stage2_spec import WAREHOUSE


TASK_NAMES = {1:'SIT',2:'CLIMB',3:'AT',4:'ON_TOP',5:'PUSH'}
PANEL_LEFT = 323
PLATE_SIZE, PLATE_THICKNESS = 120., .01
build_env = HumanoidMASensorStage2._build_env


def build_env_with_plates(self, env_id, env_ptr, humanoid_asset):
    build_env(self, env_id, env_ptr, humanoid_asset)
    if env_id == 0:
        options = gymapi.AssetOptions()
        options.fix_base_link = True
        self._plate_asset = self.gym.create_box(self.sim, PLATE_SIZE, PLATE_SIZE, PLATE_THICKNESS, options)
    origin = self.gym.get_env_origin(env_ptr)
    for index in range(len(self._scene_names)):
        offset = self._scene_offsets[index].cpu().tolist()
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(origin.x+offset[0], origin.y+offset[1], .002-PLATE_THICKNESS/2)
        handle = self.gym.create_actor(env_ptr, self._plate_asset, pose, 'capture_plate_{}'.format(index), env_id, 0)
        self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(.62, .62, .60))


def render_offset(task, env):
    origin = task.gym.get_env_origin(task.envs[env])
    return np.array([origin.x, origin.y, origin.z])


def agent_rgb(task, person):
    color = task._agent_color(person)
    return color.x, color.y, color.z


def draw_bindings(task, env):
    gym, viewer, ptr = task.gym, task.viewer, task.envs[env]
    objects = task._assigned_box_values(task._box_states)[env, :, :3]
    roots = task._humanoid_root_states[env, :, :3]
    goals = task._tar_pos[env]
    for person in range(task.num_agents):
        segments = [torch.stack((roots[person], objects[person]))]
        template = int(task._templates[env, person])
        if template in (3, 5):
            segments.append(torch.stack((objects[person], goals[person])))
            angle = np.linspace(0, 2*np.pi, 25)
            circle = np.stack((.35*np.cos(angle), .35*np.sin(angle), np.zeros_like(angle)), -1)
            ring = goals[person].cpu().numpy()+circle
            ring = torch.as_tensor(ring, device=task.device, dtype=torch.float)
            segments.extend(torch.stack((ring[i], ring[i+1])) for i in range(24))
        lines = (torch.stack(segments)+task._env_origins.new_tensor([0., 0., .03])).cpu().numpy()
        rgb = np.tile(np.array(agent_rgb(task, person), dtype=np.float32), (len(lines), 1))
        gym.add_lines(viewer, ptr, len(lines), lines, rgb)


def shoot(task, env, path, target, position):
    offset = render_offset(task, env)
    task.gym.viewer_camera_look_at(task.viewer, None, gymapi.Vec3(*(np.array(position)+offset)),
                                   gymapi.Vec3(*(np.array(target)+offset)))
    for _ in range(3):
        task.gym.step_graphics(task.sim)
        task.gym.draw_viewer(task.viewer, task.sim, True)
    task.gym.write_viewer_image_to_file(task.viewer, str(path))
    image = Image.open(path).convert('RGB')
    return image.crop((PANEL_LEFT, 0, image.width-PANEL_LEFT, image.height))


def main():
    args = get_args()
    torch.set_num_threads(4)
    cfg,train,_ = load_cfg(args)
    cfg['env']['motion_file'],cfg['args'] = args.motion_file,args
    set_seed(args.seed)
    HumanoidMASensorStage2._build_env = build_env_with_plates
    task,env = parse_task(args,cfg,train['params']['config'],parse_sim_params(args,cfg,train))
    output = Path(args.output_path)
    output.mkdir(parents=True,exist_ok=True)
    env.reset()
    env.reset()
    ids = (task._scene_ids == WAREHOUSE).nonzero().flatten().tolist()
    solo = [e for e in ids if not (task._before_predecessor[e] >= 0).any()]
    picked = solo[:2]
    for family in (1,2,4):
        choices = [e for e in ids if ((task._before_predecessor[e] >= 0) & (task._templates[e] == family)).any()]
        if choices:
            picked.append(choices[0])
    picked += [e for e in ids if e not in picked][:5-len(picked)]
    if len(picked) != 5:
        raise RuntimeError('Need at least five warehouse reset scenes')
    task._marker_pos[:,:,2] = scene.HIDDEN
    task.gym.set_actor_root_state_tensor(task.sim,gymtorch.unwrap_tensor(task._root_states))
    task.gym.simulate(task.sim)
    task.gym.fetch_results(task.sim,True)
    task._refresh_sim_tensors()

    closed,opened,cameras = {},{},{}
    for order,e in enumerate(picked):
        task.gym.clear_lines(task.viewer)
        draw_bindings(task,e)
        half = task._surface_half[e,WAREHOUSE,0]
        yaw = task._structure_yaw[e,WAREHOUSE]
        center = task._env_origins[e,:2]+task._structure_shift[e,WAREHOUSE]
        focus = scene.rotate_xy(half.new_tensor([[-1.,0.]]),yaw[None])[0]
        target = np.array([float(center[0]+focus[0]),float(center[1]+focus[1]),.6])
        radius = max(float(2*half[0])+6.,float(2*half[1]))
        offset = scene.rotate_xy(half.new_tensor([[-.02,0.]]),yaw[None])[0]
        camera = target+np.array([float(offset[0]),float(offset[1]),radius*1.35])
        name = 'sample_{:02d}'.format(order+1)
        closed[e] = shoot(task,e,output/(name+'_closed.png'),target,camera)
        closed[e].save(output/(name+'_closed.png'))
        cameras[e] = (target,camera)

    selected = torch.tensor(ids,device=task.device)
    task._box_states[selected,task._bridge_slot[selected],:3] = task._warehouse_landing[selected]
    task._box_states[selected,task._bridge_slot[selected],7:] = 0
    task.gym.set_actor_root_state_tensor(task.sim,gymtorch.unwrap_tensor(task._root_states))
    task.gym.simulate(task.sim)
    task.gym.fetch_results(task.sim,True)
    task._refresh_sim_tensors()
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
    images,report = [],[]
    for order,e in enumerate(picked):
        task.gym.clear_lines(task.viewer)
        draw_bindings(task,e)
        name = 'sample_{:02d}'.format(order+1)
        opened[e] = shoot(task,e,output/(name+'_opened.png'),*cameras[e])
        opened[e].save(output/(name+'_opened.png'))
        width,height = closed[e].size
        canvas = Image.new('RGB',(2*width,height+190),(20,22,26))
        canvas.paste(closed[e],(0,190))
        canvas.paste(opened[e],(width,190))
        draw = ImageDraw.Draw(canvas)
        dimensions = 2*task._surface_half[e,WAREHOUSE,0]
        draw.text((12,8),'Integrated warehouse {} | {:.1f} x {:.1f}m | all agents start outside'.format(
            order+1,float(dimensions[1]),float(dimensions[0])),fill='white',font=font)
        labels = []
        for person in range(task.num_agents):
            pre = int(task._before_predecessor[e,person])
            leader = bool((task._before_predecessor[e] == person).any())
            role = 'sideways PUSH' if person == int(task._roles[e,0]) else (
                'BEFORE follower of P{}'.format(pre) if pre >= 0 else 'BEFORE leader' if leader else 'independent')
            label = 'P{} {:6s} {}'.format(person,TASK_NAMES[int(task._templates[e,person])],role)
            labels.append(label)
            color = tuple(int(255*x) for x in agent_rgb(task,person))
            draw.rectangle((12,48+30*person,30,68+30*person),fill=color)
            draw.text((42,44+30*person),label,fill=(235,235,235),font=font)
        draw.text((12,154),'CLOSED - production reset',fill='white',font=font)
        draw.text((width+12,154),'OPEN - box placed at goal, no policy rollout',fill='white',font=font)
        canvas.save(output/(name+'.png'))
        images.append(canvas)
        report.append(dict(file=name+'.png',env=e,size=dimensions.tolist(),roles=labels))
    cell = (images[0].width//2,images[0].height//2)
    sheet = Image.new('RGB',(2*cell[0],3*cell[1]),(20,22,26))
    for i,canvas in enumerate(images):
        sheet.paste(canvas.resize(cell),(i%2*cell[0],i//2*cell[1]))
    sheet.save(output/'contact_sheet.png')
    (output/'samples.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PRODUCTION CAPTURE COMPLETE',output,flush=True)
    task.gym.destroy_viewer(task.viewer)
    task.gym.destroy_sim(task.sim)


if __name__ == '__main__':
    main()
