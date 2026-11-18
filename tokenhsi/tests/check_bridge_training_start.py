import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).resolve().parents[1])]
import isaacgym
from env.tasks.base_task import BaseTask
from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from utils.parse_task import parse_task
from env.tasks.multi_agent import square_sensor_scene as scene
from check_warehouse_sensor import inspect as inspect_warehouse, check_motion
import check_sensor_stage2 as checks
from isaacgym import gymapi
import numpy as np
from PIL import Image, ImageDraw
import torch

from utils.sensor_stage2_spec import BRIDGE, WAREHOUSE


base_init = BaseTask.__init__


def camera_init(self, cfg, enable_camera_sensors=False):
    base_init(self, cfg, enable_camera_sensors=True)


def camera(task, env, target, offset, width, height):
    props = gymapi.CameraProperties()
    props.width, props.height, props.horizontal_fov = width, height, 65.
    handle = task.gym.create_camera_sensor(task.envs[env], props)
    if handle < 0:
        raise RuntimeError('Camera creation failed')
    task.gym.set_camera_location(handle, task.envs[env], gymapi.Vec3(*(target+offset)), gymapi.Vec3(*target))
    return handle


def pixels(task, env, handle, width, height):
    data = task.gym.get_camera_image(task.sim, task.envs[env], handle, gymapi.IMAGE_COLOR)
    return Image.fromarray(np.asarray(data).reshape(height, width, 4)[..., :3])


def inspect_starts(task, totals):
    ids = (task._scene_ids == BRIDGE).nonzero().flatten()
    mode, count = task._bridge_start_mode[ids], task._bridge_count[ids]
    totals += torch.bincount(mode*3+count-1, minlength=9).reshape(3, 3).cpu()
    boxes = task._box_states[ids, task._bridge_slot[ids]]
    size = task._box_size[ids, task._bridge_slot[ids]]
    xy = checks.local_xy(task, boxes[:, :2], ids, task._scene_ids[ids])
    landing = task._bridge_landing[ids]
    present, near, inserted = mode == 0, mode == 1, mode == 2
    torch.testing.assert_close(xy[present], landing[present], atol=.002, rtol=0)
    delta = (xy[:, 1]-landing[:, 1]).abs()
    assert ((delta[near] >= .1-.002) & (delta[near] <= 1.5+.002)).all()
    depth = task._bridge_half_a[ids, 1]-(xy[:, 1]-size[:, 1]/2)
    torch.testing.assert_close(depth[inserted], torch.full_like(depth[inserted], .015), atol=.002, rtol=0)
    torch.testing.assert_close(task._bridge_far[ids, 1]+task._bridge_half_b[ids, 1],
                               task._bridge_half_a[ids, 1], atol=.002, rtol=0)
    p = ids[present]
    person = task._roles[p, 1]
    root = checks.local_xy(task, task._humanoid_root_states[p, person, :2], p, task._scene_ids[p])
    edge_distance = (root[:, 0]-task._bridge_landing[p, 0]).abs()-task._bridge_gap[p]/2
    assert ((edge_distance >= .8-.002) & (edge_distance <= 2.+.002)).all()
    source = task._agent_box_assignment[p, person]
    payload = checks.local_xy(task, task._box_states[p, source, :2], p, task._scene_ids[p])
    carrying = task._bridge_carry_from_start[p]
    distance = (payload-root).norm(dim=-1)
    assert ((distance[carrying] >= 1.-.002) & (distance[carrying] <= 2.+.002)).all()
    template = task._templates[p, person]
    support = 3*task.num_agents+person
    target = torch.where((template == 3)[:, None], task._tar_pos[p, person, :2],
        torch.where((template == 4)[:, None], task._box_states[p, support, :2], task._box_states[p, source, :2]))
    target = torch.where(carrying[:, None], target, task._box_states[p, source, :2])
    target = checks.local_xy(task, target, p, task._scene_ids[p])
    landing = task._bridge_landing[p]
    slope = (target[:, 1]-landing[:, 1])/(target[:, 0]-landing[:, 0])
    torch.testing.assert_close(root[:, 1], landing[:, 1]+(root[:, 0]-landing[:, 0])*slope, atol=.002, rtol=0)
    margin = task._box_size[p, task._bridge_slot[p], 1]/2-slope.abs()*task._bridge_gap[p]/2
    assert (margin >= .2-.002).all()


def screenshots(task, output):
    cameras = []
    for mode, name in enumerate(('present', 'near', 'insert')):
        for count in (1, 2, 3):
            chosen = ((task._scene_ids == BRIDGE) & (task._bridge_start_mode == mode) & (task._bridge_count == count)).nonzero().flatten()
            if not len(chosen):
                raise RuntimeError('Missing screenshot mode/count')
            e = int(chosen[0])
            target = task._static_centers[e, BRIDGE, :2].mean(0).cpu().numpy()
            radius = float(task._bridge_gap[e]+2*task._bridge_half_a[e, 0]+2*task._bridge_half_b[e, 0])
            offset = scene.rotate_xy(torch.tensor([[.15*radius, radius]]), task._structure_yaw[e, BRIDGE].cpu())[0].numpy()
            handle = camera(task, e, target, np.array([*offset, radius]), 1280, 720)
            cameras.append((e, name, count, handle))
    chosen = (task._scene_ids == WAREHOUSE).nonzero().flatten()
    if not len(chosen):
        raise RuntimeError('Missing warehouse screenshot')
    e = int(chosen[0])
    center = task._static_centers[e, WAREHOUSE, 0].cpu().numpy()
    target = .5*center+.5*task._humanoid_root_states[e, :, :3].mean(0).cpu().numpy()
    offset = scene.rotate_xy(torch.tensor([[4., 10.]]), task._structure_yaw[e, WAREHOUSE].cpu())[0].numpy()
    handle = camera(task, e, target, np.array([*offset, 12.]), 1280, 720)
    cameras.append((e, 'warehouse', 1, handle))
    return cameras


def main():
    args = get_args()
    cfg, train, _ = load_cfg(args)
    cfg['env']['motion_file'], cfg['args'] = args.motion_file, args
    torch.set_num_threads(4)
    set_seed(42)
    BaseTask.__init__ = camera_init
    task, env = parse_task(args, cfg, train['params']['config'], parse_sim_params(args, cfg, train))
    output = Path(args.output_path)
    output.mkdir(parents=True, exist_ok=True)
    stats = dict(scenes={name:dict(episodes=0, before_pairs=0, episodes_with_pair=0, unreachable=0) for name in task._scene_names})
    totals = torch.zeros(3, 3, dtype=torch.long)
    fixed = task._root_states[task._terrain_actor_records[:, -1]].clone()
    for iteration in range(4):
        obs = env.reset()
        checks.inspect(task, stats)
        stats['warehouse_samples'] = stats.get('warehouse_samples', 0)+inspect_warehouse(task)
        if task._warehouse_start:
            warehouse = task._scene_ids == WAREHOUSE
            modes = task._warehouse_start_mode[warehouse]
            counts = torch.bincount(modes,minlength=3).cpu().tolist()
            previous = stats.setdefault('warehouse_start_counts',[0,0,0])
            stats['warehouse_start_counts'] = [a+b for a,b in zip(previous,counts)]
        checks.check_sensor(task, obs, stats)
        inspect_starts(task, totals)
        records = task._terrain_actor_records
        e, kind, variant, slot, index = records.unbind(-1)
        active = (kind == task._scene_ids[e]) & ((kind != BRIDGE) | (variant == task._bridge_count[e]-1))
        torch.testing.assert_close(task._root_states[index[active], :3], task._static_centers[e[active], kind[active], slot[active]], atol=.002, rtol=0)
        torch.testing.assert_close(fixed, task._root_states[index], atol=0, rtol=0)
        if iteration == 0:
            cameras = screenshots(task, output)
            obs, _, _, _ = env.step(torch.zeros(task.num_envs*4, task.num_actions, device=task.device))
            task.gym.fetch_results(task.sim, True)
            task.gym.step_graphics(task.sim)
            task.gym.render_all_camera_sensors(task.sim)
            sheet = Image.new('RGB', (1920, 1440))
            for k, (e, mode, count, camera) in enumerate(cameras):
                frame = pixels(task, e, camera, 1280, 720)
                if np.asarray(frame).std() < 2:
                    raise RuntimeError('Empty camera image')
                draw = ImageDraw.Draw(frame)
                a = int(task._roles[e, 1])
                draw.rectangle((0, 0, 1280, 36), fill=(20, 23, 28))
                role = f'crosser P{a}' if mode != 'warehouse' else f'pusher P{int(task._roles[e, 0])}'
                draw.text((12, 12), f'{mode.upper()} | {count} PUSHERS | {role} | env {e}', fill='white')
                frame.save(output/f'{mode}_{count}push.png')
                sheet.paste(frame.resize((640, 360)), ((k%3)*640, (k//3)*360))
            sheet.save(output/'starts_overview.png')
        assert torch.isfinite(obs).all()
    before = task._root_states.clone()
    modes = task._bridge_start_mode.clone()
    warehouse_modes = task._warehouse_start_mode.clone()
    env.reset(torch.tensor([0], device=task.device))
    actors = task.get_num_actors_per_env()
    torch.testing.assert_close(before[actors:], task._root_states[actors:], atol=0, rtol=0)
    assert torch.equal(modes[1:], task._bridge_start_mode[1:])
    assert torch.equal(warehouse_modes[1:],task._warehouse_start_mode[1:])
    warehouse_ids = (task._scene_ids == WAREHOUSE).nonzero().flatten()
    assert (task._bridge_count[warehouse_ids] == 1).all()
    actual = torch.tensor([stats['scenes'][name]['episodes'] for name in task._scene_names]).float()
    actual /= actual.sum()
    torch.testing.assert_close(actual, torch.tensor([.05, .4, .25, .3]), atol=.04, rtol=0)
    stats['scene_fractions'] = actual.tolist()
    if task._warehouse_start:
        counts = torch.tensor(stats['warehouse_start_counts']).float()
        torch.testing.assert_close(counts/counts.sum(),torch.tensor([.5,.25,.25]),atol=.04,rtol=0)
    stats['warehouse_motion'] = check_motion(task, env)
    task._consume_sampling_diagnostics()
    sample = totals.sum(1).float()/totals.sum()
    torch.testing.assert_close(sample, torch.tensor([.2, .4, .4]), atol=.04, rtol=0)
    stats.update(start_mode_by_pusher=totals.tolist(), start_fractions=sample.tolist(),
        num_envs=task.num_envs, fixed_actors=True, partial_reset=True, finite=True, screenshots=10)
    (output/'physics.json').write_text(json.dumps(stats, indent=2))
    task.gym.destroy_sim(task.sim)
    print(json.dumps(stats), flush=True)


if __name__ == '__main__':
    main()
