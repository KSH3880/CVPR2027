"""Live check of the apron BEFORE-pairs variant: ratio, roles, apron layout, parked boxes and short physics."""
import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).resolve().parents[1])]
import isaacgym
import torch

from env.tasks.multi_agent import square_sensor_scene as scene
from env.tasks.multi_agent.warehouse_before import apron_region
from utils.before_spec import task_objects
from utils.sensor_stage2_spec import WAREHOUSE


def local(task, ids, xy):
    shape = (len(ids),)+(1,)*(xy.ndim-2)
    offset = xy-task._env_origins[ids, :2].view(*shape, 2)-task._structure_shift[ids, WAREHOUSE].view(*shape, 2)
    return scene.rotate_xy(offset, -task._structure_yaw[ids, WAREHOUSE].view(shape))


def inside(points, center, half, tol=1e-3):
    shape = (len(center),)+(1,)*(points.ndim-2)
    return ((points-center.view(*shape, 2)).abs() <= half.view(*shape, 2)+tol).all(-1)


def inspect(task, stats):
    n, m = task.num_envs, task.num_agents
    assert (task._scene_ids == WAREHOUSE).all()
    flag = task._warehouse_before_pairs
    stats['envs'] += n
    stats['flagged'] += int(flag.sum())
    rows = torch.arange(n, device=task.device)[:, None]
    templates, pre = task._templates, task._before_predecessor
    # Fixed episodes keep two pushers and one BEFORE pair.
    plain = (~flag).nonzero().flatten()
    assert ((templates[plain] == 5).sum(-1) == 2).all() and ((pre[plain] >= 0).sum(-1) == 1).all()
    ids = flag.nonzero().flatten()
    if not len(ids):
        return
    t, p = templates[ids], pre[ids]
    assert not (t == 5).any() and ((p >= 0).sum(-1) == 2).all()
    if task._state_init == task.StateInit.Default:
        torch.testing.assert_close(task._tar_pos[ids][p >= 0], task._humanoid_root_states[ids, :, :3][p >= 0],
                                   rtol=0, atol=1e-3)
    e, f = (p >= 0).nonzero(as_tuple=True)
    assert (t[e, p[e, f]] == 3).all()
    for family, name in ((1, 'SIT'), (2, 'CLIMB'), (4, 'ON_TOP')):
        stats['followers'][name] += int((t[e, f] == family).sum())
    center, half = apron_region(task, ids)
    roots = local(task, ids, task._humanoid_root_states[ids, :, :2])
    assert inside(roots, center, half).all(), 'people must start on the apron'
    valid = task_objects(p, t)
    support = 3*m+torch.arange(m, device=task.device)[None].expand(len(ids), -1)
    slots = torch.cat((task._agent_box_assignment[ids], support), 1)
    boxes = task._box_states[ids[:, None], slots]
    xy = local(task, ids, boxes[..., :2])
    assert inside(xy, center, half)[valid].all(), 'task boxes must lie on the apron'
    floor = task._env_origins[ids, 2, None]+task._warehouse['floor_height']
    z = boxes[..., 2]-task._box_size[ids[:, None], slots, 2]/2-floor
    assert (z[valid].abs() < .02).all(), 'task boxes must rest on the apron'
    at = t == 3
    if task._state_init == task.StateInit.Default:
        source = task._box_states[ids[:, None], task._agent_box_assignment[ids], :2]
        distance = (task._humanoid_root_states[ids, :, :2]-source).norm(dim=-1)
        maximum = task.cfg['env']['stage2JointPush']['solo_source_max_distance']
        assert ((distance[at] >= 1.-1e-3) & (distance[at] <= maximum+1e-3)).all(), 'shared source was resampled by a follower'
    goals = local(task, ids, task._tar_pos[ids, :, :2])
    assert inside(goals, center, half)[at].all(), 'AT goals must lie on the apron'
    door = task._box_states[ids, task._bridge_slot[ids], :2]-task._env_origins[ids, :2]
    assert (door.norm(dim=-1) > 100).all(), 'the unused door box stays parked'
    active = task._active_physical(ids)
    used = torch.zeros_like(active)
    r, s = valid.nonzero(as_tuple=True)
    used[r, slots[r, s]] = True
    parked = active & ~used
    if parked.any():
        r, s = parked.nonzero(as_tuple=True)
        spare = local(task, ids, task._box_states[ids[:, None].expand(-1, task.num_physical_objects), torch.arange(
            task.num_physical_objects, device=task.device)[None].expand(len(ids), -1), :2])
        assert not inside(spare[r, s][:, None], center[r], half[r])[:, 0].any(), 'spare boxes must leave the apron'
    stats['root_height_above_apron_min'] = min(stats['root_height_above_apron_min'],
        float((task._humanoid_root_states[ids, :, 2]-floor).min()))


def capture(task, output):
    """Top-down-oblique images of three apron-team scenes and one fixed-role scene."""
    from isaacgym import gymapi
    import numpy as np
    from PIL import Image
    flagged = task._warehouse_before_pairs.nonzero().flatten()[:3].tolist()
    fixed = (~task._warehouse_before_pairs).nonzero().flatten()[:1].tolist()
    cameras = []
    for e in flagged+fixed:
        ids = torch.tensor([e], device=task.device)
        center, _ = apron_region(task, ids)
        focus_xy = scene.to_world(center[:, None], task._structure_yaw[ids, WAREHOUSE],
                                  task._structure_shift[ids, WAREHOUSE], task._env_origins[ids])[0, 0]
        focus = torch.cat((focus_xy, task._env_origins[e, 2:]+.5))
        offset = scene.rotate_xy(focus.new_tensor([[-7., -7.]]), task._structure_yaw[e, WAREHOUSE, None])[0]
        props = gymapi.CameraProperties()
        props.width, props.height, props.horizontal_fov = 1280, 900, 70.
        handle = task.gym.create_camera_sensor(task.envs[e], props)
        eye = focus+torch.cat((offset, offset.new_tensor([11.])))
        task.gym.set_camera_location(handle, task.envs[e], gymapi.Vec3(*eye.tolist()), gymapi.Vec3(*focus.tolist()))
        cameras.append((e, 'apron_pairs' if e in flagged else 'fixed_roles', handle))
    task.gym.simulate(task.sim)
    task.gym.fetch_results(task.sim, True)
    task._refresh_sim_tensors()
    task.gym.step_graphics(task.sim)
    task.gym.render_all_camera_sensors(task.sim)
    for e, name, handle in cameras:
        pixels = task.gym.get_camera_image(task.sim, task.envs[e], handle, gymapi.IMAGE_COLOR)
        Image.fromarray(np.asarray(pixels).reshape(900, 1280, 4)[..., :3]).save(output/f'{name}_env{e}.png')
        task.gym.destroy_camera_sensor(task.sim, task.envs[e], handle)


def main():
    import os
    from utils.config import get_args, load_cfg, parse_sim_params, set_seed
    from utils.parse_task import parse_task
    args = get_args()
    torch.set_num_threads(4)
    cfg, train, _ = load_cfg(args)
    cfg['env']['motion_file'], cfg['args'] = args.motion_file, args
    set_seed(42)
    if os.environ.get('CAPTURE') == '1':
        from env.tasks.base_task import BaseTask
        original = BaseTask.__init__
        BaseTask.__init__ = lambda self, cfg, enable_camera_sensors=False: original(self, cfg, enable_camera_sensors=True)
    task, env = parse_task(args, cfg, train['params']['config'], parse_sim_params(args, cfg, train))
    if os.environ.get('CAPTURE') == '1':
        env.reset()
        output = Path(args.output_path)
        output.mkdir(parents=True, exist_ok=True)
        capture(task, output)
        print('captured', flush=True)
        return
    stats = dict(envs=0, flagged=0, followers=dict(SIT=0, CLIMB=0, ON_TOP=0), root_height_above_apron_min=1e9)
    retries = []
    for _ in range(6):
        before = float(task._sampling_retries)
        env.reset()
        inspect(task, stats)
        retries.append(float(task._sampling_retries)-before)
    from env.tasks.multi_agent import square_sensor as sensor
    action = torch.zeros(task.num_envs*task.num_agents, task.get_action_size(), device=task.device)
    flag = task._warehouse_before_pairs.clone()
    push = task._templates == 5
    feet = sensor.lowest_foot_height(task._kinematic_humanoid_rigid_body_states, task._foot_body_ids, task._foot_vertices)
    feet = feet-task._env_origins[:, 2, None]-task._warehouse['floor_height']
    ended = torch.zeros(task.num_envs, task.num_agents, dtype=torch.bool, device=task.device)
    wall = torch.zeros_like(ended)
    for _ in range(30):
        obs, reward, _, _ = env.step(action)
        assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
        ended |= task._terminate_buf.view(task.num_envs, task.num_agents).bool()
        wall |= task._wall_contact_steps >= 9
    stats['flagged_fraction'] = stats['flagged']/stats['envs']
    stats['sampling_retries_per_reset'] = retries
    groups = dict(flagged=flag[:, None].expand_as(push), fixed_workers=~flag[:, None] & ~push, fixed_pushers=~flag[:, None] & push)
    stats['zero_action_30step_termination'] = {k: float(ended[v].float().mean()) for k, v in groups.items()}
    stats['zero_action_30step_wall_contact_limit'] = {k: float(wall[v].float().mean()) for k, v in groups.items()}
    stats['start_foot_height_above_floor'] = {k: [float(feet[v].min()), float(feet[v].median())] for k, v in groups.items()}
    diagnostics = task._consume_sampling_diagnostics()
    stats['variant_diagnostics'] = {k: float(v) for k, v in diagnostics.items() if k.startswith('warehouse/variant/')}
    output = Path(args.output_path)
    output.mkdir(parents=True, exist_ok=True)
    (output/'beforepairs.json').write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats), flush=True)


if __name__ == '__main__':
    main()
