"""Live check of leader-done Warehouse starts: ratio, open door, delivered box, leader spot, release and short physics."""
import json
import os
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).resolve().parents[1])]
import isaacgym
import torch

from env.tasks.multi_agent import square_sensor_scene as scene
from utils.sensor_stage2_spec import WAREHOUSE


def local(task, ids, xy):
    shape = (len(ids),)+(1,)*(xy.ndim-2)
    offset = xy-task._env_origins[ids, :2].view(*shape, 2)-task._structure_shift[ids, WAREHOUSE].view(*shape, 2)
    return scene.rotate_xy(offset, -task._structure_yaw[ids, WAREHOUSE].view(shape))


def pairs(task, ids):
    pre = task._before_predecessor[ids]
    follower = (pre >= 0).long().argmax(1)
    return follower, pre[torch.arange(len(ids), device=task.device), follower]


def inspect(task, stats):
    n = task.num_envs
    assert (task._scene_ids == WAREHOUSE).all()
    flag = task._warehouse_leader_done
    stats['envs'] += n
    stats['flagged'] += int(flag.sum())
    templates, pre = task._templates, task._before_predecessor
    assert ((templates == 5).sum(-1) == 2).all() and ((pre >= 0).sum(-1) == 1).all()
    ids = flag.nonzero().flatten()
    if not len(ids):
        return
    rows = torch.arange(len(ids), device=task.device)
    assert (task._warehouse_start_mode[ids] == 2).all()
    door = task._box_states[ids, task._bridge_slot[ids], :2]
    stats['door_to_landing_max'] = max(stats['door_to_landing_max'], float((door-task._warehouse_landing[ids, :2]).norm(dim=-1).max()))
    follower, leader = pairs(task, ids)
    slot = task._agent_box_assignment[ids, leader]
    box, goal = task._box_states[ids, slot, :3], task._tar_pos[ids, leader]
    stats['box_to_goal_max'] = max(stats['box_to_goal_max'], float((box-goal).norm(dim=-1).max()))
    half = task._box_size[ids, slot, :2]/2
    roots = local(task, ids, task._humanoid_root_states[ids, :, :2])
    centre = local(task, ids, box[:, None, :2])[:, 0]
    gap = ((roots[rows, leader]-centre).abs()-half).clamp_min(0).norm(dim=-1)
    stats['leader_gap_min'] = min(stats['leader_gap_min'], float(gap.min()))
    stats['leader_gap_max'] = max(stats['leader_gap_max'], float(gap.max()))
    hx = task._surface_half[ids, WAREHOUSE, 0, 0]
    front = -hx-task._warehouse['wall_thickness']
    stats['leader_margin_to_front_wall_min'] = min(stats['leader_margin_to_front_wall_min'], float((front-roots[rows, leader, 0]).min()))
    stats['follower_distance_to_box'].extend((roots[rows, follower]-centre).norm(dim=-1).tolist())
    distance = (roots[:, :, None]-roots[:, None]).norm(dim=-1)+torch.eye(task.num_agents, device=task.device)*99
    stats['people_distance_min'] = min(stats['people_distance_min'], float(distance.min()))
    stats['near_pairs_left'] += int(task._before_near_pairs[ids].sum())


def main():
    from utils.config import get_args, load_cfg, parse_sim_params, set_seed
    from utils.parse_task import parse_task
    args = get_args()
    torch.set_num_threads(4)
    cfg, train, _ = load_cfg(args)
    cfg['env']['motion_file'], cfg['args'] = args.motion_file, args
    set_seed(42)
    capture = os.environ.get('CAPTURE') == '1'
    if capture:
        from env.tasks.base_task import BaseTask
        original = BaseTask.__init__
        BaseTask.__init__ = lambda self, cfg, enable_camera_sensors=False: original(self, cfg, enable_camera_sensors=True)
    task, env = parse_task(args, cfg, train['params']['config'], parse_sim_params(args, cfg, train))
    output = Path(args.output_path)
    output.mkdir(parents=True, exist_ok=True)
    assert task._warehouse['wall_cpa'] == 'box' and task._warehouse_done_probability == .3
    if capture:
        env.reset()
        save_images(task, output)
        print('captured', flush=True)
        return
    stats = dict(envs=0, flagged=0, door_to_landing_max=0., box_to_goal_max=0., leader_gap_min=9., leader_gap_max=0.,
                 leader_margin_to_front_wall_min=9., follower_distance_to_box=[], people_distance_min=9., near_pairs_left=0)
    retries = []
    for _ in range(6):
        before = float(task._sampling_retries)
        env.reset()
        inspect(task, stats)
        retries.append(float(task._sampling_retries)-before)
    distances = torch.tensor(stats.pop('follower_distance_to_box'))
    stats['follower_distance_to_box_quantiles'] = distances.quantile(torch.tensor([.1, .5, .9])).tolist()
    stats['flagged_fraction'] = stats['flagged']/stats['envs']
    stats['sampling_retries_per_reset'] = retries
    flag = task._warehouse_leader_done.clone()
    ids = flag.nonzero().flatten()
    follower, leader = pairs(task, ids)
    rows = torch.arange(len(ids), device=task.device)
    action = torch.zeros(task.num_envs*task.num_agents, task.get_action_size(), device=task.device)
    push = task._templates == 5
    ended = torch.zeros(task.num_envs, task.num_agents, dtype=torch.bool, device=task.device)
    for step in range(30):
        obs, reward, _, _ = env.step(action)
        assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
        if step == 0:
            stats['after_1_step_leader_success'] = float(task._edge_ever_goal[ids, leader].float().mean())
            wait = task._before_wait
            mine = flag[wait.rows]
            stats['after_1_step_follower_released'] = float(wait.released[wait.rows[mine], wait.edges[mine]].float().mean())
            stats['after_1_step_follower_released_main'] = float(wait.released[wait.rows[~mine], wait.edges[~mine]].float().mean())
            stats['after_1_step_leader_reward_done_vs_main'] = [float(reward.view(task.num_envs, -1)[ids, leader].mean()),
                float(reward.view(task.num_envs, -1)[~flag].mean())]
        ended |= task._terminate_buf.view(task.num_envs, task.num_agents).bool()
        if step in (0, 1, 9, 29):
            # A one-off delivery bonus would show as a step-0 spike; the open-start pushers are already done too.
            per = reward.view(task.num_envs, -1)
            opened = (~flag & (task._warehouse_start_mode == 2))[:, None] & push
            stats.setdefault('reward_by_step_done_leader_vs_open_start_pusher', []).append(
                [step+1, float(per[ids, leader].mean()), float(per[opened].mean())])
    stats['after_30_steps_leader_success'] = float(task._edge_ever_goal[ids, leader].float().mean())
    leader_mask = torch.zeros_like(push)
    leader_mask[ids, leader] = True
    groups = dict(done_leader=leader_mask, done_follower=flag[:, None] & ~push & ~leader_mask,
                  done_pushers=flag[:, None] & push, main_workers=~flag[:, None] & ~push, main_pushers=~flag[:, None] & push)
    stats['zero_action_30step_termination'] = {k: float(ended[v].float().mean()) for k, v in groups.items()}
    diagnostics = task._consume_sampling_diagnostics()
    stats['diagnostics'] = {k: float(v) for k, v in diagnostics.items() if k.startswith(('warehouse/main/', 'warehouse/leader_done/'))}
    (output/'leaderdone.json').write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats), flush=True)


def save_images(task, output):
    from isaacgym import gymapi
    import numpy as np
    from PIL import Image
    flagged = task._warehouse_leader_done.nonzero().flatten()[:3].tolist()
    cameras = []
    for e in flagged+(~task._warehouse_leader_done).nonzero().flatten()[:1].tolist():
        ids = torch.tensor([e], device=task.device)
        follower, leader = pairs(task, ids)
        focus = task._box_states[e, task._agent_box_assignment[e, leader[0]], :3].clone()
        offset = scene.rotate_xy(focus.new_tensor([[-6., -6.]]), task._structure_yaw[e, WAREHOUSE, None])[0]
        props = gymapi.CameraProperties()
        props.width, props.height, props.horizontal_fov = 1280, 900, 70.
        handle = task.gym.create_camera_sensor(task.envs[e], props)
        eye = focus+torch.cat((offset, offset.new_tensor([9.])))
        task.gym.set_camera_location(handle, task.envs[e], gymapi.Vec3(*eye.tolist()), gymapi.Vec3(*focus.tolist()))
        cameras.append((e, 'leader_done' if e in flagged else 'main', handle))
    task.gym.simulate(task.sim)
    task.gym.fetch_results(task.sim, True)
    task._refresh_sim_tensors()
    task.gym.step_graphics(task.sim)
    task.gym.render_all_camera_sensors(task.sim)
    for e, name, handle in cameras:
        pixels = task.gym.get_camera_image(task.sim, task.envs[e], handle, gymapi.IMAGE_COLOR)
        Image.fromarray(np.asarray(pixels).reshape(900, 1280, 4)[..., :3]).save(output/f'{name}_env{e}.png')
        task.gym.destroy_camera_sensor(task.sim, task.envs[e], handle)


if __name__ == '__main__':
    main()
