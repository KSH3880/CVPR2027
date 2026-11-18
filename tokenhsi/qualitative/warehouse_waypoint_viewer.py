import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import isaacgym
from isaacgym import gymapi
import numpy as np
import torch
import yaml

from qualitative.warehouse_waypoint import (
    ONTOP_TEMPLATE, ONTOP_WAYPOINT_RADIUS, switch_margin, update_switch, update_return,
)
from qualitative.warehouse_waypoint_policy import WaypointPolicy
from qualitative.warehouse_waypoint_task import WarehouseBoxOnlyViewerTask, WarehouseWaypointTask
from learning.multi_agent.warehouse_carry_eval import WarehouseCarryEval
from env.tasks.multi_agent.vec_task_wrapper_ma import VecTaskPythonWrapperMA
from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from utils.experiments import SENSOR_WAREHOUSE_WAYPOINT_BOXONLY
from utils.sensor_stage2_spec import WAREHOUSE

DEFAULT_CHECKPOINT = '/home/hwanhee/jhh/TokenHSI_stage12_bridge_fix_reward/output/stage2/Stage2Distill900V4_00002000.pth'


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--switch-margin', type=switch_margin, default=1.)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--envs', type=int, default=1)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--steps', type=int, default=900)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--capture', action='store_true')
    parser.add_argument('--follower-ontop-waypoint', action='store_true',
                        help='Guide ON_TOP followers and their carried payloads through the entrance')
    parser.add_argument('--box-only', action='store_true',
                        help='Use the box-only variant with the original DistillV4 policy')
    parser.add_argument('--ontop-only', action='store_true', help='Sample only ON_TOP followers for inference')
    parser.add_argument('--record', action='store_true', help='Record one rendered episode as MP4')
    parser.add_argument('--out')
    opts = parser.parse_args()
    if min(opts.envs, opts.episodes, opts.steps) < 1:
        parser.error('envs, episodes and steps must be positive')
    if opts.record and (opts.headless or opts.envs != 1 or opts.episodes != 1):
        parser.error('--record requires a viewer, --envs 1 and --episodes 1')
    opts.out = opts.out or ('output_etc/check/warehouse_waypoint/' + time.strftime('%Y%m%d_%H%M%S'))
    return opts


def environment(opts):
    output = Path(opts.out).resolve()
    if Path('output_etc/check').resolve() not in output.parents:
        raise ValueError('--out must be below output_etc/check/')
    box_only = getattr(opts, 'box_only', False)
    name = (SENSOR_WAREHOUSE_WAYPOINT_BOXONLY if box_only
            else 'stage2_sensor_900_warehouse_fixedroles_v4_beforepairs_doorprogress')
    sys.argv = [sys.argv[0], '--task', 'HumanoidMACoupledPush', '--test', '--eval',
        '--checkpoint', opts.checkpoint, '--cfg_train', f'tokenhsi/data/cfg/train/rlg/{name}.yaml',
        '--cfg_env', f'tokenhsi/data/cfg/multi_agent/{name}.yaml', '--num_envs', str(opts.envs),
        '--num_agents', '4', '--num_objects', '8', '--seed', str(opts.seed),
        '--episode_length', str(opts.steps), '--motion_file', 'tokenhsi/data/dataset_stage1_jointpush.yaml',
        '--no_video', '--output_path', str(output)]
    if opts.headless:
        sys.argv.append('--headless')
    args = get_args()
    cfg, train, _ = load_cfg(args)
    cfg['env']['motion_file'] = args.motion_file
    cfg['env']['cameraFollow'] = False
    os.environ['WAREHOUSE_VARIANT'] = 'push_before'
    os.environ['WAREHOUSE_START'] = 'closed'
    set_seed(opts.seed)
    task_type = WarehouseBoxOnlyViewerTask if box_only else WarehouseWaypointTask
    task = task_type(cfg=cfg, sim_params=parse_sim_params(args, cfg, train),
        physics_engine=args.physics_engine, device_type=args.device, device_id=args.device_id,
        headless=args.headless)
    task.ontop_only = getattr(opts, 'ontop_only', False)
    env = VecTaskPythonWrapperMA(task, args.rl_device, float('inf'), 1.)
    output.mkdir(parents=True, exist_ok=True)
    (output/'resolved_env.yaml').write_text(yaml.safe_dump(
        {key: cfg[key] for key in ('env', 'sim', 'experiment')}, sort_keys=False))
    return task, env, output


def draw(task, virtual, sizes, radius, switched, returning, leader, ontop=None):
    if task.viewer is None:
        return
    gym = task.gym
    gym.clear_lines(task.viewer)
    env = task.envs[0]
    origin = task.gym.get_env_origin(env)
    offset = np.array([origin.x, origin.y, origin.z])
    center = virtual[0].cpu().numpy()-offset
    color = (.6, .6, .6) if bool(switched[0]) else (1., .15, 1.)
    if returning[0]:
        color = (1., .5, .1)
    angle = np.linspace(0, 2*np.pi, 97)
    circle = np.column_stack((center[0]+float(radius[0])*np.cos(angle),
        center[1]+float(radius[0])*np.sin(angle), np.full_like(angle, center[2])))
    lines = [np.stack((circle[:-1], circle[1:]), 1)]
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    corners = corners*sizes[0].cpu().numpy()/2
    yaw = float(task._structure_yaw[0, WAREHOUSE])
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    corners[:, :2] = corners[:, :2]@rotation.T
    corners += center
    edges = [(i, j) for i in range(8) for j in range(i+1, 8) if (i ^ j) in (1, 2, 4)]
    lines.append(np.array([[corners[i], corners[j]] for i, j in edges]))
    lines = np.concatenate(lines).astype(np.float32)
    gym.add_lines(task.viewer, env, len(lines), lines, np.tile(color, (len(lines), 1)).astype(np.float32))
    payload = task._agent_box_assignment[0, leader[0]]
    markers = [(task._box_states[0, payload, :3], (.1, 1., 1.)),
               (task._tar_pos[0, leader[0]], (.1, 1., .1))]
    if ontop is not None:
        follower, active, target = ontop
        follower_payload = task._agent_box_assignment[0, follower[0]]
        markers.append((task._box_states[0, follower_payload, :3], (.2, .55, 1.)))
        for person, rgb in ((leader, (1., .15, 1.)), (follower, (.2, .55, 1.))):
            root = task._humanoid_root_states[0, person[0], :3].cpu().numpy()-offset
            ring = np.column_stack((root[0]+.35*np.cos(angle), root[1]+.35*np.sin(angle),
                                    np.full_like(angle, root[2]+.8)))
            segments = np.stack((ring[:-1], ring[1:]), 1).astype(np.float32)
            gym.add_lines(task.viewer, env, len(segments), segments,
                          np.tile(rgb, (len(segments), 1)).astype(np.float32))
        if active[0]:
            ring_color = (1., .85, .15) if target[0] else (.2, .55, 1.)
            ring = np.column_stack((center[0]+float(radius[0])*np.cos(angle),
                                    center[1]+float(radius[0])*np.sin(angle),
                                    np.full_like(angle, center[2]+.08)))
            segments = np.stack((ring[:-1], ring[1:]), 1).astype(np.float32)
            gym.add_lines(task.viewer, env, len(segments), segments,
                          np.tile(ring_color, (len(segments), 1)).astype(np.float32))
    for point, rgb in markers:
        point = point.cpu().numpy()-offset
        delta = np.eye(3)*.3
        cross = np.stack((point-delta, point+delta), 1).astype(np.float32)
        gym.add_lines(task.viewer, env, 3, cross, np.tile(rgb, (3, 1)).astype(np.float32))


def camera(task, virtual):
    if task.viewer is None:
        return
    env = task.envs[0]
    origin = task.gym.get_env_origin(env)
    center = virtual[0].cpu().numpy()-np.array([origin.x, origin.y, origin.z])
    yaw = float(task._structure_yaw[0, WAREHOUSE])
    c, s = np.cos(yaw), np.sin(yaw)
    center += np.array([1.5*c, 1.5*s, 0.])
    eye = center + np.array([-.01*c, -.01*s, 18.])
    task.gym.viewer_camera_look_at(task.viewer, env, gymapi.Vec3(*eye), gymapi.Vec3(*center))


def record_video(task, output, opts):
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'image2pipe',
           '-framerate', str(round(1/task.dt)), '-vcodec', 'png', '-i', '-']
    if opts.box_only:
        label = f"DistillV4 2000 | Virtual box only | Seed {opts.seed}"
        legend = ("Magenta ring - leader   Blue ring - ON_TOP follower   Door ring - virtual box approach"
                  "   Green cross - final stack goal")
    elif opts.follower_ontop_waypoint:
        label = f"DistillV4 2000 | ON_TOP follower | Seed {opts.seed}"
        legend = ("Magenta - leader   Blue - ON_TOP follower   Yellow door ring - follower waypoint goal"
                  "   Green cross - final stack goal")
    if opts.box_only or opts.follower_ontop_waypoint:
        cmd += ['-vf', f"drawtext=text='{label}':x=24:y=22:fontsize=26:fontcolor=white:"
                       "box=1:boxcolor=black@0.75:boxborderw=8,"
                       f"drawtext=text='{legend}':x=24:y=62:fontsize=19:fontcolor=white:"
                       "box=1:boxcolor=black@0.75:boxborderw=8"]
    cmd += ['-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(output/'warehouse.mp4')]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


@torch.no_grad()
def main():
    opts = parse()
    torch.set_num_threads(4)
    task, env, output = environment(opts)
    records = []
    encoder = None
    frame = output/'record_frame.png'
    try:
        if opts.record:
            encoder = record_video(task, output, opts)
        source = torch.load(opts.checkpoint, map_location='cpu', weights_only=False)
        policy = WaypointPolicy(task, source)
        ids = torch.arange(task.num_envs, device=task.device)
        fixed_radius = ONTOP_WAYPOINT_RADIUS if (opts.box_only or opts.follower_ontop_waypoint) else None
        if opts.box_only:
            print('[waypoint] box only: entrance radius 1.2m; original goals throughout', flush=True)
        else:
            print('[waypoint] magenta: inbound virtual box; orange: outbound virtual target; '
                  'cyan: real payload; green: final goal; gray: original observation', flush=True)
        for episode in range(opts.episodes):
            obs = env.reset()
            leader, follower = task.waypoint_people(ids)
            virtual, sizes = task.virtual_box(ids, leader)
            camera(task, virtual)
            carry = WarehouseCarryEval(task) if opts.box_only else None
            switched = torch.zeros(task.num_envs, dtype=torch.bool, device=task.device)
            switch_step = torch.full_like(ids, -1)
            picked = torch.zeros_like(switched)
            returned = torch.zeros_like(switched)
            follower_switched = torch.zeros_like(switched)
            follower_picked = torch.zeros_like(switched)
            follower_returned = torch.zeros_like(switched)
            follower_switch_step = torch.full_like(ids, -1)
            follower_pickup_step = torch.full_like(ids, -1)
            follower_return_step = torch.full_like(ids, -1)
            follower_enter_step = torch.full_like(ids, -1)
            follower_hold_step = torch.full_like(ids, -1)
            follower_ontop_step = torch.full_like(ids, -1)
            ontop = ((task._templates[ids, follower] == ONTOP_TEMPLATE) &
                     ~task._warehouse_before_pairs[ids])
            pickup_step = torch.full_like(ids, -1)
            return_step = torch.full_like(ids, -1)
            ended = torch.zeros_like(switched)
            final_goals = task._tar_pos.clone()
            for step in range(opts.steps):
                started = time.monotonic()
                if task.viewer is not None and task.gym.query_viewer_has_closed(task.viewer):
                    return
                if opts.box_only:
                    # Read the rollout metadata produced by the actual box-only task.
                    metadata = obs[:, -task._waypoint_obs_width:].masked_fill(ended[:, None], -1)
                    phases = metadata[:, [1, 6]]
                    changed = (phases[:, 0] > 0) & ~switched
                    changed_follower = (phases[:, 1] > 0) & ~follower_switched
                    switch_step[changed] = step
                    follower_switch_step[changed_follower] = step
                    switched |= changed
                    follower_switched |= changed_follower
                    radius = virtual.new_full((task.num_envs,), fixed_radius)
                    returning = torch.zeros_like(switched)
                    markers = (follower, phases[:, 1] == 0, returning) if task.viewer is not None and ontop[0] else None
                    capture = changed | changed_follower
                    draw(task, virtual, sizes, radius, phases[:, 0] != 0, returning, leader, markers)
                    actions = policy.box_only_actions(obs, metadata)
                else:
                    leader_xy = task._humanoid_root_states[ids, leader, :2]
                    updated, distance, radius = update_switch(
                        leader_xy, virtual[:, :2], sizes, opts.switch_margin, switched,
                        fixed_radius=fixed_radius)
                    changed = updated & ~switched & ~ended
                    switch_step[changed] = step
                    switched |= updated & ~ended
                    for row in changed.nonzero().flatten().tolist():
                        print(f'[waypoint] episode={episode} env={row} switch step={step} '
                              f'distance={distance[row]:.3f} radius={radius[row]:.3f}', flush=True)
                    payload = task._agent_box_assignment[ids, leader]
                    holding = task.leader_holding(leader) & switched & ~ended
                    payload_xy = task._box_states[ids, payload, :2]
                    new_picked, new_returned, exit_distance, _ = update_return(
                        holding, payload_xy, virtual[:, :2], sizes, opts.switch_margin, picked, returned,
                        fixed_radius=fixed_radius)
                    pickup = new_picked & ~picked & ~ended
                    restored = new_returned & ~returned & ~ended
                    pickup_step[pickup] = step
                    return_step[restored] = step
                    picked |= new_picked & ~ended
                    returned |= new_returned & ~ended
                    for event, mask in (('pickup_target', pickup), ('restore_target', restored)):
                        for row in mask.nonzero().flatten().tolist():
                            print(f'[waypoint] episode={episode} env={row} {event} step={step} '
                                  f'payload_distance={exit_distance[row]:.3f} radius={radius[row]:.3f}', flush=True)
                    returning = picked & ~returned & ~ended
                    follower_view = {}
                    markers = None
                    capture = changed | pickup | restored
                    if opts.follower_ontop_waypoint:
                        follower_xy = task._humanoid_root_states[ids, follower, :2]
                        reached = (follower_xy - virtual[:, :2]).norm(dim=-1) <= radius
                        changed_follower = ontop & reached & ~follower_switched & ~ended
                        follower_switch_step[changed_follower] = step
                        follower_switched |= changed_follower
                        eligible = ontop & follower_switched & ~ended
                        follower_payload = task._agent_box_assignment[ids, follower]
                        follower_holding = task.leader_holding(follower) & eligible
                        new_picked, new_returned, follower_distance, _ = update_return(
                            follower_holding, task._box_states[ids, follower_payload, :2],
                            virtual[:, :2], sizes, opts.switch_margin, follower_picked, follower_returned,
                            fixed_radius=fixed_radius)
                        follower_pickup = new_picked & ~follower_picked & eligible
                        follower_restored = new_returned & ~follower_returned & eligible
                        follower_picked |= follower_pickup
                        follower_returned |= follower_restored
                        follower_pickup_step[follower_pickup] = step
                        follower_return_step[follower_restored] = step
                        for event, mask in (('follower_pickup_target', follower_pickup),
                                            ('follower_restore_target', follower_restored)):
                            for row in mask.nonzero().flatten().tolist():
                                print(f'[waypoint] episode={episode} env={row} {event} step={step} '
                                      f'payload_distance={follower_distance[row]:.3f} radius={radius[row]:.3f}',
                                      flush=True)
                        follower_target = follower_picked & ~follower_returned & eligible
                        follower_active = ontop & (~follower_switched | follower_target) & ~ended
                        follower_view = dict(follower=follower, follower_active=follower_active,
                                             follower_target=follower_target)
                        capture |= changed_follower | follower_pickup | follower_restored
                        if task.viewer is not None and ontop[0]:
                            markers = (follower, follower_active, follower_target)
                    draw(task, virtual, sizes, radius, switched, returning, leader, markers)
                    actions = policy.actions(obs, leader, virtual-task._env_origins,
                        (~switched | returning) & ~ended, target=returning, **follower_view)
                actions.view(task.num_envs, 4, -1)[ended] = 0
                obs, _, done, info = env.step(actions)
                if encoder is not None:
                    task.gym.write_viewer_image_to_file(task.viewer, str(frame))
                    encoder.stdin.write(frame.read_bytes())
                if carry is not None:
                    alive = ~ended[:, None] & ~info['terminate'].view(-1, 4).bool()
                    carry.record(task, alive, step + 1)
                if opts.box_only or opts.follower_ontop_waypoint:
                    for condition, event in (
                            (task._warehouse_entered[ids, follower], follower_enter_step),
                            (task.leader_holding(follower), follower_hold_step),
                            (task.relation_runtime.done[ids, follower], follower_ontop_step)):
                        event[(event < 0) & condition & ontop & ~ended] = step + 1
                if opts.capture and task.viewer is not None and (step == 0 or capture[0]):
                    task.gym.write_viewer_image_to_file(task.viewer,
                        str(output/f'episode_{episode:03d}_step_{step:04d}.png'))
                if not torch.isfinite(obs).all() or not torch.isfinite(actions).all():
                    raise RuntimeError('Nonfinite observation or action')
                if not torch.equal(task._tar_pos, final_goals):
                    raise RuntimeError('Virtual box changed an actual task goal')
                ending = (done.view(-1, 4).any(-1) | (step+1 == opts.steps)) & ~ended
                for row in ending.nonzero().flatten().tolist():
                    record = dict(episode=episode, env=row, steps=step+1, leader=int(leader[row]),
                        follower=int(follower[row]), follower_task=int(task._templates[row, follower[row]]),
                        switch_step=int(switch_step[row]), radius=float(radius[row]),
                        pickup_step=int(pickup_step[row]), return_step=int(return_step[row]),
                        leader_at=bool(task.relation_runtime.done[row, leader[row]]),
                        terminated=info['terminate'].view(-1, 4)[row].bool().cpu().tolist())
                    if opts.box_only or opts.follower_ontop_waypoint:
                        record.update(follower_switch_step=int(follower_switch_step[row]),
                            follower_enter_step=int(follower_enter_step[row]),
                            follower_hold_step=int(follower_hold_step[row]),
                            follower_pickup_step=int(follower_pickup_step[row]),
                            follower_return_step=int(follower_return_step[row]),
                            follower_ontop_step=int(follower_ontop_step[row]))
                    if carry is not None:
                        crossings = carry.crossing_step[row].tolist()
                        exits = [max(points) if min(points) >= 0 else -1 for points in crossings]
                        record.update(leader_hold_step=int(carry.pickup_step[row, 0]),
                                      leader_exit_step=exits[0], follower_exit_step=exits[1])
                    records.append(record)
                    print('[waypoint] '+json.dumps(record), flush=True)
                ended |= ending
                if ended.all():
                    break
                if task.viewer is not None and encoder is None:
                    time.sleep(max(0., task.dt-(time.monotonic()-started)))
    finally:
        (output/'results.json').write_text(json.dumps(dict(options=vars(opts),
            variant='push_before', start='closed', episodes=records), indent=2)+'\n')
        if task.viewer is not None:
            task.gym.destroy_viewer(task.viewer)
        task.gym.destroy_sim(task.sim)
        if encoder is not None:
            encoder.stdin.close()
            status = encoder.wait()
            frame.unlink(missing_ok=True)
            if status:
                raise RuntimeError(f'ffmpeg exited with status {status}')


if __name__ == '__main__':
    main()
