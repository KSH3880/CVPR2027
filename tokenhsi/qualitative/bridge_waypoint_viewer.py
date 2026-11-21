import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import isaacgym
from isaacgym import gymapi
import numpy as np
import torch
import yaml

from qualitative.bridge_waypoint import bridge_goal, switch_margin, update_switch
from qualitative.bridge_waypoint_policy import WaypointPolicy
from qualitative.bridge_waypoint_task import BridgeWaypointTask
from qualitative.bridge_waypoint_physics import MODES, BridgeWaypointPhysicsTask, physics_info
from env.tasks.multi_agent.vec_task_wrapper_ma import VecTaskPythonWrapperMA
from utils.config import get_args, load_cfg, parse_sim_params, set_seed


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--switch-margin', type=switch_margin, default=1.)
    parser.add_argument('--no-waypoint', action='store_true',
        help='Use the original goal from reset, without the bridge waypoint or radius switch')
    parser.add_argument('--bridge-physics', choices=MODES, default='baseline')
    parser.add_argument('--bridge-start', choices=('present', 'near', 'insert'), default='present')
    parser.add_argument('--cap-lift', type=float, default=0.)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--envs', type=int, default=1)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--steps', type=int, default=900)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--capture', action='store_true')
    parser.add_argument('--out')
    opts = parser.parse_args()
    if min(opts.envs, opts.episodes, opts.steps) < 1:
        parser.error('envs, episodes and steps must be positive')
    if not math.isfinite(opts.cap_lift) or opts.cap_lift < 0:
        parser.error('cap lift must be finite and nonnegative, in meters')
    if opts.cap_lift and opts.bridge_physics != 'follow_cap':
        parser.error('--cap-lift requires --bridge-physics follow_cap')
    opts.out = opts.out or ('output_etc/check/bridge_waypoint/' + time.strftime('%Y%m%d_%H%M%S'))
    return opts


def environment(opts):
    name = 'stage2_sensor_900_bridge_before_fixed_layout_cache'
    sys.argv = [sys.argv[0], '--task', 'HumanoidMACoupledPush', '--test', '--eval',
        '--checkpoint', opts.checkpoint, '--cfg_train', f'tokenhsi/data/cfg/train/rlg/{name}.yaml',
        '--cfg_env', f'tokenhsi/data/cfg/multi_agent/{name}.yaml', '--num_envs', str(opts.envs),
        '--num_agents', '4', '--num_objects', '8', '--seed', str(opts.seed),
        '--motion_file', 'tokenhsi/data/dataset_stage1_jointpush.yaml', '--no_video', '--output_path', opts.out]
    if opts.headless:
        sys.argv.append('--headless')
    args = get_args()
    cfg, train, _ = load_cfg(args)
    cfg['env']['motion_file'] = args.motion_file
    cfg['env']['cameraFollow'] = False
    cfg['env']['sensorScene']['bridge']['before_training']['same_platform_probability'] = 0.
    start = getattr(opts, 'bridge_start', 'present')
    cfg['env']['sensorScene']['bridge']['start_distribution']['probabilities'] = [float(start == name) for name in ('present', 'near', 'insert')]
    os.environ['BRIDGE_START'] = start
    set_seed(opts.seed)
    mode = getattr(opts, 'bridge_physics', 'baseline')
    task_type = BridgeWaypointTask if mode == 'baseline' else BridgeWaypointPhysicsTask
    physics_args = {} if mode == 'baseline' else {'bridge_physics': mode, 'cap_lift': getattr(opts, 'cap_lift', 0.)}
    task = task_type(cfg=cfg, sim_params=parse_sim_params(args, cfg, train),
        physics_engine=args.physics_engine, device_type=args.device, device_id=args.device_id,
        headless=args.headless, **physics_args)
    env = VecTaskPythonWrapperMA(task, args.rl_device, float('inf'), 1.)
    output = Path(args.output_path)
    output.mkdir(parents=True, exist_ok=True)
    (output/'resolved_env.yaml').write_text(yaml.safe_dump(
        {key: cfg[key] for key in ('env', 'sim', 'experiment')}, sort_keys=False))
    return task, env, output


def draw(task, goal, radius, switched):
    if task.viewer is None:
        return
    gym = task.gym
    gym.clear_lines(task.viewer)
    origin = task.gym.get_env_origin(task.envs[0])
    offset = np.array([origin.x, origin.y, origin.z])
    points = []
    if goal is not None:
        center = goal[0].cpu().numpy()-offset
        angle = np.linspace(0, 2*np.pi, 97)
        circle = np.column_stack((center[0]+float(radius[0])*np.cos(angle),
            center[1]+float(radius[0])*np.sin(angle), np.full_like(angle, center[2])))
        lines = np.stack((circle[:-1], circle[1:]), 1)
        active_color = (.6, .6, .6) if bool(switched[0]) else (1., .15, 1.)
        gym.add_lines(task.viewer, task.envs[0], len(lines), lines.astype(np.float32),
                      np.tile(active_color, (len(lines), 1)).astype(np.float32))
        points.append((center, active_color))
    leader = task._roles[0, 1]
    final = task._tar_pos[0, leader].cpu().numpy()-offset
    points.append((final, (.1, 1., .1)))
    for point, color in points:
        delta = np.eye(3)*.3
        cross = np.stack((point-delta, point+delta), 1).astype(np.float32)
        gym.add_lines(task.viewer, task.envs[0], 3, cross, np.tile(color, (3, 1)).astype(np.float32))


def camera(task):
    if task.viewer is None:
        return
    origin = task.gym.get_env_origin(task.envs[0])
    center = task._bridge_landing_world[0].cpu().numpy()-np.array([origin.x, origin.y, origin.z])
    task.gym.viewer_camera_look_at(task.viewer, task.envs[0],
        gymapi.Vec3(center[0]+8., center[1]-10., center[2]+8.), gymapi.Vec3(*center))


@torch.no_grad()
def main():
    opts = parse()
    torch.set_num_threads(4)
    task, env, output = environment(opts)
    source = torch.load(opts.checkpoint, map_location='cpu', weights_only=False)
    policy = WaypointPolicy(task, source)
    records = []
    physics = []
    ids = torch.arange(task.num_envs, device=task.device)
    if opts.no_waypoint:
        print('[waypoint] disabled: original goal from reset; green: original goal', flush=True)
    else:
        print('[waypoint] magenta: bridge goal/radius; green: original goal; gray: switched', flush=True)
    try:
        for episode in range(opts.episodes):
            obs = env.reset()
            if episode == 0:
                physics = physics_info(task)
                print('[waypoint physics] '+json.dumps(physics), flush=True)
            camera(task)
            leader = task._roles[:, 1]
            payload = task._agent_box_assignment[ids, leader]
            sizes = task._box_size[ids, payload]
            switched = torch.zeros(task.num_envs, dtype=torch.bool, device=task.device)
            switch_step = torch.full_like(ids, -1)
            ended = torch.zeros_like(switched)
            final_goals = task._tar_pos.clone()
            for step in range(opts.steps):
                started = time.monotonic()
                goal, radius = None, None
                changed, active = torch.zeros_like(ended), torch.zeros_like(ended)
                if not opts.no_waypoint:
                    goal = bridge_goal(task._box_states[ids, task._bridge_slot],
                        task._box_size[ids, task._bridge_slot], sizes)
                    updated, distance, radius = update_switch(task._box_states[ids, payload, :2],
                        goal[:, :2], sizes, opts.switch_margin, switched)
                    changed = updated & ~switched & ~ended
                    switch_step[changed] = step
                    switched |= updated & ~ended
                    active = ~switched & ~ended
                    for row in changed.nonzero().flatten().tolist():
                        print(f'[waypoint] episode={episode} env={row} switch step={step} '
                              f'distance={distance[row]:.3f} radius={radius[row]:.3f}', flush=True)
                draw(task, goal, radius, switched)
                actions = policy.actions(obs, leader, None if goal is None else goal-task._env_origins, active)
                actions.view(task.num_envs, 4, -1)[ended] = 0
                obs, _, done, info = env.step(actions)
                if opts.capture and task.viewer is not None and (step == 0 or changed[0]):
                    task.gym.write_viewer_image_to_file(task.viewer,
                        str(output/f'episode_{episode:03d}_step_{step:04d}.png'))
                if not torch.isfinite(obs).all() or not torch.isfinite(actions).all():
                    raise RuntimeError('Nonfinite observation or action')
                if not torch.equal(task._tar_pos, final_goals):
                    raise RuntimeError('Virtual waypoint changed an actual task goal')
                ending = (done.view(-1, 4).any(-1) | (step+1 == opts.steps)) & ~ended
                for row in ending.nonzero().flatten().tolist():
                    record = dict(episode=episode, env=row, steps=step+1,
                        switch_step=int(switch_step[row]), radius=None if radius is None else float(radius[row]),
                        entered=bool(task._cross_entered[row, leader[row]]),
                        crossed=bool(task._cross_completed[row, leader[row]]),
                        leader_at=bool(task.relation_runtime.done[row, leader[row]]),
                        falls=info['terminate'].view(-1, 4)[row].bool().cpu().tolist())
                    records.append(record)
                    print('[waypoint] '+json.dumps(record), flush=True)
                ended |= ending
                if ended.all():
                    break
                if task.viewer is not None:
                    time.sleep(max(0., task.dt-(time.monotonic()-started)))
    finally:
        (output/'results.json').write_text(json.dumps(dict(options=vars(opts),
            source_epoch=source['epoch'], physics=physics, episodes=records), indent=2)+'\n')
        if task.viewer is not None:
            task.gym.destroy_viewer(task.viewer)
        task.gym.destroy_sim(task.sim)


if __name__ == '__main__':
    main()
