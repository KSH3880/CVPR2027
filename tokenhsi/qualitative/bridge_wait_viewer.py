import argparse
from functools import partial
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

import run
from env.tasks.multi_agent import humanoid_ma_bridge_follow_cap as training_task
from env.tasks.multi_agent import square_sensor_scene as scene
from learning.multi_agent.ma_players import MAPlayerContinuous
from qualitative.bridge_wait_layout import BridgeWaitTask


def parse():
    parser = argparse.ArgumentParser(description='Trained follow-cap viewer; face across the bridge: lateral -1=left, +1=right')
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--leader-lateral', type=float, default=-1.)
    parser.add_argument('--leader-depth', type=float, default=2.2, help='Leader distance inward from the source platform cliff edge, meters')
    parser.add_argument('--pickup-distance', type=float, default=1.1, help='Payload distance straight ahead of leader, meters')
    parser.add_argument('--goal-lateral', type=float, default=0.)
    parser.add_argument('--goal-depth', type=float, default=2.)
    parser.add_argument('--bridge-entry-side', choices=('left', 'right'), default='right')
    parser.add_argument('--bridge-distance', type=float, help='Bridge center distance to final center along insertion channel, meters; default native insert start')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--envs', type=int, default=1)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--steps', type=int, default=900)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--capture', action='store_true')
    parser.add_argument('--out', default='output_etc/check/bridge_wait_viewer/'+time.strftime('%Y%m%d_%H%M%S'))
    opts = parser.parse_args()
    for key in ('leader_lateral', 'goal_lateral'):
        value = getattr(opts, key)
        if not math.isfinite(value) or not -1 <= value <= 1:
            parser.error(key+' must be in [-1, 1]')
    for key in ('leader_depth', 'pickup_distance', 'goal_depth', 'bridge_distance'):
        value = getattr(opts, key)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(key+' must be positive and finite')
    if min(opts.envs, opts.episodes, opts.steps) <= 0 or opts.steps > 900:
        parser.error('envs/episodes/steps must be positive; steps <= 900')
    return opts


def camera(task):
    if task.viewer is None:
        return
    origin = task.gym.get_env_origin(task.envs[0])
    landing = task._bridge_landing_world[0].cpu().numpy()-np.array([origin.x, origin.y, origin.z])
    yaw = float(task._structure_yaw[0, 1])
    direction = np.array([math.cos(yaw), math.sin(yaw), 0.])
    target = landing.copy()
    target[2] += float(task._surface_height[0, 1])
    eye = target-13*direction+np.array([0., 0., 11.])
    task.gym.viewer_camera_look_at(task.viewer, task.envs[0], gymapi.Vec3(*eye), gymapi.Vec3(*target))


def initial_layout(task):
    ids = torch.arange(task.num_envs, device=task.device)
    leader = task._roles[:, 1]
    payload = task._agent_box_assignment[ids, leader]
    poses = torch.stack((task._humanoid_root_states[ids, leader, :2], task._box_states[ids, payload, :2],
        task._tar_pos[ids, leader, :2], task._box_states[ids, task._bridge_slot, :2]), 1)
    local = scene.rotate_xy(poses-task._env_origins[:, None, :2]-task._structure_shift[:, 1, None],
        -task._structure_yaw[:, 1, None])
    return dict(leader_payload_goal_bridge_local_xy=local.cpu().tolist(),
        platform_centers=task._surface_center[:, 1].cpu().tolist(),
        platform_half_sizes=task._surface_half[:, 1].cpu().tolist(),
        bridge_landing=task._bridge_landing.cpu().tolist(),
        roles=task._roles.cpu().tolist(), source_surface=task._start_surface[ids, leader].cpu().tolist(),
        same_platform=task._before_same_platform.cpu().tolist(),
        bridge_side=task._bridge_side.cpu().tolist(),
        cap_lift=task.cap_lift, waypoint=task._waypoint_spec)


@torch.no_grad()
def view(player, opts):
    task = player.env.task
    output = Path(task.cfg['args'].output_path)
    output.mkdir(parents=True, exist_ok=True)
    player._reward_debug = False
    player._post_step = lambda info: None
    ids = torch.arange(task.num_envs, device=task.device)
    records = []
    try:
        for episode in range(opts.episodes):
            obs = player.env_reset()
            player.get_batch_size(obs['obs'], 1)
            camera(task)
            initial = initial_layout(task)
            (output/f'initial_{episode:03d}.json').write_text(json.dumps(initial, indent=2)+'\n')
            torch.save({key: value.cpu().clone() for key, value in task._layout_tensors().items()}, output/f'initial_{episode:03d}.pt')
            print('[bridge wait] '+json.dumps(initial), flush=True)
            ended = torch.zeros(task.num_envs, dtype=torch.bool, device=task.device)
            final_goals = task._tar_pos.clone()
            leader = task._roles[:, 1]
            for step in range(opts.steps):
                started = time.monotonic()
                if task.viewer is not None and task.gym.query_viewer_has_closed(task.viewer):
                    return
                action = player.get_action(obs, True).reshape(task.num_envs, task.num_agents, -1)
                action[ended] = 0
                obs, _, done, info = player.env_step(player.env, action.flatten(0, 1))
                if not isinstance(obs, dict):
                    obs = {'obs': obs}
                if not torch.isfinite(obs['obs']).all() or not torch.isfinite(action).all():
                    raise RuntimeError('Nonfinite policy input or action')
                if not torch.equal(task._tar_pos, final_goals):
                    raise RuntimeError('Final goals changed during episode')
                if opts.capture and task.viewer is not None and step in (0, 100, 200, 400, 800):
                    task.gym.write_viewer_image_to_file(task.viewer, str(output/f'episode_{episode:03d}_step_{step:04d}.png'))
                finish = (done.reshape(task.num_envs, task.num_agents).bool().any(-1) | (step+1 == opts.steps)) & ~ended
                for row in finish.nonzero().flatten().tolist():
                    record = dict(episode=episode, env=row, steps=step+1,
                        crossed=bool(task._cross_completed[row, leader[row]]),
                        leader_goal=bool(task.relation_runtime.done[row, leader[row]]),
                        falls=info['terminate'].reshape(task.num_envs, task.num_agents)[row].cpu().tolist())
                    records.append(record)
                    print('[bridge wait result] '+json.dumps(record), flush=True)
                ended |= finish
                if ended.all():
                    break
                if task.viewer is not None:
                    time.sleep(max(0., task.dt-(time.monotonic()-started)))
    finally:
        (output/'results.json').write_text(json.dumps(dict(options=vars(opts), episodes=records), default=str, indent=2)+'\n')
        if task.viewer is not None:
            task.gym.destroy_viewer(task.viewer)
        task.gym.destroy_sim(task.sim)


def main():
    opts = parse()
    checkpoint = opts.checkpoint.resolve(strict=True)
    saved = yaml.safe_load((checkpoint.parent.parent/'relation_config.yaml').read_text())
    name = saved['experiment']['name']
    if not name.startswith('stage2_sensor_900_bridge_follow_cap'):
        raise ValueError('Use a trained follow-cap checkpoint, not the original DistillV4 checkpoint')
    os.environ['BRIDGE_START'] = 'insert'
    os.environ['BRIDGE_MODE'] = 'cooperative'
    os.environ['SCENARIO'] = 'bridge'
    torch.set_num_threads(4)
    training_task.HumanoidMABridgeFollowCap = partial(BridgeWaitTask, positions=opts)
    MAPlayerContinuous.run_eval = lambda player: view(player, opts)
    original_load = run.load_cfg
    def load(args):
        cfg, train, log = original_load(args)
        cfg['env']['cameraFollow'] = False
        return cfg, train, log
    run.load_cfg = load
    sys.argv = [sys.argv[0], '--task', 'HumanoidMACoupledPush', '--test', '--eval',
        '--checkpoint', str(checkpoint), '--cfg_train', f'tokenhsi/data/cfg/train/rlg/{name}.yaml',
        '--cfg_env', '', '--num_envs', str(opts.envs), '--seed', str(opts.seed),
        '--task_graph', 'random_scenario', '--motion_file', 'tokenhsi/data/dataset_stage1_jointpush.yaml',
        '--no_video', '--output_path', opts.out]
    if opts.headless:
        sys.argv.append('--headless')
    run.main()


if __name__ == '__main__':
    main()
