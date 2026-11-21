import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import isaacgym
import torch
import yaml

from qualitative.bridge_waypoint_policy import WaypointPolicy
from qualitative.bridge_waypoint_viewer import camera
from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from utils.parse_task import parse_task


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='Original V4 policy in the rear90 training layout')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--envs', type=int, default=1)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--steps', type=int, default=900)
    parser.add_argument('--bridge-start', choices=('mixed', 'present', 'near', 'insert'), default='mixed')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--out', default='output_etc/check/bridge_rear90_zeroshot/'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid()))
    opts = parser.parse_args()
    if min(opts.envs, opts.episodes, opts.steps) < 1 or opts.steps > 900:
        parser.error('Positive envs/episodes/steps required; steps <= 900')
    name = 'stage2_sensor_900_bridge_follow_cap_rear90'
    for key in ('BRIDGE_START', 'BRIDGE_MODE', 'BRIDGE_PUSHERS', 'SCENARIO', 'BRIDGE_CURRICULUM_EVAL'):
        os.environ.pop(key, None)
    sys.argv = [sys.argv[0], '--task', 'HumanoidMACoupledPush', '--test',
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
    if opts.bridge_start != 'mixed':
        cfg['env']['sensorScene']['bridge']['start_distribution']['probabilities'] = [
            float(opts.bridge_start == mode) for mode in ('present', 'near', 'insert')]
    torch.set_num_threads(4)
    set_seed(opts.seed)
    task, env = parse_task(args, cfg, train, parse_sim_params(args, cfg, train))
    source = torch.load(opts.checkpoint, map_location='cpu', weights_only=False)
    policy = WaypointPolicy(task, source)
    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)
    (out/'resolved_env.yaml').write_text(yaml.safe_dump({key: cfg[key] for key in ('env', 'sim', 'experiment')}, sort_keys=False))
    manifest = dict(checkpoint=str(Path(opts.checkpoint).resolve()), source_epoch=source['epoch'],
        config=name, policy='original_v4_deterministic_mean', training_updates=0,
        training_layout_cache=True, options=vars(opts), episodes=[])
    try:
        for episode in range(opts.episodes):
            obs = env.reset()
            camera(task)
            modes = [('present', 'near', 'insert')[int(mode)] for mode in task._bridge_start_mode]
            print(f'[rear90 zero-shot] episode={episode} env0={modes[0]} same_platform={bool(task._before_same_platform[0])}', flush=True)
            torch.save({key: value.cpu().clone() for key, value in task._layout_tensors().items()}, out/f'initial_{episode:03d}.pt')
            ended = torch.zeros(task.num_envs, device=task.device, dtype=torch.bool)
            inactive = torch.zeros_like(ended)
            for step in range(opts.steps):
                started = time.monotonic()
                if task.viewer is not None and task.gym.query_viewer_has_closed(task.viewer):
                    return
                actions = policy.actions(obs, task._roles[:, 1], None, inactive)
                actions.view(task.num_envs, 4, -1)[ended] = 0
                obs, _, done, _ = env.step(actions)
                if not torch.isfinite(obs).all() or not torch.isfinite(actions).all():
                    raise RuntimeError('Nonfinite observation/action')
                ended |= done.view(-1, 4).bool().any(-1)
                if ended.all():
                    break
                if task.viewer is not None:
                    time.sleep(max(0., task.dt-(time.monotonic()-started)))
            manifest['episodes'].append(dict(episode=episode, steps=step+1, modes=modes,
                same_platform=task._before_same_platform.cpu().tolist()))
    finally:
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        if task.viewer is not None:
            task.gym.destroy_viewer(task.viewer)
        task.gym.destroy_sim(task.sim)


if __name__ == '__main__':
    main()
