import json
import os
from pathlib import Path

import isaacgym
import torch

import run as launch
from env.tasks.multi_agent.humanoid_ma_mixed_sensor import HumanoidMAMixedSensor
from env.tasks.multi_agent.mixed_sensor_geometry import lowest_foot_height
from env.tasks.multi_agent.vec_task_wrapper_ma import VecTaskPythonWrapperMA
from utils.config import get_args, load_cfg, parse_sim_params, set_seed


def main():
    args = get_args()
    cfg, train, _ = load_cfg(args)
    cfg['env']['motion_file'] = args.motion_file
    seed = set_seed(47 if args.seed is None or args.seed < 0 else args.seed)
    player = None
    if args.checkpoint:
        launch.args, launch.cfg, launch.cfg_train = args, cfg, train
        train['params']['seed'] = train['params']['config']['seed'] = seed
        runner = launch.build_alg_runner(launch.RLGPUAlgoObserver())
        runner.load(train); runner.reset()
        player = runner.create_player()
        player.restore(args.checkpoint)
        env, task = player.env, player.env.task
    else:
        cfg['env']['enableTrackInitState'] = True
        task = HumanoidMAMixedSensor(cfg, parse_sim_params(args, cfg, train), args.physics_engine,
                                     args.device, args.device_id, args.headless)
        env = VecTaskPythonWrapperMA(task, args.rl_device)
    counts = torch.zeros(3, 3, dtype=torch.long, device=task.device)
    bad_counts = torch.zeros_like(counts)
    previous = task._scenario_ids.clone()
    actions = torch.zeros(task.num_envs * 2, task.num_dof, device=task.device)
    repeats = int(os.environ.get('RESET_REPEATS', 64))
    max_vz = max_rise = 0.
    min_clearance = float('inf')
    failures = []
    try:
        with torch.no_grad():
            for repeat in range(repeats):
                if player:
                    obs = player.env_reset()
                    player.get_batch_size(obs['obs'], 1)
                else:
                    env.reset()
                scene = task._scenario_ids.clone()
                pair = previous * 3 + scene
                counts += torch.bincount(pair, minlength=9).view(3, 3)
                initial = task._humanoid_root_states.clone()
                foot_z = lowest_foot_height(task._kinematic_humanoid_rigid_body_states,
                                           task._foot_body_ids, task._foot_vertices)
                clearance = foot_z - task._floor_height[:, None] - task._env_origins[:, None, 2]
                min_clearance = min(min_clearance, float(clearance.min()))
                hit = torch.zeros(task.num_envs, dtype=torch.bool, device=task.device)
                for step in range(6):
                    if player:
                        obs = player.env_reset([])
                        actions = player.get_action(obs, player.is_determenistic)
                        player.env_step(env, actions)
                    else:
                        env.step(actions)
                    root = task._humanoid_root_states
                    rise = root[..., 2] - initial[..., 2]
                    vz = root[..., 9]
                    max_vz, max_rise = max(max_vz, float(vz.max())), max(max_rise, float(rise.max()))
                    bad = ((vz > 3.) | (rise > .4)).any(-1)
                    for e in (bad & ~hit).nonzero().flatten()[:max(0, 5 - len(failures))].tolist():
                        failures.append(dict(repeat=repeat, step=step + 1, env=e, scenario=int(scene[e]),
                            root=root[e].tolist(), initial=initial[e].tolist()))
                    hit |= bad
                bad_counts += torch.bincount(pair[hit], minlength=9).view(3, 3)
                previous = scene
        report = dict(mode='checkpoint' if player else 'hold_rsi_pose', checkpoint=args.checkpoint,
            physics=task.physics_metadata, seed=seed, environments=task.num_envs, resets=repeats,
            seconds_after_reset=6 * task.dt, transitions=counts.tolist(), launch_counts=bad_counts.tolist(),
            max_upward_speed_mps=max_vz, max_root_rise_m=max_rise, min_foot_clearance_m=min_clearance,
            failures=failures)
        out = Path(args.output_path)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'reset_physics.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        assert min_clearance >= .004, report
        assert not bad_counts.any(), report
    finally:
        task.gym.destroy_sim(task.sim)


if __name__ == '__main__':
    main()
