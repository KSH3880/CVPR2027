"""Frozen single-agent evaluation; no training or changes to existing source files."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CORE = REPO/'ports/jhh_stage1_push_raw_cuboid'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--task', choices=('at', 'ontop', 'sit', 'climb', 'push'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--policy', default='policy')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--num-envs', type=int, default=512)
    p.add_argument('--steps', type=int, default=900)
    p.add_argument('--start', choices=('loco', 'rsi'), default='loco')
    p.add_argument('--distance-min', type=float, default=2.)
    p.add_argument('--distance-max', type=float, default=5.)
    p.add_argument('--viewer', action='store_true')
    p.add_argument('--smoke', action='store_true', help='Mark a short run; never a benchmark result')
    opt = p.parse_args()
    # Resolve caller paths before entering the independent code/data port.
    opt.checkpoint = opt.checkpoint.resolve()
    opt.output = opt.output.resolve()
    if not opt.checkpoint.is_file():
        p.error('Checkpoint does not exist')
    if opt.output.exists():
        p.error('Output already exists; select a new directory')
    if CORE/'output_etc/check' not in opt.output.parents:
        p.error('Evaluation output must be below '+str(CORE/'output_etc/check'))
    if opt.num_envs < 1 or opt.steps < 3:
        p.error('Positive environment count and at least 3 steps required')
    if not opt.smoke and (opt.num_envs != 512 or opt.steps != 900):
        p.error('Protocol runs use 512 environments/900 steps; use --smoke for short checks')
    if opt.start == 'rsi' and opt.task != 'push':
        p.error('The four-task protocol uses loco starts only')
    if not 0 < opt.distance_min < opt.distance_max:
        p.error('Invalid distance range')
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        p.error('Use run.sh with TOKENHSI_GPU explicitly selected')
    sys.path[:0] = [str(CORE/'tokenhsi'), str(CORE), str(HERE)]
    os.chdir(CORE)
    import isaacgym  # Must precede torch.
    import torch
    import numpy as np
    import yaml
    from metrics import PROTOCOL, PUSH_PROTOCOL, score_trace
    from backend import task_class, build_frozen_actor
    from utils.config import get_args, load_cfg, parse_sim_params, set_seed
    import utils.parse_task as parser

    weights = torch.load(opt.checkpoint, map_location='cpu', weights_only=False)
    saved = weights['relation_experiment_config']
    is_push = 'stage1_push_contract' in weights
    if opt.task == 'push' and not is_push:
        p.error('PUSH requires a model with a PUSH token; the 73000 model has four task tokens')
    if saved['env']['relationReward'].get('stage1_variant') != 'stage1_v2':
        p.error('This evaluator supports the imported Stage1V2Fixed architecture only')
    env = deepcopy(saved['env'])
    push_spec = deepcopy(env.pop('stage1Push', None))
    if is_push:
        from utils.stage1_push import check_contract
        check_contract(weights, saved['env'])
    env.update(numEnvs=opt.num_envs, numAgents=1, numObjects=2 if opt.task == 'ontop' else 1,
               episodeLength=opt.steps, mode='test', enableDebugVis=False)
    env['eval'].update(skill=['loco'], skillInitProb=[1.])
    env['video']['enable'] = False
    env['debug']['reward'] = False
    env['relationReward']['diagnostics']['enabled'] = False
    env['relationReward']['success']['terminate_when_all_subgoals_done'] = False
    env['envSpacing'] = max(float(env['envSpacing']), 12. if opt.task == 'push' else 10.)
    sim = deepcopy(saved['sim'])
    native_train = 'stage1_v2_fixed_push_360_raw_cuboid' if is_push else 'stage1_v2_fixed'
    train = yaml.safe_load((CORE/f'tokenhsi/data/cfg/train/rlg/{native_train}.yaml').read_text())
    # The checkpoint stores the complete resolved train config for reproducibility.
    saved_train = saved.get('train')
    if isinstance(saved_train, dict) and 'params' in saved_train:
        train = deepcopy(saved_train)
    opt.output.mkdir(parents=True)
    snapshot = opt.output/'resolved_env.yaml'
    snapshot.write_text(yaml.safe_dump(dict(env=env, sim=sim), sort_keys=False))
    trainfile = opt.output/'resolved_train.yaml'
    trainfile.write_text(yaml.safe_dump(train, sort_keys=False))
    preset = dict(at='holding_at', ontop='holding_ontop', sit='sit', climb='climb', push='holding_at')[opt.task]
    sys.argv = [sys.argv[0], '--task', 'HumanoidMACarry', '--cfg_env', str(snapshot),
        '--cfg_train', str(trainfile), '--checkpoint', str(opt.checkpoint), '--test', '--eval',
        '--no_video', '--seed', str(opt.seed), '--task_graph', preset, '--task_camera', 'stack',
        '--output_path', str(opt.output), '--sim_device', 'cuda:0', '--rl_device', 'cuda:0', '--pipeline', 'gpu']
    if not opt.viewer:
        sys.argv += ['--headless']
    args = get_args()
    cfg, train, _ = load_cfg(args)
    set_seed(opt.seed)
    train['seed'] = opt.seed
    parser.HumanoidMACarry = task_class(opt, 4 if is_push else 3, push_spec)
    task, wrapper = parser.parse_task(args, cfg, train, parse_sim_params(args, cfg, train))
    try:
        actor = build_frozen_actor(task, weights, train, is_push)
        # Actor construction consumes RNG. Reset it before the common scene draw.
        set_seed(opt.seed)
        with torch.no_grad():
            obs = wrapper.reset()
            def cpu(t):
                return t.detach().cpu().numpy().copy()
            initial = dict(root=cpu(task._humanoid_root_states[:, 0]), boxes=cpu(task._box_states),
                dof=cpu(task._dof_state.view(opt.num_envs, 1, task.num_dof, 2)[:, 0]),
                sizes=cpu(task._box_size), goal=cpu(task._tar_pos[:, 0]), packet=cpu(task.obs_buf[:, -5:]),
                origins=cpu(task._env_origins), mass=cpu(task._box_density.reshape(opt.num_envs, -1)*task._box_size.prod(-1)))
            physics = np.empty((opt.num_envs, task.num_objects, 3), dtype=np.float64)
            for env_id, env_ptr in enumerate(task.envs):
                for obj in range(task.num_objects):
                    handle = task._box_handles[env_id*task.num_objects+obj]
                    bodies = task.gym.get_actor_rigid_body_properties(env_ptr, handle)
                    shapes = task.gym.get_actor_rigid_shape_properties(env_ptr, handle)
                    physics[env_id, obj] = [sum(b.mass for b in bodies), min(s.friction for s in shapes), max(s.friction for s in shapes)]
            initial['actual_physics_mass_friction_min_max'] = physics
            if opt.task == 'push':
                initial['zone_yaw'] = cpu(task._single_zone_yaw)
                if not np.allclose(physics[..., 0], initial['mass'], rtol=1e-5, atol=1e-5):
                    raise RuntimeError('Actual PUSH mass differs from requested mass')
                if not np.allclose(physics[..., 1:], push_spec['box']['friction'], rtol=0, atol=1e-6):
                    raise RuntimeError('Actual PUSH shape friction differs from config')
            np.savez_compressed(opt.output/'initial_0.npz', **initial)
            alive = np.ones(opt.num_envs, dtype=bool)
            trace = {k: [] for k in ('root', 'boxes', 'goal', 'active', 'done', 'terminate')}
            fixed_pose = initial['boxes'][..., :7]
            start = time.monotonic()
            for step in range(opt.steps):
                if not torch.isfinite(obs).all():
                    raise RuntimeError('Non-finite policy observation')
                action = actor(obs)
                obs, reward, done, info = wrapper.step(action)
                if opt.viewer:
                    task.render()
                for key, value in (('root', task._humanoid_root_states[:, 0, :3]),
                                   ('boxes', task._box_states), ('goal', task._tar_pos[:, 0]),
                                   ('done', done.bool()), ('terminate', info['terminate'].bool())):
                    trace[key].append(cpu(value))
                trace['active'].append(alive.copy())
                if opt.task in ('sit', 'climb') and not np.allclose(trace['boxes'][-1][..., :7], fixed_pose, rtol=0, atol=1e-6):
                    raise RuntimeError('Fixed SIT/CLIMB object pose changed')
                alive &= ~trace['done'][-1]
                if not alive.any():
                    break
                # Never reset finished environments. Their later frames are masked.
            if alive.any():
                raise RuntimeError('First episode exceeded configured horizon')
            trace = {k: np.stack(v) for k, v in trace.items()}
            trace['sizes'] = initial['sizes']
            if opt.task == 'push':
                trace['initial_box'] = initial['boxes'][:, 0, :3]
                trace['zone_yaw'] = initial['zone_yaw']
            np.savez_compressed(opt.output/'trace_0.npz', **trace)
            scores = score_trace(opt.task, trace, task.dt)
            manifest = dict(protocol=PUSH_PROTOCOL if opt.task == 'push' else PROTOCOL,
                policy=opt.policy, task=opt.task, seed=opt.seed, start=opt.start,
                distance_range=[opt.distance_min, opt.distance_max] if opt.task == 'push' else [],
                checkpoint=str(opt.checkpoint), checkpoint_sha256=sha(opt.checkpoint), epoch=int(weights['epoch']),
                core=str(CORE), num_envs=opt.num_envs, steps=opt.steps, recorded_steps=len(trace['active']),
                control_dt=task.dt, smoke=opt.smoke, deterministic_actor=True, policy_weights_changed=False,
                model_entity_counts=[1, task.num_objects], amp_families=4 if is_push else 3,
                source_file_sha256={str(f.relative_to(HERE)): sha(f) for f in HERE.glob('*.py')},
                push_box=push_spec['box'] if opt.task == 'push' else None,
                goal_zone_size_m=[1., 1.] if opt.task == 'push' else None,
                push_initialization='goal sampled from actual initial box; no RSI progress displacement' if opt.task == 'push' else None,
                elapsed_seconds=time.monotonic()-start)
            (opt.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
            result = dict(manifest, scores=scores)
            (opt.output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
            completed = dict(exit_code=0, sampling_failures=int(task._sampling_failures),
                float32_float64_ever_match=True, first_episode_only=True,
                push_mass_friction_checked=opt.task == 'push',
                fixed_object_pose_checked=opt.task in ('sit', 'climb'),
                artifact_sha256={f: sha(opt.output/f) for f in ('initial_0.npz', 'trace_0.npz', 'resolved_env.yaml', 'resolved_train.yaml', 'results.json')})
            (opt.output/'completed.json').write_text(json.dumps(completed, indent=2)+'\n')
            print(json.dumps(dict(task=opt.task, policy=opt.policy, scores=scores), indent=2))
    finally:
        if task.viewer is not None:
            task.gym.destroy_viewer(task.viewer)
        task.gym.destroy_sim(task.sim)


if __name__ == '__main__':
    main()
