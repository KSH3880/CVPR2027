"""First-episode, reward-independent single-skill evaluation and CPU replay audit."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from learning.cvpr2027_scoring import HORIZON, TASKS, spatial_metrics, counts, ever_from_trace, trial_minimum_distances, error_statistics


def metrics_frame(name, roots, boxes, sizes, goals, xy_fraction=.8, scoring_mode='protocol_region'):
    if scoring_mode == 'native_point':
        if name not in ('sit', 'climb'):
            raise ValueError('Native point mode only supports SIT/CLIMB')
        distance = np.linalg.norm(np.asarray(roots[:, 0, :3], dtype=np.float64)-np.asarray(goals[:, 0], dtype=np.float64), axis=-1)
        distances = np.stack((distance, distance), -1)
        return distances <= .20, distances
    if scoring_mode != 'protocol_region':
        raise ValueError('Unknown scoring mode')
    task = TASKS.index(name.upper())
    support_index = 1 if name == 'on_top' else 0
    point = roots[:, 0, :3] if name in ('sit', 'climb') else boxes[:, 0, :3]
    return spatial_metrics(task, point, boxes[:, support_index], sizes[:, support_index],
                           goals[:, 0], sizes[:, 0, 2], xy_fraction)


def score_frame(name, roots, boxes, sizes, goals, xy_fraction=.8):
    return metrics_frame(name, roots, boxes, sizes, goals, xy_fraction)[0]


def cpu_rescore(directory):
    directory = Path(directory)
    report = json.loads((directory/'result.json').read_text())
    initial = np.load(directory/'initial.npz', allow_pickle=False)
    roots = np.load(directory/'roots.npy', mmap_mode='r')
    boxes = np.load(directory/'boxes.npy', mmap_mode='r')
    falls = np.load(directory/'fallen.npy', mmap_mode='r')
    frames = report['frames']
    metrics = [metrics_frame(report['task'], roots[i], boxes[i], initial['sizes'], initial['goals'], report.get('xy_fraction', .8), report.get('scoring_mode', 'protocol_region')) for i in range(frames)]
    success = np.stack([m[0] for m in metrics])
    distances = np.stack([m[1] for m in metrics])
    ever, valid = ever_from_trace(success, falls[:frames])
    stored_success = np.load(directory/'success.npy', mmap_mode='r')[:frames]
    stored_alive = np.load(directory/'alive.npy', mmap_mode='r')[:frames]
    if not np.array_equal(valid, stored_alive) or not np.array_equal(success & valid[..., None], stored_success):
        raise AssertionError('CPU trace scoring differs from recorded frame results')
    if not np.array_equal(ever, np.load(directory/'ever.npy')):
        raise AssertionError('CPU ever success differs from recorded results')
    result = dict(verified=True, frames=frames, region_success=int(ever[:, 1].sum()), point_success=int(ever[:, 0].sum()))
    if 'errors' in report:
        minimum = trial_minimum_distances(distances, valid)
        np.testing.assert_allclose(minimum, np.load(directory/'minimum_distances_m.npy'), rtol=0, atol=1e-12)
        np.testing.assert_allclose(distances, np.load(directory/'distances_m.npy', mmap_mode='r')[:frames], rtol=0, atol=1e-12)
        errors = error_statistics(minimum, ever[:, 1])
        if errors != report['errors']:
            raise AssertionError('CPU errors differ from online report')
        result['errors'] = errors
        result['xy_fraction'] = report['xy_fraction']
    (directory/'cpu_rescore.json').write_text(json.dumps(result, indent=2))
    return result


@torch.no_grad()
def run_single_eval(player):
    from isaacgym import gymapi
    ref = player.env.reference
    name = ref.cvpr_task
    scoring_mode = getattr(ref, 'scoring_mode', 'protocol_region')
    revision = os.environ.get('CVPR_PROTOCOL_REVISION', '1109')
    if revision not in ('1108', '1109'):
        raise ValueError('Unknown CVPR protocol revision')
    xy_fraction = 1. if revision == '1109' else .8
    out = Path(os.environ['CVPR_EVAL_OUT'])
    out.mkdir(parents=True, exist_ok=True)
    player.model.eval()
    if player.normalize_input:
        player.running_mean_std.eval()
    frozen_stats = {k: v.clone() for k, v in player.running_mean_std.state_dict().items()} if player.normalize_input else {}
    if hasattr(player, '_amp_input_mean_std'):
        player._amp_input_mean_std.eval()
    obs = player.env_reset()  # Uses the already constructed FIRST loco reset; no warm-up.
    player.get_batch_size(obs['obs'], 1)
    cpu = lambda x: x.detach().cpu().numpy().copy()
    n = ref.num_envs
    initial = dict(roots=cpu(ref._humanoid_root_states), boxes=cpu(ref._box_states),
                   observed_boxes=cpu(ref.cvpr_observed_initial_boxes), sizes=cpu(ref._box_size),
                   goals=cpu(ref._tar_pos), dof_pos=cpu(ref._dof_pos), dof_vel=cpu(ref._dof_vel),
                   kinematic=cpu(ref._kinematic_humanoid_rigid_body_states),
                   origins=cpu(ref._env_origins), density=cpu(ref._box_density))
    if hasattr(ref, 'snapshot_extras'):
        initial.update(ref.snapshot_extras())
    replay = os.environ.get('CVPR_REPLAY_INITIAL')
    if replay:
        archived = np.load(replay, allow_pickle=False)
        for key in ('sizes', 'density', 'origins'):
            if not np.array_equal(archived[key], initial[key]):
                raise ValueError('Replay asset/layout mismatch: '+key)
        if hasattr(ref, 'restore_initial'):
            ref.restore_initial(archived)
            initial = {k: archived[k].copy() for k in initial}
        else:
            for key, tensor in [('roots', ref._humanoid_root_states), ('boxes', ref._box_states),
                                ('observed_boxes', ref.cvpr_observed_initial_boxes), ('goals', ref._tar_pos),
                                ('dof_pos', ref._dof_pos), ('dof_vel', ref._dof_vel),
                                ('kinematic', ref._kinematic_humanoid_rigid_body_states)]:
                tensor.copy_(torch.as_tensor(archived[key], device=ref.device))
                initial[key] = archived[key].copy()
            ids = torch.arange(n, device=ref.device)
            ref._reset_env_tensors(ids)
            ref._refresh_sim_tensors()
        player.env._projection_reset = torch.arange(n, device=ref.device)
        obs = player.env._obs()
        obs = player.obs_to_torch(obs)
    np.savez_compressed(out/'initial.npz', **initial)
    specifications = dict(roots=((n, 1, 13), np.float32), boxes=((n, 2, 13), np.float32),
                          observed_boxes=((n, 2, 13), np.float32),
                          dof_pos=(tuple(ref._dof_pos.shape), np.float32),
                          dof_vel=(tuple(ref._dof_vel.shape), np.float32),
                          body_pos=(tuple(ref._rigid_body_pos.shape), np.float32),
                          body_rot=(tuple(ref._rigid_body_rot.shape), np.float32),
                          contact=(tuple(ref._contact_forces.shape), np.float32),
                          alive=((n,), bool), fallen=((n,), bool), success=((n, 2), bool),
                          distances_m=((n, 2), np.float64))
    trace = {key: np.lib.format.open_memmap(out/(key+'.npy'), mode='w+', dtype=dtype,
                                          shape=(HORIZON,)+shape) for key, (shape, dtype) in specifications.items()}
    alive = np.ones(n, bool)
    ever = np.zeros((n, 2), bool)
    minimum = np.full((n, 2), np.inf, dtype=np.float64)
    fall_step = np.full(n, -1, np.int32)
    for step in range(HORIZON):
        actions = player.get_action(obs, True)
        actions *= torch.as_tensor(alive, device=actions.device)[:, None]
        obs, _, _, _ = player.env_step(player.env, actions)
        if not isinstance(obs, dict):
            obs = {'obs': obs}
        roots, boxes = cpu(ref._humanoid_root_states), cpu(ref._box_states)
        raw_success, distances = metrics_frame(name, roots, boxes, initial['sizes'], initial['goals'], xy_fraction, scoring_mode)
        success = raw_success & alive[:, None]
        minimum = np.minimum(minimum, np.where(alive[:, None], distances, np.inf))
        trace['distances_m'][step] = distances
        ever |= success
        fallen = cpu(ref._terminate_buf).reshape(n).astype(bool) & alive
        trace['roots'][step], trace['boxes'][step] = roots, boxes
        observed_boxes = boxes.copy()
        if name != 'on_top':
            observed_boxes[:, 1] = initial['observed_boxes'][:, 1]
        trace['observed_boxes'][step] = observed_boxes
        for key, tensor in [('dof_pos', ref._dof_pos), ('dof_vel', ref._dof_vel),
                            ('body_pos', ref._rigid_body_pos), ('body_rot', ref._rigid_body_rot),
                            ('contact', ref._contact_forces)]:
            trace[key][step] = cpu(tensor)
        trace['alive'][step], trace['fallen'][step], trace['success'][step] = alive, fallen, success
        fall_step[fallen] = step+1
        for index in np.flatnonzero(fallen):
            handle = ref.humanoid_handles_all[int(index)][0]
            props = ref.gym.get_actor_dof_properties(ref.envs[int(index)], handle)
            props['stiffness'][:] = 0
            props['damping'][:] = 0
            props['driveMode'][:] = gymapi.DOF_MODE_NONE
            ref.gym.set_actor_dof_properties(ref.envs[int(index)], handle, props)
        alive &= ~fallen
        if (step+1) % 100 == 0:
            print(f'CVPR {name}: {step+1}/900 steps; ever={int(ever[:, 1].sum())}/{n}', flush=True)
        if not alive.any():
            break
    for array in trace.values():
        array.flush()
    np.save(out/'ever.npy', ever)
    np.save(out/'minimum_distances_m.npy', minimum)
    np.save(out/'fall_step.npy', fall_step)
    for key, before in frozen_stats.items():
        if not torch.equal(before, player.running_mean_std.state_dict()[key]):
            raise AssertionError('Evaluation changed normalization statistics: '+key)
    root = Path(__file__).resolve().parents[2]
    protocol_file = 'cvpr2027_evaluation_protocol_1109.md' if revision == '1109' else 'cvpr2027_evaluation_protocol.md'
    paths = [root/protocol_file, Path(__file__),
             root/'tokenhsi/learning/cvpr2027_scoring.py', root/'tokenhsi/env/cvpr2027_single.py',
             root/'tokenhsi/env/cvpr2027_scene.py', root/'tokenhsi/env/reference_projection.py']
    paths += sorted((root/'push_reference').rglob('*.py'))
    if scoring_mode == 'native_point':
        paths += [root/'tokenhsi/env/cvpr2027_native.py',
                  root/'tokenhsi/env/tasks/multi_task/humanoid_traj_sit_carry_climb.py',
                  root/'tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml']
    (out/'environment.json').write_text(json.dumps(ref.cfg, indent=2, default=str))
    report = dict(protocol='cvpr2027_single_'+revision, protocol_file=protocol_file, xy_fraction=xy_fraction,
                  scoring_mode=scoring_mode,
                  region_definition='native_point_alias' if scoring_mode == 'native_point' else 'yaw_xy_region',
                  scene_kind='original_tokenhsi_native' if scoring_mode == 'native_point' else 'protocol_boxes',
                  native_asset_paths=getattr(ref, 'asset_paths', None),
                  native_override_note='User override: original native assets/layout/target and 20cm Point SR. Region SR/Error fields alias Point, not XY100. Density placeholder is inapplicable for fixed-base furniture.' if scoring_mode == 'native_point' else None,
                  errors=error_statistics(minimum, ever[:, 1]), task=name, scenes=n, seed=int(ref.cfg['seed']),
                  horizon=HORIZON, frames=step+1, dt=ref.dt, warmup_steps=0,
                  deterministic=True, first_episode_only=True, success_reset=False,
                  normalization_frozen_verified=True,
                  checkpoint=player._checkpoint_fn,
                  checkpoint_sha256=hashlib.sha256(Path(player._checkpoint_fn).read_bytes()).hexdigest(),
                  source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  point=counts(ever[:, 0], np.ones(n, bool)), region=counts(ever[:, 1], np.ones(n, bool)),
                  falls=int((fall_step >= 0).sum()), success_unit='independent individual ever',
                  contact_body_ids=cpu(ref._contact_body_ids).tolist(),
                  termination_heights=cpu(ref._termination_heights).tolist(),
                  observation='Unmodified TokenHSI task projection; architecture has no distractor/object-token attention',
                  non_paper_smoke=n != 512)
    (out/'result.json').write_text(json.dumps(report, indent=2))
    audit = cpu_rescore(out)
    print('CVPR_EVAL '+json.dumps(dict(result=report, cpu_rescore=audit)), flush=True)
