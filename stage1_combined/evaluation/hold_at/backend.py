"""Deterministic first-episode evaluator. Does not call training success metrics."""
import json
import csv
import os
from pathlib import Path
import numpy as np
import torch
import yaml
from evaluation.hold_at.metrics import PROTOCOL, classify, sha256
from evaluation.hold_at.carry_metrics import score as carry_score, first_success_steps, PROTOCOL as CARRY_PROTOCOL
from tokenhsi.utils.hold_at_ft import object_catalog

def cpu(value):
    return value.detach().cpu().numpy().copy()

@torch.no_grad()
def run_eval(player):
    task = player.env.task
    assert task._is_eval and task.num_agents == task.num_objects == 1
    assert task.max_episode_length == 900 and abs(task.dt - 1/30) < 1e-7
    assert task.cfg['env']['holdAtFT']['asset_split'] == 'test'
    out = Path(task.cfg['args'].output_path)
    if (out / 'manifest.json').exists():
        raise FileExistsError('Do not reuse an evaluation output directory')
    out.mkdir(parents=True, exist_ok=True)
    # Preserve both the input and the actual config after task/CLI overrides.
    (out / 'input_env.yaml').write_text((out / 'resolved_env.yaml').read_text())
    (out / 'resolved_env.yaml').write_text(yaml.safe_dump(
        dict(env=task.cfg['env'], sim=task.cfg.get('sim', {})), sort_keys=False))
    checkpoint = Path(task.cfg['args'].checkpoint).resolve()
    manifest = dict(protocol=PROTOCOL, action='AT', template='HOLDING_AT',
                    seed=task.cfg['args'].seed, num_envs=task.num_envs, num_agents=1,
                    num_objects=1, episode_length=900, dt=task.dt, deterministic=True,
                    initial_frame_scored=False, reset_after_step=False,
                    checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                    asset_split='test', category='ArmChair_Normal',
                    official_sample_count=task.num_envs == 512,
                    secondary_protocol=CARRY_PROTOCOL,
                    source_sha256={str(p): sha256(p) for base in ('tokenhsi', 'evaluation/hold_at') for p in sorted(Path(base).rglob('*.py'))})
    if hasattr(task, '_public_policy_metadata'):
        manifest['scene_checkpoint'] = manifest['checkpoint']
        manifest['scene_checkpoint_sha256'] = manifest['checkpoint_sha256']
        manifest['public_policy'] = task._public_policy_metadata
        manifest['checkpoint'] = task._public_policy_metadata['checkpoint']
        manifest['checkpoint_sha256'] = task._public_policy_metadata['checkpoint_sha256']
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    obs = player.env_reset()
    player.get_batch_size(obs['obs'], 1)
    n = task.num_envs
    initial = dict(root=cpu(task._humanoid_root_states[:, 0]),
                   objects=cpu(task._box_states[:, 0]), goals=cpu(task._tar_pos[:, 0]),
                   dof_pos=cpu(task._dof_pos), dof_vel=cpu(task._dof_vel),
                   sizes=cpu(task._box_size[:, 0]), model_ids=cpu(task._object_model_ids),
                   origins=cpu(task._env_origins), ground_z=cpu(task._object_ground_z),
                   front=cpu(task._object_front))
    np.savez_compressed(out / 'initial_0.npz', **initial)
    if hasattr(task, '_native_up'):
        up = cpu(task._native_up).reshape(n, 3)
    else:
        cfg = task.cfg['env']['holdAtFT']
        catalog = object_catalog(os.environ.get('HOLD_AT_OBJECT_ROOT', cfg['object_root']),
                                 cfg['asset_split'], cfg['category'])
        up = np.asarray([catalog[int(i)]['up_facing'] for i in initial['model_ids']], dtype=np.float32)
    np.savez_compressed(out / 'object_metadata.npz', up_facing=up, model_ids=initial['model_ids'])
    active = np.ones(n, bool)
    ever = np.zeros(n, bool)
    lengths = np.zeros(n, int)
    trace = {k: [] for k in ('root', 'objects', 'goals', 'target', 'active', 'done', 'terminate', 'point', 'region')}
    for step in range(900):
        if not isinstance(obs, dict):
            obs = {'obs': obs}
        if not torch.isfinite(obs['obs']).all():
            raise ValueError('Nonfinite policy input')
        action = player.get_action(obs, True)
        if not torch.isfinite(action).all():
            raise ValueError('Nonfinite policy action')
        obs, _, done, info = player.env_step(player.env, action)
        objects, goals = cpu(task._box_states[:, 0]), cpu(task._tar_pos[:, 0])
        if not np.isfinite(objects).all() or not np.isfinite(goals).all():
            raise ValueError('Nonfinite simulation state')
        point = classify(objects, goals)
        ending = cpu(done).reshape(n).astype(bool)
        term = cpu(info['terminate']).reshape(n).astype(bool)
        values = dict(root=cpu(task._humanoid_root_states[:, 0]), objects=objects,
                      goals=goals, target=goals[:, :3].copy(), active=active.copy(),
                      done=ending, terminate=term, point=point, region=point.copy())
        for key, value in values.items():
            trace[key].append(value)
        ever |= point & active
        lengths += active
        active &= ~ending
        if not active.any():
            break
        # Never reset done environments: later episodes cannot enter ever.
    if active.any():
        raise RuntimeError('First episode did not finish within horizon')
    assert float(task._sampling_failures) == 0
    data = {k: np.stack(v) for k, v in trace.items()}
    np.savez_compressed(out / 'trace_0.npz', **data)
    tokenhsi = carry_score(data, up)
    result = dict(protocol=PROTOCOL, seed=manifest['seed'], num_envs=n,
                  point_ever_count=int(ever.sum()), region_ever_count=int(ever.sum()),
                  point_ever=float(ever.mean()), region_ever=float(ever.mean()),
                  episode_steps=lengths.tolist(), official_sample_count=n == 512,
                  tokenhsi_criteria=tokenhsi)
    ours_first = first_success_steps(data['point'] & data['active'])
    with (out / 'episodes.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['seed', 'env_id', 'episode_steps', 'ours_30s_success', 'ours_first_success_step',
                         'tokenhsi_20s_success', 'tokenhsi_first_success_step', 'tokenhsi_min_success_error_cm'])
        for i in range(n):
            writer.writerow([manifest['seed'], i, lengths[i], int(ever[i]), ours_first[i],
                             int(tokenhsi['success_by_env'][i]), tokenhsi['first_success_step'][i],
                             tokenhsi['min_success_error_cm'][i]])
    (out / 'results.json').write_text(json.dumps(result, indent=2))
    artifacts = ['manifest.json', 'input_env.yaml', 'resolved_env.yaml', 'initial_0.npz', 'trace_0.npz',
                 'object_metadata.npz', 'episodes.csv', 'results.json']
    (out / 'completed.json').write_text(json.dumps(dict(rc=0, sha256={x: sha256(out / x) for x in artifacts}), indent=2))
    print('HOLD_AT_EVAL', json.dumps(result), flush=True)
    return result
