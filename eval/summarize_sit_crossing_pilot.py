import contextlib
import io
import json

import numpy as np
import torch

from ind_mechanisms_common import digest, write
from ind_mechanisms_metrics import temporal
from run_sit_crossing_pilot import OUT
from rescore import rescore
from utils.evaluation_protocol import spatial_success


def analyze(path, scene):
    result = json.loads((path/'result.json').read_text())
    initial = torch.load(path/'initial.pt', map_location='cpu', weights_only=False)
    if result['num_envs'] != 64 or result['horizon'] != 900 or not result['policy_preserved']:
        raise ValueError('Evaluation settings or policy changed')
    for key in ('packet', 'groups', 'sizes', 'goals', 'dof_pos', 'dof_vel'):
        if not torch.equal(initial[key], scene[key]):
            raise ValueError('Initial state mismatch: '+key)
    for key in ('roots', 'boxes'):
        a, b = initial[key], scene[key]
        if key == 'boxes':
            a, b = a[scene['used']], b[scene['used']]
        if not torch.equal(a[..., 2:], b[..., 2:]) or not torch.allclose(a[..., :2], b[..., :2], atol=1e-4, rtol=0):
            raise ValueError('Initial world-coordinate roundtrip mismatch: '+key)
    with contextlib.redirect_stdout(io.StringIO()):
        proof = rescore(path)
    if not all(proof[k] for k in ('finite','cpu_gpu_exact','json_counts_exact','errors_match_recorded')):
        raise ValueError('CPU rescore failed')
    data = np.load(path/'trace.npz')
    if len(data['roots']) != 900:
        raise ValueError('Unexpected rollout length')
    packet, sizes, goals = [torch.from_numpy(data[k]) for k in ('packet','sizes','goals')]
    current = np.stack([(spatial_success(packet, torch.from_numpy(r), torch.from_numpy(b), sizes, goals)[..., 1]
                        & torch.from_numpy(v)).numpy() for r,b,v in zip(data['roots'],data['boxes'],data['valid'])])
    dt = result['control_dt']
    v = temporal(current, data['valid'], data['fall_step'], dt)
    if not np.array_equal(v['success'], data['ever'][...,1]):
        raise ValueError('Temporal score mismatch')
    extra = np.load(path/'mechanisms.npz')
    centers = scene['crossing_centers'][..., :2].numpy()
    zone = (np.linalg.norm(data['roots'][..., :2]-centers[None], axis=-1) < .8) & v['window']
    entered = zone.any(0)
    first_entry = np.where(entered, (zone.argmax(0)+1)*dt, np.nan)
    concurrent = zone.all(-1).any(0)
    near = (extra['human_body_point_distance'] < .2) & v['window'].any(-1)
    # Both actors must be active for the approach-only proximity metric.
    active_pair = v['window'].all(-1)
    pair_near = (extra['human_body_point_distance'] < .2) & active_pair
    chair_xy = scene['boxes'][:, :2, :2].numpy()
    reached = ((np.linalg.norm(data['roots'][..., :2]-chair_xy[None], axis=-1) < 1.) & v['window']).any(0)
    reach_frames = (np.linalg.norm(data['roots'][..., :2]-chair_xy[None], axis=-1) < 1.) & v['window']
    reach_time = np.where(reached, (reach_frames.argmax(0)+1)*dt, np.nan)
    time_all = np.where(v['success'], v['first']*dt, 900*dt)
    initial_xy = initial['roots'][None, ..., :2].numpy()
    path_length = (np.linalg.norm(np.diff(np.concatenate((initial_xy,data['roots'][...,:2])),axis=0),axis=-1)*v['window']).sum(0)
    metrics = dict(sr=float(v['success'].mean()), successes=int(v['success'].sum()), trials=128,
        both_success=float(v['success'].all(-1).mean()), pre_success_fall=float(v['prefall'].mean()),
        hold1=float(v['hold1'].mean()), time_all=float(time_all.mean()),
        time_success=float(time_all[v['success']].mean()) if v['success'].any() else None,
        human_near20_scene=float(near.any(0).mean()),
        human_near20_both_active_scene=float(pair_near.any(0).mean()),
        zone_entry=float(entered.mean()), simultaneous_zone_scene=float(concurrent.mean()),
        both_entered_scene=float(entered.all(-1).mean()),
        entry_time_gap_both_entered_s=float(np.abs(first_entry[:,0]-first_entry[:,1])[entered.all(-1)].mean()) if entered.all(-1).any() else None,
        chair_1m_reached=float(reached.mean()),
        chair_1m_reach_time_s=float(reach_time[reached].mean()) if reached.any() else None,
        success_given_chair_1m_reached=float(v['success'][reached].mean()) if reached.any() else None,
        travel_m=float(path_length.mean()), cpu_verified=True)
    np.savez_compressed(path/'pilot_metrics.npz', success=v['success'], prefall=v['prefall'],
        first=v['first'], time_all=time_all, entered=entered, first_entry=first_entry,
        concurrent=concurrent, pair_near=pair_near.any(0), reached=reached,
        reach_time=reach_time, travel_m=path_length)
    write(path/'pilot_metrics.json', metrics)
    return metrics


def main():
    summary = {}
    manifest = json.loads((OUT/'manifest.json').read_text())
    for path, expected in manifest['checkpoints'].items():
        if digest(path) != expected:
            raise ValueError('Checkpoint changed')
    for condition in ('separated','cross'):
        scene_path = OUT/f'{condition}_seed1.pt'
        if digest(scene_path) != manifest['scenes'][condition]:
            raise ValueError('Scene changed')
        scene = torch.load(scene_path, map_location='cpu', weights_only=False)
        paths = [OUT/'runs'/condition/model for model in ('sct','ours')]
        for name in ('initial.pt','first_policy_input.pt'):
            a,b = [torch.load(p/name, map_location='cpu', weights_only=False) for p in paths]
            equal = torch.equal(a,b) if torch.is_tensor(a) else a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)
            if not equal:
                raise ValueError('Model inputs differ: '+name)
        summary[condition] = {model:analyze(path,scene) for model,path in zip(('sct','ours'),paths)}
    summary['sr_drop_pp'] = {m:100*(summary['separated'][m]['sr']-summary['cross'][m]['sr']) for m in ('sct','ours')}
    write(OUT/'summary.json', summary)
    lines = ['# SIT crossing pilot: 64 scenes, seed 1, 2 agents, 900 steps', '',
        '| Condition | Model | SR (%) | Falls before success (%) | Both succeed (%) | Time all (s) | Body proximity <20cm, both active (% scenes) |',
        '|---|---|---:|---:|---:|---:|---:|']
    for c in ('separated','cross'):
        for m in ('sct','ours'):
            x=summary[c][m]
            lines.append(f'| {c} | {m} | {100*x["sr"]:.2f} | {100*x["pre_success_fall"]:.2f} | {100*x["both_success"]:.2f} | {x["time_all"]:.2f} | {100*x["human_near20_both_active_scene"]:.2f} |')
    lines += ['', 'SR drop separated → cross (pp): '+json.dumps(summary['sr_drop_pp']), '',
        'Pilot only; one evaluation seed, fixed checkpoints. Proximity is not physical contact.',
        'Crossing-zone radius: 0.8m. Geometry is paired, but separation also changes partner/object context.',
        'Task-information use is not causally identified by this experiment.',
        'Entry times, concurrent occupancy, chair approach, and per-trial results are saved separately.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    write(OUT/'verification.json', dict(completed_runs=4, model_initial_states_exact=True,
        first_policy_inputs_exact=True, cpu_rescore_exact=True, checkpoints_preserved=True))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
