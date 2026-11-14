import contextlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch

from ind_mechanisms_common import ROOT, OUT, SUMMARY, REFERENCE, digest, write
sys.path.insert(0, str(ROOT/'tokenhsi'))
import stage2_xy100
from rescore import rescore
from utils.evaluation_protocol import spatial_success

TASKS = ('SIT', 'CLIMB', 'AT', 'ON_TOP', 'PUSH')
stage2_xy100.install()
torch.set_num_threads(1)


def temporal(current, valid, falls, dt):
    success = current.any(0)
    first = np.where(success, current.argmax(0)+1, -1)
    streak = np.zeros_like(first)
    longest = np.zeros_like(first)
    for frame in current:
        streak = (streak+1)*frame
        longest = np.maximum(longest, streak)
    prefall = (falls >= 0) & ((first < 0) | (falls < first))
    window = valid & ((first[None] < 0) | (np.arange(len(current))[:, None, None]+1 <= first[None]))
    return dict(success=success, first=first, longest=longest, prefall=prefall, window=window,
                hold05=longest >= round(.5/dt), hold1=longest >= round(1./dt))


def analyze(path):
    path = Path(path)
    with contextlib.redirect_stdout(io.StringIO()):
        proof = rescore(path)
    if not all(proof[k] for k in ('finite','cpu_gpu_exact','json_counts_exact','errors_match_recorded')):
        raise ValueError('CPU rescore failed')
    result = json.loads((path/'result.json').read_text())
    data = np.load(path/'trace.npz')
    packet, sizes, goals = [torch.from_numpy(data[k]) for k in ('packet','sizes','goals')]
    frames = []
    for roots, boxes, valid in zip(data['roots'], data['boxes'], data['valid']):
        frames.append((spatial_success(packet, torch.from_numpy(roots), torch.from_numpy(boxes), sizes, goals)[...,1]
                       & torch.from_numpy(valid)).numpy())
    current = np.stack(frames)
    values = temporal(current, data['valid'], data['fall_step'], result['control_dt'])
    if not np.array_equal(values['success'], data['ever'][...,1]):
        raise ValueError('Temporal success mismatch')
    types = data['packet'][...,1]
    saved = {k:v for k,v in values.items() if k != 'window'}
    saved['types'] = types
    saved['minimum_point_error_m'] = data['minimum_errors'][...,0]
    saved['time_all'] = np.where(values['success'], values['first']*result['control_dt'], result['horizon']*result['control_dt'])
    rows = {}
    for kind, name in enumerate(TASKS):
        mask = types == kind
        if not mask.any():
            continue
        rows[name] = dict(trials=int(mask.sum()), sr=float(values['success'][mask].mean()),
            pre_success_fall=float(values['prefall'][mask].mean()),
            hold05=float(values['hold05'][mask].mean()), hold1=float(values['hold1'][mask].mean()),
            time_all=float(saved['time_all'][mask].mean()),
            time_success=float(saved['time_all'][mask & values['success']].mean()) if (mask & values['success']).any() else None)
        if kind in (2,4):
            rows[name]['tolerance_sr'] = {str(cm):float((saved['minimum_point_error_m'][mask] <= cm/100).mean()) for cm in (10,20,30,50)}
            rows[name]['within50_but_not20'] = float(((saved['minimum_point_error_m'][mask] <= .5) & (saved['minimum_point_error_m'][mask] > .2)).mean())
    if (path/'mechanisms.npz').exists() and types.shape[1] == 2:
        extra = np.load(path/'mechanisms.npz')
        scene = torch.load(result['mechanism_scene'], map_location='cpu', weights_only=False)
        initial = torch.load(path/'initial.pt', map_location='cpu', weights_only=False)
        center = (scene['boxes'][:, :2, :2]+scene['goals'][..., :2]).numpy()/2
        zone = (np.linalg.norm(data['roots'][..., :2]-center[None],axis=-1) < 1.5) & values['window']
        speed = np.linalg.norm(extra['root_velocity'][..., :2],axis=-1)
        saved['zone_frames'] = zone.sum(0)
        saved['zone_speed_sum'] = (speed*zone).sum(0)
        saved['zone_slow_frames'] = ((speed < .2) & zone).sum(0)
        points = np.concatenate((initial['roots'][None,...,:2].numpy(),data['roots'][...,:2]),axis=0)
        saved['travel_m'] = (np.linalg.norm(np.diff(points,axis=0),axis=-1)*values['window']).sum(0)
        for cm in (10,20,30):
            near = extra['human_body_point_distance'][...,None] < cm/100
            saved[f'human_near_{cm}'] = (near & values['window']).any(0)
        for cm in (6,12,20):
            near = extra['other_box_surface_distance'] < cm/100
            saved[f'box_near_{cm}'] = (near & values['window']).any(0)
    np.savez_compressed(path/'trial_metrics.npz', **saved)
    overview = dict(sr=float(values['success'].mean()), both_success=float(values['success'].all(-1).mean()),
        pre_success_fall=float(values['prefall'].mean()), time_all=float(saved['time_all'].mean()), tasks=rows,
        source_sha256=digest(path/'trace.npz'), cpu_verified=True)
    for k in saved:
        if '_near_' in k:
            overview[k] = float(saved[k].mean())
    if 'zone_frames' in saved:
        entered = saved['zone_frames'] > 0
        overview.update(zone_entered_rate=float(entered.mean()), travel_m=float(saved['travel_m'].mean()),
            zone_speed_entered=float((saved['zone_speed_sum'][entered]/saved['zone_frames'][entered]).mean()) if entered.any() else None,
            zone_slow_fraction_entered=float((saved['zone_slow_frames'][entered]/saved['zone_frames'][entered]).mean()) if entered.any() else None)
    write(path/'metrics.json', overview)
    return overview


def validate_run(path, scene_path, horizon=900):
    path = Path(path)
    s = torch.load(scene_path, map_location='cpu', weights_only=False)
    initial = torch.load(path/'initial.pt', map_location='cpu', weights_only=False)
    root_roundtrip = float((initial['roots']-s['roots']).abs().max())
    for key in ('roots','dof_pos','dof_vel','sizes','goals','packet','groups'):
        if key == 'roots' and 'expected_observation' not in s:
            if not torch.equal(initial[key][...,2:], s[key][...,2:]) or root_roundtrip > 1e-4:
                raise ValueError('Generated root world/local roundtrip exceeds 0.1mm')
            continue
        if not torch.equal(initial[key], s[key]):
            raise ValueError('Initial snapshot mismatch: '+key)
    actual_boxes, expected_boxes = initial['boxes'][s['used']], s['boxes'][s['used']]
    box_roundtrip = float((actual_boxes-expected_boxes).abs().max())
    if 'expected_observation' in s:
        if not torch.equal(actual_boxes, expected_boxes):
            raise ValueError('Extracted task object states changed')
    elif not torch.equal(actual_boxes[...,2:], expected_boxes[...,2:]) or box_roundtrip > 1e-4:
        raise ValueError('Generated box world/local roundtrip exceeds 0.1mm')
    observed = torch.load(path/'first_policy_input.pt', map_location='cpu', weights_only=False)
    if 'expected_observation' in s and not torch.equal(observed, s['expected_observation']):
        raise ValueError('Solo observation differs from extracted input')
    result = json.loads((path/'result.json').read_text())
    if result['horizon'] != horizon or result['num_envs'] != 512 or result['scene_sha256'] != digest(scene_path):
        raise ValueError('Wrong run configuration')
    if not result['policy_preserved'] or not result['actual_first_input_verified']:
        raise ValueError('Missing policy proof')
    if result['simulated_steps'] < horizon and result['falls'] != 512*s['roots'].shape[1]:
        raise ValueError('Invalid early termination')
    metrics = analyze(path)
    write(path/'verification.json', dict(initial_exact=root_roundtrip == box_roundtrip == 0,
        root_roundtrip_max_m=root_roundtrip, box_roundtrip_max_m=box_roundtrip, cpu_scores_exact=True,
        scene_sha256=digest(scene_path), horizon=horizon, metrics=metrics))
    return metrics


def prepare_baselines():
    import shutil
    entries = json.loads(SUMMARY.read_text())['seed_runs']
    for row in entries:
        if row['model'] not in ('stage1','ours') or row.get('name') != 'independent_2h':
            continue
        model = 'sct' if row['model'] == 'stage1' else 'ours'
        seed = row['seed']
        source = Path(row['source'])
        out = OUT/f'baseline/{model}_seed{seed}'
        if (out/'metrics.json').exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        for name in ('trace.npz','initial.pt'):
            if not (out/name).exists():
                (out/name).symlink_to(source/name)
        shutil.copy2(source/'result.json', out/'result.json')
        reference = torch.load(REFERENCE/f'general_seed{seed}/initial.pt', map_location='cpu', weights_only=False)
        actual = torch.load(source/'initial.pt', map_location='cpu', weights_only=False)
        for key in ('roots','goals','sizes','packet','dof_pos','dof_vel'):
            if not torch.equal(reference[key], actual[key]):
                raise ValueError('Baseline initial mismatch: '+key)
        hashes = {name:digest(source/name) for name in ('trace.npz','result.json','initial.pt')}
        analyze(out)
        if hashes != {name:digest(source/name) for name in hashes}:
            raise ValueError('Original baseline changed')
        write(out/'source.json', dict(source=str(source), sha256=hashes, original_unchanged=True))


if __name__ == '__main__':
    if sys.argv[1] == 'baseline':
        prepare_baselines()
    else:
        validate_run(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv)>3 else 900)
