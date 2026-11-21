import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
import torch

from random_delay import ROOT, OUT, BASE, SETTINGS, LABELS, CONDITION, jobs, key, read, write, sha
sys.path[:0] = [str(ROOT/'eval'), str(ROOT/'tokenhsi')]
import stage2_xy100
from utils.evaluation_protocol import spatial_success
from analyze_coordination_behavior import behavior_arrays
from coupled_cooperation_metrics import counters, ratios, surface_contact, reference_function, world_points
from table2_metrics import surface_distance

stage2_xy100.install()
torch.set_num_threads(1)
REFERENCE = ROOT/'output_etc/check/coupled_cooperation_four_models_20261111/reference'


def first(mask):
    return np.where(mask.any(0), mask.argmax(0)+1, -1)


def mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else None


def signature(values):
    digest = hashlib.sha256()
    for name, tensor in sorted(values.items()):
        value = tensor.numpy() if isinstance(tensor, torch.Tensor) else tensor
        digest.update(name.encode())
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def score(args):
    path = args.source
    inputs = [path/name for name in ('initial.pt', 'trace.npz', 'freeze_trace.npz', 'freeze.json',
        'table2_trials.npz', 'table2.json', 'policy_verification.json', 'first_policy_input.pt',
        'initial_before_normalization.pt', 'result.json')]
    if (path/'cooperation.npz').exists():
        inputs.append(path/'cooperation.npz')
    hashes = {str(p): sha(p) for p in inputs}
    initial = torch.load(path/'initial.pt', map_location='cpu', weights_only=False)
    with np.load(path/'trace.npz') as z:
        roots, boxes, alive, fall = (z[k] for k in ('roots', 'boxes', 'valid', 'fall_step'))
    with np.load(path/'freeze_trace.npz') as z:
        freeze_trace = {k: z[k] for k in z.files}
    info, proof, freeze = (read(path/n) for n in ('table2.json', 'policy_verification.json', 'freeze.json'))
    result = read(path/'result.json')
    steps, n, m = alive.shape
    if n != 64 or m != SETTINGS[args.setting] or steps < args.steps:
        raise ValueError('Unexpected cohort/horizon')
    if steps not in (210, 900) or not np.isfinite(roots).all() or not np.isfinite(boxes).all():
        raise ValueError('Invalid trajectory')
    if (result['seed'], info['model'], info['setting'], info['delay_seconds']) != (args.seed, args.model, args.setting, args.delay):
        raise ValueError('Job metadata mismatch')
    epochs = dict(sct=26000, fsm=26000, coopft=7000, ours=2000, tokenhsi=40500)
    if proof['epoch'] != epochs[args.model] or not proof['weights_unchanged'] or not proof['normalization_unchanged']:
        raise ValueError('Checkpoint preservation mismatch')
    schedule_path = OUT/'schedules'/f'{args.setting}.json'
    schedule = read(schedule_path)
    delay_steps = np.asarray(schedule['steps'])
    if freeze['schedule_sha256'] != sha(schedule_path) or info['delay_schedule_sha256'] != sha(schedule_path):
        raise ValueError('Schedule hash mismatch')
    if result['delay_schedule_sha256'] != sha(schedule_path):
        raise ValueError('Result schedule mismatch')
    np.testing.assert_array_equal(freeze['delay_steps'], delay_steps)
    np.testing.assert_array_equal(freeze['release_before_action_step'], delay_steps+1)
    if freeze['max_frozen_root_error'] or freeze['max_frozen_dof_error']:
        raise ValueError('Freeze error')
    idx = np.arange(n)
    selected = np.asarray(freeze['selected'])
    parent = initial['predecessor'].numpy()
    expected_selected = np.where((parent >= 0).any(1), parent.max(1), idx % m)
    np.testing.assert_array_equal(selected, expected_selected)
    expected = alive & (np.arange(steps)[:, None, None] < delay_steps[None, :, None])
    expected &= np.arange(m)[None, None] == selected[None, :, None]
    np.testing.assert_array_equal(freeze_trace['delayed'], expected)
    np.testing.assert_array_equal(roots[expected], np.broadcast_to(initial['roots'][None, ..., :3].numpy(), roots.shape)[expected])
    reference = BASE/f'reference/{args.setting}/general_seed{args.seed}/observed_scene.pt'
    observed = torch.load(reference, map_location='cpu', weights_only=False)
    raw = torch.load(path/'initial_before_normalization.pt', map_location='cpu', weights_only=False)
    if not torch.equal(raw, observed):
        raise ValueError('Shared raw input mismatch')
    scene = torch.load(path/'first_policy_input.pt', map_location='cpu', weights_only=False)
    native_signature = None
    if args.model == 'tokenhsi':
        native = torch.load(path/'tokenhsi_first_input.pt', map_location='cpu', weights_only=False)
        if native['raw'].shape != (n*m, 397) or not torch.equal(native['scene'], scene):
            raise ValueError('TokenHSI input mismatch')
        if proof['action_calls'] != steps or proof['other_policy_actions'] != 0 or not result['protocol_limitations']:
            raise ValueError('TokenHSI policy mismatch')
        native_signature = signature(native)
    enabling = args.setting.startswith('enable')
    if enabling:
        current = []
        for start in range(0, steps, 30):
            width = min(30, steps-start)
            v = spatial_success(initial['packet'].repeat(width, 1, 1),
                torch.from_numpy(roots[start:start+width]).reshape(width*n, m, 3),
                torch.from_numpy(boxes[start:start+width]).reshape(width*n, -1, 7),
                initial['sizes'].repeat(width, 1, 1), initial['goals'].repeat(width, 1, 1))[..., 1]
            current.append(v.reshape(width, n, m).numpy())
        current = np.concatenate(current) & alive
        group = current.all(-1)
    else:
        with np.load(path/'cooperation.npz') as z:
            pose, sizes, hands = (z[k] for k in ('physical_boxes', 'physical_sizes', 'hands'))
        goal = initial['shared_center_goals'][:, 0].numpy().astype(float)
        group = (np.linalg.norm(pose[..., :3].astype(float)-goal, axis=-1) <= .2) & alive.any(-1)
    first_full = first(group)
    with np.load(path/'table2_trials.npz') as z:
        np.testing.assert_array_equal(first_full, z['first_success'])
    if int((first_full > 0).sum()) != info['successes']:
        raise ValueError('Full success count mismatch')
    first_group = first(group[:args.steps])
    if args.model == 'fsm':
        released = np.zeros((n, m), dtype=bool)
        arrived = released.copy()
        follower_mask = parent >= 0
        if not enabling:
            distance = surface_distance(torch.from_numpy(hands).reshape(steps*n, m, 2, 3).double(),
                torch.from_numpy(pose).reshape(steps*n, 7).double(),
                torch.from_numpy(sizes).repeat(steps, 1).double()).reshape(steps, n, m, 2).numpy()
            near = (distance < .12).all(-1)
        for t in range(steps):
            if t:
                if enabling:
                    released |= follower_mask & np.take_along_axis(current[t-1], parent.clip(min=0), axis=1)
                else:
                    arrived |= near[t-1] & alive[t]
                    released |= arrived.all(-1)[:, None]
            expected_fsm = ((follower_mask if enabling else arrived) & ~released) & alive[t]
            np.testing.assert_array_equal(freeze_trace['fsm_frozen'][t], expected_fsm)
            np.testing.assert_array_equal(freeze_trace['released'][t], released)
            np.testing.assert_array_equal(freeze_trace['frozen'][t], expected_fsm | expected[t])
    row = dict(setting=args.setting, model=args.model, seed=args.seed, delay=args.delay, source=str(path),
        recorded_steps=steps, analyzed_steps=args.steps, trials=n, successes=int((first_group > 0).sum()),
        group_sr_percent=float((first_group > 0).mean()*100), initial_signature=signature(initial),
        input_signature=signature(dict(scene=scene)), native_signature=native_signature,
        checkpoint_epoch=proof['epoch'], verified=True)
    baseline = read(BASE/'analysis'/f'{args.setting}/{args.model}_d0_seed1.json')
    for field in ('initial_signature', 'input_signature', 'native_signature', 'checkpoint_epoch'):
        if row[field] != baseline[field]:
            raise ValueError('Unpaired baseline: '+field)
    row.update(schedule_sha256=sha(schedule_path), paired_baseline_exact=True)
    arrays = dict(first_group=first_group, delay_steps=delay_steps)
    if enabling:
        follower = (parent >= 0).argmax(1)
        leader = parent[idx, follower]
        at = first(current[:args.steps, idx, leader])
        box = initial['packet'].numpy()[idx, leader, 3]-m
        q = initial['boxes'].numpy()[idx, box, 3:7].astype(float)
        yaw = np.arctan2(2*(q[:, 3]*q[:, 2]+q[:, 0]*q[:, 1]), 1-2*(q[:, 1]**2+q[:, 2]**2))
        positions = np.concatenate((initial['roots'].numpy()[idx, follower, :2][None], roots[:args.steps, idx, follower, :2])).astype(float)
        behavior = behavior_arrays(positions, initial['goals'].numpy()[idx, leader, :2].astype(float),
            initial['sizes'].numpy()[idx, box, :2].astype(float)/2, yaw, at, first_group)
        arrays.update(first_at=at, prep_percent=behavior['prep_percent'], occupancy_percent=behavior['occupancy_percent'])
        row.update(at_successes=int((at > 0).sum()), at_sr_percent=float((at > 0).mean()*100),
            prep_percent=mean(behavior['prep_percent']), occupancy_percent=mean(behavior['occupancy_percent']))
    else:
        pose, hands = pose[:args.steps], hands[:args.steps]
        start, begin, end = counters(pose, initial['shared_centers'][:, 0].numpy(), first_group, .2)
        raw_ratio, ratio = ratios(surface_contact(hands, pose, sizes).sum(0), begin, end)
        jerk = reference_function(REFERENCE, 'task_util.py', 'compute_rim_velocity_variance_norm')
        j = np.zeros(n, dtype=np.float32)
        for offset in range(0, n, 8):
            stop = min(offset+8, n)
            points = world_points(pose[:, offset:stop], sizes[offset:stop], REFERENCE)
            a, b = torch.from_numpy(begin[offset:stop]), torch.from_numpy(end[offset:stop])
            value = jerk(points, a, b, torch.arange(stop-offset), delta_t=.0333)
            velocity = (points[:, 1:]-points[:, :-1])/.0333
            acc = (velocity[:, 1:]-velocity[:, :-1])/.0333
            absolute = ((acc[:, 1:]-acc[:, :-1])/.0333).abs()
            t = torch.arange(len(pose)-3)[None]
            mask = (t >= a[:, None]) & (t+2 < b[:, None])
            check = (absolute*mask[..., None, None]).sum((1, 2, 3))/(mask.sum(1)*64*3).clamp_min(1)
            if not torch.equal(value, check):
                raise ValueError('Jerk formula mismatch')
            j[offset:stop] = value.numpy()
        arrays.update(tcoop=ratio, jerk=j)
        row.update(tcoop_percent=mean(ratio*100), jerk_m_s3=mean(j[j != 0]),
            tcoop_valid=int(np.isfinite(ratio).sum()), tcoop_clipped=int((raw_ratio > 1).sum()),
            jerk_valid=int((j != 0).sum()))
    if any(sha(Path(p)) != h for p, h in hashes.items()):
        raise ValueError('Source modified')
    row['source_sha256'] = hashes
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.destination.with_suffix('.npz'), **arrays)
    write(args.destination, row)



def summarize():
    manifest = read(OUT/'manifest.json')
    baseline = []
    for item in manifest['baselines']:
        path = Path(item['analysis'])
        if sha(path) != item['sha256']:
            raise ValueError('Baseline modified')
        baseline.append(read(path))
    random = [read(OUT/'analysis'/f'{key(j)}.json') for j in jobs()
              if (OUT/'analysis'/f'{key(j)}.json').exists()]
    for row in random:
        if row['schedule_sha256'] != manifest['schedule_sha256'][row['setting']]:
            raise ValueError('Unpaired delay schedule')
    rows = baseline+random
    macro = []
    for family in ('enable', 'coupled'):
        metrics = ('group_sr_percent', 'prep_percent', 'occupancy_percent', 'at_sr_percent') if family == 'enable' else ('group_sr_percent', 'tcoop_percent', 'jerk_m_s3')
        for model in LABELS:
            for condition in (0, CONDITION):
                selected = [r for r in rows if r['setting'].startswith(family) and r['model'] == model and r['delay'] == condition]
                if len(selected) == 3:
                    macro.append(dict(setting=family+'_average', model=model, delay=condition,
                        **{metric: mean([r[metric] for r in selected]) for metric in metrics}))
    write(OUT/'summary.json', dict(completed=len(random) == 30, verified_random=len(random),
        baseline_runs=30, runs=rows, macro=macro))
    lines = ['# Random delay: 64 environments, seed1, 30 seconds', '',
        f'Random runs verified: {len(random)}/30. Each cell: 0s → Uniform(1–5s). Single-seed means.', '']
    for setting in [*SETTINGS, 'enable_average', 'coupled_average']:
        metrics = ('group_sr_percent', 'prep_percent', 'occupancy_percent', 'at_sr_percent') if setting.startswith('enable') else ('group_sr_percent', 'tcoop_percent', 'jerk_m_s3')
        lines += ['## '+setting, '', '| Model | '+' | '.join(metrics)+' |', '|---|'+'---|'*len(metrics)]
        for model, label in LABELS.items():
            selected = {r['delay']: r for r in rows+macro if r['setting'] == setting and r['model'] == model}
            cells = [' → '.join('N/A' if d not in selected or selected[d][metric] is None else f"{selected[d][metric]:.2f}" for d in (0, CONDITION)) for metric in metrics]
            lines.append('| '+label+' | '+' | '.join(cells)+' |')
        lines.append('')
    lines += ['Preparation/occupancy: AT-success trials only. Family means weight the three settings equally. t_coop/Jerk retain the previous valid-trial rules. TokenHSI retains its397D input limitation.', '']
    (OUT/'report.md').write_text('\n'.join(lines))
    if len(random) == 30:
        write(OUT/'completion_verification.json', dict(completed=True, new_runs=30, baseline_runs=30,
            random_trials=1920, paired_initial_and_policy_input=True, shared_schedule=True,
            baseline_hashes_unchanged=True, steps=900, seed=1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--source', type=Path)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--setting')
    parser.add_argument('--model')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--steps', type=int, default=900)
    args = parser.parse_args()
    args.delay = CONDITION
    summarize() if args.summarize else score(args)
