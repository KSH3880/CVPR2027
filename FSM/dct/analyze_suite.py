import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
import torch

from suite import ROOT, OUT, SETTINGS, SEEDS, DELAYS, LABELS, jobs, key, read, write, sha
sys.path[:0] = [str(ROOT/'eval'), str(ROOT/'tokenhsi')]
import stage2_xy100
from utils.evaluation_protocol import spatial_success
from analyze_coordination_behavior import behavior_arrays
from coupled_cooperation_metrics import counters, ratios, surface_contact, reference_function, world_points
from table2_metrics import surface_distance

stage2_xy100.install()
torch.set_num_threads(1)
REFERENCE = ROOT/'FSM/dct/reference'


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
    if steps not in (135, 900, 1050) or not np.isfinite(roots).all() or not np.isfinite(boxes).all():
        raise ValueError('Invalid trajectory')
    if (result['seed'], info['model'], info['setting'], info['delay_seconds']) != (args.seed, args.model, args.setting, args.delay):
        raise ValueError('Job metadata mismatch')
    epochs = dict(sct=26000, fsm=26000, coopft=7000, ours=2000, tokenhsi=40500)
    if proof['epoch'] != epochs[args.model] or not proof['weights_unchanged'] or not proof['normalization_unchanged']:
        raise ValueError('Checkpoint preservation mismatch')
    if freeze['max_frozen_root_error'] or freeze['max_frozen_dof_error'] or freeze['delay_steps'] != args.delay*30:
        raise ValueError('Freeze error')
    idx = np.arange(n)
    selected = np.asarray(freeze['selected'])
    parent = initial['predecessor'].numpy()
    expected_selected = np.where((parent >= 0).any(1), parent.max(1), idx % m)
    np.testing.assert_array_equal(selected, expected_selected)
    expected = alive & (np.arange(steps)[:, None, None] < args.delay*30)
    expected &= np.arange(m)[None, None] == selected[None, :, None]
    np.testing.assert_array_equal(freeze_trace['delayed'], expected)
    np.testing.assert_array_equal(roots[expected], np.broadcast_to(initial['roots'][None, ..., :3].numpy(), roots.shape)[expected])
    reference = OUT/f'reference/{args.setting}/general_seed{args.seed}/observed_scene.pt'
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
    arrays = dict(first_group=first_group)
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
    rows = [read(OUT/'analysis'/f'{key(j)}.json') for j in jobs() if (OUT/'analysis'/f'{key(j)}.json').exists()]
    groups = {}
    for row in rows:
        k = (row['setting'], row['seed'])
        proof = (row['initial_signature'], row['input_signature'])
        if k in groups and groups[k] != proof:
            raise ValueError('Initial state/common input differs across models or delays')
        groups[k] = proof
    for setting in SETTINGS:
        for seed in SEEDS:
            natives = {r['native_signature'] for r in rows if r['setting'] == setting and r['seed'] == seed and r['model'] == 'tokenhsi'}
            if len(natives) > 1:
                raise ValueError('TokenHSI first input/action differs across delays')
    macro = []
    for family in ('enable', 'coupled'):
        names = [s for s in SETTINGS if s.startswith(family)]
        metrics = ('group_sr_percent', 'prep_percent', 'occupancy_percent', 'at_sr_percent') if family == 'enable' else ('group_sr_percent', 'tcoop_percent', 'jerk_m_s3')
        for model in LABELS:
            for delay in DELAYS:
                for seed in SEEDS:
                    selected = [r for r in rows if r['setting'] in names and (r['model'], r['delay'], r['seed']) == (model, delay, seed)]
                    if len(selected) == 3:
                        macro.append(dict(setting=family+'_average', model=model, delay=delay, seed=seed,
                            **{k: None if any(r[k] is None for r in selected) else mean([r[k] for r in selected]) for k in metrics}))
    write(OUT/'summary.json', dict(completed=len(rows) == 270, verified_runs=len(rows), runs=rows, macro_per_seed=macro))
    lines = ['# 지연 대응 · 64환경·seed1/6/7·30초', '', f'검증 완료 {len(rows)}/270조건. 미완료 seed는 결과에 포함하지 않는다. 각 셀은 seed 평균±표본표준편차; n은 지연0/2/4초별 완료 seed 수다.', '']
    for setting in [*SETTINGS, 'enable_average', 'coupled_average']:
        enabling = setting.startswith('enable')
        metrics = ('group_sr_percent', 'prep_percent', 'occupancy_percent') if enabling else ('group_sr_percent', 'tcoop_percent', 'jerk_m_s3')
        titles = ('Group SR(%)', '준비율(%)', '사전점유율(%)') if enabling else ('Group SR(%)', 't_coop(%)', 'Jerk(m/s³)')
        lines += ['## '+setting, '', '| 모델 | n(0/2/4초) | '+' | '.join(titles)+' |', '|---|---|---|---|---|']
        for model, label in LABELS.items():
            selected = [r for r in rows+macro if r['setting'] == setting and r['model'] == model]
            counts = [sum(r['delay'] == d for r in selected) for d in DELAYS]
            cells = []
            for metric in metrics:
                parts = []
                for delay in DELAYS:
                    values = [r[metric] for r in selected if r['delay'] == delay and r[metric] is not None]
                    parts.append('N/A' if not values else f'{np.mean(values):.1f}'+(f'±{np.std(values, ddof=1):.1f}' if len(values)>1 else ''))
                cells.append('→'.join(parts))
            lines.append('| '+label+' | '+'/'.join(map(str, counts))+' | '+' | '.join(cells)+' |')
        lines.append('')
    lines += ['종류별 평균은 각 seed에서 Enabling3종 또는 Coupled2/3/4명을 동일 가중 평균한 뒤 seed 통계를 구한다. 준비율/점유율은30초 내 AT 성공 표본 기준이며 낙상 이후 위치도 포함한다. t_coop은 기존 전체 기록 분자/별도 분모·clip/NaN 제외, Jerk는20cm 출발·원본 차분 mask·0값 제외 규약이다. TokenHSI397D 입력 구조 차이를 유지한다. 재사용35초 기록은30초까지만 재채점한다.', '', '[상태](status.json) · [전체 통계·검증](summary.json) · [실험 manifest](manifest.json)']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--source', type=Path)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--setting')
    parser.add_argument('--model')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--delay', type=int)
    parser.add_argument('--steps', type=int, default=900)
    args = parser.parse_args()
    summarize() if args.summarize else score(args)
