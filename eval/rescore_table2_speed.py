import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tokenhsi'))
from table2_metrics import surface_distance


def speed_window_counts(speed, nearby, contact, first_success, min_speed, required_frames):
    steps, trials = speed.shape
    end = np.where(first_success >= 1, first_success, steps)
    good = speed >= min_speed
    if nearby is not None:
        good &= nearby
    prefix = np.concatenate([np.zeros((1, trials), dtype=np.int64), good.cumsum(axis=0)])
    hits = prefix[required_frames:] - prefix[:-required_frames] == required_frames
    hits &= np.arange(required_frames, steps+1)[:, None] <= end
    found = hits.any(axis=0)
    start = np.where(found, hits.argmax(axis=0), -1)
    window = np.where(found, end-start, 0)
    contact_prefix = np.concatenate([np.zeros((1, trials), dtype=np.int64), contact.cumsum(axis=0)])
    ids = np.arange(trials)
    counts = np.where(found, contact_prefix[end, ids]-contact_prefix[start.clip(0), ids], 0)
    return dict(start=np.where(found, start+1, -1), window_frames=window, contact_frames=counts)


def stats(values):
    return dict(mean=statistics.mean(values), std=statistics.stdev(values) if len(values)>1 else 0., seeds=values)


def summarize_runs(runs, seeds, variants):
    results = {}
    for model in ('stage1', 'stage2'):
        results[model] = {}
        for key in variants:
            settings = {}
            for people in (2, 3, 4):
                rows = [next(row for row in runs if row['model']==model and row['people']==people and row['seed']==seed)
                        ['variants'][key] for seed in seeds]
                settings[str(people)] = {
                    metric: stats([row[metric] for row in rows])
                    for metric in ('cooperative_time_ratio_percent', 'started_percent')}
            settings['average'] = {
                metric: stats([statistics.mean(settings[str(k)][metric]['seeds'][i] for k in (2,3,4))
                               for i in range(len(seeds))])
                for metric in ('cooperative_time_ratio_percent', 'started_percent')}
            results[model][key] = settings
    return results


def rescore(args):
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    variants = {f'{speed:g}mps_{duration:g}s': dict(speed_mps=speed, duration_s=duration)
                for speed in args.speeds for duration in args.durations}
    runs = []
    for model in ('stage1', 'stage2'):
        for people in (2,3,4):
            for seed in args.seeds:
                path = args.source/model/f'coupled_{people}'/f'general_seed{seed}'
                protected = [path/name for name in ('table2.json', 'table2_trials.npz', 'result.json')]
                before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in protected]
                initial = torch.load(path/'initial.pt', map_location='cpu', weights_only=False)
                recorded = json.loads((path/'table2.json').read_text())
                dt = recorded['control_dt']
                with np.load(path/'cooperation.npz') as data:
                    boxes, hands, sizes = (data[k] for k in ('physical_boxes','hands','physical_sizes'))
                with np.load(path/'trace.npz') as data:
                    valid = data['valid']
                with np.load(path/'table2_trials.npz') as data:
                    first_success = data['first_success']
                    old_start = data['start']
                    old_contacts = data['contact_frames'][:,2]
                center = boxes[...,:3].astype(np.float64)
                previous = np.concatenate([initial['shared_centers'][:,0,:2].numpy()[None], center[:-1,:,:2]])
                speed = np.linalg.norm(center[:,:,:2]-previous, axis=-1)/dt
                goal = initial['shared_center_goals'][:,0].numpy().astype(np.float64)
                success = (np.linalg.norm(center-goal, axis=-1)<=.2) & valid.any(-1)
                computed_first = np.where(success.any(0), success.argmax(0)+1, -1)
                if not np.array_equal(computed_first, first_success):
                    raise ValueError(f'First success changed: {path}')
                nearby = np.zeros(speed.shape, dtype=bool) if args.start_hand_cm is not None else None
                contact = np.zeros(speed.shape, dtype=bool)
                size_tensor = torch.from_numpy(sizes).double()
                for frame in range(len(boxes)):
                    distance = surface_distance(torch.from_numpy(hands[frame]).double(),
                                                torch.from_numpy(boxes[frame]).double(), size_tensor).numpy()
                    if nearby is not None:
                        nearby[frame] = ((distance < args.start_hand_cm/100) & valid[frame,...,None]).any(-1).any(-1)
                    contact[frame] = ((distance < args.contact_cm/100) & valid[frame,...,None]).all(-1).all(-1)
                if args.contact_cm == 12:
                    end = np.where(first_success>=1, first_success, len(boxes))
                    prefix = np.concatenate([np.zeros((1, len(end)), dtype=np.int64), contact.cumsum(axis=0)])
                    ids = np.arange(len(end))
                    original_counts = np.where((old_start>=1)&(old_start<=end),
                        prefix[end,ids]-prefix[(old_start-1).clip(0),ids], 0)
                    if not np.array_equal(original_counts, old_contacts):
                        raise ValueError(f'Original 12cm contact counts changed: {path}')
                row = dict(model=model, people=people, seed=seed, source=str(path), variants={},
                           first_success_frames_exact=True, source_results_unchanged=True,
                           original_12cm_contact_counts_exact=args.contact_cm==12)
                trial_arrays = dict(first_success=first_success)
                for key, setting in variants.items():
                    frames = max(1, math.ceil(setting['duration_s']/dt-1e-9))
                    if frames > len(speed):
                        raise ValueError('Duration exceeds episode length')
                    arrays = speed_window_counts(speed, nearby, contact, first_success, setting['speed_mps'], frames)
                    ratio = arrays['contact_frames']/arrays['window_frames'].clip(1)
                    row['variants'][key] = dict(**setting, required_frames=frames, control_dt=dt,
                        cooperative_time_ratio_percent=float(ratio.mean()*100),
                        started_percent=float((arrays['start']>=0).mean()*100),
                        no_start_trials=int((arrays['start']<0).sum()))
                    for name, values in arrays.items():
                        trial_arrays[key+'_'+name] = values
                if before != [hashlib.sha256(p.read_bytes()).hexdigest() for p in protected]:
                    raise ValueError(f'Source modified: {path}')
                name = f'{model}_coupled_{people}_seed{seed}'
                np.savez_compressed(args.output/(name+'_trials.npz'), **trial_arrays)
                (args.output/(name+'.json')).write_text(json.dumps(row, indent=2))
                runs.append(row)
                print(name, 'completed', flush=True)
    results = summarize_runs(runs, args.seeds, variants)
    data = dict(seeds=args.seeds, variants=variants, results=results, runs=runs,
                contact_cm=args.contact_cm, start_hand_cm=args.start_hand_cm,
                start_rule=('XY finite-difference speed'+(' and any valid hand proximity' if args.start_hand_cm is not None else '')+' in every consecutive frame; '
                           'backdate to first qualifying frame; confirmation must occur by first success; no displacement gate'),
                end_rule='first shared-box 3D goal success inclusive; failure through episode end',
                failures_included=True, no_start_percent=0., contact_force_used=False)
    (args.output/'summary.json').write_text(json.dumps(data, indent=2))
    def fmt(value):
        return f"{value['mean']:.2f}±{value['std']:.2f}"
    lines = ['# t_coop 속도·지속 시간 시작 조건 비교', '',
             '512환경 × seed '+','.join(map(str,args.seeds))+'. 단위 %, 평균±표본표준편차.',
             ('시작에 손 접근 조건 없음.' if args.start_hand_cm is None else
              f'연속 구간 내 매 프레임 한 손 이상 표면{args.start_hand_cm:g}cm 미만.')+f' 전원 양손 접촉{args.contact_cm:g}cm 미만.',
             '시작은 연속 구간 첫 프레임으로 소급하며 최초 성공까지 확인된 구간만 인정한다. 변위20cm 조건은 사용하지 않는다.',
             '실패 포함·미시작0%. 최초 성공 프레임 exact·원본 결과 hash 불변. 기본 프로토콜 변경 없음.', '',
             '| XY 속도(m/s) | 지속(s) | Stage1 t_coop | Stage2 t_coop | Stage1 시작률 | Stage2 시작률 |',
             '| ---: | ---: | ---: | ---: | ---: | ---: |']
    for key, setting in variants.items():
        a,b = (results[model][key]['average'] for model in ('stage1','stage2'))
        lines.append(f"| {setting['speed_mps']:g} | {setting['duration_s']:g} | {fmt(a['cooperative_time_ratio_percent'])} | "
                     f"{fmt(b['cooperative_time_ratio_percent'])} | {fmt(a['started_percent'])} | {fmt(b['started_percent'])} |")
    baseline = json.loads((args.source/'summary.json').read_text())['results']
    lines += ['', f"기존20cm 시작·접촉12cm: Stage1 {fmt(baseline['stage1']['coupled_average']['cooperative_time_ratio_percent']['12'])}, "
              f"Stage2 {fmt(baseline['stage2']['coupled_average']['cooperative_time_ratio_percent']['12'])}.", '',
              '## 인원별 결과', '', '| 모델 | 속도 | 지속 | 2H | 3H | 4H |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    for model in ('stage1','stage2'):
        for key, setting in variants.items():
            values = [fmt(results[model][key][str(k)]['cooperative_time_ratio_percent']) for k in (2,3,4)]
            lines.append(f"| {model} | {setting['speed_mps']:g} | {setting['duration_s']:g} | "+' | '.join(values)+' |')
    lines += ['', '## Seed별 평균', '', '| 모델 | 속도 | 지속 | seed | t_coop | 시작률 |', '| --- | ---: | ---: | --- | ---: | ---: |']
    for model in ('stage1','stage2'):
        for key, setting in variants.items():
            item = results[model][key]['average']
            for i, seed in enumerate(args.seeds):
                lines.append(f"| {model} | {setting['speed_mps']:g} | {setting['duration_s']:g} | {seed} | "
                             f"{item['cooperative_time_ratio_percent']['seeds'][i]:.2f} | {item['started_percent']['seeds'][i]:.2f} |")
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:20]), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('output_etc/check/table2_20261109'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--speeds', nargs='+', type=float, default=[.05,.1,.2])
    parser.add_argument('--durations', nargs='+', type=float, default=[.1,.3,.5])
    parser.add_argument('--seeds', nargs='+', type=int, default=[1,3,6])
    start_hand = parser.add_mutually_exclusive_group()
    start_hand.add_argument('--start-hand-cm', type=float, default=16)
    start_hand.add_argument('--no-start-hand', dest='start_hand_cm', action='store_const', const=None)
    parser.add_argument('--contact-cm', type=float, default=12)
    args = parser.parse_args()
    distances = [args.contact_cm]+([] if args.start_hand_cm is None else [args.start_hand_cm])
    if any(value<=0 for value in args.speeds+args.durations+distances):
        parser.error('Speeds, durations and distances must be positive')
    check = Path(__file__).resolve().parents[1]/'output_etc/check'
    if check not in args.output.resolve().parents:
        parser.error('Output must be under output_etc/check')
    rescore(args)
