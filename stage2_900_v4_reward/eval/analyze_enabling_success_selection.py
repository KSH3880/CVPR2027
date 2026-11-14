import csv
import json
from pathlib import Path
import statistics

import numpy as np
import torch

from reaggregate_enabling_post_at_time import sha, SEEDS, SETTINGS

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT/'output_etc/check/v4_2000_tables12_radius_seeds167_20261112'
POST = SUITE/'post_at_time'
OUTPUT = SUITE/'success_selection_tokenhsi_vs_v4'
COHORTS = ('all', 'tokenhsi_success', 'ours_success', 'both_success', 'tokenhsi_only', 'ours_only')


def stats(values):
    values = np.asarray(values)
    if not len(values):
        return dict(count=0, mean=None, median=None, q25=None, q75=None)
    return dict(count=len(values), mean=float(values.mean()), median=float(np.median(values)),
                q25=float(np.quantile(values, .25)), q75=float(np.quantile(values, .75)))


def aggregate(records, cohort, metric):
    selected = [r['cohorts'][cohort] for r in records]
    values = [statistics.mean(next(r['cohorts'][cohort]['metrics'][metric]['mean'] for r in records
        if r['seed'] == seed and r['setting'] == setting) for setting in SETTINGS) for seed in SEEDS]
    return dict(mean=statistics.mean(values), std=statistics.stdev(values), seeds=values,
        trials=sum(r['trials'] for r in selected))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    jobs = json.loads((POST/'manifest.json').read_text())['jobs']
    records, csv_rows, hashes = [], [], {}
    for setting in SETTINGS:
        for seed in SEEDS:
            sources = {model: Path(next(j['source'] for j in jobs
                if (j['model'], j['setting'], j['seed']) == (model, setting, seed)))
                for model in ('tokenhsi', 'ours')}
            initial, conditions, trials = {}, {}, {}
            for model, source in sources.items():
                trial_path = POST/f'{model}_{setting}_seed{seed}'/'trials.npz'
                for path in (source/'initial.pt', source/'new_conditions.pt', trial_path):
                    hashes[str(path)] = sha(path)
                initial[model] = torch.load(source/'initial.pt', map_location='cpu', weights_only=False)
                conditions[model] = torch.load(source/'new_conditions.pt', map_location='cpu', weights_only=False)
                with np.load(trial_path) as data:
                    trials[model] = {key: data[key].copy() for key in data.files}
            for key in ('packet', 'predecessor', 'roots', 'goals', 'sizes'):
                if not torch.equal(initial['tokenhsi'][key], initial['ours'][key]):
                    raise ValueError('Unmatched initial scene: '+key)
            for key in ('distances', 'shared_sizes'):
                if not torch.equal(conditions['tokenhsi'][key], conditions['ours'][key]):
                    raise ValueError('Unmatched initial condition: '+key)
            data = initial['ours']
            d = conditions['ours']['distances'].double().numpy()
            shared = conditions['ours']['shared_sizes'].double().numpy()
            follower = (data['predecessor'] >= 0).long().argmax(-1)
            index = torch.arange(512)
            leader = data['predecessor'][index, follower]
            payload = data['packet'][index, leader, 3]-2
            box = data['boxes'][index, payload, :2].double().numpy()
            root = data['roots'][index, leader, :2].double().numpy()
            goal = data['goals'][index, leader, :2].double().numpy()
            actual = np.stack((np.linalg.norm(box-root, axis=-1), np.linalg.norm(goal-box, axis=-1)), -1)
            if not np.allclose(actual, d[:, :2], atol=.002, rtol=0):
                raise ValueError('Recorded and reconstructed predecessor distances differ')
            follow_root = data['roots'][index, follower, :2].double().numpy()
            if not np.allclose(np.linalg.norm(follow_root-goal, axis=-1), d[:, 2], atol=.002, rtol=0):
                raise ValueError('Recorded and reconstructed follower distance differ')
            if setting == 'enable_ontop':
                follow_payload = data['packet'][index, follower, 3]-2
                payload_xy = data['boxes'][index, follow_payload, :2].double().numpy()
                actual_top = np.stack((np.linalg.norm(follow_root-payload_xy, axis=-1),
                                       np.linalg.norm(payload_xy-goal, axis=-1)), -1)
                if not np.allclose(actual_top, d[:, 3:5], atol=.002, rtol=0):
                    raise ValueError('Recorded and reconstructed ON_TOP distances differ')
            approach, transport = box-root, goal-box
            cosine = (approach*transport).sum(-1)/(np.linalg.norm(approach, axis=-1)*np.linalg.norm(transport, axis=-1))
            features = dict(at_approach_m=d[:, 0], at_transport_m=d[:, 1], at_route_m=d[:, :2].sum(-1),
                follower_goal_m=d[:, 2], follower_route_m=d[:, 3:5].sum(-1) if setting == 'enable_ontop' else d[:, 2],
                predecessor_turn_degrees=np.degrees(np.arccos(np.clip(cosine, -1, 1))),
                shared_x_m=shared[:, 0], shared_y_m=shared[:, 1], shared_z_m=shared[:, 2],
                shared_volume_m3=shared.prod(-1))
            features['follower_minus_at_route_m'] = features['follower_route_m']-features['at_route_m']
            features['follower_longer_than_at_percent'] = 100*(features['follower_route_m'] > features['at_route_m'])
            success = {model: value['first_group'] >= 1 for model, value in trials.items()}
            masks = dict(all=np.ones(512, dtype=bool), tokenhsi_success=success['tokenhsi'],
                ours_success=success['ours'], both_success=success['tokenhsi'] & success['ours'],
                tokenhsi_only=success['tokenhsi'] & ~success['ours'], ours_only=success['ours'] & ~success['tokenhsi'])
            times = {}
            for model in trials:
                times[model+'_completion_s'] = trials[model]['first_group']/30
                times[model+'_at_s'] = trials[model]['first_at']/30
                times[model+'_post_at_s'] = trials[model]['post_at_time_s']
            cohorts = {}
            for name, mask in masks.items():
                metrics = {key: stats(values[mask]) for key, values in features.items()}
                for model in trials:
                    if name == 'both_success' or name == model+'_success':
                        metrics.update({key: stats(values[mask]) for key, values in times.items() if key.startswith(model+'_')})
                cohorts[name] = dict(trials=int(mask.sum()), metrics=metrics)
            records.append(dict(setting=setting, seed=seed, initial_conditions_exact=True, cohorts=cohorts))
            for env in range(512):
                row = dict(setting=setting, seed=seed, scene=env,
                    tokenhsi_success=bool(success['tokenhsi'][env]), ours_success=bool(success['ours'][env]))
                row.update({key: float(value[env]) for key, value in features.items()})
                row.update({key: float(value[env]) if success[key.split('_')[0]][env] else None for key, value in times.items()})
                csv_rows.append(row)
    metrics = list(features)
    overview = {name: dict(trials=sum(r['cohorts'][name]['trials'] for r in records),
        metrics={key: aggregate(records, name, key) for key in metrics})
        for name in COHORTS if name not in ('tokenhsi_only', 'ours_only')}
    common_times = {model: {kind: aggregate(records, 'both_success', model+'_'+kind+'_s')
        for kind in ('at', 'completion', 'post_at')} for model in trials}
    own_times = {model: {kind: aggregate(records, model+'_success', model+'_'+kind+'_s')
        for kind in ('at', 'completion', 'post_at')} for model in trials}
    bins = []
    for at_long in (False, True):
        for follower_long in (False, True):
            selected = [row for row in csv_rows if (row['at_route_m'] >= 6) == at_long
                        and (row['follower_route_m'] >= 6) == follower_long]
            bins.append(dict(at_route='6–10m' if at_long else '2–6m',
                follower_route='6–10m' if follower_long else '2–6m', trials=len(selected),
                tokenhsi_success_rate_percent=100*statistics.mean(row['tokenhsi_success'] for row in selected),
                ours_success_rate_percent=100*statistics.mean(row['ours_success'] for row in selected)))
    if any(sha(p) != digest for p, digest in hashes.items()):
        raise ValueError('Input records changed')
    summary = dict(seeds=SEEDS, envs=512, settings=SETTINGS, overview=overview,
        own_success_times=own_times, both_success_times=common_times, runs=records, route_bins_pooled=bins,
        aggregation='per setting/seed conditional mean -> equal mean of three actions -> seed mean/sample std',
        distance_definition='initial XY straight-segment lengths, not actual trajectory distance', source_sha256=hashes)
    (OUTPUT/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    with (OUTPUT/'trials.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    labels = dict(all='전체 초기 장면', tokenhsi_success='TokenHSI 성공', ours_success='Ours V4 성공', both_success='두 모델 공통 성공')
    def fmt(record):
        return f"{record['mean']:.2f}±{record['std']:.2f}"
    lines = ['# Enabling 성공 표본·초기 거리·동일 성공 장면 비교', '',
        '512환경 × seed1/6/7 × AT→SIT/CLIMB/ON_TOP. 초기 상태/거리/공유 상자 크기가 두 모델에서 exact 일치함을 확인했다. CPU 후처리이며 GPU 재평가 없음.', '',
        '거리 단위m. 초기 XY 직선구간합이며 실제 움직인 경로 길이가 아니다. 모든 평균은 행동별 동일 가중→seed 평균±표본std.', '',
        '| 성공 표본 | 장면 수 | AT 접근 | AT 운반 | AT 합 | 후행 수행 경로 |',
        '|---|---:|---:|---:|---:|---:|']
    for name, row in overview.items():
        lines.append('| '+labels[name]+' | '+str(row['trials'])+' | '+' | '.join(fmt(row['metrics'][key])
            for key in ('at_approach_m', 'at_transport_m', 'at_route_m', 'follower_route_m'))+' |')
    lines += ['', '## 각자 성공한 장면과 공통 성공 장면의 시간', '',
        '| 비교 표본 | 모델 | AT Time(s) | Completion Time(s) | Post-AT Time(s) |', '|---|---|---:|---:|---:|']
    for name, values in (('각자 성공', own_times), ('공통 성공', common_times)):
        for model, row in values.items():
            lines.append('| '+name+' | '+model+' | '+' | '.join(fmt(row[k]) for k in ('at', 'completion', 'post_at'))+' |')
    lines += ['', '## 행동별 성공 표본 거리', '',
        '| 후행 | 표본 | 성공 수 | AT 합(m) | 후행 경로(m) |', '|---|---|---:|---:|---:|']
    for setting in SETTINGS:
        selected = [r for r in records if r['setting'] == setting]
        for name in ('all', 'tokenhsi_success', 'ours_success', 'both_success'):
            values = [r['cohorts'][name] for r in selected]
            lines.append('| '+setting+' | '+labels[name]+' | '+str(sum(r['trials'] for r in values))+' | '+
                ' | '.join(f"{statistics.mean(r['metrics'][key]['mean'] for r in values):.2f}" for key in ('at_route_m', 'follower_route_m'))+' |')
    lines += ['', '## 선행·후행 거리 구간별 성공률', '',
        '| AT 경로 | 후행 경로 | 전체 장면 수 | TokenHSI SR(%) | Ours V4 SR(%) |', '|---|---|---:|---:|---:|']
    for row in bins:
        lines.append(f"| {row['at_route']} | {row['follower_route']} | {row['trials']} | {row['tokenhsi_success_rate_percent']:.2f} | {row['ours_success_rate_percent']:.2f} |")
    lines += ['', '거리 구간 표는 해당 장면의 성공/전체 수로 직접 집계한 pooled 비율이며 행동별 동일 가중 평균과 구분한다.6m 경계는 긴 구간에 포함한다.', '',
        '성공 표본/거리가 달라지는지와 동일 장면의 시간을 분리해 확인하는 관찰 분석이다. 공통 성공 표본도 조건부 표본이며 전체 장면의 보편적 속도 순위나 거리만의 인과 효과를 증명하지 않는다.',
        '[원시 장면별 거리·성공·시간](trials.csv) · [전체 통계·상자 크기·회전각](summary.json)']
    (OUTPUT/'report.md').write_text('\n'.join(lines)+'\n')
    (OUTPUT/'verification.json').write_text(json.dumps(dict(completed=True, paired_scenes=4608,
        initial_states_exact=True, recorded_distances_reconstructed=True, input_hashes_unchanged=True,
        gpu_executions=0), indent=2)+'\n')
    print(OUTPUT/'report.md')


if __name__ == '__main__':
    main()
