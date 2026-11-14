import csv
import json
import os
from pathlib import Path
import statistics

import numpy as np
import torch

from reaggregate_enabling_post_at_time import SEEDS, SETTINGS, sha

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT/'output_etc/check/v4_2000_tables12_radius_seeds167_20261112'
POST = SUITE/'post_at_time'
OUTPUT = SUITE/'movement_speed_tokenhsi_vs_v4'
MODELS = dict(tokenhsi='TokenHSI 40.5k', ours='Ours V4 2k')
DT = 1/30
FIELDS = ('speed', 'distance', 'duration', 'moving_speed_020', 'moving_fraction_020',
          'moving_speed_050', 'moving_fraction_050')
LABELS = dict(leader_approach_r100='선행 사람 접근 ≤1m', follower_approach_r100='후행 사람 접근 ≤1m',
    leader_carry='선행 상자 운반', follower_carry='후행 payload 운반 → 최초 후행 성공',
    leader_full='선행 사람 전체 → 최초 AT 성공', follower_full='후행 사람 전체 → 그룹 성공',
    follower_before_at='후행 사람 AT 완료 전', follower_after_at='후행 사람 AT 완료 후',
    leader_until_group='선행 사람 전체 → 그룹 성공', leader_box_full='선행 상자 전체 → 최초 AT 성공',
    follower_payload_full='후행 payload 전체 → 최초 후행 성공',
    follower_carry_until_group='후행 payload 운반 → 그룹 성공')
for role, label in (('leader', '선행'), ('follower', '후행')):
    for suffix, radius in (('050', .5), ('150', 1.5)):
        LABELS[f'{role}_approach_r{suffix}'] = f'{label} 사람 접근 ≤{radius}m'


def describe(values):
    values = [v for v in values if v is not None]
    return dict(count=len(values), mean=float(np.mean(values)) if values else None,
        median=float(np.median(values)) if values else None)


def seed_stats(values):
    complete = None not in values
    return dict(mean=statistics.mean(values) if complete else None,
        std=statistics.stdev(values) if complete else None, seeds=dict(zip(SEEDS, values)))


def window(path, start, end):
    if start is None or end <= start:
        return dict({key: None for key in FIELDS}, start=start, end=end,
            missing_reason='zero_duration' if start is not None else 'no_start')
    increments = np.linalg.norm(np.diff(path[start:end+1], axis=0), axis=-1)
    speeds = increments/DT
    distance = float(increments.sum())
    duration = (end-start)*DT
    value = dict(speed=distance/duration, distance=distance, duration=duration, start=start, end=end)
    for suffix, threshold in (('020', .2), ('050', .5)):
        moving = speeds > threshold
        value['moving_speed_'+suffix] = float(speeds[moving].mean()) if moving.any() else None
        value['moving_fraction_'+suffix] = float(100*moving.mean())
    return value


def approach(path, target, end, radius):
    distance = np.linalg.norm(path[:end+1]-target[:end+1], axis=-1)
    hits = np.flatnonzero(distance <= radius)
    value = window(path, 0, int(hits[0])) if len(hits) else window(path, None, end)
    value['missing_reason'] = ('initially_near' if len(hits) and hits[0] == 0 else
                               'never_reached' if not len(hits) else None)
    return value


def carry(path, end):
    hits = np.flatnonzero(np.linalg.norm(path[:end+1]-path[0], axis=-1) >= .2)
    value = window(path, int(hits[0]) if len(hits) else None, end)
    value['missing_reason'] = ('never_started' if not len(hits) else
                               'zero_duration' if hits[0] == end else None)
    return value


def collect(jobs):
    rows, hashes = [], {}
    for setting in SETTINGS:
        for seed in SEEDS:
            loaded = {}
            for model in MODELS:
                job = next(j for j in jobs if (j['model'], j['setting'], j['seed']) == (model, setting, seed))
                source, trials = Path(job['source']), POST/job['name']/'trials.npz'
                result = json.loads((POST/job['name']/'result.json').read_text())
                for path in (source/'trace.npz', source/'initial.pt', trials):
                    hashes[str(path)] = sha(path)
                    expected = result['source_sha256'].get(str(path))
                    if expected and hashes[str(path)] != expected:
                        raise ValueError('Raw input differs from verified Post-AT record')
                initial = torch.load(source/'initial.pt', map_location='cpu', weights_only=False)
                with np.load(trials) as data:
                    events = {k: data[k].copy() for k in ('first_at', 'first_follower', 'first_group')}
                with np.load(source/'trace.npz') as data:
                    trace = {k: data[k].copy() for k in ('roots', 'boxes', 'valid', 'packet', 'predecessor', 'goals')}
                if trace['valid'].shape != (900, 512, 2):
                    raise ValueError('Unexpected trace dimensions')
                for key in ('packet', 'predecessor', 'goals'):
                    np.testing.assert_array_equal(initial[key].numpy(), trace[key])
                loaded[model] = initial, events, trace
            for key in ('roots', 'packet', 'predecessor', 'goals', 'sizes'):
                np.testing.assert_array_equal(loaded['tokenhsi'][0][key], loaded['ours'][0][key])
            successes = {m: loaded[m][1]['first_group'] >= 1 for m in MODELS}
            for model, (initial, events, trace) in loaded.items():
                for env in np.flatnonzero(successes[model]):
                    follower = int((trace['predecessor'][env] >= 0).argmax())
                    leader = int(trace['predecessor'][env, follower])
                    at, follow, group = (int(events[k][env]) for k in ('first_at', 'first_follower', 'first_group'))
                    if not (0 < at <= group <= 900 and 0 < follow <= group
                            and trace['valid'][:group, env].all()):
                        raise ValueError('Successful trial has an invalid event/window')
                    people = np.concatenate((initial['roots'][env:env+1, :, :2].numpy(),
                                             trace['roots'][:, env, :, :2])).astype(np.float64)
                    objects = np.concatenate((initial['boxes'][env:env+1, :, :2].numpy(),
                                              trace['boxes'][:, env, :, :2])).astype(np.float64)
                    if not (np.isfinite(people).all() and np.isfinite(objects).all()):
                        raise ValueError('Non-finite position')
                    shared = int(trace['packet'][env, leader, 3])-2
                    a, b, box = people[:, leader], people[:, follower], objects[:, shared]
                    goal = np.broadcast_to(initial['goals'][env, leader, :2].numpy(), b.shape)
                    payload = objects[:, int(trace['packet'][env, follower, 3])-2] if setting == 'enable_ontop' else None
                    target = payload if payload is not None else goal
                    metrics = dict(leader_full=window(a, 0, at), follower_full=window(b, 0, group),
                        leader_until_group=window(a, 0, group), leader_box_full=window(box, 0, at),
                        follower_before_at=window(b, 0, at), follower_after_at=window(b, at, group),
                        leader_carry=carry(box, at))
                    for suffix, radius in (('050', .5), ('100', 1.), ('150', 1.5)):
                        metrics['leader_approach_r'+suffix] = approach(a, box, at, radius)
                        metrics['follower_approach_r'+suffix] = approach(b, target, group, radius)
                    if payload is not None:
                        metrics.update(follower_carry=carry(payload, follow),
                            follower_carry_until_group=carry(payload, group),
                            follower_payload_full=window(payload, 0, follow))
                    rows.append(dict(model=model, setting=setting, seed=seed, scene=int(env),
                        both_success=bool(successes['tokenhsi'][env] and successes['ours'][env]),
                        first_at=at, first_follower=follow, first_group=group, metrics=metrics))
            print(f'{setting} seed{seed}: processed both models', flush=True)
    return rows, hashes


def summarize(rows, cohort, model, metric, setting=None):
    settings = (setting,) if setting else (('enable_ontop',) if metric.startswith('follower_carry')
        or metric == 'follower_payload_full' else SETTINGS)
    selected = [r for r in rows if r['model'] == model and r['setting'] in settings
        and (cohort == 'own_success' or r['both_success'])]
    cells = []
    for task in settings:
        for seed in SEEDS:
            subset = [r['metrics'][metric] for r in selected if r['seed'] == seed and r['setting'] == task]
            cells.append(dict(setting=task, seed=seed, trials=len(subset),
                **{key: describe([r[key] for r in subset]) for key in FIELDS},
                missing_reasons={reason: sum(r.get('missing_reason') == reason for r in subset)
                    for reason in ('initially_near', 'never_reached', 'never_started', 'zero_duration')}))
    aggregate = {}
    for key in FIELDS:
        values = []
        for seed in SEEDS:
            means = [r[key]['mean'] for r in cells if r['seed'] == seed]
            values.append(statistics.mean(means) if None not in means else None)
        aggregate[key] = seed_stats(values)
    aggregate.update(count=sum(c['speed']['count'] for c in cells), trials=len(selected), cells=cells,
        pooled={key: describe([r['metrics'][metric][key] for r in selected]) for key in FIELDS})
    return aggregate


def fmt(value):
    return 'N/A' if value['mean'] is None else f"{value['mean']:.3f}±{value['std']:.3f}"


def report(summary):
    lines = ['# Enabling 이동속력: TokenHSI / Ours V4', '',
        '512환경 × seed1/6/7 × 세 후행 행동. 기존18개 trace의 CPU 후처리이며 GPU 재평가 없음.', '',
        '실제 수평 궤적 길이÷구간 시간(m/s). 최초 초기 위치와 step1 사이부터 포함하고 정지 프레임을 유지한다. 목표 방향 성분·3D 속력·물리 substep 속력이 아니다.', '',
        '성공은 기존 유효 프레임의 그룹 동시 성공이다. 공통 성공402장면을 주 비교로, 각자 성공(TokenHSI479/Ours3903)을 보조로 제시한다. Trial별 속력 평균→행동 동일 가중→seed 평균±표본std. ON_TOP payload는 해당 행동만 평균한다.', '',
        '접근은 시작부터 대상의 수평 중심거리≤1m 최초 진입까지. 선행은 현재 공유 상자, 후행 SIT/CLIMB는 최종 목표XY, ON_TOP은 현재 payload가 대상이다. 시작부터 근처인 경우와 미진입은 제외한다.', '',
        '운반은 초기 물체 중심 대비 수평 직선 변위≥20cm 최초 시점부터 최초 개인 성공까지. 문턱 도달 이전 구간은 제외하며 이후 정지는 포함한다. 후행 최초 성공은 그룹 성공보다 빠를 수 있어 그룹 성공까지 버전도 따로 기록한다.', '',
        '사람 전체는 선행 최초 AT 성공/후행 최초 그룹 성공까지. 후행 AT 전후는 선행 최초 AT 성공으로 나눈다. 0길이 구간은 제외하고 N/A·표본 수를 기록하며 0m/s로 대체하지 않는다.', '']
    primary = ('leader_approach_r100', 'follower_approach_r100', 'leader_carry', 'follower_carry',
               'leader_full', 'follower_full', 'follower_before_at', 'follower_after_at')
    for cohort, title in (('both_success', '동일 장면에서 둘 다 성공'), ('own_success', '각 모델 전체 성공')):
        lines += ['', f'## {title}', '', '| 구간 | TokenHSI (m/s) | Ours V4 (m/s) | 유효 표본 Token/Ours |', '|---|---:|---:|---:|']
        for metric in primary:
            a, b = (summary['cohorts'][cohort][m]['average'][metric] for m in MODELS)
            lines.append(f"| {LABELS[metric]} | {fmt(a['speed'])} | {fmt(b['speed'])} | {a['count']}/{b['count']} |")
    lines += ['', '## 공통 성공: 행동별 속력', '', '| 후행 | 구간 | TokenHSI (m/s) | Ours V4 (m/s) | 유효 표본 Token/Ours |', '|---|---|---:|---:|---:|']
    for setting in SETTINGS:
        for metric in primary:
            if metric == 'follower_carry' and setting != 'enable_ontop':
                continue
            a, b = (summary['cohorts']['both_success'][m]['by_setting'][setting][metric] for m in MODELS)
            lines.append(f"| {setting} | {LABELS[metric]} | {fmt(a['speed'])} | {fmt(b['speed'])} | {a['count']}/{b['count']} |")
    lines += ['', '## 공통 성공: 접근 문턱 민감도', '', '| 구간 | TokenHSI (m/s) | Ours V4 (m/s) | 유효 표본 Token/Ours |', '|---|---:|---:|---:|']
    for role in ('leader', 'follower'):
        for suffix in ('050', '100', '150'):
            metric = role+'_approach_r'+suffix
            a, b = (summary['cohorts']['both_success'][m]['average'][metric] for m in MODELS)
            lines.append(f"| {LABELS[metric]} | {fmt(a['speed'])} | {fmt(b['speed'])} | {a['count']}/{b['count']} |")
    lines += ['', '## 공통 성공: 구간 길이·시간', '', '| 구간 | Token 거리(m) | Ours 거리(m) | Token 시간(s) | Ours 시간(s) |', '|---|---:|---:|---:|---:|']
    for metric in primary:
        a, b = (summary['cohorts']['both_success'][m]['average'][metric] for m in MODELS)
        lines.append(f"| {LABELS[metric]} | {fmt(a['distance'])} | {fmt(b['distance'])} | {fmt(a['duration'])} | {fmt(b['duration'])} |")
    lines += ['', '## 공통 성공: 이동 프레임만의 속력', '',
        '보조 진단: 수평 속력>0.2 또는0.5m/s인 프레임만의 평균(m/s)과 해당 프레임의 구간 내 비율(%). 문턱 미만 프레임은 주 평균에는 유지된다. 어느 프레임도 문턱을 넘지 않은 trial의 조건부 속력은 N/A다. 문턱은 논문 확정값이 아니다.', '',
        '| 구간 | 문턱(m/s) | Token 이동 속력 | Ours 이동 속력 | Token 이동 비율(%) | Ours 이동 비율(%) |',
        '|---|---:|---:|---:|---:|---:|']
    for metric in primary:
        a, b = (summary['cohorts']['both_success'][m]['average'][metric] for m in MODELS)
        for suffix, threshold in (('020', .2), ('050', .5)):
            lines.append(f"| {LABELS[metric]} | {threshold} | {fmt(a['moving_speed_'+suffix])} | {fmt(b['moving_speed_'+suffix])} | {fmt(a['moving_fraction_'+suffix])} | {fmt(b['moving_fraction_'+suffix])} |")
    lines += ['', '평균 거리÷평균 시간과 trial별 속력 평균은 다르다. 높은 속력에는 우회·흔들림도 포함되므로 완료 시간이나 보상 설계의 인과적 효과와 동일시하지 않는다.', '',
        '[전체·seed별 수치/제외 사유](summary.json) · [장면·구간별 원시 수치](trials.csv) · [검증](verification.json)']
    (OUTPUT/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only: set CUDA_VISIBLE_DEVICES=')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    jobs = json.loads((POST/'manifest.json').read_text())['jobs']
    rows, hashes = collect(jobs)
    counts = {m: sum(r['model'] == m for r in rows) for m in MODELS}
    common = {m: sum(r['model'] == m and r['both_success'] for r in rows) for m in MODELS}
    if counts != dict(tokenhsi=479, ours=3903) or common != dict(tokenhsi=402, ours=402):
        raise ValueError('Success cohorts differ from previous paired analysis')
    cohorts = {}
    for cohort in ('both_success', 'own_success'):
        cohorts[cohort] = {m: dict(average={k: summarize(rows, cohort, m, k) for k in LABELS},
            by_setting={s: {k: summarize(rows, cohort, m, k, s) for k in LABELS
                if s == 'enable_ontop' or (not k.startswith('follower_carry') and k != 'follower_payload_full')}
                for s in SETTINGS}) for m in MODELS}
    summary = dict(models=MODELS, seeds=SEEDS, time_step_s=DT, own_success_counts=counts,
        both_success_counts=common, definitions=LABELS, source_sha256=hashes, cohorts=cohorts)
    (OUTPUT/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    records = []
    for row in rows:
        for metric, value in row['metrics'].items():
            if value['speed'] is not None and not np.isclose(value['speed']*value['duration'], value['distance'], atol=1e-10):
                raise ValueError('Speed/distance/time identity failed')
            records.append(dict({k: v for k, v in row.items() if k != 'metrics'}, metric=metric, **value))
    fields = list(dict.fromkeys(k for row in records for k in row))
    with (OUTPUT/'trials.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    if any(sha(path) != digest for path, digest in hashes.items()):
        raise ValueError('Source changed during analysis')
    report(summary)
    (OUTPUT/'verification.json').write_text(json.dumps(dict(completed=True, runs=18, gpu_executions=0,
        paired_scenes=4608, matched_success_scenes=402, own_success_counts=counts,
        initial_scenes_exact=True, success_cohorts_match=True, raw_matches_verified_post_at_hashes=True,
        successful_windows_valid=True, speed_distance_time_identity=True, input_hashes_unchanged=True), indent=2)+'\n')
    print(OUTPUT/'report.md', flush=True)


if __name__ == '__main__':
    main()
