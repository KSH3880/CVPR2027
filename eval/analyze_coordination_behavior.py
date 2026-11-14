import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch

import reaggregate_enabling_post_at_time as post

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'output_etc/check/enabling_post_at_time_seeds167_20261111'
V4 = ROOT/'output_etc/check/v4_2000_tables12_radius_seeds167_20261112/post_at_time'
LABELS = dict(tokenhsi='TokenHSI 40.5k', stage1='SCT 26k', fsm='SCT + FSM (참고)',
              coopft='SCT + Coop. FT 4k', ours='Ours V4 2k')
MAIN = ('prep_percent', 'occupancy_percent', 'post_at_s', 'at_sr_percent', 'group_sr_percent')
RADII = (.25, .35, .45)
MARGINS = (0., .1, .2)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def mean(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else None


def occupancy_key(radius, margin):
    return f'occ_r{radius:.2f}_m{margin:.2f}'


def behavior_arrays(positions, goal, half, yaw, first_at, first_group):
    count = len(first_at)
    rows = np.arange(count)
    at = first_at > 0
    group = first_group > 0
    distances = np.linalg.norm(positions-goal, axis=-1)
    before = np.maximum(first_at-1, 0)
    prep = (distances[0]-distances[before, rows])/np.maximum(distances[0], 1e-6)
    prep[~at] = np.nan
    pre = (np.arange(len(positions))[:, None] < first_at) & at
    delta = positions-goal
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.stack((delta[..., 0]*c+delta[..., 1]*s,
                      -delta[..., 0]*s+delta[..., 1]*c), -1)
    result = dict(first_at=first_at, first_group=first_group, d0_m=distances[0],
        d_pre_m=np.where(at, distances[before, rows], np.nan), prep_percent=100*prep,
        pre_frames=np.where(at, first_at, 0),
        post_at_s=np.where(group, (first_group-first_at)/30, np.nan))
    for radius in RADII:
        for margin in MARGINS:
            distance = np.linalg.norm(np.maximum(np.abs(local)-half-margin, 0), axis=-1)
            occupied = distance <= radius
            result[occupancy_key(radius, margin)] = np.where(at,
                100*(occupied & pre).sum(0)/np.maximum(first_at, 1), np.nan)
    result['occupancy_percent'] = result[occupancy_key(.35, .1)]
    circle = distances <= np.linalg.norm(half, axis=-1)+.35+.1
    result['occ_rotation_envelope'] = np.where(at,
        100*(circle & pre).sum(0)/np.maximum(first_at, 1), np.nan)
    return result


def audit_geometry(positions, goal, half, yaw, a):
    checked = 0
    for env in np.flatnonzero(a['first_at'] > 0)[::64]:
        k = int(a['first_at'][env])
        start = np.linalg.norm(positions[0, env]-goal[env])
        end = np.linalg.norm(positions[k-1, env]-goal[env])
        np.testing.assert_allclose(a['prep_percent'][env], 100*(start-end)/max(start, 1e-6), atol=1e-10)
        c, s = np.cos(yaw[env]), np.sin(yaw[env])
        rotation = np.array([[c, -s], [s, c]])
        corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])*(half[env]+.1)
        corners = corners@rotation.T+goal[env]
        p = positions[:k, env]
        edge_distances, crosses = [], []
        for i in range(4):
            begin, finish = corners[i], corners[(i+1) % 4]
            edge = finish-begin
            fraction = np.clip(((p-begin)*edge).sum(-1)/(edge@edge), 0, 1)
            edge_distances.append(np.linalg.norm(p-begin-fraction[:, None]*edge, axis=-1))
            crosses.append(edge[0]*(p[:, 1]-begin[1])-edge[1]*(p[:, 0]-begin[0]))
        inside = (np.stack(crosses) >= -1e-12).all(0)
        occupied = inside | (np.min(edge_distances, axis=0) <= .35)
        np.testing.assert_allclose(a['occupancy_percent'][env], 100*occupied.mean(), atol=1e-10)
        checked += 1
    return checked


def summarize(a, mask=None):
    if mask is None:
        mask = np.ones(len(a['first_at']), dtype=bool)
    at, group = a['first_at'] > 0, a['first_group'] > 0
    fields = ('prep_percent', 'occupancy_percent', 'post_at_s', 'd0_m', 'd_pre_m',
              'occ_rotation_envelope', *[occupancy_key(r, m) for r in RADII for m in MARGINS])
    result = {key: mean(a[key][mask]) for key in fields}
    result.update(trials=int(mask.sum()), at_count=int((mask & at).sum()),
        group_count=int((mask & group).sum()), at_failed=int((mask & ~at).sum()),
        at_without_group=int((mask & at & ~group).sum()),
        fell_before_at=int((mask & at & a['fell_before_at']).sum()),
        negative_preparation=int((mask & (a['prep_percent'] < 0)).sum()),
        at_sr_percent=mean(100*at[mask]), group_sr_percent=mean(100*group[mask]))
    return result


def aggregate(rows, fields):
    result = {}
    for key in fields:
        seeds = {}
        for seed in post.SEEDS:
            values = [r[key] for r in rows if r['seed'] == seed]
            seeds[str(seed)] = float(np.mean(values)) if values and None not in values else None
        values = list(seeds.values())
        result[key] = dict(mean=float(np.mean(values)) if None not in values else None,
            std=float(np.std(values, ddof=1)) if None not in values else None, seeds=seeds)
    return result


def fmt(value):
    return 'N/A' if value['mean'] is None else f"{value['mean']:.2f}±{value['std']:.2f}"


def report(output, summary):
    lines = ['# Coordination Behavior Analysis', '',
        'TokenHSI40.5k / SCT·FSM26k / Coop. FT4k / Ours V4 2k. '
        '512환경 × seed1/6/7 × AT→SIT/CLIMB/ON_TOP, 총45개 기존 기록을 CPU로 분석했다.', '',
        '## 정의와 집계', '',
        '- A: follower root XY→선행 AT 목표 XY의 거리 감소율. ON_TOP도 동일한 직선거리이며 payload 경유 거리가 아니다. '
        '초기 snapshot을 t=0으로 포함하고 최초 AT 성공 step k 직전인 k−1을 사용한다. ε=1e−6m; 음수 보존.',
        '- B: root 중심 반경0.35m 원과 예정 배치 직사각형(양 축에0.10m 여유)의 겹침 시간 비율. '
        '원 반경은 agentCollisionDist 기본0.7m의 절반을 분석용 근사로 채택했으며 실제 몸체 형상이 아니다. '
        '직사각형은 AT 물체의 실제 크기·초기 yaw를 목표 XY로 평행 이동해 고정한다. '
        'AT에는 목표 회전 조건이 없으므로 이 회전은 분석용 규약이다.',
        '- B의 분모는 t=0,…,k−1의 k개 프레임이다. 낙상 후에도 물리 actor가 남으므로 위치를 포함한다. '
        '전신/높이/접촉력은 반영하지 않는다. 실제 물리 방해나 충돌률로 해석하지 않는다.',
        '- A·B는 AT 성공 episode 전체, C는 group 성공 episode만 평균한다. '
        'AT 실패의 A/B/C, group 실패의 C는 N/A이며 실패시간 대입은 없다.',
        '- C: 최초 유효 동시 group 성공−최초 유효 AT 성공, 단위 초. 원래 Table2 성공·낙상 규약 유지.',
        '- episode 평균→seed별 세 행동 동일 가중 평균→3seed 평균±표본표준편차. '
        'A는 백분율로 표시(100%=1). 공통 장면 비교도 각 모델 자신의 AT 시점을 사용한다.', '',
        '## 전체 AT 성공 표본의 A·B / Group 성공 표본의 C', '',
        '| 모델 | AT SR (%) | Group SR (%) | Preparation (%) ↑ | Premature Occupancy (%) ↓ | Post-AT (s) ↓ |',
        '|---|---:|---:|---:|---:|---:|']
    for model in LABELS:
        a = summary['models'][model]['average']
        keys = ('at_sr_percent', 'group_sr_percent', *MAIN[:3])
        lines.append('| '+LABELS[model]+' | '+' | '.join(fmt(a[k]) for k in keys)+' |')
    lines += ['', '## 표본 및 예외 건수', '',
        '| 모델 | 전체 | AT 성공 | Group 성공 | AT 실패 | AT 성공·Group 미완료 | AT 전에 후행 낙상 | A<0 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for m in LABELS:
        rows = [r for r in summary['runs'] if r['model'] == m]
        keys = ('trials','at_count','group_count','at_failed','at_without_group','fell_before_at','negative_preparation')
        lines.append('| '+LABELS[m]+' | '+' | '.join(str(sum(r[k] for r in rows)) for k in keys)+' |')
    for cohort, title in [('common_at', '공통 AT 성공 장면'), ('common_group', '공통 Group 성공 장면')]:
        lines += ['', '## '+title, '',
            '각 비교군과 Ours의 공통 장면이다. 조건부 표본이며 비교군마다 장면 집합이 다르다. '
            '공통 AT 표의 Group SR도 같은 장면에서 비교한다.', '',
            '| 비교군 | 장면 수 | A 비교군/Ours (%) | B 비교군/Ours (%) | C 비교군/Ours (s) | Group SR 비교군/Ours (%) |',
            '|---|---:|---|---|---|---|']
        for model in list(LABELS)[:-1]:
            a = summary['paired'][cohort][model]
            pairs = [fmt(a['average']['baseline_'+k])+' / '+fmt(a['average']['ours_'+k])
                     for k in (*MAIN[:3], 'group_sr_percent')]
            lines.append('| '+LABELS[model]+f" | {a['count']} | "+' | '.join(pairs)+' |')
        if cohort == 'common_at':
            lines += ['', '이 표의 C는 공통 AT 장면 중 각자 Group 성공 표본으로 계산하므로 표본이 다를 수 있다. '
                      '동일 성공 표본의 C 비교는 아래 공통 Group 표를 사용한다.']
    lines += ['', '## 행동별 결과', '',
        '| 모델 | 후행 | A (%) | B (%) | C (s) | AT SR (%) | Group SR (%) |',
        '|---|---|---:|---:|---:|---:|---:|']
    for m in LABELS:
        for setting in post.SETTINGS:
            a = summary['models'][m]['by_setting'][setting]
            lines.append('| '+LABELS[m]+' | '+setting+' | '+' | '.join(fmt(a[k]) for k in MAIN)+' |')
    lines += ['', '## Occupancy 민감도', '',
        '반경·margin은 모델에 동일하게 적용한다. 회전 외접원은 물체의 모든 yaw를 감싸는 '
        '반경 hypot(Lx/2,Ly/2)+0.35+0.10m의 별도 기준이며 직사각형 지표와 동일하지 않다.', '',
        '| root 반경(m) | margin(m) | '+' | '.join(LABELS.values())+' |',
        '|---|---|'+'---:|'*len(LABELS)]
    for r in RADII:
        for margin in MARGINS:
            key = occupancy_key(r, margin)
            lines.append(f'| {r:.2f} | {margin:.2f} | '+' | '.join(
                fmt(summary['models'][m]['average'][key]) for m in LABELS)+' |')
    lines.append('| 회전 외접원 | 0.10 | '+' | '.join(
        fmt(summary['models'][m]['average']['occ_rotation_envelope']) for m in LABELS)+' |')
    lines += ['', '## 해석 범위', '',
        'A가 높아도 B가 높으면 배치 공간을 미리 점유했을 수 있다. '
        'A는 접근량이며 준비 완료나 ON_TOP payload 준비 여부를 직접 나타내지 않는다. '
        'C는 성공 표본 조건부 시간이고 AT 완료 시간·성공률이 다른 모델의 인과적 속도 비교가 아니다. '
        'FSM은 첫 정책 관측 불일치가 있는 기존 참고 기록이다. '
        'TokenHSI는 고정397D 입력 차이를 유지한다. LLM Scheduler는 원시 기록이 없어 제외했다.', '',
        '[전체 통계·seed별 값](summary.json) · [episode CSV](trials.csv) · '
        '[규약·출처](manifest.json) · [검증](verification.json)']
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='output_etc/check/coordination_behavior_v4_2000_20261114')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if ROOT/'output_etc/check' not in output.parents or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Use output_etc/check and CUDA_VISIBLE_DEVICES= for CPU analysis')
    output.mkdir(parents=True, exist_ok=False)
    hashes, cache, runs, trial_rows, jobs, geometry = {}, {}, [], [], [], {}
    audit_count = 0
    for setting in post.SETTINGS:
        for seed in post.SEEDS:
            reference = None
            for model in LABELS:
                name = f'{model}_{setting}_seed{seed}'
                previous = (V4 if model == 'ours' else OLD)/name
                previous_result = json.loads((previous/'result.json').read_text())
                source = Path(previous_result['source'])
                for path in (previous/'result.json', previous/'trials.npz'):
                    hashes[str(path)] = post.sha(path)
                job = dict(model=model, setting=setting, seed=seed, name=name, source=str(source))
                jobs.append(job)
                scored = post.calculate(job, output)
                hashes.update(scored['source_sha256'])
                for path, digest in scored['source_sha256'].items():
                    if previous_result['source_sha256'].get(path) != digest:
                        raise ValueError('Raw source differs from verified previous Post-AT record')
                initial = torch.load(source/'initial.pt', map_location='cpu', weights_only=False)
                initial = {k: v.numpy() for k, v in initial.items() if torch.is_tensor(v)}
                if reference is None:
                    reference = initial
                for key in ('packet','predecessor','sizes','goals'):
                    np.testing.assert_array_equal(reference[key], initial[key])
                np.testing.assert_array_equal(reference['roots'][..., :7], initial['roots'][..., :7])
                packet = initial['packet']
                rows = np.arange(len(packet))
                follower = (initial['predecessor'] >= 0).argmax(-1)
                leader = initial['predecessor'][rows, follower]
                box = packet[rows, leader, 3]-2
                np.testing.assert_array_equal(packet[rows, follower, 4]-2, box)
                poses = initial['boxes'][rows, box, :7]
                half = initial['sizes'][rows, box, :2].astype(float)/2
                goal = initial['goals'][rows, leader, :2].astype(float)
                q = poses[:, 3:7].astype(float)
                yaw = np.arctan2(2*(q[:, 3]*q[:, 2]+q[:, 0]*q[:, 1]), 1-2*(q[:, 1]**2+q[:, 2]**2))
                current_geometry = (half, goal, yaw)
                if (setting, seed) not in geometry:
                    geometry[setting, seed] = current_geometry
                for a, b in zip(geometry[setting, seed], current_geometry):
                    np.testing.assert_array_equal(a, b)
                with np.load(source/'trace.npz') as trace:
                    roots = trace['roots'][:, rows, follower, :2].astype(float)
                    fall_step = trace['fall_step'][rows, follower]
                positions = np.concatenate((initial['roots'][rows, follower, :2][None], roots))
                with np.load(output/name/'trials.npz') as events, np.load(previous/'trials.npz') as old:
                    for key in ('first_at','first_follower','first_group'):
                        np.testing.assert_array_equal(events[key], old[key])
                    a = behavior_arrays(positions, goal, half, yaw, events['first_at'], events['first_group'])
                a['fell_before_at'] = (fall_step >= 1) & (fall_step < a['first_at'])
                a['fall_step'] = fall_step
                audit_count += audit_geometry(positions, goal, half, yaw, a)
                at, group = a['first_at'] > 0, a['first_group'] > 0
                if np.any(a['post_at_s'][group] < 0) or not np.isnan(a['prep_percent'][~at]).all():
                    raise ValueError('Invalid event masks')
                np.testing.assert_allclose(mean(a['post_at_s']), scored['post_at_time_s'], atol=1e-12)
                for r in RADII:
                    values = np.stack([a[occupancy_key(r, m)][at] for m in MARGINS])
                    if (np.diff(values, axis=0) < -1e-10).any():
                        raise ValueError('Occupancy must increase with margin')
                for margin in MARGINS:
                    values = np.stack([a[occupancy_key(r, margin)][at] for r in RADII])
                    if (np.diff(values, axis=0) < -1e-10).any():
                        raise ValueError('Occupancy must increase with radius')
                cache[model, setting, seed] = a
                runs.append(dict(**job, **summarize(a)))
                np.savez_compressed(output/name/'behavior.npz', **a)
                for env in rows:
                    row = dict(model=model, setting=setting, seed=seed, scene=int(env))
                    row.update({k: v[env].item() if np.isfinite(v[env]) else '' for k, v in a.items()})
                    trial_rows.append(row)
                print(f'{len(runs)}/45 {name}', flush=True)
    fields = (*MAIN, 'd0_m','d_pre_m','occ_rotation_envelope',
              *[occupancy_key(r, m) for r in RADII for m in MARGINS])
    models = {m: dict(average=aggregate([r for r in runs if r['model']==m], fields),
        by_setting={s: aggregate([r for r in runs if r['model']==m and r['setting']==s], fields)
                    for s in post.SETTINGS}) for m in LABELS}
    paired, paired_runs = {}, []
    for cohort, event in [('common_at','first_at'), ('common_group','first_group')]:
        paired[cohort] = {}
        for model in list(LABELS)[:-1]:
            selected = []
            for setting in post.SETTINGS:
                for seed in post.SEEDS:
                    a, b = cache[model, setting, seed], cache['ours', setting, seed]
                    mask = (a[event] > 0) & (b[event] > 0)
                    row = dict(cohort=cohort, model=model, setting=setting, seed=seed, count=int(mask.sum()))
                    for label, values in [('baseline',a), ('ours',b)]:
                        row.update({label+'_'+k: v for k,v in summarize(values, mask).items()})
                    for key in MAIN:
                        x, y = row['baseline_'+key], row['ours_'+key]
                        row['delta_'+key] = y-x if x is not None and y is not None else None
                    selected.append(row)
            paired_runs.extend(selected)
            paired[cohort][model] = dict(count=sum(r['count'] for r in selected),
                average=aggregate(selected, [label+'_'+k for label in ('baseline','ours','delta') for k in MAIN]))
    with (output/'trials.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trial_rows[0]))
        writer.writeheader()
        writer.writerows(trial_rows)
    summary = dict(models=models, paired=paired, runs=runs, paired_runs=paired_runs)
    dump(output/'summary.json', summary)
    for path in [Path(__file__), Path(post.__file__), ROOT/'tokenhsi/utils/evaluation_protocol.py',
                 ROOT/'eval/stage2_xy100.py', ROOT/'tokenhsi/env/tasks/multi_agent/collision_reward.py']:
        hashes[str(path)] = post.sha(path)
    dump(output/'manifest.json', dict(protocol='coordination-behavior-v1',
        base_protocol='cvpr2027-evaluation-v1.1.0', models=LABELS, jobs=jobs,
        seeds=post.SEEDS, envs=512, steps=900, dt_s=1/30, root_radius_m=.35, margin_m=.1,
        radius_sensitivity_m=RADII, margin_sensitivity_m=MARGINS,
        placement_orientation='initial AT object yaw, fixed at target XY',
        pre_at_frames='initial snapshot at t=0 through first_at_step-1 inclusive',
        failed_at='A/B/C missing', failed_group='C missing', source_sha256=hashes))
    if any(post.sha(p) != h for p,h in hashes.items()):
        raise ValueError('Source changed during analysis')
    dump(output/'verification.json', dict(completed=True, runs=len(runs), trials=len(trial_rows),
        initial_geometry_equal=True, independent_polygon_audit_episodes=audit_count,
        first_at_follower_group_exact=True, original_table2_sr_time_match=True,
        radius_margin_monotonic=True, source_hashes_unchanged=True, gpu_executions=0))
    report(output, summary)
    print(output/'report.md', flush=True)


if __name__ == '__main__':
    main()
