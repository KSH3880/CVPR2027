import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import statistics
import sys

import numpy as np
import torch

from coupled_cooperation_metrics import sha, surface_contact, counters, ratios

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tokenhsi'))
SUITE = ROOT/'output_etc/check/v4_2000_tables12_radius_seeds167_20261112'
OLD_POST = ROOT/'output_etc/check/enabling_post_at_time_seeds167_20261111'
OUTPUT = ROOT/'output_etc/check/paper43_analysis_20261112'
MODELS = ('tokenhsi', 'stage1', 'fsm', 'coopft', 'ours')
SEEDS = (1, 6, 7)
SETTINGS = ('enable_sit', 'enable_climb', 'enable_ontop')


def mean(values):
    a = np.asarray(values, dtype=float)
    return float(a.mean()) if len(a) else None


def aggregate(rows, metric):
    values = []
    for seed in SEEDS:
        cells = [r[metric] for r in rows if r['seed'] == seed]
        values.append(statistics.mean(cells) if cells and None not in cells else None)
    complete = None not in values
    return dict(mean=statistics.mean(values) if complete else None,
        std=statistics.stdev(values) if complete else None, seeds=dict(zip(SEEDS, values)))


def enabling():
    runs, hashes, cache, paired, trial_rows = [], {}, {}, [], []
    for setting in SETTINGS:
        for seed in SEEDS:
            initial_reference = None
            for model in MODELS:
                base = SUITE/'post_at_time' if model == 'ours' else OLD_POST
                folder = base/f'{model}_{setting}_seed{seed}'
                result = json.loads((folder/'result.json').read_text())
                initial_path = Path(result['source'])/'initial.pt'
                for path in (folder/'trials.npz', folder/'result.json', initial_path):
                    hashes[str(path)] = sha(path)
                initial = torch.load(initial_path, map_location='cpu', weights_only=False)
                if initial_reference is None:
                    initial_reference = initial
                np.testing.assert_array_equal(initial_reference['roots'][..., :7], initial['roots'][..., :7])
                for key in ('packet', 'predecessor', 'goals', 'sizes'):
                    np.testing.assert_array_equal(initial_reference[key], initial[key])
                with np.load(folder/'trials.npz') as data:
                    a = {k: data[k].copy() for k in ('first_at', 'first_follower', 'first_group')}
                cache[model, setting, seed] = a
                at, follower, group = (a[k] > 0 for k in ('first_at', 'first_follower', 'first_group'))
                if group.sum() != result['successes'] or np.any(group & ~(at & follower)):
                    raise ValueError('Enabling event mismatch')
                metrics = dict(at_success_percent=100*at.mean(), follower_success_percent=100*follower.mean(),
                    group_success_percent=100*group.mean(), no_at_percent=100*(~at).mean(),
                    at_without_follower_percent=100*(at & ~follower).mean(),
                    both_without_simultaneous_percent=100*(at & follower & ~group).mean(),
                    group_given_at_percent=100*group.sum()/at.sum(),
                    completion_s=mean(a['first_group'][group]/30),
                    at_s=mean(a['first_at'][group]/30),
                    post_at_s=mean((a['first_group'][group]-a['first_at'][group])/30))
                if not np.isclose(sum(metrics[k] for k in ('group_success_percent', 'no_at_percent',
                    'at_without_follower_percent', 'both_without_simultaneous_percent')), 100):
                    raise ValueError('Failure partition does not sum to 100')
                runs.append(dict(model=model, setting=setting, seed=seed, source=result['source'],
                    initial_root_velocity_equal=bool(torch.equal(initial_reference['roots'][..., 7:], initial['roots'][..., 7:])),
                    **metrics))
                for env in range(512):
                    trial_rows.append(dict(model=model, setting=setting, seed=seed, scene=env,
                        **{k: int(v[env]) for k, v in a.items()}))
            ours = cache['ours', setting, seed]
            for model in MODELS[:-1]:
                baseline = cache[model, setting, seed]
                both = (ours['first_group'] > 0) & (baseline['first_group'] > 0)
                ready = (ours['first_at'] > 0) & (baseline['first_at'] > 0)
                row = dict(model=model, setting=setting, seed=seed, both_success_count=int(both.sum()),
                    both_at_count=int(ready.sum()),
                    baseline_group_on_both_at_percent=100*mean(baseline['first_group'][ready] > 0),
                    ours_group_on_both_at_percent=100*mean(ours['first_group'][ready] > 0))
                for label, a in (('baseline', baseline), ('ours', ours)):
                    row[label+'_completion_s'] = mean(a['first_group'][both]/30)
                    row[label+'_at_s'] = mean(a['first_at'][both]/30)
                    row[label+'_post_at_s'] = mean((a['first_group'][both]-a['first_at'][both])/30)
                row['completion_delta_ours_minus_baseline_s'] = row['ours_completion_s']-row['baseline_completion_s']
                paired.append(row)
    summary = {m: {k: aggregate([r for r in runs if r['model'] == m], k) for k in metrics} for m in MODELS}
    paired_metrics = [k for k in paired[0] if k.endswith(('_percent', '_s'))]
    pairs = {m: dict(both_success_count=sum(r['both_success_count'] for r in paired if r['model'] == m),
        both_at_count=sum(r['both_at_count'] for r in paired if r['model'] == m),
        **{k: aggregate([r for r in paired if r['model'] == m], k) for k in paired_metrics}) for m in MODELS[:-1]}
    with (OUTPUT/'enabling_trials.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trial_rows[0]))
        writer.writeheader()
        writer.writerows(trial_rows)
    return dict(models=summary, paired=pairs, runs=runs, paired_runs=paired, source_sha256=hashes)


def coupled(job):
    torch.set_num_threads(1)
    source = Path(job['source'])
    first_path = source/'first_success.npy' if job['model'] == 'tokenhsi' else source/'table2_trials.npz'
    paths = [source/'cooperation.npz', source/'trace.npz', source/'initial.pt', first_path]
    hashes = {str(p): sha(p) for p in paths}
    initial = torch.load(source/'initial.pt', map_location='cpu', weights_only=False)
    if job['model'] == 'tokenhsi':
        first = np.load(first_path)
    else:
        with np.load(first_path) as a:
            first = a['first_success'].copy()
    with np.load(source/'cooperation.npz') as data:
        pose, hands, sizes = (data[k] for k in ('physical_boxes', 'hands', 'physical_sizes'))
    with np.load(source/'trace.npz') as data:
        alive = data['valid']
    origin = initial['shared_centers'][:, 0].numpy()
    goal = initial['shared_center_goals'][:, 0].numpy()
    distance = np.linalg.norm(pose[..., :3].astype(float)-goal[None], axis=-1)
    goal_hit = (distance <= .2) & alive.any(-1)
    observed = np.where(goal_hit.any(0), goal_hit.argmax(0)+1, -1)
    np.testing.assert_array_equal(first, observed)
    raw_contact = np.concatenate([surface_contact(hands[t:t+30], pose[t:t+30], sizes)
                                  for t in range(0, len(pose), 30)])
    active_contact = raw_contact & alive.all(-1)
    success = first > 0
    end = np.where(success, first, len(pose))
    t = np.arange(1, len(pose)+1)[:, None]
    before_end = t <= end
    first_joint = np.where((active_contact & before_end).any(0), (active_contact & before_end).argmax(0)+1, -1)
    joint = first_joint > 0
    start, a, b = counters(pose, origin, first, .2)
    raw, canonical = ratios(raw_contact.sum(0), a, b)
    started = start > 0
    # The same start/end, inclusive frames, and active original participants define this diagnostic.
    push_window = started[None] & (t >= start[None]) & before_end
    window_frames = push_window.sum(0)
    window_ratio = np.divide((active_contact & push_window).sum(0), window_frames,
        out=np.zeros(len(first), dtype=float), where=window_frames > 0)
    ready_by_start = started & joint & (first_joint <= start)
    last_index = end-1
    index = np.arange(len(first))
    final_distance = distance[last_index, index]
    initial_distance = np.linalg.norm(origin.astype(float)-goal, axis=-1)
    final_progress = (initial_distance-final_distance)/initial_distance
    fallen_by_end = ((~alive.all(-1)) & before_end).any(0)
    arrays = dict(first_success=first, first_active_joint_near=first_joint, first_20cm=start,
        canonical_tcoop=canonical, window_active_joint_ratio=window_ratio,
        window_frames=window_frames, joint_near_ever=joint, ready_by_start=ready_by_start,
        fallen_by_end=fallen_by_end, initial_goal_distance=initial_distance,
        terminal_goal_distance=final_distance, terminal_progress_fraction=final_progress)
    dest = OUTPUT/'coupled'/f"{job['model']}_{job['name']}_seed{job['seed']}"
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest/'trials.npz', **arrays)
    rate = lambda mask: 100*float(mask.mean())
    conditional = lambda values, mask: mean(values[mask])
    metrics = dict(group_success_percent=rate(success), joint_near_ever_percent=rate(joint),
        group_given_joint_percent=100*conditional(success, joint),
        success_with_joint_percent=rate(success & joint), success_without_joint_percent=rate(success & ~joint),
        failure_with_joint_percent=rate(~success & joint), failure_without_joint_percent=rate(~success & ~joint),
        ready_before_start_percent=100*conditional(ready_by_start, started),
        canonical_tcoop_percent=100*float(np.nanmean(canonical)),
        canonical_upper_clipped_percent=rate(raw > 1),
        window_active_joint_percent=100*float(window_ratio.mean()),
        window_active_joint_success_percent=100*conditional(window_ratio, success),
        window_active_joint_failure_percent=100*conditional(window_ratio, ~success),
        failed_terminal_distance_m=conditional(final_distance, ~success),
        failed_terminal_progress_percent=100*conditional(final_progress, ~success),
        failed_with_fall_percent=100*conditional(fallen_by_end, ~success),
        successful_with_fall_percent=100*conditional(fallen_by_end, success),
        completion_s=conditional(first/30, success))
    if any(sha(p) != digest for p, digest in hashes.items()):
        raise ValueError('Coupled input changed')
    result = dict(**job, metrics=metrics, counts=dict(trials=len(first), success=int(success.sum()),
        joint=int(joint.sum()), started=int(started.sum()), canonical_nan=int(np.isnan(canonical).sum())),
        source_sha256=hashes, first_success_exact=True)
    (dest/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def fmt(r):
    return 'N/A' if r['mean'] is None else f"{r['mean']:.2f}±{r['std']:.2f}"


def supplement(e, c):
    readiness, hashes = {}, {}
    for run in e['runs']:
        model, setting, seed = (run[k] for k in ('model','setting','seed'))
        source = Path(run['source'])
        path = source/'trace.npz'
        hashes[str(path)] = sha(path)
        initial = torch.load(source/'initial.pt', map_location='cpu', weights_only=False)
        base = SUITE/'post_at_time' if model == 'ours' else OLD_POST
        with np.load(base/f'{model}_{setting}_seed{seed}'/'trials.npz') as a:
            at, success = a['first_at'], a['first_group'] > 0
        with np.load(path) as a:
            roots = a['roots'].astype(float)
            boxes = a['boxes'].astype(float) if setting == 'enable_ontop' else None
        ids = np.arange(512)
        f = (initial['predecessor'].numpy() >= 0).argmax(-1)
        leader = initial['predecessor'].numpy()[ids, f]
        goal = initial['goals'].numpy()[ids, leader, :2].astype(float)
        root0 = initial['roots'].numpy()[ids, f, :2].astype(float)
        follower = roots[:, ids, f, :2]
        frame = np.maximum(at-1, 0)
        root_at = follower[frame, ids]
        increments = np.linalg.norm(np.diff(np.concatenate((root0[None], follower)), axis=0), axis=-1)
        path_before_at = increments.cumsum(0)[frame, ids]
        if boxes is None:
            d0 = np.linalg.norm(root0-goal, axis=-1)
            d_at = np.linalg.norm(root_at-goal, axis=-1)
        else:
            payload = initial['packet'].numpy()[ids, f, 3]-2
            p0 = initial['boxes'].numpy()[ids, payload, :2].astype(float)
            p_at = boxes[frame, ids, payload, :2]
            d0 = np.linalg.norm(root0-p0, axis=-1)+np.linalg.norm(p0-goal, axis=-1)
            d_at = np.linalg.norm(root_at-p_at, axis=-1)+np.linalg.norm(p_at-goal, axis=-1)
        readiness[model, setting, seed] = dict(success=success, initial_route_m=d0,
            remaining_route_at_at_m=d_at, route_reduction_before_at_percent=100*(d0-d_at)/d0,
            follower_path_before_at_m=path_before_at)
    readiness_rows = []
    for model in MODELS[:-1]:
        for setting in SETTINGS:
            for seed in SEEDS:
                a, b = (readiness[m, setting, seed] for m in (model, 'ours'))
                mask = a['success'] & b['success']
                row = dict(model=model, setting=setting, seed=seed, count=int(mask.sum()))
                for label, values in (('baseline', a), ('ours', b)):
                    row.update({label+'_'+k: mean(v[mask]) for k, v in values.items() if k != 'success'})
                readiness_rows.append(row)
    readiness_summary = {m: {s: {k: aggregate([r for r in readiness_rows if r['model']==m and r['setting']==s], k)
        for k in readiness_rows[0] if k.startswith(('baseline_', 'ours_'))} for s in SETTINGS} for m in MODELS[:-1]}
    proximity_pairs, tolerance_rows = [], []
    for run in c['runs']:
        model, setting, seed = (run[k] for k in ('model','name','seed'))
        source = Path(run['source'])
        initial = torch.load(source/'initial.pt', map_location='cpu', weights_only=False)
        with np.load(source/'cooperation.npz') as data:
            positions = data['physical_boxes'][..., :3].astype(float)
        with np.load(source/'trace.npz') as data:
            valid = data['valid'].any(-1)
        goal = initial['shared_center_goals'][:, 0].numpy().astype(float)
        dist = np.linalg.norm(positions-goal[None], axis=-1)
        minimum = np.where(valid, dist, np.inf).min(0)
        with np.load(OUTPUT/'coupled'/f'{model}_{setting}_seed{seed}'/'trials.npz') as a:
            first = a['first_success']
            joint = a['joint_near_ever']
        np.testing.assert_array_equal(minimum <= .2, first > 0)
        tolerance_rows.append(dict(model=model, setting=setting, seed=seed,
            **{'within_'+str(round(r*100))+'cm_percent': 100*float((minimum <= r).mean()) for r in (.2,.3,.5,1.)}))
        if model != 'ours':
            with np.load(OUTPUT/'coupled'/f'ours_{setting}_seed{seed}'/'trials.npz') as b:
                ours_joint, ours_first = b['joint_near_ever'], b['first_success']
            mask = joint & ours_joint
            proximity_pairs.append(dict(model=model, setting=setting, seed=seed, count=int(mask.sum()),
                baseline_sr_percent=100*mean(first[mask] > 0), ours_sr_percent=100*mean(ours_first[mask] > 0)))
    paired_summary = {m: dict(count=sum(r['count'] for r in proximity_pairs if r['model']==m),
        **{k: aggregate([r for r in proximity_pairs if r['model']==m], k) for k in ('baseline_sr_percent','ours_sr_percent')})
        for m in MODELS[:-1]}
    tolerance_summary = {m: {k: aggregate([r for r in tolerance_rows if r['model']==m], k)
        for k in tolerance_rows[0] if k.startswith('within_')} for m in MODELS}
    if any(sha(p) != digest for p, digest in hashes.items()):
        raise ValueError('Readiness trace changed')
    result = dict(readiness_pairs=readiness_summary, readiness_runs=readiness_rows,
        coupled_common_joint=paired_summary, coupled_common_joint_runs=proximity_pairs,
        coupled_goal_tolerance=tolerance_summary, coupled_goal_tolerance_runs=tolerance_rows,
        readiness_trace_sha256=hashes)
    (OUTPUT/'supplement.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    lines = ['# 4.3 추가 대조', '',
        '성공 문턱/공식 SR은 유지한다. 아래 목표 거리 문턱은 달성 위치의 분포를 설명하는 민감도 분석이다.', '',
        '## Coupled 두 모델 모두 공동 근접을 형성한 동일 장면', '',
        '| 비교군 | 공통 장면 | 비교군 SR | Ours SR |', '|---|---:|---:|---:|']
    for m, r in paired_summary.items():
        lines.append(f"| {m} | {r['count']} | {fmt(r['baseline_sr_percent'])} | {fmt(r['ours_sr_percent'])} |")
    lines += ['', '## Coupled 유효 frame의 최소3D 목표 거리', '',
        '| 모델 | ≤20cm | ≤30cm | ≤50cm | ≤1m |', '|---|---:|---:|---:|---:|']
    for m, r in tolerance_summary.items():
        lines.append('| '+m+' | '+' | '.join(fmt(r[k]) for k in r)+' |')
    lines += ['', '## Enabling 공통 성공 장면의 후행 준비 상태', '',
        '각 모델 자신의 최초 AT 성공 시점에 측정한다. SIT/CLIMB는 사람→목표 수평 거리, ON_TOP은 사람→payload + payload→목표의 수평 직선거리합이다. 실제 남은 최적 경로나 완료 시간은 아니다. 초기 상태 대비 감소율은 음수도 보존한다.', '',
        '| 비교군 | 후행 | 비교군 남은 거리(m) | Ours 남은 거리(m) | 비교군 거리 감소(%) | Ours 거리 감소(%) |',
        '|---|---|---:|---:|---:|---:|']
    for m, settings in readiness_summary.items():
        for setting, r in settings.items():
            keys = ('baseline_remaining_route_at_at_m','ours_remaining_route_at_at_m',
                'baseline_route_reduction_before_at_percent','ours_route_reduction_before_at_percent')
            lines.append('| '+m+' | '+setting+' | '+' | '.join(fmt(r[k]) for k in keys)+' |')
    lines += ['', '[전체 수치](supplement.json) · [주 분석](report.md)']
    (OUTPUT/'supplement.md').write_text('\n'.join(lines)+'\n')
    return result


def write_report(enabling_data, coupled_data):
    lines = ['# 4.3 기존 기록 분석', '',
        'PDF (15)의 비교군 유지: Enabling Coop. FT4k / Coupled Coop. FT6k / Ours V4 2k.512환경×seed1/6/7. LLM Scheduler는 이 저장소에 원시 기록이 없어 PDF 수치만 사용한다.', '',
        '아래 통계는 행동/인원별 trial 평균→seed별 세 설정 동일 가중→seed 평균±표본std다. 조건부 표본 수는 JSON에 저장한다. GPU 재평가 없음.', '',
        '## Enabling 실패 분해', '',
        'AT/F는 유효한 개인 성공이 한 번이라도 존재함, Group은 같은 프레임의 두 성공이다. 실패 분해 세 열과 Group SR의 합은100%. 후행 미성공은 관찰된 결과이며 충돌 등 원인을 단정하지 않는다.', '',
        '| 모델 | AT SR | Group SR | AT 미성공 | AT 성공·후행 미성공 | 둘 다 성공 경험·동시 성공 없음 | Group / AT 성공 (%) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for m, d in enabling_data['models'].items():
        keys = ('at_success_percent','group_success_percent','no_at_percent','at_without_follower_percent',
                'both_without_simultaneous_percent','group_given_at_percent')
        lines.append('| '+m+' | '+' | '.join(fmt(d[k]) for k in keys)+' |')
    lines += ['', '## Enabling 두 모델 공통 성공 장면의 시간', '',
        '각 비교군과 Ours 모두 성공한 장면만 사용한다. 각 행의 표본이 달라 비교군끼리 가로질러 비교하지 않는다. 음의 Δ는 Ours의 짧은 완료 시간이다.', '',
        '| 비교군 | 공통 성공 수 | 비교군 완료(s) | Ours 완료(s) | Δ Ours−비교군(s) | 비교군 Post-AT(s) | Ours Post-AT(s) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for m, d in enabling_data['paired'].items():
        keys = ('baseline_completion_s','ours_completion_s','completion_delta_ours_minus_baseline_s',
                'baseline_post_at_s','ours_post_at_s')
        lines.append('| '+m+' | '+str(d['both_success_count'])+' | '+' | '.join(fmt(d[k]) for k in keys)+' |')
    lines += ['', '## Enabling 두 모델 모두 AT에 성공한 장면의 Group SR', '',
        '각 모델에서 선행이 한 번 이상 성공한 동일 장면의 조건부 SR이다. 여전히 결과에 따른 선택이며 순수한 모듈 인과 효과는 아니다.', '',
        '| 비교군 | 공통 AT 성공 수 | 비교군 Group SR | Ours Group SR |', '|---|---:|---:|---:|']
    for m, d in enabling_data['paired'].items():
        lines.append(f"| {m} | {d['both_at_count']} | {fmt(d['baseline_group_on_both_at_percent'])} | {fmt(d['ours_group_on_both_at_percent'])} |")
    lines += ['', '## Coupled 공동 근접과 목표 도달', '',
        '추가 분석의 공동 근접은 원래 전원 생존·양손 중심의 상자 표면거리<12cm가 같은 프레임에 성립한 경우다. 최초 성공 또는900step까지 검사한다. 접촉력/유효 미는 힘을 뜻하지 않는다.', '',
        '| 모델 | Group SR | 공동 근접 경험 | 근접 경험 후 성공률 | 공동 근접 경험·실패 | 공동 근접 없이 성공 |',
        '|---|---:|---:|---:|---:|---:|']
    for m, d in coupled_data['models'].items():
        keys = ('group_success_percent','joint_near_ever_percent','group_given_joint_percent',
                'failure_with_joint_percent','success_without_joint_percent')
        lines.append('| '+m+' | '+' | '.join(fmt(d[k]) for k in keys)+' |')
    lines += ['', '## Coupled 출발과 실제 밀기 구간', '',
        '준비 후 출발:20cm 출발 전에 위 공동 근접이 한 번 이상 있었던 started trial 비율. 구간 근접:20cm 출발부터 최초 성공/종료까지(양 끝 포함) 전원 생존·양손 근접 프레임 비율; 미출발0. 기존 t_coop과 분자/분모·alive 처리 모두 구분하며 대체 지표로 확정하지 않는다.', '',
        '| 모델 | 준비 후 출발(%) | 구간 근접 전체(%) | 구간 근접 성공(%) | 구간 근접 실패(%) | 기존 t_coop(%) | 상한 clip(%) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for m, d in coupled_data['models'].items():
        keys = ('ready_before_start_percent','window_active_joint_percent','window_active_joint_success_percent',
                'window_active_joint_failure_percent','canonical_tcoop_percent','canonical_upper_clipped_percent')
        lines.append('| '+m+' | '+' | '.join(fmt(d[k]) for k in keys)+' |')
    lines += ['', '## Coupled 실패 상황', '',
        '실패 trial의900step 시점 물리 상자와 목표 간3D 거리, 초기 대비 거리 감소, 종료 전 원래 참여자 중 한 명이라도 낙상한 비율이다. 낙상은 실패의 확정 원인과 구분한다.', '',
        '| 모델 | 실패 시 최종 목표 거리(m) | 실패 시 목표 거리 감소(%) | 실패 중 낙상(%) | 성공 중 낙상(%) |',
        '|---|---:|---:|---:|---:|']
    for m, d in coupled_data['models'].items():
        keys = ('failed_terminal_distance_m','failed_terminal_progress_percent','failed_with_fall_percent','successful_with_fall_percent')
        lines.append('| '+m+' | '+' | '.join(fmt(d[k]) for k in keys)+' |')
    lines += ['', 'TokenHSI 고정397D 입력 차이와 FSM 초기 정책 관측의 기존 불일치는 원본 기록의 한계로 유지한다. 새로운 CPU 집계가 이를 보정하지 않는다.', '',
        '[전체 수치](summary.json) · [추가 대조: 준비 상태·목표 정확도](supplement.md) · [실패 분해 그림](failure_partitions.png) · [Enabling trial](enabling_trials.csv) · [검증](verification.json)']
    (OUTPUT/'report.md').write_text('\n'.join(lines)+'\n')


def plot_partitions(e, extra):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    labels = ['TokenHSI', 'SCT', 'SCT + FSM', 'SCT + Coop. FT', 'Ours']
    colors = ['#3B8C6E', '#C0C4CD', '#E8A351', '#A65E78']
    parts = [([e['models'][m][k]['mean'] for m in MODELS], title) for k, title in (
        ('group_success_percent','Group success'), ('no_at_percent','No AT success'),
        ('at_without_follower_percent','AT success, no follower success'),
        ('both_without_simultaneous_percent','Both ever, never simultaneous'))]
    d = extra['coupled_goal_tolerance']
    close = np.array([d[m]['within_20cm_percent']['mean'] for m in MODELS])
    near = np.array([d[m]['within_50cm_percent']['mean'] for m in MODELS])
    coupled_parts = [(close, 'Minimum goal error ≤20cm'), (near-close, '20–50cm'), (100-near, '>50cm')]
    for ax, entries, palette, title in zip(axes, (parts, coupled_parts), (colors, [colors[0],colors[2],colors[1]]),
            ('Enabling: failure stage (Coop. FT 4k)', 'Coupled: goal attainment (Coop. FT 6k)')):
        left = np.zeros(5)
        for (values, label), color in zip(entries, palette):
            ax.barh(labels, values, left=left, color=color, label=label, height=.65)
            for y, value in enumerate(values):
                if value >= 5:
                    ax.text(left[y]+value/2, y, f'{value:.1f}', ha='center', va='center', fontsize=9)
            left += values
        ax.set_xlim(0,100)
        ax.invert_yaxis()
        ax.set_xlabel('All evaluation trials (%)')
        ax.set_title(title, fontsize=11)
        ax.legend(loc='upper center', bbox_to_anchor=(.5,-.19), fontsize=8, frameon=False)
        ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    for extension in ('png','pdf'):
        fig.savefig(OUTPUT/f'failure_partitions.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only: set CUDA_VISIBLE_DEVICES=')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    e = enabling()
    print('Enabling 45 runs complete', flush=True)
    current = json.loads((SUITE/'summary.json').read_text())
    jobs = [dict(model=r['model'], name=r['name'], seed=r['seed'], source=r['source'])
        for r in current['seed_runs'] if r['table']=='table2' and r['name'].startswith('coupled_')]
    runs = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(coupled, job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            runs.append(row)
            print(f"Coupled {len(runs)}/{len(jobs)} {row['model']} {row['name']} seed{row['seed']}", flush=True)
    models = {}
    for m in MODELS:
        selected = [dict(seed=r['seed'], **r['metrics']) for r in runs if r['model']==m]
        models[m] = {k: aggregate(selected, k) for k in runs[0]['metrics']}
        expected = current['models'][m]['coupled_average']
        for actual, key in (('group_success_percent','Coupled Interaction Group Success Rate'),
                            ('canonical_tcoop_percent','t_coop_percent')):
            if not np.isclose(models[m][actual]['mean'], expected[key]['mean'], atol=1e-5):
                raise ValueError('Existing Coupled metric mismatch')
    if any(sha(p) != digest for p, digest in e['source_sha256'].items()):
        raise ValueError('Enabling source changed')
    c = dict(models=models, runs=runs)
    (OUTPUT/'summary.json').write_text(json.dumps(dict(enabling=e, coupled=c), indent=2, allow_nan=False)+'\n')
    write_report(e, c)
    extra = supplement(e, c)
    plot_partitions(e, extra)
    (OUTPUT/'verification.json').write_text(json.dumps(dict(completed=True, enabling_runs=45, coupled_runs=45,
        initial_enabling_geometry_equal=True, enabling_success_counts_exact=True, failure_partition_identity=True,
        coupled_first_success_exact=True, coupled_existing_sr_tcoop_match=True, source_hashes_unchanged=True,
        gpu_executions=0, enabling_coopft_checkpoint='4k', coupled_coopft_checkpoint='6k', ours='V4 2k'), indent=2)+'\n')
    print(OUTPUT/'report.md', flush=True)


if __name__ == '__main__':
    main()
