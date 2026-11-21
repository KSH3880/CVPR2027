import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import torch

from common import ROOT, read, write, sha
from suite import LABELS

sys.path.insert(0, str(ROOT/'eval'))
from analyze_coordination_behavior import behavior_arrays


def calculate(row, analysis, hashes):
    source = Path(row['source'])
    paths = [source/'initial.pt', source/'trace.npz', analysis, analysis.with_suffix('.npz')]
    for p in paths:
        hashes[str(p)] = sha(p)
    for p in paths[:2]:
        if hashes[str(p)] != row['source_sha256'][str(p)]:
            raise ValueError('Raw source changed')
    initial = torch.load(paths[0], map_location='cpu', weights_only=False)
    with np.load(paths[1]) as z:
        roots = z['roots'][:row['analyzed_steps']]
    with np.load(paths[3]) as z:
        at, group, prep = (z[k] for k in ('first_at', 'first_group', 'prep_percent'))
    n = row['trials']
    idx = np.arange(n)
    parent = initial['predecessor'].numpy()
    follower = (parent >= 0).argmax(1)
    leader = parent[idx, follower]
    goal = initial['goals'].numpy()[idx, leader, :2].astype(float)
    positions = np.concatenate((initial['roots'].numpy()[idx, follower, :2][None],
                                roots[:, idx, follower, :2])).astype(float)
    valid = at > 0
    before = np.maximum(at-1, 0)
    d0 = np.linalg.norm(positions[0]-goal, axis=-1)
    distance = np.linalg.norm(positions[before, idx]-goal, axis=-1)
    distance[~valid] = np.nan
    replay = 100*(d0-distance)/np.maximum(d0, 1e-6)
    np.testing.assert_allclose(replay, prep, rtol=1e-12, atol=1e-10, equal_nan=True)
    behavior = behavior_arrays(positions, goal, np.ones((n, 2)), np.zeros(n), at, group)
    np.testing.assert_array_equal(distance, behavior['d_pre_m'])
    for env in np.flatnonzero(valid):
        k = int(at[env])
        point = initial['roots'][env, follower[env], :2].numpy() if k == 1 else roots[k-2, env, follower[env], :2]
        np.testing.assert_allclose(distance[env], np.linalg.norm(point.astype(float)-goal[env]), atol=1e-12)
    if int(valid.sum()) != row['at_successes']:
        raise ValueError('AT cohort changed')
    result = {k: row[k] for k in ('setting', 'model', 'seed', 'delay', 'trials', 'at_successes', 'at_sr_percent', 'group_sr_percent')}
    result.update(d_pre_m=float(np.mean(distance[valid])) if valid.any() else None,
                  median_m=float(np.median(distance[valid])) if valid.any() else None,
                  d0_at_success_m=float(np.mean(d0[valid])) if valid.any() else None)
    trials = [dict(env_id=i, at_step=int(at[i]), d_pre_m=float(distance[i]) if valid[i] else None,
                   initial_distance_m=float(d0[i]), group_success=bool(group[i] > 0)) for i in idx]
    return result, trials


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    suite, out = args.suite.resolve(), args.output.resolve()
    if ROOT/'output_etc/check' not in out.parents:
        raise ValueError('Output must be under output_etc/check')
    out.mkdir(parents=True, exist_ok=False)
    summary_path = suite/'summary.json'
    summary = read(summary_path)
    if not summary['completed']:
        raise ValueError('Wait for completed suite before final aggregation')
    files = list((suite/'analysis').glob('*/*.json'))
    manifest = read(suite/'manifest.json')
    files += [Path(item['analysis']) for item in manifest.get('baselines', [])]
    analyses = {}
    for p in files:
        row = read(p)
        analyses[tuple(row[k] for k in ('setting', 'model', 'seed', 'delay'))] = p
    hashes = {str(summary_path): sha(summary_path), str(Path(__file__).resolve()): sha(Path(__file__))}
    runs, trial_rows = [], []
    for row in summary['runs']:
        if not row['setting'].startswith('enable'):
            continue
        key = tuple(row[k] for k in ('setting', 'model', 'seed', 'delay'))
        result, trials = calculate(row, analyses[key], hashes)
        runs.append(result)
        for trial in trials:
            trial_rows.append(dict(zip(('setting', 'model', 'seed', 'delay'), key), **trial))
    seeds = sorted({r['seed'] for r in runs})
    delays = sorted({r['delay'] for r in runs}, key=str)
    settings = sorted({r['setting'] for r in runs})
    macro = []
    for model in LABELS:
        for seed in seeds:
            for delay in delays:
                selected = [r for r in runs if r['model'] == model and r['seed'] == seed and r['delay'] == delay]
                if len(selected) != 3:
                    raise ValueError('Incomplete type coverage')
                macro.append(dict(setting='enable_average', model=model, seed=seed, delay=delay,
                    d_pre_m=float(np.mean([r['d_pre_m'] for r in selected])),
                    at_successes=sum(r['at_successes'] for r in selected), trials=sum(r['trials'] for r in selected)))
    stats = []
    lines = ['# AT 완료 직전 거리 (m)', '',
        '후행 root XY → 선행 AT 배치 목표 XY. 최초 AT 성공 step k의 직전 k−1; k=1이면 초기 snapshot.',
        'AT 성공 전체 표본, Group 실패·낙상 포함. AT 실패는 N/A. 유형별 평균 → 세 유형 동일 가중 → seed 평균.',
        '거리만으로 적절한 대기 여부를 판정하지 않는다. 1m offset을 적용하지 않는다.', '',
        '조건 순서: '+' → '.join(map(str, delays))+f'; seeds={seeds}.', '']
    for setting in [*settings, 'enable_average']:
        lines += ['## '+setting, '', '| 모델 | 거리(m) | AT 성공/전체 |', '|---|---:|---:|']
        for model,label in LABELS.items():
            cells, counts = [], []
            for delay in delays:
                rows = [r for r in runs+macro if r['setting'] == setting and r['model'] == model and r['delay'] == delay]
                values = [r['d_pre_m'] for r in rows]
                value, std = float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else None
                stats.append(dict(setting=setting, model=model, delay=delay, mean_m=value, std_m=std))
                cells.append(f'{value:.3f}'+(f' ± {std:.3f}' if std is not None else ''))
                counts.append(f"{sum(r['at_successes'] for r in rows)}/{sum(r['trials'] for r in rows)}")
            lines.append('| '+label+' | '+' → '.join(cells)+' | '+' → '.join(counts)+' |')
        lines.append('')
    write(out/'summary.json', dict(source_suite=str(suite), runs=runs, macro_per_seed=macro, statistics=stats))
    with (out/'trials.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trial_rows[0]))
        writer.writeheader()
        writer.writerows(trial_rows)
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    if any(sha(Path(p)) != h for p,h in hashes.items()):
        raise ValueError('Input changed during analysis')
    write(out/'verification.json', dict(passed=True, runs=len(runs), trial_rows=len(trial_rows),
        raw_frame_index_verified=True, matches_existing_d_pre_m=True, preparation_replay_exact=True,
        at_sample_counts_exact=True, source_sha256=hashes))


if __name__ == '__main__':
    main()
