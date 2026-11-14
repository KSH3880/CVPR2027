import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'output_etc/check/enabling_completion_time_succ_all_seeds167_20261111'
SEEDS = (1, 6, 7)
SETTINGS = ('enable_sit', 'enable_climb', 'enable_ontop')
LABELS = {'tokenhsi': 'TokenHSI (+ PUSH)', 'stage1': 'SCT', 'coopft': 'SCT + Coop. FT (4k)',
          'fsm': 'SCT + FSM', 'ours': 'Ours (V3 2k)', 'coopft6k': 'SCT + Coop. FT (6k)'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    return dict(mean=statistics.mean(values) if None not in values else None,
                std=statistics.stdev(values) if None not in values else None, seed_values=values)


def fmt(value):
    return 'N/A' if value['mean'] is None else f"{value['mean']:.2f}±{value['std']:.2f}"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source_manifest = ROOT/'output_etc/check/enabling_post_at_time_seeds167_20261111/manifest.json'
    jobs = json.loads(source_manifest.read_text())['jobs']
    extra = ROOT/'output_etc/check/coopft_3000_6000_tables12_seeds167_20261111/6000/table2'
    jobs += [dict(model='coopft6k', setting=setting, seed=seed,
                  source=str(extra/setting/f'general_seed{seed}')) for setting in SETTINGS for seed in SEEDS]
    rows, hashes = [], {str(source_manifest): sha(source_manifest)}
    for job in jobs:
        source = Path(job['source'])
        table_path, result_path = source/'table2.json', source/'result.json'
        table, result = json.loads(table_path.read_text()), json.loads(result_path.read_text())
        trials_path = source/'table2_trials.npz'
        if trials_path.exists():
            with np.load(trials_path) as data:
                first = data['first_success'].copy()
        else:
            trials_path = source/'first_success.npy'
            first = np.load(trials_path)
        for path in (table_path, result_path, trials_path):
            hashes[str(path)] = sha(path)
        dt = table.get('control_dt', 1/30)
        horizon = result['horizon']
        cap = horizon*dt
        success = first >= 0
        if (first.shape, horizon, int(success.sum())) != ((512,), 900, table['successes']):
            raise ValueError('Trial count, horizon or success count mismatch: '+str(source))
        if success.any() and ((first[success] < 1).any() or (first[success] > horizon).any()):
            raise ValueError('Completion step outside episode: '+str(source))
        times = np.where(success, first*dt, cap)
        succ = float(times[success].mean()) if success.any() else None
        all_time = float(times.mean())
        if succ is not None and not np.isclose(succ, table['completion_time_s'], atol=2e-6, rtol=0):
            raise ValueError('Original successful completion time mismatch: '+str(source))
        identity = (int(success.sum())*(succ or 0)+(512-int(success.sum()))*cap)/512
        if not np.isclose(all_time, identity, atol=1e-10, rtol=0):
            raise ValueError('Failure penalty identity mismatch')
        rows.append(dict(model=job['model'], setting=job['setting'], seed=job['seed'], source=str(source),
            trials=512, successes=int(success.sum()), success_rate_percent=float(success.mean()*100),
            time_success_s=succ, time_all_s=all_time, maximum_episode_time_s=cap, control_dt=dt))
    models = {}
    metrics = ('success_rate_percent', 'time_success_s', 'time_all_s')
    for model in LABELS:
        settings = {}
        for setting in SETTINGS:
            selected = [next(r for r in rows if (r['model'], r['setting'], r['seed']) ==
                             (model, setting, seed)) for seed in SEEDS]
            settings[setting] = {metric: stats([r[metric] for r in selected]) for metric in metrics}
        average = {}
        for metric in metrics:
            values = [[settings[s][metric]['seed_values'][i] for s in SETTINGS] for i in range(3)]
            average[metric] = stats([statistics.mean(v) if None not in v else None for v in values])
        models[model] = dict(by_setting=settings, average=average)
    summary = dict(envs=512, seeds=SEEDS, horizon=900, failure_penalty='900 * recorded control_dt (approximately 30s)',
        aggregation='per-action trial mean, equal-weight mean over three actions, seed mean/sample std',
        models=models, seed_runs=rows, excluded_models=['llm_scheduler'])
    (OUTPUT/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    lines = ['# Enabling Completion Time · Succ. / All', '',
        '512환경 × seeds1/6/7 · 900step · XY100%. 기존 최초 유효 그룹 동시 성공 기록을 CPU 재집계했다.', '',
        'Succ.: 성공 trial만 포함. All: 성공은 완료 시간, 실패는 최대 episode 시간(약30초)을 사용.',
        '행동별 평균 → SIT/CLIMB/ON_TOP 동일 가중 평균 → seed 평균±표본표준편차(ddof=1).', '',
        '| Method | Group SR (%) ↑ | Time Succ. (s) ↓ | Time All (s) ↓ |', '|---|---:|---:|---:|']
    for model, label in LABELS.items():
        avg = models[model]['average']
        lines.append('| '+label+' | '+' | '.join(fmt(avg[k]) for k in metrics)+' |')
    lines += ['', '원래 표의 Coop. FT는4k이며6k 결과를 별도 행으로 제공한다. LLM Scheduler는 외부 결과라 포함하지 않는다.',
        'TokenHSI397D 입력 차이와 FSM 기존 첫 관측 차이는 재집계로 변경되지 않는다.', '',
        '## Seed별 행동 동일 가중 평균', '',
        '| Method | Seed | Group SR (%) | Time Succ. (s) | Time All (s) |', '|---|---:|---:|---:|---:|']
    for model, label in LABELS.items():
        avg = models[model]['average']
        for i, seed in enumerate(SEEDS):
            values = [avg[k]['seed_values'][i] for k in metrics]
            lines.append('| '+label+' | '+str(seed)+' | '+
                         ' | '.join('N/A' if v is None else f'{v:.6f}' for v in values)+' |')
    lines += ['', '[원시 수치·행동별 결과](summary.json) · [검증·원본 SHA](verification.json)']
    (OUTPUT/'report.md').write_text('\n'.join(lines)+'\n')
    if any(sha(Path(path)) != value for path, value in hashes.items()):
        raise ValueError('Original records changed')
    (OUTPUT/'verification.json').write_text(json.dumps(dict(completed=True, runs=len(rows),
        trials_per_run=512, original_success_counts_exact=True, original_successful_time_matches=True,
        failure_penalty_identity_verified=True, source_hashes_unchanged=True,
        gpu_executions=0, source_sha256=hashes, script_sha256=sha(Path(__file__))), indent=2)+'\n')
    print('\n'.join(lines[:15]))
    print(OUTPUT/'report.md')


if __name__ == '__main__':
    main()
