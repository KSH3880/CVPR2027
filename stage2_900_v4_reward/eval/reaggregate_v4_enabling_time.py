import json
from pathlib import Path
import shutil
import statistics

import numpy as np

from reaggregate_enabling_completion_time import sha


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT/'output_etc/check/v4_2000_tables12_radius_seeds167_20261112'
OUTPUT = SUITE/'time_succ_all'
SEEDS = (1, 6, 7)
SETTINGS = ('enable_sit', 'enable_climb', 'enable_ontop')
LABELS = dict(tokenhsi='TokenHSI (+ PUSH) 40.5k', stage1='SCT 26k', fsm='SCT + FSM (reference)',
              coopft='SCT + Coop. FT 6k', ours='Ours V4 2k')


def read(path):
    return json.loads(path.read_text())


def stats(values):
    return dict(mean=statistics.mean(values) if None not in values else None,
                std=statistics.stdev(values) if None not in values else None, seeds=values)


def pair(average):
    return ' / '.join('N/A' if average[key]['mean'] is None else f"{average[key]['mean']:.2f}"
                      for key in ('completion_time_success_s', 'completion_time_all_s'))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    before = OUTPUT/'before'
    before.mkdir(exist_ok=True)
    for name in ('report.md', 'summary.json', 'completion_verification.json'):
        if not (before/name).exists():
            shutil.copy2(SUITE/name, before/name)
    summary = read(before/'summary.json')
    jobs = [row for row in summary['seed_runs'] if row['table'] == 'table2' and row['name'] in SETTINGS]
    if len(jobs) != 45:
        raise ValueError('Expected five models, three actions, three seeds')
    rows, hashes = [], {}
    for job in jobs:
        source = Path(job['source'])
        table_path, result_path = source/'table2.json', source/'result.json'
        table, result = read(table_path), read(result_path)
        trials = source/'table2_trials.npz'
        if trials.exists():
            with np.load(trials) as saved:
                first = saved['first_success'].copy()
        else:
            trials = source/'first_success.npy'
            first = np.load(trials)
        for path in (table_path, result_path, trials):
            hashes[str(path)] = sha(path)
        dt = table.get('control_dt', 1/30)
        cap = result['horizon']*dt
        success = first >= 1
        if (first.shape, result['horizon'], int(success.sum())) != ((512,), 900, table['successes']):
            raise ValueError('Trial count or success mismatch: '+str(source))
        if (first == 0).any() or (first[success] > 900).any():
            raise ValueError('Invalid first completion step')
        times = np.where(success, first*dt, cap)
        succ = float(times[success].mean()) if success.any() else None
        all_time = float(times.mean())
        if succ is not None and not np.isclose(succ, table['completion_time_s'], atol=2e-6, rtol=0):
            raise ValueError('Existing successful time differs')
        identity = (int(success.sum())*(succ or 0)+(512-int(success.sum()))*cap)/512
        if not np.isclose(all_time, identity, atol=1e-10, rtol=0):
            raise ValueError('Failure penalty identity differs')
        rows.append(dict(model=job['model'], setting=job['name'], seed=job['seed'], source=str(source),
            trials=512, successes=int(success.sum()), maximum_episode_time_s=cap, control_dt=dt,
            completion_time_success_s=succ, completion_time_all_s=all_time))
    metrics = ('completion_time_success_s', 'completion_time_all_s')
    models = {}
    for model in LABELS:
        settings = {}
        for setting in SETTINGS:
            selected = [next(row for row in rows if (row['model'], row['setting'], row['seed']) ==
                             (model, setting, seed)) for seed in SEEDS]
            settings[setting] = {metric: stats([row[metric] for row in selected]) for metric in metrics}
        average = {metric: stats([statistics.mean([settings[s][metric]['seeds'][i] for s in SETTINGS])
            if all(settings[s][metric]['seeds'][i] is not None for s in SETTINGS) else None
            for i in range(3)]) for metric in metrics}
        original = summary['models'][model]['enabling_average']['completion_time_s']
        if not np.allclose(original['seeds'], average['completion_time_success_s']['seeds'], atol=2e-6, rtol=0):
            raise ValueError('Existing seed/action aggregation differs')
        models[model] = dict(by_setting=settings, average=average)
        for setting in SETTINGS:
            summary['models'][model]['table2'][setting].update(settings[setting])
        summary['models'][model]['enabling_average'].update(average)
    data = dict(envs=512, seeds=SEEDS, horizon=900, models=models, seed_runs=rows,
        failure_penalty='900 * recorded control_dt (approximately 30s)',
        aggregation='per-action trial mean -> equal mean over three actions -> seed mean/sample std')
    (OUTPUT/'summary.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    summary['enabling_time_succ_all'] = dict(source=str(OUTPUT/'summary.json'), **{
        key: data[key] for key in ('failure_penalty', 'aggregation')})
    (SUITE/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    report = (before/'report.md').read_text()
    start = report.index('## Table 2(a)')
    end = report.index('## Table 2(b)', start)
    section = report[start:end].replace('Completion Time (s) ↓', 'Time (Succ. / All) (s) ↓')
    lines = section.splitlines()
    for model, label in LABELS.items():
        index = next(i for i, line in enumerate(lines) if line.startswith('| '+label+' |'))
        cells = lines[index].split('|')
        cells[-2] = ' '+pair(models[model]['average'])+' '
        lines[index] = '|'.join(cells)
    report = report[:start]+'\n'.join(lines)+'\n\n'+report[end:]
    old_note = 'SR/Time은 seed 평균±표본표준편차(ddof=1). Enable Time은 성공 trial의 최초 유효 동시 성공 시간이며 행동별 동일 가중 평균이다.'
    new_note = 'SR은 seed 평균±표본표준편차(ddof=1). Time (Succ. / All)은 두 seed 평균을 초 단위로 표기한다. Succ.는 성공 trial의 최초 유효 그룹 동시 성공 시간, All은 성공 시 실제 시간·실패 시 최대 episode 시간(900×control_dt, 약30초)을 사용한다. 행동별 평균→세 행동 동일 가중→seed 평균이며 각 표준편차/seed 값은 summary.json에 보존한다.'
    if old_note not in report:
        raise ValueError('Existing Time definition note missing')
    report = report.replace(old_note, new_note)
    (SUITE/'report.md').write_text(report)
    lines = ['# Table 2(a) Time · Succ. / All', '',
        '512환경·seed1/6/7·900step. 기존 기록45회를 CPU 재집계했으며 GPU 평가는 실행하지 않았다.', '',
        '| Method | Time (Succ. / All) (s) ↓ |', '|---|---:|']
    lines += ['| '+label+' | '+pair(models[model]['average'])+' |' for model, label in LABELS.items()]
    lines += ['', new_note, '', '[행동/seed별 수치](summary.json) · [검증](verification.json)']
    (OUTPUT/'report.md').write_text('\n'.join(lines)+'\n')
    updated = read(SUITE/'summary.json')
    for model in LABELS:
        original, current = read(before/'summary.json')['models'][model], updated['models'][model]
        if original['table1'] != current['table1'] or original['coupled_average'] != current['coupled_average']:
            raise ValueError('Unrelated metrics changed')
        for setting in SETTINGS:
            old = original['table2'][setting]
            new = {key: value for key, value in current['table2'][setting].items() if key not in metrics}
            if old != new:
                raise ValueError('Existing Enabling metrics changed')
    if any(sha(Path(path)) != digest for path, digest in hashes.items()):
        raise ValueError('Raw source changed')
    verification = dict(completed=True, runs=45, original_success_counts_exact=True,
        original_successful_time_matches=True, failure_penalty_identity_verified=True,
        unrelated_metrics_preserved=True, raw_source_hashes_unchanged=True, gpu_executions=0,
        source_sha256=hashes, script_sha256=sha(Path(__file__)))
    (OUTPUT/'verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    proof = read(before/'completion_verification.json')
    proof['time_succ_all_reaggregation'] = str(OUTPUT/'verification.json')
    (SUITE/'completion_verification.json').write_text(json.dumps(proof, indent=2)+'\n')
    print('\n'.join(lines[:11]))


if __name__ == '__main__':
    main()
