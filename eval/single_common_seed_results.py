import os
from pathlib import Path
import statistics

from single_size_results import ROOT, MODELS, TASKS, read, write

OUTPUT = ROOT/'output_etc/check/single_common_env512_seeds0to9_20261120'
PREVIOUS = ROOT/'output_etc/check/single_size_native_common_env512_seeds167_20261120'
REUSED = (1, 6, 7)
NEW_SEEDS = (0, 2, 3, 4, 5, 8, 9)
SEED_RANGE = os.environ.get('SINGLE_COMMON_SEEDS', '0to9')
if SEED_RANGE not in ('0to9', '10to20', '21to25'):
    raise ValueError('Expected SINGLE_COMMON_SEEDS=0to9/10to20/21to25')
REUSED_RESULTS = {}
if SEED_RANGE == '10to20':
    PREVIOUS = OUTPUT
    OUTPUT = ROOT/'output_etc/check/single_common_env512_seeds10to20_20261120'
    REUSED = tuple(range(10))
    NEW_SEEDS = tuple(range(10, 21))
    REUSED_RESULTS = {(row['model'], row['seed']):Path(row['result']) for row in read(PREVIOUS/'summary.json')['rows']}
if SEED_RANGE == '21to25':
    PREVIOUS = ROOT/'output_etc/check/single_common_env512_seeds10to20_20261120'
    OUTPUT = ROOT/'output_etc/check/single_common_env512_seeds21to25_20261120'
    REUSED = tuple(range(21))
    NEW_SEEDS = tuple(range(21, 26))
    REUSED_RESULTS = {(row['model'], row['seed']):Path(row['result']) for row in read(PREVIOUS/'summary.json')['rows']}
ALL_SEEDS = tuple(sorted((*REUSED, *NEW_SEEDS)))
NEW_RUNS = len(NEW_SEEDS)*len(MODELS)
TOTAL_RUNS = len(ALL_SEEDS)*len(MODELS)


def result_path(model, seed):
    if (model, seed) in REUSED_RESULTS:
        return REUSED_RESULTS[model, seed]
    root = PREVIOUS if seed in REUSED else OUTPUT
    return root/f'rollouts/common/{model}/general_seed{seed}/result.json'


def summarize(verified):
    rows = []
    aggregates = []
    lines = [f'# ② 기존 공통 크기 · seed 0–{ALL_SEEDS[-1]}', '',
        f'추가 평가 검증 {len(verified)}/{NEW_RUNS}회 + 기존{len(REUSED)*4}회. 1명·2토큰·512환경·900step.',
        'SIT/CLIMB/AT/ON_TOP 받침은 공통 크기, ON_TOP payload·PUSH는 기존 크기다. 원본 carry9종 조건이 아니다.',
        f'수치는 SR(%). seed {list(REUSED)}는 기존 결과를 재사용하며 평균±표본std는 seed별 SR로 계산한다.', '']
    counts = {}
    for model in MODELS:
        lines += ['## '+dict(tokenhsi='TokenHSI', sct='SCT', coopft='CoopFT', ours='Ours')[model], '',
            '| seed | SIT | CLIMB | AT | ON_TOP | PUSH | 전체 |', '|---|---:|---:|---:|---:|---:|---:|']
        model_rows = []
        for seed in ALL_SEEDS:
            if seed not in REUSED and f'common/{model}/seed{seed}' not in verified:
                continue
            data = read(result_path(model, seed))
            if data.get('carry_sizes', 'table1') != 'table1' or data['size_variant'] != 'common':
                raise ValueError('Wrong size condition')
            if data['seed'] != seed or data['num_envs'] != 512 or data['horizon'] != 900:
                raise ValueError('Wrong seed or rollout dimensions')
            scores = data['scores']['region']
            task_counts = {task:scores[task]['trials'] for task in TASKS}
            if sum(task_counts.values()) != 512 or (seed in counts and counts[seed] != task_counts):
                raise ValueError('Trial counts differ')
            counts[seed] = task_counts
            metrics = {task:100*scores[task]['success']/scores[task]['trials'] for task in TASKS}
            metrics['overall'] = 100*sum(scores[task]['success'] for task in TASKS)/512
            for task in TASKS:
                if abs(metrics[task]-100*scores[task]['rate']) > 1e-10:
                    raise ValueError('Recorded rate differs from raw counts')
            row = dict(model=model, seed=seed, reused=seed in REUSED, metrics=metrics,
                counts=task_counts, result=str(result_path(model, seed)))
            rows.append(row)
            model_rows.append(row)
            lines.append('| '+str(seed)+' | '+' | '.join(f'{value:.2f}' for value in metrics.values())+' |')
        if len(model_rows) == len(ALL_SEEDS):
            stats = {task:dict(mean=statistics.mean(r['metrics'][task] for r in model_rows),
                std=statistics.stdev(r['metrics'][task] for r in model_rows)) for task in (*TASKS, 'overall')}
            entry = dict(model=model, metrics=stats)
            lines.append(f'| 0–{ALL_SEEDS[-1]} 평균±std | '+' | '.join(f'{v["mean"]:.2f}±{v["std"]:.2f}' for v in stats.values())+' |')
            if SEED_RANGE != '0to9':
                recent = [r for r in model_rows if r['seed'] in NEW_SEEDS]
                entry['new_seed_metrics'] = {task:dict(mean=statistics.mean(r['metrics'][task] for r in recent),
                    std=statistics.stdev(r['metrics'][task] for r in recent)) for task in (*TASKS, 'overall')}
                lines.append(f'| {NEW_SEEDS[0]}–{NEW_SEEDS[-1]} 평균±std | '+' | '.join(f'{v["mean"]:.2f}±{v["std"]:.2f}' for v in entry['new_seed_metrics'].values())+' |')
            aggregates.append(entry)
        lines.append('')
    lines += ['## seed별 행동 표본 수', '', '| seed | SIT | CLIMB | AT | ON_TOP | PUSH |', '|---|---:|---:|---:|---:|---:|']
    for seed, values in sorted(counts.items()):
        lines.append('| '+str(seed)+' | '+' | '.join(str(values[t]) for t in TASKS)+' |')
    lines += ['', '각 seed의 동일 초기 장면을 네 모델이 공유한다. 전체SR은 전체 성공/512이며 행동별SR의 단순 평균이 아니다.',
        '거리·초기화·물성·성공 기준은 기존 Table1과 같다. TokenHSI는 기존397D 입력을 유지한다.', '',
        '[원시 집계](summary.json) · [실행 구성](manifest.json) · [상태](status.json)']
    write(OUTPUT/'summary.json', dict(completed=len(rows)==TOTAL_RUNS, new_verified=len(verified),
        total_runs=len(rows), rows=rows, aggregates=aggregates, task_counts=counts))
    (OUTPUT/'report.md').write_text('\n'.join(lines)+'\n')
