import json
from pathlib import Path

import numpy as np

from summarize_coopft_tables import stats, macro


SEEDS = (1, 6, 7)
ENABLE = ('enable_sit', 'enable_climb', 'enable_ontop')
COUPLED = ('coupled_2', 'coupled_3', 'coupled_4')
METRICS = ('Independent Interaction Success Rate', 'Enabling Interaction Group Success Rate',
           'Coupled Interaction Group Success Rate', 'Mixed Interaction Group Success Rate')


def read(path):
    return json.loads(path.read_text())


def summarize(output):
    manifest = read(output/'manifest.json')
    data, runs = dict(table1={}, table2={}), []
    settings = [('table1', f'{mode}_{n}h') for n in (2, 4, 8, 16)
                for mode in ('independent', 'mixed')]
    settings += [('table2', name) for name in ENABLE+COUPLED]
    for table, name in settings:
        rows, times, contacts = [], [], []
        for seed in SEEDS:
            path = output/table/name/f'general_seed{seed}'
            row = read(path/'result.json')
            proof = read(output/'checks'/table/name/f'seed{seed}.json')
            if not all(proof[k] for k in ('first_policy_input_exact', 'policy_preserved',
                                         'cpu_gpu_sr_exact', 'cpu_gpu_errors_match')):
                raise ValueError('Missing verification: '+str(path))
            rows.append(row)
            run = dict(table=table, name=name, seed=seed, source=str(path),
                       scores=row['interaction_group_scores']['region'], errors=row['errors'])
            if name in ENABLE:
                record = read(path/'table2.json')
                with np.load(path/'table2_trials.npz') as saved:
                    first = saved['first_success'].copy()
                success = first >= 1
                if first.shape != (512,) or int(success.sum()) != record['successes']:
                    raise ValueError('Successful trial count differs')
                duration = np.where(success, first*record['control_dt'], 900*record['control_dt'])
                succ = float(duration[success].mean()) if success.any() else None
                if succ is not None and not np.isclose(succ, record['completion_time_s'], atol=2e-6, rtol=0):
                    raise ValueError('Successful completion time differs')
                times.append(dict(completion_time_success_s=succ, completion_time_all_s=float(duration.mean())))
                run.update(times[-1])
            if name in COUPLED:
                source = output/'latest_cooperation'/f'v5_{name[-1]}h_seed{seed}'
                with np.load(source/'trials.npz') as saved:
                    values = saved['20__tcoop'].astype(np.float64)*100
                contacts.append(values[np.isfinite(values)])
                run['t_coop_percent'] = float(contacts[-1].mean())
            runs.append(run)
        item = {metric: stats([100*r['interaction_group_scores']['region'][metric]['rate']
                if r['interaction_group_scores']['region'][metric]['rate'] is not None else None
                for r in rows]) for metric in METRICS}
        if times:
            item.update({key: stats([t[key] for t in times]) for key in times[0]})
        if contacts:
            item['t_coop_percent'] = dict(stats([float(v.mean()) for v in contacts]),
                pooled_mean=float(np.concatenate(contacts).mean()))
        data[table][name] = item
    data['enabling_average'] = {key: macro([data['table2'][n] for n in ENABLE], key)
        for key in (METRICS[1], 'completion_time_success_s', 'completion_time_all_s')}
    data['coupled_average'] = {key: macro([data['table2'][n] for n in COUPLED], key)
        for key in (METRICS[2], 't_coop_percent')}
    data['coupled_average']['t_coop_percent']['pooled_mean'] = float(np.mean([
        data['table2'][n]['t_coop_percent']['pooled_mean'] for n in COUPLED]))
    summary = dict(protocol_id=manifest['protocol_id'], checkpoint=manifest['checkpoint'],
        checkpoint_sha256=manifest['checkpoint_sha256'], envs=512, seeds=SEEDS, horizon=900,
        models={'ours': data}, seed_runs=runs)
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')

    def fmt(value):
        return 'N/A' if value['mean'] is None else f"{value['mean']:.2f}±{value['std']:.2f}"

    lines = ['# Ours V5 2k · Table 1/2/3', '',
        '512환경 × seeds 1/6/7 · 900step · XY100% · M=2N · 8H R3m / 16H R7.5m.', '',
        '## Table 1 · Success Rate (%)', '',
        '| 인원 | Independent | Mixed |', '|---|---:|---:|']
    for n in (2, 4, 8, 16):
        values = [fmt(data['table1'][f'{mode}_{n}h'][METRICS[i]])
                  for mode, i in (('independent', 0), ('mixed', 3))]
        lines.append(f'| {n}H | '+ ' | '.join(values)+' |')
    lines += ['', '## Table 2 · Enabling', '',
        '| 조건 | Group SR (%) | Time Succ. (s) | Time All (s) |', '|---|---:|---:|---:|']
    for name in ENABLE+('Avg.',):
        item = data['enabling_average'] if name == 'Avg.' else data['table2'][name]
        lines.append('| '+name+' | '+' | '.join(fmt(item[k]) for k in
            (METRICS[1], 'completion_time_success_s', 'completion_time_all_s'))+' |')
    lines += ['', '## Table 3 · Coupled', '',
        '| 조건 | Group SR (%) | Cooperative Time Ratio (%) |', '|---|---:|---:|']
    for name in COUPLED+('Avg.',):
        item = data['coupled_average'] if name == 'Avg.' else data['table2'][name]
        lines.append('| '+name+' | '+fmt(item[METRICS[2]])+' | '+fmt(item['t_coop_percent'])+' |')
    lines += ['', '평균±표본표준편차. 행동/인원별 동일 가중 평균 후 seed 통계. Time All은 실패에 최대 episode 시간을 부여한다.',
        't_coop은 출발변위20cm·표면거리12cm·기록 전체 누적 규약. pooled 값은 summary.json에 별도로 보존한다.',
        '', '[Seed별 통계](summary.json) · [실행 설정](manifest.json) · [검증](completion_verification.json)']
    (output/'report.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    import sys
    summarize(Path(sys.argv[1]).resolve())
