import json
from pathlib import Path
import statistics

import numpy as np

from run_stage2_scalability import ROOT
from summarize_coopft_tables import stats, macro


SEEDS = (1, 6, 7)
ENABLE = ('enable_sit', 'enable_climb', 'enable_ontop')
COUPLED = ('coupled_2', 'coupled_3', 'coupled_4')
METRICS = ('Independent Interaction Success Rate', 'Enabling Interaction Group Success Rate',
           'Coupled Interaction Group Success Rate', 'Mixed Interaction Group Success Rate')
OLD = ROOT/'output_etc/check/v3_2000_tables12_seeds167_20261110'
RADIUS = ROOT/'output_etc/check/table1_group_radius_8r3_16r7p5_env512_seeds167_baselines_20261112'
COOP = ROOT/'output_etc/check/coopft_3000_6000_tables12_seeds167_20261111'
TOKEN = ROOT.parent/'tokenhsi_push/output_etc/check/tokenhsi_tables_512_seeds167_20261111_154749'
PRIOR = ROOT/'output_etc/check/coupled_cooperation_four_models_20261111'
LABELS = dict(tokenhsi='TokenHSI (+ PUSH) 40.5k', stage1='SCT 26k', fsm='SCT + FSM (reference)',
              coopft='SCT + Coop. FT 6k', ours='Ours V4 2k')


def read(path):
    return json.loads(path.read_text())


def source(output, model, table, name, seed):
    if model == 'ours':
        return output/table/name/f'general_seed{seed}'
    if table == 'table1' and name.endswith(('8h', '16h')):
        actual = 'stage1' if model == 'fsm' and name.startswith('independent') else model
        return RADIUS/actual/name/f'general_seed{seed}'
    if model == 'coopft':
        return COOP/'6000'/table/name/f'general_seed{seed}'
    if model == 'tokenhsi':
        return TOKEN/f'seed_{seed}'/name/f'general_seed{seed}'
    return OLD/table/model/name/f'general_seed{seed}'


def latest(output, model, people):
    if model == 'tokenhsi':
        data = read(TOKEN/'summary.json')['results'][f'coupled_{people}']
        return {key: data[key] for key in ('t_coop_percent', 'mean_absolute_jerk')}
    arrays = {'t_coop_percent': [], 'mean_absolute_jerk': []}
    counts = []
    for seed in SEEDS:
        directory = (output/'latest_cooperation'/f'v4_{people}h_seed{seed}' if model == 'ours' else
            COOP/'latest_cooperation'/f'coopft_6000_{people}h_seed{seed}' if model == 'coopft' else
            PRIOR/f'{model}_{people}h_seed{seed}')
        with np.load(directory/'trials.npz') as saved:
            c = saved['20__tcoop'].astype(np.float64)*100
            j = saved['20__jerk'].astype(np.float64)
        arrays['t_coop_percent'].append(c[np.isfinite(c)])
        arrays['mean_absolute_jerk'].append(j[j != 0])
        counts.append(read(directory/'result.json')['variants']['20'])
    result = {}
    for name, parts in arrays.items():
        joined = np.concatenate(parts)
        result[name] = dict(stats([float(p.mean()) if len(p) else None for p in parts]),
            pooled_mean=float(joined.mean()) if len(joined) else None, valid_per_seed=[len(p) for p in parts])
    result['counts'] = counts
    return result


def summarize(output):
    manifest = read(output/'manifest.json')
    models, runs, missing = {}, [], []
    for model in LABELS:
        data = dict(table1={}, table2={})
        for table, names in (('table1', [f'{mode}_{n}h' for n in (2, 4, 8, 16)
                                       for mode in ('independent', 'mixed')]), ('table2', ENABLE+COUPLED)):
            for name in names:
                paths = [source(output, model, table, name, seed) for seed in SEEDS]
                if not all((p/'result.json').is_file() for p in paths):
                    missing.append(dict(model=model, table=table, name=name))
                    continue
                rows = [read(p/'result.json') for p in paths]
                for seed, row, path in zip(SEEDS, rows, paths):
                    if (row['seed'], row['num_envs'], row['horizon'], row['simulated_steps'], row['xy_extent_percent']) != (
                            seed, 512, 900, 900, 100):
                        raise ValueError('Reused result dimensions differ: '+str(path))
                    if table == 'table1' and name.endswith(('8h', '16h')):
                        if row.get('group_spawn_radius') != (7.5 if name.endswith('16h') else 3.):
                            raise ValueError('Reused result has old placement: '+str(path))
                    if model == 'ours':
                        proof = read(output/'checks'/table/name/f'seed{seed}.json')
                        if not all(proof[k] for k in ('first_policy_input_exact', 'policy_preserved',
                                                     'cpu_gpu_sr_exact', 'cpu_gpu_errors_match')):
                            raise ValueError('Missing V4 verification')
                    runs.append(dict(model=model, table=table, name=name, seed=seed, source=str(path),
                        scores=row['interaction_group_scores']['region'], errors=row['errors']))
                item = {metric: stats([100*r['interaction_group_scores']['region'][metric]['rate']
                    if r['interaction_group_scores']['region'][metric]['rate'] is not None else None for r in rows])
                    for metric in METRICS}
                if table == 'table2':
                    tables = [read(p/'table2.json') for p in paths]
                    item['completion_time_s'] = stats([r.get('completion_time_s') for r in tables])
                    item['successes'] = [r['successes'] for r in tables]
                    if name.startswith('coupled'):
                        item.update(latest(output, model, int(name[-1])))
                data[table][name] = item
        if all(name in data['table2'] for name in ENABLE):
            data['enabling_average'] = {metric: macro([data['table2'][name] for name in ENABLE], metric)
                for metric in (METRICS[1], 'completion_time_s')}
        if all(name in data['table2'] for name in COUPLED):
            parts = [data['table2'][name] for name in COUPLED]
            data['coupled_average'] = {METRICS[2]: macro(parts, METRICS[2])}
            for metric in ('t_coop_percent', 'mean_absolute_jerk'):
                data['coupled_average'][metric] = dict(macro(parts, metric),
                    pooled_mean=statistics.mean(p[metric]['pooled_mean'] for p in parts))
        models[model] = data
    summary = dict(protocol_id=manifest['protocol_id'], envs=512, seeds=SEEDS, horizon=900,
        group_spawn_radii_m={8: 3., 16: 7.5}, models=models, seed_runs=runs, missing_baseline_settings=missing,
        checkpoint=manifest['checkpoint'], checkpoint_sha256=manifest['checkpoint_sha256'])
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')

    def fmt(item):
        return 'N/A' if item is None or item['mean'] is None else f"{item['mean']:.2f}±{item['std']:.2f}"

    lines = ['# V4 2k 채택 · Table 1/2', '',
        '512환경 × seed1/6/7 × 900step·XY100%·M=2N. Ours 신규42회. 8H R=3m /16H R=7.5m.', '',
        '## Table 1 · Success Rate (%)', '',
        '| Method | 2H Ind. | 2H Mixed | 4H Ind. | 4H Mixed | 8H Ind. | 8H Mixed | 16H Ind. | 16H Mixed |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for model, data in models.items():
        values = [fmt(data['table1'].get(f'{mode}_{n}h', {}).get(METRICS[0 if mode == 'independent' else 3]))
                  for n in (2, 4, 8, 16) for mode in ('independent', 'mixed')]
        lines.append('| '+LABELS[model]+' | '+' | '.join(values)+' |')
    lines += ['', '## Table 2(a) · Enabling', '',
        '| Method | AT→SIT SR (%) | AT→CLIMB SR (%) | AT→ON_TOP SR (%) | Avg. SR (%) | Completion Time (s) ↓ |',
        '|---|---:|---:|---:|---:|---:|']
    for model, data in models.items():
        values = [fmt(data['table2'].get(name, {}).get(METRICS[1])) for name in ENABLE]
        average = data.get('enabling_average', {})
        values += [fmt(average.get(METRICS[1])), fmt(average.get('completion_time_s'))]
        lines.append('| '+LABELS[model]+' | '+' | '.join(values)+' |')
    lines += ['', '## Table 2(b) · Coupled', '',
        '| Method | 2H SR (%) | 3H SR (%) | 4H SR (%) | Avg. SR (%) | t_coop pooled (%) ↑ | Jerk pooled (m/s³) ↓ |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for model, data in models.items():
        values = [fmt(data['table2'].get(name, {}).get(METRICS[2])) for name in COUPLED]
        average = data.get('coupled_average', {})
        values.append(fmt(average.get(METRICS[2])))
        values += [f"{average[key]['pooled_mean']:.2f}" if key in average else 'N/A'
                   for key in ('t_coop_percent', 'mean_absolute_jerk')]
        lines.append('| '+LABELS[model]+' | '+' | '.join(values)+' |')
    lines += ['', 'SR/Time은 seed 평균±표본표준편차(ddof=1). Enable Time은 성공 trial의 최초 유효 동시 성공 시간이며 행동별 동일 가중 평균이다.',
        't_coop/Jerk 주 값은20cm 출발 조건의 seed trial 합산→인원별 평균→2/3/4명 동일 가중 평균이다. 별도 seed 통계·표본 수·제외/clip 수는 summary.json에 저장한다.',
        'TokenHSI는 고정397D 입력 차이(fully_protocol_compliant=false), FSM은 기존 첫 정책 관측 불일치가 남은 참고값이다. 기존 결과를 수정하거나 이 차이를 보정하지 않았다.',
        '반경 조건의 baseline이 미완료면 해당 칸은 N/A이며 완료 후 이 집계기만 다시 실행한다. 64환경 탐색·이전5m 간격 결과는 정식 표에 넣지 않는다.', '',
        '[원시 통계](summary.json) · [Manifest](manifest.json) · [검증](completion_verification.json)']
    (output/'report.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    import sys
    summarize(Path(sys.argv[1]).resolve())
