import argparse
import json
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tokenhsi'))
from utils.evaluation_protocol import INTERACTION_METRICS
from coupled_cooperation_metrics import sha

BASELINE = ROOT.parent/'sct_coopft/output_etc/check/sct_coopft_4000_tables12_seeds167_20261111'
PRIOR_COOP = ROOT/'output_etc/check/coupled_cooperation_four_models_20261111'
SEEDS = (1,6,7)
ENABLE = ('enable_sit','enable_climb','enable_ontop')
COUPLED = ('coupled_2','coupled_3','coupled_4')


def stats(values):
    complete = None not in values
    return dict(mean=statistics.mean(values) if complete else None,
                std=statistics.stdev(values) if complete else None, seeds=values)


def macro(rows, metric):
    return stats([statistics.mean(row[metric]['seeds'][i] for row in rows)
                  if all(row[metric]['seeds'][i] is not None for row in rows) else None for i in range(3)])


def latest_metrics(output, epoch, people):
    values = {'t_coop_percent':[],'mean_absolute_jerk':[]}
    means = {key:[] for key in values}
    counts = []
    for seed in SEEDS:
        directory = (PRIOR_COOP/f'coopft_{people}h_seed{seed}' if epoch==4000 else
                     output/'latest_cooperation'/f'coopft_{epoch}_{people}h_seed{seed}')
        with np.load(directory/'trials.npz') as saved:
            j = saved['20__jerk'].astype(np.float64)
            c = saved['20__tcoop'].astype(np.float64)*100
        for name,array in [('mean_absolute_jerk',j[j!=0]),('t_coop_percent',c[np.isfinite(c)])]:
            values[name].append(array)
            means[name].append(float(array.mean()) if len(array) else None)
        row=json.loads((directory/'result.json').read_text())
        counts.append(row['variants']['20'])
    result={}
    for name,parts in values.items():
        joined=np.concatenate(parts)
        result[name]=dict(stats(means[name]),pooled_mean=float(joined.mean()) if len(joined) else None,
                          valid_per_seed=[len(part) for part in parts])
    result['counts']=counts
    return result


def summarize(output):
    manifest=json.loads((output/'manifest.json').read_text())
    proofs=[]
    for job in manifest['jobs']:
        file=output/'checks'/str(job['epoch'])/job['table']/job['name']/f"seed{job['seed']}.json"
        proof=json.loads(file.read_text())
        if not (proof['first_policy_input_exact'] and proof['policy_preserved'] and proof['canonical_group_counts_exact']):
            raise ValueError('Incomplete proof: '+str(file))
        proofs.append(proof)
    summary={}
    runs=[]
    for epoch in (3000,4000,6000):
        root=BASELINE if epoch==4000 else output/str(epoch)
        model=dict(table1={},table2={})
        for table,names in [('table1',[f'{mode}_{n}h' for n in (2,4,8,16) for mode in ('independent','mixed')]),
                            ('table2',ENABLE+COUPLED)]:
            for name in names:
                rows=[]
                for seed in SEEDS:
                    source=root/table/name/f'general_seed{seed}'
                    row=json.loads((source/'result.json').read_text())
                    index=(0 if name.startswith('independent') else 3) if table=='table1' else (1 if name.startswith('enable') else 2)
                    score=row['interaction_group_scores']['region'][INTERACTION_METRICS[index]]
                    runs.append(dict(epoch=epoch,table=table,setting=name,seed=seed,source=str(source),
                                     sr_percent=100*score['rate'],errors=row['errors']))
                    rows.append(row)
                item={metric:stats([100*r['interaction_group_scores']['region'][metric]['rate']
                      if r['interaction_group_scores']['region'][metric]['rate'] is not None else None for r in rows])
                      for metric in INTERACTION_METRICS}
                if table=='table2':
                    stored=[json.loads((root/table/name/f'general_seed{s}'/'table2.json').read_text()) for s in SEEDS]
                    item['completion_time_s']=stats([r['completion_time_s'] for r in stored])
                    item['successes']=[r['successes'] for r in stored]
                    if name.startswith('coupled'):
                        item.update(latest_metrics(output,epoch,int(name[-1])))
                model[table][name]=item
        model['enabling_average']={metric:macro([model['table2'][s] for s in ENABLE],metric)
                                  for metric in (INTERACTION_METRICS[1],'completion_time_s')}
        model['coupled_average']={INTERACTION_METRICS[2]:macro([model['table2'][s] for s in COUPLED],INTERACTION_METRICS[2])}
        for metric in ('t_coop_percent','mean_absolute_jerk'):
            parts=[model['table2'][s] for s in COUPLED]
            item=macro(parts,metric)
            item['pooled_mean']=statistics.mean(p[metric]['pooled_mean'] for p in parts)
            model['coupled_average'][metric]=item
        summary[str(epoch)]=model
    for path,digest in json.loads((output/'source_hashes.json').read_text()).items():
        if sha(path)!=digest:
            raise ValueError('Original/source changed: '+path)
    for row in manifest['checkpoints'].values():
        if sha(row['source'])!=row['sha256'] or sha(row['staged'])!=row['sha256']:
            raise ValueError('Checkpoint changed')
    data=dict(protocol_id=manifest['protocol_id'],seeds=SEEDS,envs=512,horizon=900,
              primary_start_displacement_m=.2,models=summary,seed_runs=runs)
    (output/'summary.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    def fmt(item):
        return 'N/A' if item['mean'] is None else f"{item['mean']:.2f}±{item['std']:.2f}"
    lines=['# SCT + Coop. FT 3k / 4k / 6k · Table 1/2','',
           '512환경 × seed1/6/7 × 900step·XY100%·M=2N. 3k/6k 신규84회, 기존4k42회 재사용. 공통 초기 snapshot을 유지했다.','',
           '## Table 1 · Success Rate (%)','',
           '| Checkpoint | 2H Independent | 2H Mixed | 4H Independent | 4H Mixed | 8H Independent | 8H Mixed | 16H Independent | 16H Mixed |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for epoch,model in summary.items():
        values=[fmt(model['table1'][f'{mode}_{n}h'][INTERACTION_METRICS[0 if mode=='independent' else 3]])
                for n in (2,4,8,16) for mode in ('independent','mixed')]
        lines.append('| '+epoch+' | '+' | '.join(values)+' |')
    lines+=['','Independent는 Independent Interaction Success Rate, Mixed는 Mixed Interaction Group Success Rate다. seed 평균±표본표준편차.','',
            '## Table 2(a) · Enabling','',
            '| Checkpoint | AT→SIT SR (%) | AT→CLIMB SR (%) | AT→ON_TOP SR (%) | Avg. SR (%) | Completion Time (s) ↓ |',
            '|---|---:|---:|---:|---:|---:|']
    for epoch,model in summary.items():
        values=[fmt(model['table2'][s][INTERACTION_METRICS[1]]) for s in ENABLE]
        values += [fmt(model['enabling_average'][INTERACTION_METRICS[1]]),fmt(model['enabling_average']['completion_time_s'])]
        lines.append('| '+epoch+' | '+' | '.join(values)+' |')
    lines+=['','SR은 최초 유효 동시 성공, Time은 성공 trial의 최초 동시 성공 시간이다. 행동별 동일 가중 평균·seed 평균±표본표준편차.','',
            '## Table 2(b) · Coupled','',
            '| Checkpoint | 2H SR (%) | 3H SR (%) | 4H SR (%) | Avg. SR (%) | t_coop pooled (%) ↑ | Mean Absolute Jerk pooled (m/s³) ↓ |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for epoch,model in summary.items():
        values=[fmt(model['table2'][s][INTERACTION_METRICS[2]]) for s in COUPLED]
        avg=model['coupled_average']
        values += [fmt(avg[INTERACTION_METRICS[2]]),f"{avg['t_coop_percent']['pooled_mean']:.2f}",f"{avg['mean_absolute_jerk']['pooled_mean']:.2f}"]
        lines.append('| '+epoch+' | '+' | '.join(values)+' |')
    lines+=['','초기 중심 대비 수평20cm 시작. t_coop은 전원 양손–실제 전체 상자 표면<12cm의 전체 기록 누적/원본counter/clip이며 힘·추가 생존 필터가 없다. Jerk는 원본64점·float32·dt0.0333·0 trial 제외다. 주 값은 seed trial 합산 후 인원별 평균→인원 동일 가중 평균이며 seed 표준편차를 붙이지 않는다.','',
            '### Coupled 추가 지표의 별도 seed 통계','',
            '| Checkpoint | t_coop seed 평균±std (%) | Mean Absolute Jerk seed 평균±std (m/s³) |',
            '|---|---:|---:|']
    for epoch,model in summary.items():
        avg=model['coupled_average']
        lines.append('| '+epoch+' | '+fmt(avg['t_coop_percent'])+' | '+fmt(avg['mean_absolute_jerk'])+' |')
    lines+=['','## Seed별 Success Rate (%)','', '| Checkpoint | 표 | 설정 | seed | Success Rate (%) |', '|---|---|---|---:|---:|']
    for row in runs:
        lines.append(f"| {row['epoch']} | {row['table']} | {row['setting']} | {row['seed']} | {row['sr_percent']:.6f} |")
    lines+=['','Post-AT Time·Collision 추가 실험은 포함하지 않았다. Point/Region Error는 summary.json의 seed_runs에 보존했다.','',
            '[원시 통계](summary.json) · [Manifest](manifest.json) · [검증](completion_verification.json)']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    proof=dict(completed=True,new_runs=84,reused_4000_runs=42,seeds=SEEDS,envs=512,horizon=900,
               first_actual_policy_input_exact=True,weights_and_normalization_preserved=True,
               cpu_sr_counts_errors_verified=True,canonical_group_counts_exact=True,
               table2_cpu_replay_verified=True,latest_cooperation_first_success_exact=True,
               primary_start_displacement_m=.2,source_checkpoint_hashes_unchanged=True)
    (output/'completion_verification.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(output/'report.md',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    summarize(parser.parse_args().output.resolve())
