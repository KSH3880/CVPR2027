import fcntl
import json
from pathlib import Path

import numpy as np
import torch

from ind_mechanisms_common import OUT, write

SEEDS = (1,6,7)
MODELS = ('sct','ours')


def stats(values):
    return dict(mean=float(np.mean(values)), std=float(np.std(values,ddof=1)), seeds=list(map(float,values)))


def paired_ci(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(314159)
    boot = np.zeros(2000)
    for seed_values in values:
        sampled = rng.integers(0,len(seed_values),size=(2000,len(seed_values)))
        boot += seed_values[sampled].mean(axis=tuple(range(1,seed_values[sampled].ndim)))/len(values)
    return dict(mean=float(values.mean()), ci95=list(map(float,np.quantile(boot,[.025,.975]))),
                unit='original scene; both people sampled together; stratified by evaluation seed')


def main():
    lock=(OUT/'.summary.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX)
    runs, metrics, trials = {}, {}, {}
    for condition in ('baseline','solo_a','solo_b','separated','cross'):
        for model in MODELS:
            for seed in SEEDS:
                path = OUT/f'baseline/{model}_seed{seed}' if condition=='baseline' else OUT/f'runs/{condition}/{model}/seed{seed}'
                marker = 'source.json' if condition=='baseline' else 'verification.json'
                if not (path/marker).exists() or not (path/'metrics.json').exists():
                    continue
                key = (condition,model,seed)
                runs[key] = path
                metrics[key] = json.loads((path/'metrics.json').read_text())
                with np.load(path/'trial_metrics.npz') as archive:
                    trials[key] = {k:archive[k] for k in archive.files}
    for condition in ('solo_a','solo_b','separated','cross'):
        for seed in SEEDS:
            keys = [(condition,model,seed) for model in MODELS]
            if not all(k in runs for k in keys):
                continue
            for name in ('first_policy_input.pt','initial.pt'):
                a,b = [torch.load(runs[k]/name,map_location='cpu',weights_only=False) for k in keys]
                matches = torch.equal(a,b) if torch.is_tensor(a) else a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)
                if not matches:
                    raise ValueError(f'Models received different initial states: {condition} seed{seed} {name}')
    for model in MODELS:
        for seed in SEEDS:
            keys = [(c,model,seed) for c in ('solo_a','solo_b')]
            if not all(k in trials for k in keys):
                continue
            trials[('solo',model,seed)] = {k:np.concatenate([trials[j][k] for j in keys],axis=1) for k in trials[keys[0]]}
    comparisons = {}
    if all(('solo',model,seed) in trials and ('baseline',model,seed) in trials for model in MODELS for seed in SEEDS):
        solo_diff, pair_diff, interaction, loss = [], [], [], {m:[] for m in MODELS}
        for seed in SEEDS:
            solo = {m:trials[('solo',m,seed)]['success'].astype(float) for m in MODELS}
            pair = {m:trials[('baseline',m,seed)]['success'].astype(float) for m in MODELS}
            solo_diff.append(solo['ours']-solo['sct'])
            pair_diff.append(pair['ours']-pair['sct'])
            interaction.append((pair['ours']-solo['ours'])-(pair['sct']-solo['sct']))
            for model in MODELS:
                loss[model].append((solo[model]-pair[model]).mean())
        comparisons['a'] = dict(solo_ours_minus_sct=paired_ci(solo_diff), pair_ours_minus_sct=paired_ci(pair_diff),
            reduced_interference_penalty=paired_ci(interaction), solo_minus_pair={m:stats(v) for m,v in loss.items()})
    if all((c,m,s) in trials for c in ('separated','cross') for m in MODELS for s in SEEDS):
        differences=[]
        for seed in SEEDS:
            gain={m:trials[('cross',m,seed)]['success'].astype(float)-trials[('separated',m,seed)]['success'].astype(float) for m in MODELS}
            differences.append(gain['ours']-gain['sct'])
        comparisons['b'] = dict(reduced_crossing_penalty=paired_ci(differences))
    aggregates={}
    for condition in ('baseline','solo','separated','cross'):
        for model in MODELS:
            keys=[(condition,model,s) for s in SEEDS]
            if not all(k in trials for k in keys):
                continue
            fields=('success','prefall','hold05','hold1','time_all')
            row={field:stats([trials[k][field].mean() for k in keys]) for field in fields}
            row['time_success']=(stats([trials[k]['time_all'][trials[k]['success']].mean() for k in keys])
                                 if all(trials[k]['success'].any() for k in keys) else None)
            row['both_success']=stats([trials[k]['success'].all(-1).mean() for k in keys])
            row['tasks']={}
            for kind,name in enumerate(('SIT','CLIMB','AT','ON_TOP','PUSH')):
                if not all((trials[k]['types']==kind).any() for k in keys):
                    continue
                row['tasks'][name]={field:stats([trials[k][field][trials[k]['types']==kind].mean() for k in keys]) for field in fields}
                if kind in (2,4):
                    row['tasks'][name]['tolerance_sr']={str(cm):stats([(trials[k]['minimum_point_error_m'][trials[k]['types']==kind] <= cm/100).mean() for k in keys]) for cm in (10,20,30,50)}
                    row['tasks'][name]['within50_but_not20']=stats([((trials[k]['minimum_point_error_m'][trials[k]['types']==kind] <= .5) & (trials[k]['minimum_point_error_m'][trials[k]['types']==kind] > .2)).mean() for k in keys])
            for field in trials[keys[0]]:
                if '_near_' in field:
                    row[field]=stats([trials[k][field].mean() for k in keys])
            if all('zone_frames' in trials[k] for k in keys):
                row['zone_entered_rate']=stats([(trials[k]['zone_frames']>0).mean() for k in keys])
                row['travel_m']=stats([trials[k]['travel_m'].mean() for k in keys])
                if all((trials[k]['zone_frames']>0).any() for k in keys):
                    for field,label in [('zone_speed_sum','zone_speed_entered'),('zone_slow_frames','zone_slow_fraction_entered')]:
                        row[label]=stats([(trials[k][field][trials[k]['zone_frames']>0]/trials[k]['zone_frames'][trials[k]['zone_frames']>0]).mean() for k in keys])
            aggregates[f'{condition}/{model}']=row
    matched_times={}
    for condition in ('baseline','solo','separated','cross'):
        if not all((condition,m,s) in trials for m in MODELS for s in SEEDS):
            continue
        times={m:[] for m in MODELS}
        counts=[]
        for seed in SEEDS:
            common=trials[(condition,'sct',seed)]['success'] & trials[(condition,'ours',seed)]['success']
            counts.append(int(common.sum()))
            if common.any():
                for model in MODELS:
                    times[model].append(float(trials[(condition,model,seed)]['time_all'][common].mean()))
        if all(counts):
            matched_times[condition]=dict(counts_by_seed=counts, **{m:stats(v) for m,v in times.items()},
                ours_minus_sct=stats(np.array(times['ours'])-np.array(times['sct'])))
    summary=dict(verified_runs=len([k for k in runs if k[0]!='baseline']), expected_runs=24,
        aggregates=aggregates, comparisons=comparisons, common_success_times=matched_times,
        limitations=['Fixed checkpoints; evaluation-seed variation only.',
            'Solo changes token count; A is not a pure physical-collision intervention.',
            'B is a controlled carry-only distribution, separate from Table 1.',
            'Distances between body centers and box surfaces are proximity proxies, not contact-pair measurements.',
            'No training control; positive transfer from cooperative training is not causally established.'])
    write(OUT/'summary.json',summary)
    lines=['# IND mechanism evaluation · SCT 26k / Ours V4 2k','',
        f'검증 완료 {summary["verified_runs"]}/24회. 512환경 × seed 1/6/7 × 900step. GPU 6/7.',
        '', '| 조건 | 모델 | SR (%) | 성공 전 낙상 (%) | 1초 유지 성공 (%) | Time Succ. / All (s) |',
        '|---|---|---:|---:|---:|---:|']
    for key,row in aggregates.items():
        condition,model=key.split('/')
        cells=[f'{100*row[k]["mean"]:.2f}±{100*row[k]["std"]:.2f}' for k in ('success','prefall','hold1')]
        success_time=f'{row["time_success"]["mean"]:.2f}' if row['time_success'] else 'N/A'
        lines.append(f'| {condition} | {model} | '+ ' | '.join(cells)+f' | {success_time} / {row["time_all"]["mean"]:.2f} |')
    for section, entries in comparisons.items():
        lines.extend(['',f'## 실험 {section.upper()} · 짝지은 차이',''])
        for name,value in entries.items():
            if 'ci95' in value:
                lo,hi=value['ci95'];lines.append(f'- {name}: {100*value["mean"]:.2f}%p, scene bootstrap 95% CI [{100*lo:.2f}, {100*hi:.2f}].')
    if 'solo/ours' in aggregates and 'solo/sct' in aggregates:
        lines.extend(['','## 단독 과제별 성공률','', '| 과제 | SCT (%) | Ours (%) |','|---|---:|---:|'])
        for task in ('SIT','CLIMB','AT','ON_TOP','PUSH'):
            cells=[f'{100*aggregates["solo/"+model]["tasks"][task]["success"]["mean"]:.2f}' for model in MODELS]
            lines.append('| '+task+' | '+' | '.join(cells)+' |')
        lines.extend(['','## 단독 목표 허용오차별 성공률','', '| 과제/모델 | 10cm | 20cm | 30cm | 50cm |','|---|---:|---:|---:|---:|'])
        for task in ('AT','PUSH'):
            for model in MODELS:
                values=aggregates['solo/'+model]['tasks'][task]['tolerance_sr']
                cells=[f'{100*values[str(cm)]["mean"]:.2f}' for cm in (10,20,30,50)]
                lines.append('| '+task+'/'+model+' | '+' | '.join(cells)+' |')
    if 'cross/ours' in aggregates and 'cross/sct' in aggregates:
        lines.extend(['','## 교차 실험 보조지표','',
            '| 조건/모델 | 둘 다 성공 (%) | 몸체 중심간20cm 근접 (%) | 상대 상자 표면12cm 근접 (%) | 중점 구간 진입 (%) |',
            '|---|---:|---:|---:|---:|'])
        for condition in ('separated','cross'):
            for model in MODELS:
                row=aggregates.get(condition+'/'+model)
                if row and 'zone_entered_rate' in row:
                    cells=[f'{100*row[k]["mean"]:.2f}' for k in ('both_success','human_near_20','box_near_12','zone_entered_rate')]
                    lines.append('| '+condition+'/'+model+' | '+' | '.join(cells)+' |')
        lines.extend(['','구간은 각자의 계획 운반 중점 반경1.5m이며, 구간 내 속도/저속(<0.2m/s) 비율은 진입한 trial에 조건부로 summary.json에 저장한다.'])
    lines.extend(['','목표 허용오차·행동별 통계는 summary.json에 저장한다. 공식 SR은 기존 ever 판정이며 0.5/1초 유지율은 별도 보조지표다.',
        '두 모델이 모두 성공한 동일 표본의 완료 시간과 표본 수는 common_success_times에 별도 저장한다.',
        '', '근접 지표는 실제 충돌률이 아니다. 단독 조건은 관측 토큰 수도 달라지며, A의 차이를 원인별 기여율로 해석하지 않는다. 이 평가만으로 협력 학습의 인과적 전이를 입증하지 않는다.'])
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    if 'solo/ours' in aggregates and 'solo/sct' in aggregates:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figure,axes=plt.subplots(1,2,figsize=(8,3.3))
        for ax,task in zip(axes,('AT','PUSH')):
            for model in MODELS:
                values=aggregates['solo/'+model]['tasks'][task]['tolerance_sr']
                ax.errorbar([10,20,30,50],[100*values[str(x)]['mean'] for x in (10,20,30,50)],
                    yerr=[100*values[str(x)]['std'] for x in (10,20,30,50)],marker='o',label=model)
            ax.set(title=task,xlabel='Goal tolerance (cm)',ylabel='Success (%)',ylim=(0,100))
            ax.legend()
        figure.tight_layout();figure.savefig(OUT/'solo_precision.pdf');plt.close(figure)


if __name__=='__main__':
    main()
