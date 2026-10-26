"""Compare the FT trajectory against verified zero-shot and 20k endpoints."""
from pathlib import Path
import json,sys,datetime
import numpy as np
import yaml
root=Path(sys.argv[1]);base=root.parent
paths={'zero-shot':base/'tokenhsi_climb_78k_zeroshot_20261025','FT5k':root/'ft_05000','FT10k':root/'ft_10000','FT15k':root/'ft_15000','FT20k':base/'tokenhsi_climb_20k_official_20261025'}
reference=yaml.safe_load((paths['FT20k']/'env.yaml').read_text());summary={};verification={}
for name,p in paths.items():
    assert yaml.safe_load((p/'env.yaml').read_text())==reference,(name,'config differs')
    files=list(p.glob('metrics_*.json'));assert len(files)==1
    metrics=json.loads(files[0].read_text());checks={}
    for i in range(3):
        t=np.load(p/f'trace_{i}.npz');distance=np.linalg.norm(t['root'].astype('float64')-t['target'].astype('float64'),axis=-1)
        assert distance.shape[1:]==(512,) and len(distance)<=299 and np.isfinite(distance).all()
        good=((distance<=.2)&t['alive']).any(0);minimum=np.where(t['alive'],distance,np.inf).min(0)
        m=metrics[f'ObjectSet_test_{i}'];assert m['num_trials']==512 and int(good.sum())==m['success_trials']
        assert abs(float(minimum[good].mean())-m['success_precision'])<1e-7
        assert ((t['done'].astype(bool)&t['alive']).sum(0)==1).all()
        checks[str(i)]=dict(success_count=int(good.sum()),precision_float64=float(minimum[good].mean()),first_episode_verified=True,finite=True)
    rates=[metrics[f'ObjectSet_test_{i}']['success_rate'] for i in range(3)]
    row=dict(success_rate=float(np.mean(rates[1:])),raw_success_rates=rates,precision_m=float(np.mean([metrics[f'ObjectSet_test_{i}']['success_precision'] for i in (1,2)])),artifact=str(p.resolve()))
    row['delta_vs_zero_shot_percentage_points']=100*(row['success_rate']-summary['zero-shot']['success_rate']) if name!='zero-shot' else 0.
    summary[name]=row;verification[name]=checks
best=max(summary,key=lambda k:summary[k]['success_rate'])
result=dict(summary=summary,best_success_rate_checkpoint=best,verification=verification,env_sim_exact_match=True,actual_mixture=True)
(root/'summary.json').write_text(json.dumps(result,indent=2))
report='''# CLIMB FT 체크포인트 비교 — 실제 물체 혼합

조건:78k출발모델·1명/1고정물체,test Box/Cabinet/Table_Square각1/3확률혼합,512환경/seed0/프로세스내3repeat,loco100%,300설정step(실제최대299≈9.97초),IET off,골반목표20cm ever. 종류별성공률을산술평균한추정치가아니라실제혼합환경에서측정했다. 학습추가실행없음.

5k/10k/15k를새로평가하고zero-shot/20k는같은평가기와조건의기존원시결과를재검증해재사용했다. 전5조건env/sim YAML exact동일. 표는0회차제외repeat1·2평균이며학습seed산포가아니다.

| 정책 | repeat0 | repeat1 | repeat2 | 주 성공률 | zero-shot 대비 | 성공 시 최소 목표 거리 |
|---|---:|---:|---:|---:|---:|---:|
'''
for name,row in summary.items():
    rates=row['raw_success_rates'];report+=f"| {name} | {rates[0]*100:.2f}% | {rates[1]*100:.2f}% | {rates[2]*100:.2f}% | {row['success_rate']*100:.2f}% | {row['delta_vs_zero_shot_percentage_points']:+.2f}%p | {row['precision_m']*100:.2f}cm |\n"
report+=f"\n관측된최고성공률은 **{best} {summary[best]['success_rate']*100:.2f}%**. 단일학습정책/단일평가seed의반복이며작은차이의통계적우열이나연장학습효과를확정하지않는다.\n\n전15trace를독립NumPy64로재채점해성공수/precision일치·finite·첫episode종료각1회·10초길이검증. 원시eval.log/metrics/trace보존. summary.json에조건별원시경로와재채점기록이있다.\n"
(root/'REPORT.md').write_text(report)
(root/'completed.json').write_text(json.dumps(dict(finished_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),new_eval_exit_codes=dict(ft5000=0,ft10000=0,ft15000=0),reuse=['zero-shot','FT20k'],verified_traces=15),indent=2))
print(json.dumps(dict(summary=summary,best=best),indent=2))
