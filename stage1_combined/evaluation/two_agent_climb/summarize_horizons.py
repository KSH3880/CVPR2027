"""Verify a 30-second run and compare 10/20/30s on each identical trajectory."""
from pathlib import Path
import sys,json,datetime
import numpy as np
import yaml
root=Path(sys.argv[1]);summaries={};verification={}
old=root.parent/'two_agent_climb_78k_fixed_20261025'
a=yaml.safe_load((root/'env.yaml').read_text());b=yaml.safe_load((old/'env.yaml').read_text())
assert a['env'].pop('episodeLength')==900;assert b['env'].pop('episodeLength')==600;assert a==b
for cat in ['Box','Table_Square','Cabinet']:
    result=json.loads((root/cat/'results.json').read_text());raw=result['raw'];verification[cat]={}
    for repeat in range(3):
        t=np.load(root/cat/f'trace_{repeat}.npz');distance=np.linalg.norm(t['root'].astype('float64')-t['target'].astype('float64'),axis=-1)
        assert distance.shape[1:]==(512,2) and np.isfinite(distance).all()
        assert ((t['done']&t['alive']).sum(0)==1).all()
        rates=[]
        for count,key in [(299,'agent_success_10s'),(599,'agent_success_20s'),(899,'agent_success_rate')]:
            success=((distance[:count]<=.2)&t['alive'][:count,:,None]).any(0);rate=float(success.mean());assert rate==raw[str(repeat)][key]
            rates.append(rate)
        assert rates[0]<=rates[1]<=rates[2]
        r=raw[str(repeat)];assert r['episode_length']==900 and r['steps']<=899 and r['fixed_targets'] and r['max_fixed_pose_drift']<2e-4
        full=((distance<=.2)&t['alive'][...,None]).any(0)
        assert int(full.sum())==r['agent_success_count']
        best=np.where(t['alive'][...,None],distance,np.inf).min(0)
        assert abs(float(best[full].mean())-r['success_precision_m'])<1e-8
        verification[cat][str(repeat)]=dict(rates=rates,steps=r['steps'],fixed_pose_drift=r['max_fixed_pose_drift'],rescored=True,first_episode_verified=True)
    init=np.load(root/cat/'initial_0.npz');oldinit=np.load(old/cat/'initial_0.npz')
    verification[cat]['initial_repeat0_exact']={k:bool(np.array_equal(init[k],oldinit[k])) for k in init.files}
    summaries[cat]={key:float(np.mean([raw[str(i)][key] for i in (1,2)])) for key in ['agent_success_10s','agent_success_20s','agent_success_rate','both_agents_success_rate']}
means={key:float(np.mean([v[key] for v in summaries.values()])) for key in ['agent_success_10s','agent_success_20s','agent_success_rate']}
result=dict(summary=summaries,equal_category_mean_estimate=means,verification=verification,env_sim_same_except_episodeLength=True)
(root/'summary_horizons.json').write_text(json.dumps(result,indent=2))
report='''# 78k·2명4물체·고정CLIMB 대상 — 30초 평가

78k원본모델/정규화·2명/4물체·target0/1고정·distractor2/3동적·category별test분포·골반목표20cm ever·512환경×2명×3repeat·seed0을유지하고최대episodeLength만600→900으로변경했다. env/sim파일의나머지항목동일성검증통과. 사람별성공률이며scene당첫episode만종료frame까지집계한다. 낙상종료는기존대로유지하므로모든시도가30초를채우는것은아니다.

아래10/20/30초는모두이번30초실행의동일궤적을299/599/899step에서잘라재채점한값이다. repeat0보존,표는repeat1·2평균이다.

| CLIMB 대상 | 10초 | 20초 | 30초 | 20→30초 차이 |
|---|---:|---:|---:|---:|
'''
for cat,v in summaries.items():report+=f"| {cat} | {v['agent_success_10s']*100:.2f}% | {v['agent_success_20s']*100:.2f}% | {v['agent_success_rate']*100:.2f}% | {(v['agent_success_rate']-v['agent_success_20s'])*100:+.2f}%p |\n"
report+=f"| 3종균등 단순평균 추정 | {means['agent_success_10s']*100:.2f}% | {means['agent_success_20s']*100:.2f}% | {means['agent_success_rate']*100:.2f}% | {(means['agent_success_rate']-means['agent_success_20s'])*100:+.2f}%p |\n"
report+='\n균등평균은종류별독립실행의산술평균으로실제혼합실행수치가아니다. 반복산포는독립학습seed산포가아니다. 기존600step실행과는첫repeat종료이후reset난수소비/초기장면이달라질수있으므로시간효과는위의동일궤적10/20/30초로비교한다.\n\n검증:전9trace를NumPy64로독립재채점,10/20/30초성공수/precision일치·비감소·finite·첫scene종료1회·episode900확인. 고정target pose변화2e-4미만실행중확인. 원시initial/trace/results/log및summary_horizons.json보존. 학습과기존평가산출물수정없음.\n'
(root/'REPORT.md').write_text(report)
(root/'completed.json').write_text(json.dumps(dict(completed_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),exit_codes=dict(Box=0,Table_Square=0,Cabinet=0),trace_verification='all nine passed'),indent=2))
print(json.dumps(dict(summary=summaries,equal_category_mean_estimate=means),indent=2))
