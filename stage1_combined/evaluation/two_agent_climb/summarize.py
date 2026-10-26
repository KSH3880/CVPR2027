"""Rescore saved trajectories and summarize all target substitutions."""
from pathlib import Path
import sys,json
import numpy as np
root=Path(sys.argv[1]);summaries={};validation={}
base=np.load(root/'Box/initial_0.npz')['size']
for category in ['Box','Table_Square','Cabinet']:
    result=json.loads((root/category/'results.json').read_text());raw=result['raw'];validation[category]={}
    if raw['0'].get('episode_length',600)>600:raise ValueError('Use summarize_horizons.py for 30-second runs')
    initial=np.load(root/category/'initial_0.npz')
    assert initial['root'].shape[:2]==(512,2) and initial['objects'].shape[:2]==(512,4)
    np.testing.assert_array_equal(initial['size'][:,2:],base[:,2:])
    if category!='Box':
        assetroot=Path(__file__).resolve().parents[2]/'tokenhsi/data/dataset_amass_climb/objects/test'/category
        entries=[json.loads((p/'config.json').read_text()) for p in sorted(assetroot.iterdir()) if p.is_dir()]
        np.testing.assert_allclose(initial['size'][:,:2],np.asarray([v['bbox'] for v in entries])[initial['models']],atol=1e-6,rtol=0)
    for i in range(3):
        t=np.load(root/category/f'trace_{i}.npz');d=np.linalg.norm(t['root'].astype('float64')-t['target'].astype('float64'),axis=-1)
        success=((d<=.2)&t['alive'][...,None]).any(0)
        assert int(success.sum())==raw[str(i)]['agent_success_count'];assert np.isfinite(d).all()
        assert ((t['done']&t['alive']).sum(0)==1).all()
        short=((d[:299]<=.2)&t['alive'][:299,:,None]).any(0)
        assert float(short.mean())==raw[str(i)]['agent_success_10s']
        validation[category][str(i)]=dict(agent_count=int(success.sum()),finite=True,first_episode_verified=True)
    summaries[category]={key:float(np.mean([raw[str(i)][key] for i in (1,2)])) for key in ['agent_success_rate','both_agents_success_rate','agent_success_10s','success_precision_m']}
fixed=bool(result.get('fixed_targets',False))
report='''# 78k zero-shot — 2명·4물체 CLIMB 대상 교체

모델:원본78k,adapter/학습없음. checkpoint에저장된학습env/sim/train을기준으로2명·4물체/600step을유지한다. 두사람이각각자기대상에CLIMB하며물체0/1만교체,나머지2/3은원래박스다. Table_Square test8종/Cabinet test9종에서각각균등복원추출. 원래학습처럼동적물체이며native URDF질량·관성을사용한다. 두target을모두Table또는모두Cabinet으로바꾼조건을따로실행했다.

평가512환경×2명×3repeat,seed0,걷기참조초기화100%,원래multi-agent scene종료/낙상/reset유지. 골반과물체metadata목표3D거리≤20cm ever. 학습의RSI50%혼합/랜덤과제혼합은평가에적용하지않는다. 모든수치는사람별성공률이며두사람모두성공률은별도열이다. 첫299step(약10초)과599step(약20초)을함께집계했다.

| CLIMB 대상 | 10초 성공률 | 20초 성공률 | 20초 두사람 모두 성공 | 성공한 시도의 최소 거리 |
|---|---:|---:|---:|---:|
'''
for cat,m in summaries.items():report+=f"| {cat} | {m['agent_success_10s']*100:.2f}% | {m['agent_success_rate']*100:.2f}% | {m['both_agents_success_rate']*100:.2f}% | {m['success_precision_m']*100:.2f}cm |\n"
report+='\n표는repeat1·2평균,repeat0도원시results.json에보존. 동일학습정책/평가seed의repeat산포이며독립학습seed산포가아니다. 이전1명평가는고정가구/10초/단독낙상코드등도달라서이번결과차이를인원수한변수의효과로확정할수없다. 초기장면은카테고리간exact pair가아니다.\n\n검증:전9trace를별도NumPy64로재채점해성공수/10초수치일치,finite,scene당첫종료1회확인. initial.npz에서2명/4물체shape,각조건distractor박스2개의크기가Box대조와exact동일,교체target native bbox와model ID일치확인. 학습/원본baseline수정없음.\n'
if fixed:
    report=report.replace('원래학습처럼동적물체이며native URDF질량·관성을사용한다.','CLIMB대상0/1만fix_base_link=True로고정하고나머지박스2/3은원래동적물체다. native URDF형상/metadata를사용한다.')
    report=report.replace('이전1명평가는고정가구/10초/단독낙상코드등도달라서이번결과차이를인원수한변수의효과로확정할수없다.','이전1명평가와는주변박스/scene종료/낙상코드등도달라서이번결과차이를인원수한변수의효과로확정할수없다.')
    dynamic=root.parent/'two_agent_climb_78k_20261025'
    comparison={}
    report+='\n## 고정 vs 동적\n\n| 대상 | 동적10초 | 고정10초 | 동적20초 | 고정20초 |\n|---|---:|---:|---:|---:|\n'
    for cat,m in summaries.items():
        old=json.loads((dynamic/cat/'results.json').read_text())['raw']
        d10=float(np.mean([old[str(i)]['agent_success_10s'] for i in (1,2)]));d20=float(np.mean([old[str(i)]['agent_success_rate'] for i in (1,2)]))
        old_init=np.load(dynamic/cat/'initial_0.npz');new_init=np.load(root/cat/'initial_0.npz')
        paired={key:bool(np.array_equal(old_init[key],new_init[key])) for key in old_init.files}
        for i in range(3):
            row=json.loads((root/cat/'results.json').read_text())['raw'][str(i)]
            assert row['fixed_targets'] and row['max_fixed_pose_drift']<2e-4
        comparison[cat]=dict(dynamic_10s=d10,fixed_10s=m['agent_success_10s'],dynamic_20s=d20,fixed_20s=m['agent_success_rate'],initial_repeat0_exact=paired)
        report+=f"| {cat} | {d10*100:.2f}% | {m['agent_success_10s']*100:.2f}% | {d20*100:.2f}% | {m['agent_success_rate']*100:.2f}% |\n"
    report+='\n전9repeat에서고정target pose변화2e-4미만을실행중검증했다.\n'
    (root/'fixed_vs_dynamic.json').write_text(json.dumps(comparison,indent=2))
(root/'REPORT.md').write_text(report);(root/'summary.json').write_text(json.dumps(dict(summary=summaries,validation=validation),indent=2));print(json.dumps(summaries,indent=2))
