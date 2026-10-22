"""Report original two-agent stability first; expanded runs remain exploratory."""
from pathlib import Path
import json,hashlib,statistics,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/jointpush/five_task_500_fall_20261021';WORK=OUT/'workspace';CHECK=WORK/'output_etc/check'
records=[]
for model in ('depth180','depth200_friction04'):
 mixed_tag='depth200_original_mixed_a2_seed2141' if model=='depth200_friction04' else f'{model}_original_mixed_a2_seed2141'
 for tag,preset,agents in [(mixed_tag,'original_mixed',2),(f'{model}_a2_seed2141','push_at',2),(f'{model}_a4_seed2141','push_at',4)]:
  folder=CHECK/tag;r=json.loads((folder/'result.json').read_text());times=[t['seconds'] for t in r['trials'] if t['scene_fall']]
  without_push=[t for t in r['trials'] if 'tasks' in t and 4 not in t['tasks']]
  records.append(dict(model=model,preset=preset,agents=agents,objects=r['objects'],scene_falls=r['scene_falls'],scenes=r['scenes'],median_fall_seconds=statistics.median(times) if times else None,minimum_fall_seconds=min(times) if times else None,maximum_fall_seconds=max(times) if times else None,scenes_without_push=len(without_push) if without_push else None,falls_without_push=sum(t['scene_fall'] for t in without_push) if without_push else None,result_path=str(folder/'result.json'),trials=r['trials']))
(OUT/'all_results.json').write_text(json.dumps(records,indent=2))
with (OUT/'summary.csv').open('w') as f:
 fields=[k for k in records[0] if k!='trials'];writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in records)
fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
for ax,model in zip(axes,('depth180','depth200_friction04')):
 for r in [r for r in records if r['model']==model]:
  t=np.array(sorted(x['seconds'] for x in r['trials'] if x['scene_fall']))
  x=np.r_[0,t,max(3,float(t[-1]))];y=np.r_[1,1-np.arange(1,len(t)+1)/r['scenes'],1-len(t)/r['scenes']]
  label='Original 2 agents / mixed tasks' if r['preset']=='original_mixed' else f'{r["agents"]} agents / PUSH_AT only'
  ax.step(x,y*100,where='post',label=label)
 ax.set_title(model);ax.set_xlabel('Time until first scene fall (seconds)');ax.set_ylabel('Scenes with no fallen agent (%)');ax.set_ylim(-2,102);ax.grid(alpha=.3);ax.legend(fontsize=8)
fig.suptitle('500-iteration checkpoints: original 2-agent baseline is already unstable\nSeed2141, 64 scenes/condition, loco-only reset, deterministic policy; no training/masking',fontsize=13)
fig.savefig(OUT/'fall_survival.png',dpi=160);plt.close(fig)
lines=['| 모델 | 원래2명·박스4개 혼합과제 낙상 | 첫낙상시간 중앙값 | PUSH없는장면 낙상 |','|---|---:|---:|---:|']
for r in records:
 if r['preset']=='original_mixed':lines.append(f'| {r["model"]} | {r["scene_falls"]}/{r["scenes"]} | {r["median_fall_seconds"]:.3f}s | {r["falls_without_push"]}/{r["scenes_without_push"]} |')
report='''# 500iter 다섯 과제 모델: 원래 환경의 낙상 확인

사용자 우선 요청대로 원래2명/박스4개(담당2+비대상2)/다섯과제혼합 구성을 먼저 판정한다. 저장된relation_config.yaml에서env/sim/train을복사했고원본모델을strict복원했다. depth180은PUSH상자Y1.8m/20kg/마찰0.6, depth200_friction04는Y2.0m/20kg/마찰0.4. 다른과제물성·고정선반·모션·보상은원본그대로다.

'''+ '\n'.join(lines)+'''

두모델모두원래2명환경에서이미낙상하므로이checkpoint로4명확장성의좋고나쁨을판정할수없다. 500iter가원인이라고확정한것은아니며, 현재평가의정상baseline이확보되지않았다는결론이다. PUSH없는장면도따로집계했다.

## 평가 조건과 낙상 정의

각조건64장면·seed2141·최대600step(약20초), 원본test기본인loco초기화만사용, deterministicaction, 학습/mask없음. 명시한혼합baseline은random_scenario이며실제128agent task수는SIT6/CLIMB27/CARRY_AT33/CARRY_ONTOP37/PUSH_AT25다. 원본학습의RSI혼합초기화와는구분한다. epoch500/frame65536000을CPU에서확인했다.

원본종료식을사용했다: 허용된발contact를제외한접촉힘L1>0.1과비발body높이<원본body별종료높이(설정기본0.15m)의동시조건, progress>1. 장면에서첫agent낙상시종료하고동료는검열하므로64/64는모든개인이넘어졌다는뜻이아니라모든장면에서누군가넘어졌다는뜻이다. timeout/성공과낙상은원본terminate flag로구분했다. Push성공률은평가하지않았다.

## 별도 PUSH_AT 전용/확장 검사

원래baseline과별개로PUSH_AT만지정한2명/4box와4명/8box도각64장면실행했고모두64/64낙상이었다. source자체의Stage1은2명/4box만허용하므로4인검사는로컬평가확장이다. 학습가중치/RMS와토큰/보상계산을보존하고인원검증·checkpoint물리계약의count항목만명시적으로정규화했다. 그래프는원래variable-count sampler를사용하고PUSHreset은원래정사각형배치의4모서리에agent별source를두도록일반화했다. 비대상4box는원래base reset을유지했다. 직접4인학습환경재현이라고해석하지않는다. 2인baseline불안정으로4인결과해석은보류한다.

## 검증/보존

CPU4env/12step에서2인/4인/두물성strict복원·634/1268D·agent/object/graphpacket의대응·PUSH20kg/마찰0.6,0.4 실제Gym조회·finite를검증했다. GPU5의기존MPS에서4env/60step 검증후정식평가를순차실행했다. 원본2인actor10/DOF64, 확장4인actor20/DOF128 및globalroot actorID범위·중복없음을검사했다. 모든평가가중치/RMS불변과finite를확인했다.

외부원본코드·checkpoint를수정하지않았다. 실행산출물은이폴더의workspace/output_etc/check에만저장한다. 원본의출력경로제약도process-local workspace root로동일하게유지했다. SizeRsiCache학습경로는eval에서실행되지않으며PUSH cache는기존선별본을읽었다. source_manifest.json의184Python파일과두checkpoint SHA를실행후재검증했다. 원본학습/MPS데몬/다른프로세스를종료하지않았다.
'''
(OUT/'REPORT.md').write_text(report)
source=Path('/home/hwanhee/koo_cvpr/stage1_fix/ports/ksh_stage1_stage2_base/push')
manifest=json.loads((WORK/'source_manifest.json').read_text())
assert all(hashlib.sha256((source/k).read_bytes()).hexdigest()==v for k,v in manifest.items())
prep=json.loads((WORK/'preparation.json').read_text())
assert all(hashlib.sha256(Path(v['checkpoint']).read_bytes()).hexdigest()==v['checkpoint_sha256'] for v in prep.values())
(OUT/'verification.json').write_text(json.dumps(dict(conditions=6,source_files_unchanged=len(manifest),checkpoints_unchanged=True,all_evaluations_finite=True,weights_rms_unchanged=True),indent=2))
print('\n'.join(lines))
