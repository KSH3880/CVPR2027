"""Report the requested 7000-iteration four-agent zero-shot PUSH evaluation."""
import json,csv,hashlib,statistics
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/jointpush/five_task_push7000_scaling_20261022';WORK=OUT/'workspace'
prep=json.loads((WORK/'preparation.json').read_text());source=Path(prep['source']);manifest=json.loads((WORK/'source_manifest.json').read_text())
assert all(hashlib.sha256((source/k).read_bytes()).hexdigest()==v for k,v in manifest.items())
assert hashlib.sha256(Path(prep['checkpoint']).read_bytes()).hexdigest()==prep['checkpoint_sha256']
assert hashlib.sha256(Path(prep['saved_config']).read_bytes()).hexdigest()==prep['saved_config_sha256']
folder=WORK/'output_etc/check/a4_seed2141';r=json.loads((folder/'result.json').read_text())
assert r['finite'] and r['model_rms_unchanged'] and r['checkpoint_epoch']==7000 and (r['agents'],r['objects'],r['scenes'])==(4,8,64)
assert not r['attention_intervention']
rows=[]
for t in r['trials']:
 for i in range(4):
  rows.append(dict(scene=t['scene'],agent=i,scene_fall=t['scene_fall'],agent_fall=i in t['fallen_agents'],observed_seconds=t['seconds'],goal_success=t['ever_at'][i],contact_success=t['ever_contact'][i],force_contact=t['ever_force'][i],simultaneous_contact_goal=t['ever_success'][i],initial_distance_m=t['initial_goal_distance'][i],final_distance_m=t['final_goal_distance'][i],minimum_distance_m=t['minimum_goal_distance'][i],displacement_m=t['box_displacement_m'][i],goal_progress_m=t['net_goal_progress_m'][i]))
with (OUT/'per_agent.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
falls=sorted(t['seconds'] for t in r['trials'] if t['scene_fall']);end=max(t['seconds'] for t in r['trials'])
fig,axs=plt.subplots(1,3,figsize=(16,4.8),layout='constrained')
x=[0]+falls+[end];y=[100]+[100*(1-(i+1)/64) for i in range(len(falls))]+[100*(1-len(falls)/64)]
axs[0].step(x,y,where='post');axs[0].set_ylim(-2,102);axs[0].set_xlabel('Seconds');axs[0].set_ylabel('Scenes without a fallen agent (%)');axs[0].grid(alpha=.2);axs[0].set_title('Four-agent scene survival')
labels=['Contact','Goal AT','Contact + AT'];values=[r['agent_contact_success_rate']*100,r['agent_goal_success_rate']*100,r['agent_joint_contact_goal_success_rate']*100]
bars=axs[1].bar(labels,values,color=['tab:blue','tab:green','tab:orange']);axs[1].bar_label(bars,fmt='%.2f');axs[1].set_ylim(0,105);axs[1].set_title('Agent ever success (%)')
initial=np.array([v['initial_distance_m'] for v in rows]);final=np.array([v['final_distance_m'] for v in rows]);axs[2].scatter(initial,final,s=12,alpha=.5);axs[2].plot([0,6],[0,6],'--',color='gray');axs[2].set_xlim(0,6);axs[2].set_ylim(0,6);axs[2].set_xlabel('Initial goal distance (m)');axs[2].set_ylabel('Final goal distance (m)');axs[2].set_title('256 agents; below diagonal = progress');axs[2].grid(alpha=.2)
fig.suptitle('Five-task PUSH100 checkpoint7000 | 4 humans, 8 boxes | original bias, no mask | seed2141, 64 scenes')
fig.savefig(OUT/'scaling4_summary.png',dpi=170);plt.close(fig)
summary={k:v for k,v in r.items() if k not in ('trials','physics_sample')}
summary.update(agents_with_goal=sum(v['goal_success'] for v in rows),agents_with_contact=sum(v['contact_success'] for v in rows),median_box_displacement_m=float(np.median([v['displacement_m'] for v in rows])),agents_progress_over_0_5m=sum(v['goal_progress_m']>.5 for v in rows),agents_displaced_over_0_1m=sum(v['displacement_m']>.1 for v in rows),median_first_fall_seconds=statistics.median(falls) if falls else None)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
(OUT/'verification.json').write_text(json.dumps(dict(source_files_verified=len(manifest),source_unchanged=True,checkpoint_unchanged=True,saved_config_unchanged=True,strict_weights_restore=True,weights_rms_unchanged=True,finite=True,checkpoint_epoch=7000,original_two_agent_evaluation_skipped_by_user_preference=True),indent=2))
report=f'''# PUSH100 7000iter: 4명 확장 평가

구동/가중치 복원은 가능하지만, 이번 무개입4인 평가에서 PUSH 수행은 충분하지 않았다. 낙상은 {r['scene_falls']}/64장면({100*r['scene_fall_rate']:.2f}%), goal AT도달은 {summary['agents_with_goal']}/256, 접촉성공은 {summary['agents_with_contact']}/256({100*r['agent_contact_success_rate']:.2f}%)였다.

| 항목 | 결과 |
|---|---:|
| 낙상 없는 장면 | {64-r['scene_falls']}/64 |
| Agent 목표 도달 | {summary['agents_with_goal']}/256 |
| Agent 손 접촉 성공 | {summary['agents_with_contact']}/256 |
| 평균 초기 목표거리 | {r['mean_initial_goal_distance_m']:.4f}m |
| 평균 종료 목표거리 | {r['mean_final_goal_distance_m']:.4f}m |
| 평균 목표 방향 순진척 | {r['mean_net_goal_progress_m']:.4f}m |
| 평균/중앙값 상자 이동거리 | {r['mean_box_displacement_m']:.4f}/{summary['median_box_displacement_m']:.4f}m |
| 목표 방향으로 0.5m 이상 전진 | {summary['agents_progress_over_0_5m']}/256 |

원본: `{prep['checkpoint']}`. epoch7000/frame{prep['frame']}. 저장 relation_config에서 env/sim/train을 복사했다. 원래 학습은2명/4박스/PUSH_AT100%, 모델은 task_shared_embedding_action·GTA·64D/4층/2head이다. 상자는가로1.2~1.5m/깊이2m/높이1.3m,20kg,마찰0.4,바닥마찰1.0이다. 목표거리는상자진행방향1~5m다.

평가:4명/8박스(담당4+비대상4)/4goal, seed2141,64장면,최대600step(~20초),loco-only평가시작. 훈련의RSI혼합과구분한다. 추가음수bias·mask·fine-tuning은없다. CPU2장면24step과GPU5기존MPS4장면60step을통과한뒤64장면을순차실행했다. CPU/GPU smoke는성능평가표에합치지않는다.

원본은일부설정검증/스폰에서2명·4박스만허용하므로 process-local로count검증의학습기준값만정규화하고4개작업구역샘플러를사용했다.4개담당구역은원래정사각형의네모서리이며여분상자는원본base reset을유지한다. 원본소스/설정/가중치는수정하지않았다. 실제actor수는{r['actor_count']}(human4+관측box8+원본고정받침actor8),DOF수{r['dof_count']}다. 관측폭1268D/encoder counts[4,8,4],task packet의agent/payload/goal할당과물성20kg/마찰0.4를assert검사했다. 체크포인트복원의물성/모델계약은count변경부분외유지했다.

낙상은원본per-agent terminate가하나라도나온첫장면종료에서집계하며동료agent는그시점에검열한다. goal은원본AT own_success,contact는원본PUSH_CONTACT own_success,simultaneous는같은step의두조건동시만족이다. 원본runtime.done도별도계측했으며PUSH_CONTACT는required=false/AT는required=true라동시접촉성공으로간주하지않는다. 목표판정은저장config의AT success_distance0.2m와원본z/자세조건을따른다.

사용자의직전선호에따라원래2명평가는생략했다. 따라서현재4인수행이낮다는관측은확실하지만,원래2인수행부족과인원증가에따른성능저하를구분하지는못한다. 단일평가seed결과이며4인공유상자협력검사가아니라각담당상자를미는zero-shot검사다. 스폰/물리배치도인원수에맞춰확장되어토큰수만의인과효과로해석하지않는다.

원시result/initial/log: workspace/output_etc/check/a4_seed2141/. per_agent.csv에는256개agent의초기/종료/최소거리·접촉·성공·검열시간이있다. 원본Python{len(manifest)}파일/체크포인트/저장config무결성과가중치/RMS불변을검증했다.

재현: TOKENHSI_GPU=5 PROBE_AGENTS=4 NUM_ENVS=64 PROBE_STEPS=600 SEED=2141 OUTPUT_PATH=<새로컬workspace/output_etc/check/이름> bash jointpush/scripts/five_task_scaling7000.sh
'''
(OUT/'REPORT.md').write_text(report)
print(json.dumps(summary))
