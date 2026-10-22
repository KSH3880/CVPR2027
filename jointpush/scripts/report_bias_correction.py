"""Render measured inference correction outcomes and actual corrected maps."""
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
out=ROOT/'runs/jointpush/attn_view/native_y2m_bias_correction_20261021'
records=json.loads((out/'results.json').read_text())
fields=['tag','agents','scene_falls','scene_fall_rate','agent_ever_goal_success_rate','mean_final_goal_distance_m']
with (out/'summary.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(records)
fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
conditions=[('none',0)]+[(t,l) for t in ('humans','boxes','goals') for l in (2,4,6)]
labels=['baseline']+[f'{t}\n-{l}' for t,l in conditions[1:]]
for agents,color in ((2,'tab:blue'),(4,'tab:orange')):
 rows=[next(r for r in records if r['agents']==agents and r['attention_ablation']['target']==t and r['attention_ablation']['strength']==l) for t,l in conditions]
 for ax,metric in zip(axes,('scene_fall_rate','agent_ever_goal_success_rate')):
  ax.plot(range(10),[r[metric]*100 for r in rows],'o-',label=f'{agents} agents',color=color)
for ax,title in zip(axes,('Scene falls (%) - lower better','Agent ever goal success (%) - higher better')):
 ax.set_xticks(range(10),labels,rotation=35,ha='right');ax.set_title(title);ax.set_ylim(-2,102);ax.grid(alpha=.2);ax.legend()
fig.suptitle('Native Y2m 6000 | actor layers 3-4 | 64 scenes, seed 2141, max 600 steps')
fig.savefig(out/'bias_correction_comparison.png',dpi=170);plt.close(fig)
lines=['# Native Y2m 후반 attention bias 보정 평가','','원본 6000iter 가중치로 actor의 3·4층에서만 Hi→남의 사람/담당 박스/목표 logit에 -λ를 더했다. 자기 토큰·다른 query·앞 두 층·critic·가중치·RMS는 유지했다. 학습은 실행하지 않았다.','','GPU5 기존 MPS, 64장면, seed2141, 최대600step. 원본 loco 평가 시작을 사용했다. 2명은 2박스·2목표, 4명은 4박스·4목표이며 여분 박스는 없다. 조건별 같은 인원수 초기 상태 hash 일치와 전체 가중치/RMS 불변, 실제 층 출력 및 보정 위치를 검증했다.','','| 보정 | λ | 인원 | 장면 낙상 | agent 목표 도달 | 종료 목표거리(m) |','|---|---:|---:|---:|---:|---:|']
for r in records:
 a=r['attention_ablation'];lines.append(f"| {a['target']} | {a['strength']:g} | {r['agents']} | {r['scene_falls']}/64 ({r['scene_fall_rate']*100:.2f}%) | {r['agent_ever_goal_success_rate']*100:.2f}% | {r['mean_final_goal_distance_m']:.3f} |")
lines+=['','낙상은 첫 agent terminate가 나온 장면의 비율이다. 이 시점에서 동료도 검열하며 목표 도달은 그 이전 누적 runtime.done 기준이다. 단일 평가 seed의 탐색 결과로 학습 seed 산포나 독립 검증 성능을 뜻하지 않는다. 2명과 4명은 배치도 달라 인원수만의 인과 효과로 해석하지 않는다.','','각 조건의 attention_initial.png는 첫 장면·첫 프레임의 실제 실행을 저장했다. Raw QK는 sqrt(d)로 나눈 softmax 전 값이며 original bias / corrected bias / 실제 attention을 별도로 표시한다.']
lines += ['','## 관측 결과','','보정 없는 4명은 낙상63/64(98.44%), agent 목표 도달1/256(0.39%)였다. 남의 목표 token에 -6을 준 조건은 낙상48/64(75.00%), 목표 도달19/256(7.42%)로 이번 탐색에서 변화가 가장 컸다. 같은 보정의 2명은 낙상2/64, 목표 도달102/128(79.69%)이며 기준은2/64·98/128(76.56%)다. 사람만 또는 박스만 보정해서는 4명 낙상이61~64/64로 높았다.','','후반층의 다른 owner 목표 경로가 동작 실패에 영향을 줄 수 있다는 개입 근거지만, 여전히 4명 장면의75%가 넘어지므로 충분한 해결책은 아니다. 앞 두 층, 다른 query를 통한 간접 전달, V/GTA 및 잔차 경로를 통제한 실험은 아니다. 이 결과만으로 전체 relation bias의 원인 여부를 확정하거나 배제할 수 없다. 독립 평가 seed에서 재검증하지 않았으므로 -6을 최적값으로 확정하지 않는다.']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
mass_rows=[]
for r in records:
 folder=Path(r['folder']);data=np.load(folder/'attention_raw.npz');m=r['agents'];labels=[f'H{i}' for i in range(m)]+[f'B{i}' for i in range(m)]+[f'G{i}' for i in range(m)]
 q,b,e,p=[data[k][0,0] for k in ('qk_scores','relation_bias','effective_bias','attention')]
 initial=data['attention'][0,:,3] # scenes,heads,query,key
 masses={}
 for name,offset in (('humans',0),('boxes',m),('goals',2*m)):
  values=[initial[:,:,i,[offset+j for j in range(m) if j!=i]].sum(-1).mean() for i in range(m)]
  masses[name]=float(np.mean(values))
 mass_rows.append(dict(tag=r['tag'],**masses))
 z=q+e;z=np.exp(z-z.max(-1,keepdims=True));assert np.abs(z/z.sum(-1,keepdims=True)-p).max()<2e-6
 bound=max(np.abs(v).max() for v in (q,b,e));fig,axs=plt.subplots(2,4,figsize=(17,8),layout='constrained')
 for row,layer in enumerate((2,3)):
  for col,(value,title) in enumerate(((q,'Raw QK / sqrt(d)'),(b,'Original bias'),(e,'Corrected bias'),(p,'Actual attention'))):
   # Mean over heads, after computing actual head-wise attention.
   ax=axs[row,col];im=ax.imshow(value[layer].mean(0),cmap='magma' if col==3 else 'coolwarm',vmin=0 if col==3 else -bound,vmax=1 if col==3 else bound)
   ax.set_xticks(range(3*m),labels,rotation=90);ax.set_yticks(range(3*m),labels);ax.set_title(f'Layer {layer+1}: {title}');ax.set_xlabel('Key');ax.set_ylabel('Query');fig.colorbar(im,ax=ax,shrink=.65)
 fig.suptitle(r['tag']+' | scene0 step0 | head mean (attention averaged AFTER softmax)')
 fig.savefig(folder/'attention_initial.png',dpi=140);plt.close(fig)
print(out/'REPORT.md')

with (out/'initial_layer4_foreign_mass.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=['tag','humans','boxes','goals']);writer.writeheader();writer.writerows(mass_rows)
