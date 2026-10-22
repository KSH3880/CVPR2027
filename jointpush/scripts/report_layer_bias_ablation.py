"""Layer-selection all-owner bias ablation: measured outcomes and actual maps."""
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
out=ROOT/'runs/jointpush/attn_view/native_y2m_layer_bias_ablation_20261021'
records=[r for r in json.loads((out/'results.json').read_text()) if r['agents']==4]
assert len(records)==11 and (out/'complete.json').exists()
fields=['tag','agents','scene_falls','scene_fall_rate','agent_ever_goal_success_rate','mean_final_goal_distance_m']
with (out/'summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(records)
conditions=[('last2',0)]+[('first2',l) for l in (4,6,8,12)]+[(f'layer{i}',4) for i in range(1,5)]+[('last2',4),('all',4)]
labels=['baseline']+[f'{mode}\n-{l}' for mode,l in conditions[1:]]
fig,axs=plt.subplots(1,2,figsize=(18,5),layout='constrained')
for agents,color in ((4,'tab:orange'),):
 rows=[next(r for r in records if r['agents']==agents and r['layer_mode']==mode and r['attention_ablation']['strength']==l) for mode,l in conditions]
 for ax,key in zip(axs,('scene_fall_rate','agent_ever_goal_success_rate')):ax.plot(range(len(conditions)),[100*r[key] for r in rows],'o-',color=color,label=f'{agents} agents')
for ax,title in zip(axs,('Scene falls (%) - lower better','Agent ever goal success (%) - higher better')):
 ax.set_xticks(range(len(conditions)),labels,rotation=25,ha='right');ax.set_ylim(-2,102);ax.set_title(title);ax.grid(alpha=.2);ax.legend()
fig.suptitle('Layer selection | Every H/B/G query -> foreign-owner H/B/G: -lambda | Native Y2m 6000 | 64 scenes')
fig.savefig(out/'layer_ablation_comparison.png',dpi=170);plt.close(fig)
lines=['# All-owner bias 층별 ablation 및 초반2층 강도 비교','','Native Y2m 6000 원본 가중치에서 owner i의 H_i/B_i/G_i는 같은 owner 토큰의 기존 bias를 유지하고, 다른 모든 owner의 H/B/G key로 향하는 logit에 -λ를 더했다. 원래 bias를 덮어쓰지 않고 한 번 더한다. actor만 바꾸고 critic·물리·가중치·RMS를 유지했다. 자기 그룹 안의 기존 음수 bias를 제거하는 것은 아니다.','','사용자 요청에 따라 4명/4박스/4목표만 비교하며 여분 박스는 없다. 이미 완료한 일부2명 결과는 raw results.json에 보존했고, 나머지2명 조건은 생략했다. 같은 seed2141·64장면·최대600step·loco 평가 시작·GPU5 기존MPS를 사용했다. 각 단일층(layer1/2/3/4), 초반2층(first2), 후반2층(last2), 전체4층(all)을 -4에서 비교했다. 초반2층은 -6/-8/-12도 확인했다. 층 번호는1부터4까지다. 매 조건의 초기상태 hash가 같은 인원수 내에서 일치하고, 가중치/RMS 불변·정확한 보정 위치·실제 층출력 재구성을 검사했다.','','| 층 | λ | 인원 | 장면 낙상 | agent 목표 도달 | 종료 거리(m) |','|---|---:|---:|---:|---:|---:|']
for r in records:
 a=r['attention_ablation'];lines.append(f"| {r['layer_mode']} | {a['strength']:g} | {r['agents']} | {r['scene_falls']}/64 ({r['scene_fall_rate']*100:.2f}%) | {r['agent_ever_goal_success_rate']*100:.2f}% | {r['mean_final_goal_distance_m']:.3f} |")
lines+=['','단일층 보정은 해당 층만 바꿨을 때 충분한지를 비교하며, 그 층을 제거했을 때의 필요성을 입증하는 leave-one-out 검사는 아니다. 강도별 차이도 동일한 평가 seed에서 비교했으며 독립 검증은 아니다.','','낙상은 원본 per-agent terminate가 하나라도 나온 장면 비율이며, 첫 장면 종료에서 동료 agent도 검열한다. 목표 도달은 그 이전 누적 원본 runtime.done 기준이다. 같은 seed의 탐색 비교이며 독립 seed 검증이나 학습 seed 간 일반화 결과는 아니다. 같은4인 배치를 유지한 층 선택/강도 개입 비교다.']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
mass=[]
for r in records:
 folder=Path(r['folder']);d=np.load(folder/'attention_raw.npz');m=r['agents'];labels=[f'H{i}' for i in range(m)]+[f'B{i}' for i in range(m)]+[f'G{i}' for i in range(m)]
 q,b,e,p=[d[k][0,0] for k in ('qk_scores','relation_bias','effective_bias','attention')]
 z=q+e;z=np.exp(z-z.max(-1,keepdims=True));assert np.abs(z/z.sum(-1,keepdims=True)-p).max()<2e-6
 owner=np.arange(3*m)%m;cross=owner[:,None]!=owner[None,:]
 expected=np.zeros_like(e)
 for layer in r['attention_ablation']['layers']:expected[layer-1,:,cross]=-r['attention_ablation']['strength']
 assert np.abs((e-b)-expected).max()<2e-6
 first=d['attention'][0] # stored first2 scenes, layer,head,query,key
 for layer in range(4):
  for group,offset in (('human',0),('box',m),('goal',2*m)):
   values=[np.take(first[:,layer,:,i,:],np.flatnonzero(cross[i]),axis=-1).sum(-1).mean() for i in range(offset,offset+m)]
   mass.append(dict(tag=r['tag'],layer=layer+1,query_type=group,foreign_attention_mass=float(np.mean(values))))
 fig,axs=plt.subplots(4,4,figsize=(18,17),layout='constrained');bound=max(np.abs(v).max() for v in (q,b,e))
 for layer in range(4):
  for col,(value,title) in enumerate(((q,'Raw QK / sqrt(d)'),(b,'Original bias'),(e,'Applied bias'),(p,'Actual attention'))):
   ax=axs[layer,col];im=ax.imshow(value[layer].mean(0),cmap='magma' if col==3 else 'coolwarm',vmin=0 if col==3 else -bound,vmax=1 if col==3 else bound)
   ax.set_xticks(range(3*m),labels,rotation=90,fontsize=7);ax.set_yticks(range(3*m),labels,fontsize=7);ax.set_title(f'Layer {layer+1}: {title}');ax.set_xlabel('Key');ax.set_ylabel('Query')
   if layer==0:fig.colorbar(im,ax=ax,shrink=.65)
 fig.suptitle(r['tag']+' | scene0 step0 | head mean; actual attention averaged AFTER softmax')
 fig.savefig(folder/'attention_initial.png',dpi=135);plt.close(fig)
with (out/'initial_foreign_attention.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['tag','layer','query_type','foreign_attention_mass']);w.writeheader();w.writerows(mass)
print(out/'REPORT.md')
# Context for the layer intervention: initial stored two scenes, not all64.
fig,axs=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
selected=[('last2_lambda0_a4','Baseline','gray'),('first2_lambda4_a4','First2 -4','tab:blue'),('first2_lambda12_a4','First2 -12','tab:cyan'),('last2_lambda4_a4','Last2 -4','tab:orange'),('all_lambda4_a4','All -4','tab:green')]
for ax,kind in zip(axs,('human','box','goal')):
 for tag,label,color in selected:
  vals=[next(row['foreign_attention_mass'] for row in mass if row['tag']==tag and row['layer']==layer and row['query_type']==kind)*100 for layer in range(1,5)]
  ax.plot(range(1,5),vals,'o-',label=label,color=color)
 ax.set_title(f'{kind.capitalize()} query: all foreign-owner key mass');ax.set_xlabel('Actor layer');ax.set_ylabel('Attention mass (%)');ax.set_xticks(range(1,5));ax.set_ylim(bottom=0);ax.grid(alpha=.2)
axs[-1].legend(fontsize=8)
fig.suptitle('Initial two stored scenes | probabilities averaged over heads and owned queries | not a rollout-wide statistic')
fig.savefig(out/'foreign_attention_by_layer.png',dpi=170);plt.close(fig)
