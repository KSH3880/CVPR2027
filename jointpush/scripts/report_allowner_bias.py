"""All-query cross-owner bias: metrics and actual canonical attention maps."""
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
out=ROOT/'runs/jointpush/attn_view/native_y2m_allowner_bias_20261021'
records=json.loads((out/'results.json').read_text())
assert len(records)==14 and (out/'complete.json').exists()
fields=['tag','agents','scene_falls','scene_fall_rate','agent_ever_goal_success_rate','mean_final_goal_distance_m']
with (out/'summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(records)
conditions=[('last2',0)]+[(mode,l) for mode in ('last2','all') for l in (2,4,6)]
labels=['baseline']+[f'{mode}\n-{l}' for mode,l in conditions[1:]]
fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
for agents,color in ((2,'tab:blue'),(4,'tab:orange')):
 rows=[next(r for r in records if r['agents']==agents and r['layer_mode']==mode and r['attention_ablation']['strength']==l) for mode,l in conditions]
 for ax,key in zip(axs,('scene_fall_rate','agent_ever_goal_success_rate')):ax.plot(range(7),[100*r[key] for r in rows],'o-',color=color,label=f'{agents} agents')
for ax,title in zip(axs,('Scene falls (%) - lower better','Agent ever goal success (%) - higher better')):
 ax.set_xticks(range(7),labels);ax.set_ylim(-2,102);ax.set_title(title);ax.grid(alpha=.2);ax.legend()
fig.suptitle('Every H/B/G query -> other-owner H/B/G keys: -lambda | Native Y2m 6000 | 64 scenes')
fig.savefig(out/'allowner_comparison.png',dpi=170);plt.close(fig)
lines=['# 모든 human·box·goal query의 다른 owner bias 보정','','Native Y2m 6000 원본 가중치에서 owner i의 H_i/B_i/G_i는 같은 owner 토큰의 기존 bias를 유지하고, 다른 모든 owner의 H/B/G key로 향하는 logit에 -λ를 더했다. 원래 bias를 덮어쓰지 않고 한 번 더한다. actor만 바꾸고 critic·물리·가중치·RMS를 유지했다. 자기 그룹 안의 기존 음수 bias를 제거하는 것은 아니다.','','2명/2박스/2목표, 4명/4박스/4목표이며 여분 박스는 없다. 같은 seed2141·64장면·최대600step·loco 평가 시작·GPU5 기존MPS를 사용했다. 후반2층(last2)과 전체4층(all), λ2/4/6을 각각 비교했다. 매 조건의 초기상태 hash가 같은 인원수 내에서 일치하고, 가중치/RMS 불변·정확한 보정 위치·실제 층출력 재구성을 검사했다.','','| 층 | λ | 인원 | 장면 낙상 | agent 목표 도달 | 종료 거리(m) |','|---|---:|---:|---:|---:|---:|']
for r in records:
 a=r['attention_ablation'];lines.append(f"| {r['layer_mode']} | {a['strength']:g} | {r['agents']} | {r['scene_falls']}/64 ({r['scene_fall_rate']*100:.2f}%) | {r['agent_ever_goal_success_rate']*100:.2f}% | {r['mean_final_goal_distance_m']:.3f} |")
lines+=['','낙상은 원본 per-agent terminate가 하나라도 나온 장면 비율이며, 첫 장면 종료에서 동료 agent도 검열한다. 목표 도달은 그 이전 누적 원본 runtime.done 기준이다. 같은 seed의 탐색 비교이며 독립 seed 검증이나 학습 seed 간 일반화 결과는 아니다. 2/4인 배치도 달라 확장 실패의 단일 원인으로 해석하지 않는다.','','앞선 human-query-only 시도는 사용자 범위 변경으로 중단했다. native_y2m_allforeign_bias_20261021/interrupted.json에 표시했고 완료 실험으로 집계하지 않았다.']
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
