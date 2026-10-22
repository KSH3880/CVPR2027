"""Compare full owner hard mask with matched recent baseline/soft controls."""
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
base=ROOT/'runs/jointpush/attn_view'
out=base/'native_y2m_allowner_hardmask_20261021'
controls=base/'native_y2m_layer_bias_ablation_20261021'
folders=[controls/'last2_lambda0_a4',controls/'all_lambda4_a4',out]
labels=['Original','All layers -4','All layers hard mask']
records=[json.loads((p/'result.json').read_text()) for p in folders]
assert all(r['model_and_rms_unchanged'] and r['agents']==4 and r['scenes']==64 and r['seed']==2141 for r in records)
assert len({r['attention_ablation']['initial_state_sha256'] for r in records})==1
hard=records[-1]['attention_ablation'];assert hard['hard_mask'] and hard['layers']==[1,2,3,4] and hard['scope']=='all_owner'
assert hard['blocked_attention_max_error']==0 and hard['layer_reconstruction_max_error']<2e-5
rows=[dict(condition=label,scene_falls=r['scene_falls'],scene_fall_rate=r['scene_fall_rate'],agent_ever_goal_success_rate=r['agent_ever_goal_success_rate'],all_agents_goal_rate=r['scene_all_agents_ever_goal_success_rate'],final_distance_m=r['mean_final_goal_distance_m']) for label,r in zip(labels,records)]
with (out/'comparison.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
for ax,key,title in zip(axs,('scene_fall_rate','agent_ever_goal_success_rate'),('Scene falls (%)','Agent ever goal success (%)')):
 values=[100*r[key] for r in records];bars=ax.bar(labels,values,color=['gray','tab:blue','tab:green'])
 ax.bar_label(bars,fmt='%.2f');ax.set_ylim(0,108);ax.set_title(title);ax.tick_params(axis='x',labelrotation=15)
fig.suptitle('Native Y2m 6000 | 4 agents / 4 boxes / 4 goals | 64 scenes, seed2141')
fig.savefig(out/'hardmask_comparison.png',dpi=170);plt.close(fig)
d=np.load(out/'attention_raw.npz');q,b,e,p=[d[k][0,0] for k in ('qk_scores','relation_bias','effective_bias','attention')]
m=4;owners=np.arange(12)%m;mask=owners[:,None]!=owners[None,:]
assert np.isneginf(e[...,mask]).all() and np.array_equal(e[...,~mask],b[...,~mask]) and (p[...,mask]==0).all()
z=q+e;z=np.exp(z-z.max(-1,keepdims=True));assert np.abs(z/z.sum(-1,keepdims=True)-p).max()<2e-6
names=[f'H{i}' for i in range(m)]+[f'B{i}' for i in range(m)]+[f'G{i}' for i in range(m)]
bound=max(np.abs(q).max(),np.abs(b).max())
fig,axs=plt.subplots(4,4,figsize=(18,17),layout='constrained')
for layer in range(4):
 for col,(v,title) in enumerate(((q,'Raw QK / sqrt(d)'),(b,'Original bias'),(e,'Applied bias (gray = -inf)'),(p,'Actual attention'))):
  ax=axs[layer,col];value=v[layer].mean(0);value=np.ma.masked_invalid(value)
  cmap=plt.get_cmap('magma' if col==3 else 'coolwarm').copy();cmap.set_bad('lightgray')
  im=ax.imshow(value,cmap=cmap,vmin=0 if col==3 else -bound,vmax=1 if col==3 else bound)
  ax.set_xticks(range(12),names,rotation=90,fontsize=7);ax.set_yticks(range(12),names,fontsize=7);ax.set_xlabel('Key');ax.set_ylabel('Query');ax.set_title(f'Layer{layer+1}: {title}')
  if layer==0:fig.colorbar(im,ax=ax,shrink=.65)
fig.suptitle('All-query owner hard mask | scene0 step0 | head mean; attention averaged AFTER softmax')
fig.savefig(out/'attention_initial.png',dpi=135);plt.close(fig)
lines=['# 全owner hard mask 비교'.replace('全owner','모든 query의 owner'),'', 'Native Y2m 6000의 actor1~4층에서 H_i/B_i/G_i가 다른 owner의 H/B/G key를 보는 항목을 -inf로 대체했다. 같은 owner 세 토큰의 기존 bias, critic, 가중치/RMS, 물리 설정은 유지했다. 4명/4박스/4목표로 각 agent가 별도 상자를 미는 실험이며 공동 상자 협력 수행 검사는 아니다.','','64장면·seed2141·최대600step·loco 평가시작·GPU5 기존MPS. 비교 대조군은 바로 앞 층별 배치의 원본 및 전체층 -4이며, 세 조건의 초기상태 hash가 동일하다. 전체층 -4는 다른 owner 확률을 약하게 남기고 hard mask는 정확히0이다.','','| 조건 | 장면 낙상 | agent 목표 도달 | 4명 전원 도달 장면 |','|---|---:|---:|---:|']
for label,r in zip(labels,records):lines.append(f"| {label} | {r['scene_falls']}/64 ({r['scene_fall_rate']*100:.2f}%) | {r['agent_ever_goal_success_rate']*100:.2f}% | {r['scene_all_agents_ever_goal_success_rate']*100:.2f}% |")
lines+=['','실제 저장 QK+적용bias의 softmax와 attention 일치, 막힌 항목의 모든 저장 층/head/query 확률0, 같은 owner bias 불변, 실제 층출력 재구성을 검증했다.','', '낙상은 첫 per-agent terminate에서 장면 단위로 집계하고 동료는 그 시점에 검열한다. 목표 도달은 원본 runtime.done 누적값이다. 단일 seed 평가이며 독립 seed 검증은 아니다. Hard mask는 모든 attention 층의 그룹간 정보교환을 차단한다. 이 성능을 여러 agent의 협력 학습이나 unrestricted attention의 일반화 증거로 해석하지 않는다. 물리 세계는 여전히 공유하며 관측 전처리까지 별도 단일-agent 실험과 동일한지는 별도 조건이다.']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
(out/'verification.json').write_text(json.dumps(dict(matched_initial_state=True,all_weights_rms_unchanged=True,all_blocked_attention_exact_zero=True,own_bias_unchanged=True,hard_mask_metadata=hard),indent=2))
print(out/'REPORT.md')
