"""Original KSH Sit/Climb versus Koo PUSH: directed NONE bias and attention mass."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runs/jointpush/attn_view';OUT=BASE/'none_relation_comparison_20261021'
sources=[('Koo',2,OUT/'koo2'),('Koo',4,BASE/'agentwise_attention_20261021/push'),('KSH',2,OUT/'ksh2'),('KSH',4,OUT/'ksh4')]
rows=[];typed=[];tables={};meta_rows=[]
for model,m,folder in sources:
 initial=(folder/'initial_attention.npz').exists()
 meta=json.loads((folder/('initial_attention.json' if initial else 'capture.json')).read_text())
 raw=np.load(folder/('initial_attention.npz' if initial else 'attention_raw.npz'))
 q,b,a=[(raw[k] if initial else raw[k][0]).astype(np.float64) for k in ('qk_scores','relation_bias','attention')]
 n,layers,heads,L,_=a.shape;assert (n,layers,heads,L)==(32,4,2,3*m)
 assert meta['agents']==m and meta.get('objects',meta.get('boxes'))==m
 assert not meta.get('mask',False) and not meta.get('policy_intervention',False)
 z=q+b;z=np.exp(z-z.max(-1,keepdims=True));assert np.abs(z/z.sum(-1,keepdims=True)-a).max()<2e-6
 task=np.zeros((L,L),bool)
 for i in range(m):
  task[i,m+i]=True
  if model=='Koo':task[m+i,2*m+i]=True
 self_mask=np.eye(L,dtype=bool);none=~(task|self_mask)
 owner=np.arange(L)%m;foreign=owner[:,None]!=owner[None,:]
 assert not (foreign&~none).any()
 types=np.repeat(np.arange(3),m)
 table=np.zeros((4,2,3,3))
 for st in range(3):
  for dt in range(3):
   mask=none&(types[:,None]==st)&(types[None,:]==dt)
   values=b[...,mask];mean=values.mean((0,3)) # layer,head
   assert np.abs(values-mean[None,:,:,None]).max()<2e-5
   table[:,:,st,dt]=mean
   for layer in range(4):
    for head in range(2):
     typed.append(dict(model=model,agents=m,layer=layer+1,head=head+1,source_type='HBG'[st],target_type='HBG'[dt],none_bias=float(mean[layer,head]),count_per_scene=int(mask.sum())))
 tables[model,m]=table
 for layer in range(4):
  for head in range(2):
   bs,ps,qs=b[:,layer,head],a[:,layer,head],q[:,layer,head]
   for kind,ids in [('ALL',np.arange(L)),('H',np.arange(m)),('B',np.arange(m,2*m)),('G',np.arange(2*m,3*m))]:
    nm=none.copy();nm[~np.isin(np.arange(L),ids)]=False
    task_mass=(ps*task).sum(-1)[:,ids];self_mass=(ps*self_mask).sum(-1)[:,ids];none_mass=(ps*none).sum(-1)[:,ids]
    assert np.abs(task_mass+self_mass+none_mass-1).max()<2e-6
    row=dict(model=model,agents=m,scenes=n,layer=layer+1,head=head+1,query_type=kind,none_bias_mean=float(bs[:,nm].mean()),none_qk_mean=float(qs[:,nm].mean()),none_attention_per_key=float(ps[:,nm].mean()),none_attention_row_mass=float(none_mass.mean()),foreign_attention_row_mass=float((ps*foreign).sum(-1)[:,ids].mean()),within_owner_none_mass=float((ps*(none&~foreign)).sum(-1)[:,ids].mean()),assigned_attention_row_mass=float(task_mass.mean()),self_attention_row_mass=float(self_mass.mean()),none_keys_per_query=float(nm.sum()/len(ids)))
    if kind=='H':
     own=bs[:,np.arange(m),m+np.arange(m)]
     foreign_box=np.stack([bs[:,i,m+np.arange(m)[np.arange(m)!=i]].mean(-1) for i in range(m)],-1)
     row.update(task_box_bias=float(own.mean()),foreign_box_none_bias=float(foreign_box.mean()),own_vs_foreign_box_bias_margin=float((own-foreign_box).mean()))
    rows.append(row)
 meta_rows.append(dict(model=model,agents=m,source=str(folder),checkpoint=meta['checkpoint'],seed=meta['seed'],scenes=n,tokens=L,task_directed_edges=int(task.sum()),none_directed_pairs=int(none.sum()),scope='initial state only; original biased actor; 32 CPU scenes'))
for model in ('Koo','KSH'):assert np.abs(tables[model,2]-tables[model,4]).max()<2e-5
keys=list(dict.fromkeys(k for row in rows for k in row))
with (OUT/'per_layer_head.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
with (OUT/'none_bias_by_type.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(typed[0]));w.writeheader();w.writerows(typed)
# Head means retain query type; scenes were averaged inside each head.
summary=[]
for model,m,_ in sources:
 for layer in range(1,5):
  for kind in ('ALL','H','B','G'):
   selected=[r for r in rows if (r['model'],r['agents'],r['layer'],r['query_type'])==(model,m,layer,kind)]
   summary.append(dict(model=model,agents=m,layer=layer,query_type=kind,**{k:float(np.mean([r[k] for r in selected])) for k in selected[0] if k not in ('model','agents','layer','head','query_type','scenes')}))
(OUT/'summary.json').write_text(json.dumps(dict(sources=meta_rows,rows=summary,none_type_bias_invariant_to_2_vs_4=True),indent=2))
def get(model,m,layer,kind):return next(r for r in summary if (r['model'],r['agents'],r['layer'],r['query_type'])==(model,m,layer,kind))
fig,axs=plt.subplots(1,3,figsize=(17,4.8),layout='constrained')
for model,color in (('Koo','tab:orange'),('KSH','tab:blue')):
 for m,style in ((2,'--'),(4,'-')):
  vals=[get(model,m,i,'H') for i in range(1,5)]
  for ax,key,scale in zip(axs,('none_bias_mean','own_vs_foreign_box_bias_margin','foreign_attention_row_mass'),(1,1,100)):
   ax.plot(range(1,5),[r[key]*scale for r in vals],marker='o',ls=style,color=color,label=f'{model} {m} agents')
for ax,title in zip(axs,('Human query: NONE bias mean','Own box task bias minus foreign box NONE','Human query: foreign-owner attention (%)')):
 ax.set_title(title);ax.set_xticks(range(1,5));ax.set_xlabel('Actor layer');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Koo PUSH6000 vs KSH Sit/Climb39000 | agents = boxes = goals | 32 initial scenes; different tasks/layouts')
fig.savefig(OUT/'human_none_comparison.png',dpi=180);plt.close(fig)
fig,axs=plt.subplots(4,2,figsize=(10,16),layout='constrained');bound=max(np.abs(tables[model,4]).max() for model in ('Koo','KSH'))
for layer in range(4):
 for col,model in enumerate(('Koo','KSH')):
  ax=axs[layer,col];v=tables[model,4][layer].mean(0)
  im=ax.imshow(v,cmap='coolwarm',vmin=-bound,vmax=bound)
  ax.set_xticks(range(3),['H','B','G']);ax.set_yticks(range(3),['H','B','G']);ax.set_xlabel('Key type');ax.set_ylabel('Query type');ax.set_title(f'{model} layer{layer+1}: NONE bias, head mean')
  for i in range(3):
   for j in range(3):ax.text(j,i,f'{v[i,j]:.3f}',ha='center',va='center',color='white' if abs(v[i,j])>.6*bound else 'black')
  if layer==0:fig.colorbar(im,ax=ax,shrink=.7)
fig.suptitle('Only NONE category; SELF and assigned directed task edges excluded\nPhysical type pair tables are unchanged between 2 and 4 agents')
fig.savefig(OUT/'none_bias_type_tables.png',dpi=170);plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
for ax,kind in zip(axs,('H','B','G')):
 for model,color in (('Koo','tab:orange'),('KSH','tab:blue')):
  for m,style in ((2,'--'),(4,'-')):
   ax.plot(range(1,5),[100*get(model,m,l,kind)['foreign_attention_row_mass'] for l in range(1,5)],marker='o',ls=style,color=color,label=f'{model} {m}')
 ax.set_title(f'{kind} query: foreign-owner attention (%)');ax.set_xticks(range(1,5));ax.set_xlabel('Actor layer');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Query types separated: only HUMAN output is read by the action head after layer4')
fig.savefig(OUT/'foreign_mass_by_query_type.png',dpi=170);plt.close(fig)
lines=['# KSH / Koo task 미할당 relation 비교','','NONE은 diagonal SELF와 실제 directed task edge를 제외한 항목이다. Koo는 H_i→B_i(PUSH), B_i→G_i(AT); KSH Sit/Climb은 H_i→B_i만 task edge이다. 따라서 task edge의 역방향이나 같은 owner의 간접 관계도 NONE일 수 있다. 다른 owner로 향하는 항목은 별도 subset으로 비교했다.','','두 모델 모두2명2박스2goal 및4명4박스4goal,32개 초기장면/seed2141/CPU/no mask/no bias modification을 사용했다. Koo4는 기존32장면 canonical 자료를 재사용했다. 같은 seed라도 태스크·초기물리상태·학습이력은 서로 다르며 입력이 같은 paired 비교는 아니다. bias의절댓값은row constant shift에불변인softmax효과를단독으로나타내지않으므로 task-NONE margin을같이제공한다.','','## Human query: 4명 평균(32장면·2head)','','| 층 | Koo NONE bias | KSH NONE bias | Koo task−foreign box | KSH task−foreign box | Koo foreign mass | KSH foreign mass |','|---|---:|---:|---:|---:|---:|---:|']
for l in range(1,5):
 k,s=get('Koo',4,l,'H'),get('KSH',4,l,'H')
 lines.append(f"| {l} | {k['none_bias_mean']:.3f} | {s['none_bias_mean']:.3f} | {k['own_vs_foreign_box_bias_margin']:.3f} | {s['own_vs_foreign_box_bias_margin']:.3f} | {100*k['foreign_attention_row_mass']:.4f}% | {100*s['foreign_attention_row_mass']:.4f}% |")
lines+=['','## Human query: 2명→4명에서 foreign attention 총량','','| 층 | Koo 2명 | Koo 4명 | KSH 2명 | KSH 4명 |','|---|---:|---:|---:|---:|']
for l in range(1,5):lines.append('| '+str(l)+' | '+' | '.join(f"{100*get(model,m,l,'H')['foreign_attention_row_mass']:.4f}%" for model,m in [('Koo',2),('Koo',4),('KSH',2),('KSH',4)])+' |')
lines+=['','NONE의 physical type별 learned bias 값은 같은 모델의2명/4명에서 동일했다. 인원이 늘면 단위 항목값이 바뀌는 것이 아니라 다른 owner의 key 개수가 늘어난다. attention 가중치는 합1이므로 value가 무한히 단순합산되는 뜻은 아니며 불필요한 key가 차지하는 합계 비율과 후속 layer의feature변화를봐야한다.','','전체query NONE 평균은per_layer_head.csv의ALL, 실질readout인human은H로분리했다. KSH 마지막층 B/G query의bias=0인항목이전체평균을크게올리지만 그층의 B/G output을actionhead가읽지않는다. 앞층B/G output은다음층을통해human에영향을줄수있다. 따라서 전체항목을모두강하게억제해야한다는명제로이수치를일반화하지않는다.','','Attention mass는 V의크기/GTA/잔차를포함한실제action기여도와동일하지않다. 이분석은모델별초기상태32장면이며학습seed변동성이나학습과정에서왜이런bias가생겼는지는검증하지않았다. full-layer -4/hard mask 개선결과는별도개입근거로함께해석할수있다.','','원자료위치는summary.json.sources, 모든층/head/방향별NONE bias는none_bias_by_type.csv에있다. 새GPU/학습/외부소스수정은없다.']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
for l in range(1,5):print('layer',l,'Koo',get('Koo',4,l,'H'),'KSH',get('KSH',4,l,'H'))
