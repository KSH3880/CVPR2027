"""Read existing captures only: owner-vs-foreign logit decomposition."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'runs/jointpush/attn_view/agentwise_attention_20261021'
OUT=SOURCE/'logit_decomposition';OUT.mkdir(exist_ok=True)
groups=[('Push','push',[0,1,2,3]),('Sit','sitclimb',[0,2]),('Climb','sitclimb',[1,3]),('Holding+AT','holding_at',[0,1,2,3])]
records=[];checks={}
for name,folder,agents in groups:
 d=np.load(SOURCE/folder/'initial_attention.npz');q,b,p=[d[k] for k in ('qk_scores','relation_bias','attention')]
 for layer in range(4):
  parts={}
  for label,values in [('qk',q),('bias',b),('combined',q+b)]:
   own=np.stack([values[:,layer,:,i,4+i] for i in agents],-1)
   foreign=np.stack([np.stack([values[:,layer,:,i,4+j] for j in range(4) if j!=i],-1) for i in agents],-2)
   margin=own[...,None]-foreign
   parts[label]=dict(own_mean=float(own.mean()),foreign_mean=float(foreign.mean()),margin_mean=float(margin.mean()),margin_std=float(margin.std()),negative_margin_fraction=float((margin<0).mean()))
  own_p=np.stack([p[:,layer,:,i,4+i] for i in agents],-1)
  foreign_p=np.stack([np.stack([p[:,layer,:,i,4+j] for j in range(4) if j!=i],-1) for i in agents],-2)
  assert (own_p>0).all() and (foreign_p>0).all()
  ratio=np.log(own_p[...,None])-np.log(foreign_p)
  raw=(q+b)
  expected=np.stack([np.stack([raw[:,layer,:,i,4+i]-raw[:,layer,:,i,4+j] for j in range(4) if j!=i],-1) for i in agents],-2)
  error=float(np.abs(ratio-expected).max());assert error<2e-5
  checks[f'{name}_L{layer+1}']=error
  records.append(dict(task=name,layer=layer+1,agents=agents,qk=parts['qk'],bias=parts['bias'],combined=parts['combined'],own_box_probability_mean=float(own_p.mean()),foreign_box_probability_sum_mean=float(foreign_p.sum(-1).mean()),log_ratio_identity_max_error=error))
(OUT/'summary.json').write_text(json.dumps(dict(records=records,checks=checks),indent=2))
with (OUT/'margins.csv').open('w') as f:
 fields=['task','layer','qk_margin','bias_margin','total_margin','own_bias','foreign_bias','foreign_boxes_attention']
 writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
 for r in records:writer.writerow(dict(task=r['task'],layer=r['layer'],qk_margin=r['qk']['margin_mean'],bias_margin=r['bias']['margin_mean'],total_margin=r['combined']['margin_mean'],own_bias=r['bias']['own_mean'],foreign_bias=r['bias']['foreign_mean'],foreign_boxes_attention=r['foreign_box_probability_sum_mean']))
fig,axes=plt.subplots(1,3,figsize=(18,5),layout='constrained')
for name,_,_ in groups:
 rows=[r for r in records if r['task']==name]
 for ax,key,title in zip(axes,('qk','bias','combined'),('QK / sqrt(d) margin','Learned bias margin','Combined logit margin')):
  ax.plot(range(1,5),[r[key]['margin_mean'] for r in rows],marker='o',label=name);ax.set_title(title);ax.set_xticks(range(1,5));ax.set_xlabel('Transformer layer');ax.set_ylabel('Own box score - other assigned box score');ax.grid(alpha=.3)
axes[-1].legend();fig.suptitle('Same captures: direct H_i query preference for own B_i vs foreign B_j\n32 initial scenes, both heads; positive margin favors own box. No new policy/physics run.',fontsize=14)
fig.savefig(OUT/'logit_margin_comparison.png',dpi=170);plt.close(fig)
lines=['| Task | L1 QK margin | L1 bias margin | L4 QK margin | L4 bias margin |','|---|---:|---:|---:|---:|']
for name,_,_ in groups:
 a=next(r for r in records if r['task']==name and r['layer']==1);z=next(r for r in records if r['task']==name and r['layer']==4)
 lines.append(f'| {name} | {a["qk"]["margin_mean"]:.3f} | {a["bias"]["margin_mean"]:.3f} | {z["qk"]["margin_mean"]:.3f} | {z["bias"]["margin_mean"]:.3f} |')
report='''# 자기 박스 vs 남의 담당 박스: QK/bias 분해

새시뮬레이션없이기존agentwise초기32장면을사용했다. 각Hi에서자기Bi와모든Bj(j!=i)사이의score차를구해32장면·2head·해당agent·다른box3개의평균을냈다. QK는GTA변환후QK/sqrt(d)이고bias는실제로더해진learned bias다. margin은양수일수록자기박스를선호한다.

'''+ '\n'.join(lines)+'''

결과: 첫층QK에서자기/남의박스차이는세모델모두거의0이다. Push와구HOLDING+AT는첫층bias margin도비슷하지만,후반층에서는HOLDING+AT가훨씬큰margin을만든다. 마지막층QK자체는Push가더자기박스를선호하지만최종attention은KSH쪽이더자기박스에묶인다. 따라서이번직접박스attention차이에서는후반층의학습된bias 분리가중요한차이로관찰된다.

softmax분모는같은query의두key확률비에서상쇄된다. 각원소별log(p_own/p_foreign)=QK차+bias차를수치검증했다. 평균margin의exp를평균확률비라고해석하면안된다. source/config/task/배치가다른모델간관찰이므로bias만이낙상원인이라고확정하지않는다. QK에는이전층bias영향이있으며V·GTA·residual·FFN의영향은이분해만으로판정할수없다. 다른human/goal로향하는항목까지모두설명한것도아니다.

다음검증후보는(1)같은상태에서attentionV출력을기존/추가토큰기여와재정규화항으로분해,(2)bias만제한적으로조절해원래2명성능을보존하면서4명행동이안정화되는지확인,(3)여분박스학습모델은2인baseline안정이확보된뒤같은지표로대조하는것이다. 여기서는추가정책실행을하지않았다.
'''
(OUT/'REPORT.md').write_text(report)
print('\n'.join(lines))
for name,_,_ in groups:
 r=next(r for r in records if r['task']==name and r['layer']==4)
 print(name,'L4 bias own/foreign',r['bias']['own_mean'],r['bias']['foreign_mean'])
