"""Render captured raw QK, bias and actual attention without model inference."""
import sys,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path(sys.argv[1]);data=np.load(out/'attention_raw.npz');states=json.loads((out/'states.json').read_text());meta=json.loads((out/'capture.json').read_text())
m,o=meta['agents'],meta['boxes'];L=2*m+o
masked=meta.get('masked_token_ids',[])
pairs=meta.get('blocked_pairs',[])
mask_names=(', '.join(f'X{i-8}' if i>=8 else f'B{i-4}' for i in masked) or 'none (original biased forward)')+(' + H_i->foreign B_j blocked' if pairs else '')
if meta.get('strict_owner_box'):mask_names+=f' | perspective H{meta.get("stored_root_perspective",0)} ONLY; other human rows not used for their actions'
labels=[f'H{i}' for i in range(m)]+[f'B{i}' for i in range(m)]+[f'X{i}' for i in range(o-m)]+[f'G{i}' for i in range(m)]
jobs=[(f,c,s['step']) for f,s in enumerate(states) for c,active in enumerate(s['active']) if active]
bound=float(sys.argv[2]) if len(sys.argv)>2 else max(float(np.abs(data[k][f,c]).max()) for f,c,_ in jobs for k in ('qk_scores','relation_bias'))
images=[];stats=[]
for f,c,step in jobs:
 qk,bias,actual=[data[k][f,c] for k in ('qk_scores','relation_bias','attention')]
 logits=qk+bias
 if masked:logits[...,masked]=-np.inf
 for query,key in pairs:logits[...,query,key]=-np.inf
 exp=np.exp(logits-logits.max(-1,keepdims=True));assert np.max(np.abs(exp/exp.sum(-1,keepdims=True)-actual))<2e-6
 fig,axes=plt.subplots(4,6,figsize=(29,20),layout='constrained')
 for layer in range(4):
  for head in range(2):
   for part,(values,title) in enumerate(((qk,'Raw QK / sqrt(d)'),(bias,'Original relation bias'),(actual,'Actual softmax(QK + bias + mask)' if 'masked_token_ids' in meta else 'Actual softmax(QK + bias)'))):
    ax=axes[layer,head*3+part];v=values[layer,head]
    im=ax.imshow(v,cmap='magma' if part==2 else 'coolwarm',vmin=0 if part==2 else -bound,vmax=1 if part==2 else bound)
    ax.set_xticks(range(L),labels,rotation=90,fontsize=7);ax.set_yticks(range(L),labels,fontsize=7)
    ax.set_xlabel('Key');ax.set_ylabel('Query');ax.set_title(f'L{layer+1} H{head+1} | {title}',fontsize=10)
    for border in (m,m+o):
     ax.axhline(border-.5,color='gray',lw=.6);ax.axvline(border-.5,color='gray',lw=.6)
    if part==2:
     for key in masked:
      ax.axvspan(key-.5,key+.5,facecolor='gray',alpha=.3,hatch='//')
     for query,key in pairs:
      ax.add_patch(plt.Rectangle((key-.5,query-.5),1,1,facecolor='gray',alpha=.4,hatch='//'))
     for i,j in zip(*np.where(v>=.08)):ax.text(j,i,f'{v[i,j]:.2f}',ha='center',va='center',fontsize=6,color='black' if v[i,j]>.65 else 'white')
    if part in (0,1):
     for i in range(L):
      for j in range(L):
       ax.text(j,i,f'{v[i,j]:.3g}',ha='center',va='center',fontsize=6,color='white' if abs(v[i,j])>.6*bound else 'black')
    if layer==0:fig.colorbar(im,ax=ax,shrink=.65)
   a=actual[layer,head,:m]
   stats.append(dict(scene=c,step=step,layer=layer+1,head=head+1,own_box=float(np.mean([a[i,m+i] for i in range(m)])),other_target_boxes=float(np.mean(a[:,m:2*m].sum(-1))-np.mean([a[i,m+i] for i in range(m)])),extra_boxes=float(a[:,2*m:m+o].sum(-1).mean()),humans=float(a[:,:m].sum(-1).mean()),goals=float(a[:,m+o:].sum(-1).mean()),qk_human_row_std=float(qk[layer,head,:m].std(-1).mean())))
 fig.suptitle(f'KSH Sit/Climb Distill39000 | {m} agents / {o} boxes | scene {c}, step {step} | masked: {mask_names}\nH0: SIT | H1: CLIMB | B0/B1: assigned boxes | G0/G1: goals | original biases retained\nRaw QK and bias share a fixed logit scale across all samples. Later-layer QK includes earlier bias effects.',fontsize=14)
 name=f'scene_{c:03d}_step_{step:06d}.png';fig.savefig(out/name,dpi=135);plt.close(fig);images.append(name)
 print('SAVED',name,flush=True)
(out/'metrics.json').write_text(json.dumps(stats,indent=2))
html=['<!doctype html><meta charset="utf-8"><title>KSH Sit/Climb 2-agent 2-box attention</title><style>body{font:16px sans-serif;margin:24px}img{width:500px}a{display:inline-block;margin:12px}</style><h1>KSH Sit/Climb 2-agent 2-box attention</h1><p>Raw QK / relation bias / actual attention. H0=SIT, H1=CLIMB, B0/B1=assigned boxes, G0/G1=goals. No masking or bias correction. Click for full resolution.</p>']
for name in images:html.append(f'<a href="{name}"><p>{name}</p><img loading="lazy" src="{name}"></a>')
(out/'index.html').write_text(''.join(html))
summary=[]
for l in range(1,5):
 rows=[r for r in stats if r['layer']==l];keys=('own_box','other_target_boxes','extra_boxes','humans','goals','qk_human_row_std')
 summary.append(dict(layer=l,**{k:float(np.mean([r[k] for r in rows])) for k in keys}))
(out/'render.json').write_text(json.dumps(dict(png_count=len(images),raw_logit_bound=bound,layer_average=summary),indent=2))
print(json.dumps(summary),flush=True)
# Dedicated raw-QK panels for an initial and middle frame, numeric values shown.
qk_bound=max(float(np.abs(data['qk_scores'][f,c]).max()) for f,c,_ in jobs)
for f,c,step in [job for job in jobs if job[1]==0 and job[2] in (0,120)]:
 qk=data['qk_scores'][f,c]
 fig,axs=plt.subplots(4,2,figsize=(12,19),layout='constrained')
 for layer in range(4):
  for head in range(2):
   ax=axs[layer,head];v=qk[layer,head]
   im=ax.imshow(v,cmap='coolwarm',vmin=-qk_bound,vmax=qk_bound)
   ax.set_xticks(range(L),labels,fontsize=11);ax.set_yticks(range(L),labels,fontsize=11);ax.set_xlabel('Key');ax.set_ylabel('Query')
   ax.set_title(f'Layer{layer+1} Head{head+1}: raw QK / sqrt(d)')
   for i in range(L):
    for j in range(L):ax.text(j,i,f'{v[i,j]:.3g}',ha='center',va='center',fontsize=9,color='white' if abs(v[i,j])>.6*qk_bound else 'black')
   fig.colorbar(im,ax=ax,shrink=.65)
 fig.suptitle(f'KSH Sit/Climb Distill39000 | 2 humans / 2 boxes / 2 goals | scene{c}, step{step}\nRAW QK BEFORE bias and softmax; later-layer inputs include earlier biased attention',fontsize=13)
 fig.savefig(out/f'raw_qk_scene_{c:03d}_step_{step:06d}.png',dpi=150);plt.close(fig)
