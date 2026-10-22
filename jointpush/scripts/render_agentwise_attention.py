"""Owner-based attention accounting: no two-agent-group aggregation."""
from pathlib import Path
import sys,json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(sys.argv[1]);models=[('push','Push'),('sitclimb','Sit/Climb'),('holding_at','Holding+AT')]
categories=['self_human','other_humans','own_box','other_assigned_boxes','unassigned_boxes','own_goal','other_goals']
category_names=['Self human','Other humans','Own box','Other assigned boxes','Unassigned boxes','Own goal','Other goals']
records=[];summary={};arrays={};owner_arrays={};details={}
for key,label in models:
 folder=root/key;data=np.load(folder/'initial_attention.npz');meta=json.loads((folder/'initial_attention.json').read_text())
 a=data['attention'][...,:4,:];N,L,H,M,K=a.shape;o=meta['objects']
 assert N==32 and M==4 and K==8+o
 assert meta['own_box_token_ids']==list(range(4,8))
 mask=np.zeros((M,K,7),dtype=np.float32)
 for i in range(M):
  other=[j for j in range(M) if j!=i]
  mask[i,i,0]=1;mask[i,other,1]=1;mask[i,4+i,2]=1;mask[i,[4+j for j in other],3]=1
  mask[i,8:4+o,4]=1;mask[i,4+o+i,5]=1;mask[i,[4+o+j for j in other],6]=1
 assert np.all(mask.sum(-1)==1)
 mass=np.einsum('nlhak,akg->nlhag',a,mask)
 assert np.max(np.abs(mass.sum(-1)-1))<2e-6
 owner=np.stack([a[...,j]+a[...,4+j]+a[...,4+o+j] for j in range(4)],-1)
 assert np.max(np.abs(owner.sum(-1)+mass[...,4]-1))<2e-6
 np.savez_compressed(folder/'agentwise_mass.npz',category_mass=mass,owner_mass=owner)
 mean=mass.mean((0,2));owners=owner.mean((0,2));arrays[key]=mean;owner_arrays[key]=owners;details[key]=meta
 by_agent=[]
 for i in range(M):
  values={category:float(mean[-1,i,c]) for c,category in enumerate(categories)}
  values.update(agent=i,other_agents_total=sum(values[c] for c in ('other_humans','other_assigned_boxes','other_goals')))
  by_agent.append(values)
 for l in range(L):
  for h in range(H):
   for i in range(M):
    values=mass[:,l,h,i].mean(0)
    records.append(dict(model=label,layer=l+1,head=h+1,agent=i,**{c:float(values[j]) for j,c in enumerate(categories)}))
 summary[key]=dict(label=label,scenes=N,objects=o,definition='H_i query: foreign assigned box keys B_j,j!=i; humans/goals/unassigned separately. ALL other agents included regardless of prior pair grouping.',averaging='32 initial scenes, two heads; last transformer layer',last_layer_agents=by_agent,last_layer_mean={c:float(mean[-1,:,j].mean()) for j,c in enumerate(categories)},last_layer_owner_matrix=owners[-1].tolist(),layer_reconstruction_max_error=meta['layer_reconstruction_max_error'])
(root/'summary.json').write_text(json.dumps(summary,indent=2))
with (root/'per_agent_layer_head.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
foreign=[1,3,6,4];foreign_labels=['Other humans','Other assigned boxes','Other goals','Unassigned boxes']
max_foreign=max(float(arrays[key][-1,:,foreign].max())*100 for key,_ in models)
fig,axes=plt.subplots(2,3,figsize=(19,11),layout='constrained')
for col,(key,label) in enumerate(models):
 ax=axes[0,col];v=owner_arrays[key][-1]*100
 im=ax.imshow(v,cmap='magma',vmin=0,vmax=100)
 ax.set_title(f'{label}: keys grouped by INDIVIDUAL owner');ax.set_xticks(range(4),[f'Owner {i}' for i in range(4)]);ax.set_yticks(range(4),[f'Query H{i}' for i in range(4)])
 ax.set_xlabel('Owner of human + assigned box + goal keys')
 for i in range(4):
  for j in range(4):ax.text(j,i,f'{v[i,j]:.3f}%',ha='center',va='center',fontsize=10,color='black' if v[i,j]>70 else 'white')
 ax=axes[1,col];v=arrays[key][-1][:,foreign]*100
 im=ax.imshow(v,cmap='viridis',vmin=0,vmax=max_foreign)
 ax.set_xticks(range(4),foreign_labels,rotation=25,ha='right');ax.set_yticks(range(4),[f'Query H{i}' for i in range(4)]);ax.set_title(f'{label}: foreign categories separated')
 for i in range(4):
  for j in range(4):ax.text(j,i,f'{v[i,j]:.3f}%',ha='center',va='center',fontsize=10,color='black' if v[i,j]>.65*max_foreign else 'white')
fig.suptitle('Agent-wise attention: every H_i distinguishes its own B_i/G_i from ALL other agents\nLast layer, 32 initial scenes and 2 heads averaged; same original biased forward. Push: 4 boxes; KSH: 8 boxes.',fontsize=15)
fig.savefig(root/'agentwise_attention.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(4,3,figsize=(16,20),layout='constrained')
for l in range(4):
 for col,(key,label) in enumerate(models):
  ax=axes[l,col];v=owner_arrays[key][l]*100;ax.imshow(v,cmap='magma',vmin=0,vmax=100)
  ax.set_xticks(range(4),[f'Owner {i}' for i in range(4)]);ax.set_yticks(range(4),[f'H{i}' for i in range(4)]);ax.set_title(f'{label} | layer {l+1}')
  for i in range(4):
   for j in range(4):ax.text(j,i,f'{v[i,j]:.3f}%',ha='center',va='center',fontsize=9,color='black' if v[i,j]>70 else 'white')
fig.suptitle('Individual-owner attention across all layers; no pair/group aggregation\nEach key owner contains its human, assigned box and goal. Unassigned boxes are excluded from owner cells.',fontsize=15)
fig.savefig(root/'agentwise_all_layers.png',dpi=150);plt.close(fig)
lines=['| Model | Agent0 | Agent1 | Agent2 | Agent3 |','|---|---:|---:|---:|---:|']
for key,label in models:lines.append('| '+label+' | '+' | '.join(f"{a['other_assigned_boxes']:.3%}" for a in summary[key]['last_layer_agents'])+' |')
report='''# Agent별 attention 재집계

이전2인group 단위 집계를 대체한다. 같은pair 안의 다른agent도 남의agent로계산한다. 각Hi에서다른담당박스Bj(j!=i)로향하는확률을합하고,다른사람Hj/다른목표Gj/비대상박스는별도항목으로분리했다. 7개category합이항상1인지검사했다. owner별4x4그림은각owner의human+담당박스+goal에대한합이며2명묶음이아니다.

마지막층, 초기32장면,2head평균의 **다른agent담당박스** 확률:

'''+ '\n'.join(lines)+'''

Sit/Climb에서H0/H2는Sit,H1/H3는Climb이다. 물체physical assignment와 logical순서의앞4개가같은지각모델에서검증해자기박스를정확히식별했다. 원본forward를수정하지않는hook으로QK/bias/attention을수집하고,그attention으로재구성한각층출력과원본출력이수치정밀도내에서일치하는지검증했다. weights/RMS불변. CPU만사용하고KSH원본수정/GPU사용은없다.

이번은feature비교와동일한seed2141/초기32장면이며Push4box, KSH8box다. 앞선12.6%/0.29%/0.21%는서로다른기존샘플과2명group경계기준이었으므로이번수치와같은통계가아니다. 이전수치를개인별남의박스비율로해석하면안된다. 모델간task/환경/물체수/학습이력교란은남는다. attention확률만으로V/geometry의기여크기나낙상인과를판정할수없다.
'''
(root/'REPORT.md').write_text(report)
(root/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Agent-wise attention</h1><p>Each H_i versus each individual owner. Foreign boxes/humans/goals/unassigned boxes are separate.</p><a href="agentwise_attention.png"><img width="1200" src="agentwise_attention.png"></a><p><a href="agentwise_all_layers.png">All four layers</a></p>')
print('\n'.join(lines))
for key,label in models:
 mean=summary[key]['last_layer_mean'];print(label,'mean other boxes',mean['other_assigned_boxes']*100,'mean other humans',mean['other_humans']*100,'mean other goals',mean['other_goals']*100)
