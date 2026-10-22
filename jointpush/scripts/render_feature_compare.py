"""Matched-state 2-vs-4 feature and action diagnostics."""
from pathlib import Path
import sys,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(sys.argv[1]);data=np.load(p/'features.npz');states=json.loads((p/'states.json').read_text());meta=json.loads((p/'capture.json').read_text())
model_label=meta.get('model_label', 'KSH Sit/Climb 39000' if 'tasks' in meta else 'Native Y2m 6000')
f=data['features'];actions=data['actions'];active=np.array([s['active'] for s in states])
# F,N,C,L,A,D -> select matched scene samples, then aggregate agents independently.
def metrics(a,b):
 diff=b-a;norm=np.linalg.norm(a,axis=-1)
 cosine=np.sum(a*b,axis=-1)/(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)+1e-12)
 return dict(rms=float(np.sqrt(np.mean(diff**2))),relative_l2_mean=float(np.mean(np.linalg.norm(diff,axis=-1)/(norm+1e-12))),cosine_mean=float(np.mean(cosine)),cosine_min=float(np.min(cosine)),max_abs=float(np.max(np.abs(diff))))
reports={}
for scope,selected in [('initial',np.arange(len(states))==0),('all_valid',np.ones(len(states),dtype=bool))]:
 keep=active&selected[:,None];a=f[:,:,0][keep];b=f[:,:,1][keep];c=f[:,:,2][keep]
 mu2=actions[:,:,0][keep];mu4=actions[:,:,1][keep];mum=actions[:,:,2][keep]
 layers=[]
 for layer in range(5):layers.append(dict(layer=meta['layers'][layer],**metrics(a[:,layer],b[:,layer]),mask_control=metrics(a[:,layer],c[:,layer])))
 reports[scope]=dict(scene_time_samples=int(keep.sum()),agent_time_samples=int(keep.sum()*4),layers=layers,action_raw=metrics(mu2,mu4),action_clipped=metrics(np.clip(mu2,-1,1),np.clip(mu4,-1,1)),action_mask_control=metrics(mu2,mum),clipped_dof_abs_difference_gt_0_1_fraction=float(np.mean(np.abs(np.clip(mu2,-1,1)-np.clip(mu4,-1,1))>.1)))
(p/'summary.json').write_text(json.dumps(reports,indent=2))
fig,axes=plt.subplots(2,2,figsize=(15,11),layout='constrained');x=np.arange(5)
for scope,label in [('initial','Initial matched states'),('all_valid','All pre-termination samples')]:
 r=reports[scope]
 axes[0,0].plot(x,[v['relative_l2_mean'] for v in r['layers']],marker='o',label=label)
 axes[0,1].plot(x,[v['cosine_mean'] for v in r['layers']],marker='o',label=label)
for ax in axes[0]:ax.set_xticks(x,meta['layers']);ax.grid(alpha=.3);ax.legend()
axes[0,0].set_title('Relative L2 difference: ||f4-f2|| / ||f2||');axes[0,1].set_title('Feature cosine similarity')
a=actions[0,:,0].ravel();b=actions[0,:,1].ravel();ax=axes[1,0]
ax.scatter(a,b,s=3,alpha=.25);lo=min(a.min(),b.min());hi=max(a.max(),b.max());ax.plot([lo,hi],[lo,hi],'k--',lw=1);ax.set_xlabel('Same head: raw action mean from f2');ax.set_ylabel('Same head: raw action mean from f4');ax.set_title('Initial same-state action outputs')
delta=(f[0,:,1,-1]-f[0,:,0,-1]).reshape(-1,64);ax=axes[1,1];bound=float(np.abs(delta).max());im=ax.imshow(delta,aspect='auto',cmap='coolwarm',vmin=-bound,vmax=bound);fig.colorbar(im,ax=ax);ax.set_xlabel('Feature dimension');ax.set_ylabel('Scene x agent');ax.set_title('Layer4 feature delta: f4 - f2 (initial)')
fig.suptitle(model_label+': identical retained observations and weights\n2-agent pair vs full 4-agent context; 4-agent pair-mask equals 2-agent features',fontsize=15)
fig.savefig(p/'feature_comparison.png',dpi=160);plt.close(fig)
images=[]
for frame,state in enumerate(states):
 if not state['active'][0]:continue
 for agent in range(2):
  fig,axes=plt.subplots(3,5,figsize=(23,9),layout='constrained')
  feat2=f[frame,0,0,:,agent];feat4=f[frame,0,1,:,agent];diff=feat4-feat2
  bound=max(float(np.abs(feat2).max()),float(np.abs(feat4).max()));db=max(float(np.abs(diff).max()),1e-8)
  for layer in range(5):
   for row,(values,title,limit) in enumerate(((feat2,'2-agent feature',bound),(feat4,'4-agent feature',bound),(diff,'Difference: f4 - f2',db))):
    ax=axes[row,layer];im=ax.imshow(values[layer].reshape(8,8),cmap='coolwarm',vmin=-limit,vmax=limit)
    ax.set_xticks([]);ax.set_yticks([]);ax.set_title(f'{meta["layers"][layer]} | {title}',fontsize=10)
    if layer==4:fig.colorbar(im,ax=ax,shrink=.8)
  fig.suptitle(f'Same state | scene0 H{agent}, step{state["step"]}\nEach 64D vector displayed as an 8x8 grid (feature indices only, not spatial locations). Difference panels share one scale within this image.',fontsize=13)
  name=f'scene000_H{agent}_step{state["step"]:06d}.png';fig.savefig(p/name,dpi=140);plt.close(fig);images.append(name)
html=['<!doctype html><meta charset="utf-8"><title>2 vs 4 feature comparison</title><style>body{font:16px sans-serif;margin:24px}img{width:700px}a{display:inline-block;margin:12px}</style><h1>Same-state feature comparison: 2 vs 4 agents</h1><p>Raw vectors and metrics compare the same retained humans, boxes and goals, using identical learned weights and input normalization. Four-agent pair-mask is the numerical control. The 2-agent feature is a counterfactual forward, not a separate physical trajectory.</p><a href="feature_comparison.png"><img src="feature_comparison.png"></a>']
for name in images:html.append(f'<a href="{name}"><p>{name}</p><img loading="lazy" src="{name}"></a>')
(p/'index.html').write_text(''.join(html))
rows=['| Layer | RMS delta | Relative L2 | Cosine |','|---|---:|---:|---:|']
for v in reports['initial']['layers']:rows.append(f'| {v["layer"]} | {v["rms"]:.6f} | {v["relative_l2_mean"]:.4f} | {v["cosine_mean"]:.4f} |')
r=reports['initial']
report="""# 같은 상태의 feature 비교: 2명 vs 4명

같은 checkpoint와 RMS를 사용했다. 4명 장면의 한 상태에서 두 source pair의 사람·박스·목표·pose를 각각 추출하고 graph index만 remap해2인 encoder로 계산했다. 두 pair의 결과를 합쳐4인 full-attention과 비교했다. 별도 rollout끼리 비교한 것이 아니다. 각 모델의 원래 agent당 물체 비율을 유지했다. 실제 물체 수는 capture.json에 기록했다. 4인 그룹간 mask를 수치 대조군으로 사용했다. tokenizer 직후와 Transformer 각층 뒤의64D human feature, 동일 action_head의 최종 출력을 저장했다.

"""+'\n'.join(rows)+f"""

초기{meta['scenes']}장면×4명 비교. action raw RMS 차이={r['action_raw']['rms']:.6f}, clipped RMS 차이={r['action_clipped']['rms']:.6f}. clipped action에서 차이0.1초과 DOF 비율={r['clipped_dof_abs_difference_gt_0_1_fraction']:.2%}. clipped는rawmu를[-1,1]로 제한한 비교이며 이후PD/물리변화를 측정한 것은 아니다.

Tokenizer human입력 최대차이={meta['same_tokenizer_human_max_error']}. pair-mask feature최대차이={meta['pair_mask_feature_max_error']}, action최대차이={meta['pair_mask_action_max_error']}. 가중치/RMS불변, CPU PhysX/CPU추론. 초기치는 낙상 이전의 비교이며 나머지 시점은4인unmasked궤적의 장면 최초종료 전만 사용했다. 시간별샘플과 한 장면4명은 독립시행이 아니다. feature차이만으로 낙상의 인과나head단독의 정상여부를 판정할 수 없다. 8x8그림은64개feature의 편의적 배치이며 공간구조를 뜻하지 않는다.
"""
(p/'REPORT.md').write_text(report)
print(json.dumps(reports['initial'],indent=2));print('SAVED',len(images)+1,'PNGs',flush=True)
