"""Four-condition owner-box comparison and faithful raw attention PNGs."""
import os,sys,json,subprocess,concurrent.futures
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(sys.argv[1]).resolve();records=json.loads((root/'results.json').read_text());scripts=Path(__file__).resolve().parent
assert len(records)==4
bound=0.
for r in records:
 data=np.load(root/r['tag']/'attention_raw.npz');states=json.loads((root/r['tag']/'states.json').read_text())
 for f,state in enumerate(states):
  for c,active in enumerate(state['active']):
   if active:bound=max(bound,*(float(np.abs(data[k][f,c]).max()) for k in ('qk_scores','relation_bias')))
 if r['owner_only']:
  for i in range(4):
   for j in range(4):
    if i!=j:assert np.max(np.abs(data['attention'][...,i,4+j]))==0

def render(r):
 folder=root/r['tag']
 with (folder/'render.log').open('w') as log:
  subprocess.run([sys.executable,str(scripts/'attention_render.py'),str(folder),str(bound)],check=True,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
 print('RENDERED',r['tag'],flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(render,records))
labels=[f'H{i}' for i in range(4)]+[f'B{i}' for i in range(4)]+[f'X{i}' for i in range(4)]+[f'G{i}' for i in range(4)]
fig,axes=plt.subplots(4,4,figsize=(24,23),layout='constrained')
for col,r in enumerate(records):
 folder=root/r['tag'];meta=json.loads((folder/'capture.json').read_text());a=np.load(folder/'attention_raw.npz')['attention'][0,0].mean(1)
 for layer in range(4):
  ax=axes[layer,col];ax.imshow(a[layer],cmap='magma',vmin=0,vmax=1);ax.set_xticks(range(16),labels,rotation=90,fontsize=7);ax.set_yticks(range(16),labels,fontsize=7)
  ax.set_title(f'{r["tag"]}\nL{layer+1}, falls {r["scene_falls"]}/64',fontsize=11)
  for key in meta.get('masked_token_ids',[]):ax.axvspan(key-.5,key+.5,facecolor='gray',alpha=.3,hatch='//')
  for query,key in meta.get('blocked_pairs',[]):ax.add_patch(plt.Rectangle((key-.5,query-.5),1,1,facecolor='gray',alpha=.4,hatch='//'))
fig.suptitle('Same initial scene 0 | actual attention averaged over 2 heads\nDirect H_i -> other assigned B_j blocking; indirect paths through humans/goals/boxes remain',fontsize=17)
fig.savefig(root/'owner_box_comparison.png',dpi=140);plt.close(fig)
html=['<!doctype html><meta charset="utf-8"><title>Owner box attention ablation</title><style>body{font:16px sans-serif;margin:24px}img{width:450px}a{display:inline-block;margin:10px}td,th{padding:8px;border:1px solid #bbb}table{border-collapse:collapse}</style><h1>4 agents / 8 boxes: own target box visibility</h1><p>Direct H_i to foreign assigned B_j blocked in every actor layer/head; own box remains visible. Human/goal tokens remain visible, so indirect information paths remain. Physics and weights unchanged. Same seed2141,64 scenes,max600steps.</p><a href="owner_box_comparison.png"><img src="owner_box_comparison.png"></a><table><tr><th>Condition</th><th>Falls</th><th>Agent goals</th><th>Images</th></tr>']
rows=[];total=0
for r in records:
 tag=r['tag'];count=json.loads((root/tag/'render.json').read_text())['png_count'];total+=count
 html.append(f'<tr><td>{tag}</td><td>{r["scene_falls"]}/64</td><td>{r["agent_ever_goal_success_rate"]:.2%}</td><td><a href="{tag}/index.html">{count} PNG</a></td></tr>')
 rows.append(f'| {tag} | {r["scene_falls"]}/64 | {r["agent_ever_goal_success_rate"]:.2%} | {count} |')
html.append('</table>')
for r in records:
 tag=r['tag'];html.append(f'<h2>{tag}</h2>')
 for p in sorted((root/tag).glob('scene_*.png')):html.append(f'<a href="{tag}/{p.name}"><p>{p.stem}</p><img loading="lazy" src="{tag}/{p.name}"></a>')
(root/'index.html').write_text(''.join(html))
report='''# 담당 박스 직접 attention ablation

물리 환경:4명/대상박스4+비대상박스4/목표4. 같은6000 가중치/seed2141/64장면/최대600step/GPU5 기존MPS. H_i는 자기 담당B_i를 유지하고 다른 담당B_j(j!=i)로 향하는 직접 attention12셀을 모든4층·2head에서 -inf로 차단했다. 논리slot은 reset시 담당자 순으로 재정렬되므로 실제 물리담당자 swap도 반영한다. 비대상4개의 key 전체 차단을 별도 요인으로 비교했다. 사람·목표·물리박스·보상·critic·가중치/RMS는 유지했다.

대조군2개는 직전 extra-box ablation의 결과/원자료를 복사해 재사용했고 새 조건2개는 재평가했다. 4조건의 초기physics+obs SHA256은 동일하다. 실제 masked attention확률0, 층출력 재구성오차0, 모델/RMS불변을 확인했다. 원래shuffle은 forward당1회 유지하며 이미지에서만 canonical순서로 복원한다.

| 조건 | 장면낙상 | Agent 목표도달 | PNG |
|---|---:|---:|---:|
'''+ '\n'.join(rows)+'''

full_attention:모두표시. extra_boxes_hidden:비대상4개만차단. foreign_targets_hidden:남의담당박스 직접경로만차단,비대상은유지. own_box_only:두mask동시적용으로 human query의 직접박스key는자기담당것만허용.

직접 attention만의 실험이다. 다른사람/목표/자기박스 등의 토큰을 거친 간접 정보전파는 남는다. 따라서 foreign box 정보 자체를 완전히 제거했다고 해석하면 안 된다. 같은64장면 단일seed 반복이며 첫scene종료에서 동료는 검열된다. 성공은 원본CONTACT+AT done. PNG는첫2장면의종료전시점,rawQK=GTA변환후QK/sqrt(d),원래bias,mask적용attention을4층×2head로배치했다. blocked셀/열은빗금이며 모든조건공통logit색상범위다. later-layerQK는이전mask효과를포함한다.
'''
(root/'REPORT.md').write_text(report)
(root/'summary.json').write_text(json.dumps(dict(sample_pngs=total,comparison_pngs=1,shared_logit_bound=bound,conditions=[dict(tag=r['tag'],scene_falls=r['scene_falls'],agent_goal_rate=r['agent_ever_goal_success_rate']) for r in records]),indent=2))
print('TOTAL_PNG',total,flush=True)
