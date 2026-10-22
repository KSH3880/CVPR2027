"""Render every captured mask condition and a shared-scale comparison gallery."""
import sys,json,subprocess,concurrent.futures,os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(sys.argv[1]).resolve();records=json.loads((root/'results.json').read_text());scripts=Path(__file__).resolve().parent
bound=0.
for record in records:
 folder=root/record['tag'];data=np.load(folder/'attention_raw.npz');states=json.loads((folder/'states.json').read_text())
 for f,state in enumerate(states):
  for c,active in enumerate(state['active']):
   if active:
    bound=max(bound,*(float(np.abs(data[k][f,c]).max()) for k in ('qk_scores','relation_bias')))
def render(record):
 folder=root/record['tag']
 with (folder/'render.log').open('w') as log:
  subprocess.run([sys.executable,str(scripts/'attention_render.py'),str(folder),str(bound)],check=True,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
 print('RENDERED',record['tag'],flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(render,records))
labels=[f'H{i}' for i in range(4)]+[f'B{i}' for i in range(4)]+[f'X{i}' for i in range(4)]+[f'G{i}' for i in range(4)]
selected=['mask_none','mask_0','mask_01','mask_012','mask_0123']
fig,axes=plt.subplots(4,5,figsize=(24,19),layout='constrained')
for col,tag in enumerate(selected):
 r=next(r for r in records if r['tag']==tag);data=np.load(root/tag/'attention_raw.npz');a=data['attention'][0,0].mean(1)
 for layer in range(4):
  ax=axes[layer,col];ax.imshow(a[layer],cmap='magma',vmin=0,vmax=1)
  ax.set_xticks(range(16),labels,rotation=90,fontsize=6);ax.set_yticks(range(16),labels,fontsize=6)
  ax.set_title(f'{tag} | L{layer+1} | falls {r["scene_falls"]}/64')
  for i in r['masked_extras']:ax.axvspan(8+i-.5,8+i+.5,facecolor='gray',alpha=.3,hatch='//')
fig.suptitle('Same initial scene 0 | actual masked attention averaged over 2 heads\nProgressive example (all 16 combinations available in gallery); physical 8 boxes retained',fontsize=17)
fig.savefig(root/'mask_progression.png',dpi=140);plt.close(fig)
html=['<!doctype html><meta charset="utf-8"><title>Native Y2m mask ablation</title><style>body{font:16px sans-serif;margin:24px}img{width:420px}a{display:inline-block;margin:8px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #bbb}</style><h1>4 agents / 8 boxes: extra-box masking ablation</h1><p>Original weights, same initial state / seed 2141 / 64 scenes / max 600 steps for all 16 conditions. Physical boxes retained. X0..X3 are unassigned boxes; selected keys blocked in every actor layer/head. QK and bias use a shared logit color scale across every condition. Each sample PNG contains raw QK, original bias and actual masked attention, 4 layers x 2 heads. Hatched columns are masked.</p><a href="mask_progression.png"><img src="mask_progression.png"></a><table><tr><th>Mask</th><th>Falls /64</th><th>Agent goal success</th><th>PNG gallery</th></tr>']
rows=[]
for r in records:
 tag=r['tag'];render=json.loads((root/tag/'render.json').read_text())
 html.append(f'<tr><td>{tag}</td><td>{r["scene_falls"]}</td><td>{r["agent_ever_goal_success_rate"]:.2%}</td><td><a href="{tag}/index.html">{render["png_count"]} PNG</a></td></tr>')
 rows.append(f'| {tag} | {r["scene_falls"]}/64 | {r["agent_ever_goal_success_rate"]:.2%} | {render["png_count"]} |')
html.append('</table>')
for r in records:
 tag=r['tag'];html.append(f'<h2>{tag}</h2>')
 for p in sorted((root/tag).glob('scene_*.png')):html.append(f'<a href="{tag}/{p.name}"><p>{p.stem}</p><img loading="lazy" src="{tag}/{p.name}"></a>')
(root/'index.html').write_text(''.join(html))
count_summary=[]
for k in range(5):
 group=[r for r in records if r['masked_count']==k]
 rates=[r['scene_fall_rate'] for r in group]
 count_summary.append(dict(masked_count=k,combinations=len(group),fall_rate_mean=float(np.mean(rates)),fall_rate_min=min(rates),fall_rate_max=max(rates),agent_goal_rate_mean=float(np.mean([r['agent_ever_goal_success_rate'] for r in group]))))
(root/'summary.json').write_text(json.dumps(dict(count_summary=count_summary,shared_raw_logit_bound=bound,total_png=sum(json.loads((root/r['tag']/'render.json').read_text())['png_count'] for r in records)),indent=2))
report='''# Native Y2m: 비대상 박스 key masking ablation

4명·대상4개+비대상4개·목표4개, 업로드6000 모델. GPU5 기존MPS, seed2141, 조건당64장면×최대600step, 모든16개 subset. 모든 조건의 초기 physics/obs SHA256 동일. actor의 모든층·head·query에서 선택한 비대상박스로 향하는 key만 -inf로 차단. 물리 박스·담당박스·목표·보상·critic·가중치/RMS 유지. shuffle permutation은 원래와 같이 forward당1회 소비하며 저장할 때 canonical 순서로 복원. mask 열 확률0과 실제층 출력 재구성을 검증.

| 조건(X index) | 장면낙상 | Agent 목표도달 | PNG 수 |
|---|---:|---:|---:|
'''+ '\n'.join(rows)+'''

같은64장면을 반복한 단일seed 비교이며 조합을 독립 표본으로 취급하지 않는다. 첫 scene 종료에서 동료를 검열한다. 성공은 원본 CONTACT+AT done. 비대상박스 masking만의 실험으로, 다른 agent 담당박스/사람/목표의 영향이나 토큰 수 자체를 제거하는 실험은 아니다. 그림은 첫2개장면에서 종료 전 시점만 포함한다. later-layer QK는 앞층의 mask 효과를 이미 포함한다. QK는 GTA 변환 후 QK/sqrt(d), softmax 전 값. bias패널은 원래 bias, 실제 attention패널은 mask 적용 결과. 전체조건 공통색상범위.
'''
(root/'REPORT.md').write_text(report)
print('SUMMARY',json.dumps(count_summary),flush=True)
