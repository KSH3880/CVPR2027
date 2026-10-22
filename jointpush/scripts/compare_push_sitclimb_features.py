"""Compare within-model matched-state context sensitivity by task."""
import sys,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(sys.argv[1]);push=Path(sys.argv[2]);ksh=Path(sys.argv[3]);root.mkdir(parents=True,exist_ok=True)
def metrics(a,b):
 d=b-a;na=np.linalg.norm(a,axis=-1);nb=np.linalg.norm(b,axis=-1)
 return dict(rms=float(np.sqrt(np.mean(d*d))),relative_l2_mean=float(np.mean(np.linalg.norm(d,axis=-1)/(na+1e-12))),cosine_mean=float(np.mean(np.sum(a*b,axis=-1)/(na*nb+1e-12))),max_abs=float(np.max(np.abs(d))))
rows=[];scatter=[]
groups=[('Push',push,[0,1,2,3]),('Sit',ksh,[0,2]),('Climb',ksh,[1,3])]
if len(sys.argv)>4:groups.append(('Holding+AT',Path(sys.argv[4]),[0,1,2,3]))
for label,folder,ids in groups:
 data=np.load(folder/'features.npz');f=data['features'][0];a=data['actions'][0]
 two=np.take(f[:,0],ids,axis=2);four=np.take(f[:,1],ids,axis=2)
 mu2=np.take(a[:,0],ids,axis=1);mu4=np.take(a[:,1],ids,axis=1)
 layers=[dict(layer=l,**metrics(two[:,l],four[:,l])) for l in range(5)]
 rows.append(dict(task=label,source=str(folder),scenes=len(f),agents_per_scene=len(ids),layers=layers,action_raw=metrics(mu2,mu4),action_clipped=metrics(mu2.clip(-1,1),mu4.clip(-1,1)),action_dof_difference_gt_0_1_fraction=float(np.mean(np.abs(mu4.clip(-1,1)-mu2.clip(-1,1))>.1))))
 scatter.append((mu2.ravel(),mu4.ravel()))
(root/'summary.json').write_text(json.dumps(rows,indent=2))
columns=max(3,len(rows))
fig,axes=plt.subplots(2,columns,figsize=(6.3*columns,11),layout='constrained');labels=['Tokenizer','L1','L2','L3','L4']
for row in rows:
 axes[0,0].plot(range(5),[v['relative_l2_mean']*100 for v in row['layers']],marker='o',label=row['task'])
 axes[0,1].plot(range(5),[v['cosine_mean'] for v in row['layers']],marker='o',label=row['task'])
for ax in axes[0,:2]:ax.set_xticks(range(5),labels);ax.grid(alpha=.3);ax.legend()
axes[0,0].set_title('Relative human feature change (%)');axes[0,1].set_title('Human feature cosine similarity')
x=np.arange(len(rows));w=.35
axes[0,2].bar(x-w/2,[r['layers'][-1]['relative_l2_mean']*100 for r in rows],w,label='Final feature')
axes[0,2].bar(x+w/2,[r['action_raw']['relative_l2_mean']*100 for r in rows],w,label='Action mean')
axes[0,2].set_xticks(x,[r['task'] for r in rows]);axes[0,2].set_title('Relative L2 change: feature vs action (%)');axes[0,2].legend()
if columns>3:
 axes[0,3].axis('off')
 text='Initial matched-state relative L2\n\n'+ '\n'.join(f'{r["task"]}: feature {r["layers"][-1]["relative_l2_mean"]:.2%}, action {r["action_raw"]["relative_l2_mean"]:.2%}' for r in rows)
 axes[0,3].text(0,.75,text,fontsize=12,va='top')
lo=min(min(a.min(),b.min()) for a,b in scatter);hi=max(max(a.max(),b.max()) for a,b in scatter)
for col,((a,b),row) in enumerate(zip(scatter,rows)):
 ax=axes[1,col];ax.scatter(a,b,s=3,alpha=.25);ax.plot([lo,hi],[lo,hi],'k--',lw=1);ax.set_xlim(lo,hi);ax.set_ylim(lo,hi);ax.set_xlabel('Raw action mean: 2-agent context');ax.set_ylabel('Raw action mean: 4-agent context');ax.set_title(f'{row["task"]}: RMS delta {row["action_raw"]["rms"]:.4f}')
fig.suptitle('Same retained state within each model: 2-agent vs 4-agent context\n32 initial scenes/model; Push: boxes2->4, KSH: boxes4->8. Models/tasks/layouts differ.',fontsize=16)
fig.savefig(root/'push_sit_climb_comparison.png',dpi=160);plt.close(fig)
table=['| Task | Final feature relative L2 | Final feature cosine | Action relative L2 | Action RMS |','|---|---:|---:|---:|---:|']
for r in rows:table.append(f'| {r["task"]} | {r["layers"][-1]["relative_l2_mean"]:.2%} | {r["layers"][-1]["cosine_mean"]:.6f} | {r["action_raw"]["relative_l2_mean"]:.2%} | {r["action_raw"]["rms"]:.6f} |')
report='''# Push vs Sit/Climb: 같은 상태의2명→4명 feature/action 변화

각 모델 안에서 같은 사람의 proprioception, 같은 담당물체·목표·pose와 task를 유지하고 추가토큰만 넣거나뺐다. 동일 학습가중치/RMS를 사용했다. 초기32장면을 주비교로 쓰며 Push는128agent, Sit/Climb은각64agent이다. Sit/Climb은기존39000 distill checkpoint이며 같은4인scene의H0/H2가Sit,H1/H3가Climb이다. Push는6000checkpoint다.

'''+ '\n'.join(table)+'''

`상대L2`는 에이전트별 ||x4-x2||/||x2||의 평균이다. feature차원64, action은같은head의rawmean이다. 따라서 feature와action의 스케일이 달라 rawRMS만으로head증폭을판정하지않는다. clippedaction은rawmean을[-1,1]로제한한값이며 실제PD후물리변화를측정한것은아니다.

Push는2명/박스2개→4명/박스4개, KSH는원래학습비율인2명/박스4개→4명/박스8개로비교했다. KSH의비대상박스4개는logical extra0/1을첫pair,2/3을둘째pair에배정했다. source관측에서각pair의사람·물체·목표·GTApose와taskpacket을추출하고index만remap했다. 모든시점에서tokenizer입력human차이0, 그룹간mask4인과2인encoder출력은수치정밀도범위에서일치했다. 가중치/RMS불변, CPU만사용.

관찰: KSH는최종human feature와action의상대변화가작고, Push는encoder층을거치며차이가커진다. 따라서이번샘플에서는확장성차이가head에서만생긴것이아니라encoder출력단계부터뚜렷하다. KSHhead가큰feature변화를흡수했다는설명보다입력feature자체가안정적이라는설명에맞는다. 다만모델간태스크·장면·물체수·학습이력이다르므로bias방식이나distillation을단독원인으로확정할수없다. feature/action차이만으로낙상인과를확정하지않는다.
'''
if len(rows)>3:
 h=rows[-1]
 report+='\n구버전 HOLDING+AT unified6000도 추가했다. HOLDING과AT를별도edge로표현하는edge_mlp 방식이며2명/4box→4명/8box를비교했다. 최종feature상대L2 '+f"{h['layers'][-1]['relative_l2_mean']:.2%}"+'와action상대L2 '+f"{h['action_raw']['relative_l2_mean']:.2%}"+'를얻었다. 따라서edge_mlp라는bias방식이같다는사실만으로확장성실패를설명할수없다. 모델간환경·학습이력차이는남아있다.\n'
(root/'REPORT.md').write_text(report)
(root/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Push / Sit / Climb feature comparison</h1><a href="push_sit_climb_comparison.png"><img width="1200" src="push_sit_climb_comparison.png"></a><p>See REPORT.md and summary.json for metrics and limitations.</p>')
print('\n'.join(table),flush=True)
