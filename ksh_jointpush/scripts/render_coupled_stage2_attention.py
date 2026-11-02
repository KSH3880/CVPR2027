"""Render human-to-human Stage2 attention from actual actor rollout inputs."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'ksh_jointpush/coupled_base/tokenhsi')]
import numpy as np
import torch
from ksh_jointpush.ksh_torch_compat import install_attention_compat
install_attention_compat()
from ksh_jointpush.ksh_stage2_attention import canonical_coordination_maps
from learning.multi_agent.human_coordination import HumanCoordination
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

OUT = Path(sys.argv[1]).resolve()
meta = json.loads((OUT/'capture.json').read_text())
assert meta['groups'] == [0, 1, 1, 1], 'This legacy renderer is for 1plus3; use render_coupled_stage2_comparison.py for three layouts'
states = json.loads((OUT/'states.json').read_text())
data = torch.load(OUT/'actor_coordination_inputs.pt', map_location='cpu')
checkpoint = torch.load(meta['checkpoint'], map_location='cpu')
branch = HumanCoordination(num_layers=meta['layer_count'], heads=meta['head_count'],
                           shuffle_human_order=True, relations=4)
prefix = 'a2c_network.coordination.'
branch.load_state_dict({key[len(prefix):]: value for key, value in checkpoint['model'].items()
                        if key.startswith(prefix)}, strict=True)
trace = canonical_coordination_maps(branch, data['humans'], data['packet'], data['groups'])
error = float((trace['output']-data['output']).abs().max())
assert error < 3e-5, error
arrays = {key: trace[key].numpy() for key in ('qk_scores', 'relation_bias', 'attention')}
np.savez_compressed(OUT/'stage2_attention_raw.npz', **arrays)
assert np.isfinite(arrays['attention']).all()
row_error = float(np.abs(arrays['attention'].sum(-1)-1).max())
assert row_error < 2e-6
labels = ['H0 solo', 'H1 team', 'H2 team', 'H3 team']
colors = ['#6ebf24', '#3984d3', '#d48b22', '#ad42bc']
relation = np.array([[1,0,0,0], [0,1,3,3], [0,3,1,3], [0,3,3,1]])
attention = arrays['attention']
frames, layers, heads = attention.shape[:3]
mean = attention.mean(0)


def heatmap_figure(values, path, subtitle):
    fig, axes = plt.subplots(layers, heads, figsize=(10.5, 15), constrained_layout=True)
    axes = np.asarray(axes).reshape(layers, heads)
    for layer in range(layers):
        for head in range(heads):
            ax = axes[layer, head]
            image = ax.imshow(values[layer, head], vmin=0, vmax=1, cmap='viridis')
            ax.set_xticks(range(4), labels, rotation=15, fontsize=9)
            ax.set_yticks(range(4), labels, fontsize=9)
            for index, color in enumerate(colors):
                ax.get_xticklabels()[index].set_color(color)
                ax.get_yticklabels()[index].set_color(color)
            ax.axvline(.5, color='white', lw=1.5)
            ax.axhline(.5, color='white', lw=1.5)
            ax.set_title(f'Layer {layer+1} / Head {head+1}')
            ax.set_xlabel('Key agent (attended to)')
            ax.set_ylabel('Query agent (attending)')
            for i in range(4):
                for j in range(4):
                    v = float(values[layer, head, i, j])
                    ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                            color='black' if v>.65 else 'white', fontsize=11)
    fig.colorbar(image, ax=axes, shrink=.6, label='Attention probability; each row sums to 1')
    fig.suptitle(f'Stage2 actor human coordination | {checkpoint["epoch"]} iter | 1 solo + 3 coupled\n'+subtitle,
                 fontsize=14)
    fig.savefig(path, dpi=170)
    plt.close(fig)


heatmap_figure(mean, OUT/'stage2_attention_mean.png',
              f'Uniform mean of {frames} sampled pre-action states / {meta["episodes"]} rollouts')
heatmap_figure(attention[0], OUT/'stage2_attention_initial.png',
              'First rollout / initial state / canonical agent IDs')
bias = torch.stack([block.relation_bias for block in branch.blocks]).detach().numpy()
fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
bound = max(float(np.abs(bias).max()), 1e-4)
im = ax.imshow(bias.reshape(layers*heads, 4), cmap='coolwarm', vmin=-bound, vmax=bound)
ax.set_xticks(range(4), ['NONE', 'SELF', 'BEFORE\n(absent here)', 'COUPLED'])
ax.set_yticks(range(layers*heads), [f'L{l+1} H{h+1}' for l in range(layers) for h in range(heads)])
for i, row in enumerate(bias.reshape(-1, 4)):
    for j, value in enumerate(row):
        ax.text(j, i, f'{value:+.4f}', ha='center', va='center', fontsize=10,
                color='white' if abs(value)>.6*bound else 'black')
ax.set_title('Learned Stage2 relation bias added to QK logits\nThis is an additive bias, not a hard group mask')
fig.colorbar(im, ax=ax, shrink=.8, label='Logit bias')
fig.savefig(OUT/'stage2_relation_bias.png', dpi=170)
plt.close(fig)

by_relation = {name: attention[..., relation == index].mean(-1)
               for index, name in [(1, 'SELF'), (3, 'COUPLED peer'), (0, 'NONE cross-group')]}
episodes = sorted({state['episode'] for state in states})
fig, axes = plt.subplots(len(episodes), 1, figsize=(12, 3.3*len(episodes)), constrained_layout=True)
for ax, episode in zip(np.atleast_1d(axes), episodes):
    selected = np.array([s['episode']==episode for s in states])
    times = np.array([s['step']*s['control_dt'] for s in states])[selected]
    for name, values in by_relation.items():
        ax.plot(times, values.mean((1,2))[selected], label=name)
    ax.axhline(.25, ls=':', color='gray', label='Uniform / 4 agents')
    ax.set_ylim(0,1)
    ax.set_xlabel('Simulation seconds')
    ax.set_ylabel('Mean weight per edge')
    ax.set_title(f'Rollout {episode+1} | all 4 layers and 2 heads averaged')
    ax.legend(loc='best', ncol=2)
fig.savefig(OUT/'stage2_attention_timeline.png', dpi=160)
plt.close(fig)


def corners(box, size):
    qx,qy,qz,qw = box[3:7]
    yaw = np.arctan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))
    rot = np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
    return np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*np.asarray(size[:2])/2 @ rot.T + np.asarray(box[:2])


state = states[0]
fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), constrained_layout=True,
                         gridspec_kw={'width_ratios':[1.4,1,1]})
ax = axes[0]
for slot,box,size in zip(state['physical_slots'],state['boxes'],state['box_sizes']):
    color = colors[0] if slot == state['assignment'][0] else '#d2b52a'
    ax.add_patch(Polygon(corners(box,size), fc=color, alpha=.5, ec='black'))
    ax.text(box[0],box[1], 'solo box' if slot==state['assignment'][0] else '3-agent box',
            ha='center', va='center', fontsize=9)
for i,(human,goal) in enumerate(zip(state['humans'],state['goals'])):
    ax.scatter(human[0],human[1],c=colors[i],s=100,edgecolors='black')
    ax.text(human[0]+.15,human[1]+.15,f'H{i}',fontweight='bold')
    ax.scatter(goal[0],goal[1],c=colors[i],marker='x',s=90)
ax.autoscale(); ax.margins(.2); ax.set_aspect('equal'); ax.grid(alpha=.2)
ax.set_title('Actual initial scene: dots=agents / crosses=goals')
ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
for head, ax in enumerate(axes[1:]):
    ax.imshow(attention[0,-1,head],vmin=0,vmax=1,cmap='viridis')
    ax.set_xticks(range(4),labels,rotation=35); ax.set_yticks(range(4),labels)
    ax.set_title(f'Last layer / head {head+1}')
    for i in range(4):
        for j in range(4):
            ax.text(j,i,f'{attention[0,-1,head,i,j]:.2f}',ha='center',va='center',color='white')
fig.savefig(OUT/'stage2_initial_scene.png',dpi=165); plt.close(fig)
summary = dict(checkpoint_epoch=checkpoint['epoch'], frames=frames, episodes=meta['episodes'],
    layout='frame, layer, head, canonical query agent, canonical key agent',
    canonical_replay_output_max_error=error, mha_reference_max_error=trace['mha_reference_max_error'],
    row_sum_max_error=row_error, actual_policy_inputs=True,
    model_and_rms_unchanged=meta['model_and_rms_unchanged'],
    unchanged_action_max_error=meta['unchanged_action_max_error'],
    mean_per_edge={name:float(value.mean()) for name,value in by_relation.items()},
    layer_head_mean_per_edge={name:value.mean(0).tolist() for name,value in by_relation.items()},
    learned_relation_bias=bias.tolist(), groups=[0,1,1,1],
    note='Attention probability describes routing; output projections and residuals also affect influence. No group-specific hard attention mask is applied in this checkpoint. QK uses the original upstream features, including prior-layer biases; displaying QK alone is not a no-bias ablation.')
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

payload = dict(bounds={key:float(np.abs(value).max()) for key,value in arrays.items()}, attention=attention.round(6).tolist(), qk_scores=arrays['qk_scores'].round(6).tolist(),
               relation_bias=arrays['relation_bias'].round(6).tolist(), states=states, summary=summary)
html = r'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Stage2 agent attention</title>
<style>body{margin:0;background:#f3f5f8;color:#1f2937;font:15px system-ui}main{max-width:1280px;margin:auto;padding:28px}h1{font-size:25px;margin:0 0 9px}p{line-height:1.5}.toolbar,.card{background:white;border-radius:10px;padding:15px;margin:12px 0;box-shadow:0 2px 10px #0001}.toolbar{position:sticky;top:0;z-index:2;display:flex;gap:16px;align-items:center;flex-wrap:wrap}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}table{border-collapse:collapse;margin:auto}td,th{padding:9px;text-align:center}td{width:70px;height:48px;border:1px solid #ffffff44;font-variant-numeric:tabular-nums}th{font-size:12px}.name{color:#536174;font-size:13px}canvas{max-width:100%;background:#fcfcfd}a{color:#2164a8}select{padding:6px}input[type=range]{width:350px}@media(max-width:760px){.grid{grid-template-columns:1fr}}</style>
<main><h1>Stage2: agent 간 attention</h1><p>__ITERATION__ iter · Actor coordination · 1명 단독(H0/초록) + 3명 공동(H1–H3) · 4 layers × 2 heads<br>행: 보는 agent(query), 열: 정보를 받는 agent(key). 원본 추론 입력을 agent 번호 순서로 재생했습니다.</p>
<div class="toolbar"><label>표시 <select id="component"><option value="attention">Attention probability</option><option value="qk_scores">QK / sqrt(d)</option><option value="relation_bias">Learned relation bias</option></select></label><label><input type="checkbox" id="mean"> 전체 수집 시점 평균</label><input id="frame" type="range" min="0" value="0"><b id="caption"></b></div>
<div class="card"><canvas id="scene" width="900" height="370"></canvas><p id="status" class="name"></p></div><div class="grid" id="maps"></div>
<p><a href="stage2_attention_mean.png">평균 PNG</a> · <a href="stage2_attention_initial.png">첫 시점 PNG</a> · <a href="stage2_relation_bias.png">관계 bias PNG</a> · <a href="stage2_attention_timeline.png">시간 변화 PNG</a> · <a href="stage2_attention_raw.npz">원시 배열</a></p>
<p class="name">관측한 attention은 정보의 경로이며, 행동에 대한 인과적 기여량과 같지는 않습니다. NONE 관계도 현재 구조에서는 hard mask로 차단되지 않습니다. 초기화가 identity여도 attention weight 자체는 균등하지 않을 수 있습니다.</p></main>
<script>const D=PAYLOAD;const colors=['#6ebf24','#3984d3','#d48b22','#ad42bc'];const labels=['H0 solo','H1 team','H2 team','H3 team'];const f=document.getElementById('frame');f.max=D.states.length-1;let means={};for(const name of ['attention','qk_scores','relation_bias']){means[name]=D[name][0].map((l,li)=>l.map((h,hi)=>h.map((row,i)=>row.map((v,j)=>D[name].reduce((a,x)=>a+x[li][hi][i][j],0)/D.states.length))))}
function color(v,name,b){if(name!=='attention'){let t=Math.min(1,Math.abs(v)/b);return v>=0?`rgb(\${255-40*t},\${255-170*t},\${255-190*t})`:`rgb(\${255-190*t},\${255-150*t},255)`}let p=[[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]],t=Math.min(3.999,Math.max(0,v)*4),i=Math.floor(t),u=t-i;return `rgb(\${p[i].map((x,j)=>Math.round(x*(1-u)+p[i+1][j]*u)).join(',')})`}
function drawScene(s){let c=document.getElementById('scene'),x=c.getContext('2d');x.clearRect(0,0,c.width,c.height);let pts=[...s.humans,...s.boxes,...s.goals],minX=Math.min(...pts.map(p=>p[0]))-4,maxX=Math.max(...pts.map(p=>p[0]))+4,minY=Math.min(...pts.map(p=>p[1]))-4,maxY=Math.max(...pts.map(p=>p[1]))+4,scale=Math.min((c.width-50)/(maxX-minX),(c.height-40)/(maxY-minY));let P=p=>[25+(p[0]-minX)*scale,c.height-20-(p[1]-minY)*scale];x.font='14px system-ui';for(let n=0;n<s.boxes.length;n++){let b=s.boxes[n],z=s.box_sizes[n],q=b.slice(3),yaw=Math.atan2(2*(q[3]*q[2]+q[0]*q[1]),1-2*(q[1]*q[1]+q[2]*q[2])),p=P(b);x.save();x.translate(...p);x.rotate(-yaw);x.fillStyle=s.physical_slots[n]===s.assignment[0]?'#92cd4d88':'#ddc34488';x.strokeStyle='#444';x.fillRect(-z[0]*scale/2,-z[1]*scale/2,z[0]*scale,z[1]*scale);x.strokeRect(-z[0]*scale/2,-z[1]*scale/2,z[0]*scale,z[1]*scale);x.restore()}s.humans.forEach((h,i)=>{let p=P(h),g=P(s.goals[i]);x.fillStyle=colors[i];x.beginPath();x.arc(...p,7,0,2*Math.PI);x.fill();x.fillText('H'+i,p[0]+9,p[1]-9);x.strokeStyle=colors[i];x.beginPath();x.moveTo(g[0]-6,g[1]-6);x.lineTo(g[0]+6,g[1]+6);x.moveTo(g[0]-6,g[1]+6);x.lineTo(g[0]+6,g[1]-6);x.stroke()});document.getElementById('status').textContent='점=agent / 사각형=상자 / ×=개인 목표 · contact: '+s.contact.map(v=>v?'1':'0').join(' ')+' · group gate: '+s.group_gate.map(v=>v?'1':'0').join(' ')}
function update(){let index=+f.value,name=document.getElementById('component').value,avg=document.getElementById('mean').checked,a=avg?means[name]:D[name][index],bound=Math.max(.0001,D.bounds[name]);f.disabled=avg;let s=D.states[index];document.getElementById('caption').textContent=avg?D.states.length+' 시점 평균':`장면 \${s.episode+1} · step \${s.step}`;let html='';for(let l=0;l<a.length;l++)for(let h=0;h<a[l].length;h++){html+=`<div class="card"><b>Layer \${l+1} / Head \${h+1}</b><p class="name">\${name==='attention'?'행 합 = 1':'Logit 값; 0 중심 색상'}</p><table><tr><th>Query ↓ / Key →</th>\${labels.map((v,j)=>`<th style="color:\${colors[j]}">\${v}</th>`).join('')}</tr>`;for(let i=0;i<4;i++){html+=`<tr><th style="color:\${colors[i]}">\${labels[i]}</th>`;for(let j=0;j<4;j++){let v=a[l][h][i][j],sep=(i===0?'border-bottom:3px solid #ddd;':'')+(j===0?'border-right:3px solid #ddd;':'');html+=`<td style="\${sep}background:\${color(v,name,bound)};color:\${name==='attention'&&v<.6?'white':'#172032'}">\${v.toFixed(3)}</td>`}html+='</tr>'}html+='</table></div>'}document.getElementById('maps').innerHTML=html;drawScene(s)}
for(const id of ['frame','component','mean'])document.getElementById(id).addEventListener('input',update);update();</script></html>'''
# Python string literals escape template interpolation solely for readability.
html = html.replace('\\${', '${').replace('__ITERATION__',str(checkpoint['epoch'])).replace('PAYLOAD', json.dumps(payload, separators=(',',':')))
(OUT/'index.html').write_text(html)
(OUT/'README.md').write_text('Stage2 actor coordination attention\n\nOpen index.html for per-state maps. Rows=query agent; columns=key agent. H0 is solo; H1-H3 share a box. Raw arrays use frame/layer/head/query/key. The actual rollout and RNG were unchanged. Canonical replay matches the recorded actor output. Relation bias is separate from the softmax attention and is not a hard mask.\n')
print('STAGE2_ATTN_RENDERED', json.dumps(summary), flush=True)
