"""Build an offline skeleton player and contact sheet from retarget outputs."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[3]

HTML = r'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>BONES AMP 모션 확인</title>
<style>body{font:16px system-ui;background:#101827;color:#e5edf8;margin:24px}select,button,input{font:inherit;margin:8px}canvas{width:100%;height:65vh;background:#172337;border-radius:12px}p{color:#b7c7dc}#scrub{width:60%}</style>
<h2>BONES → Humanoid 모션 확인</h2><p>파랑: 변환된 humanoid · 회색: 크기를 맞춘 원본 관절 위치. 문·상자·접촉 정보는 포함하지 않습니다.</p>
<select id="clip"></select><button id="play">일시 정지</button><select id="speed"><option value=".5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select>
<label>시점 <input id="yaw" type="range" min="-180" max="180" value="30"></label><label><input id="follow" type="checkbox" checked>사람 따라가기</label><label><input id="source" type="checkbox" checked>원본 비교</label>
<canvas id="canvas"></canvas><br><input id="scrub" type="range" min="0" value="0"><span id="time"></span><p id="audit"></p>
<script>
const clips=__DATA__;
const $=id=>document.getElementById(id),c=$('canvas'),ctx=c.getContext('2d');
clips.forEach((m,i)=>{let o=document.createElement('option');o.value=i;o.textContent=m.name;$('clip').append(o)});
let m,frame=0,playing=true,last=0,elapsed=0;
function select(){m=clips[+$('clip').value];frame=0;elapsed=0;$('scrub').max=m.positions.length-1;$('scrub').value=0;$('audit').textContent=m.flags.length?'검토 항목: '+m.flags.join(', '):'수치 검사: 검토 기준 초과 없음';draw()}
$('clip').onchange=select;$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'일시 정지':'재생'};
$('scrub').oninput=()=>{frame=+$('scrub').value;elapsed=frame/m.fps;draw()};
['yaw','source','follow'].forEach(id=>$(id).oninput=draw);
function draw(){if(!m)return;let rect=c.getBoundingClientRect();c.width=rect.width*devicePixelRatio;c.height=rect.height*devicePixelRatio;ctx.scale(devicePixelRatio,devicePixelRatio);let w=rect.width,h=rect.height;ctx.clearRect(0,0,w,h);
let pos=m.positions[frame],ref=m.source_positions[frame],angle=+$('yaw').value*Math.PI/180;
let center=$('follow').checked?pos[0]:m.positions[0][0];let scale=Math.min(h/2.5,w/6);
function project(p){let x=p[0]-center[0],y=p[1]-center[1];return [w/2+scale*(x*Math.cos(angle)-y*Math.sin(angle)),h*.83-scale*p[2]+scale*.22*(x*Math.sin(angle)+y*Math.cos(angle))]}
function line(a,b,col,width=2){a=project(a);b=project(b);ctx.strokeStyle=col;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke()}
for(let i=-6;i<=6;i++){line([center[0]+i,center[1]-6,0],[center[0]+i,center[1]+6,0],'#2b3a50',1);line([center[0]-6,center[1]+i,0],[center[0]+6,center[1]+i,0],'#2b3a50',1)}
function skeleton(p,col,width){for(let j=1;j<m.parents.length;j++)line(p[m.parents[j]],p[j],col,width);for(let j=0;j<p.length;j++){let a=project(p[j]);ctx.fillStyle=col;ctx.beginPath();ctx.arc(...a,width+1,0,Math.PI*2);ctx.fill()}}
if($('source').checked)skeleton(ref,'#7c899b',2);skeleton(pos,'#59c8ff',4);
$('time').textContent=(frame/m.fps).toFixed(2)+' / '+((m.positions.length-1)/m.fps).toFixed(2)+'초';$('scrub').value=frame}
function tick(t){let dt=last?Math.min((t-last)/1000,.1):0;last=t;if(playing&&m){elapsed+=dt*+$('speed').value;frame=Math.floor(elapsed*m.fps)%m.positions.length;draw()}requestAnimationFrame(tick)}select();requestAnimationFrame(tick);window.onresize=draw;
</script></html>'''


def build():
    clips=[]
    for folder in ('dataset_bones_dooropen','dataset_bones_push'):
        for path in sorted((ROOT/'tokenhsi/data'/folder/'motions').glob('*/phys_humanoid_v3/preview_data.json')):
            m=json.loads(path.read_text())
            audit=json.loads((path.parent/'conversion_report.json').read_text())
            m['flags']=audit['review_flags']
            clips.append(m)
    if not clips: raise ValueError('No converted motions')
    out=ROOT/'output/bones_amp_conversion_check'
    out.mkdir(parents=True,exist_ok=True)
    (out/'preview.html').write_text(HTML.replace('__DATA__',json.dumps(clips)))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # One original left/right door and one push; samples span the full clip.
    names=['inside_door_handle_left_side_open_walk_R_001__A515',
           'inside_door_handle_right_side_open_walk_R_001__A515','push_obstacle_180_101__A341']
    chosen=[m for name in names for m in clips if m['name']==name]
    fig,axes=plt.subplots(len(chosen),5,figsize=(18,4*len(chosen)),squeeze=False)
    for row,m in enumerate(chosen):
        positions=np.asarray(m['positions']); source=np.asarray(m['source_positions'])
        for col,f in enumerate(np.linspace(0,len(positions)-1,5).astype(int)):
            ax=axes[row,col]; center=positions[f,0,:2]
            for points,color,width in [(source[f],'#9ca3af',1),(positions[f],'#0284c7',2)]:
                for j,parent in enumerate(m['parents']):
                    if parent<0:continue
                    pair=points[[parent,j]]
                    x=(pair[:,0]-center[0])*.85-(pair[:,1]-center[1])*.52
                    z=pair[:,2]+(pair[:,0]-center[0])*.1
                    ax.plot(x,z,color=color,lw=width,marker='o',ms=3)
            ax.axhline(0,color='black',lw=.8);ax.set_xlim(-1,1);ax.set_ylim(-.1,2);ax.set_aspect('equal')
            ax.set_title(f'{f/m["fps"]:.2f}s');ax.set_xlabel('projected metres')
            if col==0:ax.set_ylabel(m['name'].replace('inside_door_handle_','').replace('_open_walk_R_001',''))
    fig.tight_layout();fig.savefig(out/'contact_sheet.png',dpi=130);plt.close(fig)
    print('Preview:',out/'preview.html','clips:',len(clips),flush=True)


if __name__=='__main__':build()
