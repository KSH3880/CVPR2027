"""CPU-only preview of the converted humanoid and the current push-door geometry."""
from pathlib import Path
import json
import sys
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation
from PIL import Image
from lpanlib.poselib.skeleton.skeleton3d import SkeletonMotion

OUTPUT = ROOT/'tokenhsi/data/dataset_bones_door/motions/inside_door_handle_right_side_open_walk_R_001__A515_M/phys_humanoid_v3'
XML = ROOT/'tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml'


def cube(center, size, rotation=None):
    vertices = np.array([[x,y,z] for x in (-.5,.5) for y in (-.5,.5) for z in (-.5,.5)])*size
    if rotation is not None:
        vertices = vertices@rotation.T
    vertices += center
    return vertices, [[0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5]]


def build():
    m=SkeletonMotion.from_file(str(OUTPUT/'ref_motion.npy'))
    turn=Rotation.from_euler('z',90,degrees=True).as_matrix()
    pos=m.global_translation.numpy()@turn.T
    rot=turn@Rotation.from_quat(m.global_rotation.numpy().reshape(-1,4)).as_matrix().reshape(-1,15,3,3)
    anchor=38
    pos[:,:,:2]+=np.array([.78,-.1])-pos[anchor,8,:2]
    angles=np.arctan2(pos[:,8,1],pos[:,8,0])-np.arctan2(-.1,.78)
    angles=np.maximum.accumulate(np.clip(angles,0,1.7));angles[:anchor]=0
    handle=np.stack([.78*np.cos(angles)+.1*np.sin(angles),.78*np.sin(angles)-.1*np.cos(angles),np.ones(len(angles))],-1)
    distance=np.linalg.norm(pos[:,8]-handle,axis=-1)
    data=dict(fps=30,nodes=[dict(name=n,parent=int(p)) for n,p in zip(m.skeleton_tree.node_names,m.skeleton_tree.parent_indices)],
              positions=pos.round(5).tolist(),angles=angles.round(5).tolist(),handle=handle.round(5).tolist(),distance=distance.round(5).tolist())
    (OUTPUT/'door_preview_data.json').write_text(json.dumps(data))
    html='''<!doctype html><meta charset="utf-8"><title>TokenHSI BONES Door AMP preview</title>
<style>body{font:15px system-ui;margin:20px;background:#f5f7fa;color:#182333}canvas{width:100%;height:65vh;background:white;border:1px solid #ccd4df}button{padding:8px}input[type=range]{width:45%}</style>
<h2>왼손으로 왼쪽에 여는 AMP reference — 현재 문과 비교</h2>
<p>현재 humanoid 관절 길이·제한과 실제 door.urdf 치수로 표시합니다. 문 각도는 손의 XY 방향에서 추정한 비교용 값입니다. <b>실측 문 데이터·접촉 검증·정책 성공 영상이 아닙니다.</b></p>
<button id="play">일시정지</button> <label><input id="door" type="checkbox" checked>문 표시</label>
<label>시점 <input id="yaw" type="range" min="-3.14" max="3.14" step=".02" value="-.8"></label>
<canvas id="view"></canvas><p><input id="time" type="range" min="0" max="102" step="1"><span id="stamp"></span></p>
<p>파랑: 왼팔·왼다리, 빨강: 오른팔·오른다리. 손과 손잡이 사이 선은 거리 차이를 보여줍니다. 접촉을 강제로 맞춘 데이터가 아닙니다.</p>
<script>const data=DATA;let t=0,playing=true,last=performance.now();const e=id=>document.getElementById(id),c=e('view'),ctx=c.getContext('2d');
e('play').onclick=()=>{playing=!playing;e('play').textContent=playing?'일시정지':'재생'};e('time').oninput=()=>{t=+e('time').value;playing=false;e('play').textContent='재생'};
function draw(now){let dt=Math.min(.1,(now-last)/1000);last=now;if(playing)t=(t+dt*30)%data.positions.length;let f=Math.floor(t),p=data.positions[f];c.width=c.clientWidth*devicePixelRatio;c.height=c.clientHeight*devicePixelRatio;ctx.clearRect(0,0,c.width,c.height);let scale=Math.min(c.width/4.5,c.height/2.7),yaw=+e('yaw').value,co=Math.cos(yaw),si=Math.sin(yaw);
function proj(p){let x=p[0]-.45,y=p[1]-.35;return[c.width/2+scale*(x*co-y*si),c.height*.85-scale*(p[2]+.18*(x*si+y*co))]}
function line(a,b,color,width){ctx.beginPath();ctx.moveTo(...proj(a));ctx.lineTo(...proj(b));ctx.strokeStyle=color;ctx.lineWidth=width*devicePixelRatio;ctx.stroke()}
for(let k=-2;k<=2;k+=.5){line([k,-2,0],[k,2,0],'#e5e9ef',.5);line([-2,k,0],[2,k,0],'#e5e9ef',.5)}
if(e('door').checked){let a=data.angles[f],tip=[.9*Math.cos(a),.9*Math.sin(a),0],corners=[[0,0,0],tip,[tip[0],tip[1],2],[0,0,2]];ctx.beginPath();corners.forEach((v,i)=>i?ctx.lineTo(...proj(v)):ctx.moveTo(...proj(v)));ctx.closePath();ctx.fillStyle='rgba(155,110,65,.20)';ctx.fill();for(let j=0;j<4;j++)line(corners[j],corners[(j+1)%4],'#80552a',2);line([0,0,0],[0,0,2.1],'#444',5);let h=proj(data.handle[f]);ctx.beginPath();ctx.arc(...h,5*devicePixelRatio,0,Math.PI*2);ctx.fillStyle='#bb8800';ctx.fill();line(p[8],data.handle[f],'#cc7700',1)}
data.nodes.forEach((n,j)=>{if(n.parent>=0)line(p[n.parent],p[j],n.name.startsWith('right')?'#dd4867':'#2777bd',4)});
e('time').value=f;e('stamp').textContent=` ${(f/30).toFixed(2)}s / 3.40s · 추정 문각 ${(data.angles[f]*180/Math.PI).toFixed(1)}° · 손–손잡이 거리 ${(100*data.distance[f]).toFixed(1)}cm`;
requestAnimationFrame(draw)}requestAnimationFrame(draw);</script>'''
    (OUTPUT/'door_preview.html').write_text(html.replace('DATA',json.dumps(data),1))
    body_elements={b.attrib['name']:b for b in ET.parse(XML).iter('body')}
    geom_by_body=[body_elements[n].findall('geom') for n in m.skeleton_tree.node_names]
    images=[]
    selected=list(range(0,len(pos),4))
    if selected[-1]!=len(pos)-1:selected.append(len(pos)-1)
    for f in selected:
        fig=plt.figure(figsize=(7,6));ax=fig.add_subplot(111,projection='3d')
        for j,geoms in enumerate(geom_by_body):
            color='#2777bd' if m.skeleton_tree.node_names[j].startswith('left') else '#dd4867' if m.skeleton_tree.node_names[j].startswith('right') else '#777777'
            for g in geoms:
                if g.get('type')=='mesh':continue
                if g.get('fromto'):
                    points=np.fromstring(g.get('fromto'),sep=' ').reshape(2,3)@rot[f,j].T+pos[f,j]
                    ax.plot(*points.T,color=color,lw=8,solid_capstyle='round')
                else:
                    local=np.fromstring(g.get('pos','0 0 0'),sep=' ');center=rot[f,j]@local+pos[f,j]
                    size=float(g.get('size','.04').split()[0]);u=np.linspace(0,2*np.pi,12);v=np.linspace(0,np.pi,8)
                    sphere=np.stack([size*np.outer(np.cos(u),np.sin(v)),size*np.outer(np.sin(u),np.sin(v)),size*np.outer(np.ones_like(u),np.cos(v))],-1)
                    sphere=sphere@rot[f,j].T+center
                    ax.plot_surface(sphere[:,:,0],sphere[:,:,1],sphere[:,:,2],color=color,alpha=.9,linewidth=0)
        for side in ('left','right'):
            mesh=XML.parent.parent/'smpl_foot_meshes'/f'{side}_foot.obj';vertices=[];faces=[]
            for line in mesh.read_text().splitlines():
                if line.startswith('v '):vertices.append([float(v) for v in line.split()[1:4]])
                elif line.startswith('f '):faces.append([int(v.split('/')[0])-1 for v in line.split()[1:]])
            b=m.skeleton_tree.index(side+'_foot');verts=np.array(vertices)@rot[f,b].T+pos[f,b]
            ax.add_collection3d(Poly3DCollection([verts[face] for face in faces],facecolor='#2777bd' if side=='left' else '#dd4867',alpha=.9))
        r=Rotation.from_euler('z',angles[f]).as_matrix()
        for center,size,rotation,color,alpha in [(r@np.array([.45,0,1]),[.9,.06,2],r,'#a5713c',.25),
                                                ([-.06,0,1],[.1,.12,2.1],None,'#555555',.8),
                                                (handle[f],[.14,.08,.04],r,'#ddad25',1)]:
            vertices,faces=cube(np.array(center),np.array(size),rotation)
            ax.add_collection3d(Poly3DCollection([vertices[face] for face in faces],facecolor=color,edgecolor=color,alpha=alpha))
        ax.plot(*np.stack([pos[f,8],handle[f]]).T,color='#ee9900',lw=1)
        ax.set(xlim=(-.6,1.4),ylim=(-1.,1.6),zlim=(0,2.1),xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)')
        ax.set_box_aspect([2.,2.6,2.1]);ax.view_init(20,-60)
        ax.set_title(f'Retarget FK + current door | {f/30:.2f}s\nDoor angle inferred; NOT contact/policy validation',fontsize=10)
        fig.tight_layout();fig.canvas.draw()
        image=Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy());images.append(image);plt.close(fig)
    images[0].save(OUTPUT/'door_preview.gif',save_all=True,append_images=images[1:],duration=133,loop=0)
    chosen=[images[i] for i in [0,9,14,19,26]]
    sheet=Image.new('RGB',(chosen[0].width*len(chosen),chosen[0].height),'white')
    for i,image in enumerate(chosen):sheet.paste(image,(i*image.width,0))
    sheet.save(OUTPUT/'door_contact_sheet.jpg')
    active=np.arange(len(pos))/30
    window=(active>=1.2666)&(active<=2.4)
    report=dict(door_angle_measured=False,physics_contact_verified=False,contact_anchor_s=anchor/30,
                inferred_angle_max_rad=float(angles.max()),
                hand_handle_distance_1_27_to_2_4_s_median_m=float(np.median(distance[window])),
                hand_handle_distance_1_27_to_2_4_s_max_m=float(distance[window].max()),
                note='Positions from final SkeletonMotion; hand XY aligned once, no Z correction. Door is a diagnostic overlay.')
    (OUTPUT/'door_geometry_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    print(OUTPUT/'door_preview.html')


if __name__=='__main__':build()
