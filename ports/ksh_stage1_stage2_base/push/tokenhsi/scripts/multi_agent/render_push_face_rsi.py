"""Render actual RSI placements with MJCF collision shapes and original foot meshes.

CPU projection of simulator-extracted poses; explicitly not a physics rollout.
"""
import argparse,itertools,json,math,subprocess
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull
from PIL import Image,ImageDraw,ImageFont

parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);parser.add_argument('--config',type=Path,required=True)
parser.add_argument('--show-goal',action='store_true',help='Include the real goal and four-object scene in a top view inset')
args=parser.parse_args();out=args.input;cfg=yaml.safe_load(args.config.read_text())['env'];spec=cfg['pushBox']
data=json.loads((out/'timeline.json').read_text());names=data['names'];parents=data['parents'];scenes=data['scenes'];review=json.loads((out/'scenes.json').read_text())['scenes']
xml=Path(cfg['asset']['assetRoot'])/cfg['asset']['assetFileName'];tree=ET.parse(xml).getroot()
meshes={}
for mesh in tree.findall('./asset/mesh'):
    path=xml.parent/mesh.get('file');v=np.array([[float(x) for x in line.split()[1:4]] for line in path.read_text().splitlines() if line.startswith('v ')])
    meshes[mesh.get('name')]=v*np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
# Samples of each exact sphere/capsule shape, for a projected surface silhouette.
sphere=np.array([[math.cos(a)*math.sin(b),math.sin(a)*math.sin(b),math.cos(b)] for a in np.linspace(0,2*math.pi,16,endpoint=False) for b in np.linspace(0,math.pi,9)])
shapes=[];feet={}
for body in tree.findall('.//body'):
    if body.get('name') not in names:continue
    index=names.index(body.get('name'))
    for geom in body.findall('geom'):
        if geom.get('type')=='mesh':v=meshes[geom.get('mesh')].copy()
        else:
            radius=float(geom.get('size').split()[0]);v=sphere*radius
            if geom.get('fromto'):
                ends=np.fromstring(geom.get('fromto'),sep=' ').reshape(2,3);v=np.concatenate([v+ends[0],v+ends[1]])
        if geom.get('quat'):
            q=np.fromstring(geom.get('quat'),sep=' ');v=R.from_quat(q[[1,2,3,0]]).apply(v)
        v+=np.fromstring(geom.get('pos','0 0 0'),sep=' ')
        shapes.append((index,geom.get('name',body.get('name')),v))
        if body.get('name') in ('right_foot','left_foot'):feet[body.get('name')]=v
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',23);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
signs=np.array(list(itertools.product([-1,1],repeat=3)));edges=[(i,j) for i in range(8) for j in range(i+1,8) if np.count_nonzero(signs[i]!=signs[j])==1]

def canonical(scene):
    body=np.array(scene['body']);box=np.array(scene['box']);r=R.from_quat(box[3:7]).inv()
    p=r.apply(body[:,:3]-box[:3]);p[:,2]+=scene['size'][2]/2;q=(r*R.from_quat(body[:,3:7])).as_quat()
    vertices={n:R.from_quat(q[names.index(n)]).apply(v)+p[names.index(n)] for n,v in feet.items()}
    return p,q,vertices

rows=[]
for s in scenes:
    _,_,f=canonical(s);rows.append([s['clip'],s['time'],f['right_foot'][:,2].min(),f['left_foot'][:,2].min()])
arr=np.array(rows);support=arr[:,2:].min(1)
(out/'feet_audit.json').write_text(json.dumps(dict(cases=len(rows),min_sole_z_m=float(support.min()),max_lower_sole_z_m=float(support.max()),both_soles_above_4cm=int((support>.04).sum()),method='MJCF foot OBJ vertices transformed by actual RSI reset body poses'),indent=2)+'\n')


def draw(scene,mode=None):
    if mode is None:mode='ACTUAL RSI RESET SAMPLES' if data.get('sampled_resets') else 'SOURCE-TIME RSI POSES'
    p,q,f=canonical(scene);size=np.array(scene['size']);co=si=2**-.5
    center=np.array([0.,-.7]);scale=180.
    if scene['label'].startswith('loco'):
        center=(p[0,:2]+np.array([0.,0.]))/2
        scale=min(140. if args.show_goal else 180.,(620 if args.show_goal else 850)/(np.linalg.norm(p[0,:2])+2.5))
    def uv(v):
        x,y=v[:2]-center;return ((470 if args.show_goal else 650)+scale*(co*x+si*y),520-scale*(v[2]+.27*(-si*x+co*y)))
    im=Image.new('RGB',(1280,720),'#f6f8fc');d=ImageDraw.Draw(im)
    for k in np.arange(-7,4.01,.5):
        d.line([uv(np.array([-4.,k,0])),uv(np.array([4.,k,0]))],fill='#e1e7ef')
        d.line([uv(np.array([k,-7.,0])),uv(np.array([k,4.,0]))],fill='#e1e7ef')
    corners=signs*size/2;corners[:,2]+=size[2]/2
    front=[i for i,s in enumerate(signs) if s[1]==-1]
    d.polygon([uv(corners[i]) for i in [front[0],front[1],front[3],front[2]]],fill='#dceef2')
    for i,j in edges:d.line([uv(corners[i]),uv(corners[j])],fill='#71859c',width=3)
    ordered=[]
    for index,name,local in shapes:
        vertices=R.from_quat(q[index]).apply(local)+p[index]
        ordered.append((float((-si*vertices[:,0]+co*vertices[:,1]).mean()),name,vertices))
    for _,name,vertices in sorted(ordered,reverse=True):
        xy=np.array([uv(v) for v in vertices]);hull=ConvexHull(xy)
        color='#59a686' if 'left' in name else '#668db9' if 'right' in name else '#8b9db3'
        d.polygon([tuple(v) for v in xy[hull.vertices]],fill=color,outline='#344b66')
    # Joint overlay shows source skeleton alignment on the actual collision shapes.
    for j,parent in enumerate(parents):
        if parent>=0:d.line([uv(p[parent]),uv(p[j])],fill='#344b66',width=2)
    d.rectangle((0,0,1280,120),fill='#f6f8fc')
    d.text((22,13),f'COMMON FACE XY CONTACT | Y={spec["size_y_m"]:.1f}m | {spec["mass"]:.0f}kg | FRICTION {spec["friction"]:g}',font=font,fill='#172c47')
    d.text((22,49),f'X 1.20-1.50m / Z 1.30m | shown X={size[0]:.2f}m | {scene["label"]} | clip {scene["clip"]} t={scene["time"]:.3f}s',font=small,fill='#41536a')
    d.text((22,80),'Actual reset poses + MJCF collision shapes + original foot meshes. No joint/velocity retargeting.',font=small,fill='#41536a')
    if args.show_goal:
        box=np.array(scene['box']);inverse=R.from_quat(box[3:7]).inv()
        all_boxes=np.array(scene.get('all_boxes',[scene['box']]))
        all_sizes=np.array(scene.get('all_sizes',[scene['size']]))
        centers=inverse.apply(all_boxes[:,:3]-box[:3])
        roots=inverse.apply(np.array(scene.get('human_roots',[np.array(scene['body'])[0,:3]]))-box[:3])
        goals=inverse.apply(np.array(scene.get('all_goals',[scene['goal']]))-box[:3])
        points=np.concatenate((centers[:,:2],roots[:,:2],goals[:,:2]))
        low,high=points.min(0)-2,points.max(0)+2
        center_top=(low+high)/2
        top_scale=min(330/max(high[0]-low[0],1),235/max(high[1]-low[1],1))
        def top(v):return (1080+(v[0]-center_top[0])*top_scale,285-(v[1]-center_top[1])*top_scale)
        d.rounded_rectangle((890,130,1270,450),radius=8,fill='#ffffff',outline='#9badbf',width=2)
        d.text((902,140),'TOP: 2 HUMANS / 4 BOXES',font=small,fill='#172c47')
        for index,(other,other_size) in enumerate(zip(all_boxes,all_sizes)):
            corners2=np.array([[-1,-1,0],[-1,1,0],[1,1,0],[1,-1,0]])*other_size/2
            vertices=R.from_quat(other[3:7]).apply(corners2)+other[:3]
            vertices=inverse.apply(vertices-box[:3])
            active=np.linalg.norm(other[:3]-box[:3])<1e-4
            d.polygon([top(v) for v in vertices],fill='#f2bd61' if active else '#d2dae3',outline='#687b90')
        for index,root in enumerate(roots):
            x,y=top(root);color='#298a67' if index==0 else '#466fd2'
            d.ellipse((x-5,y-5,x+5,y+5),fill=color)
            if index<len(goals):
                gx,gy=top(goals[index]);d.ellipse((gx-5,gy-5,gx+5,gy+5),outline=color,width=2)
        goal_local=inverse.apply(np.array(scene['goal'])-box[:3])
        d.line([top(np.zeros(3)),top(goal_local)],fill='#bc7319',width=3)
        d.text((902,414),'Solid: human | ring: goal',font=small,fill='#41536a')
    mid=p[[names.index('right_hand'),names.index('left_hand')]].mean(0)
    d.rectangle((0,610,1280,720),fill='#f6f8fc')
    line=(f'LOCO: 360-degree random start | full-body collider / box surface gap 1-5m | random heading'
          if scene['label'].startswith('loco') else
          f'Hand midpoint Z {mid[2]:.3f}m (RSI limit 1m; CONTACT ignores height) | face gap {(-mid[1]-size[1]/2)*100:.1f}cm')
    d.text((22,622),line,font=small,fill='#172c47')
    d.text((22,653),f'Actual foot soles: R {f["right_foot"][:,2].min()*1000:.1f}mm / L {f["left_foot"][:,2].min()*1000:.1f}mm above floor',font=small,fill='#172c47')
    d.text((22,685),mode+' | NOT A POLICY OR PHYSICS ROLLOUT',font=small,fill='#586577')
    return im

byclip={c:sorted([s for s in scenes if s['clip']==c],key=lambda s:s['time']) for c in sorted({s['clip'] for s in scenes})}
sheet=Image.new('RGB',(1920,len(byclip)*360),'white')
for row,(clip,seq) in enumerate(byclip.items()):
    for col,idx in enumerate((0,len(seq)//2,len(seq)-1)):
        im=draw(seq[idx]);im.save(out/f'clip_{clip:02d}_{col}.png');sheet.paste(im.resize((640,360)),(640*col,360*row))
sheet.save(out/'review.jpg',quality=94)
if review:
    starts_sheet=Image.new('RGB',(2560,360*math.ceil(len(review)/4)),'white')
    for index,scene in enumerate(review):
        preview=draw(scene,'ACTUAL RESET START / NOT A POLICY ROLLOUT')
        preview.save(out/f'start_{index:02d}.png')
        starts_sheet.paste(preview.resize((640,360)),(640*(index%4),360*(index//4)))
    starts_sheet.save(out/'starts_review.jpg',quality=94)
video=out/'rsi_review.mp4'
proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','10','-i','-','-an','-c:v','libx264','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
count=0
try:
    # All five clips at original 30fps source time, sampled every third frame.
    for clip,seq in byclip.items():
        for scene in seq[::1 if data.get('sampled_resets') else 3]:proc.stdin.write(np.asarray(draw(scene)).tobytes());count+=1
        print('RENDER_CLIP',clip,flush=True)
    extreme=scenes[int(support.argmax())].copy();extreme['label']='maximum_lower_foot_sole'
    for scene in review+[extreme]:
        frame=np.asarray(draw(scene,'STATIC RSI / LOCO REVIEW CASE')).tobytes()
        for _ in range(10):proc.stdin.write(frame);count+=1
finally:
    proc.stdin.close();assert proc.wait()==0
(out/'video.json').write_text(json.dumps(dict(path=str(video),frames=count,fps=10,seconds=count/10,width=1280,height=720,mass=spec['mass'],size_y_m=spec['size_y_m'],physics_rollout=False),indent=2)+'\n')
print('VIDEO',video,'frames',count,flush=True)
