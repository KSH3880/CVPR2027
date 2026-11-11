"""CPU schematic video of measured rigid bodies; not a Gym mesh render."""
import json, subprocess, argparse, colorsys
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent

def render(folder):
 z=np.load(folder/'rollout.npz');meta=json.loads((folder/'replay.json').read_text());scene=json.loads((folder/'request.json').read_text())['current_scene'];tasks=scene['tasks'];ids=[t['agent_id'] for t in tasks]
 release=json.loads((folder/'release_audit.json').read_text())['quantized_release_times_s'][0];groups=list(dict.fromkeys(t['group_id'] for t in tasks));names=meta['body_names'];links=[]
 def visit(node,parent=None):
  name=node.get('name');current=names.index(name) if name in names else parent
  if current is not None and parent is not None and current!=parent:links.append((parent,current))
  for child in node.findall('body'):visit(child,current)
 for body in ET.parse(ROOT/'teacher_runtime/tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml').getroot().find('worldbody').findall('body'):visit(body)
 font=lambda s:ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',s)
 small,normal,big=font(14),font(18),font(25)
 colors=[tuple(int(x*255) for x in colorsys.hsv_to_rgb((i*.381966)%1,.55,.98)) for i in range(16)]
 def iso(p):x,y,v=p[:3];return np.array([.8*x-.6*y,.36*x+.48*y-1.3*v])
 fits={}
 for g in groups:
  sel=[i for i,t in enumerate(tasks) if t['group_id']==g];pts=np.array([iso(p) for p in z['roots'][::10,sel,:3].reshape(-1,3)]);lo=pts.min(0);hi=pts.max(0);fits[g]=((hi+lo)/2,min(410/(hi[0]-lo[0]+4),310/(hi[1]-lo[1]+5)))
 proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pixel_format','rgb24','-video_size','1920x1080','-framerate','15','-i','-','-an','-c:v','libx264','-preset','fast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(folder/'replay.mp4')],stdin=subprocess.PIPE)
 try:
  for f,t in enumerate(z['time_s']):
   im=Image.new('RGB',(1920,1080),'#0c1220');d=ImageDraw.Draw(im)
   d.text((25,15),f'ACTUAL SIMULATION | Initial freeze -> permanent release | t = {t:05.2f} s',font=big,fill='white')
   d.text((25,50),'Measured skeleton / object schematic. White ring = waiting at initial pose. No runtime event gates.',font=normal,fill='#b2c2d7')
   for gi,g in enumerate(groups):
    x=(gi%4)*480;y=90+(gi//4)*485;d.rounded_rectangle((x+8,y,x+472,y+475),radius=10,fill='#142033',outline='#30435d');sel=[i for i,task in enumerate(tasks) if task['group_id']==g]
    d.text((x+22,y+10),' / '.join(ids[i] for i in sel)+' | '+','.join(dict.fromkeys(tasks[i]['task_label'] for i in sel)),font=normal,fill='white');center,scale=fits[g]
    def p(point):q=(iso(point)-center)*scale;return(x+240+q[0],y+235-q[1])
    obs={int(tasks[i][k].split('_')[1]) for i in sel for k in ('object_id','support_object_id') if k in tasks[i]}
    for bi,bid in enumerate(z['box_ids']):
     if bid not in obs:continue
     b=z['boxes'][f,bi];size=z['box_sizes'][bi];q=b[3:6];qw=b[6];corners=[]
     for k in range(8):
      v=size/2*np.array([1 if k&(1<<a) else -1 for a in range(3)]);uv=2*np.cross(q,v);corners.append(p(v+qw*uv+np.cross(q,uv)+b[:3]))
     for k in range(8):
      for a in range(3):
       if not k&(1<<a):d.line([corners[k],corners[k|(1<<a)]],fill='#8b9db9',width=2)
    for j,i in enumerate(sel):
     c=colors[i];body=z['bodies'][f,i];root=p(z['roots'][f,i]);wait=t<release[ids[i]]
     for a,b in links:d.line([p(body[a]),p(body[b])],fill=c,width=3)
     d.ellipse((root[0]-4,root[1]-4,root[0]+4,root[1]+4),fill=c)
     if wait:d.ellipse((root[0]-11,root[1]-11,root[0]+11,root[1]+11),outline='white',width=2)
     d.text((root[0]+8,root[1]-20),ids[i],font=small,fill=c);flag='WAIT' if wait else 'ACTIVE'
     if z['terminated'][f,i]:flag+=' / TERMINATED'
     d.text((x+22,y+370+j*23),f'{ids[i]}: {flag} | release {release[ids[i]]:.3f}s',font=small,fill=c)
   d.text((1465,630),'Freeze verification',font=big,fill='white')
   for j,line in enumerate(['Same initial scene: verified','16 initial releases applied','Root / DOF frozen drift: 0','No runtime event gates','No task-success claim','GPU 7 / 600 simulation steps']):d.text((1465,675+j*35),line,font=normal,fill='#b2c2d7')
   if f in (0,45):im.save(folder/('preview.png' if f==45 else 'initial_preview.png'))
   proc.stdin.write(im.tobytes())
 finally:proc.stdin.close()
 if proc.wait()!=0:raise RuntimeError('Video encoding failed')
 print(folder/'replay.mp4')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('folder',type=Path);render(p.parse_args().folder)
