"""CPU projection of continuous reset-pose sequences, actual foot OBJ meshes."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ.pop('CUDA_MPS_PIPE_DIRECTORY',None);os.environ.pop('CUDA_MPS_LOG_DIRECTORY',None)
from isaacgym import gymapi
import torch
from pathlib import Path
import argparse
import json,yaml,subprocess,xml.etree.ElementTree as ET
from PIL import Image,ImageDraw,ImageFont
from utils.rsi_preview_render import PreviewRenderer
from utils.motion_lib import MotionLib
from utils.native_push_rsi import geometry,safe_place,digest
from utils.open_push_rsi import align_push_pose
from utils.door_reference import rotate
from env.tasks.multi_agent.open_push_reward import push_rsi_hand_targets

parser=argparse.ArgumentParser()
parser.add_argument('--cfg-env',type=Path,default=Path('tokenhsi/data/cfg/multi_agent/push_native_phase_xy.yaml'))
parser.add_argument('--bank',type=Path)
parser.add_argument('--output',type=Path)
parser.add_argument('--candidates-only',action='store_true')
parser.add_argument('--fps',type=int,default=15)
parser.add_argument('--repeat',type=int,default=2)
parser.add_argument('--expected-candidates',type=int)
args=parser.parse_args()
if args.fps<=0 or args.repeat<=0:parser.error('fps and repeat must be positive')
torch.set_num_threads(1)
env=yaml.safe_load(args.cfg_env.read_text())['env']
default_bank=args.bank or Path(env['openPushRSI']['bank_file'])
output_name=default_bank.stem[:-len('_rsi_bank')] if default_bank.stem.endswith('_rsi_bank') else default_bank.stem
out=args.output.parent if args.output else Path('output_etc/check')/output_name
out.mkdir(parents=True,exist_ok=True)
asset=Path(env['asset']['assetRoot'])/env['asset']['assetFileName'];tree=ET.parse(asset)
bodies=tree.findall('.//worldbody//body');names=[b.get('name') for b in bodies]
dofs=[];offsets=[0]
for i,b in enumerate(bodies):
    joints=b.findall('./joint')
    if joints:dofs.append(i);offsets.append(offsets[-1]+len(joints))
hands=[names.index('right_hand'),names.index('left_hand')]
lib=MotionLib('tokenhsi/data/dataset_push_native_phase_xy.yaml','push',dofs,offsets,hands+[11,14],'cpu')
geo=geometry(asset,names,'cpu')
bank=args.bank or Path(env['openPushRSI']['bank_file']);payload=json.loads(bank.read_text());rows=payload['candidates']['push']
if args.expected_candidates is not None:assert len(rows)==args.expected_candidates
source_bank_hash=digest(bank)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',21)
small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
W,H=1280,720;fps=args.fps
video=args.output or out/'rsi_preview_v2.mp4'
renderer=PreviewRenderer(asset,names)
proc=subprocess.Popen(['/usr/bin/ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(fps),'-i','-','-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p',str(video)],stdin=subprocess.PIPE)
records=[];nframes=0

def frame_image(body,rot,box,dims,label,metrics):
    im=Image.new('RGB',(W,H),(18,24,35));d=ImageDraw.Draw(im)
    d.text((24,14),'PUSH RSI | ORIGINAL JOINTS | CPU RESET-POSE SEQUENCE',font=font,fill='#eeeeee')
    d.text((24,47),f'Not physics rollout. Source speed {fps/(30*args.repeat):g}x; each frame is independently aligned.',font=small,fill='#f9bf69')
    d.text((24,81),label,font=font,fill='#f5f5f5')
    d.text((24,112),f"Box XYZ {dims[0]:.1f} / {dims[1]:.1f} / 1.3 m | hand midpoint {float(metrics['placed_hand_midpoint_z'][0]):.3f} m | floor Z shift {float(metrics['z_shift'][0]):.3f} m",font=small,fill='#a5d8e8')
    d.text((24,140),f"Hand-plane gaps {float(metrics['hand_gap'][0,0]):.3f} / {float(metrics['hand_gap'][0,1]):.3f} m | original DOF position / velocity preserved",font=small,fill='#a5d8e8')
    soles=metrics['sole_height_m'][0]*100
    d.text((24,167),f'Sole-to-ground: right {float(soles[0]):.2f} cm / left {float(soles[1]):.2f} cm',font=small,fill='#9de6a2')
    for view in range(2):
        panel=renderer.panel(body.numpy(),rot.numpy(),box.numpy(),dims,side=(view==0))
        im.paste(panel,(view*640,210))
    return im

try:
    rechecks={}
    for clip_index,mid in enumerate(sorted(set(r['motion_id'] for r in rows))):
        clip=[r for r in rows if r['motion_id']==mid]
        profiles=env['pushBox']['size_profiles']
        profile_index=([0,5,30,35,17][clip_index] if len(profiles)==36 else
            round(clip_index*(len(profiles)-1)/max(1,len(set(r['motion_id'] for r in rows))-1)))
        dims=profiles[profile_index]
        # Export every accepted source frame; fps/repeat control playback speed.
        chosen=clip
        for row in chosen:
            mids=torch.tensor([mid]);times=torch.tensor([row['time']])
            body,rot,vel,ang=lib.get_motion_state_max(mids,times)
            box=torch.zeros(1,1,13);box[...,6]=1;box[...,2]=.65;goal=box[...,:3].clone();goal[...,1]+=4
            target=push_rsi_hand_targets(box,goal,dims,env['relationReward'])[:,0]
            body,rot,_,_,_=align_push_pose(body,rot,vel,ang,target,hands,.06)
            body,metrics=safe_place(body,rot,box[:,0],torch.tensor(dims).expand(1,3),target,geo,True)
            im=frame_image(body[0],rot[0],box[0,0],dims,f"Selected clip {mid:02d} | source t={row['time']:.2f}s | {clip_index+1}/{len(set(r['motion_id'] for r in rows))}",metrics)
            for _ in range(args.repeat):proc.stdin.write(im.tobytes());nframes+=1
            if nframes==args.repeat:im.save(video.with_suffix('.jpg'))
        records.append(dict(motion_id=mid,source_start=clip[0]['time'],source_end=clip[-1]['time'],size=dims,export_frames=args.repeat*len(chosen)))
        print('RENDERED',mid,len(chosen),flush=True)
    audit_path=out/'geometry.json'
    if not args.candidates_only and audit_path.exists():
        extremes=json.loads(audit_path.read_text())['extremes']
        for key,extreme in extremes.items():
            row=extreme['row'];dims=extreme['size']
            body,rot,vel,ang=lib.get_motion_state_max(torch.tensor([row['motion_id']]),torch.tensor([row['time']]))
            box=torch.tensor(extreme['box']).reshape(1,1,13);goal=torch.tensor(extreme['goal']).reshape(1,1,3)
            target=push_rsi_hand_targets(box,goal,dims,env['relationReward'])[:,0]
            body,rot,_,_,_=align_push_pose(body,rot,vel,ang,target,hands,.06)
            body,metrics=safe_place(body,rot,box[:,0],torch.tensor(dims).expand(1,3),target,geo,True)
            rechecks[key]=dict(source_motion_id=row['motion_id'],source_time=row['time'],sole_height_m=metrics['sole_height_m'][0].tolist(),box=dims)
            if key=='min_head_clearance_m':
                frame_image(body[0],rot[0],box[0,0],dims,'Reported frame: clip 8 / source t=4.700s',metrics).save(out/'reported_frame_v2.jpg')
            label=f"Extreme: {key.replace('_m','')} = {extreme['value']:.3f}m | clip {row['motion_id']}"
            im=frame_image(body[0],rot[0],box[0,0],dims,label,metrics)
            for _ in range(2*fps):proc.stdin.write(im.tobytes());nframes+=1
finally:
    proc.stdin.close();code=proc.wait()
if code:raise RuntimeError('ffmpeg failed')
if digest(bank)!=source_bank_hash:raise RuntimeError('Source bank changed during render')
if args.candidates_only:assert nframes==len(rows)*args.repeat
if not args.candidates_only:
    (out/'rechecked_frames.json').write_text(json.dumps(rechecks,indent=2)+'\n')
metadata=video.with_suffix('.json') if args.output else out/'video_v2.json'
metadata.write_text(json.dumps(dict(video=str(video),fps=fps,frames=nframes,duration=nframes/fps,
    bank=str(bank),bank_sha256=source_bank_hash,unique_candidates=len(rows),candidates_only=args.candidates_only,clips=records,
    limitations='CPU reset-pose sequence, independently aligned each frame; not physical rollout. MJCF collision surfaces, actual foot OBJ, opaque box with depth.'),indent=2)+'\n')
print('VIDEO_DONE',video,nframes/fps,flush=True)
