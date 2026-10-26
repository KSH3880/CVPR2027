"""CPU PUSH-only RSI candidate video; legacy task RSI is not inspected or changed."""
from isaacgym import gymapi
import json
from pathlib import Path
import subprocess
import importlib.util
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from utils.config import get_args
from utils.native_push_rsi import safe_place, digest, load_cpu_reference
from utils.open_push_rsi import align_push_pose
from env.tasks.multi_agent.push_contact import push_rsi_hand_targets
import yaml

args = get_args()
assert args.headless and args.device == 'cpu' and not args.use_gpu
torch.set_num_threads(1)
cfg = yaml.safe_load(Path(args.cfg_env).read_text())
env = cfg['env']
source = yaml.safe_load(Path(env['pushExtension']['source_env']).read_text())['env']
source['motion_file'] = 'tokenhsi/data/dataset_push_native_phase_xy.yaml'
lib,names,geo = load_cpu_reference(source)
contact = dict(push=env['pushExtension']['physical_push'],
    contact_at=env['pushExtension']['contact'],push_at=env['pushExtension']['push_at'])
out = Path(args.output_path); out.mkdir(parents=True,exist_ok=True)
bank = Path(env['openPushRSI']['bank_file']); bank_payload = json.loads(bank.read_text())
rows = bank_payload['candidates']['push']
assert len(rows) == 949
asset = Path(env['asset']['assetRoot'])/env['asset']['assetFileName']
preview = Path(__file__).resolve().parents[2]/'tokenhsi/utils/rsi_preview_render.py'
spec = importlib.util.spec_from_file_location('finetune_preview',preview)
render = importlib.util.module_from_spec(spec);spec.loader.exec_module(render)
renderer = render.PreviewRenderer(asset,names)
font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font = ImageFont.truetype(font_path,22); small = ImageFont.truetype(font_path,17)
video = out/'push_rsi_candidates.mp4'
proc = subprocess.Popen(['/usr/bin/ffmpeg','-y','-loglevel','error','-f','rawvideo',
    '-pix_fmt','rgb24','-s','1280x720','-r','30','-i','-','-an','-c:v','libx264',
    '-threads','2','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
frames = 0

def write(im, repeat=1):
    global frames
    for _ in range(repeat):
        proc.stdin.write(im.tobytes()); frames += 1


def card(title, subtitle):
    im = Image.new('RGB',(1280,720),(18,24,35)); d = ImageDraw.Draw(im)
    d.text((22,14),title,font=font,fill='#eeeeee')
    d.text((22,50),subtitle,font=small,fill='#f6c36e')
    d.text((22,78),'CPU reset poses / original DOF | NOT policy rollout | fixed goal after RSI',font=small,fill='#b7dce8')
    return im,d


try:
    # Continuous original source-frame order. Each frame is independently aligned
    # at its actual bank progress along a 10m segment to a fixed goal.
    profiles = cfg['env']['pushBox']['size_profiles']
    clips = sorted({r['motion_id'] for r in rows})
    for index,row in enumerate(rows):
        clip = clips.index(row['motion_id'])
        size = torch.tensor(profiles[round((index//30)*(len(profiles)-1)/max(1,(len(rows)-1)//30))])
        angle = (-45.,0.,45.)[(index//30)%3]
        theta = np.deg2rad(angle)
        direction = torch.tensor([np.sin(theta),np.cos(theta),0.],dtype=torch.float)
        goal = torch.tensor([0.,0.,float(size[2]/2)])
        source = goal-10*direction
        box = torch.zeros(13);box[6]=1.;box[:3]=source+row['box_progress']*(goal-source)
        targets = push_rsi_hand_targets(box[None,None],goal[None,None],size,contact)[:,0]
        body,rot,vel,ang = lib.get_motion_state_max(torch.tensor([row['motion_id']]),torch.tensor([row['time']]))
        body,rot,_,_,_ = align_push_pose(body,rot,vel,ang,targets,geo['hands'],.06)
        body,metrics = safe_place(body,rot,box[None],size[None],targets,geo,True)
        progress = 10*row['box_progress']; remaining = 10-progress
        im,d = card('ALL 949 PUSH RSI CANDIDATES | fixed-goal distance example',
            f'Clip {row["motion_id"]} | source {row["time"]:.2f}s | angle {angle:+.0f} deg | friction 0.4 / 20kg')
        d.text((22,111),f'Initial 10.00 m - RSI progress {progress:.2f} m = remaining {remaining:.2f} m',font=font,fill='#9de6a2')
        for view in range(2):
            im.paste(renderer.panel(body[0].numpy(),rot[0].numpy(),box.numpy(),size.tolist(),
                side=(view==0),width=640,height=450),(640*view,170))
        d.text((22,634),f'XYZ {size[0]:.2f} / {size[1]:.2f} / {size[2]:.2f} m | feet to floor {metrics["sole_height_m"][0,0]*100:.1f} / {metrics["sole_height_m"][0,1]*100:.1f} cm',font=small,fill='#bedeea')
        d.text((22,667),'10m is a distance example; actual goals are sampled inside environment XY +/-4.5m and +/-45 deg.',font=small,fill='#b6c5d4')
        if index == 0: im.save(out/'push_candidates.jpg')
        write(im)
        if index%200 == 0: print('CANDIDATE_FRAME',index,len(rows),flush=True)
finally:
    proc.stdin.close(); code = proc.wait()
if code: raise RuntimeError('Video encoding failed')
final_payload = json.loads(bank.read_text())
# A parallel CPU screen may add its report; pose/config changes invalidate video.
assert final_payload['contract'] == bank_payload['contract']
assert final_payload['candidates'] == bank_payload['candidates']
bank_hash = digest(bank)
assert frames == len(rows)
meta = dict(video=str(video.resolve()),fps=30,frames=frames,duration=frames/30,
    bank=str(bank),bank_sha256=bank_hash,candidates=len(rows),legacy_task_rsi='unchanged',
    limitations='PUSH-only independently aligned CPU source poses; not physical or policy rollout. 10m is a distance example; actual goals use the square/cone sampler.')
(out/'review.json').write_text(json.dumps(meta,indent=2)+'\n')
print('VIDEO_DONE',video,frames/30,flush=True)
