"""All retained RSI poses/profiles at both cone boundaries and straight ahead."""
from isaacgym import gymapi
from pathlib import Path
import json
import math
import torch
import yaml
from utils.native_push_rsi import load_cpu_reference, safe_place, digest
from utils.open_push_rsi import align_push_pose
from utils.door_reference import rotate
from env.tasks.multi_agent.push_contact import push_rsi_hand_targets

torch.set_num_threads(1)
env = yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/stage1_push_finetune_source.yaml').read_text())['env']
env['motion_file'] = 'tokenhsi/data/dataset_push_native_phase_xy.yaml'
bank = Path(env['openPushRSI']['bank_file'])
rows = json.loads(bank.read_text())['candidates']['push']
lib,names,geo = load_cpu_reference(env)
body,rot,vel,ang = lib.get_motion_state_max(torch.tensor([r['motion_id'] for r in rows]),torch.tensor([r['time'] for r in rows]))
report = dict(device='cpu',poses=len(rows),profiles=len(env['pushBox']['size_profiles']),angles_degrees=[-45,0,45],combinations=0,
    penetration=0,invalid_face=0,min_clearance_m=100.,bank_sha256=digest(bank))
for dims in env['pushBox']['size_profiles']:
    sizes = torch.tensor(dims).expand(len(rows),3)
    for degrees in report['angles_degrees']:
        box = torch.zeros(len(rows),1,13); box[...,6]=1.;box[...,2]=dims[2]/2
        goal = box[...,:3].clone()
        theta = math.radians(degrees)
        goal[:,:,0] += 4*math.sin(theta);goal[:,:,1] += 4*math.cos(theta)
        target = push_rsi_hand_targets(box,goal,sizes[:,None],env['relationReward'])[:,0]
        placed,q,_,_,_ = align_push_pose(body,rot,vel,ang,target,geo['hands'],.06)
        placed,metrics = safe_place(placed,q,box[:,0],sizes,target,geo,True)
        points = placed[:,geo['ids']]+rotate(q[:,geo['ids']],geo['offsets'].expand(len(rows),-1,-1))
        local = (points-box[:,0,None,:3]).abs()-sizes[:,None]/2
        clearance = local.clamp_min(0).norm(dim=-1)+local.amax(-1).clamp_max(0)-geo['radii']
        report['penetration'] += int((clearance.amin(-1)<-1e-5).sum())
        report['invalid_face'] += int((~metrics['face_valid']).sum())
        report['min_clearance_m'] = min(report['min_clearance_m'],float(clearance.min()))
        report['combinations'] += len(rows)
out = Path('../../output/stage1_base_push_finetune_stable_check');out.mkdir(parents=True,exist_ok=True)
(out/'geometry.json').write_text(json.dumps(report,indent=2)+'\n')
assert not report['penetration'] and not report['invalid_face'],report
print('FINETUNE_GEOMETRY_DONE',json.dumps(report),flush=True)
