"""Exhaustive CPU placed-collider audit at all 36 asset profiles, random directions."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ.pop('CUDA_MPS_PIPE_DIRECTORY',None);os.environ.pop('CUDA_MPS_LOG_DIRECTORY',None)
from isaacgym import gymapi
import torch,json,yaml,argparse
from pathlib import Path
from utils.native_push_rsi import load_cpu_reference,safe_place,digest
from utils.open_push_rsi import align_push_pose
from utils.door_reference import rotate
from env.tasks.multi_agent.open_push_reward import push_rsi_hand_targets

torch.set_num_threads(1);torch.manual_seed(20261020)
parser=argparse.ArgumentParser()
parser.add_argument('--cfg-env',default='tokenhsi/data/cfg/multi_agent/push_native_phase_xy.yaml')
args=parser.parse_args()
env=yaml.safe_load(Path(args.cfg_env).read_text())['env']
env['motion_file']='tokenhsi/data/dataset_push_native_phase_xy.yaml'
lib,names,geo=load_cpu_reference(env);bank=Path(env['openPushRSI']['bank_file'])
rows=json.loads(bank.read_text())['candidates']['push'];count=len(rows)
mids=torch.tensor([r['motion_id'] for r in rows]);times=torch.tensor([r['time'] for r in rows])
body,rot,vel,ang=lib.get_motion_state_max(mids,times)
profiles=len(env['pushBox']['size_profiles'])
summary=dict(device='cpu',simulator=False,poses=count,profiles=profiles,combinations=count*profiles,violations=dict(penetration=0,face_projection=0,height=0,initial_support=0),
    min_collider_clearance_m=100.,min_head_clearance_m=100.,max_hand_gap_m=0.,max_foot_sole_height_m=0.,max_floor_shift_m=0.,max_lower_foot_sole_height_m=0.,min_foot_sole_height_m=100.,extremes={})
head=names.index('head')
for dims in env['pushBox']['size_profiles']:
    sizes=torch.tensor(dims).expand(count,3)
    box=torch.zeros(count,1,13);box[...,6]=1;box[...,2]=.65
    yaw=torch.rand(count)*2*torch.pi;box[:,0,5]=torch.sin(yaw/2);box[:,0,6]=torch.cos(yaw/2)
    a=torch.rand(count)*2*torch.pi;goals=box[...,:3].clone();goals[:,0,:2]+=4*torch.stack((torch.cos(a),torch.sin(a)),-1)
    if env['pushScene']['random_at_goal'].get('sampling')=='box_forward':
        from utils.push_scene import forward_xy
        goals=box[...,:3].clone();goals[:,0,:2]+=4*forward_xy(box[:,0,3:7])
    targets=push_rsi_hand_targets(box,goals,sizes[:,None],env['relationReward'])[:,0]
    placed,rotation,_,_,_=align_push_pose(body,rot,vel,ang,targets,geo['hands'],.06)
    placed,metrics=safe_place(placed,rotation,box[:,0],sizes,targets,geo,True)
    points=placed[:,geo['ids']]+rotate(rotation[:,geo['ids']],geo['offsets'].expand(count,-1,-1))
    inv=box[:,0,3:7].clone();inv[:,:3]*=-1
    local=rotate(inv[:,None].expand(*points.shape[:-1],4),points-box[:,0,None,:3])
    q=local.abs()-sizes[:,None]/2
    clearance=q.clamp_min(0).norm(dim=-1)+q.amax(-1).clamp_max(0)-geo['radii']
    perpose=clearance.amin(-1)
    summary['violations']['penetration']+=int((perpose< -1e-5).sum())
    summary['violations']['face_projection']+=int((~metrics['face_valid']).sum())
    summary['violations']['height']+=int((metrics['placed_hand_midpoint_z']>1.001).sum())
    summary['min_collider_clearance_m']=min(summary['min_collider_clearance_m'],float(perpose.min()))
    summary['max_floor_shift_m']=max(summary['max_floor_shift_m'],float(metrics['z_shift'].max()))
    hc=clearance[:,geo['ids']==head].amin(-1)
    ground=placed[:,geo['ground_ids']]+rotate(rotation[:,geo['ground_ids']],geo['ground_offsets'].expand(count,-1,-1))
    # Mesh sole is the lowest actual vertex in each current foot, not ankle Z.
    sole=torch.stack([ground[:,geo['ground_ids']==names.index(f),2].amin(-1) for f in ('right_foot','left_foot')],-1)
    summary['violations']['initial_support']+=int((sole.amin(-1)>env['openPushRSI'].get('max_support_sole_height_m',float('inf'))+1e-6).sum())
    summary['max_lower_foot_sole_height_m']=max(summary['max_lower_foot_sole_height_m'],float(sole.amin(-1).max()))
    summary['min_foot_sole_height_m']=min(summary['min_foot_sole_height_m'],float(sole.min()))
    vals={'min_head_clearance_m':(hc,False),'max_hand_gap_m':(metrics['hand_gap'].amax(-1),True),'max_foot_sole_height_m':(sole.amax(-1),True)}
    for key,(value,maximize) in vals.items():
        index=int(value.argmax() if maximize else value.argmin());v=float(value[index])
        if (v>summary[key]) if maximize else (v<summary[key]):
            summary[key]=v;summary['extremes'][key]=dict(row=rows[index],size=dims,box=box[index,0].tolist(),goal=goals[index,0].tolist(),value=v)
summary['bank_sha256']=digest(bank)
summary['limitations']='CPU conservative sphere/capsule covers and exact foot OBJ floor vertices; not simulator snapshots or policy rollout. One seeded random yaw/direction per candidate/profile, plus builder canonical faces.'
out=Path('output_etc/check')/bank.stem[:-len('_rsi_bank')];out.mkdir(parents=True,exist_ok=True)
(out/'geometry.json').write_text(json.dumps(summary,indent=2)+'\n')
assert not any(summary['violations'].values()),summary['violations']
print('GEOMETRY_DONE',json.dumps({k:v for k,v in summary.items() if k!='extremes'}),flush=True)
