"""CPU-only source/placement intersection; never reuse the imported size cache."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ.pop('CUDA_MPS_PIPE_DIRECTORY',None)
os.environ.pop('CUDA_MPS_LOG_DIRECTORY',None)
from isaacgym import gymapi
import torch
import yaml
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from utils.motion_lib import MotionLib
from utils.native_push_rsi import geometry,phase_mask,trim_runs,safe_place,native_contract
from utils.open_push_rsi import align_push_pose
from env.tasks.multi_agent.open_push_reward import push_rsi_hand_targets

torch.set_num_threads(1)
env=yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/push_native_phase_xy.yaml').read_text())['env']
env['motion_file']='tokenhsi/data/dataset_push_native_phase_xy.yaml'
if Path(env['openPushRSI']['bank_file']).exists():raise FileExistsError('Preserve the existing bank; use a new configuration/output for rebuilding')
asset=Path(env['asset']['assetRoot'])/env['asset']['assetFileName']
bodies=ET.parse(asset).findall('.//worldbody//body');names=[b.get('name') for b in bodies]
dof_ids=[];offsets=[0];limits=[]
for i,b in enumerate(bodies):
    joints=b.findall('./joint')
    if joints:
        dof_ids.append(i);offsets.append(offsets[-1]+len(joints))
        limits.extend([[float(x)*torch.pi/180 for x in j.get('range').split()] for j in joints])
hands=[names.index('right_hand'),names.index('left_hand')]
lib=MotionLib(env['motion_file'],'push',dof_ids,offsets,hands+[names.index('right_foot'),names.index('left_foot')],'cpu')
geo=geometry(asset,names,'cpu');limits=torch.tensor(limits)
rows=[];report=[]
for mid in range(lib.num_motions()):
    n=int(lib._motion_num_frames[mid]);dt=float(lib._motion_dt[mid]);fps=round(1/dt)
    times=torch.arange(n)*dt;mids=torch.full((n,),mid,dtype=torch.long)
    states=lib.get_motion_state(mids,times);body,rot,vel,ang=lib.get_motion_state_max(mids,times)
    valid=phase_mask(body,rot,names,fps)
    dof=states[2];valid &= ((dof>=limits[:,0]-1e-4)&(dof<=limits[:,1]+1e-4)).all(-1)
    worst_gap=0.;max_z=0.
    # Both local axes and asymmetric rectangle faces are covered by all profiles.
    for dims in env['pushBox']['size_profiles']:
        for axis in (0,1):
            box=torch.zeros(n,1,13);box[...,6]=1;box[...,2]=dims[2]/2
            goals=box[...,:3].clone();goals[...,axis]+=4
            targets=push_rsi_hand_targets(box,goals,dims,env['relationReward'])[:,0]
            aligned,rotation,_,_,_=align_push_pose(body,rot,vel,ang,targets,hands,env['openPushRSI']['stand_off'])
            placed,metrics=safe_place(aligned,rotation,box[:,0],torch.tensor(dims).expand(n,3),targets,geo,True)
            good=(metrics['placed_hand_midpoint_z']<=1.001)&metrics['face_valid']&(metrics['hand_gap']>=.005).all(-1)&(metrics['hand_gap']<=.18).all(-1)
            if 'max_support_sole_height_m' in env['openPushRSI']:
                good &= metrics['sole_height_m'].amin(-1)<=env['openPushRSI']['max_support_sole_height_m']
            valid &= good
    if mid in env['openPushRSI'].get('excluded_motion_ids',[]):valid[:]=False
    selected=trim_runs(valid,fps);frames=selected.nonzero().flatten().tolist()
    runs=[]
    for f in frames:
        if not runs or f!=runs[-1][-1]+1:runs.append([])
        runs[-1].append(f)
    report.append(dict(motion_id=mid,frames=n,selected=len(frames),intervals_s=[[r[0]*dt,r[-1]*dt] for r in runs]))
    for f in frames:rows.append(dict(motion_id=mid,time=f*dt,phase=f/(n-1),angle=0.,box_progress=.85*f/(n-1)))
    print('PHASE',mid,len(frames),flush=True)
if not rows:raise RuntimeError('No phase candidates; refusing fallback')
payload=dict(contract=native_contract(env),screen='CPU_geometry_only',candidates={'push':rows},selection=report,
    limitations='Conservative collider covers, exact foot mesh floor minima; no simulator/PD or policy validation yet. Training blocked until PD screen.')
path=Path(env['openPushRSI']['bank_file']);path.write_text(json.dumps(payload,indent=2)+'\n')
print('SAVED',path,len(rows),flush=True)
