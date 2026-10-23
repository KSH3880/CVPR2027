"""Filter an immutable PD-screened donor by reference and first-step hand height."""
from isaacgym import gymapi
import copy,hashlib,json,os,subprocess
from pathlib import Path
import torch
from utils.config import get_args,load_cfg,parse_sim_params,set_seed
from utils.open_push_rsi import bank_contract
from utils.push_point import point_geometry
from env.tasks.multi_agent.humanoid_ma_open_push_point import HumanoidMAOpenPushPoint

torch.set_num_threads(1)
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
uuid=subprocess.check_output(['nvidia-smi','-i','6','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
assert os.environ.get('TOKENHSI_GPU') in ('6',uuid)
args=get_args();assert args.test and args.eval and args.headless and args.num_envs==64
cfg,train,_=load_cfg(args);cfg['args']=args;cfg['env']['motion_file']=args.motion_file
cfg['env']['enableTrackInitState']=True;cfg['_rsi_build']=True
source=Path('tokenhsi/data/push_fixed130_point_goal3to5_bank.json')
source_hash=hashlib.sha256(source.read_bytes()).hexdigest();donor=json.loads(source.read_text())
contract=bank_contract(cfg['env']);assert donor['contract']==contract
target=Path(cfg['env']['openPushRSI']['bank_file']);assert target!=source
lo,hi=cfg['env']['pushPointReward']['height_band_m']
if target.exists():
    saved=json.loads(target.read_text())
    assert saved['contract']==contract and saved['height_alignment_screen']['source_sha256']==source_hash
    assert saved['height_alignment_screen']['height_band_m']==[lo,hi]
    print('Reusing height-matched bank:',len(saved['candidates']['push']),flush=True)
    raise SystemExit(0)
sim=parse_sim_params(args,cfg,train);sim.physx.num_threads=1;set_seed(42)
task=HumanoidMAOpenPushPoint(cfg,sim,args.physics_engine,args.device,args.device_id,True)
rows=donor['candidates']['push'];ids=torch.arange(64,device=task.device);records=[];accepted=[]
def pairs(x):return x.reshape(32,2,-1).all(-1).all(-1)
try:
    for begin in range(0,len(rows),32):
        batch=rows[begin:begin+32];padded=batch+[batch[-1]]*(32-len(batch))
        task._reset_envs(ids)
        sample={k:torch.tensor([r.get(k,0.) for r in padded],device=task.device).repeat_interleave(2)
                for k in ('motion_id','time','phase','box_progress')}
        task._apply_joint_rsi(ids,torch.zeros_like(ids),'push',sample)
        task._reset_task(ids);task._reset_env_tensors(ids);task._sync_door_reset_states(ids)
        task._compute_observations(ids);task._init_amp_obs(ids)
        ref=task._kinematic_humanoid_rigid_body_states[:,0,task._key_body_ids[:2],2].mean(-1)
        valid=pairs((ref>=lo)&(ref<=hi));root=task._humanoid_root_states[:,0,:3].clone()
        support=torch.zeros_like(valid);near=torch.zeros_like(valid);first=None;last=None
        for step in range(12):
            task.pre_physics_step(torch.zeros(128,task.num_actions,device=task.device))
            task._physics_step();task.post_physics_step()
            hands=task._rigid_body_pos[:,:,task._key_body_ids[:2]]
            z=hands[:,0,:,2].mean(-1)
            if step==0:
                first=z.clone();valid &= pairs((z>=lo)&(z<=hi))
            last=z.clone()
            valid &= pairs(torch.isfinite(task.obs_buf)) & pairs(torch.isfinite(task.rew_buf))
            valid &= pairs(task.reset_buf==0)&pairs(task._humanoid_root_states[:,0,2]>.65)
            valid &= pairs((task._humanoid_root_states[:,0,:3]-root).norm(dim=-1)<.45)
            valid &= pairs((task._owner_push_boxes()[:,0,2]-.65).abs()<.08)
            if step>=2:
                feet=task._key_body_ids[2:]
                supported=(task._rigid_body_pos[:,0,feet,2]<.35)&(task._contact_forces[:,0,feet,2]>=1.)
                support |= pairs(supported.any(-1))
            _,_,distance,_=point_geometry(hands,task._owner_push_boxes(),task._owner_push_sizes(),
                task.relation_runtime.face_side,task.relation_runtime.point_local)
            near |= pairs(distance[:,0].amax(-1)<.18)
        valid &= support&near
        for i,row in enumerate(batch):
            passed=bool(valid[i]);sl=slice(i*2,i*2+2)
            records.append(dict(index=begin+i,motion_id=row['motion_id'],time=row['time'],accepted=passed,
                reference_z=ref[sl].tolist(),first_step_z=first[sl].tolist(),last_step_z=last[sl].tolist()))
            if passed:accepted.append(row)
        print('HEIGHT_SCREEN',begin+len(batch),len(rows),'accepted',len(accepted),flush=True)
    assert accepted,'No stable height-matched RSI candidates'
    payload=copy.deepcopy(donor);payload['candidates']['push']=accepted
    payload['height_alignment_screen']=dict(source_bank=str(source),source_sha256=source_hash,
        height_band_m=[lo,hi],source_candidates=len(rows),accepted_candidates=len(accepted),
        motion_ids=sorted(set(r['motion_id'] for r in accepted)),scenes_per_candidate=2,control_steps=12,
        criteria='Reference and first simulated step midpoint in band in both scenes; finite/no reset/root>0.65m/drift<0.45m/boxheight within0.08m; foot support and hand-face proximity observed.',
        limitations='Initial height alignment only; subsequent zero-action PD steps need not stay in the band. Fewer RSI clips; all21 AMP experts unchanged.')
    assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash
    target.write_text(json.dumps(payload,indent=2)+'\n')
    out=Path(args.output_path);out.mkdir(parents=True,exist_ok=True)
    (out/'height_screen.json').write_text(json.dumps(dict(metadata=payload['height_alignment_screen'],candidates=records),indent=2)+'\n')
    print(json.dumps(payload['height_alignment_screen'],indent=2),flush=True)
finally:task.gym.destroy_sim(task.sim)
