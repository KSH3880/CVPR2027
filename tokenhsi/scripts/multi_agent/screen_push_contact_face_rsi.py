"""Rescreen the dedicated fixed-box common-face RSI bank on CPU, without training."""
from isaacgym import gymapi
import json
import os
import hashlib
from pathlib import Path
from collections import Counter
import torch
from utils.config import get_args,load_cfg,parse_sim_params,set_seed
from utils.open_push_rsi import bank_contract,OpenPushRSIBank
from env.tasks.multi_agent.humanoid_ma_open_push import HumanoidMAOpenPush
from env.tasks.multi_agent.open_push_reward import push_rear_xy

args=get_args()
assert args.test and args.eval and args.num_envs==2 and args.num_objects==2
assert args.device=='cpu', 'This screen is CPU-only to protect active training'
cfg,train,_=load_cfg(args);cfg['args']=args;cfg['env']['motion_file']=args.motion_file
cfg['env']['enableTrackInitState']=True
assert cfg['env'].get('pushGoalContact') is True
set_seed(42);torch.set_num_threads(1)
sim=parse_sim_params(args,cfg,train);sim.physx.num_threads=1
task=HumanoidMAOpenPush(cfg,sim,args.physics_engine,args.device,args.device_id,True)
ids=torch.arange(2,device=task.device)
bank_path=Path(cfg['env']['openPushRSI']['bank_file'])
payload=json.loads(bank_path.read_text());accepted=[];rejected=[]
bank_hash=hashlib.sha256(bank_path.read_bytes()).hexdigest()
readonly=os.environ.get('TOKENHSI_RSI_SCREEN_READONLY')=='1'
try:
    task._is_eval=False;task._task_graph_preset='push'
    candidates=task._joint_rsi_bank.rows['push']
    for index,row in enumerate(candidates):
        task._reset_envs(ids)
        sample={k:torch.tensor([row[k],row[k]],device=task.device) for k in row}
        for agent in range(2):
            task._apply_joint_rsi(ids,torch.full_like(ids,agent),'push',sample)
        task._reset_task(ids);task._reset_env_tensors(ids)
        task._compute_observations(ids);task._init_amp_obs(ids)
        initial=task._humanoid_root_states[...,:3].clone()
        valid=True;support=torch.zeros(2,2,dtype=torch.bool);near=torch.zeros_like(support)
        for step in range(12):
            task.pre_physics_step(torch.zeros(4,32));task._physics_step();task.post_physics_step()
            valid &= bool(torch.isfinite(task.obs_buf).all() and torch.isfinite(task.rew_buf).all())
            valid &= bool((task._humanoid_root_states[...,2]>.65).all() and not task.reset_buf.any())
            valid &= bool(((task._humanoid_root_states[...,:3]-initial).norm(dim=-1)<.45).all())
            boxes=task._owned_push_states()
            valid &= bool((boxes[...,2]-cfg['env']['pushBox']['size'][2]/2).abs().max()<.08)
            if step>=2:
                feet=task._posture_body_ids[2:]
                support |= ((task._rigid_body_pos[:,:,feet,2]<.35) &
                            (task._contact_forces[:,:,feet,2]>=1.)).any(-1)
            hands=task._rigid_body_pos[:,:,task._key_body_ids[:2]]
            distance=push_rear_xy(hands,boxes,task._tar_pos,cfg['env']['pushBox']['size'])
            # Screen pose proximity, not actual contact or policy success.
            near |= (distance<.18).all(-1)
        if valid and support.all() and near.all():accepted.append(row)
        else:rejected.append(dict(index=index,valid=valid,support=support.tolist(),near=near.tolist()))
        if index%20==0:print('FACE_RSI_SCREEN',index,len(candidates),len(accepted),flush=True)
    assert set(Counter(r['motion_id'] for r in accepted))==set(Counter(r['motion_id'] for r in candidates))
    payload['candidates']['push']=accepted
    payload['contract']=bank_contract(cfg['env'])
    OpenPushRSIBank(cfg['env']['openPushRSI'],task._motion_lib,payload['contract'],payload['candidates'])
    payload['face_contact_reuse']=dict(source='push_toy_random_at_10deg_at_tilt_soft_rsi_bank.json',
        candidate_rescreened=True,screen_device='cpu',physics_steps=12,
        proposed=len(candidates),accepted=len(accepted),both_agents=True,
        notes='Random scene/yaw, same-face XY proximity <18cm and stable supported PD pose; not the contact state >0.9 force gate or policy success')
    if not readonly:
        bank_path.write_text(json.dumps(payload,indent=2)+'\n')
    else:
        assert bank_hash==hashlib.sha256(bank_path.read_bytes()).hexdigest()
    out=Path(args.output_path);out.mkdir(parents=True,exist_ok=True)
    report=dict(training=False,device='cpu',proposed=len(candidates),accepted=len(accepted),rejected=rejected,
        readonly=readonly,bank_sha256=bank_hash,config=args.cfg_env,
        notes='12-step PD pose support/proximity screen; not policy rollout or force-contact success')
    (out/'screen.json').write_text(json.dumps(report,indent=2)+'\n')
    print('FACE_RSI_SCREEN_DONE',len(accepted),'/',len(candidates),flush=True)
finally:
    task.gym.destroy_sim(task.sim)
