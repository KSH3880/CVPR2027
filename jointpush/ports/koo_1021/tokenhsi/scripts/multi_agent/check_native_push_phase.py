"""CPU-only exhaustive 0.1s initial PD screen, plus a 0.4s frozen-pose audit."""
from isaacgym import gymapi
import json
import hashlib
from pathlib import Path
import torch
from utils.config import get_args,load_cfg,parse_sim_params,set_seed
from utils.native_push_rsi import digest
from env.tasks.multi_agent.humanoid_ma_open_push import HumanoidMAOpenPush

args=get_args()
assert args.device=='cpu' and args.test and args.eval
cfg,train,_=load_cfg(args);cfg['args']=args;env=cfg['env'];env['motion_file']=args.motion_file
nprofiles=len(env['pushBox']['size_profiles'])
assert args.num_envs==nprofiles
nagents=2*nprofiles
cfg['_native_all_profiles']=True;env['_native_geometry_check']=True
env['enableTrackInitState']=True
set_seed(20261020);torch.set_num_threads(1)
sim=parse_sim_params(args,cfg,train);sim.physx.num_threads=1
bank=Path(env['openPushRSI']['bank_file']);before=digest(bank)
payload=json.loads(bank.read_text())
task=HumanoidMAOpenPush(cfg,sim,args.physics_engine,args.device,args.device_id,True)
ids=torch.arange(task.num_envs,device=task.device);records=[];failed=[]
try:
    task._is_eval=False;task._task_graph_preset='push'
    # Sample each retained continuous clip at start/middle/end, all asset sizes.
    rows=payload['candidates']['push'];representatives=[]
    for mid in sorted(set(r['motion_id'] for r in rows)):
        clip=[r for r in rows if r['motion_id']==mid]
        representatives.extend([clip[0],clip[len(clip)//2],clip[-1]])
    for index,row in enumerate(representatives):
        task._reset_envs(ids)
        sample={k:torch.full((nprofiles,),v,device=task.device) for k,v in row.items()}
        for a in range(2):task._apply_joint_rsi(ids,torch.full_like(ids,a),'push',sample)
        task._reset_task(ids);task._reset_env_tensors(ids);task._compute_observations(ids);task._init_amp_obs(ids)
        initial=task._humanoid_root_states[...,:3].clone()
        dof=task._dof_pos.clone();dofvel=task._dof_vel.clone()
        lib=task._motion_lib['push'];reference=lib.get_motion_state(sample['motion_id'].long(),sample['time'])
        assert torch.allclose(dof[:,0],reference[2]) and torch.allclose(dof[:,1],reference[2])
        assert torch.allclose(dofvel[:,0],reference[5]) and torch.allclose(dofvel[:,1],reference[5])
        support=torch.zeros(nprofiles,2,dtype=torch.bool);valid=torch.ones(nprofiles,2,dtype=torch.bool)
        for step in range(12):
            task.pre_physics_step(torch.zeros(nagents,32));task._physics_step();task.post_physics_step()
            assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task.rew_buf).all()
            valid &= (task._humanoid_root_states[...,2]>.65)&((task._humanoid_root_states[...,:3]-initial).norm(dim=-1)<.45)
            valid &= ~task.reset_buf[:,None].bool()
            if step>=2:
                feet=task._posture_body_ids[2:]
                support |= ((task._rigid_body_pos[:,:,feet,2]<.35)&(task._contact_forces[:,:,feet,2]>=1.)).any(-1)
        ok=valid&support
        records.append(dict(motion_id=row['motion_id'],time=row['time'],passed=int(ok.sum()),total=nagents))
        if not ok.all():failed.append(dict(row=row,failed=(~ok).nonzero().tolist(),stable=int(valid.sum()),supported=int(support.sum()),root_z=task._humanoid_root_states[...,2].tolist()))
        print('NATIVE_PD',index,len(representatives),int(ok.sum()),nagents,flush=True)
    # The 0.4s frozen-pose audit above is retained, including failures. Native
    # stepping clips need a separate 0.1s initial-state screen (guideline scope).
    short_records=[];accepted=[]
    for index,row in enumerate(rows):
        task._reset_envs(ids)
        sample={k:torch.full((nprofiles,),v,device=task.device) for k,v in row.items()}
        for a in range(2):task._apply_joint_rsi(ids,torch.full_like(ids,a),'push',sample)
        task._reset_task(ids);task._reset_env_tensors(ids);task._compute_observations(ids);task._init_amp_obs(ids)
        initial=task._humanoid_root_states[...,:3].clone();initial_boxes=task._owned_push_states().clone()
        stable=torch.ones(nprofiles,2,dtype=torch.bool)
        for step in range(3):
            task.pre_physics_step(torch.zeros(nagents,32));task._physics_step();task.post_physics_step()
            assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task.rew_buf).all()
            stable &= (task._humanoid_root_states[...,2]>.65)&((task._humanoid_root_states[...,:3]-initial).norm(dim=-1)<.25)
            stable &= ~task.reset_buf[:,None].bool()
            stable &= (task._owned_push_states()[...,:3]-initial_boxes[...,:3]).norm(dim=-1)<.08
        feet=task._posture_body_ids[2:]
        if 'max_support_sole_height_m' in env['openPushRSI']:
            stable &= (task._contact_forces[:,:,feet,2]>=1.).any(-1)
        if stable.all():accepted.append(row)
        else:short_records.append(dict(motion_id=row['motion_id'],time=row['time'],passed=int(stable.sum())))
        if index%100==0:print('SHORT_PD',index,len(rows),len(accepted),flush=True)
    # Partial reset must not move any other owner's scene.
    task._reset_envs(ids);untouched=task._box_states[1:].clone();task._reset_envs(ids[:1]);assert torch.equal(untouched,task._box_states[1:])
    assert before==digest(bank)
    report=dict(device='cpu',profiles=nprofiles,physics_steps=12,representatives=records,failed=failed,
        limitations='Representative initial-PD stability/support screen, not every candidate physics, policy success, long balance, or rendered video.')
    out=Path(args.output_path);out.mkdir(parents=True,exist_ok=True);(out/'screen.json').write_text(json.dumps(report,indent=2)+'\n')
    report['short_screen']=dict(seconds=3*task.dt,control_steps=3,proposed=len(rows),accepted=len(accepted),rejected=short_records,all_profiles=True,both_agents=True)
    (out/'screen.json').write_text(json.dumps(report,indent=2)+'\n')
    if len(accepted)!=len(rows):raise RuntimeError('Initial PD validation failed; preserve the requested pool and keep the bank blocked')
    # A repeat validation must not invalidate prepared checkpoints/video hashes.
    report['bank_preserved']=payload.get('screen')=='PD_0.1s_all_profiles'
    if not report['bank_preserved']:
        payload['candidates']['push']=accepted
        payload['screen']='PD_0.1s_all_profiles';payload['physics_screen']=report
        payload['limitations']='All candidates/sizes: 0.1s initial PD only. Some 0.4s frozen-pose representatives lose height. No policy success/long balance/GPU training validation.'
        bank.write_text(json.dumps(payload,indent=2)+'\n')
    report['contract']=task.get_open_push_contract()
    report['bank_sha256']=digest(bank)
    report['partial_reset']=True
    report['original_dof_preserved']=True
    (out/'screen.json').write_text(json.dumps(report,indent=2)+'\n')
    print('NATIVE_SCREEN_DONE',flush=True)
finally:
    task.gym.destroy_sim(task.sim)
