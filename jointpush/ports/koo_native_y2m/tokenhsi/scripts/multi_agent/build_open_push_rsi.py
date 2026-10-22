"""Offline non-training PD screen and reset/AMP audit of joint RSI proxies."""
from isaacgym import gymapi
import os
import json
from pathlib import Path
import torch
from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from utils.open_push_rsi import bank_contract, OpenPushRSIBank
from env.tasks.multi_agent.humanoid_ma_open_push import HumanoidMAOpenPush
from env.tasks.multi_agent.open_push_reward import push_hand_point_distance

torch.set_num_threads(1)
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
args=get_args()
assert args.test and args.eval and args.num_envs==2
cfg,train,_=load_cfg(args);cfg['args']=args;cfg['_rsi_build']=os.environ.get('RSI_REUSE_BANK')!='1'
cfg['env']['motion_file']=args.motion_file;cfg['env']['enableTrackInitState']=True
sim=parse_sim_params(args,cfg,train);sim.physx.num_threads=1;set_seed(42)
task=HumanoidMAOpenPush(cfg,sim,args.physics_engine,args.device,args.device_id,True)
ids=torch.arange(2,device=task.device)
out=Path(args.output_path);out.mkdir(parents=True,exist_ok=True)
screen_skills=os.environ.get('RSI_SCREEN_SKILLS','doorOpen,push').split(',')
assert screen_skills and len(set(screen_skills))==len(screen_skills) and set(screen_skills)<= {'doorOpen','push'}
accepted=dict(task._joint_rsi_bank.rows) if not cfg['_rsi_build'] else {};counts={};rejected=[]
if cfg['_rsi_build'] and set(screen_skills)!={'doorOpen','push'}:
    # Preserve the already screened, contract-matching unused task. This does
    # not claim it has been screened on the current physics device.
    previous=json.loads(Path(cfg['env']['openPushRSI']['bank_file']).read_text())
    assert previous['contract']==bank_contract(cfg['env']) and previous['screen']=='PD_12_steps'
    accepted={k:v for k,v in previous['candidates'].items() if k not in screen_skills}
try:
    for skill in (screen_skills if cfg['_rsi_build'] else ()):
        proposals=task._joint_rsi_bank.rows[skill];accepted[skill]=[]
        task._task_graph_preset='open' if skill=='doorOpen' else 'push'
        for index,row in enumerate(proposals):
            task._reset_envs(ids)
            sample={k:torch.tensor([row[k],row[k]],device=task.device) for k in row}
            task._apply_joint_rsi(ids,torch.zeros_like(ids),skill,sample)
            task._reset_task(ids);task._reset_env_tensors(ids);task._sync_door_reset_states(ids)
            if task._manipulation_spec is not None:
                task.relation_runtime.initialize_manipulation(ids,task._box_states[:,2:],
                    task._tar_pos,task.door_hinge_angle)
            task._compute_observations(ids);task._init_amp_obs(ids)
            assert not task.relation_runtime.done.any() and not task.relation_runtime.elapsed.any()
            assert not task.relation_runtime.push_eligible.any()
            initial=task._humanoid_root_states[:,0,:3].clone()
            valid=True;support=False;near=False
            for step in range(12):
                task.pre_physics_step(torch.zeros(4,32,device=task.device));task._physics_step();task.post_physics_step()
                valid &= bool(torch.isfinite(task.obs_buf).all() and torch.isfinite(task.rew_buf).all())
                valid &= bool((task._humanoid_root_states[:,0,2]>.65).all() and not task.reset_buf.any())
                valid &= bool(((task._humanoid_root_states[:,0,:3]-initial).norm(dim=-1)<.45).all())
                if step>=2:
                    if skill=='doorOpen':
                        support |= bool(task.relation_runtime.posture[:,0].all())
                    else:
                        feet=task._posture_body_ids[2:]
                        low=task._rigid_body_pos[:,0,feet,2]<.35
                        force=task._contact_forces[:,0,feet,2]>=1.
                        support |= bool((low & force).any(-1).all())
                hands=task._rigid_body_pos[:,:,task._key_body_ids[:2]]
                if skill=='doorOpen':
                    distance=(hands-task.door_handle_position[:,:,None]).norm(dim=-1).amin(-1)[:,0]
                    near |= bool((distance<.2).all())
                else:
                    distance=push_hand_point_distance(hands,task._box_states[:,2:],cfg['env']['pushBox']['size'],cfg['env']['relationReward']['push'])[:,0]
                    near |= bool((distance<cfg['env']['relationReward']['push']['hand_distance']).all())
                    boxes=task._box_states[:,2]
                    valid &= bool(((boxes[:,2]-cfg['env']['pushBox']['size'][2]/2).abs()<cfg['env']['relationReward']['push']['ground_tolerance']).all())
            if valid and support and near:
                accepted[skill].append(row)
            else:
                rejected.append(dict(skill=skill,index=index,valid=valid,support=support,near=near,**row))
            if index%20==0:print('RSI_SCREEN',skill,index,len(proposals),'accepted',len(accepted[skill]),flush=True)
        counts[skill]={'proposed':len(proposals),'accepted':len(accepted[skill])}
    contract=bank_contract(cfg['env'])
    (out/'screen_counts.json').write_text(json.dumps(counts,indent=2)+'\n')
    (out/'rejected.json').write_text(json.dumps(rejected,indent=2)+'\n')
    (out/'accepted.json').write_text(json.dumps(accepted,indent=2)+'\n')
    # Require each clip to have survivors before producing a runnable bank.
    bank=OpenPushRSIBank(cfg['env']['openPushRSI'],task._motion_lib,contract,accepted)
    payload=dict(contract=contract,screen='PD_12_steps',candidates=accepted,
        screened_skills=screen_skills if cfg['_rsi_build'] else [],
        screen_device=task.device if cfg['_rsi_build'] else None,
        notes='Object trajectories are geometric proxies, not measured BONES states. PD screen verifies short pose stability/foot support/hand proximity, not policy success or sustained manipulation.')
    bank_path=Path(cfg['env']['openPushRSI']['bank_file'])
    if cfg['_rsi_build']:
        bank_path.write_text(json.dumps(payload,indent=2)+'\n')
    task._joint_rsi_bank=bank;task._is_eval=False
    task._skill=list(cfg['env']['skill'])
    task._skill_init_prob=torch.tensor(cfg['env']['skillInitProb'],device=task.device)
    task._task_graph_preset='random_open_push'
    # Training reset path, task matching, ratios, all combinations and AMP history.
    fractions=[];late=[];combinations=[];phases=[]
    for _ in range(250):
        task._reset_envs(ids)
        assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task._amp_obs_buf).all()
        assert not task.relation_runtime.done.any() and not task.relation_runtime.elapsed.any()
        assert not task.relation_runtime.push_eligible.any()
        assert (task._amp_obs_buf[...,-2:].argmax(-1)==task.relation_runtime.tasks[:,:,None]).all(), (
            task.relation_runtime.tasks.tolist(),task._amp_obs_buf[...,-2:].argmax(-1).tolist(),
            {k:(v[0].tolist(),v[1].tolist()) for k,v in task._reset_ref_slots.items()})
        for skill,slots in task._reset_ref_slots.items():
            env,agent=slots
            if skill!='loco':assert (task.relation_runtime.tasks[env,agent]==(skill=='push')).all()
        fractions.extend(task._rsi_interaction.flatten().tolist())
        late.extend(task._rsi_late[task._rsi_interaction].tolist())
        phases.extend(task._rsi_phase[task._rsi_interaction].tolist())
        combinations.extend(task.relation_runtime.tasks.tolist())
        expected=task.relation_runtime.start.clone();expected[...,1]+=cfg['env']['relationReward']['push']['target_distance']
        mask=task.relation_runtime.tasks==1
        torch.testing.assert_close(task._tar_pos[mask],expected[mask])
    human=task._humanoid_root_states[1].clone();boxes=task._box_states[1].clone()
    hinge=task._door_dof_state[1].clone();graph=task.relation_runtime.suffix()[1].clone()
    task._reset_envs(ids[:1])
    for a,b in ((task._humanoid_root_states[1],human),(task._box_states[1],boxes),
                (task._door_dof_state[1],hinge),(task.relation_runtime.suffix()[1],graph)):
        torch.testing.assert_close(a,b)
    assert .4<sum(fractions)/len(fractions)<.6
    assert .6<sum(late)/len(late)<.8
    demo=task.fetch_amp_obs_demo(128);assert demo.shape==(128,1310) and torch.isfinite(demo).all()
    smoke_steps=int(os.environ.get('RSI_SMOKE_STEPS','0'))
    assert 0<=smoke_steps<=30
    for _ in range(smoke_steps):
        task.pre_physics_step(torch.zeros(4,32,device=task.device))
        task._physics_step();task.post_physics_step()
        assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task.rew_buf).all()
        assert torch.isfinite(task.relation_runtime.push_hand_distance).all()
    report=dict(training=False,device=task.device,counts=counts,bank_file=str(bank_path),
        reset_samples=len(fractions),interaction_fraction=sum(fractions)/len(fractions),
        late_request_fraction=sum(late)/len(late),phase_range=[min(phases),max(phases)],
        combinations=sorted(set(tuple(x) for x in combinations)),partial_reset=True,
        timers_contact_history_zero=True,goal_fixed=True,amp_finite=True,
        physics_smoke_steps=smoke_steps,contract=task.get_open_push_contract())
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'rejected.json').write_text(json.dumps(rejected,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
finally:
    task.gym.destroy_sim(task.sim)
