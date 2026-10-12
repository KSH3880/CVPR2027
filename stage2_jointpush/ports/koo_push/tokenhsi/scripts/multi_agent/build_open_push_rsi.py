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
screen_batch=int(os.environ.get('RSI_SCREEN_BATCH_SIZE','1'))
assert screen_batch>=1 and args.test and args.eval and args.num_envs==2*screen_batch
cfg,train,_=load_cfg(args);cfg['args']=args;cfg['_rsi_build']=os.environ.get('RSI_REUSE_BANK')!='1'
cfg['env']['motion_file']=args.motion_file;cfg['env']['enableTrackInitState']=True
sim=parse_sim_params(args,cfg,train);sim.physx.num_threads=1;set_seed(42)
task=HumanoidMAOpenPush(cfg,sim,args.physics_engine,args.device,args.device_id,True)
ids=torch.arange(args.num_envs,device=task.device)
out=Path(args.output_path);out.mkdir(parents=True,exist_ok=True)
accepted=dict(task._joint_rsi_bank.rows) if not cfg['_rsi_build'] else {};counts={};rejected=[]
try:
    for skill in (('doorOpen','push') if cfg['_rsi_build'] else ()):
        proposals=task._joint_rsi_bank.rows[skill];accepted[skill]=[]
        task._task_graph_preset='open' if skill=='doorOpen' else 'push'
        for index in range(0,len(proposals),screen_batch):
            rows=proposals[index:index+screen_batch]
            padded=rows+[rows[-1]]*(screen_batch-len(rows))
            task._reset_envs(ids)
            sample={k:torch.tensor([r[k] for r in padded],device=task.device).repeat_interleave(2) for k in rows[0]}
            task._apply_joint_rsi(ids,torch.zeros_like(ids),skill,sample)
            task._reset_task(ids);task._reset_env_tensors(ids);task._sync_door_reset_states(ids)
            task._compute_observations(ids);task._init_amp_obs(ids)
            assert not task.relation_runtime.done.any() and not task.relation_runtime.elapsed.any()
            assert not task.relation_runtime.push_eligible.any()
            initial=task._humanoid_root_states[:,0,:3].clone()
            # Each candidate must pass in its own pair of independent scenes.
            # Preserve simultaneous pair support/proximity at a physics tick.
            pair_all=lambda values:values.reshape(screen_batch,-1).all(-1)
            valid=torch.ones(screen_batch,dtype=torch.bool,device=task.device)
            support=torch.zeros_like(valid);near=torch.zeros_like(valid)
            actions=torch.zeros(args.num_envs*task.num_agents,32,device=task.device)
            for step in range(12):
                task.pre_physics_step(actions);task._physics_step();task.post_physics_step()
                valid &= pair_all(torch.isfinite(task.obs_buf)) & pair_all(torch.isfinite(task.rew_buf))
                valid &= pair_all(task._humanoid_root_states[:,0,2]>.65) & pair_all(task.reset_buf==0)
                valid &= pair_all((task._humanoid_root_states[:,0,:3]-initial).norm(dim=-1)<.45)
                if step>=2:
                    if skill=='doorOpen':
                        support |= pair_all(task.relation_runtime.posture[:,0])
                    else:
                        feet=task._posture_body_ids[2:]
                        low=task._rigid_body_pos[:,0,feet,2]<.35
                        force=task._contact_forces[:,0,feet,2]>=1.
                        support |= pair_all((low & force).any(-1))
                hands=task._rigid_body_pos[:,:,task._key_body_ids[:2]]
                if skill=='doorOpen':
                    distance=(hands-task.door_handle_position[:,:,None]).norm(dim=-1).amin(-1)[:,0]
                    near |= pair_all(distance<.2)
                else:
                    distance=push_hand_point_distance(hands,task._owner_push_boxes(),cfg['env']['pushBox']['size'],cfg['env']['relationReward']['push'])[:,0]
                    near |= pair_all(distance<cfg['env']['relationReward']['push']['hand_distance'])
                    boxes=task._owner_push_boxes()[:,0]
                    valid &= pair_all((boxes[:,2]-cfg['env']['pushBox']['size'][2]/2).abs()<cfg['env']['relationReward']['push']['ground_tolerance'])
            verdicts=torch.stack((valid,support,near),-1).cpu().tolist()
            for offset,row in enumerate(rows):
                v,s,n=verdicts[offset]
                if v and s and n:
                    accepted[skill].append(row)
                else:
                    rejected.append(dict(skill=skill,index=index+offset,valid=v,support=s,near=n,**row))
            if index%40==0:print('RSI_SCREEN',skill,index,len(proposals),'accepted',len(accepted[skill]),flush=True)
        counts[skill]={'proposed':len(proposals),'accepted':len(accepted[skill])}
    contract=bank_contract(cfg['env'])
    (out/'screen_counts.json').write_text(json.dumps(counts,indent=2)+'\n')
    (out/'rejected.json').write_text(json.dumps(rejected,indent=2)+'\n')
    (out/'accepted.json').write_text(json.dumps(accepted,indent=2)+'\n')
    # Require each clip to have survivors before producing a runnable bank.
    bank=OpenPushRSIBank(cfg['env']['openPushRSI'],task._motion_lib,contract,accepted)
    payload=dict(contract=contract,screen='PD_12_steps',candidates=accepted,
        notes='Object trajectories are geometric proxies, not measured BONES states. PD screen verifies short pose stability/foot support/hand proximity, not policy success or sustained manipulation.')
    bank_path=Path(cfg['env']['openPushRSI']['bank_file']);bank_path.write_text(json.dumps(payload,indent=2)+'\n')
    task._joint_rsi_bank=bank;task._is_eval=False
    task._skill=list(cfg['env']['skill'])
    task._skill_init_prob=torch.tensor(cfg['env']['skillInitProb'],device=task.device)
    task._task_graph_preset='random_open_push'
    # Training reset path, task matching, ratios, all combinations and AMP history.
    fractions=[];late=[];combinations=[];phases=[]
    for _ in range((250+screen_batch-1)//screen_batch):
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
        expected=task.relation_runtime.start+task._owner_push_values(task._push_goal_offsets)
        mask=task.relation_runtime.tasks==1
        torch.testing.assert_close(task._tar_pos[mask],expected[mask])
    human=task._humanoid_root_states[1:].clone();boxes=task._box_states[1:].clone()
    hinge=task._door_dof_state[1:].clone();graph=task.relation_runtime.suffix()[1:].clone()
    task._reset_envs(ids[:1])
    for a,b in ((task._humanoid_root_states[1:],human),(task._box_states[1:],boxes),
                (task._door_dof_state[1:],hinge),(task.relation_runtime.suffix()[1:],graph)):
        torch.testing.assert_close(a,b)
    assert .4<sum(fractions)/len(fractions)<.6
    assert .6<sum(late)/len(late)<.8
    demo=task.fetch_amp_obs_demo(128);assert demo.shape==(128,1310) and torch.isfinite(demo).all()
    report=dict(training=False,device=task.device,screen_batch=screen_batch,num_envs=args.num_envs,counts=counts,bank_file=str(bank_path),
        reset_samples=len(fractions),interaction_fraction=sum(fractions)/len(fractions),
        late_request_fraction=sum(late)/len(late),phase_range=[min(phases),max(phases)],
        combinations=sorted(set(tuple(x) for x in combinations)),partial_reset=True,
        timers_contact_history_zero=True,goal_fixed=True,amp_finite=True,contract=task.get_open_push_contract())
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'rejected.json').write_text(json.dumps(rejected,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
finally:
    task.gym.destroy_sim(task.sim)
