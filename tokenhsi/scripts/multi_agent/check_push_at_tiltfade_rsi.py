"""Non-training audit: restricted creation sizes, physical masses, RSI and AMP."""
from isaacgym import gymapi
import hashlib
import json
import os
from pathlib import Path
import torch
from utils.config import get_args,load_cfg,parse_sim_params,set_seed
from env.tasks.multi_agent.humanoid_ma_open_push import HumanoidMAOpenPush
from env.tasks.multi_agent.open_push_reward import contact_at_reward,push_hand_point_distance
from utils.open_push_sizes import size_values

torch.set_num_threads(1)
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
args=get_args();assert args.test and args.eval
cfg,train,_=load_cfg(args);cfg['args']=args;cfg['env']['motion_file']=args.motion_file;cfg['_size_audit']=True
sizes=size_values(cfg['env']);assert args.num_envs==2*len(sizes)
bank=Path(cfg['env']['openPushRSI']['bank_file']);before=hashlib.sha256(bank.read_bytes()).hexdigest()
sim=parse_sim_params(args,cfg,train);sim.physx.num_threads=1;set_seed(42)
task=HumanoidMAOpenPush(cfg,sim,args.physics_engine,args.device,args.device_id,True)
ids=torch.arange(task.num_envs,device=task.device)
try:
    assert task.get_obs_size()==570 and task.get_clean_object_obs_size()==30
    assert task.get_num_amp_obs()==1310 and set(task._motion_lib)=={'loco','push'}
    actual=[]
    for e in range(task.num_envs):
        for lane in range(2):
            handle=task._box_handles[e*4+lane+2]
            mass=sum(p.mass for p in task.gym.get_actor_rigid_body_properties(task.envs[e],handle))
            side=float(task._box_size[e,lane+2,0]);assert abs(mass-20*(side/1.3)**3)<.001
            actual.append(mass)
    task._is_eval=False;task._skill=list(cfg['env']['skill'])
    task._skill_init_prob=torch.tensor(cfg['env']['skillInitProb'],device=task.device)
    interaction=late=initial=0;hand_distances=[];seen=set()
    for _ in range(256):
        task._reset_envs(ids)
        interaction+=int(task._rsi_interaction.sum());late+=int(task._rsi_late.sum())
        boxes=task._owner_push_boxes();hands=task._rigid_body_pos[:,:,task._key_body_ids[:2]]
        r=contact_at_reward(task._humanoid_root_states[...,:3],hands,boxes,task._tar_pos,task._relation_cfg,task._owner_push_sizes(),None,True)
        initial+=int(r['success'].sum());assert not r['success'].any()
        assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task._amp_obs_buf).all()
        mask=task._rsi_interaction
        # Read initialized reference hands, before the first simulator refresh.
        ref_hands=task._kinematic_humanoid_rigid_body_states[:,:,task._key_body_ids[:2],:3]
        distance=push_hand_point_distance(ref_hands,boxes,task._owner_push_sizes(),task._relation_cfg['push'],task._tar_pos).amax(-1)
        hand_distances.extend(distance[mask].cpu().tolist())
        seen.update(round(float(v),2) for v in task._owner_push_sizes()[...,0].flatten())
    count=256*task.num_envs*2
    assert .47<interaction/count<.53 and .65<late/interaction<.75
    assert seen==set(sizes) and min(seen)==1.2
    assert max(hand_distances)<task._relation_cfg['push']['hand_distance']
    demo=task.fetch_amp_obs_demo(128);assert demo.shape==(128,1310) and torch.isfinite(demo).all()
    states=[v[1:].clone() for v in (task._humanoid_root_states,task._box_states,task._amp_obs_buf)]
    task._reset_envs(ids[:1])
    for old,new in zip(states,(task._humanoid_root_states,task._box_states,task._amp_obs_buf)):
        torch.testing.assert_close(old,new[1:],rtol=0,atol=0)
    actions=torch.zeros(task.num_envs*2,task.num_actions,device=task.device)
    for _ in range(12):
        task.pre_physics_step(actions);task._physics_step();task.post_physics_step()
        assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task.rew_buf).all() and torch.isfinite(task._amp_obs_buf).all()
    assert hashlib.sha256(bank.read_bytes()).hexdigest()==before
    report=dict(training=False,reset_agent_samples=count,sizes=sorted(seen),physical_mass_kg_range=[min(actual),max(actual)],
        initial_at_success_count=initial,rsi_fraction=interaction/count,late_request_fraction=late/interaction,
        initial_rsi_hand_target_distance_max_m=max(hand_distances),partial_reset_preserved=True,
        policy_width=task.get_obs_size(),object_width=task.get_clean_object_obs_size(),amp_width=task.get_num_amp_obs(),
        finite_simulation_steps=12,bank_sha256=before,bank_unchanged=True,
        candidates=len(task._joint_rsi_bank.rows['push']))
    out=Path(args.output_path);out.mkdir(parents=True,exist_ok=True)
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
finally:task.gym.destroy_sim(task.sim)
