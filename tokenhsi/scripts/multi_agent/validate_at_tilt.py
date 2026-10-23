"""Non-training audit of unchanged RSI/AMP and migrated AT-only reward gate."""
from isaacgym import gymapi
from pathlib import Path
import hashlib
import json
import os
import torch
from utils.config import get_args,load_cfg,parse_sim_params,set_seed
from utils.relation_task_spec import checkpoint_metadata
from env.tasks.multi_agent.humanoid_ma_open_push import HumanoidMAOpenPush
from env.tasks.multi_agent.open_push_reward import contact_at_reward
from learning.multi_agent.amp_network_builder_ma import AMPMultiAgentBuilder
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from learning.multi_agent.scene_normalizer import SceneRunningMeanStd


def equal(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


torch.set_num_threads(1)
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
args=get_args();assert args.test and args.eval and args.num_envs==62
cfg,train,_=load_cfg(args);cfg['args']=args;cfg['env']['motion_file']=args.motion_file;cfg['_size_audit']=True
bank=Path(cfg['env']['openPushRSI']['bank_file']);before=hashlib.sha256(bank.read_bytes()).hexdigest()
sim=parse_sim_params(args,cfg,train);sim.physx.num_threads=1;set_seed(42)
task=HumanoidMAOpenPush(cfg,sim,args.physics_engine,args.device,args.device_id,True)
ids=torch.arange(task.num_envs,device=task.device)
try:
    w=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    source=Path(w['at_tilt_migration']['source_checkpoint']) if 'source_checkpoint' in w['at_tilt_migration'] else next(Path('output/open_push_contact_at_friction06_random_size_goal1to25_at01').glob('*/nn/*_00006000.pth'))
    original=torch.load(source,map_location='cpu',weights_only=False)
    changed={'relation_metadata','open_push_contract','relation_experiment_config'}
    for key in original:
        if key not in changed:equal(original[key],w[key])
    assert w['open_push_contract']==task.get_open_push_contract()
    assert w['relation_metadata']==checkpoint_metadata(task._relation_cfg)
    assert w['epoch']==w['at_tilt_migration']['source_epoch']
    if cfg['env'].get('pushOnly',False):
        assert task.get_obs_size()==570 and task.get_clean_object_obs_size()==30
        assert task.get_policy_num_objects()==2 and set(task._motion_lib)=={'loco','push'}
    # Training reset path, with unchanged sampling probabilities and bank.
    task._is_eval=False
    task._skill=list(cfg['env']['skill']);task._skill_init_prob=torch.tensor(cfg['env']['skillInitProb'],device=task.device)
    interaction=late=initial=0;tilts=[];remaining=[]
    for _ in range(256):
        task._reset_envs(ids)
        interaction+=int(task._rsi_interaction.sum());late+=int(task._rsi_late.sum())
        boxes=task._owner_push_boxes();q=boxes[...,3:7]
        tilt=torch.acos((1-2*(q[...,0].square()+q[...,1].square())).clamp(-1,1))
        tilts.append(tilt.flatten());remaining.append((boxes[...,:2]-task._tar_pos[...,:2]).norm(dim=-1).flatten())
        roots=task._humanoid_root_states[...,:3]
        hands=task._rigid_body_pos[:,:,task._key_body_ids[:2]]
        r=contact_at_reward(roots,hands,boxes,task._tar_pos,task._relation_cfg,task._owner_push_sizes(),None,True)
        initial+=int(r['success'].sum())
        assert not r['success'].any()
        assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task._amp_obs_buf).all()
        assert (task._amp_obs_buf[...,-2:].argmax(-1)==1).all()
    samples=256*62*2
    assert .47<interaction/samples<.53 and .65<late/interaction<.75
    assert len(torch.unique(task._box_size[:,2:,0]))==31
    demo=task.fetch_amp_obs_demo(128);assert demo.shape==(128,1310) and torch.isfinite(demo).all()
    assert (demo.reshape(128,task._num_amp_obs_steps,-1)[...,-2:].argmax(-1)==1).all()
    # Load the actual preserved actor, critic, AMP discriminator and input RMS.
    builder=AMPMultiAgentBuilder();builder.load(train['params']['network'])
    network=builder.build('amp',actions_num=task.num_actions,input_shape=(task.get_obs_size(),),amp_input_shape=(task.get_num_amp_obs(),),value_size=1,
        num_agents=2,num_objects=task.get_policy_num_objects(),humanoid_obs_size=task.get_humanoid_obs_size(),object_obs_size=task.get_object_obs_size(),goal_obs_size=task.get_goal_obs_size(),observation_mode=task.get_policy_obs_mode(),
        scene_entity_sizes=task.get_scene_entity_sizes(),scene_kinematic_size=task.get_scene_kinematic_size(),scene_arena_scale=task.get_scene_arena_scale(),
        relation_reward_mode=task._relation_cfg['mode'],relation_graph_spec=task._relation_graph_spec,device=task.device).to(task.device)
    network.load_state_dict({k[len('a2c_network.') :]:v for k,v in w['model'].items() if k.startswith('a2c_network.')},strict=True);network.eval()
    obs_rms=SceneRunningMeanStd(entity_sizes=task.get_scene_entity_sizes(),entity_counts=[2,task.get_policy_num_objects(),2],normalized_sizes=task.get_scene_normalized_entity_sizes(),kinematic_size=task.get_scene_kinematic_size(),extra_passthrough_size=task.get_relation_suffix_size()).to(task.device);obs_rms.load_state_dict(w['running_mean_std']);obs_rms.eval()
    amp_rms=RunningMeanStd((131,)).to(task.device);amp_rms.load_state_dict(w['amp_input_mean_std']);amp_rms.eval()
    with torch.no_grad():
        logits=network.eval_disc(amp_rms(demo.reshape(-1,131)).reshape(-1,1310))
        demo_reward=-torch.log(torch.clamp(1-torch.sigmoid(logits),min=1e-4))
        assert torch.isfinite(logits).all() and torch.isfinite(demo_reward).all()
        for _ in range(12):
            obs=obs_rms(task.obs_buf)
            actor=network.eval_actor(obs);critic=network.eval_critic(obs)
            assert all(torch.isfinite(x).all() for x in actor) and torch.isfinite(critic).all()
            task.pre_physics_step(actor[0].reshape(-1,32));task._physics_step();task.post_physics_step()
            assert torch.isfinite(task.rew_buf).all() and torch.isfinite(task._amp_obs_buf).all()
            agent_logits=network.eval_disc(amp_rms(task._amp_obs_buf.reshape(-1,131)).reshape(-1,1310))
            assert torch.isfinite(agent_logits).all()
    states=[x[1:].clone() for x in (task._humanoid_root_states,task._box_states,task._amp_obs_buf)]
    task._reset_envs(ids[:1])
    for old,new in zip(states,(task._humanoid_root_states,task._box_states,task._amp_obs_buf)):torch.testing.assert_close(old,new[1:])
    assert hashlib.sha256(bank.read_bytes()).hexdigest()==before
    tilt=torch.cat(tilts);distance=torch.cat(remaining)
    report=dict(training=False,epoch=w['epoch'],samples=samples,initial_at_success_count=initial,interaction_fraction=interaction/samples,late_request_fraction=late/interaction,
        tilt_max_rad=float(tilt.max()),tilt_over_limit_fraction=float((tilt>.35).float().mean()),remaining_distance_min_m=float(distance.min()),
        policy_width=task.get_obs_size(),object_width=task.get_clean_object_obs_size(),amp_width=task.get_num_amp_obs(),sizes=31,
        bank_sha256=before,bank_unchanged=True,all_learned_states_preserved=True,partial_reset_preserved=True,simulation_steps=12,
        actor_critic_amp_checkpoint_inference_finite=True,amp_demo_reward_mean=float(demo_reward.mean()),contract=w['open_push_contract'])
    out=Path(args.output_path);out.mkdir(parents=True,exist_ok=True);(out/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
finally:task.gym.destroy_sim(task.sim)
