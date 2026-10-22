"""Non-training reset, ownership, RSI, AMP and physics checks for random PUSH scenes."""
from isaacgym import gymapi
import json
import hashlib
import os
from pathlib import Path
import torch
from utils.config import get_args,load_cfg,parse_sim_params,set_seed
from utils.push_scene import forward_xy,layout_valid,random_at_goal_valid
from env.tasks.multi_agent.humanoid_ma_open_push import HumanoidMAOpenPush
from env.tasks.multi_agent.open_push_reward import push_rsi_hand_targets

args=get_args();assert args.test and args.eval and args.num_envs==2 and args.num_objects in (2,4)
cfg,train,_=load_cfg(args);cfg['args']=args;cfg['env']['motion_file']=args.motion_file
cfg['env']['enableTrackInitState']=True
set_seed(42);torch.set_num_threads(1)
sim=parse_sim_params(args,cfg,train);sim.physx.num_threads=1
out=Path(args.output_path);out.mkdir(parents=True,exist_ok=True)
bank=Path(cfg['env']['openPushRSI']['bank_file']);bank_hash=hashlib.sha256(bank.read_bytes()).hexdigest()
task=HumanoidMAOpenPush(cfg,sim,args.physics_engine,args.device,args.device_id,True)
ids=torch.arange(2,device=task.device)
try:
    task._is_eval=False;task._skill=list(cfg['env']['skill'])
    task._skill_init_prob=torch.tensor(cfg['env']['skillInitProb'],device=task.device)
    interactions=[];late=[];assignments=[];headings=[];bearings=[];counts=[0,0]
    tilt_shaping=cfg['env']['relationReward'].get('at',{}).get('tilt_shaping')
    contact_tilt=cfg['env']['relationReward'].get('contact_at',{}).get('tilt_full_angle')
    reset_tilts=[];goal_distances=[];remaining_distances=[]
    for _ in range(250):
        task._reset_envs(ids)
        zones=task._agent_zone
        anchors=task._agent_spawn_offsets[zones][...,:2]
        original=task._zone_values(task._push_reset_states)
        relative=original[...,:2]-anchors
        distances=relative.norm(dim=-1)
        assert (distances>=1-1e-5).all() and (distances<=2+1e-5).all()
        torch.testing.assert_close(task.relation_runtime.start,original[...,:3])
        if task._random_at_goal:
            expected=task._zone_values(task._push_random_goals)
            assert random_at_goal_valid(task._push_reset_states[...,:2],task._push_reset_states[...,3:7],task._push_random_goals[...,:2],cfg['env']).all()
            delta=expected-original[...,:3]
            goal_distances.extend(delta[...,:2].norm(dim=-1).flatten().tolist())
            remaining=(expected[...,:2]-task._owned_push_states()[...,:2]).norm(dim=-1)
            remaining_distances.extend(remaining.flatten().tolist())
            assert (remaining<=delta[...,:2].norm(dim=-1)+1e-5).all()
            # RSI progress changes the box along this episode's goal ray only.
            moved=task._owned_push_states()[...,:2]-original[...,:2]
            direction=delta[...,:2]/delta[...,:2].norm(dim=-1,keepdim=True)
            along=(moved*direction).sum(-1)
            torch.testing.assert_close(moved,along[...,None]*direction,atol=1e-5,rtol=1e-5)
            assert (along>=-1e-5).all()
            torch.testing.assert_close(moved[~task._rsi_interaction],torch.zeros_like(moved[~task._rsi_interaction]))
            torch.testing.assert_close(task._push_goal_direction,delta/delta.norm(dim=-1,keepdim=True))
            roots=task._humanoid_root_states[...,:2]
            assert (roots[:,0]-roots[:,1]).norm(dim=-1).min()>=.7
            assert (expected[...,:2]-roots.flip(1)).norm(dim=-1).min()>=1.42
            assert (task._owned_push_states()[...,:2]-roots.flip(1)).norm(dim=-1).min()>=1.42
        else:
            torch.testing.assert_close(task._push_goal_direction[...,:2],forward_xy(original[...,3:7]))
            expected=original[...,:3]+2*task._push_goal_direction
        torch.testing.assert_close(task._tar_pos,expected)
        torch.testing.assert_close(task._logical_box_values(task._box_states)[:,task._push_slot_start:],task._owned_push_states())
        for agent in range(2):
            logical=task.relation_runtime.graph.edge_dst[:,2*agent]-2
            assert torch.equal(task._logical_box_order[ids,logical],task._agent_box_assignment[:,agent])
        local=task._rsi_interaction
        if tilt_shaping or contact_tilt:
            box=task._owned_push_states()
            up=1-2*(box[...,3].square()+box[...,4].square())
            angles=torch.acos(up.clamp(-1.,1.))
            assert (angles<=(contact_tilt or tilt_shaping['full_angle'])).all(), 'RSI starts with reduced AT tilt reward'
            reset_tilts.extend(angles.flatten().tolist())
        torch.testing.assert_close(task._humanoid_root_states[...,:2][~local],anchors[~local])
        assert not task.relation_runtime.push_hand_distance.any() and not task.relation_runtime.push_eligible.any()
        torch.testing.assert_close(task._push_episode_start,task._owned_push_states()[...,:3])
        assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task._amp_obs_buf).all()
        assert (task._amp_obs_buf[...,-2:].argmax(-1)==1).all()
        for agent in range(2):counts[agent]+=int((zones[:,agent]==0).sum())
        assignments.extend(zones[:,0].tolist());interactions.extend(local.flatten().tolist())
        late.extend(task._rsi_late[local].tolist())
        headings.extend(torch.atan2(task._push_goal_direction[...,1],task._push_goal_direction[...,0]).flatten().tolist())
        bearings.extend(torch.atan2(relative[...,1],relative[...,0]).flatten().tolist())
    assert .4<sum(assignments)/len(assignments)<.6
    assert .4<sum(interactions)/len(interactions)<.6 and .6<sum(late)/len(late)<.8
    # Partial reset preserves the other scene's geometry, assignment and histories.
    reset_tensors=[task._agent_zone,task._push_reset_states,task._push_goal_direction,
        task._box_states,task._humanoid_root_states,task._tar_pos,task._amp_obs_buf]
    if task._random_at_goal:reset_tensors.extend((task._push_random_goals,task._push_goal_distance))
    snapshots=[x[1].clone() for x in reset_tensors]
    task._reset_envs(ids[:1])
    for x,before in zip(reset_tensors,snapshots):
        torch.testing.assert_close(x[1],before)
    # Force both assignment maps while testing transformed RSI hand alignment.
    sampler=task._sample_zone_assignments
    for swapped in (False,True):
        task._sample_zone_assignments=lambda env_ids:torch.tensor([1,0] if swapped else [0,1],device=task.device).expand(len(env_ids),2)
        task._reset_envs(ids)
        for agent in range(2):
            sample=task._joint_rsi_bank.sample('push',2)
            task._apply_joint_rsi(ids,torch.full_like(ids,agent),'push',sample)
        task._reset_task(ids)
        task.relation_runtime.initialize_manipulation(ids,task._owned_push_states(),task._tar_pos,task.door_hinge_angle)
        task._push_episode_start.copy_(task._owned_push_states()[...,:3])
        task._reset_env_tensors(ids);task._compute_observations(ids);task._init_amp_obs(ids)
        targets=push_rsi_hand_targets(task._owned_push_states(),task._tar_pos,task._push_sizes(),task._relation_cfg)
        hands=task._kinematic_humanoid_rigid_body_states[:,:,task._key_body_ids[:2],:3]
        midpoint=hands[...,:2].mean(-2)-targets[...,:2].mean(-2)
        pair=targets[:,:,0,:2]-targets[:,:,1,:2]
        normal=torch.stack((-pair[...,1],pair[...,0]),-1)
        normal=normal/normal.norm(dim=-1,keepdim=True)
        if cfg['env']['openPushRSI'].get('pose_mode')=='native_phase':
            # Native collider retreat can exceed stand-off; each hand must
            # remain on the finite face, with original joint pose preserved.
            from env.tasks.multi_agent.open_push_reward import push_rear_xy
            assert (push_rear_xy(hands,task._owned_push_states(),task._tar_pos,task._push_sizes())<=.18+1e-5).all()
        else:
            torch.testing.assert_close(midpoint,-cfg['env']['openPushRSI']['stand_off']*normal,atol=1e-5,rtol=1e-5)
        goals=task._tar_pos.clone()
        for _ in range(12):
            task.pre_physics_step(torch.zeros(4,32,device=task.device));task._physics_step();task.post_physics_step()
            assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task.rew_buf).all()
            torch.testing.assert_close(task._tar_pos,goals)
    task._sample_zone_assignments=sampler
    assert torch.isfinite(task.fetch_amp_obs_demo(128)).all()
    assert bank_hash==hashlib.sha256(bank.read_bytes()).hexdigest()
    network_shapes=None
    if task._push_only:
        assert task.num_objects==2 and task._extra_object_dofs==0 and not hasattr(task,'_door_asset')
        assert not hasattr(task,'_door_dof_state')
        assert task.gym.get_env_rigid_body_count(task.envs[0])==2*task.num_bodies+2
        assert task.gym.get_env_dof_count(task.envs[0])==2*task.num_dof
        names=[task.gym.get_actor_name(task.envs[0],i) for i in range(task.gym.get_actor_count(task.envs[0]))]
        assert not any('door' in name for name in names)
        from learning.multi_agent.amp_network_builder_ma import AMPMultiAgentBuilder
        builder=AMPMultiAgentBuilder();builder.load(train['params']['network'])
        network=builder.build('amp',actions_num=32,input_shape=(task.get_obs_size(),),
            amp_input_shape=(task.get_num_amp_obs(),),value_size=1,num_agents=2,num_objects=2,
            humanoid_obs_size=230,object_obs_size=39,goal_obs_size=6,observation_mode='clean_scene',
            scene_entity_sizes=task.get_scene_entity_sizes(),scene_kinematic_size=7,scene_arena_scale=7.,
            relation_reward_mode=cfg['env']['relationReward']['mode'],
            relation_graph_spec=cfg['env']['relationGraph'],device=task.device).to(task.device)
        checkpoint=os.environ.get('TOKENHSI_CHECKPOINT')
        if checkpoint:
            from utils.open_push_spec import checkpoint_contracts_match
            weights=torch.load(checkpoint,map_location=task.device)
            assert checkpoint_contracts_match(weights['open_push_contract'],task.get_open_push_contract())
            assert all(k.startswith('a2c_network.') for k in weights['model'])
            network.load_state_dict({k[len('a2c_network.'):]:v for k,v in weights['model'].items()},strict=True)
        with torch.no_grad():
            actor=network.eval_actor(task.obs_buf);critic=network.eval_critic(task.obs_buf)
            disc=network.eval_disc(task.fetch_amp_obs_demo(128))
        assert all(torch.isfinite(x).all() for x in actor) and torch.isfinite(critic).all() and torch.isfinite(disc).all()
        network_shapes=dict(obs_size=task.get_obs_size(),actor=[list(x.shape) for x in actor],
            critic=list(critic.shape),discriminator=list(disc.shape),entity_tokens=6,
            rigid_bodies=task.gym.get_env_rigid_body_count(task.envs[0]),dofs=task.gym.get_env_dof_count(task.envs[0]))
        network_shapes['checkpoint_loaded']=checkpoint
    report=dict(push_only=task._push_only,network_shapes=network_shapes,training=False,device=task.device,reset_agent_samples=len(interactions),
        swap_fraction=sum(assignments)/len(assignments),interaction_fraction=sum(interactions)/len(interactions),
        late_request_fraction=sum(late)/len(late),yaw_range=[min(headings),max(headings)],
        bearing_range=[min(bearings),max(bearings)],owner_distance_range=[1,2],size=cfg['env']['pushBox']['size'],
        both_assignments_rsi=True,physical_steps=24,partial_reset=True,graph_owner=True,
        goals_follow_reset_yaw=not task._random_at_goal,random_at_goals=task._random_at_goal,
        initial_goal_distance_range=[min(goal_distances),max(goal_distances)] if goal_distances else None,
        post_rsi_remaining_distance_range=[min(remaining_distances),max(remaining_distances)] if remaining_distances else None,
        goal_fixed_after_rsi=True,amp_finite=True,bank_preserved=True,
        reset_max_tilt_degrees=max(reset_tilts)*180/3.141592653589793 if reset_tilts else None,
        contract=task.get_open_push_contract())
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
finally:task.gym.destroy_sim(task.sim)
