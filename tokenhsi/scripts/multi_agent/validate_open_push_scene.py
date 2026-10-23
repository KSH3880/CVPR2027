"""Non-training reset, owner binding, arbitrary PUSH yaw and RSI physics audit."""
from isaacgym import gymapi
import json
from pathlib import Path
import torch
from utils.config import get_args,load_cfg,parse_sim_params,set_seed
from env.tasks.multi_agent.humanoid_ma_open_push import HumanoidMAOpenPush
from env.tasks.multi_agent.open_push_reward import push_hand_targets,push_hand_contact,push_face_geometry

torch.set_num_threads(1)
args=get_args();assert args.num_envs in (2,62) and args.test and args.eval
cfg,train,_=load_cfg(args);cfg['args']=args;cfg['env']['motion_file']=args.motion_file
cfg['env']['enableTrackInitState']=True
if args.num_envs==62:cfg['_size_audit']=True
set_seed(42)
task=HumanoidMAOpenPush(cfg,parse_sim_params(args,cfg,train),args.physics_engine,args.device,args.device_id,True)
ids=torch.arange(args.num_envs,device=task.device);out=Path(args.output_path);out.mkdir(parents=True,exist_ok=True)
report={'training':False,'checks':{}}
physical=[]
for e in range(args.num_envs):
    for b in (2,3):
        handle=task._box_handles[4*e+b]
        mass=sum(p.mass for p in task.gym.get_actor_rigid_body_properties(task.envs[e],handle))
        mu=[p.friction for p in task.gym.get_actor_rigid_shape_properties(task.envs[e],handle)]
        size=task._box_size[e,b].cpu().tolist()
        from utils.open_push_sizes import density
        expected_mass=density(cfg['env'])*size[0]*size[1]*size[2]
        assert abs(mass-expected_mass)<.001
        assert all(abs(x-cfg['env']['pushBox']['friction'])<1e-5 for x in mu)
        physical.append(dict(env=e,box=b,mass_kg=mass,size_m=size,friction=mu))
report['physical_properties']=physical
if 'pushSizeRandomization' in cfg['env']:
    sides=sorted(set(round(row['size_m'][0],2) for row in physical))
    if args.num_envs==62:assert len(sides)==31 and sides[0]==1.1 and sides[-1]==1.4
    report['checks']['physical_sides_m']=sides
    torch.testing.assert_close(task._box_bps[:,2:].abs().amax(-2)*2,task._box_size[:,2:])
    report['checks']['bps_matches_physical_size']=True
if cfg['env']['relationGraph']['task_probabilities']['PUSH']==1.:
    demo=task.fetch_amp_obs_demo(128).reshape(128,task._num_amp_obs_steps,-1)
    assert (demo[...,-2:].argmax(-1)==1).all()
    report['checks']['amp_push_family_only']=True

def check():
    logical=task._logical_box_values(task._box_states);boxes=task._owner_push_boxes()
    torch.testing.assert_close(logical[:,2:],boxes)
    binding=torch.where(task.relation_runtime.tasks==1,task._push_binding+2,task._agent_lanes)
    torch.testing.assert_close(task._agent_box_assignment,binding)
    direction=task._owner_push_values(task._push_directions)
    goal=task.relation_runtime.start+task._owner_push_values(task._push_goal_offsets)
    expected=torch.where((task.relation_runtime.tasks==1)[...,None],goal,task._owner_values(task.door_handle_position))
    torch.testing.assert_close(task._tar_pos,expected)
    packet=task.obs_buf[:,608:664].reshape(args.num_envs,8,7)
    torch.testing.assert_close(packet[:,4:6,:3],boxes[...,:3]);torch.testing.assert_close(packet[:,6:8,:3],expected)
    assert (task._amp_obs_buf[...,-2:].argmax(-1)==task.relation_runtime.tasks[:,:,None]).all()
    assert torch.isfinite(task.obs_buf).all()
    if cfg['env']['relationGraph']['task_probabilities']['PUSH']==1.:assert (task.relation_runtime.tasks==1).all()
    assert ((task._push_reset_states[:,0,:2]-task._push_reset_states[:,1,:2]).norm(dim=-1)>=cfg['env']['pushSceneRandomization']['min_station_distance']).all()
    goals=task._push_contact_goals()
    _,normal,_,_,_=push_face_geometry(boxes,task._owner_push_sizes() if 'pushSizeRandomization' in cfg['env'] else cfg['env']['pushBox']['size'],goals)
    physical_force=task._box_forces.clone();physical_force.zero_()
    physical_force.scatter_(1,task._push_binding[...,None].expand(-1,-1,3),normal*2)
    physical_force[:,1]=0
    hands=push_hand_targets(boxes,task._owner_push_sizes() if 'pushSizeRandomization' in cfg['env'] else cfg['env']['pushBox']['size'],cfg['env']['relationReward']['push'],goals)
    hand_force=-normal[:,:,None].expand_as(hands)
    hit=push_hand_contact(hands,hand_force,boxes,task._owner_push_values(physical_force),task._owner_push_sizes() if 'pushSizeRandomization' in cfg['env'] else cfg['env']['pushBox']['size'],cfg['env']['relationReward']['push'],task._push_contact_goals())
    if cfg['env']['relationReward']['push'].get('contact_mode')=='any_surface':
        # Arbitrary surface contact depends on each hand normal, not net box force.
        assert hit.all()
        assert not push_hand_contact(hands,torch.zeros_like(hand_force),boxes,task._owner_push_values(physical_force),cfg['env']['pushBox']['size'],cfg['env']['relationReward']['push'],goals).any()
    else:
        assert torch.equal(hit,task._push_binding==0)

def step():
    task.pre_physics_step(torch.zeros(args.num_envs*2,32,device=task.device));task._physics_step();task.post_physics_step()
    assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task.rew_buf).all() and torch.isfinite(task._amp_obs_buf).all()

try:
    locations=[];directions=[];goal_offsets=[];bindings=set();distances=[];face_sides=set()
    for _ in range(100):
        task._task_graph_preset='push';task._reset_envs(ids);check()
        if cfg['env'].get('pushGoalContact',False):
            center,normal,_,_,sides=push_face_geometry(task._owner_push_boxes(),task._owner_push_sizes() if 'pushSizeRandomization' in cfg['env'] else cfg['env']['pushBox']['size'],task._tar_pos)
            boxes=task._owner_push_boxes()[...,:3]
            assert ((center-boxes)*(task._tar_pos-boxes)).sum(-1).lt(0).all()
            face_sides.update(sides.flatten().cpu().tolist())
        locations.append(task._push_reset_states[...,:2].clone());directions.append(task._push_directions.clone());goal_offsets.append(task._push_goal_offsets.clone())
        bindings.update(zip(task._agent_lanes[:,0].cpu().tolist(),task._push_binding[:,0].cpu().tolist()))
        distance=(task._humanoid_root_states[...,:2]-task._owner_push_boxes()[...,:2]).norm(dim=-1)
        assert (distance>=.9999).all() and (distance<=2.0001).all();distances.append(distance)
    assert bindings=={(0,0),(0,1),(1,0),(1,1)}
    xy=torch.stack(locations);heading=torch.stack(directions)
    assert xy[...,0].std()>.5 and xy[...,1].std()>.5 and heading[...,0].min()<-.9 and heading[...,0].max()>.9
    report['checks']['200_reset_geometry_binding']={'xy_std':xy.flatten(0,2).std(0).cpu().tolist(),'distance_min':float(torch.stack(distances).min()),'distance_max':float(torch.stack(distances).max()),'binding_combinations':[list(x) for x in sorted(bindings)]}
    before=[x[1].clone() for x in (task._agent_lanes,task._push_binding,task._box_states,task._push_directions,task._humanoid_root_states,task._tar_pos,task.relation_runtime.start,task._push_goal_offsets)]
    task._reset_envs(ids[:1])
    for a,b in zip(before,(task._agent_lanes,task._push_binding,task._box_states,task._push_directions,task._humanoid_root_states,task._tar_pos,task.relation_runtime.start,task._push_goal_offsets)):torch.testing.assert_close(a,b[1])
    report['checks']['partial_reset']=True
    if 'pushGoalRandomization' in cfg['env']:
        offsets=torch.stack(goal_offsets)
        goal_spec=cfg['env']['pushGoalRandomization']
        if goal_spec.get('sampling') in ('radial_mixture','radial_uniform'):
            distance=offsets.norm(dim=-1)
            lo,hi=goal_spec['distance_range']
            assert (distance>=lo-1e-5).all() and (distance<=hi+1e-5).all()
            near=None
            if goal_spec['sampling']=='radial_mixture':
                near=(distance<goal_spec['near_distance']).float().mean().item()
                assert abs(near-goal_spec['near_probability'])<.08
            else:
                assert abs(distance.mean().item()-(lo+hi)/2)<.08
            assert (offsets[...,2]==0).all()
            report['checks']['radial_goal_sampling']={'min':float(distance.min()),'max':float(distance.max()),'near_fraction':near}
        else:
            assert (offsets[...,:2].abs()<=goal_spec['half_extent']+1e-5).all()
            assert (offsets.norm(dim=-1)>=goal_spec['min_distance']-1e-5).all()
        assert ((offsets*heading).sum(-1)<0).any()
        assert offsets[...,0].min()< -1.4 and offsets[...,0].max()>1.4
        report['checks']['random_goal_independent_of_yaw']=True
        if cfg['env'].get('pushGoalContact',False):
            assert face_sides=={0,1,2,3}
            report['checks']['goal_opposite_four_faces']=sorted(face_sides)
    presets=('push',) if cfg['env']['relationGraph']['task_probabilities']['PUSH']==1. else ('open','push','open_push','push_open')
    for preset in presets:
        task._task_graph_preset=preset
        for _ in range(4):
            task._reset_envs(ids);check()
            for _ in range(12):step()
    report['checks']['tasks_loco_12steps']=list(presets)
    task._is_eval=False;task._skill=list(cfg['env']['skill']);task._skill_init_prob=torch.tensor(cfg['env']['skillInitProb'],device=task.device)
    task._joint_rsi_spec=dict(task._joint_rsi_spec,interaction_probability=1.)
    max_hand=0.
    for preset in presets:
        task._task_graph_preset=preset
        for _ in range(8):
            task._reset_envs(ids);check();assert task._rsi_interaction.all()
            push=task.relation_runtime.tasks==1
            if push.any():
                hands=task._kinematic_humanoid_rigid_body_states[:,:,task._key_body_ids[:2],:3]
                targets=push_hand_targets(task._owner_push_boxes(),task._owner_push_sizes() if 'pushSizeRandomization' in cfg['env'] else cfg['env']['pushBox']['size'],cfg['env']['relationReward']['push'],task._push_contact_goals())
                error=(hands-targets).norm(dim=-1)[push].max();max_hand=max(max_hand,float(error));assert error<.18, float(error)
            for _ in range(12):step()
    report['checks']['tasks_RSI_12steps']=list(presets);report['checks']['max_RSI_hand_point_error_m']=max_hand
    task._joint_rsi_spec=cfg['env']['openPushRSI'];report['contract']=task.get_open_push_contract()
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
finally:task.gym.destroy_sim(task.sim)
