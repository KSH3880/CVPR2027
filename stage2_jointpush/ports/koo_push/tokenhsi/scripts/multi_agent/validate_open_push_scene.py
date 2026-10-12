"""Non-training reset, owner binding, arbitrary PUSH yaw and RSI physics audit."""
from isaacgym import gymapi
import json
from pathlib import Path
import torch
from utils.config import get_args,load_cfg,parse_sim_params,set_seed
from env.tasks.multi_agent.humanoid_ma_open_push import HumanoidMAOpenPush
from env.tasks.multi_agent.open_push_reward import push_hand_targets,push_hand_contact

torch.set_num_threads(1)
args=get_args();assert args.num_envs==2 and args.test and args.eval
cfg,train,_=load_cfg(args);cfg['args']=args;cfg['env']['motion_file']=args.motion_file
cfg['env']['enableTrackInitState']=True
set_seed(42)
task=HumanoidMAOpenPush(cfg,parse_sim_params(args,cfg,train),args.physics_engine,args.device,args.device_id,True)
ids=torch.arange(2,device=task.device);out=Path(args.output_path);out.mkdir(parents=True,exist_ok=True)
report={'training':False,'checks':{}}

def check():
    logical=task._logical_box_values(task._box_states);boxes=task._owner_push_boxes()
    torch.testing.assert_close(logical[:,2:],boxes)
    binding=torch.where(task.relation_runtime.tasks==1,task._push_binding+2,task._agent_lanes)
    torch.testing.assert_close(task._agent_box_assignment,binding)
    direction=task._owner_push_values(task._push_directions)
    goal=task.relation_runtime.start+task._owner_push_values(task._push_goal_offsets)
    expected=torch.where((task.relation_runtime.tasks==1)[...,None],goal,task._owner_values(task.door_handle_position))
    torch.testing.assert_close(task._tar_pos,expected)
    packet=task.obs_buf[:,608:664].reshape(2,8,7)
    torch.testing.assert_close(packet[:,4:6,:3],boxes[...,:3]);torch.testing.assert_close(packet[:,6:8,:3],expected)
    assert (task._amp_obs_buf[...,-2:].argmax(-1)==task.relation_runtime.tasks[:,:,None]).all()
    assert torch.isfinite(task.obs_buf).all()
    assert ((task._push_reset_states[:,0,:2]-task._push_reset_states[:,1,:2]).norm(dim=-1)>=cfg['env']['pushSceneRandomization']['min_station_distance']).all()
    physical_force=task._box_forces.clone();physical_force.zero_();physical_force[:,0]=task._push_directions[:,0]*2
    hands=push_hand_targets(boxes,cfg['env']['pushBox']['size'],cfg['env']['relationReward']['push'])
    hand_force=-direction[:,:,None].expand_as(hands)
    hit=push_hand_contact(hands,hand_force,boxes,task._owner_push_values(physical_force),cfg['env']['pushBox']['size'],cfg['env']['relationReward']['push'])
    assert torch.equal(hit,task._push_binding==0)

def step():
    task.pre_physics_step(torch.zeros(4,32,device=task.device));task._physics_step();task.post_physics_step()
    assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task.rew_buf).all() and torch.isfinite(task._amp_obs_buf).all()

try:
    locations=[];directions=[];goal_offsets=[];bindings=set();distances=[]
    for _ in range(100):
        task._task_graph_preset='push';task._reset_envs(ids);check()
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
        assert (offsets[...,:2].abs()<=1.50001).all()
        assert (offsets.norm(dim=-1)>=.99999).all()
        assert ((offsets*heading).sum(-1)<0).any()
        assert offsets[...,0].min()< -1.4 and offsets[...,0].max()>1.4
        report['checks']['random_goal_independent_of_yaw']=True
    for preset in ('open','push','open_push','push_open'):
        task._task_graph_preset=preset
        for _ in range(4):
            task._reset_envs(ids);check()
            for _ in range(12):step()
    report['checks']['four_tasks_loco_16reset_12steps']=True
    task._is_eval=False;task._skill=list(cfg['env']['skill']);task._skill_init_prob=torch.tensor(cfg['env']['skillInitProb'],device=task.device)
    task._joint_rsi_spec=dict(task._joint_rsi_spec,interaction_probability=1.)
    max_hand=0.
    for preset in ('open','push','open_push','push_open'):
        task._task_graph_preset=preset
        for _ in range(8):
            task._reset_envs(ids);check();assert task._rsi_interaction.all()
            push=task.relation_runtime.tasks==1
            if push.any():
                hands=task._kinematic_humanoid_rigid_body_states[:,:,task._key_body_ids[:2],:3]
                targets=push_hand_targets(task._owner_push_boxes(),cfg['env']['pushBox']['size'],cfg['env']['relationReward']['push'])
                error=(hands-targets).norm(dim=-1)[push].max();max_hand=max(max_hand,float(error));assert error<.18
            for _ in range(12):step()
    report['checks']['four_tasks_RSI_32reset_12steps']=True;report['checks']['max_RSI_hand_point_error_m']=max_hand
    task._joint_rsi_spec=cfg['env']['openPushRSI'];report['contract']=task.get_open_push_contract()
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
finally:task.gym.destroy_sim(task.sim)
