"""Check physical ownership, partial reset, RSI and graph under both zone maps."""
from isaacgym import gymapi  # before torch
import json
from pathlib import Path
import torch
from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from utils.open_push_spec import graph_for_tasks
from env.tasks.multi_agent.humanoid_ma_open_push import HumanoidMAOpenPush
from env.tasks.multi_agent.open_push_reward import push_hand_targets, push_hand_contact

args=get_args()
assert args.test and args.eval and args.num_envs==2
cfg,train,_=load_cfg(args);cfg['args']=args
cfg['env']['motion_file']=args.motion_file
cfg['env']['enableTrackInitState']=True
set_seed(42);torch.set_num_threads(1)
sim=parse_sim_params(args,cfg,train);sim.physx.num_threads=1
out=Path(args.output_path);out.mkdir(parents=True,exist_ok=True)
task=HumanoidMAOpenPush(cfg,sim,args.physics_engine,args.device,args.device_id,True)
ids=torch.arange(2,device=task.device)
try:
    original_sampler=task._sample_zone_assignments
    maps=ids[:,None].expand(2,2)^ids[None,:]
    task._sample_zone_assignments=lambda env_ids:maps[env_ids]
    for preset in ('open','push','open_push','push_open'):
        task._task_graph_preset=preset;task._reset_envs(ids)
        assert torch.equal(task._agent_zone,maps)
        assert torch.equal(task._logical_box_order,torch.cat((maps,maps+2),1))
        assert torch.equal(task._agent_box_assignment,maps+2*task.relation_runtime.tasks)
        # Physical roots remain in their original zones, regardless of owners.
        torch.testing.assert_close(task._box_states,task._initial_box_states)
        owned=task._owned_push_states()
        torch.testing.assert_close(task.relation_runtime.start,owned[...,:3])
        push=task.relation_runtime.tasks==1
        torch.testing.assert_close(task._tar_pos[push],(owned[...,:3]+owned.new_tensor([0,2,0]))[push])
        centers=(maps-.5)*5
        dy=task._humanoid_root_states[...,1]-centers
        assert (dy<0).all() and (dy.abs()<4).all()
        graph=graph_for_tasks(task._relation_graph_spec,task.relation_runtime.tasks)
        for agent in range(2):
            logical=graph.edge_dst[:,2*agent]-2
            actual=task._logical_box_order[ids,logical]
            assert torch.equal(actual,task._agent_box_assignment[:,agent])
        logical=task._logical_box_values(task._box_states)
        torch.testing.assert_close(logical[:,2:],owned)
        # RSI must write the assigned physical actor and preserve its fixed goal.
        for skill,family in (('doorOpen',0),('push',1)):
            selected=task.relation_runtime.tasks[:,0]==family
            env=ids[selected]
            if not len(env):continue
            task._apply_joint_rsi(env,torch.zeros_like(env),skill,task._joint_rsi_bank.sample(skill,len(env)))
            task._reset_task(env)
            if family==1:
                torch.testing.assert_close(task._push_episode_start[env,0],task._box_states[env,maps[env,0]+2,:3])
                expected=task.relation_runtime.start[env,0]+owned.new_tensor([0,2,0])
                torch.testing.assert_close(task._tar_pos[env,0],expected)
        task._reset_env_tensors(ids);task._compute_observations(ids);task._init_amp_obs(ids)
        assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task._amp_obs_buf).all()
    task._task_graph_preset='push';task._reset_envs(ids)
    # Distinguishable forces prove the force lookup follows the same owner map.
    task._box_forces.zero_();task._box_forces[:,0,1]=2.
    forces=task._zone_values(task._box_forces)
    assert torch.equal(forces[...,1]>0,maps==0)
    boxes=task._owned_push_states();hands=push_hand_targets(boxes,cfg['env']['pushBox']['size'],task._relation_cfg['push'])
    hand_forces=torch.zeros_like(hands);hand_forces[...,1]=-2.
    contact=push_hand_contact(hands,hand_forces,boxes,forces,cfg['env']['pushBox']['size'],task._relation_cfg['push'])
    assert torch.equal(contact,maps==0)
    rt=task.relation_runtime
    boxes=boxes.clone();boxes[...,1]+=.1
    rt.collect_manipulation(boxes,task._tar_pos,task.door_hinge_angle,torch.zeros_like(contact),contact)
    torch.testing.assert_close(rt.push_hand_distance,.1*contact.float())
    # A partial reset cannot change the other environment's map, goal or credit.
    before=[x[1].clone() for x in (task._agent_zone,task._tar_pos,rt.push_hand_distance)]
    task._reset_envs(ids[:1])
    for value,expected in zip((task._agent_zone,task._tar_pos,rt.push_hand_distance),before):
        torch.testing.assert_close(value[1],expected)
    task._sample_zone_assignments=original_sampler
    sampled=original_sampler(torch.arange(10000,device=task.device))
    fraction=float((sampled[:,0]==1).float().mean())
    assert .48<fraction<.52
    task._reset_envs(ids)
    for _ in range(12):
        task.pre_physics_step(torch.zeros(4,32,device=task.device));task._physics_step();task.post_physics_step()
        assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task.rew_buf).all()
    metrics=task.consume_relation_diagnostics()
    assert len([k for k in metrics if k.startswith('push/zones/')])==16
    report=dict(device=task.device,task_combinations=4,both_assignments=True,physical_placement=True,
        loco_spawn=True,rsi_owner=True,graph_owner=True,observation_owner=True,force_owner=True,
        same_tick_credit=True,partial_reset=True,swap_fraction=fraction,physics_steps=12,
        contract=task.get_open_push_contract())
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
finally:task.gym.destroy_sim(task.sim)
