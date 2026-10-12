"""Non-training physical-lane/owner mapping and RSI reset audit."""
from isaacgym import gymapi
import os
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
report={'training':False,'checks':{},'samples':[]}

def check():
    lanes=task._agent_lanes
    assert torch.equal(lanes.sort(-1).values,torch.arange(2,device=task.device).expand_as(lanes))
    torch.testing.assert_close(task._agent_box_assignment,lanes+2*task.relation_runtime.tasks)
    logical=task._logical_box_values(task._box_states)
    boxes=task._owner_push_boxes()
    torch.testing.assert_close(logical[:,2:],boxes)
    torch.testing.assert_close(task.relation_runtime.start[...,1],(lanes-.5)*10.)
    goal=task.relation_runtime.start.clone();goal[...,1]+=2.
    expected=torch.where((task.relation_runtime.tasks==1)[...,None],goal,task._owner_values(task.door_handle_position))
    torch.testing.assert_close(task._tar_pos,expected)
    packet=task.obs_buf[:,608:664].reshape(2,8,7)
    torch.testing.assert_close(packet[:,4:6,:3],boxes[...,:3])
    torch.testing.assert_close(packet[:,6:8,:3],expected)
    labels=task._amp_obs_buf[...,-2:].argmax(-1)
    assert (labels==task.relation_runtime.tasks[:,:,None]).all()
    assert torch.isfinite(task.obs_buf).all()

try:
    for preset in ('open','push','open_push','push_open'):
      task._task_graph_preset=preset
      for mode in ('fixed','swapped'):
        os.environ['OPEN_PUSH_LANES']=mode;task._reset_envs(ids);check()
        before=task._box_states[:,2:].clone()
        # Asymmetric physical forces must follow the assigned lane, never agent ID.
        physical_force=task._box_forces.clone();physical_force.zero_();physical_force[:,0,1]=2.
        boxes=task._owner_push_boxes();hands=push_hand_targets(boxes,cfg['env']['pushBox']['size'],cfg['env']['relationReward']['push'])
        hand_force=hands.new_zeros(hands.shape);hand_force[...,1]=-1.
        contact=push_hand_contact(hands,hand_force,boxes,task._owner_values(physical_force),cfg['env']['pushBox']['size'],cfg['env']['relationReward']['push'])
        assert torch.equal(contact,task._agent_lanes==0)
        for _ in range(12):
          task.pre_physics_step(torch.zeros(4,32,device=task.device));task._physics_step();task.post_physics_step()
          assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task.rew_buf).all()
        assert (task._box_states[:,2:,:3]-before[...,:3]).norm(dim=-1).max()<.03
        report['samples'].append({'preset':preset,'mode':mode,'lanes':task._agent_lanes.cpu().tolist()})
    report['checks']['four_tasks_both_assignments_12steps']=True
    os.environ['OPEN_PUSH_LANES']='random'
    count=0
    for _ in range(100):
      task._reset_envs(ids);check();count+=int(task._agent_lanes[:,0].sum())
    assert 60<count<140
    report['checks']['swap_fraction']=count/200
    lane=task._agent_lanes[1].clone();root=task._humanoid_root_states[1].clone();box=task._box_states[1].clone();packet=task.relation_runtime.suffix()[1].clone()
    task._reset_envs(ids[:1])
    torch.testing.assert_close(task._agent_lanes[1],lane);torch.testing.assert_close(task._humanoid_root_states[1],root)
    torch.testing.assert_close(task._box_states[1],box);torch.testing.assert_close(task.relation_runtime.suffix()[1],packet)
    report['checks']['partial_reset']=True
    # Actual mixed OPEN/PUSH RSI candidates in both owner-lane assignments.
    task._is_eval=False;task._skill=list(cfg['env']['skill']);task._skill_init_prob=torch.tensor(cfg['env']['skillInitProb'],device=task.device)
    task._joint_rsi_spec=dict(task._joint_rsi_spec,interaction_probability=1.)
    for preset in ('open','push','open_push','push_open'):
      task._task_graph_preset=preset
      seen=set()
      for _ in range(12):
        task._reset_envs(ids);check();seen.update(task._agent_lanes[:,0].cpu().tolist())
        assert task._rsi_interaction.all()
        for _ in range(12):
          task.pre_physics_step(torch.zeros(4,32,device=task.device));task._physics_step();task.post_physics_step()
          assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task._amp_obs_buf).all()
      assert seen=={0,1}
    report['checks']['four_tasks_RSI_both_assignments']=True
    task._joint_rsi_spec=cfg['env']['openPushRSI']
    report['contract']=task.get_open_push_contract()
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
finally:task.gym.destroy_sim(task.sim)
