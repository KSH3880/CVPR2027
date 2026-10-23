"""Random PUSH geometry invariants and rigid-motion reward/credit equivalence."""
import copy
import sys
from pathlib import Path
import torch
import pytest
import yaml
from utils.open_push_scene import validate_scene,sample_boxes,sample_loco_positions
from utils.door_reference import rotate
from utils.open_push_spec import validate_env
from utils.open_push_lanes import owner_values
from env.tasks.multi_agent.open_push_reward import OpenPushRuntime,push_hand_targets,push_hand_contact,push_reward,push_debug_metrics

ROOT=Path(__file__).resolve().parents[1]


def env():return yaml.safe_load((ROOT/'data/cfg/multi_agent/open_push_push180_random_scene.yaml').read_text())['env']


def test_scene_sampling_varies_positions_yaw_and_stays_disjoint():
    e=env();validate_env(e);torch.manual_seed(71)
    boxes,direction=sample_boxes(2048,'cpu',e['pushSceneRandomization'],e['pushBox']['size'])
    assert ((boxes[:,:,:2]-boxes.new_tensor([-4.,0.])).norm(dim=-1)<=4.50001).all()
    assert ((boxes[:,0,:2]-boxes[:,1,:2]).norm(dim=-1)>=6.).all()
    torch.testing.assert_close(direction.norm(dim=-1),torch.ones(2048,2))
    torch.testing.assert_close(boxes[...,2],torch.full((2048,2),.65))
    assert boxes[...,0].std()>.5 and boxes[...,1].std()>.5
    assert direction[...,0].min()<-.9 and direction[...,0].max()>.9
    bad=copy.deepcopy(e);bad['pushSceneRandomization']['min_station_distance']=1.
    with pytest.raises(ValueError):validate_scene(bad)


def test_loco_bearings_ranges_and_body_clearance():
    e=env();boxes,_=sample_boxes(256,'cpu',e['pushSceneRandomization'],e['pushBox']['size']);boxes=boxes[:,0]
    relative=boxes.new_tensor([[0.,0.,0.],[.5,0.,.1],[-.2,0.,-.6]]).expand(256,3,3)
    target=sample_loco_positions(boxes,relative,boxes.new_full((256,),.9),e['pushBox']['size'],[1.,2.])
    distance=(target[:,:2]-boxes[:,:2]).norm(dim=-1)
    assert (distance>=.99999).all() and (distance<=2.00001).all()
    delta=target[:,:2]-boxes[:,:2];assert (delta[:,0]>0).any() and (delta[:,0]<0).any() and (delta[:,1]>0).any() and (delta[:,1]<0).any()
    inv=boxes[:,3:7].clone();inv[:,:3]*=-1
    local=rotate(inv[:,None].expand(-1,3,-1),relative+target[:,None]-boxes[:,None,:3])
    assert not ((local.abs()<.77).all(-1)).any()


def test_rotated_goal_contact_credit_reward_and_diagnostics():
    e=env();boxes=torch.zeros(1,2,13);boxes[...,2]=.65;boxes[...,6]=1
    yaw=boxes.new_tensor([[torch.pi/2,-torch.pi/2]])
    boxes[...,5]=(yaw/2).sin();boxes[...,6]=(yaw/2).cos()
    direction=rotate(boxes[...,3:7],boxes.new_tensor([0.,1.,0.]).expand(1,2,3))
    start=boxes[...,:3].clone();goals=start+direction*2
    hands=push_hand_targets(boxes,e['pushBox']['size'],e['relationReward']['push'])
    force=-direction[:,:,None].expand_as(hands)
    hit=push_hand_contact(hands,force,boxes,direction*2,e['pushBox']['size'],e['relationReward']['push']);assert hit.all()
    runtime=OpenPushRuntime(1,e['relationGraph'],e['relationReward'],e['doorHold'],1/30,'cpu',e['pushBox']['size'])
    runtime.reset(torch.tensor([0]),torch.ones(1,2,dtype=torch.long),boxes)
    runtime.reset_push_credit(torch.tensor([0]),boxes,goals)
    boxes[...,:3]+=direction*.2;runtime.accumulate_push_credit(boxes,goals,hit)
    torch.testing.assert_close(runtime.push_credit,torch.full((1,2),.2))
    metric=push_debug_metrics(hands,force,boxes,direction*2,start,e['pushBox']['size'],e['relationReward']['push'],hit,runtime.push_eligible,start,direction)
    torch.testing.assert_close(metric['box_forward_m'],torch.full((1,2),.2))
    torch.testing.assert_close(metric['bottom_edge_forward_m'],torch.full((1,2),.2),atol=1e-6,rtol=1e-5)


def test_transfer_allows_scene_only_and_preserves_training_state():
    sys.path.insert(0,str(ROOT/'scripts/multi_agent'))
    from transfer_open_push_random_scene import prepare,expected_contract,SOURCE_CONFIG,TRAIN_CONFIG
    from utils.relation_task_spec import checkpoint_metadata
    source=yaml.safe_load(SOURCE_CONFIG.read_text())['env'];target=env()
    state={k:{'test':torch.tensor([1.,2.])} for k in ('model','optimizer','running_mean_std','reward_mean_std','amp_input_mean_std')}
    state.update(epoch=3500,frame=123,amp_enabled=True,open_push_contract=expected_contract(source,'old'),relation_metadata=checkpoint_metadata(source['relationReward']),relation_experiment_config=dict(num_envs=2048,num_agents=2,num_objects=4,observation_size=684,env=source,train=yaml.safe_load(TRAIN_CONFIG.read_text())))
    result=prepare(state,source,target,'old','new',Path('/tmp/scene-test.pth'))
    for k in ('model','optimizer','running_mean_std','reward_mean_std','amp_input_mean_std','epoch','frame'):assert result[k] is state[k]
    for key in ('mass','friction'):
        bad=copy.deepcopy(target);bad['pushBox'][key]+=.1
        with pytest.raises(ValueError):prepare(state,source,bad,'old','new',Path('/tmp/scene-test.pth'))


def test_random_goal_square_separation_and_loco_exclusion():
    from utils.open_push_scene import sample_goal_offsets
    e=yaml.safe_load((ROOT/'data/cfg/multi_agent/open_push_contact_at_friction1_random_goal.yaml').read_text())['env']
    validate_env(e);torch.manual_seed(42)
    boxes,direction=sample_boxes(2048,'cpu',e['pushSceneRandomization'],e['pushBox']['size'])
    offsets=sample_goal_offsets(boxes,e['pushGoalRandomization'])
    assert (offsets[...,:2].abs()<=1.5).all() and (offsets.norm(dim=-1)>=1.).all()
    assert ((offsets*direction).sum(-1)<0).any()
    assert offsets[...,0].min()< -1.4 and offsets[...,0].max()>1.4
    relative=boxes.new_zeros(2048,1,3)
    goals=boxes[:,0,:3]+offsets[:,0]
    roots=sample_loco_positions(boxes[:,0],relative,boxes.new_full((2048,),.9),e['pushBox']['size'],[1.,2.],goals)
    assert ((roots[:,:2]-goals[:,:2]).norm(dim=-1)>=1.).all()
    bad=copy.deepcopy(e);bad['pushSceneRandomization']['min_station_distance']=6.
    with pytest.raises(ValueError):validate_env(bad)
