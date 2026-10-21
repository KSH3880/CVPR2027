"""Owner-relative placement, yaw covariance and strict scene contracts."""
from copy import deepcopy
from pathlib import Path
from isaacgym import gymapi
import pytest
import torch
import yaml
from utils.push_scene import validate_scene,sample_layout,forward_xy,layout_valid
from utils.open_push_spec import validate_env
from utils.open_push_rsi import bank_contract
from env.tasks.multi_agent.open_push_reward import push_hand_targets,push_hand_contact,push_debug_metrics,OpenPushRuntime

ROOT=Path(__file__).resolve().parents[2]

def config():
    env=yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/push_toy_midpoint_2m_random_scene.yaml').read_text())['env']
    env['motion_file']='tokenhsi/data/dataset_open_push.yaml'
    return env


def test_layout_distribution_and_clearance():
    env=config();validate_env(env)
    xy,q=sample_layout(2048,env,'cpu',torch.Generator().manual_seed(42))
    anchors=torch.tensor(env['pushScene']['owner_anchor_xy']).expand_as(xy)
    delta=xy-anchors;distance=delta.norm(dim=-1)
    assert (distance>=1).all() and (distance<=2).all()
    # No quadrant or yaw is tied to an owner; both cover the full circle.
    for agent in range(2):
        quadrants=(delta[:,agent,0]>0).long()+2*(delta[:,agent,1]>0).long()
        assert (torch.bincount(quadrants,minlength=4)>200).all()
        direction=forward_xy(q)[:,agent]
        quadrants=(direction[:,0]>0).long()+2*(direction[:,1]>0).long()
        assert (torch.bincount(quadrants,minlength=4)>200).all()
    yaw=2*torch.atan2(q[...,2],q[...,3])
    doors=xy.new_tensor([[1.45,-2.5],[1.45,2.5]]).expand_as(xy)
    assert layout_valid(xy,yaw,anchors,doors,env['pushBox']['size'],2.,2.2).all()


@pytest.mark.parametrize('yaw',[0.,.7,1.57079632679,3.14159265359,-1.2])
def test_rotated_contact_credit_reward_and_diagnostics(yaw):
    env=config();boxes=torch.zeros(1,2,13);boxes[...,2]=.65
    boxes[...,3+2]=torch.sin(torch.tensor(yaw/2));boxes[...,6]=torch.cos(torch.tensor(yaw/2))
    direction=torch.cat((forward_xy(boxes[...,3:7]),torch.zeros(1,2,1)),-1)
    goals=boxes[...,:3]+2*direction;start=boxes[...,:3].clone()
    rt=OpenPushRuntime(1,env['relationGraph'],env['relationReward'],env['doorHold'],1/30,'cpu',env['pushBox']['size'],env['manipulationContact'])
    ids=torch.arange(1);rt.reset(ids,torch.ones(1,2,dtype=torch.long),boxes)
    # Pre-advanced RSI positions change remaining distance, never fabricate hand credit.
    boxes[...,:3]+=.8*direction
    rt.initialize_manipulation(ids,boxes,goals,torch.zeros(1,2))
    assert not rt.push_hand_distance.any()
    reset=boxes[...,:3].clone()
    for step in range(6):
        boxes[...,:3]+=.2*direction
        hands=push_hand_targets(boxes,env['pushBox']['size'],env['relationReward']['push'])
        hf=-2*direction[:,:,None].expand_as(hands);bf=2*direction
        hit=push_hand_contact(hands,hf,boxes,bf,env['pushBox']['size'],env['relationReward']['push'])
        assert hit.all()
        rt.push_contact[:]=hit;rt.clean_push_contact[:]=hit
        rt.collect_manipulation(boxes,goals,torch.zeros(1,2),torch.zeros_like(hit),hit)
        reward=rt.step(hands.mean(-2),hands,torch.zeros(1,2,3),torch.zeros(1,2),boxes,goals)
    assert reward['success'].all()
    torch.testing.assert_close(rt.push_hand_distance,torch.full((1,2),1.2),atol=1e-5,rtol=1e-5)
    metrics=push_debug_metrics(hands,hf,boxes,bf,start,env['pushBox']['size'],env['relationReward']['push'],hit,rt.push_eligible,reset,direction)
    for key,target in [('box_forward_m',2.),('bottom_edge_forward_m',2.),('since_reset_forward_m',1.2),('since_reset_bottom_edge_forward_m',1.2)]:
        torch.testing.assert_close(metrics[key],torch.full((1,2),target),atol=1e-5,rtol=1e-5)
    boxes[...,:3]-=.3*direction
    rt.collect_manipulation(boxes,goals,torch.zeros(1,2),torch.zeros_like(hit),torch.zeros_like(hit))
    torch.testing.assert_close(rt.push_hand_distance,torch.full((1,2),.9),atol=1e-5,rtol=1e-5)


def test_only_layout_changes_and_screened_bank_reused():
    env=config();old=yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/push_toy_midpoint_2m_zone_swap.yaml').read_text())['env']
    old['motion_file']=env['motion_file']
    assert bank_contract(old)==bank_contract(env)
    new=deepcopy(env);new.pop('pushScene');new['box']['reset']=old['box']['reset']
    assert new==old
    assert env['pushBox']['size']==[1.3,1.3,1.3]


@pytest.mark.parametrize('change',[{'mode':'other'},{'path_separation':1.},{'owner_anchor_xy':[[0,0]]}])
def test_reject_bad_scene(change):
    env=config();env['pushScene'].update(change)
    with pytest.raises(ValueError):validate_scene(env)
