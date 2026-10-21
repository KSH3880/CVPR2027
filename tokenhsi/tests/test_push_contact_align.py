"""CONTACT surface, positioning and completion regressions; AT stays independent."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import math
from isaacgym import gymapi  # before torch
import pytest
import torch
import yaml
from env.tasks.multi_agent.open_push_reward import (
    push_at_edge_reward, push_contact_shaping, push_rsi_hand_targets, OpenPushRuntime)
from utils.open_push_spec import validate_env
from utils.open_push_rsi import bank_contract

ROOT=Path(__file__).resolve().parents[2]
NAME='push_toy_contact_align_at'


def scene():
    env=yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent'/f'{NAME}.yaml').read_text())['env']
    env['motion_file']='tokenhsi/data/dataset_open_push.yaml'
    boxes=torch.zeros(1,2,13);boxes[...,2]=.65;boxes[...,6]=1
    start=boxes[...,:3].clone();goals=start.clone();goals[...,1]+=2
    roots=start.clone();roots[...,1]-=1.2
    hands=push_rsi_hand_targets(boxes,goals,env['pushBox']['size'],env['relationReward'])
    return env,roots,hands,boxes,goals,start


def reward(env,roots,hands,boxes,goals,start):
    return push_at_edge_reward(roots,hands,boxes,goals,start,env['pushBox']['size'],env['relationReward'])


def shaping(env,roots,hands,boxes,goals):
    return push_contact_shaping(roots,hands,boxes,goals,env['pushBox']['size'],env['relationReward'])


def test_surface_sliding_and_midpoint_false_positive():
    env,roots,hands,boxes,goals,start=scene()
    # Any two points along the selected rear face are valid, with unrestricted hand Z.
    hands[:,:,0,0]=-.55;hands[:,:,1,0]=-.35;hands[...,2]=4
    r=reward(env,roots,hands,boxes,goals,start)
    assert r['own_success'][:,::2].all()
    torch.testing.assert_close(r['edge_state'][:,::2],torch.full((1,2),.2))
    # Hands straddle the target midpoint but are both outside the finite face.
    hands[:,:,0,0]=2;hands[:,:,1,0]=-2
    assert not reward(env,roots,hands,boxes,goals,start)['own_success'][:,::2].any()
    old=deepcopy(env);old['relationReward']['push_at'].pop('contact_shaping')
    assert reward(old,roots,hands,boxes,goals,start)['own_success'][:,::2].all()


def test_one_hand_far_has_low_state_and_no_success_without_force_gate():
    env,roots,hands,boxes,goals,start=scene()
    assert reward(env,roots,hands,boxes,goals,start)['own_success'][:,::2].all()
    hands[:,:,0,1]-=1
    r=reward(env,roots,hands,boxes,goals,start)
    assert not r['own_success'][:,::2].any()
    assert (r['edge_state'][:,::2]<.001).all()


@pytest.mark.parametrize('distance',[.9,1.5,2.5])
def test_positioning_and_near_gate(distance):
    env,roots,hands,boxes,goals,start=scene()
    progress=[];align=[];gate=[]
    for xy in ((0,-distance),(distance,0),(0,distance)):
        roots[...,:2]=roots.new_tensor(xy)
        _,p,a,g,_=shaping(env,roots,hands,boxes,goals)
        progress.append(p);align.append(a);gate.append(g)
    assert (progress[0]>progress[1]).all() and (progress[1]>progress[2]).all()
    torch.testing.assert_close(align[0],torch.ones_like(align[0]))
    torch.testing.assert_close(align[2],torch.full_like(align[2],math.exp(-4)))
    torch.testing.assert_close(gate[0],torch.ones_like(gate[0]))
    if distance>=2:
        torch.testing.assert_close(gate[2],torch.ones_like(gate[2]))
    elif distance<=1:
        torch.testing.assert_close(gate[2],align[2])
    else:
        assert (gate[2]>align[2]).all() and (gate[2]<1).all()


def test_progress_rewards_repositioning_without_contact():
    env,roots,hands,boxes,goals,start=scene();hands+=10
    roots[...,1]=1.2;before=reward(env,roots,hands,boxes,goals,start)
    roots[...,1]=-1.2;after=reward(env,roots,hands,boxes,goals,start)
    assert (after['edge_progress'][:,::2]>before['edge_progress'][:,::2]).all()
    torch.testing.assert_close(after['edge_state'],before['edge_state'])
    assert not after['own_success'].any()


@pytest.mark.parametrize('degrees',[0,10,10.1,25])
@pytest.mark.parametrize('at_goal',[False,True])
def test_at_unchanged_and_completion_saturation(degrees,at_goal):
    env,roots,hands,boxes,goals,start=scene();hands+=10
    if at_goal:boxes[...,:3]=goals
    boxes[...,3]=math.sin(math.radians(degrees)/2);boxes[...,6]=math.cos(math.radians(degrees)/2)
    old=deepcopy(env);old['relationReward']['push_at'].pop('contact_shaping')
    new_r=reward(env,roots,hands,boxes,goals,start);old_r=reward(old,roots,hands,boxes,goals,start)
    for key in ('edge_state','edge_progress','edge_bonus','own_success'):
        torch.testing.assert_close(new_r[key][:,1::2],old_r[key][:,1::2])
    torch.testing.assert_close(new_r['success'],old_r['success'])
    if at_goal and degrees<=10:
        for key in ('edge_state','edge_progress','edge_bonus'):
            torch.testing.assert_close(new_r[key],torch.full((1,4),.2))
        assert not new_r['own_success'][:,::2].any()  # AT saturates CONTACT, not own success
    else:
        assert not new_r['edge_bonus'][:,::2].any()


@pytest.mark.parametrize('yaw',[.35,1.2,2.4])
def test_joint_rotation_invariance(yaw):
    env,roots,hands,boxes,goals,start=scene();hands[:,:,0,0]=.5;hands[:,:,1,0]=-.4
    baseline=reward(env,roots,hands,boxes,goals,start)
    mat=torch.tensor([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])
    for x in (roots,hands,boxes,goals,start):x[...,:2]=x[...,:2]@mat.T
    boxes[...,5]=math.sin(yaw/2);boxes[...,6]=math.cos(yaw/2)
    rotated=reward(env,roots,hands,boxes,goals,start)
    for key in ('edge_state','edge_progress','edge_bonus','success'):
        torch.testing.assert_close(rotated[key],baseline[key])


def test_degenerate_directions_finite_and_height_gate():
    env,roots,hands,boxes,goals,start=scene();roots[...,:2]=boxes[...,:2]
    _,p,a,g,_=shaping(env,roots,hands,boxes,goals)
    assert torch.isfinite(p).all() and not a.any()
    goals[...,:2]=boxes[...,:2]
    assert torch.isfinite(reward(env,roots,hands,boxes,goals,start)['reward']).all()
    goals[...,1]+=2;boxes[...,2]+=.1
    r=reward(env,roots,hands,boxes,goals,start)
    assert not r['edge_state'][:,::2].any() and not r['edge_progress'][:,::2].any()
    assert not r['own_success'][:,::2].any()


def test_runtime_owner_sharing_and_partial_reset():
    env,roots,hands,boxes,goals,start=scene()
    tensors=[x.repeat(2,*([1]*(x.ndim-1))) for x in (roots,hands,boxes,goals)]
    roots,hands,boxes,goals=tensors;hands[:,1]+=10
    rt=OpenPushRuntime(2,env['relationGraph'],env['relationReward'],env['doorHold'],1/30,'cpu',env['pushBox']['size'],env['manipulationContact'])
    ids=torch.arange(2);rt.reset(ids,torch.ones(2,2,dtype=torch.long),boxes)
    result=rt.step(roots,hands,torch.zeros_like(roots),torch.zeros(2,2),boxes,goals)
    local=(result['edge_state']+result['edge_progress']+result['edge_bonus']).reshape(2,2,2).sum(-1)
    torch.testing.assert_close(result['reward'],.9*local+.1*local.flip(-1)/2)
    before=rt.own_success[1].clone();rt.reset(ids[:1],torch.ones(1,2,dtype=torch.long),boxes[:1])
    torch.testing.assert_close(rt.own_success[1],before)
    assert not rt.own_success[0].any()


def test_config_contract_and_candidate_reuse():
    env,*_=scene();validate_env(env)
    bank=json.loads((ROOT/env['openPushRSI']['bank_file']).read_text())
    assert bank['contract']==bank_contract(env)
    provenance=bank.pop('contact_align_reuse');source_path=ROOT/provenance['source']
    assert hashlib.sha256(source_path.read_bytes()).hexdigest()==provenance['source_sha256']
    source=json.loads(source_path.read_text());bank['contract']=source['contract'];assert bank==source
    old=yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/push_toy_random_at_10deg_at_success_tilt.yaml').read_text())['env']
    current=deepcopy(env);current.pop('motion_file');current['relationReward']['push_at'].pop('contact_shaping')
    current['openPushRSI']['bank_file']=old['openPushRSI']['bank_file'];assert current==old


@pytest.mark.parametrize('key,value',[
    ('alignment_weight',True),('alignment_weight',0),('alignment_weight',1),
    ('alignment_sigma2',0),('alignment_sigma2',float('nan')),
    ('gate_full_distance',-1),('gate_full_distance',2),('gate_start_distance',float('inf')),
    ('mode','unknown'),('unknown',1)])
def test_invalid_shaping_rejected(key,value):
    env,*_=scene();env['relationReward']['push_at']['contact_shaping'][key]=value
    with pytest.raises(ValueError,match='CONTACT'):validate_env(env)


def test_shaping_requires_goal_reference():
    env,*_=scene();env['relationReward']['push_at']['hand_reference']='box_rear_midpoint_xy'
    with pytest.raises(ValueError,match='CONTACT'):validate_env(env)
