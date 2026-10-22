"""Tilt limits AT state only, including the placement saturation path."""
from copy import deepcopy
from pathlib import Path
import json
import math
from isaacgym import gymapi  # before torch
import torch
import yaml
import pytest
from env.tasks.multi_agent.open_push_reward import push_at_edge_reward, push_goal_opposite_xy_target
from utils.open_push_spec import validate_env
from utils.open_push_rsi import bank_contract

ROOT=Path(__file__).resolve().parents[2]


def config():
    env=yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/push_toy_random_at_10deg_at_state.yaml').read_text())['env']
    env['motion_file']='tokenhsi/data/dataset_open_push.yaml'
    return env


@pytest.mark.parametrize('degrees,valid',[(9.99,True),(10.,True),(10.01,False),(25.,False)])
@pytest.mark.parametrize('at_goal',[False,True])
def test_tilt_only_gates_at_state(degrees,valid,at_goal):
    env=config();validate_env(env);cfg=env['relationReward']
    boxes=torch.zeros(1,2,13);boxes[...,2]=.65;boxes[...,6]=1
    start=boxes[...,:3].clone();goals=start.clone()
    if not at_goal:goals[...,1]+=1.
    hands=torch.zeros(1,2,2,3)
    hands[...,:2]=push_goal_opposite_xy_target(boxes,goals,env['pushBox']['size']).unsqueeze(-2)
    before=push_at_edge_reward(start,hands,boxes,goals,start,env['pushBox']['size'],cfg)
    boxes[...,3]=math.sin(math.radians(degrees)/2);boxes[...,6]=math.cos(math.radians(degrees)/2)
    after=push_at_edge_reward(start,hands,boxes,goals,start,env['pushBox']['size'],cfg)
    torch.testing.assert_close(after['edge_state'][:,::2],before['edge_state'][:,::2])
    for key in ('edge_progress','edge_bonus','success','own_success'):
        torch.testing.assert_close(after[key],before[key])
    assert bool((after['edge_state'][:,1::2]>0).all())==valid
    # Returning upright restores state, including after hands are released at goal.
    if at_goal:hands+=20
    boxes[...,3:7]=boxes.new_tensor([0.,0.,0.,1.])
    restored=push_at_edge_reward(start,hands,boxes,goals,start,env['pushBox']['size'],cfg)
    torch.testing.assert_close(restored['edge_state'],before['edge_state'])


def test_push_height_gate_is_preserved():
    env=config();cfg=env['relationReward']
    boxes=torch.zeros(1,2,13);boxes[...,2]=.65;boxes[...,6]=1
    start=boxes[...,:3].clone();goals=start.clone();goals[...,1]+=1
    boxes[...,2]+=.081
    hands=torch.zeros(1,2,2,3)
    hands[...,:2]=push_goal_opposite_xy_target(boxes,goals,env['pushBox']['size']).unsqueeze(-2)
    result=push_at_edge_reward(start,hands,boxes,goals,start,env['pushBox']['size'],cfg)
    for key in ('edge_state','edge_progress','edge_bonus'):
        assert not result[key][:,::2].any()


def test_bank_preserves_candidates_and_matches_new_contract():
    env=config();data=json.loads(Path(env['openPushRSI']['bank_file']).read_text())
    old=json.loads(Path(data['at_state_tilt_reuse']['source']).read_text())
    assert data['contract']==bank_contract(env)
    data.pop('at_state_tilt_reuse');data['contract']=old['contract']
    assert data==old


@pytest.mark.parametrize('section,key',[('at','state_require_upright'),('push_at','require_tilt')])
def test_flags_require_boolean(section,key):
    env=deepcopy(config());env['relationReward'][section][key]='true'
    with pytest.raises(ValueError,match='boolean'):validate_env(env)
