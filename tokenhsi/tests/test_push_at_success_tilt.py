"""AT success-only tilt gate and unchanged distance rewards, with unchanged PUSH geometry."""
from pathlib import Path
from copy import deepcopy
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
    env=yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/push_toy_random_at_10deg_at_success_tilt.yaml').read_text())['env']
    env['motion_file']='tokenhsi/data/dataset_open_push.yaml'
    return env


@pytest.mark.parametrize('degrees,scale,allowed',[(0,1,True),(5,1,True),(7.5,1,True),(10,1,True),(10.01,1,False),(15,1,False),(25,1,False),(180,1,False)])
@pytest.mark.parametrize('at_goal',[False,True])
def test_shaping_and_success(degrees,scale,allowed,at_goal):
    env=config();validate_env(env);cfg=env['relationReward']
    boxes=torch.zeros(1,2,13);boxes[...,2]=.65;boxes[...,6]=1
    start=boxes[...,:3].clone();goals=start.clone()
    if not at_goal:goals[...,1]+=1
    hands=torch.full((1,2,2,3),20.)  # isolate AT completion saturation from own PUSH success
    roots=start.clone();roots[...,0]+=3.
    def reward():return push_at_edge_reward(roots,hands,boxes,goals,start,env['pushBox']['size'],cfg)
    baseline=reward()
    boxes[...,3]=math.sin(math.radians(degrees)/2);boxes[...,6]=math.cos(math.radians(degrees)/2)
    result=reward()
    assert bool(result['success'].all())==(at_goal and allowed)
    torch.testing.assert_close(result['edge_state'][:,1::2],baseline['edge_state'][:,1::2]*scale,atol=1e-6,rtol=1e-4)
    torch.testing.assert_close(result['edge_progress'][:,1::2],baseline['edge_progress'][:,1::2]*scale,atol=1e-6,rtol=1e-4)
    assert bool(result['edge_bonus'][:,1::2].any())==(at_goal and allowed)
    if at_goal and not allowed:
        for key in ('edge_state','edge_bonus'):
            assert not result[key][:,::2].any()  # no completion saturation with released hands
        assert (result['edge_progress'][:,::2]<.2).all()  # ordinary approach reward survives
    hands[...,:2]=push_goal_opposite_xy_target(boxes,goals,env['pushBox']['size']).unsqueeze(-2)
    assert reward()['own_success'][:,::2].all()  # PUSH has no angle gate
    hands.fill_(20.);boxes[...,3:7]=boxes.new_tensor([0,0,0,1])
    restored=reward()
    for key in ('edge_state','edge_progress','edge_bonus','success'):
        torch.testing.assert_close(restored[key],baseline[key])


def test_bank_and_physical_settings_unchanged():
    env=config();data=json.loads(Path(env['openPushRSI']['bank_file']).read_text())
    source=json.loads(Path(data['at_success_tilt_reuse']['source']).read_text())
    assert data['contract']==bank_contract(env)
    data.pop('at_success_tilt_reuse');data['contract']=source['contract'];assert data==source
    old=yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/push_toy_random_at_10deg.yaml').read_text())['env']
    new=deepcopy(env);new.pop('motion_file');new['relationReward']=old['relationReward'];new['openPushRSI']['bank_file']=old['openPushRSI']['bank_file']
    assert new==old  # only reward and bank path changed: no physical re-screen required


@pytest.mark.parametrize('value',[True,0,-1,float('nan'),math.pi])
def test_invalid_success_tilt_rejected(value):
    env=config();env['relationReward']['at']['success_max_tilt']=value
    with pytest.raises(ValueError,match='tilt'):validate_env(env)
