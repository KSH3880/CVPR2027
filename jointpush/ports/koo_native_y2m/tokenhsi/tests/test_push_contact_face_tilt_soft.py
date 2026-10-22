"""AT tilt shaping must survive completion saturation and preserve CONTACT."""
from copy import deepcopy
import json
import math
from pathlib import Path
from isaacgym import gymapi  # before torch
import pytest
import torch
import yaml
from env.tasks.multi_agent.open_push_reward import contact_at_reward
from utils.open_push_spec import validate_env
from utils.open_push_rsi import bank_contract

ROOT=Path(__file__).resolve().parents[2]


def config():
    env=yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/push_contact_face_at_tilt_soft.yaml').read_text())['env']
    env['motion_file']='tokenhsi/data/dataset_open_push.yaml'
    return env


@pytest.mark.parametrize('degrees,scale,allowed',[(0,1,True),(5,1,True),(7.5,.75,True),(10,.5,True),(10.01,.499,False),(15,0,False),(30,0,False),(180,0,False)])
@pytest.mark.parametrize('at_goal',[False,True])
@pytest.mark.parametrize('contact',[False,True])
def test_tilt_before_and_after_arrival(degrees,scale,allowed,at_goal,contact):
    env=config();validate_env(env);cfg=env['relationReward']
    boxes=torch.zeros(1,2,13);boxes[...,2]=.65;boxes[...,6]=1
    goals=boxes[...,:3].clone()
    if not at_goal:goals[...,1]+=1
    roots=boxes[...,:3].clone();roots[...,1]-=2
    hands=torch.full((1,2,2,3),20.)
    contacts=torch.full((1,2),contact,dtype=torch.bool)
    def reward(c):return contact_at_reward(roots,hands,boxes,goals,contacts,env['pushBox']['size'],c)
    baseline=reward(cfg)
    boxes[...,3]=math.sin(math.radians(degrees)/2);boxes[...,6]=math.cos(math.radians(degrees)/2)
    r=reward(cfg)
    assert bool(r['success'].all())==(at_goal and allowed)
    assert bool(r['push_success'].all())==contact
    for key in ('edge_state','edge_progress'):
        torch.testing.assert_close(r[key][:,1::2],baseline[key][:,1::2]*scale,atol=1e-6,rtol=1e-4)
    assert bool(r['edge_bonus'][:,1::2].any())==(at_goal and allowed)
    # Success-only version gives exactly the same CONTACT terms at every angle.
    legacy=deepcopy(cfg)
    for k in ('tilt_full_angle','tilt_zero_angle'):legacy['contact_at'].pop(k)
    old=reward(legacy)
    for key in ('edge_state','edge_progress','edge_bonus'):
        torch.testing.assert_close(r[key][:,::2],old[key][:,::2])
    boxes[...,3:7]=boxes.new_tensor([0.,0.,0.,1.])
    for key in ('reward','success','edge_state','edge_progress','edge_bonus'):
        torch.testing.assert_close(reward(cfg)[key],baseline[key])


def test_bank_and_physics_preserved():
    env=config();data=json.loads((ROOT/env['openPushRSI']['bank_file']).read_text())
    source=json.loads((ROOT/data['contact_at_tilt_soft_reuse']['source']).read_text())
    assert data['contract']==bank_contract(env)
    data.pop('contact_at_tilt_soft_reuse');data['contract']=source['contract']
    assert data==source


@pytest.mark.parametrize('change',[{'tilt_zero_angle':.01},{'max_tilt':True},{'tilt_full_angle':float('nan')},{'tilt_zero_angle':2.}])
def test_bad_angles(change):
    env=config();env['relationReward']['contact_at'].update(change)
    with pytest.raises(ValueError):validate_env(env)
