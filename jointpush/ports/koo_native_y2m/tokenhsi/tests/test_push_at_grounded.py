from pathlib import Path
from copy import deepcopy
import math
from isaacgym import gymapi  # before torch
import torch
import yaml
import pytest
from env.tasks.multi_agent.open_push_reward import push_at_edge_reward
from utils.open_push_spec import validate_env

ROOT=Path(__file__).resolve().parents[2]


def config():
    return yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/push_toy_random_at_10deg_at_grounded.yaml').read_text())['env']


@pytest.mark.parametrize('degrees,height,valid',[(9.99,0.,True),(10.,0.,True),(10.01,0.,False),
    (0.,.079,True),(0.,.081,False),(0.,-.081,False)])
def test_grounded_gate_blocks_all_terms_and_recovers(degrees,height,valid):
    env=config();validate_env(env);cfg=env['relationReward']
    boxes=torch.zeros(1,2,13);boxes[...,6]=1;boxes[...,2]=.65
    start=boxes[...,:3].clone()
    boxes[...,2]+=height
    boxes[...,3]=math.sin(math.radians(degrees)/2);boxes[...,6]=math.cos(math.radians(degrees)/2)
    goals=boxes[...,:3].clone();roots=start.clone();hands=torch.full((1,2,2,3),20.)
    def reward():return push_at_edge_reward(roots,hands,boxes,goals,start,env['pushBox']['size'],cfg)
    r=reward()
    assert bool(r['success'].all())==valid
    if valid:
        assert (r['edge_state']==.2).all() and (r['edge_progress']==.2).all()
        assert (r['edge_bonus']==.2).all()  # hands may be released at goal
    else:
        for key in ('edge_state','edge_progress','edge_bonus','reward'):
            assert not r[key].any()
    boxes[...,3:7]=boxes.new_tensor([0.,0.,0.,1.]);boxes[...,2]=start[...,2];goals.copy_(boxes[...,:3])
    assert reward()['success'].all()
    boxes[...,3]=math.sin(math.radians(11)/2);boxes[...,6]=math.cos(math.radians(11)/2)
    assert not reward()['success'].any()  # current completion is revoked
    legacy=deepcopy(cfg);legacy['at'].pop('require_grounded')
    r=push_at_edge_reward(roots,hands,boxes,goals,start,env['pushBox']['size'],legacy)
    assert r['success'].all()  # original experiment keeps its previous behavior


def test_invalid_tilt_gates_at_approach_before_goal():
    env=config();cfg=env['relationReward']
    boxes=torch.zeros(1,2,13);boxes[...,2]=.65;boxes[...,6]=1
    start=boxes[...,:3].clone();goals=start.clone();goals[...,1]+=1.
    hands=torch.full((1,2,2,3),20.)
    r=push_at_edge_reward(start,hands,boxes,goals,start,env['pushBox']['size'],cfg)
    assert (r['edge_progress'][:,1::2]>0).all()
    boxes[...,3]=math.sin(.2/2);boxes[...,6]=math.cos(.2/2)
    r=push_at_edge_reward(start,hands,boxes,goals,start,env['pushBox']['size'],cfg)
    assert not r['edge_progress'][:,1::2].any() and not r['edge_state'][:,1::2].any()


def test_at_gate_requires_boolean():
    env=config();env['relationReward']['at']['require_grounded']='true'
    with pytest.raises(ValueError,match='boolean'):validate_env(env)
