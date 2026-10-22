"""Common-face CONTACT+AT geometry, actual force gates and reward regressions."""
from copy import deepcopy
from pathlib import Path
import math
from isaacgym import gymapi
import pytest
import torch
import yaml
from env.tasks.multi_agent.open_push_reward import (
    push_face_geometry,push_rear_xy,push_hand_contact,push_contact_near,
    push_rsi_hand_targets,contact_at_reward,OpenPushRuntime)
from utils.open_push_spec import validate_env

ROOT=Path(__file__).resolve().parents[2]


def scene():
    env=yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/push_contact_face_at.yaml').read_text())['env']
    env['motion_file']='tokenhsi/data/dataset_open_push.yaml'
    boxes=torch.zeros(1,2,13);boxes[...,2]=.65;boxes[...,6]=1
    goals=boxes[...,:3].clone();goals[...,0:2]+=2
    hands=torch.tensor([[[[-.2,-.65,.8],[.2,-.65,.8]]]]).expand(1,2,2,3).clone()
    roots=boxes[...,:3].clone();roots[...,1]-=1.5
    return env,boxes,goals,hands,roots


def test_config_and_contract():
    env,*_=scene();validate_env(env)
    import json
    from utils.open_push_rsi import bank_contract
    bank=json.loads((ROOT/env['openPushRSI']['bank_file']).read_text())
    assert bank['contract']==bank_contract(env)
    assert env['pushBox']==dict(size=[1.3]*3,mass=20.,friction=.6)
    bad=deepcopy(env);bad['pushGoalContact']=False
    with pytest.raises(ValueError):validate_env(bad)


def test_diagonal_face_uses_both_hands_and_eligible_normals():
    env,b,g,h,_=scene();size=env['pushBox']['size']
    center,n,t,width=push_face_geometry(h,b,g,size)
    torch.testing.assert_close(n,torch.tensor([[[0.,1.],[0.,1.]]]))
    assert not push_rear_xy(h,b,g,size).any()
    h[...,0]=-.65;h[...,0,1]=-.2;h[...,1,1]=.2
    _,n,_,_=push_face_geometry(h,b,g,size)
    torch.testing.assert_close(n,torch.tensor([[[1.,0.],[1.,0.]]]))
    # Close hands on the front cannot select a face that pushes against the goal.
    g[...,0]=0;h[...,1]=.65
    _,n,_,_=push_face_geometry(h,b,g,size)
    torch.testing.assert_close(n,torch.tensor([[[0.,1.],[0.,1.]]]))


def test_face_choice_minimizes_worst_hand_not_average():
    env,b,g,h,_=scene()
    h[...,:2]=torch.tensor([[-.65,-.05],[.35,-1.25]])
    # Left face has a smaller mean distance, but bottom face has a smaller maximum.
    _,normal,_,_=push_face_geometry(h,b,g,env['pushBox']['size'])
    torch.testing.assert_close(normal,torch.tensor([[[0.,1.],[0.,1.]]]))


@pytest.mark.parametrize('degrees,expected',[(60.,[1.,0.]),(60.1,[0.,1.])])
def test_face_angle_cutoff(degrees,expected):
    env,b,g,h,_=scene();angle=math.radians(degrees)
    g[...,:2]=torch.tensor([math.cos(angle),math.sin(angle)])*2
    h[...,:2]=torch.tensor([[-.65,-.2],[-.65,.2]])
    _,normal,_,_=push_face_geometry(h,b,g,env['pushBox']['size'])
    torch.testing.assert_close(normal,torch.tensor(expected).expand_as(normal))


@pytest.mark.parametrize('yaw',[0.,.4,1.7])
def test_face_rotation_height_sliding_and_bounds(yaw):
    env,b,g,h,_=scene();size=env['pushBox']['size']
    h[...,0,0]=-.55;h[...,1,0]=.6;h[...,0,2]=-3;h[...,1,2]=7
    rot=torch.tensor([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])
    for v in (b,g,h):v[...,:2]=v[...,:2]@rot.T
    b[...,5]=math.sin(yaw/2);b[...,6]=math.cos(yaw/2)
    torch.testing.assert_close(push_rear_xy(h,b,g,size),torch.zeros(1,2,2),atol=1e-6,rtol=0)
    h[...,:2]+=torch.tensor([2.,0.])@rot.T
    assert (push_rear_xy(h,b,g,size)>0).any()


def test_strict_worst_hand_threshold_and_face_force_gates():
    env,b,g,h,_=scene();cfg=env['relationReward'];p=cfg['push'];size=env['pushBox']['size']
    force=torch.zeros_like(h);force[...,1]=-.5
    bf=torch.zeros(1,2,3);bf[...,1]=.5
    def contact():return push_hand_contact(h,force,b,bf,size,p,g,cfg)
    assert contact().all()  # Project onto face normal, not diagonal goal direction.
    h[...,1]-=.1;assert contact().all()
    h[...,0,1]-=.05;assert not contact().any()
    d=torch.full((1,2,2),.1)
    cfg['contact_at']['satisfaction_threshold']=math.exp(-10*.1**2)
    # Use the exact tensor result to avoid rounding at a strict boundary.
    cfg['contact_at']['satisfaction_threshold']=torch.exp(-10*d.square())[0,0,0].item()
    assert not push_contact_near(d,p,cfg).any()
    cfg['contact_at']['satisfaction_threshold']=.9
    h[...,1]=-.65;force[...,0,1]=-.499;assert not contact().any()
    force[...,0,1]=-.5;bf[...,1]=.499;assert not contact().any()


def test_reward_approach_contact_at_saturation_and_no_tilt_gate():
    env,b,g,h,r=scene();cfg=env['relationReward'];size=env['pushBox']['size']
    contact=torch.zeros(1,2,dtype=torch.bool)
    def reward():return contact_at_reward(r,h,b,g,contact,size,cfg)
    h[...,1]-=.18
    result=reward()
    torch.testing.assert_close(result['edge_state'][:,::2],torch.full((1,2),.4*math.exp(-10*.18**2)))
    assert not result['edge_bonus'].any()
    torch.testing.assert_close(result['edge_progress'][:,::2],torch.full((1,2),.1))
    contact[:]=True;result=reward()
    torch.testing.assert_close(result['edge_state'][:,::2],torch.full((1,2),.4))
    torch.testing.assert_close(result['edge_progress'][:,::2],torch.full((1,2),.2))
    assert result['push_success'].all()
    contact[:]=False;b[...,:3]=g;h+=10
    b[...,3]=math.sin(math.radians(30)/2);b[...,6]=math.cos(math.radians(30)/2)
    result=reward();assert result['success'].all() and not result['push_success'].any()
    torch.testing.assert_close(result['edge_state'],torch.full((1,4),.4))
    torch.testing.assert_close(result['edge_progress'],torch.full((1,4),.2))
    cfg['contact_at']['max_tilt']=math.radians(10)
    assert not reward()['success'].any()


def test_rsi_same_face_targets_height_and_runtime_actual_contact():
    env,b,g,h,r=scene();cfg=env['relationReward'];size=env['pushBox']['size']
    target=push_rsi_hand_targets(b,g,size,cfg)
    torch.testing.assert_close(push_rear_xy(target,b,g,size),torch.zeros(1,2,2))
    torch.testing.assert_close(target[...,2],torch.full((1,2,2),.8))
    torch.testing.assert_close((target[...,0,:2]-target[...,1,:2]).norm(dim=-1),torch.full((1,2),.4))
    runtime=OpenPushRuntime(1,env['relationGraph'],cfg,env['doorHold'],1/30,'cpu',size)
    runtime.reset(torch.tensor([0]),torch.ones(1,2,dtype=torch.long),b)
    runtime.push_contact[:]=True
    out=runtime.step(r,h,torch.zeros(1,2,3),torch.zeros(1,2),b,g)
    assert out['push_success'].all() and runtime.own_success[:,::2].all()
    runtime.push_contact[:]=False
    runtime.step(r,h,torch.zeros(1,2,3),torch.zeros(1,2),b,g)
    assert not runtime.own_success[:,::2].any()
