"""Regressions for body-assisted motion, recontact, RSI and clean release."""
from pathlib import Path
import pytest
import torch
import yaml

from utils.manipulation_contact import collision_samples, nonhand_contact, validate_contact
from env.tasks.multi_agent.open_push_reward import OpenPushRuntime, push_reward

ROOT=Path(__file__).resolve().parents[1]


def scene(task=1,n=2):
    e=yaml.safe_load((ROOT/'data/cfg/multi_agent/push_toy.yaml').read_text())['env']
    rt=OpenPushRuntime(n,e['relationGraph'],e['relationReward'],e['doorHold'],1/30,'cpu',
                       e['pushBox']['size'],e['manipulationContact'])
    boxes=torch.zeros(n,2,13);boxes[...,2]=.65;boxes[...,6]=1
    roots=boxes[...,:3].clone();hands=torch.zeros(n,2,2,3)
    handles=torch.zeros(n,2,3);angles=torch.zeros(n,2)
    goals=boxes[...,:3].clone();goals[...,1]=1.5
    ids=torch.arange(n);rt.reset(ids,torch.full((n,2),task),boxes)
    rt.initialize_manipulation(ids,boxes,goals,angles)
    return e,rt,roots,hands,handles,angles,boxes,goals


def tick(rt,roots,hands,handles,angles,boxes,goals,valid,body=False):
    rt.body_contact[:]=body
    rt.clean_push_contact[:]=valid and not body
    rt.clean_open_contact[:]=valid and not body
    rt.push_contact[:]=valid;rt.contact[:]=valid and not body;rt.posture[:]=True
    rt.collect_manipulation(boxes,goals,angles,
        rt.clean_open_contact & (rt.tasks==0),rt.clean_push_contact & (rt.tasks==1))
    return rt.step(roots,hands,handles,angles,boxes,goals)


def test_body_movement_cannot_be_claimed_by_later_hand_contact():
    e,rt,roots,hands,handles,angles,boxes,goals=scene()
    boxes[...,1]=1.2
    result=tick(rt,roots,hands,handles,angles,boxes,goals,False,True)
    assert (result['progress']==0).all() and not rt.push_eligible.any()
    boxes[...,1]=1.4
    result=tick(rt,roots,hands,handles,angles,boxes,goals,True)
    assert rt.push_eligible.all() and not result['success'].any()
    torch.testing.assert_close(rt.push_credited,torch.full((2,2),.2))
    assert (result['progress']<.03).all()


def test_contact_then_body_push_does_not_finish():
    e,rt,roots,hands,handles,angles,boxes,goals=scene()
    boxes[...,1]=.1;tick(rt,roots,hands,handles,angles,boxes,goals,True)
    assert rt.push_eligible.all()
    boxes[...,1]=1.5
    result=tick(rt,roots,hands,handles,angles,boxes,goals,True,True)
    assert not result['success'].any() and (result['progress']==0).all()
    result=tick(rt,roots,hands,handles,angles,boxes,goals,True)
    assert not result['success'].any()


def test_rsi_credit_and_clean_completion_allow_hand_release():
    e,rt,roots,hands,handles,angles,boxes,goals=scene()
    boxes[...,1]=1.275
    rt.initialize_manipulation(torch.arange(2),boxes,goals,angles)
    assert (rt.push_credited==1.275).all() and not rt.push_eligible.any()
    boxes[...,1]=1.4
    result=tick(rt,roots,hands,handles,angles,boxes,goals,True)
    assert result['success'].all()
    hands[:]=20
    result=tick(rt,roots,hands,handles,angles,boxes,goals,False)
    assert result['success'].all() and (result['state']<.4).all()
    assert (result['progress']==.2).all() and (result['bonus']==.2).all()
    # Current geometric success, not a permanent latch after leaving the goal.
    boxes[...,1]=1.
    result=tick(rt,roots,hands,handles,angles,boxes,goals,False)
    assert not result['success'].any() and not rt.push_completed.any()


def test_frontier_blocks_oscillation_and_partial_reset_preserves_other_scene():
    e,rt,roots,hands,handles,angles,boxes,goals=scene()
    boxes[...,1]=.1;tick(rt,roots,hands,handles,angles,boxes,goals,True)
    credit=rt.push_credited.clone()
    boxes[...,1]=0;tick(rt,roots,hands,handles,angles,boxes,goals,True)
    boxes[...,1]=.1;tick(rt,roots,hands,handles,angles,boxes,goals,True)
    torch.testing.assert_close(rt.push_credited,credit)
    rt.reset(torch.tensor([0]),torch.ones(1,2,dtype=torch.long),boxes[:1])
    assert not rt.push_credited[0].any() and not rt.push_hand_distance[0].any()
    torch.testing.assert_close(rt.push_credited[1],credit[1])


def test_open_body_opening_cannot_be_reclaimed_by_holding_handle():
    e,rt,roots,hands,handles,angles,boxes,goals=scene(task=0)
    angles[:]=1.6
    result=tick(rt,roots,hands,handles,angles,boxes,goals,True,True)
    assert (result['progress']==0).all()
    for _ in range(160):
        result=tick(rt,roots,hands,handles,angles,boxes,goals,True)
    assert not result['success'].any() and not rt.elapsed.any()


def test_open_rsi_angle_preserved_but_body_support_resets_hold():
    e,rt,roots,hands,handles,angles,boxes,goals=scene(task=0)
    angles[:]=1.4;rt.initialize_manipulation(torch.arange(2),boxes,goals,angles)
    torch.testing.assert_close(rt.open_credited,angles)
    angles[:]=1.6
    for _ in range(160):
        result=tick(rt,roots,hands,handles,angles,boxes,goals,True)
    assert result['success'].all()
    result=tick(rt,roots,hands,handles,angles,boxes,goals,True,True)
    assert not rt.elapsed.any() and not result['success'].any()
    result=tick(rt,roots,hands,handles,angles,boxes,goals,False)
    assert not result['success'].any()


def test_body_proxy_excludes_floor_only_force_and_distant_contacts():
    e,rt,roots,hands,handles,angles,boxes,goals=scene()
    points=torch.zeros(2,2,1,3);points[...,1]=-.68;points[...,2]=.8
    forces=torch.zeros_like(points);forces[...,2]=100
    target_force=torch.zeros(2,2,3);target_force[...,1]=10
    def check():
        return nonhand_contact(points,torch.tensor([.1]),forces,boxes,[1.3]*3,
                               target_force,e['manipulationContact'])[0]
    assert not check().any()
    forces[...,1]=-10;assert check().all()
    points[...,1]=-2;assert not check().any()
    points[...,1]=-.68;target_force[:]=0;assert not check().any()
    # Yaw rotates the proximity test with the object.
    target_force[...,0]=10;boxes[...,5:7]=2**-.5
    points[...,0]=.68;points[...,1]=0;assert check().all()


def test_samples_cover_collision_bodies_and_exclude_hands():
    samples=collision_samples(ROOT/'data/assets/mjcf/phys_humanoid_v3.xml')
    names={s[0] for s in samples}
    assert len(names)==13 and 'right_hand' not in names and 'left_hand' not in names
    assert {'pelvis','torso','head','left_thigh','right_shin','left_foot'}<=names
    assert all(radius>0 for _,_,radius in samples)


@pytest.mark.parametrize('bad', [float('nan'),-1,True,0])
def test_bad_contact_parameters_rejected(bad):
    with pytest.raises(ValueError):
        validate_contact(dict(body_force_threshold=bad,body_distance_margin=.02,open_credit_tolerance=.02))
