"""Goal opposite faces follow goal, box yaw and owner independently."""
from isaacgym import gymapi
import torch
import pytest
import yaml
from pathlib import Path
from utils.door_reference import rotate
from utils.open_push_spec import validate_env
from env.tasks.multi_agent.open_push_reward import push_face_geometry,push_hand_targets,push_hand_contact,push_rear_xy,push_reward

ROOT=Path(__file__).resolve().parents[1]

def config(friction=1.):
    name='open_push_contact_at_friction06_goal_face.yaml' if friction==.6 else 'open_push_contact_at_friction1_goal_face.yaml'
    return yaml.safe_load((ROOT/'data/cfg/multi_agent'/name).read_text())['env']

def scene():
    boxes=torch.zeros(2,2,13);boxes[...,2]=.65;boxes[...,6]=1.
    goals=boxes[...,:3].clone();goals[...,:2]=torch.tensor([[[2.,0.],[-2.,0.]],[[0.,2.],[0.,-2.]]])
    return boxes,goals

@pytest.mark.parametrize('friction',[.6,1.])
def test_four_faces_contact_direction_and_wrong_side_rejection(friction):
    e=config(friction);validate_env(e);boxes,goals=scene()
    center,normal,tangent,half,side=push_face_geometry(boxes,e['pushBox']['size'],goals)
    assert set(side.flatten().tolist())=={0,1,2,3}
    assert (((center-boxes[...,:3])*(goals-boxes[...,:3])).sum(-1)<0).all()
    hands=push_hand_targets(boxes,e['pushBox']['size'],e['relationReward']['push'],goals)
    force=-normal[:,:,None].expand_as(hands)
    contact=lambda h,f,bf:push_hand_contact(h,f,boxes,bf,e['pushBox']['size'],e['relationReward']['push'],goals)
    assert contact(hands,force,normal*2).all()
    assert not contact(2*boxes[:,:,None,:3]-hands,force,normal*2).any()
    assert not contact(hands,-force,normal*2).any()
    assert not contact(hands,force,-normal*2).any()
    _,distance=push_rear_xy(hands,boxes,e['pushBox']['size'],goals)
    torch.testing.assert_close(distance,torch.zeros(2,2,2))
    assert e['pushBox']['mass']==20. and e['pushBox']['friction']==friction
    assert e['plane']['staticFriction']==e['plane']['dynamicFriction']==1.

@pytest.mark.parametrize('yaw',[.3,1.57,3.2,5.1])
def test_face_geometry_rotates_with_box_and_goal(yaw):
    e=config();boxes,goals=scene()
    original=push_hand_targets(boxes,e['pushBox']['size'],e['relationReward']['push'],goals)
    q=torch.tensor([0.,0.,torch.sin(torch.tensor(yaw/2)),torch.cos(torch.tensor(yaw/2))]).expand(2,2,4)
    boxes[...,3:7]=q;goals=rotate(q,goals)
    targets=push_hand_targets(boxes,e['pushBox']['size'],e['relationReward']['push'],goals)
    torch.testing.assert_close(targets,rotate(q[:,:,None].expand(-1,-1,2,-1),original))
    assert set(push_face_geometry(boxes,e['pushBox']['size'],goals)[-1].flatten().tolist())=={0,1,2,3}


def test_contact_reward_uses_goal_face_and_at_release_still_works():
    e=config();boxes,goals=scene();flags=torch.zeros(2,2,dtype=torch.bool)
    hands=push_hand_targets(boxes,e['pushBox']['size'],e['relationReward']['push'],goals)
    roots=hands.mean(-2);start=boxes[...,:3].clone()
    r=push_reward(roots,hands,boxes,goals,start,flags,e['relationReward'],e['pushBox']['size'],flags,goal_contact=True)
    torch.testing.assert_close(r['push_phi'],torch.ones(2,2))
    bad=push_reward(roots,2*boxes[:,:,None,:3]-hands,boxes,goals,start,flags,e['relationReward'],e['pushBox']['size'],flags,goal_contact=True)
    assert (bad['push_phi']<.01).all()
    boxes[...,:3]=goals
    done=push_reward(roots,hands+10,boxes,goals,start,flags,e['relationReward'],e['pushBox']['size'],flags,goal_contact=True)
    assert done['success'].all()
    assert torch.isfinite(push_hand_targets(boxes,e['pushBox']['size'],e['relationReward']['push'],goals)).all()


def test_diagonal_goal_keeps_adjacent_opposite_face_contact():
    e=config(.6);boxes=torch.zeros(1,2,13);boxes[...,2]=.65;boxes[...,6]=1.
    before=boxes[...,:3].clone();before[...,:2]=torch.tensor([1.,1.1])
    hands=push_hand_targets(boxes,e['pushBox']['size'],e['relationReward']['push'],before)
    after=before.clone();after[...,:2]=torch.tensor([1.1,1.])
    center,normal,_,_,side=push_face_geometry(boxes,e['pushBox']['size'],after,hands)
    torch.testing.assert_close(normal,torch.tensor([0.,1.,0.]).expand(1,2,3))
    _,distance=push_rear_xy(hands,boxes,e['pushBox']['size'],after)
    torch.testing.assert_close(distance,torch.zeros(1,2,2))
    forces=-normal[:,:,None].expand_as(hands)
    assert push_hand_contact(hands,forces,boxes,normal*2,e['pushBox']['size'],e['relationReward']['push'],after).all()
    # Once the goal is near+X, the orthogonal -Y face no longer guides pushing.
    after[...,:2]=torch.tensor([2.,.1])
    assert not push_hand_contact(hands,forces,boxes,normal*2,e['pushBox']['size'],e['relationReward']['push'],after).any()
