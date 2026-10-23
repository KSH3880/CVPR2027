"""Heterogeneous owner sizes must drive faces, heights and scene clearance."""
from isaacgym import gymapi  # Isaac Gym must precede torch
from pathlib import Path
import pytest
import torch
import yaml
from env.tasks.multi_agent.open_push_reward import push_hand_targets, push_hand_contact, push_debug_metrics
from utils.open_push_sizes import size_values, density
from utils.open_push_scene import sample_boxes, sample_loco_positions


def test_density_range_and_strict_sampling_contract():
    env=yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/open_push_contact_at_friction06_random_size.yaml').read_text())['env']
    assert env['relationGraph']['task_probabilities']=={'OPEN':0.,'PUSH':1.}
    assert env['skillDiscProb']==[.33,0.,.67]
    values=size_values(env)
    assert len(values)==31 and values[0]==1.1 and values[-1]==1.4
    assert density(env)*1.1**3==pytest.approx(12.11652)
    assert density(env)*1.4**3==pytest.approx(24.97952)
    env['pushSizeRandomization']['sampling']='reset'
    with pytest.raises(ValueError):size_values(env)


def test_mixed_sizes_contact_heights_and_corners():
    boxes=torch.zeros(1,2,13);boxes[...,6]=1
    sizes=torch.tensor([[[1.1]*3,[1.4]*3]])
    boxes[...,2]=sizes[...,2]/2
    goals=boxes[...,:3].clone();goals[...,1]+=1.5
    spec=dict(hand_lateral_offset=.2,hand_target_height=.8,hand_distance=.18,min_hand_force=.5,min_box_force=.5,max_tilt=.35,ground_tolerance=.08)
    hands=push_hand_targets(boxes,sizes,spec,goals)
    torch.testing.assert_close(hands[...,2],torch.full((1,2,2),.8))
    torch.testing.assert_close(hands[0,:,0,1],torch.tensor([-.55,-.7]))
    forces=torch.zeros_like(hands);forces[...,1]=-2
    bf=torch.zeros(1,2,3);bf[...,1]=2
    assert push_hand_contact(hands,forces,boxes,bf,sizes,spec,goals).all()
    wrong=hands.clone();wrong[...,1]-=.25
    assert not push_hand_contact(wrong,forces,boxes,bf,sizes,spec,goals).any()
    start=boxes[...,:3].clone();moved=boxes.clone();moved[...,1]+=.3
    metrics=push_debug_metrics(hands,forces,moved,bf,start,sizes,spec,torch.ones(1,2,dtype=torch.bool),torch.ones(1,2,dtype=torch.bool),goals=goals,reset_states=boxes)
    torch.testing.assert_close(metrics['bottom_edge_forward_m'],torch.full((1,2),.3))


def test_mixed_sizes_reset_on_ground_and_clearance():
    sizes=torch.tensor([[[1.1]*3,[1.4]*3]]).repeat(16,1,1)
    spec=dict(arena_center=[-4.,0.],arena_radius=4.5,min_station_distance=6.3)
    boxes,_=sample_boxes(16,'cpu',spec,sizes)
    torch.testing.assert_close(boxes[...,2],sizes[...,2]/2)
    relative=torch.zeros(16,1,3)
    roots=sample_loco_positions(boxes[:,0],relative,torch.ones(16),sizes[:,0],[1.,2.])
    assert torch.isfinite(roots).all()
