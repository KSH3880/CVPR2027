"""Radial goal distribution, workspace separation and source-transfer isolation."""
from isaacgym import gymapi  # before torch
from copy import deepcopy
from pathlib import Path
import math
import pytest
import torch
import yaml
from utils.open_push_scene import goal_travel_limit,sample_goal_offsets,sample_boxes,validate_scene
from utils.open_push_spec import validate_env

ROOT=Path(__file__).resolve().parents[1]

def config():
    return yaml.safe_load((ROOT/'data/cfg/multi_agent/open_push_contact_at_friction06_random_size_goal25.yaml').read_text())['env']

def test_distribution_distance_angle_z_and_seed():
    e=config();validate_env(e);spec=e['pushGoalRandomization']
    boxes=torch.zeros(10000,2,13);boxes[...,2]=torch.linspace(.55,.7,10000)[:,None]
    torch.manual_seed(42);offsets=sample_goal_offsets(boxes,spec)
    torch.manual_seed(42);torch.testing.assert_close(sample_goal_offsets(boxes,spec),offsets)
    radius=offsets.norm(dim=-1)
    assert radius.min()>=.5-1e-6 and radius.max()<=2.5+1e-6
    assert offsets[...,2].count_nonzero()==0
    assert abs((radius<1.).float().mean().item()-spec['near_probability'])<.015
    near=radius[radius<1.];far=radius[radius>=1.]
    assert abs(near.mean().item()-.75)<.015
    assert abs(far.mean().item()-1.75)<.025
    angles=torch.atan2(offsets[...,1],offsets[...,0])
    quadrants=((angles+math.pi)/(math.pi/2)).long().clamp_max(3)
    for q in range(4):assert abs((quadrants==q).float().mean().item()-.25)<.015

def test_expanded_workspace_and_size_correct_reset():
    e=config();old=deepcopy(e);old['pushSceneRandomization']['min_station_distance']=6.3
    with pytest.raises(ValueError,match='overlap'):validate_scene(old)
    sizes=torch.full((2048,2,3),1.4)
    torch.manual_seed(42)
    for _ in range(5):
        boxes,_=sample_boxes(2048,'cpu',e['pushSceneRandomization'],sizes)
        assert (boxes[:,0,:2]-boxes[:,1,:2]).norm(dim=-1).min()>=7.
        torch.testing.assert_close(boxes[...,2],sizes[...,2]/2)

@pytest.mark.parametrize('change',[{'near_probability':0.}, {'near_probability':float('nan')}, {'distance_range':[1.,.5]}, {'near_distance':2.5}, {'distance_range':[.5]}, {'extra':1}])
def test_invalid_radial_specs_rejected(change):
    spec=config()['pushGoalRandomization'];spec.update(change)
    with pytest.raises(ValueError):goal_travel_limit(spec)

def test_legacy_square_still_obeys_its_contract():
    spec={'half_extent':1.5,'min_distance':1.}
    assert goal_travel_limit(spec)==pytest.approx(math.sqrt(2)*1.5)
    offsets=sample_goal_offsets(torch.zeros(2048,2,13),spec)
    assert offsets[...,:2].abs().max()<=1.5
    assert offsets.norm(dim=-1).min()>=1.


def test_uniform_goal_1to25_and_rsi_success_margin():
    e=yaml.safe_load((ROOT/'data/cfg/multi_agent/open_push_contact_at_friction06_random_size_goal1to25.yaml').read_text())['env']
    validate_env(e)
    spec=e['pushGoalRandomization'];assert goal_travel_limit(spec)==2.5
    torch.manual_seed(42)
    offsets=sample_goal_offsets(torch.zeros(10000,2,13),spec)
    radius=offsets.norm(dim=-1)
    assert radius.min()>=1.-1e-6 and radius.max()<=2.5+1e-6
    assert abs(radius.mean().item()-1.75)<.02
    assert offsets[...,2].count_nonzero()==0
    for lo,hi in ((1.,1.5),(1.5,2.),(2.,2.5)):
        assert abs(((radius>=lo)&(radius<hi)).float().mean().item()-1/3)<.015
    angles=torch.atan2(offsets[...,1],offsets[...,0])
    for q in range(4):
        assert abs(((angles>=-math.pi+q*math.pi/2)&(angles< -math.pi+(q+1)*math.pi/2)).float().mean().item()-.25)<.015
    # Even the latest allowed RSI phase leaves more than the AT success radius.
    remaining=radius*(1-.85)
    a=e['relationReward']['contact_at']
    assert (torch.exp(-a['near_distance_scale']*remaining.square())<a['satisfaction_threshold']).all()


@pytest.mark.parametrize('bounds',[[0.,2.5],[2.5,1.],[1.],[True,2.5],[1.,float('nan')]])
def test_invalid_uniform_radial_specs_rejected(bounds):
    with pytest.raises(ValueError):goal_travel_limit({'sampling':'radial_uniform','distance_range':bounds})

