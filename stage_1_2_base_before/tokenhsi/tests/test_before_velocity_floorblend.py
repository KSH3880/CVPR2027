from copy import deepcopy
import pytest
import torch
from test_before_target_velocity import setup
from utils.experiments import native_config,validate_experiment
from utils.unified_training import validate_unified_env
from utils.before_spec import checkpoint_contract
NAME='stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg_velocity_l2_squared_floorblend'
ENV=native_config(NAME)['env'];CFG=ENV['before']['waiting_penalty']['target_velocity']

def sample(gap,vx,vy=0,family='climb',held=True,released=False):
    s,a,g,r,e,place=setup(family);s.target_velocity_config=deepcopy(CFG);place(gap)
    v=a[2][...,7:10] if family=='stack' else a[1];index=s.source-4 if family=='stack' else s.owners
    v[r,index,0]=vx;v[r,index,1]=vy
    if family=='stack' and held:a[-1][r,s.holding_edge]=True
    if released:a[-1][r,s.predecessor]=True
    return s.penalty(*a)[r,s.owners]

@pytest.mark.parametrize('vx,vy,expected',[(-.8,0,0),(-.5,3,0),(-.2,0,.09),(0,0,.25),(0,2,.25),(.5,0,1.)])
def test_far_floor_only_goalward_projection(vx,vy,expected):
    torch.testing.assert_close(sample(5,vx,vy),torch.full((4,),float(expected)),atol=1e-6,rtol=1e-5)

@pytest.mark.parametrize('family',['sit','climb','stack'])
def test_blending_and_deceleration(family):
    for gap,expected in [(4.,0.),(3.5,.045),(3.,.09),(1.5,.3025),(0.,.64)]:
        torch.testing.assert_close(sample(gap,-.8,family=family),torch.full((4,),float(expected)),atol=2e-6,rtol=1e-5)
    for gap in [5,3.5,1,0]:
        assert not sample(gap,3,4,family=family,released=True).any()
        if family=='stack':assert not sample(gap,3,4,family=family,held=False).any()

@pytest.mark.parametrize('boundary',[4.,3.])
def test_transition_continuity(boundary):
    assert (sample(boundary-1e-4,-.8,.4)-sample(boundary+1e-4,-.8,.4)).abs().max()<1e-4


def test_config_changes_only_blend():
    validate_unified_env(ENV);validate_experiment(NAME,ENV,native_config(NAME,True))
    old=native_config(NAME[:-len('_floorblend')])['env'];other=deepcopy(ENV)
    assert other['before']['waiting_penalty']['target_velocity'].pop('far_approach_blend_width')==1
    assert other==old and checkpoint_contract(ENV)!=checkpoint_contract(old)

@pytest.mark.parametrize('width',[0,-1,True,float('nan'),float('inf')])
def test_invalid_blend(width):
    env=deepcopy(ENV);env['before']['waiting_penalty']['target_velocity']['far_approach_blend_width']=width
    with pytest.raises(ValueError,match='waiting penalty'):validate_unified_env(env)
