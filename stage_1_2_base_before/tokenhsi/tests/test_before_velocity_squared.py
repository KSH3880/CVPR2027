from copy import deepcopy
import pytest
import torch
from test_before_target_velocity import setup
from utils.experiments import native_config,validate_experiment
from utils.unified_training import validate_unified_env
from utils.before_spec import checkpoint_contract
NAME='stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg_velocity_l2_squared'
ENV=native_config(NAME)['env']

@pytest.mark.parametrize('family',['sit','climb','stack'])
def test_squared_tracking_activation_release(family):
    s,a,g,r,e,place=setup(family);s.target_velocity_config=deepcopy(ENV['before']['waiting_penalty']['target_velocity']);place(4.)
    if family=='stack':
        assert not s.penalty(*a).any()
        a[-1][r,s.holding_edge]=True
    torch.testing.assert_close(s.penalty(*a)[r,s.owners],torch.full((len(r),),.25))
    velocity=a[2][...,7:10] if family=='stack' else a[1]
    index=s.source-4 if family=='stack' else s.owners
    velocity[r,index,:2]=s.target_velocity[r,s.owners]
    assert s.penalty(*a).abs().max()<1e-6
    place(-.2);velocity.zero_();velocity[r,index,0]=.3;velocity[r,index,1]=.4;velocity[r,index,2]=8
    torch.testing.assert_close(s.penalty(*a)[r,s.owners],torch.full((len(r),),.25))
    a[-1][r,s.predecessor]=True;velocity.fill_(10)
    assert not s.penalty(*a).any()
    a[-1].zero_();assert not s.penalty(*a).any()


def test_squared_and_three_meter_deceleration_config():
    validate_unified_env(ENV);validate_experiment(NAME,ENV,native_config(NAME,True))
    old=native_config(NAME[:-len('_squared')])['env'];other=deepcopy(ENV)
    other['before']['waiting_penalty']['target_velocity']['mode']='sit_climb_l2_v1'
    assert other['before']['waiting_penalty']['target_velocity']['deceleration_time']==6.
    other['before']['waiting_penalty']['target_velocity']['deceleration_time']=2.
    assert old==other and checkpoint_contract(old)!=checkpoint_contract(ENV)


@pytest.mark.parametrize('gap,speed',[(4,.5),(3,.5),(2,1/3),(1,1/6),(0,0),(-1,0)])
def test_three_meter_deceleration(gap,speed):
    s,a,g,r,e,place=setup();s.target_velocity_config=deepcopy(ENV['before']['waiting_penalty']['target_velocity']);place(gap)
    cost=s.penalty(*a)
    torch.testing.assert_close(s.target_velocity[r,s.owners].norm(dim=-1),torch.full((len(r),),float(speed)),atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(cost[r,s.owners],torch.full((len(r),),float(speed**2)),atol=1e-6,rtol=1e-5)
