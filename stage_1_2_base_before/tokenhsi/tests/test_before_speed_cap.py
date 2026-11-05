from copy import deepcopy
import pytest
import torch
from test_before_target_velocity import setup
from env.tasks.multi_agent.before_target_velocity import target_velocity_penalty
from utils.experiments import native_config,validate_experiment
from utils.unified_training import validate_unified_env
from utils.before_spec import checkpoint_contract
NAME='stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg_speed_cap'
ENV=native_config(NAME)['env'];CFG=ENV['before']['waiting_penalty']['target_velocity']

@pytest.mark.parametrize('gap,cap',[(2.,.5),(.5,.25),(0.,0.),(-1.,0.)])
@pytest.mark.parametrize('direction',[(1.,0.),(-1.,0.),(0.,1.)])
def test_only_speed_excess_not_direction(gap,cap,direction):
    s,a,g,r,e,place=setup();s.target_velocity_config=deepcopy(CFG);place(gap)
    for speed in [0.,cap/2,cap,cap+.2]:
        a[1].zero_();a[1][r,s.owners,0]=direction[0]*speed;a[1][r,s.owners,1]=direction[1]*speed
        expected=torch.full((len(r),),max(speed-cap,0.))
        torch.testing.assert_close(s.penalty(*a)[r,s.owners],expected,atol=1e-6,rtol=1e-5)

@pytest.mark.parametrize('family',['sit','climb','stack'])
def test_activation_and_complete_release(family):
    s,a,g,r,e,place=setup(family);s.target_velocity_config=deepcopy(CFG);place(2.)
    a[1].fill_(2.);a[2][...,7:10]=2.
    if family=='stack':
        assert not s.penalty(*a).any()
        a[-1][r,s.holding_edge]=True
    assert (s.penalty(*a)[r,s.owners]>0).all()
    a[-1][r,s.predecessor]=True
    assert not s.penalty(*a).any()
    a[-1].zero_();place(-.2)
    assert not s.penalty(*a).any()


def test_config_contract():
    validate_unified_env(ENV);validate_experiment(NAME,ENV,native_config(NAME,True))
    previous=native_config(NAME.replace('_speed_cap','_velocity_l2'))['env']
    assert checkpoint_contract(ENV)!=checkpoint_contract(previous)
    other=deepcopy(ENV);other['before']['waiting_penalty']['target_velocity']['mode']='sit_climb_l2_v1'
    assert other==previous
