from copy import deepcopy
import pytest
import torch
from env.tasks.multi_agent.before_wait import BeforeWait
from utils.experiments import native_config,validate_experiment
from utils.unified_training import validate_unified_env
from utils.before_spec import checkpoint_contract
from test_before_atprogress import fixture
NAME='stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2'
ENV=native_config(NAME)['env']

@pytest.mark.parametrize('family',['sit','climb','stack'])
def test_latched_speed_penalty_entry_release_reset(family):
    g,roots,boxes,sizes,goals,rows,edges,goal_ids,support=fixture(family)
    raw=torch.zeros_like(g.edge_valid);state=BeforeWait(g,len(roots),1/30,ENV['before']['waiting_penalty'],'cpu');state.reset(torch.arange(len(roots)),g,raw)
    velocity=torch.zeros_like(roots);velocity[...,0]=.3;velocity[...,2]=.4
    entity=boxes if family=='stack' else roots;index=state.source-4 if family=='stack' else state.owners
    entity[rows,index,:2]=goals[rows,state.goal,:2];radius=sizes[rows,state.support,:2].norm(dim=-1)/2+3
    entity[rows,index,0]+=radius+.01
    args=(roots,velocity,boxes,sizes,goals,raw);assert not state.penalty(*args).any()
    entity[rows,index,0]-=.02
    expected=torch.zeros(roots.shape[:2]);expected[rows,state.owners]=.5
    torch.testing.assert_close(state.penalty(*args),expected)
    entity[rows,index,0]+=100;torch.testing.assert_close(state.penalty(*args),expected)
    velocity.zero_();assert not state.penalty(*args).any();velocity[...,0]=.5
    selected=rows==0;raw[rows[selected],state.predecessor[selected]]=True;expected[0]=0
    torch.testing.assert_close(state.penalty(*args),expected);raw.zero_();torch.testing.assert_close(state.penalty(*args),expected)
    state.reset(torch.tensor([1]),g,raw);expected[1]=0;torch.testing.assert_close(state.penalty(*args),expected)
    assert not state.moving_steps.any()

def test_config_matches_virtual_except_penalty_and_at_height():
    validate_unified_env(ENV);validate_experiment(NAME,ENV,native_config(NAME,True))
    base=native_config(NAME[:-len('_wait_penalty_v2')])['env'];copy=deepcopy(ENV)
    assert copy['before'].pop('waiting_penalty')=={'radius_buffer':3.,'coefficient':1.}
    assert copy['relationReward']['success']['at_z_tolerance']==.2
    copy['relationReward']['success']['at_z_tolerance']=.001;assert copy==base
    assert checkpoint_contract(ENV)['at_z_tolerance']==.2
    assert not any(k in ENV['before'] for k in ['asymmetric_progress','continuous_progress','approach_timeout','waiting_termination'])

@pytest.mark.parametrize('value',[0,-1,True,None,float('inf'),float('nan')])
def test_invalid_penalty(value):
    env=deepcopy(ENV);env['before']['waiting_penalty']['coefficient']=value
    with pytest.raises(ValueError,match='waiting penalty'):validate_unified_env(env)
