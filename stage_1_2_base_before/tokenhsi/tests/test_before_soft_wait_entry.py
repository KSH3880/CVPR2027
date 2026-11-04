from copy import deepcopy
import pytest
import torch
from env.tasks.multi_agent.before_wait import BeforeWait
from utils.experiments import native_config,validate_experiment
from utils.unified_training import validate_unified_env
from utils.before_spec import checkpoint_contract
from test_before_atprogress import fixture
NAME='stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_soft_entry1m'
ENV=native_config(NAME)['env']

def setup(family):
    g,roots,boxes,sizes,goals,rows,edges,goal_ids,support=fixture(family)
    raw=torch.zeros_like(g.edge_valid);s=BeforeWait(g,len(roots),1/30,ENV['before']['waiting_penalty'],'cpu');s.reset(torch.arange(len(roots)),g,raw)
    velocity=torch.zeros_like(roots);velocity[...,0]=.6;velocity[...,2]=.8
    args=(roots,velocity,boxes,sizes,goals,raw)
    def place(offset):
        entity=boxes if family=='stack' else roots;index=s.source-4 if family=='stack' else s.owners
        entity[rows,index,:2]=goals[rows,s.goal,:2]
        entity[rows,index,0]+=sizes[rows,s.support,:2].norm(dim=-1)/2+3+offset
    return s,args,g,rows,edges,place

@pytest.mark.parametrize('family',['sit','climb','stack'])
def test_soft_entry_stationary_latch_and_release(family):
    s,args,g,r,e,place=setup(family)
    for offset,expected in [(1.1,0.),(1.,0.),(.75,.15625),(.5,.5),(.25,.84375),(0.,1.),(-.1,1.)]:
        place(offset);p=s.penalty(*args);torch.testing.assert_close(p[r,s.owners],torch.full_like(p[r,s.owners],expected),atol=1e-6,rtol=1e-5)
        # Neither time nor a repeated observation increases weight.
        assert torch.equal(p,s.penalty(*args))
    place(100);assert (s.penalty(*args)[r,s.owners]==1).all()
    args[1].zero_();assert not s.penalty(*args).any();args[1][...,0]=1
    chosen=r==0;args[-1][r[chosen],s.predecessor[chosen]]=True;p=s.penalty(*args);assert not p[0].any()
    args[-1].zero_();assert not s.penalty(*args)[0].any()
    before=s.entry_weight[2:].clone();s.reset(torch.tensor([1]),g,args[-1]);assert not s.entry_weight[1].any() and torch.equal(before,s.entry_weight[2:])
    assert not s.penalty(*args)[1].any()

@pytest.mark.parametrize('family',['sit','climb','stack'])
def test_inside_start_and_partial_entry_retreat(family):
    s,args,g,r,e,place=setup(family);place(-.2);assert (s.penalty(*args)[r,s.owners]==1).all()
    s.reset(torch.arange(len(args[0])),g,args[-1]);place(.5);p=s.penalty(*args);place(2.);assert torch.equal(p,s.penalty(*args))

@pytest.mark.parametrize('boundary',[0.,1.])
def test_no_threshold_jump(boundary):
    values=[]
    for offset in [boundary+1e-4,boundary-1e-4]:
        s,args,g,r,e,place=setup('climb');place(offset);values.append(s.penalty(*args)[r,s.owners])
    assert (values[0]-values[1]).abs().max()<1e-6

def test_config_and_contract():
    validate_unified_env(ENV);validate_experiment(NAME,ENV,native_config(NAME,True))
    base=native_config(NAME[:-len('_soft_entry1m')])['env'];copy=deepcopy(ENV)
    assert copy['before']['waiting_penalty'].pop('soft_entry_width')==1.;assert copy==base
    assert checkpoint_contract(ENV)!=checkpoint_contract(base)

@pytest.mark.parametrize('width',[0,-1,True,None,float('inf'),float('nan')])
def test_bad_width(width):
    env=deepcopy(ENV);env['before']['waiting_penalty']['soft_entry_width']=width
    with pytest.raises(ValueError,match='waiting penalty'):validate_unified_env(env)
