from copy import deepcopy
import math
import pytest
import torch
from env.tasks.multi_agent.before_target_velocity import target_velocity_penalty
from env.tasks.multi_agent.before_wait import BeforeWait
from test_before_atprogress import fixture
from utils.experiments import native_config,validate_experiment
from utils.unified_training import validate_unified_env
from utils.before_spec import checkpoint_contract
BASE='stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg'
NAME=BASE+'_velocity_l2'
ENV=native_config(NAME)['env'];CFG=ENV['before']['waiting_penalty']['target_velocity']


def setup(family='climb'):
    g,roots,boxes,sizes,goals,rows,edges,goal_ids,support=fixture(family)
    raw=torch.zeros_like(g.edge_valid)
    s=BeforeWait(g,len(roots),1/30,ENV['before']['waiting_penalty'],'cpu');s.reset(torch.arange(len(roots)),g,raw)
    velocity=torch.zeros_like(roots)
    args=[roots,velocity,boxes,sizes,goals,raw]
    def place(offset):
        entity=boxes if family=='stack' else roots
        index=s.source-4 if family=='stack' else s.owners
        entity[rows,index,:2]=goals[rows,s.goal,:2]
        entity[rows,index,0]+=sizes[rows,s.support,:2].norm(dim=-1)/2+3+offset
    return s,args,g,rows,edges,place


@pytest.mark.parametrize('family',['sit','climb'])
def test_approach_stop_and_orbit(family):
    s,a,g,r,e,place=setup(family)
    for gap,speed in [(2,.5),(1,.5),(.5,.25),(0,0),(-.2,0)]:
        place(gap);a[1].zero_()
        p=s.penalty(*a);torch.testing.assert_close(p[r,s.owners],torch.full((len(r),),float(speed)),atol=1e-6,rtol=1e-5)
        a[1][r,s.owners,0]=-speed
        assert s.penalty(*a)[r,s.owners].abs().max()<1e-6
    # Orbital motion penalized; vertical skill movement is deliberately excluded.
    a[1][r,s.owners,1]=.3;a[1][r,s.owners,2]=2
    torch.testing.assert_close(s.penalty(*a)[r,s.owners],torch.full((len(r),),.3))
    # Departing the waiting region restores an approach target, not an all-speed latch.
    place(2);a[1].zero_();a[1][r,s.owners,0]=-.5
    assert s.penalty(*a)[r,s.owners].abs().max()<1e-6


def test_release_disables_velocity_cost_and_partial_reset():
    s,a,g,r,e,place=setup();place(-.1)
    assert not s.penalty(*a).any()
    a[-1][r,s.predecessor]=True
    assert not s.penalty(*a).any()  # release disables cost even far from actual box
    a[-1].zero_();assert not s.penalty(*a).any() # release remains latched
    target=s.target_velocity.clone();a[1][...,:2]=target
    assert s.penalty(*a).abs().max()<1e-6
    # Inside actual support's approach envelope: no interference with sitting/climbing.
    a[0][r,s.owners,:2]=a[2][r,s.support,:2];a[1].fill_(1.)
    assert not s.penalty(*a).any()
    saved=s.released[1:].clone();s.reset(torch.tensor([0]),g,a[-1])
    assert not s.released[0].any() and torch.equal(saved,s.released[1:])


def test_rotation_invariance_and_unsquared_norm():
    position=torch.tensor([[5.,2.,0.]])
    velocity=torch.tensor([[.3,.4,9.]])
    goal=torch.tensor([[0.,0.,0.]])
    size=torch.tensor([[.4,.6,.4]])
    def f(pos,vel,go):return target_velocity_penalty(pos,vel,go,go,size,torch.tensor([False]),3.,1.,CFG)[0]
    a=.7;rot=torch.tensor([[math.cos(a),-math.sin(a),0],[math.sin(a),math.cos(a),0],[0,0,1.]])
    torch.testing.assert_close(f(position,velocity,goal),f(position@rot.T,velocity@rot.T,goal@rot.T))
    torch.testing.assert_close(f(goal,velocity,goal),torch.tensor([.5]))


@pytest.mark.parametrize('boundary',[0.,1.])
def test_distance_continuity(boundary):
    values=[]
    for gap in [boundary-1e-5,boundary+1e-5]:
        s,a,g,r,e,place=setup();place(gap);values.append(s.penalty(*a))
    assert (values[0]-values[1]).abs().max()<2e-5


def test_ontop_is_identical_to_original():
    s,a,g,r,e,place=setup('stack');s.target_velocity_config=deepcopy(s.target_velocity_config);s.target_velocity_config.pop('ontop');old=BeforeWait(g,len(a[0]),1/30,native_config(BASE)['env']['before']['waiting_penalty'],'cpu');old.reset(torch.arange(len(a[0])),g,a[-1]);a[1][...,0]=.3;a[1][...,2]=.4
    for gap in [1,-.1,2]:
        place(gap);torch.testing.assert_close(s.penalty(*a),old.penalty(*a))
    a[-1][r,s.predecessor]=True
    assert not s.penalty(*a).any() and not old.penalty(*a).any()


def test_config_changes_only_penalty_and_preserves_checkpoint_distinction():
    validate_unified_env(ENV);validate_experiment(NAME,ENV,native_config(NAME,True))
    other=deepcopy(ENV);other['before']['waiting_penalty'].pop('target_velocity')
    assert other==native_config(BASE)['env']
    assert native_config(NAME,True)['params']['network']==native_config(BASE,True)['params']['network']
    assert checkpoint_contract(ENV)!=checkpoint_contract(other)


@pytest.mark.parametrize('key',['max_speed','deceleration_time','goal_buffer','arrival_fade_distance'])
@pytest.mark.parametrize('value',[0,-1,float('nan'),float('inf'),True])
def test_invalid_values(key,value):
    env=deepcopy(ENV);env['before']['waiting_penalty']['target_velocity'][key]=value
    with pytest.raises(ValueError,match='waiting penalty'):validate_unified_env(env)


def test_ontop_first_holding_box_velocity_latch_and_reset():
    s,a,g,r,e,place=setup('stack');place(2)
    payload=s.source-4
    a[1].fill_(5.)  # Human motion must not be used for ON_TOP tracking.
    assert not s.penalty(*a).any()
    # Unrelated successes, including predecessor placement, cannot activate pickup.
    a[-1][r,s.predecessor]=True
    assert not s.penalty(*a).any()
    a[-1].zero_();s.reset(torch.arange(len(a[0])),g,a[-1])
    a[-1][r,s.holding_edge]=True
    torch.testing.assert_close(s.penalty(*a)[r,s.owners],torch.full((len(r),),.5))
    a[2][r,payload,7:9]=s.target_velocity[r,s.owners]
    assert s.penalty(*a).abs().max()<1e-6
    a[-1].zero_();a[2][...,7:10]=0
    assert (s.penalty(*a)[r,s.owners]>.49).all()  # Losing hold does not switch cost off.
    place(-.1);a[2][r,payload,7]=.3;a[2][r,payload,9]=4.
    torch.testing.assert_close(s.penalty(*a)[r,s.owners],torch.full((len(r),),.3))
    a[-1][r,s.predecessor]=True
    assert not s.penalty(*a).any()  # No speed constraint after predecessor success.
    a[2][r,payload,:2]=a[2][r,s.support,:2]
    assert not s.penalty(*a).any()  # Final placement envelope fades velocity cost to zero.
    a[-1].zero_();saved=s.picked_up[1:].clone();s.reset(torch.tensor([0]),g,a[-1])
    assert not s.picked_up[0].any() and torch.equal(saved,s.picked_up[1:])
    s.reset(torch.arange(len(a[0])),g,a[-1]);place(2)
    assert not s.penalty(*a).any()


def test_ontop_requires_matching_actor_and_payload_not_other_hold():
    s,a,g,r,e,place=setup('stack');place(2)
    from utils.edge_context_spec import HOLDING
    from utils.edge_ontop_spec import batched
    rel=batched(g.edge_relation,len(a[0]));valid=batched(g.edge_valid,len(a[0]))
    a[-1][(rel==HOLDING)&valid]=True
    a[-1][r,s.holding_edge]=False
    assert not s.penalty(*a).any()


@pytest.mark.parametrize('bad',[{'activation':'current','goal_buffer':.1},{'activation':'first_holding_success','goal_buffer':0},{'activation':'first_holding_success','goal_buffer':.3}])
def test_invalid_ontop_config(bad):
    env=deepcopy(ENV);env['before']['waiting_penalty']['target_velocity']['ontop']=bad
    with pytest.raises(ValueError,match='waiting penalty'):validate_unified_env(env)


@pytest.mark.parametrize('family',['sit','climb','stack'])
def test_release_no_cost_for_any_velocity_or_distance(family):
    s,a,g,r,e,place=setup(family)
    if family=='stack':a[-1][r,s.holding_edge]=True
    place(4);s.penalty(*a)
    a[-1][r,s.predecessor]=True
    for gap in [-2,0,5]:
        place(gap);a[1].fill_(20);a[2][...,7:10]=20
        assert not s.penalty(*a).any()
        assert not s.velocity_weight.any() and not s.target_velocity.any()
        a[-1].zero_()


def test_old_saved_config_keeps_post_release_behavior():
    old=deepcopy(CFG);old.pop('after_release')
    pos=torch.tensor([[6.,0.,0.]]);vel=torch.zeros_like(pos);goal=torch.zeros_like(pos);size=torch.tensor([[.6,.6,.4]])
    cost,_,_=target_velocity_penalty(pos,vel,goal,goal,size,torch.tensor([True]),3.,1.,old)
    torch.testing.assert_close(cost,torch.tensor([.5]))
