from copy import deepcopy
import pytest
import torch
from env.tasks.multi_agent.before_approach_timeout import ApproachTimeout, follower_goal_distances, apply_timeout_reset, validate_approach_timeout
from utils.experiments import native_config, validate_experiment
from utils.before_spec import checkpoint_contract
from utils.before_push_spec import validate_env
from test_before_atprogress import fixture

BASE='stage2_before_push_near50_pair50_source5_virtual_asym_goal5_goalroute'
NAME=BASE+'_approach_timeout2s'

def setup():
    timer=ApproachTimeout((2, 2),'cpu',1/30,dict(mode='first_approach_v1',seconds=2.,min_improvement_m=.01))
    d=torch.ones(2,2);g=torch.zeros(2,2,dtype=torch.bool);s=g.clone();dep=g.clone();dep[:,0]=True
    return timer,d,g,s,dep

def test_no_clock_before_gate_and_exact_2_seconds():
    t,d,g,s,dep=setup()
    for _ in range(100):assert not t.step(d,g,s,dep).any()
    g[:,0]=True
    assert not t.step(d,g,s,dep).any()  # anchor frame is t=0
    for _ in range(59):assert not t.step(d,g,s,dep).any()
    expired=t.step(d,g,s,dep);assert expired[:,0].all() and not expired[:,1].any()

def test_first_approach_disarms_even_if_agent_stops_or_retreats():
    t,d,g,s,dep=setup();g[:]=True;t.step(d,g,s,dep)
    for _ in range(40):t.step(d,g,s,dep)
    d[:,0]-=.006;t.step(d,g,s,dep);assert (t.idle[:,0]==41).all()
    d[:,0]-=.006;t.step(d,g,s,dep)
    assert t.completed[:,0].all() and (t.idle[:,0]==0).all()
    for i in range(180):
        d[:,0]=1.5 if i>60 else .988
        g[:]=bool(i%2)
        assert not t.step(d,g,s,dep).any()
    t.reset(torch.tensor([0]));g[:]=True;t.step(d,g,s,dep)
    for _ in range(60):expired=t.step(d,g,s,dep)
    assert expired[0,0] and not expired[1].any()


def test_subthreshold_motion_does_not_extend_deadline():
    t,d,g,s,dep=setup();g[:]=True;t.step(d,g,s,dep)
    for i in range(59):
        d[:,0]=.994 if i%2 else 1.01
        assert not t.step(d,g,s,dep).any()
    assert t.step(d,g,s,dep)[:,0].all()


def test_first_approach_on_deadline_counts_as_response():
    t,d,g,s,dep=setup();g[:]=True;t.step(d,g,s,dep)
    for _ in range(59):t.step(d,g,s,dep)
    d[:,0]=.98
    assert not t.step(d,g,s,dep).any() and t.completed[:,0].all()

def test_gate_reclose_cannot_reset_timer_and_success_disarms():
    t,d,g,s,dep=setup();g[:,0]=True;t.step(d,g,s,dep);g[:]=False
    s[0,0]=True;t.step(d,g,s,dep);s[:]=False
    for _ in range(59):expired=t.step(d,g,s,dep)
    assert not expired[0].any() and expired[1,0]
    t.reset(torch.tensor([1]));assert t.completed[0,0] and not t.started[1].any()
    assert not t.step(d,g,s,dep).any()

@pytest.mark.parametrize('family',['sit','climb','stack'])
def test_actor_distance_uses_predecessor_goal(family):
    graph,roots,boxes,sizes,goals,rows,edges,goal_ids,_=fixture(family)
    result=follower_goal_distances(roots,goals,graph)
    torch.testing.assert_close(result[rows,edges],torch.full((4,),5.))
    goals[rows,goal_ids,:2]=torch.tensor([3.,3.])
    torch.testing.assert_close(follower_goal_distances(roots,goals,graph)[rows,edges],torch.ones(4))
    from utils.edge_ontop_spec import select_graph
    one=select_graph(graph,0)
    torch.testing.assert_close(follower_goal_distances(roots[:1],goals[:1],one)[0,edges[0]],torch.tensor(1.))

def test_timeout_resets_entire_scene_with_terminal_mask():
    expired=torch.tensor([[False,True],[False,False]])
    reset=torch.tensor([0,1]);term=torch.tensor([0,0,1,0])
    apply_timeout_reset(expired,reset,term)
    assert reset.tolist()==[1,1] and term.tolist()==[1,1,1,0]

@pytest.mark.parametrize('bad',[0,-1,float('nan'),float('inf'),True])
def test_invalid_config(bad):
    with pytest.raises(ValueError):validate_approach_timeout(dict(mode='first_approach_v1',seconds=bad,min_improvement_m=.01))

@pytest.mark.parametrize('base_name', [BASE, BASE.replace('virtual_asym', 'virtual')])
def test_config_contract_and_original_rewards_unchanged(base_name):
    name=base_name+'_approach_timeout2s'
    cfg=native_config(name);base=native_config(base_name);env=deepcopy(cfg['env'])
    assert env['before'].pop('approach_timeout')==dict(mode='first_approach_v1',seconds=2.,min_improvement_m=.01)
    assert env==base['env'] and cfg['sim']==base['sim']
    assert 'state_rescale' not in env['before']
    assert checkpoint_contract(cfg['env'])!=checkpoint_contract(base['env'])
    validate_env(cfg['env']);validate_experiment(name,cfg['env'],native_config(name,training=True))


def test_disjoint_termination_reasons_include_overlap_and_horizon():
    from env.tasks.multi_agent.before_approach_timeout import termination_reason_counts
    fall = torch.tensor([True,False,True,False,False])
    timeout = torch.tensor([False,True,True,False,False])
    reset = torch.tensor([1,1,1,1,0])
    result = termination_reason_counts(fall,timeout,reset)
    assert result.tolist()==[1,1,1,1,4]
    assert result[:-1].sum()==result[-1]
