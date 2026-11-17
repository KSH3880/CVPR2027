from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from utils.compact_bridge import NAME, choose_boundary, goal_band, progress, sample_distance, source_depth, visitors
from utils.cliff_bridge_layout import crossing_layout
from utils.cliff_bridge_spec import sample_roles_for_scene
from utils.coupled_push_spec import checkpoint_contract, check_checkpoint
from utils.experiments import native_config, validate_experiment, experiment_horizon
from env.tasks.multi_agent import square_sensor_scene as scene
from env.tasks.multi_agent.compact_rsi import CompactRsi


def test_config_isolation_and_source():
    cfg = native_config(NAME)
    validate_experiment(NAME,cfg['env'],native_config(NAME,True))
    assert experiment_horizon(NAME) == 900
    assert cfg['experiment']['transfer']['checkpoint'] == 'output/stage2/Stage2Distill900V4_00002000.pth'
    old = native_config('stage2_sensor_900_cliff_bridge_surface')
    for key in ['relationGraph','relationReward','before','stage2Policy','jointPush','stage2JointPush']:
        assert cfg['env'][key] == old['env'][key]
    assert cfg['env']['sensorScene']['bridge']['stage_size_by_people'] == [[5,6],[5.5,6.5],[6.5,7.5],[7,8]]
    weights = dict(coupled_push_contract=checkpoint_contract(cfg['env']))
    check_checkpoint(weights,cfg['env'])
    with pytest.raises(ValueError):
        check_checkpoint(weights,old['env'])


def test_distance_schedule_mixture_and_evaluation():
    spec = native_config(NAME)['env']['sensorScene']['bridge']['distance_curriculum']
    assert [progress(u,spec) for u in (0,500,1500,2500,9000)] == [0,0,.5,1,1]
    torch.manual_seed(42)
    values = sample_distance(50000,1500,spec,'cpu')
    for value,probability in ((0,.2),(.5,.6),(1,.2)):
        assert abs(float((values == value).float().mean())-probability) < .01
    assert sample_distance(100,0,spec,'cpu',True).eq(1).all()


def test_visitors_count_both_stages_and_exclude_floor_pushers():
    starts = torch.tensor([[1,1,0,2],[1,1,1,1]])
    targets = torch.tensor([[2,1,0,2],[2,2,2,2]])
    push = torch.tensor([[False,False,True,False],[False]*4])
    assert visitors(starts,targets,push).tolist() == [[2,2],[4,4]]


def test_boundary_single_person_probabilities_and_eligibility():
    spec = native_config(NAME)['env']['sensorScene']['bridge']['boundary_rsi']
    torch.manual_seed(42)
    eligible = torch.ones(50000,4,dtype=torch.bool)
    eligible[:,0] = False
    kind = choose_boundary(eligible,spec)
    assert (kind.gt(0).sum(-1) <= 1).all()
    assert kind[:,0].eq(0).all()
    selected = kind.amax(-1)
    for i,p in enumerate(spec['probabilities']):
        assert abs(float(selected.eq(i).float().mean())-p) < .01
    assert not choose_boundary(eligible,spec,True).any()
    assert not choose_boundary(torch.zeros_like(eligible),spec).any()


@pytest.mark.parametrize('alpha',[0.,.5,1.])
def test_source_rows_and_before_minimum(alpha):
    half = torch.full((500,2),3.5)
    phases = torch.zeros(500,4,dtype=torch.long)
    a = torch.full((500,),alpha)
    boundary = torch.zeros_like(phases)
    boundary[:,3] = 1
    depths = source_depth(half,phases,a,boundary)
    assert (depths <= 6.2+1e-5).all()
    assert (depths[:,3] >= .8).all()
    assert (depths[:,0]-depths[:,3] >= 1.2-1e-5).all()
    low,high = goal_band(a,torch.zeros(500),torch.full((500,),.56),torch.full((500,),5.5),3.3)
    assert low.ge(3.86-1e-5).all()
    assert (high >= low).all()


@pytest.mark.parametrize('branch,people,side',[(0,4,7.),(1,3,6.5)])
@pytest.mark.parametrize('alpha',[0.,1.])
def test_compact_layout_smallest_stages_all_spans_and_directions(branch,people,side,alpha):
    torch.manual_seed(42)
    n=192
    env=native_config(NAME)['env']
    sizes=torch.full((n,4,3),.5)
    _,_,pre,tasks,roles,push=sample_roles_for_scene(sizes,env['box']['build']['taskSizeRanges'],
        env['relationGraph']['template_probabilities'],torch.ones(n,dtype=torch.long),
        torch.full((n,),branch),torch.ones(n,dtype=torch.long))
    phases=torch.where(push,-1,0)
    bridge=torch.tensor([1.8,1.6,1.5]).expand(n,-1).clone()
    bridge[:,0] *= torch.arange(n)%3+1
    gap=bridge[:,0]+.1
    start=torch.zeros(n,2)
    destination=torch.stack((side+gap,torch.zeros(n)),-1)
    reverse=torch.arange(n)%2 == 0
    start[reverse],destination[reverse]=destination[reverse].clone(),start[reverse].clone()
    half=torch.full((n,2),side/2)
    landing=torch.stack((side/2+gap/2,torch.zeros(n)),-1)
    primary=torch.where(tasks >= 3,.6*2**.5/2,.8*2**.5/2)
    second=torch.where(tasks == 4,.8*2**.5/2,primary)
    radius=torch.stack((primary,second),-1)
    pending=torch.arange(n)
    for _ in range(32):
        payload,target,human,failed=crossing_layout(start[pending],half[pending],destination[pending],half[pending],
            landing[pending],bridge[pending],gap[pending],tasks[pending],pre[pending],radius[pending],
            phases[pending],torch.zeros(len(pending),4),(pre[pending] >= 0),
            env['sensorScene']['bridge']['curriculum'],2.5,distance_alpha=torch.full((len(pending),),alpha))
        assert torch.isfinite(human).all()
        good=~failed
        assert (((human-start[pending,None]).abs() <= half[pending,None]-.8+1e-5).all(-1) | push[pending])[good].all()
        pending=pending[failed]
        if not len(pending):
            break
    assert not len(pending), (branch,alpha,pending.tolist())


@pytest.mark.parametrize('carry',[False,True])
@pytest.mark.parametrize('present',[False,True])
@pytest.mark.parametrize('alpha',[0.,1.])
def test_cooperative_single_layout_uses_distance(carry,present,alpha):
    n=64
    cfg=native_config(NAME)['env']['sensorScene']['bridge']
    bridge=torch.tensor([1.8,1.8,1.5]).expand(n,-1)
    gap=torch.full((n,),1.9)
    source=torch.zeros(n,2)
    dest=torch.tensor([6.9,0.]).expand(n,-1)
    half=torch.full((n,2),2.5)
    landing=torch.tensor([3.45,0.]).expand(n,-1)
    radius=torch.full((n,),.6*2**.5/2)
    payload,target,human,failed=scene.crosser_layout(torch.full((n,),3),dest,half,source,half,
        radius,radius,landing,bridge,gap,torch.ones(n,dtype=torch.bool),.5,.8,cfg,
        torch.full((n,),carry),torch.full((n,),present),distance=torch.full((n,),alpha))
    assert not failed.any()
    if not carry:
        assert (2.5-human[:,0]).le(1.2+.8*alpha+1e-5).all()


def test_boundary_support_rejects_void_and_penetration():
    points=torch.tensor([[-.1,-.05,0.],[.1,-.05,0.],[.1,.05,0.],[-.1,.05,0.]])
    task=SimpleNamespace(device='cpu',_foot_body_ids=[0,1],_foot_vertices=[points,points],
        _env_origins=torch.zeros(3,3))
    task._terrain_height=lambda xy,ids: torch.where(xy[...,0] < 0,1.5,0.)
    body=torch.zeros(3,2,13)
    body[...,6]=1.
    body[0,:,0]=-.5
    body[1,:,0]=.5
    body[2,:,0]=-.5
    body[:2,:,2]=1.505
    body[2,:,2]=1.48
    supported=CompactRsi._boundary_supported(task,torch.arange(3),body)
    assert supported.tolist() == [True,False,False]


@pytest.mark.parametrize('alpha',[0.,1.])
def test_offset_entrance_keeps_approach_lanes_on_source(alpha):
    torch.manual_seed(42)
    n=128
    cfg=native_config(NAME)['env']['sensorScene']['bridge']
    source=torch.zeros(n,2)
    dest=torch.tensor([9.15,-.25]).expand(n,-1)
    start_half=torch.tensor([3.5,3.75]).expand(n,-1)
    dest_half=torch.tensor([3.75,3.5]).expand(n,-1)
    bridge=torch.tensor([1.8,1.8,1.5]).expand(n,-1)
    gap=torch.full((n,),1.9)
    landing=torch.tensor([4.45,2.02954]).expand(n,-1)
    tasks=torch.tensor([3,1,3,5]).expand(n,-1)
    pre=torch.tensor([-1,2,-1,-1]).expand(n,-1)
    phases=torch.tensor([0,0,0,-1]).expand(n,-1)
    radii=torch.full((n,4,2),.4243)
    pending=torch.arange(n)
    for _ in range(32):
        _,_,_,failed=crossing_layout(source[pending],start_half[pending],dest[pending],dest_half[pending],
            landing[pending],bridge[pending],gap[pending],tasks[pending],pre[pending],radii[pending],
            phases[pending],torch.zeros(len(pending),4),pre[pending]>=0,cfg['curriculum'],2.5,
            distance_alpha=torch.full((len(pending),),alpha))
        pending=pending[failed]
        if not len(pending): break
    assert not len(pending)
