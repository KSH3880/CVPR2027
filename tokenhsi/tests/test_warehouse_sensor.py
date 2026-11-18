from copy import deepcopy

import pytest
import torch

from env.tasks.multi_agent import square_sensor as sensor
from utils.before_goal_sampling import route_distances
from utils.before_spec import task_objects
from utils.coupled_push_spec import checkpoint_contract
from utils import stage2_spec
from utils.experiments import native_config, validate_experiment, experiment_horizon
from utils.sensor_stage2_spec import validate_sensor as validate_env
from utils.warehouse_sensor import opening_clearance, sample_roles_graph, sample_layout, walls

NAME = 'stage2_sensor_900_warehouse'


def environment():
    return native_config(NAME)['env']


def test_contract_separates_old_experiment_and_keeps_learning_terms():
    env = environment()
    old = native_config('stage2_sensor_900')['env']
    validate_env(env)
    for key in ('before','relationGraph','jointPush','relationReward','box','templateRsi','stage2Policy'):
        assert env[key] == old[key]
    assert list(env['sensorScene']['mixture'].values()) == [.25]*4
    assert checkpoint_contract(env) != checkpoint_contract(old)
    validate_env(old)
    train = native_config(NAME, True)
    old_train = native_config('stage2_sensor_900', True)
    train['params']['config']['name'] = old_train['params']['config']['name']
    assert train == old_train
    assert experiment_horizon(NAME) == 900
    for name, wrong_env in ((NAME,old), ('stage2_sensor_900',env)):
        with pytest.raises(ValueError,match='environment differs'):
            validate_experiment(name,wrong_env,native_config(name,True),evaluation=True)
    changed = deepcopy(env)
    changed['sensorScene']['warehouse']['wall_box_gap'] = .8
    with pytest.raises(ValueError):
        validate_env(changed)


def test_one_pusher_and_at_most_one_before_pair():
    torch.manual_seed(1)
    env = environment()
    sizes = torch.tensor([.45,.45,.4]).expand(8192,4,3)
    graph, pre, tasks, roles = sample_roles_graph(sizes,env['box']['build']['taskSizeRanges'],
        env['relationGraph']['template_probabilities'],'random_scenario',.5)
    graph = stage2_spec.compose_graph(tasks,graph,pre,8,False)
    rows = torch.arange(len(tasks))
    assert ((tasks == 5).sum(-1) == 1).all()
    assert (tasks[rows,roles[:,0]] == 5).all()
    assert (pre[rows,roles[:,0]] == -1).all()
    assert ((pre >= 0).sum(-1) <= 1).all()
    assert abs(float((pre >= 0).any(-1).float().mean())-.75) < .025
    assert not (graph.prereq_mask.any(-1) & (graph.edge_owner == roles[:,:1])).any()
    row, follower = (pre >= 0).nonzero(as_tuple=True)
    assert (tasks[row,pre[row,follower]] == 3).all()
    assert set(tasks[row,follower].tolist()) == {1,2,4}


def test_layout_starts_outside_and_keeps_sources_inside_and_near_constraints():
    torch.manual_seed(42)
    env = environment()
    n = 256
    sizes = torch.tensor([.45,.45,.4]).expand(n,4,3).clone()
    support = torch.tensor([.6,.6,.4]).expand_as(sizes)
    blocker = torch.tensor([2.,1.8,1.5]).expand(n,3)
    half = torch.tensor([2.,2.5]).expand(n,2)
    _,pre,tasks,roles = sample_roles_graph(sizes,env['box']['build']['taskSizeRanges'],
        env['relationGraph']['template_probabilities'],'random_scenario',.5)
    near = (torch.rand(n,4) < .5) & (pre >= 0)
    result = sample_layout(tasks,pre,roles,sizes,support,half,blocker,near,env['sensorScene']['warehouse'],16)
    assert int(result['valid'].sum()) > .8*n
    accepted = result['valid']
    roots,source,goal = [result[key][accepted] for key in ('root','source','goal')]
    selected = tasks[accepted]
    rows = torch.arange(len(roots))
    pusher = roles[accepted,0]
    assert (roots[...,0] < -.75).all()
    objects = task_objects(pre[accepted],selected)[:,:4] & (selected != 5)
    assert (source[...,0][objects] > .4).all()
    assert (source[...,0][objects] < 3.2).all()
    assert (source[...,1].abs()[objects] < 2.3).all()
    assert torch.allclose(source[rows,pusher,0],goal[rows,pusher,0])
    assert (goal[rows,pusher,1].abs() > blocker[accepted,1]-.04+.35-.001).all()
    row,follower = (pre[accepted] >= 0).nonzero(as_tuple=True)
    leader = pre[accepted][row,follower]
    first,second = route_distances(roots[row,leader],source[row,leader],goal[row,leader],
        roots[row,follower],source[row,follower],selected[row,follower])
    assert (second[near[accepted][row,follower]] < first[near[accepted][row,follower]]-.0005).all()
    assert (roots[row,follower]-goal[row,leader]).norm(dim=-1).min() >= 1.


def test_entrance_wall_gap_and_height_sensor_change_when_box_moves_sideways():
    half = torch.tensor([[2.5,2.5]])
    centers,extent,starts,ends = walls(half,.4,2.5,torch.tensor([1.72]))
    assert centers.shape == extent.shape == (1,5,3)
    assert starts.shape == ends.shape == (1,5,2)
    q = torch.tensor([0.,0.,0.,1.]).expand(1,6,4)
    obstacle = torch.tensor([[[-3.94,0.,.75]]])
    centers = torch.cat((centers,obstacle),1)
    extent = torch.cat((extent,torch.tensor([[[1.,.9,.75]]])),1)
    points = torch.tensor([[[-2.7,0.],[-2.7,1.5],[-3.94,0.]]])
    closed = sensor.vertical_height(points,centers,q,extent,torch.zeros(1))
    assert torch.allclose(closed,torch.tensor([[0.,2.5,1.5]]))
    centers[:,-1,1] = 3.
    opened = sensor.vertical_height(points,centers,q,extent,torch.zeros(1))
    assert torch.allclose(opened,torch.tensor([[0.,2.5,0.]]))


def test_opening_clearance_uses_actual_box_rotation():
    q = torch.tensor([[0.,0.,0.,1.],[0.,0.,2**-.5,2**-.5]])
    clearance = opening_clearance(torch.full((2,),2.),q,torch.tensor([[2.,1.,1.]]).expand(2,-1),
        torch.zeros(2),torch.ones(2))
    assert torch.allclose(clearance,torch.tensor([1.,.5]),atol=1e-6)


def test_crowded_two_ontop_and_carry_graph_keeps_roles_while_retrying():
    torch.manual_seed(413)
    n = 64
    tasks = torch.tensor([[4,3,4,5]]).expand(n,-1)
    pre = torch.full_like(tasks,-1)
    roles = torch.tensor([[3,0,1,2]]).expand(n,-1)
    sizes = torch.tensor([[[.2,.5,.45],[.55,.6,.6],[.55,.35,.55],[2.2,2.1,1.45]]]).expand(n,-1,-1)
    support = torch.tensor([[[.7,.7,.5],[.6,.45,.35],[.8,.65,.35],[.55,.75,.25]]]).expand_as(sizes)
    half = torch.full((n,2),2.)
    blocker = sizes[:,3]
    near = torch.zeros_like(tasks,dtype=torch.bool)
    inputs = (tasks,pre,roles,sizes,support,half,blocker,near)
    pending = torch.arange(n)
    for _ in range(8):
        result = sample_layout(*(value[pending] for value in inputs),environment()['sensorScene']['warehouse'],16)
        pending = pending[~result['valid']]
        if not len(pending):
            break
    assert not len(pending)
