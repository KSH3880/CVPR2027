from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from utils import stage2_spec
from utils.before_goal_sampling import route_distances
from utils.before_spec import task_objects
from utils.coupled_push_spec import checkpoint_contract, group_sizes
from utils.experiments import native_config, validate_experiment, experiment_horizon
from utils.warehouse_coop import box_layout, sample_roles_graph, sample_layout


NAME = 'stage2_sensor_900_warehouse_coop'
BASE = 'stage2_sensor_900_warehouse_safety'


def test_separate_contract_preserves_other_scenes_and_learning():
    cfg, old = native_config(NAME), native_config(BASE)
    validate_experiment(NAME, cfg['env'], native_config(NAME, True))
    actual = deepcopy(cfg)
    del actual['env']['sensorScene']['warehouse']['cooperative']
    actual['experiment']['name'] = BASE
    assert actual == old
    training = native_config(NAME, True)
    training['params']['config']['name'] = native_config(BASE, True)['params']['config']['name']
    assert training == native_config(BASE, True)
    assert checkpoint_contract(cfg['env']) != checkpoint_contract(old['env'])
    assert experiment_horizon(NAME) == 900


def test_roles_share_one_push_group_without_losing_workers():
    torch.manual_seed(12)
    env = native_config(NAME)['env']
    n = 12000
    counts = torch.arange(n)%3+1
    sizes = torch.tensor([.5,.5,.4]).expand(n,4,3)
    groups, pre, tasks, roles = sample_roles_graph(sizes,env['box']['build']['taskSizeRanges'],
        env['relationGraph']['template_probabilities'],counts,.8)
    push = tasks == 5
    assert torch.equal(push.sum(-1),counts)
    assert torch.equal(group_sizes(groups)[push],counts[:,None].expand_as(tasks)[push])
    assert (pre[push] == -1).all()
    assert (stage2_spec.required_tokens(groups,pre,tasks) <= 8).all()
    assert not (pre[counts == 3] >= 0).any()
    assert set(tasks[~push].tolist()) == {1,2,3,4}
    for count, expected in ((1,.96),(2,.8)):
        paired = (pre[counts == count] >= 0).any(-1).float().mean()
        assert abs(float(paired)-expected) < .025


@pytest.mark.parametrize('count',[1,2,3])
@pytest.mark.parametrize('direction',[0,1,2,3])
@pytest.mark.parametrize('mode',[0,1,2])
def test_layout_directions_rsi_workers_and_near_routes(count,direction,mode):
    torch.manual_seed(100+count*20+direction*3+mode)
    env = native_config(NAME)['env']
    cfg = env['sensorScene']['warehouse']
    n = 32
    sizes = torch.tensor([.45,.45,.4]).expand(n,4,3)
    supports = torch.tensor([.7,.7,.4]).expand_as(sizes)
    counts = torch.full((n,),count)
    directions = torch.full((n,),direction)
    modes = torch.full((n,),mode)
    groups,pre,tasks,roles = sample_roles_graph(sizes,env['box']['build']['taskSizeRanges'],
        env['relationGraph']['template_probabilities'],counts,.8)
    near = (pre >= 0) & (torch.rand(n,4) < .5)
    result = sample_layout(tasks,pre,roles,sizes,supports,near,counts,directions,modes,cfg,16)
    good = result['valid']
    assert good.float().mean() > .8
    roots,source,goal = [result[key][good] for key in ('root','source','goal')]
    push = tasks[good] == 5
    assert (roots[...,0][~push] < -.75).all()
    geometry = box_layout(counts[good],directions[good],torch.zeros(int(good.sum()),dtype=torch.long),cfg)
    start = source[torch.arange(len(roots)),roles[good,0]]
    target = goal[torch.arange(len(roots)),roles[good,0]]
    if mode == 0:
        torch.testing.assert_close(start,geometry['door'])
    elif mode == 2:
        torch.testing.assert_close(start,target)
    else:
        assert ((start-geometry['door'])*(target-geometry['door'])).sum(-1).min() > 0
        assert ((start-geometry['door']).norm(dim=-1) < (target-geometry['door']).norm(dim=-1)).all()
    if direction == 3 and mode == 0:
        assert (roots[...,0][push] > 0).all()
    elif direction < 2 or (direction == 2 and mode == 0):
        assert (roots[...,0][push] < -.35).all()
    row,follower = (pre[good] >= 0).nonzero(as_tuple=True)
    leader = pre[good][row,follower]
    first,second = route_distances(roots[row,leader],source[row,leader],goal[row,leader],
        roots[row,follower],source[row,follower],tasks[good][row,follower])
    assert (second[near[good][row,follower]] < first[near[good][row,follower]]).all()
    valid = task_objects(pre[good],tasks[good])[:,:4] & ~push
    assert (source[...,0][valid] > 0).all()
    assert (source[...,1].abs()[valid] < 4.).all()


@pytest.mark.parametrize('direction',[0,1,2,3])
@pytest.mark.parametrize('mode',[0,1,2])
def test_crowded_independent_carry_and_two_ontop_workers(direction,mode):
    torch.manual_seed(804+direction*3+mode)
    cfg = native_config(NAME)['env']['sensorScene']['warehouse']
    n = 64
    tasks = torch.tensor([[4,4,3,5]]).expand(n,-1)
    pre = torch.full_like(tasks,-1)
    roles = torch.tensor([[3,0,1,2]]).expand(n,-1)
    sizes = torch.tensor([[[.55,.6,.55],[.6,.55,.6],[.6,.6,.6],[1.8,2.,1.5]]]).expand(n,-1,-1)
    supports = torch.tensor([.8,.8,.5]).expand_as(sizes)
    result = sample_layout(tasks,pre,roles,sizes,supports,torch.zeros_like(pre,dtype=torch.bool),
        torch.ones(n,dtype=torch.long),torch.full((n,),direction),torch.full((n,),mode),cfg,32)
    assert result['valid'].all()


def test_opening_axis_and_actual_worker_episode_denominators():
    from env.tasks.multi_agent.warehouse_sensor import record
    from env.tasks.multi_agent.warehouse_coop import diagnostics
    cfg = native_config(NAME)['env']['sensorScene']['warehouse']
    n = 24
    counts = torch.arange(n)//8+1
    directions = torch.arange(n)//2%4
    modes = torch.arange(n)%2*2
    layout = box_layout(counts,directions,modes,cfg)
    boxes = torch.zeros(n,3,13)
    slot = counts-1
    rows = torch.arange(n)
    boxes[rows,slot,:2] = layout['source']-torch.tensor([3.,0.])
    boxes[rows,slot,2] = .85
    boxes[rows,slot,5] = (layout['yaw']/2).sin()
    boxes[rows,slot,6] = (layout['yaw']/2).cos()
    sizes = torch.tensor(cfg['cooperative']['box_size']).expand(n,3,3).clone()
    sizes[:,:,0] *= torch.arange(1,4)
    templates = torch.where(torch.arange(4)[None] < counts[:,None],5,3)
    half = torch.zeros(n,4,1,2)
    half[:,3,0] = torch.tensor([3.,4.])
    task = SimpleNamespace(device='cpu',num_agents=4,_scene_ids=torch.full((n,),3),
        _warehouse=cfg,_warehouse_coop=cfg['cooperative'],_warehouse_start=cfg['start_distribution'],
        _rigid_body_pos=torch.zeros(n,4,1,3),_env_origins=torch.zeros(n,3),
        _structure_shift=torch.zeros(n,4,2),_structure_yaw=torch.zeros(n,4),_surface_half=half,
        _templates=templates,_warehouse_direction=directions,_bridge_count=counts,_bridge_slot=slot,
        _box_states=boxes,_box_size=sizes,_warehouse_start_mode=modes,
        _warehouse_landing=boxes[rows,slot,:3],_relation_cfg={'at':{'success_distance':.5}},
        _warehouse_opened=torch.zeros(n,dtype=torch.bool),_warehouse_arrived=torch.zeros(n,dtype=torch.bool),
        _warehouse_entered=torch.zeros(n,4,dtype=torch.bool),_warehouse_outcomes=torch.zeros(4),
        _warehouse_start_outcomes=torch.zeros(3,4),_warehouse_worker_episodes=torch.zeros(()),
        _warehouse_start_worker_episodes=torch.zeros(3),_warehouse_coop_samples=torch.zeros(36),
        _warehouse_coop_outcomes=torch.zeros(36,5))
    record(task,torch.ones(n,dtype=torch.bool))
    assert torch.equal(task._warehouse_opened,modes == 2)
    assert int(task._warehouse_worker_episodes) == int((4-counts).sum())
    assert int(task._warehouse_entered.sum()) == int((4-counts).sum())
    results = {}
    diagnostics(task,results)
    for count in (1,2,3):
        for direction in ('left','right','in','out'):
            for mode in ('closed','open'):
                prefix='warehouse/pushers{}/{}/{}/'.format(count,direction,mode)
                assert results[prefix+'worker_episodes'] == 4-count
                assert results[prefix+'entered_people_fraction'] == 1.
