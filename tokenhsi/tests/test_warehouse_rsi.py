from copy import deepcopy

import pytest
import torch

from utils.experiments import native_config, validate_experiment, experiment_horizon
from utils.sensor_stage2_spec import validate_sensor
from utils.warehouse_sensor import sample_roles_graph, sample_layout
from utils.before_goal_sampling import route_distances

NAME = 'stage2_sensor_900_warehouse_bridge_init_rsi'
BASE = 'stage2_sensor_900_warehouse_bridge_init'


def test_only_warehouse_start_changes_and_contract_is_separate():
    new, old = native_config(NAME), native_config(BASE)
    validate_experiment(NAME,new['env'],native_config(NAME,True))
    cfg = deepcopy(new)
    del cfg['env']['sensorScene']['warehouse']['start_distribution']
    cfg['experiment']['name'] = old['experiment']['name']
    assert cfg == old
    train = native_config(NAME,True)
    train['params']['config']['name'] = native_config(BASE,True)['params']['config']['name']
    assert train == native_config(BASE,True)
    assert experiment_horizon(NAME) == 900
    with pytest.raises(ValueError,match='environment differs'):
        validate_experiment(NAME,old['env'],native_config(NAME,True),evaluation=True)
    for key,value in (('probabilities',[.5,.5,.5]),('partial_open_progress',[.1,1.])):
        broken = deepcopy(new['env'])
        broken['sensorScene']['warehouse']['start_distribution'][key] = value
        with pytest.raises(ValueError):
            validate_sensor(broken)


@pytest.mark.parametrize('mode',[0,1,2])
def test_conditional_layout_progress_keeps_external_workers_and_near_routes(mode):
    torch.manual_seed(71)
    env = native_config(NAME)['env'];n = 128
    sizes = torch.tensor([.45,.45,.4]).expand(n,4,3)
    support = torch.tensor([.7,.7,.4]).expand_as(sizes)
    blocker = torch.tensor([2.,2.2,1.5]).expand(n,3)
    half = torch.full((n,2),2.5)
    _,pre,tasks,roles = sample_roles_graph(sizes,env['box']['build']['taskSizeRanges'],
        env['relationGraph']['template_probabilities'],'random_scenario',.5)
    near = (pre >= 0) & (torch.rand(n,4) < .5)
    result = sample_layout(tasks,pre,roles,sizes,support,half,blocker,near,
        env['sensorScene']['warehouse'],16,torch.full((n,),mode))
    chosen = result['valid']
    assert chosen.float().mean() > .8
    root,source,goal = [result[k][chosen] for k in ('root','source','goal')]
    rows = torch.arange(len(root));pusher = roles[chosen,0]
    start = source[rows,pusher];landing = goal[rows,pusher]
    assert (root[...,0] < -.75).all()
    torch.testing.assert_close((root[rows,pusher]-start).norm(dim=-1),torch.full((len(rows),),1.6))
    if mode == 0:
        assert (start[:,1] == 0).all()
    elif mode == 1:
        progress = start[:,1].abs()/(blocker[chosen,1]-.04+.35)
        assert ((progress >= .25) & (progress <= .75)).all()
        assert (start[:,1]*landing[:,1] > 0).all()
    else:
        torch.testing.assert_close(start,landing)
    row,follower = (pre[chosen] >= 0).nonzero(as_tuple=True)
    leader = pre[chosen][row,follower]
    first,second = route_distances(root[row,leader],source[row,leader],goal[row,leader],
        root[row,follower],source[row,follower],tasks[chosen][row,follower])
    assert (second[near[chosen][row,follower]] < first[near[chosen][row,follower]]).all()
