from copy import deepcopy

import pytest
import torch

from utils.coupled_push_spec import group_sizes
from utils.experiments import experiment_horizon, native_config, validate_experiment
from utils.warehouse_coop import box_layout, sample_layout, sample_roles_graph


NAME = 'stage2_sensor_900_warehouse_fixedroles_v4'
BASE = 'stage2_sensor_900_room_warehouse_wallterm_boxterm_v4_reward'


def test_config_changes_only_warehouse_roles_and_geometry():
    cfg, train = native_config(NAME), native_config(NAME, True)
    validate_experiment(NAME, cfg['env'], train)
    assert experiment_horizon(NAME) == 900
    assert cfg['env']['stage2']['reward_source'] == 'v4_teachers'
    assert cfg['experiment']['transfer']['checkpoint'] == 'output/stage2/Stage2Distill900V4_00002000.pth'
    restored = deepcopy(cfg)
    restored['experiment']['name'] = BASE
    scene = restored['env']['sensorScene']
    assert scene['mixture'] == dict(cliff=0., bridge=0., room=0., warehouse=1.)
    scene['mixture'].update(room=.5, warehouse=.5)
    warehouse = scene['warehouse']
    assert warehouse.pop('worker_pairing_probability') == 1.
    warehouse['worker_pairing_probability'] = .8
    coop = warehouse['cooperative']
    expected = dict(pusher_probabilities=[0., 1., 0.], direction_probabilities=[0., 0., 1., 0.],
                    room_size=[8., 8.], frontal_distance=5.)
    assert {key: coop[key] for key in expected} == expected
    coop.update(pusher_probabilities=[1/3]*3, direction_probabilities=[.25]*4, room_size=[6., 8.],
                frontal_distance=3.)
    assert restored == native_config(BASE)
    base_train = native_config(BASE, True)
    train['params']['config']['name'] = base_train['params']['config']['name']
    assert train == base_train


def test_roles_are_two_pushers_and_one_at_before_pair():
    torch.manual_seed(7)
    env = native_config(NAME)['env']
    n = 12000
    sizes = (torch.tensor([.4, .4, .35])+torch.rand(n, 4, 3)*torch.tensor([.2, .2, .15]))
    groups, pre, tasks, roles = sample_roles_graph(sizes, env['box']['build']['taskSizeRanges'],
        env['relationGraph']['template_probabilities'], torch.full((n,), 2),
        env['sensorScene']['warehouse']['worker_pairing_probability'])
    push = tasks == 5
    assert (push.sum(-1) == 2).all()
    assert (group_sizes(groups)[push] == 2).all() and (pre[push] == -1).all()
    follower = pre >= 0
    assert (follower.sum(-1) == 1).all()
    row, person = follower.nonzero(as_tuple=True)
    assert (tasks[row, pre[row, person]] == 3).all()
    assert ((~push & ~follower).sum(-1) == 1).all()
    families = tasks[row, person]
    assert set(families.unique().tolist()) == {1, 2, 4}
    assert (torch.bincount(families, minlength=5)[[1, 2, 4]].float()/n).min() > .2


@pytest.mark.parametrize('mode', [0, 1, 2])
def test_two_pushers_push_inward_five_metres_inside_deeper_room(mode):
    torch.manual_seed(31+mode)
    env = native_config(NAME)['env']
    cfg = env['sensorScene']['warehouse']
    n = 64
    sizes = torch.tensor([.5, .5, .42]).expand(n, 4, 3)
    supports = torch.tensor([.7, .7, .45]).expand_as(sizes)
    counts, directions, modes = torch.full((n,), 2), torch.full((n,), 2), torch.full((n,), mode)
    _, pre, tasks, roles = sample_roles_graph(sizes, env['box']['build']['taskSizeRanges'],
        env['relationGraph']['template_probabilities'], counts, 1.)
    near = (pre >= 0) & (torch.rand(n, 4) < .5)
    result = sample_layout(tasks, pre, roles, sizes, supports, near, counts, directions, modes, cfg, 16)
    good = result['valid']
    assert good.float().mean() > .8
    layout = box_layout(counts, directions, modes, cfg)
    torch.testing.assert_close(layout['landing']-layout['door'], torch.tensor([5., 0.]).expand(n, 2))
    depth = cfg['cooperative']['room_size'][0]
    assert float(layout['landing'][0, 0]+layout['world_size'][0, 0]/2) < depth-2.
    source = result['source'][good]
    push = tasks[good] == 5
    start = source[push].view(-1, 2, 2)[:, 0]
    if mode == 2:
        torch.testing.assert_close(start, layout['landing'][good])
    worker = source[~push]
    assert ((worker[:, 0] > 0) & (worker[:, 0] < depth) & (worker[:, 1].abs() < 4.)).all()
