from copy import deepcopy
from types import SimpleNamespace

import isaacgym
import pytest
import torch

from env.tasks.multi_agent import warehouse_before
from utils.experiments import native_config, validate_experiment
from utils.sensor_stage2_spec import WAREHOUSE


NAME = 'stage2_sensor_900_warehouse_fixedroles_v4_beforepairs'
BASE = 'stage2_sensor_900_warehouse_fixedroles_v4'


def test_config_adds_only_the_before_pairs_probability():
    cfg, train = native_config(NAME), native_config(NAME, True)
    validate_experiment(NAME, cfg['env'], train)
    restored = deepcopy(cfg)
    restored['experiment']['name'] = BASE
    assert restored['env']['sensorScene']['warehouse']['cooperative'].pop('before_pairs_probability') == .2
    env, warehouse = restored['env'], restored['env']['sensorScene']['warehouse']
    assert (env['stateInit'], warehouse['start_distribution']['probabilities']) == ('Default', [1., 0., 0.])
    env['stateInit'], warehouse['start_distribution']['probabilities'] = 'Random', [.5, .25, .25]
    assert env.pop('warehousePushRegionMatching') is True
    assert warehouse.pop('layout_cache') == dict(capacity=32, samples=2, refresh_probability=.25, warmup_resets=32)
    assert restored == native_config(BASE)
    base_train = native_config(BASE, True)
    train['params']['config']['name'] = base_train['params']['config']['name']
    assert train == base_train
    for value in (-.1, 1.1):
        bad = deepcopy(cfg['env'])
        bad['sensorScene']['warehouse']['cooperative']['before_pairs_probability'] = value
        with pytest.raises(ValueError):
            validate_experiment(NAME, bad, native_config(NAME, True))


def test_apron_teams_are_two_at_leaders_with_one_follower_each():
    torch.manual_seed(3)
    env = native_config(NAME)['env']
    n = 12000
    sizes = torch.tensor([.4, .4, .35])+torch.rand(n, 4, 3)*torch.tensor([.2, .2, .15])
    groups, pre, tasks, roles = warehouse_before.sample_roles(sizes, env['box']['build']['taskSizeRanges'],
        env['relationGraph']['template_probabilities'])
    follower = pre >= 0
    assert (follower.sum(-1) == 2).all() and not (tasks == 5).any()
    row, person = follower.nonzero(as_tuple=True)
    leader = pre[row, person]
    assert (tasks[row, leader] == 3).all() and (pre[row, leader] == -1).all()
    assert (leader.view(n, 2)[:, 0] != leader.view(n, 2)[:, 1]).all()
    assert torch.equal(groups[row, person], leader) and (groups[~follower] == torch.nonzero(~follower)[:, 1]).all()
    assert set(tasks[row, person].unique().tolist()) == {1, 2, 4}
    assert torch.equal(roles.sort(-1).values, torch.arange(4).expand(n, 4))


def test_apron_region_lies_outside_the_room_on_the_raised_floor():
    cfg = native_config(NAME)['env']['sensorScene']['warehouse']
    half = torch.zeros(3, 4, 2, 2)
    half[:, WAREHOUSE, 0] = torch.tensor(cfg['cooperative']['room_size'])/2
    task = SimpleNamespace(_surface_half=half, _warehouse=cfg, _warehouse_coop=cfg['cooperative'])
    center, size = warehouse_before.apron_region(task, torch.arange(3))
    low, high = center-size, center+size
    hx, hy = cfg['cooperative']['room_size'][0]/2, cfg['cooperative']['room_size'][1]/2
    apron = cfg['cooperative']['apron_length']
    torch.testing.assert_close(high[:, 0], torch.full((3,), -hx-cfg['wall_thickness']))
    torch.testing.assert_close(low[:, 0], torch.full((3,), -hx-apron))
    torch.testing.assert_close(size[:, 1], torch.full((3,), hy+.5))


def test_record_splits_leader_and_follower_success_by_variant():
    n = 4
    task = SimpleNamespace(device='cpu', _scene_ids=torch.full((n,), WAREHOUSE),
        _before_predecessor=torch.tensor([[-1, -1, -1, 2], [-1, 0, -1, 2], [-1, 0, -1, 2], [-1, -1, -1, 2]]),
        _edge_ever_goal=torch.tensor([[1, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0]], dtype=torch.bool),
        _warehouse_before_pairs=torch.tensor([False, True, True, False]))
    warehouse_before.allocate(SimpleNamespace(num_envs=n, device='cpu'))
    task._warehouse_variant_samples = torch.tensor([2., 2.])
    task._warehouse_variant_outcomes = torch.zeros(2, 5)
    warehouse_before.record(task, torch.tensor([True, True, True, False]))
    torch.testing.assert_close(task._warehouse_variant_outcomes,
                               torch.tensor([[1., 1., 1., 1., 0.], [2., 4., 3., 4., 2.]]))
    result = {}
    warehouse_before.diagnostics(task, result)
    assert result['warehouse/variant/before_pairs/leader_success'] == .75
    assert result['warehouse/variant/before_pairs/follower_success'] == .5
    assert result['warehouse/variant/push_before/episodes'] == 1
