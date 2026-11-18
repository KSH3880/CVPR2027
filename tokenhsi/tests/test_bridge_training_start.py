from copy import deepcopy

import pytest
import torch

from env.tasks.multi_agent import square_sensor_scene as scene
from utils.experiments import native_config, validate_experiment, experiment_horizon
from utils.sensor_stage2_spec import sample_bridge_starts


NAME = 'stage2_sensor_900_warehouse_bridge_init'


def test_config_changes_only_initialization_and_run_name():
    original = native_config('stage2_sensor_900_warehouse')
    current = deepcopy(native_config(NAME))
    current['env']['sensorScene']['bridge'].pop('start_distribution')
    assert current['env']['sensorScene']['mixture'] == dict(cliff=.05, bridge=.4, room=.25, warehouse=.3)
    current['env']['sensorScene']['mixture'] = original['env']['sensorScene']['mixture']
    current['experiment']['name'] = original['experiment']['name']
    assert current == original
    old_train, new_train = native_config('stage2_sensor_900_warehouse', True), native_config(NAME, True)
    new_train['params']['config']['name'] = old_train['params']['config']['name']
    assert new_train == old_train
    assert experiment_horizon(NAME) == 900
    validate_experiment(NAME, native_config(NAME)['env'], native_config(NAME, True))


def test_mode_frequencies_and_independence():
    torch.manual_seed(19)
    spec = native_config(NAME)['env']['sensorScene']['bridge']['start_distribution']
    counts = torch.randint(0, 3, (60000,))
    mode = sample_bridge_starts(spec, len(counts), 'cpu')
    for count in range(3):
        fraction = torch.bincount(mode[counts == count], minlength=3).float()/(counts == count).sum()
        torch.testing.assert_close(fraction, torch.tensor([.2, .4, .4]), rtol=0, atol=.015)


@pytest.mark.parametrize('count', [1, 2, 3])
@pytest.mark.parametrize('depth', [.01, .015, .02])
def test_three_start_positions(count, depth):
    cfg = deepcopy(native_config(NAME)['env']['sensorScene']['bridge'])
    cfg['start_distribution']['entry_insert_depth'] = depth
    a, b = torch.tensor([[4., 5.]]).repeat(3, 1), torch.tensor([[5., 4.]]).repeat(3, 1)
    box = torch.tensor([[1.8*count, 2., 1.5]]).repeat(3, 1)
    far, gap, _ = scene.bridge_stages(a, b, box, cfg, align_entry=True)
    mode = torch.arange(3)
    layout = scene.bridge_layout(a, b, far, gap, box, cfg, mode == 1,
                                 torch.ones(3, dtype=torch.bool), training_mode=mode)
    torch.testing.assert_close(far[:, 1]+b[:, 1], a[:, 1])
    torch.testing.assert_close(layout['start'][0], layout['landing'][0])
    distance = layout['start'][1, 1]-layout['landing'][1, 1]
    assert .1 <= distance <= 1.5
    torch.testing.assert_close(a[2, 1]-(layout['start'][2, 1]-box[2, 1]/2), torch.tensor(depth))
    torch.testing.assert_close(layout['pusher'][:, 1]-layout['start'][:, 1], torch.full((3,), 1.6))
    torch.testing.assert_close(gap, box[:, 0]+.2)


@pytest.mark.parametrize('count', [1, 2, 3])
def test_present_crosser_all_tasks_and_both_directions(count):
    torch.manual_seed(100+count)
    cfg = native_config(NAME)['env']['sensorScene']['bridge']
    n = 1024
    tasks = torch.arange(n)%4+1
    from_start = (tasks >= 3) & ((torch.arange(n)//4)%2 == 0)
    a, b = torch.tensor([[0., 0.]]).repeat(n, 1), torch.tensor([[10.+1.8*count+.2, 1.]]).repeat(n, 1)
    half_a, half_b = torch.tensor([[5., 5.]]).repeat(n, 1), torch.tensor([[5., 4.]]).repeat(n, 1)
    reverse = ((torch.arange(n)//8)%2 == 0)[:, None]
    start, dest = torch.where(reverse, a, b), torch.where(reverse, b, a)
    start_half, dest_half = torch.where(reverse, half_a, half_b), torch.where(reverse, half_b, half_a)
    landing = torch.tensor([[5.+(1.8*count+.2)/2, 1.]]).repeat(n, 1)
    box = torch.tensor([[1.8*count, 2., 1.5]]).repeat(n, 1)
    gap = torch.full((n,), 1.8*count+.2)
    radius = torch.full((n,), .6*2**.5/2)
    payload, second, human, failed = scene.crosser_layout(tasks, dest, dest_half, start, start_half,
        radius, radius, landing, box, gap, torch.ones(n, dtype=torch.bool), .5, .8, cfg,
        from_start, present=torch.ones(n, dtype=torch.bool))
    assert not failed.any()
    depth = (human[:, 0]-landing[:, 0]).abs()-gap/2
    assert ((depth >= .8-1e-5) & (depth <= 2.+1e-5)).all()
    assert ((human-start).abs() <= start_half-.8+1e-5).all()
    distance = (payload-human).norm(dim=-1)
    assert ((distance[from_start] >= 1.-1e-5) & (distance[from_start] <= 2.+1e-5)).all()
    assert (distance[from_start] >= scene.HUMAN_RADIUS+radius[from_start]+scene.GROUP_PAD-1e-5).all()
    target = torch.where(from_start[:, None], second, payload)
    slope = (target[:, 1]-landing[:, 1])/(target[:, 0]-landing[:, 0])
    torch.testing.assert_close(human[:, 1], landing[:, 1]+(human[:, 0]-landing[:, 0])*slope)
    assert (box[:, 1]/2-slope.abs()*gap/2 >= .2-1e-5).all()


def test_four_scene_mixture_frequency():
    torch.manual_seed(71)
    mixture = native_config(NAME)['env']['sensorScene']['mixture']
    assert tuple(mixture) == ('cliff', 'bridge', 'room', 'warehouse')
    sampled = torch.multinomial(torch.tensor(list(mixture.values())), 60000, replacement=True)
    actual = torch.bincount(sampled, minlength=4).float()/len(sampled)
    torch.testing.assert_close(actual, torch.tensor([.05, .4, .25, .3]), atol=.01, rtol=0)
