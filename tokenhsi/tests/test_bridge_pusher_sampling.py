from copy import deepcopy

import torch

from utils.bridge_pusher_sampling import rear_points, rectangle_distance
from utils.experiments import native_config, validate_experiment


NAME = 'stage2_sensor_900_bridge_follow_cap_rear90'


def test_config_changes_only_pusher_sampling():
    config = native_config(NAME)
    validate_experiment(NAME, config['env'], native_config(NAME, True))
    env = deepcopy(config['env'])
    spec = env['sensorScene']['bridge'].pop('pusher_sampling')
    assert spec['angle_degrees'] == 90 and spec['surface_distance'] == [.5, 5.]
    assert spec['assignment'] == 'ordered'
    assert env == native_config('stage2_sensor_900_bridge_follow_cap')['env']
    assert config['experiment']['transfer'] == native_config('stage2_sensor_900_bridge_follow_cap')['experiment']['transfer']


def test_rear_cone_distance_is_to_rectangle_not_center_or_ray_boundary():
    torch.manual_seed(41)
    sizes = torch.tensor([[3.2, 1.6, 1.6], [3.2, 2.2, 1.6], [.4, 5., 1.]])
    draws = torch.rand(3, 4096, 2)
    draws[:, :4] = torch.tensor([[0., 0.], [1., 1.], [.5, 0.], [.5, 1.]])
    spec = native_config(NAME)['env']['sensorScene']['bridge']['pusher_sampling']
    xy = rear_points(sizes, draws, spec)
    gap = (xy.abs()-sizes[:, None, :2]/2).clamp_min(0).norm(dim=-1)
    torch.testing.assert_close(gap, .5+4.5*draws[..., 1], atol=2e-6, rtol=0)
    angle = torch.atan2(xy[..., 0], -xy[..., 1])
    torch.testing.assert_close(angle, (draws[..., 0]-.5)*torch.pi/2, atol=2e-7, rtol=0)
    assert (xy[..., 1] < 0).all()


def test_clearance_uses_rotated_rectangle():
    centers = torch.tensor([[[10., 20., 1.]]])
    q = torch.tensor([[[0., 0., 2**-.5, 2**-.5]]])
    size = torch.tensor([[[4., 2., 2.]]])
    points = torch.tensor([[[11.5, 20.], [10., 22.5], [10., 20.]]])
    torch.testing.assert_close(rectangle_distance(points, centers, q, size), torch.tensor([[[.5], [.5], [0.]]]))


def test_random_config_changes_only_assignment_and_run_name():
    config = native_config(NAME+'_random')
    validate_experiment(NAME+'_random', config['env'], native_config(NAME+'_random', True))
    env = deepcopy(config['env'])
    assert env['sensorScene']['bridge']['pusher_sampling']['assignment'] == 'random'
    env['sensorScene']['bridge']['pusher_sampling']['assignment'] = 'ordered'
    assert env == native_config(NAME)['env']
    assert config['experiment']['transfer'] == native_config(NAME)['experiment']['transfer']
