import math

import numpy as np
import pytest
import torch

from utils.coupled_push_assignment import best_assignment
from utils.coupled_push_spec import inverse_rotate, ordered_regions, region_states
from utils.jointpush_geometry import rotate
from utils.sensor_stage2_spec import validate_sensor
from utils.experiments import native_config


def scenes(n=256, seed=0):
    torch.manual_seed(seed)
    m = 4
    yaw = torch.rand(n)*2*math.pi
    box = torch.zeros(n, 13)
    box[:, :2], box[:, 2] = torch.randn(n, 2)*5, .75
    box[:, 5], box[:, 6] = (yaw/2).sin(), (yaw/2).cos()
    boxes = box[:, None].expand(n, m, 13)
    sizes = torch.tensor([3.6, 2., 1.5]).expand(n, m, 3)
    groups = torch.tensor([0, 0, 2, 3]).expand(n, m)
    members = torch.tensor([2, 2, 1, 1]).expand(n, m)
    regions = torch.zeros(n, m, dtype=torch.long)
    regions[:, :2] = torch.stack([torch.randperm(2) for _ in range(n)])
    tokens, _ = region_states(boxes, sizes, members, regions)
    behind = rotate(boxes[..., 3:7], torch.tensor([0., -1.7, 0.]).expand(n, m, 3))
    roots = tokens[..., :3]+behind+torch.randn(n, m, 3)*.05
    roots[:, 2:] = torch.randn(n, 2, 3)*4
    return groups, roots, boxes, members, regions


def test_each_pusher_gets_the_region_it_stands_behind_for_any_yaw():
    groups, roots, boxes, members, regions = scenes()
    assert torch.equal(ordered_regions(groups, roots, boxes, members), regions)


def test_swapped_people_swap_regions_and_solo_people_stay_zero():
    groups, roots, boxes, members, regions = scenes(seed=1)
    roots = roots[:, [1, 0, 2, 3]]
    expected = regions[:, [1, 0, 2, 3]]
    expected[:, 2:] = 0
    assert torch.equal(ordered_regions(groups, roots, boxes, members), expected)


def test_matches_the_cpu_ordered_assignment():
    groups, roots, boxes, members, _ = scenes(seed=2)
    result = ordered_regions(groups, roots, boxes, members)
    local = inverse_rotate(boxes[..., 3:7], roots-boxes[..., :3])[..., :2].double().numpy()
    targets = np.zeros((2, 2))
    for row in range(len(groups)):
        assert result[row, :2].tolist() == best_assignment(local[row, :2], targets, 'ordered').tolist()


def test_switch_must_be_boolean_and_needs_the_cooperative_warehouse():
    env = native_config('stage2_sensor_900_warehouse_fixedroles_v4_beforepairs')['env']
    assert env['warehousePushRegionMatching'] is True
    validate_sensor(env)
    env['warehousePushRegionMatching'] = 1
    with pytest.raises(ValueError, match='warehousePushRegionMatching'):
        validate_sensor(env)
    env['warehousePushRegionMatching'] = True
    warehouse = env['sensorScene']['warehouse']
    del warehouse['cooperative'], warehouse['layout_cache']
    with pytest.raises(ValueError, match='warehousePushRegionMatching'):
        validate_sensor(env)
