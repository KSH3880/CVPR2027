"""Direction progress shared by the retained all-edge RSI experiments."""
import math

import pytest
import torch

from env.tasks.multi_agent.relation_reward import direction_progress, distance_progress, relation_progress
from utils.relation_task_spec import STATE_MODE, validate_relation_config


@pytest.mark.parametrize('velocity,expected', [
    ([1.5, 0., 0.], 1.), ([.5, 0., 0.], 1.), ([.1, 0., 0.], 1.),
    ([.0001, 0., 0.], 1.), ([1., 1., 0.], 1 / math.sqrt(2)),
    ([0., 1., 0.], 0.), ([-1., 0., 0.], 0.), ([0., 0., 0.], 0.),
    ([0., 0., 1.], 0.), ([.1, 0., 10.], 1.),
])
def test_direction_speed_angle_and_xy_invariance(velocity, expected):
    prev = torch.zeros(1, 3)
    cur = torch.tensor([velocity]) * .1
    goal = cur + torch.tensor([[2., 0., 4.]])
    result = direction_progress(prev, cur, goal, .1)
    torch.testing.assert_close(result, torch.tensor([expected]))
    rot = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    torch.testing.assert_close(result, direction_progress(prev @ rot, cur @ rot, goal @ rot, .1))


def test_no_distance_pinning_height_mask_or_nonfinite_zero_distance():
    cur = torch.zeros(2, 3)
    prev = cur - torch.tensor([[.01, 0., 0.], [0., 0., 0.]])
    # Low box at 10 cm: forward motion gets 1, rest still gets 0.
    goal = cur + torch.tensor([[.1, 0., 0.]])
    torch.testing.assert_close(direction_progress(prev, cur, goal, .1), torch.tensor([1., 0.]))
    torch.testing.assert_close(direction_progress(prev, cur, cur, .1), torch.zeros(2))
    for dt, eps in ((0., 1e-6), (.1, 0.)):
        with pytest.raises(ValueError):
            direction_progress(prev, cur, goal, dt, eps)



def test_dispatch_defaults_to_distance_and_rejects_removed_modes():
    prev = torch.zeros(1, 3)
    cur = torch.tensor([[.05, 0., 0.]])
    goal = torch.tensor([[2., 0., 0.]])
    torch.testing.assert_close(relation_progress(prev, cur, goal, .1), distance_progress(cur, goal))
    assert relation_progress(prev, cur, goal, .1, {'kind': 'direction'}).item() == 1.
    for kind in ('velocity', 'unknown'):
        with pytest.raises(ValueError, match='Unsupported progress kind'):
            relation_progress(prev, cur, goal, .1, {'kind': kind})
        with pytest.raises(ValueError, match='Unsupported progress kind'):
            validate_relation_config({'mode': STATE_MODE, 'progress': {'kind': kind}})


@pytest.mark.parametrize('progress', [
    {'kind': 'direction'},
    {'kind': 'direction', 'approach_radius': .5, 'normalization_epsilon': 0.},
    {'kind': 'direction', 'approach_radius': .5, 'target_speed': 1.5},
])
def test_direction_requires_all_edge_approach(progress):
    with pytest.raises(ValueError):
        validate_relation_config({'mode': STATE_MODE, 'progress': progress})
