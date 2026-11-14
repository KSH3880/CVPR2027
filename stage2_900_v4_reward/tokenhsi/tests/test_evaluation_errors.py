import math

import pytest
import torch

from utils.evaluation_errors import ERROR_PROTOCOL, error_scores, spatial_errors, summarize_errors, update_error_minima
from utils.evaluation_protocol import spatial_success


def geometry(kind=0):
    packet = torch.tensor([[[1, kind, 0, 1, 2]]])
    roots = torch.tensor([[[.15, 0., .7]]], dtype=torch.float64)
    boxes = torch.tensor([[[0., 0., .7, 0., 0., 0., 1.], [0., 0., .25, 0., 0., 0., 1.]]], dtype=torch.float64)
    sizes = torch.tensor([[[.4, .4, .4], [.5, .5, .5]]], dtype=torch.float64)
    goals = torch.tensor([[[0., 0., .7]]], dtype=torch.float64)
    return packet, roots, boxes, sizes, goals


def test_region_error_uses_plane_without_height_tolerance():
    args = geometry()
    errors = spatial_errors(*args)
    assert spatial_success(*args).all()
    assert errors[0, 0, 0] == pytest.approx(math.sqrt(.15**2 + .1**2))
    assert errors[0, 0, 1] == pytest.approx(.1)


def test_region_error_respects_support_yaw_and_boundary():
    packet, roots, boxes, sizes, goals = geometry()
    sizes[0, 1, :2] = torch.tensor([1., .25])
    roots[0, 0, 0] = .3
    assert spatial_errors(packet, roots, boxes, sizes, goals)[0, 0, 1] == pytest.approx(.1)
    boxes[0, 1, 3:7] = torch.tensor([0., 0., math.sqrt(.5), math.sqrt(.5)], dtype=torch.float64)
    assert spatial_errors(packet, roots, boxes, sizes, goals)[0, 0, 1] == pytest.approx(math.sqrt(.2**2 + .1**2))


@pytest.mark.parametrize('kind', [2, 4])
def test_at_and_push_have_identical_errors(kind):
    packet, roots, boxes, sizes, goals = geometry(kind)
    packet[..., 4] = 3
    boxes[0, 0, 0] = .3
    errors = spatial_errors(packet, roots, boxes, sizes, goals)
    assert errors.tolist() == [[[.3, .3]]]


def test_ontop_goal_uses_upright_payload_height():
    packet, roots, boxes, sizes, goals = geometry(3)
    sizes[0, 0] = torch.tensor([.8, .4, .2])
    boxes[0, 0, 2] = .6
    boxes[0, 0, 3:7] = torch.tensor([0., math.sqrt(.5), 0., math.sqrt(.5)], dtype=torch.float64)
    assert spatial_errors(packet, roots, boxes, sizes, goals).max() < 1e-8


def test_minimum_includes_after_success_but_excludes_after_fall():
    packet, roots, boxes, sizes, goals = geometry()
    roots[..., 0] = 0
    minimum = torch.full((1, 1, 2), float('inf'), dtype=torch.float64)
    for z, valid in ((.75, True), (.65, True), (.6, False)):
        roots[..., 2] = z
        update_error_minima(minimum, packet, roots, boxes, sizes, goals, torch.tensor([[valid]]))
    assert minimum[0, 0].tolist() == pytest.approx([.05, .05])


def test_error_filter_requires_enabling_group_success_and_counts_shared_box_once():
    packet = torch.zeros(1, 7, 5, dtype=torch.long)
    packet[..., 0] = 1
    packet[..., 1] = torch.tensor([0, 2, 1, 2, 3, 4, 4])
    parent = torch.tensor([[-1, -1, 1, -1, 3, -1, -1]])
    groups = torch.tensor([[0, 1, 2, 3, 4, 5, 5]])
    ever = torch.ones(1, 7, 2, dtype=torch.bool)
    ever[0, 0, 0] = False
    simultaneous = torch.zeros_like(ever)
    simultaneous[0, 4, 1] = True
    minimum = torch.arange(14, dtype=torch.float64).reshape(1, 7, 2) / 100
    centers = torch.zeros(1, 7, dtype=torch.bool)
    centers[0, 5:] = True
    center_minimum = torch.full((1, 7), .03, dtype=torch.float64)
    result = error_scores(packet, parent, groups, ever, simultaneous, minimum, centers, center_minimum)
    assert result['per_task']['SIT']['successes'] == 1
    assert result['per_task']['CLIMB']['successes'] == 0
    assert result['per_task']['CLIMB']['point_error_cm'] is None
    assert result['per_task']['AT'] == pytest.approx(dict(successes=1, point_error_cm=6., region_error_cm=7.))
    assert result['per_task']['ON_TOP']['successes'] == 1
    assert result['per_task']['PUSH'] == dict(successes=1, point_error_cm=3., region_error_cm=3.)
    assert result['coupled_by_size']['2'] == result['per_task']['PUSH']


def test_shared_minimum_remains_valid_until_last_participant_falls():
    packet = torch.tensor([[[1, 4, 0, 2, 3], [1, 4, 1, 2, 4]]])
    roots = torch.zeros(1, 2, 3, dtype=torch.float64)
    boxes = torch.tensor([[[0., 0., 0., 0., 0., 0., 1.]]], dtype=torch.float64)
    sizes = torch.ones(1, 1, 3, dtype=torch.float64)
    goals = torch.zeros(1, 2, 3, dtype=torch.float64)
    groups = torch.zeros(1, 2, dtype=torch.long)
    minimum = torch.full((1, 2, 2), float('inf'), dtype=torch.float64)
    center_minimum = torch.full((1, 2), float('inf'), dtype=torch.float64)
    for distance, alive in ((.1, [True, True]), (.05, [False, True]), (0., [False, False])):
        centers = torch.full((1, 2, 3), 0., dtype=torch.float64)
        centers[..., 0] = distance
        update_error_minima(minimum, packet, roots, boxes, sizes, goals, torch.tensor([alive]),
                            center_minimum, centers, goals, groups)
    assert center_minimum.tolist()[0] == [.05, .05]


def test_error_summary_weights_seeds_equally_and_preserves_na():
    runs = []
    for seed, count, value in ((0, 2, 2.), (1, 10, 8.)):
        runs.append(dict(seed=seed, errors=dict(protocol=ERROR_PROTOCOL,
            per_task=dict(SIT=dict(successes=count, point_error_cm=value, region_error_cm=value / 2)),
            coupled_by_size={})))
    result = summarize_errors(runs)['per_task']['SIT']
    assert result['successes'] == 12
    assert result['point_error_cm']['mean'] == 5.
    assert result['point_error_cm']['std'] == pytest.approx(math.sqrt(18))
    runs[1]['errors']['per_task']['SIT'] = dict(successes=0, point_error_cm=None, region_error_cm=None)
    assert summarize_errors(runs)['per_task']['SIT']['point_error_cm']['mean'] is None
    with pytest.raises(ValueError, match='error protocols'):
        summarize_errors(runs + [dict(seed=2)])
