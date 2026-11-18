import itertools

import numpy as np
import pytest
import torch

from utils.coupled_push_assignment import assign_regions, best_assignment, crossing_counts, scores, validate_mode
from utils.coupled_push_contact import group_face_geometry
from utils.coupled_push_spec import region_states
from utils.jointpush_geometry import rotate


@pytest.mark.parametrize('mode', ['ordered', 'distance', 'optimal'])
def test_assignment_is_global_for_its_objective(mode):
    roots = np.array([[2., -5.], [-3., -.5], [0., -2.], [1., -1.]])
    targets = np.column_stack((np.arange(4)-1.5, np.full(4, -1.1)))
    result = best_assignment(roots, targets, mode)
    assert sorted(result) == list(range(4))
    distances = np.linalg.norm(roots[:, None]-targets[None], axis=-1)
    permutations = np.array(list(itertools.permutations(range(4))))
    if mode == 'ordered':
        assert np.array_equal(result, [3, 0, 1, 2])
    elif mode == 'distance':
        assert distances[np.arange(4), result].sum() == min(distances[np.arange(4), p].sum() for p in permutations)
    else:
        assert tuple(scores(roots, targets, distances, result[None])[0]) == min(map(tuple, scores(roots, targets, distances, permutations)))


@pytest.mark.parametrize('count', [2, 6, 8, 9, 12])
def test_optimal_supports_large_groups_without_rng(count):
    roots = np.random.default_rng(count).uniform([-5., -6.], [5., -1.], (count, 2))
    targets = np.column_stack((np.arange(count)-count/2, np.full(count, -.6)))
    first = best_assignment(roots, targets, 'optimal')
    assert sorted(first) == list(range(count))
    assert np.array_equal(first, best_assignment(roots, targets, 'optimal'))
    distances = np.linalg.norm(roots[:, None]-targets[None], axis=-1)
    score = tuple(scores(roots, targets, distances, first[None])[0])
    for mode in ('ordered', 'distance'):
        baseline = best_assignment(roots, targets, mode)
        assert score <= tuple(scores(roots, targets, distances, baseline[None])[0])


@pytest.mark.parametrize('mode', ['random', 'ordered', 'distance', 'optimal'])
def test_rotated_multiple_groups_regions_goals_and_rng(mode):
    groups = torch.tensor([[0, 1, 0, 1, 4], [0, 0, 2, 2, 4]])
    regions = torch.tensor([[0, 1, 1, 0, 0], [1, 0, 1, 0, 0]])
    boxes = torch.zeros(2, 5, 13)
    boxes[..., 6] = 1.
    sizes = torch.full((2, 5, 3), 2.)
    roots = torch.tensor([[[2., -3., 0.], [1., -2., 0.], [-1., -4., 0.], [-2., -3., 0.], [5., 1., 0.]]]).expand(2,-1,-1).clone()
    state = torch.get_rng_state()
    expected = assign_regions(groups, roots, boxes, sizes, regions, mode, .6)
    assert torch.equal(torch.get_rng_state(), state)
    if mode == 'random':
        assert expected is regions
    for row in range(2):
        for label in groups[row].unique():
            mask = groups[row] == label
            assert expected[row, mask].sort().values.tolist() == list(range(int(mask.sum())))
    yaw = torch.tensor(.73)
    quaternion = torch.tensor([0., 0., (yaw/2).sin(), (yaw/2).cos()]).expand(2,5,4)
    shift = torch.tensor([9., -13., .4])
    rotated = boxes.clone()
    rotated[..., 3:7] = quaternion
    rotated[..., :3] = shift
    actual = assign_regions(groups, rotate(quaternion, roots)+shift, rotated, sizes, regions, mode, .6)
    assert torch.equal(actual, expected)
    members = (groups[:, :, None] == groups[:, None]).sum(-1)
    tokens, _ = region_states(rotated, sizes, members, actual)
    centers = rotated[..., :3]+torch.tensor([0., 5., 0.])
    goals = centers+tokens[..., :3]-rotated[..., :3]
    torch.testing.assert_close(goals-tokens[..., :3], centers-rotated[..., :3])
    forward = rotate(quaternion, torch.tensor([0., 1., 0.]).expand(2,5,3))
    rear = tokens[..., :3]-forward
    face = group_face_geometry(rear[..., None, :].expand(2,5,2,3), rotated, sizes,
                               rotated[..., :3]+5*forward, groups, members, actual)
    torch.testing.assert_close(face['slot_center'], rear[..., :2])
    assert face['on_face'].all()
    order = torch.tensor([1, 0])
    partial = assign_regions(groups[order], (rotate(quaternion, roots)+shift)[order], rotated[order], sizes[order], regions[order], mode, .6)
    assert torch.equal(partial, actual[order])


def test_crossing_score_counts_intersection_and_collinear_overlap():
    roots = np.array([[-1., -2.], [1., -2.]])
    targets = np.array([[-1., 0.], [1., 0.]])
    assert crossing_counts(roots, targets, np.array([[0, 1], [1, 0]])).tolist() == [0, 1]
    assert crossing_counts(roots, roots, np.array([[1, 0]])).tolist() == [1]


@pytest.mark.parametrize('task,evaluation,protocol,resume', [
    ('HumanoidMACoupledPush', False, '', 0),
    ('HumanoidMABeforePush', True, '', 0),
    ('HumanoidMACoupledPush', True, 'general', 0),
    ('HumanoidMACoupledPush', True, '', 1),
])
def test_viewer_override_cannot_change_training_or_formal_eval(task, evaluation, protocol, resume):
    with pytest.raises(ValueError, match='viewer/test'):
        validate_mode('ordered', task, evaluation, protocol, resume)
    validate_mode('random', task, evaluation, protocol, resume)
    validate_mode('optimal', 'HumanoidMACoupledPush', True)
