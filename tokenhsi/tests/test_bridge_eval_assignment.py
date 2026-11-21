from types import SimpleNamespace

import pytest
import torch

from utils.bridge_eval_assignment import restore_ordered_assignment, restore_training_assignment
from utils.coupled_push_spec import inverse_rotate, region_states


@pytest.mark.parametrize('yaw', [-2.3, 0., .7, torch.pi])
@pytest.mark.parametrize('side', [-1, 1])
@pytest.mark.parametrize('ordered', [False, True])
def test_fixture_assignment_without_moving_actors(yaw, side, ordered):
    boxes = torch.zeros(1, 4, 13)
    boxes[..., :3] = torch.tensor([100., -90., .8])
    angle = torch.tensor(yaw + (torch.pi if side == 1 else 0.))
    boxes[..., 5], boxes[..., 6] = (angle/2).sin(), (angle/2).cos()
    roots = boxes.clone()
    local = torch.tensor([[[-.8, side*2.5], [4., 3.], [.8, side*4.], [-2., 3.]]])
    angle = torch.tensor(yaw)
    roots[..., 0] += angle.cos()*local[..., 0]-angle.sin()*local[..., 1]
    roots[..., 1] += angle.sin()*local[..., 0]+angle.cos()*local[..., 1]
    task = SimpleNamespace(_box_states=boxes, _humanoid_root_states=roots,
        _box_size=torch.tensor([3.2, 2., 1.6]).expand(1, 4, 3),
        _structure_yaw=torch.tensor([[0., yaw]]),
        _bridge_pushers=torch.tensor([[True, False, True, False]]),
        _groups=torch.tensor([[0, 1, 0, 3]]),
        _members=torch.tensor([[2, 1, 2, 1]]), _regions=torch.tensor([[1, 0, 0, 0]]),
        _center_tar_pos=torch.tensor([5., 7., .8]).expand(1, 4, 3).clone(),
        _tar_pos=torch.arange(12.).reshape(1, 4, 3),
        _assigned_box_values=lambda values, ids: values[ids])
    physical = [boxes.clone(), roots.clone()]
    original_goals = task._tar_pos.clone()
    ids = torch.tensor([0])
    restore = restore_ordered_assignment if ordered else restore_training_assignment
    assert restore(task, ids) == (0 if ordered and side == 1 else 1)
    assert task._regions.tolist() == ([[1, 0, 0, 0]] if ordered and side == 1 else [[0, 0, 1, 0]])
    assert torch.equal(boxes, physical[0]) and torch.equal(roots, physical[1])
    assert torch.equal(task._tar_pos[:, [1, 3]], original_goals[:, [1, 3]])
    tokens, _ = region_states(boxes, task._box_size, task._members, task._regions)
    if ordered:
        human_local = inverse_rotate(boxes[..., 3:7], roots[..., :3]-boxes[..., :3])
        token_local = inverse_rotate(boxes[..., 3:7], tokens[..., :3]-boxes[..., :3])
        assert (human_local[:, [0, 2], 0]*token_local[:, [0, 2], 0] > 0).all()
    expected = task._center_tar_pos+(tokens[..., :3]-boxes[..., :3])
    assert torch.equal(task._tar_pos[:, [0, 2]], expected[:, [0, 2]])
    goals = task._tar_pos.clone()
    assert restore(task, ids) == 0
    assert torch.equal(task._tar_pos, goals)


def test_random_regions_are_a_permutation_and_partial_resets_are_isolated():
    from utils.bridge_eval_assignment import randomize_pusher_assignment
    torch.manual_seed(19)
    n = 4096
    boxes = torch.zeros(n, 4, 13)
    boxes[..., 6] = 1
    task = SimpleNamespace(_box_states=boxes,
        _box_size=torch.tensor([3.2, 2., 1.6]).expand(n, 4, 3),
        _groups=torch.tensor([[0, 1, 0, 3]]).expand(n, -1),
        _members=torch.tensor([[2, 1, 2, 1]]).expand(n, -1),
        _bridge_pushers=torch.tensor([[True, False, True, False]]).expand(n, -1),
        _regions=torch.tensor([[0, 0, 1, 0]]).repeat(n, 1),
        _center_tar_pos=torch.tensor([5., 7., .8]).expand(n, 4, 3).clone(),
        _tar_pos=torch.arange(12.).reshape(1, 4, 3).repeat(n, 1, 1),
        _assigned_box_values=lambda values, ids: values[ids])
    physical = boxes.clone()
    goals = task._tar_pos.clone()
    ids = torch.arange(n)
    randomize_pusher_assignment(task, ids)
    assert torch.equal(task._regions[:, [0, 2]].sort(-1).values, torch.tensor([0, 1]).expand(n, -1))
    assert .47 < task._regions[:, 0].float().mean() < .53
    assert torch.equal(boxes, physical)
    assert torch.equal(task._tar_pos[:, [1, 3]], goals[:, [1, 3]])
    tokens, _ = region_states(boxes, task._box_size, task._members, task._regions)
    expected = task._center_tar_pos+(tokens[..., :3]-boxes[..., :3])
    assert torch.equal(task._tar_pos[:, [0, 2]], expected[:, [0, 2]])
    old_regions, old_goals = task._regions.clone(), task._tar_pos.clone()
    randomize_pusher_assignment(task, ids[::2])
    assert torch.equal(task._regions[1::2], old_regions[1::2])
    assert torch.equal(task._tar_pos[1::2], old_goals[1::2])
    changed = (task._regions[::2, 0] != old_regions[::2, 0]).float().mean()
    assert .45 < changed < .55
