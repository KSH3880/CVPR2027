import math
from types import SimpleNamespace

import pytest
import torch

from env.tasks.multi_agent.stage2_group_layout import EVAL_RADII, translate_people, layout_metadata


def make_task(people, mixed):
    torch.manual_seed(12)
    pattern = [0, 0, 2, 2, 2, 5, 6, 6]
    groups = ([g+offset for offset in range(0, people, 8) for g in pattern]
              if mixed else list(range(people)))
    task = SimpleNamespace(num_agents=people, device='cpu')
    task._groups = torch.tensor([groups]*3)
    task._humanoid_root_states = torch.randn(3, people, 13)
    task._kinematic_humanoid_rigid_body_states = torch.randn(3, people, 15, 13)
    task._env_origins = torch.tensor([[0., 0., 0.], [500., -500., 0.], [-500., 500., 0.]])
    task._group_spawn_targets = torch.zeros(3, people, 2)
    return task


@pytest.mark.parametrize('people', [8, 16])
@pytest.mark.parametrize('mixed', [False, True])
def test_fixed_radius_preserves_groups_relative_pose_and_ring_rng(people, mixed):
    task = make_task(people, mixed)
    ids = torch.arange(3)
    roots = task._humanoid_root_states.clone()
    bodies = task._kinematic_humanoid_rigid_body_states.clone()
    torch.manual_seed(27)
    translate_people(task, ids)
    old_targets = task._group_spawn_targets.clone()
    rng = torch.get_rng_state()
    origin = task._env_origins[:, None, :2]
    direction = old_targets-origin
    radius = EVAL_RADII[people]
    expected = origin+radius*direction/direction.norm(dim=-1, keepdim=True)
    task._humanoid_root_states.copy_(roots)
    task._kinematic_humanoid_rigid_body_states.copy_(bodies)
    torch.manual_seed(27)
    translate_people(task, ids, radius=radius)
    torch.testing.assert_close(task._group_spawn_targets, expected, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.equal(task._humanoid_root_states[..., 2:], roots[..., 2:])
    assert torch.equal(task._kinematic_humanoid_rigid_body_states[..., 2:], bodies[..., 2:])
    delta = task._group_spawn_targets-torch.stack([
        roots[row, task._groups[row] == group, :2].mean(0)
        for row in range(3) for group in task._groups[row]]).reshape(3, people, 2)
    torch.testing.assert_close(task._humanoid_root_states[..., :2], roots[..., :2]+delta)
    torch.testing.assert_close(task._kinematic_humanoid_rigid_body_states[..., :2],
                               bodies[..., :2]+delta[:, :, None])
    metadata = layout_metadata(task)
    count = len(task._groups[0].unique())
    assert metadata['group_spawn_gap'] is None
    assert metadata['group_spawn_radius'] == radius
    assert metadata['minimum_initial_group_gap'] == pytest.approx(2*radius*math.sin(math.pi/count), abs=.002)


def test_explicit_legacy_gap_and_wrong_radius_are_distinguished():
    task = make_task(8, False)
    translate_people(task, torch.arange(3))
    metadata = layout_metadata(task)
    assert metadata['group_spawn_radius'] is None
    assert metadata['group_spawn_gap'] == 5.
    assert metadata['minimum_initial_group_gap'] == pytest.approx(5., abs=.002)
    task._group_spawn_radius = 3.
    task._group_spawn_gap = None
    with pytest.raises(ValueError, match='configured radius'):
        layout_metadata(task)
