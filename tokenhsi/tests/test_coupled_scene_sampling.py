from types import SimpleNamespace

import torch

from utils.coupled_push_scene import paths, place_distractors, scene_infeasible
from utils.coupled_push_spec import compose_graph


def scene(objects=2, count=1):
    boxes = torch.zeros(count, objects, 13)
    boxes[..., 2] = .2
    boxes[..., 6] = 1.
    boxes[:, 1, 0] = 10.
    roots = torch.tensor([[[-3., 0., 1.], [5., 2., 1.]]]).expand(count, -1, -1).clone()
    bodies = torch.zeros(count, 2, 1, 13)
    bodies[..., :3] = roots[:, :, None]
    templates = torch.tensor([[5, 3]]).expand(count, -1).clone()
    task = SimpleNamespace(num_agents=2, num_objects=objects, device='cpu',
        _box_states=boxes, _box_size=torch.full((count, objects, 3), .4),
        _logical_box_order=torch.arange(objects)[None].expand(count, -1),
        _templates_by_owner=templates, _members=torch.ones(count, 2, dtype=torch.long),
        _tar_pos=torch.tensor([[[0., 4., .2], [20., 0., .2]]]).expand(count, -1, -1).clone(),
        _humanoid_root_states=roots, _kinematic_humanoid_rigid_body_states=bodies,
        _env_origins=torch.zeros(count, 3))
    task._gather_entity_slots = lambda values, mapping: values.gather(1, mapping[..., None].expand(-1, -1, values.shape[-1]))
    return task


def test_solo_push_still_rejects_a_non_push_person_on_its_path():
    task = scene()
    ids = torch.tensor([0])
    assert not scene_infeasible(task, ids).item()
    task._humanoid_root_states[:, 1, 0] = 0.
    task._kinematic_humanoid_rigid_body_states[:, 1, 0, 0] = 0.
    assert scene_infeasible(task, ids).item()


def test_joint_paths_use_the_full_physical_box_and_skip_solo_checks(monkeypatch):
    task = scene()
    task._logical_box_order = torch.tensor([[0, 0]])
    task._templates_by_owner[:] = 5
    task._members[:] = 2
    task._box_size[:, 0, :2] = torch.tensor([4., 2.])
    task._tar_pos[:, 1] = task._tar_pos[:, 0]
    start, end, _, half = paths(task, torch.tensor([0]))
    torch.testing.assert_close(start, torch.tensor([[[0., -1.5], [0., -1.5]]]))
    torch.testing.assert_close(end, torch.tensor([[[0., 4.], [0., 4.]]]))
    torch.testing.assert_close(half, torch.tensor([[[2., 1.], [2., 1.]]]))

    def unexpected(*args):
        raise RuntimeError('Joint scene invoked a solo-only check')

    monkeypatch.setattr('utils.coupled_push_scene.solo_path_clear', unexpected)
    scene_infeasible(task, torch.tensor([0]))


def test_distractor_retries_keep_supports_and_unsorted_environment_mapping():
    task = scene(objects=4, count=2)
    task._templates_by_owner[:] = torch.tensor([4, 3])
    task._box_states[:, 0, :2] = torch.tensor([-2., 0.])
    task._box_states[:, 1, :2] = torch.tensor([2., 0.])
    graph = compose_graph(task._templates_by_owner, torch.tensor([[0, 1], [0, 1]]),
        False, num_objects=4)
    task.relation_runtime = SimpleNamespace(graph=graph)
    before = task._box_states[:, :3].clone()
    draws = [torch.tensor([[0., 4.], [2., 0.]]), torch.tensor([[0., -4.]])]

    def sample(centers):
        return draws.pop(0) if len(centers) else torch.empty(0, 2)

    task._sample_arena_xy = sample
    place_distractors(task, torch.tensor([1, 0]))
    assert torch.equal(task._box_states[:, :3], before)
    torch.testing.assert_close(task._box_states[:, 3, :2], torch.tensor([[0., -4.], [0., 4.]]))
    assert not draws
