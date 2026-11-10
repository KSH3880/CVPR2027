from types import SimpleNamespace
import colorsys

import torch

from env.tasks.multi_agent.viewer_design import (
    group_colors, hide_goal_markers, observed_boxes, prepare_scene, push_goals, role_colors,
)
from utils import coupled_push_spec as coupled
from utils import stage2_spec as spec
from env.tasks.multi_agent.scene_features import object_task_owners


PALETTE = ((.54, .85, .2), (.28, .55, .9), (.9, .6, .2), (.8, .35, .75))


def test_enable_tones_keep_hue_and_leave_solo_coupled_unchanged():
    base = [PALETTE[0], PALETTE[0], PALETTE[1], PALETTE[1], PALETTE[2]]
    colors = role_colors(base, [-1, 0, -1, -1, -1])
    leading, following = [colorsys.rgb_to_hsv(*color) for color in colors[:2]]
    assert abs(leading[0]-following[0]) < 1e-6
    assert leading[1] < following[1]
    assert leading[2] > following[2]
    assert colors[2:] == base[2:]
    assert role_colors(base, None) == base


def test_group_colors_merge_alias_owners_and_extend_without_rng():
    rng = torch.get_rng_state()
    colors = group_colors({0: {0, 1}, 1: {1, 2}, 2: {3, 4}}, 8, PALETTE)
    assert colors[:3] == [PALETTE[0]]*3
    assert colors[3:5] == [PALETTE[1]]*2
    assert colors[5:7] == list(PALETTE[2:])
    assert len(set(colors)) == 5
    solo = group_colors({}, 16, PALETTE)
    assert solo[:4] == list(PALETTE)
    assert len(set(solo)) == 16
    assert torch.equal(torch.get_rng_state(), rng)


def test_distractors_preserve_observation_support_alias_and_partial_reset():
    templates = torch.tensor([[3, 4, 5, 5], [3, 1, 5, 5]])
    groups = torch.tensor([[0, 0, 2, 2], [0, 0, 2, 2]])
    predecessor = torch.tensor([[-1, 0, -1, -1], [-1, 0, -1, -1]])
    graph = spec.compose_graph(templates, groups, predecessor, 8, shuffle=False)
    mapping, _, _ = spec.physical_mapping(templates, groups, predecessor, graph)
    boxes = torch.randn(2, coupled.physical_box_count(4), 13)
    task = SimpleNamespace(num_agents=4, num_objects=8, num_envs=2,
        num_physical_objects=boxes.shape[1], device='cpu', viewer=None,
        _video_enabled=False, _logical_box_order=mapping, _box_states=boxes,
        _env_origins=torch.zeros(2, 3), _AGENT_COLOR_PALETTE=PALETTE,
        _before_predecessor=predecessor,
        relation_runtime=SimpleNamespace(graph=graph),
        _update_box_assignment_colors=lambda ids: None)
    task._logical_box_values = lambda values, ids=None: values.gather(
        1, mapping[..., None].expand(-1, -1, 13)) if ids is None else values[ids].gather(
        1, mapping[ids, :, None].expand(-1, -1, 13))
    before = boxes.clone()
    original = task._logical_box_values(boxes).clone()
    prepare_scene(task, torch.arange(2))
    assert torch.equal(observed_boxes(task, task._logical_box_values(boxes), None), original)
    for env in range(2):
        used = set(mapping[env, logical].item()
                   for logical in object_task_owners(graph, env))
        assert torch.equal(boxes[env, sorted(used)], before[env, sorted(used)])
        assert (boxes[env, [i for i in range(boxes.shape[1]) if i not in used], :2] >= 1000).all()
        first, second = [colorsys.rgb_to_hsv(*color) for color in task._design_colors[env][:2]]
        assert abs(first[0]-second[0]) < 1e-6 and first[1] < second[1]
        assert task._design_colors[env][2] == task._design_colors[env][3]
        assert task._design_colors[env][0] != task._design_colors[env][2]
        shared = int(mapping[env, 0])
        expected = tuple((a+b)/2 for a, b in zip(*task._design_colors[env][:2]))
        assert task._design_box_colors[env][shared] == expected
    assert task._design_box_mask.any()
    boxes[..., :3] += .1
    actual = task._logical_box_values(boxes)
    observed = observed_boxes(task, actual, None)
    mask = task._design_box_mask
    assert torch.equal(observed[mask], original[mask])
    assert torch.equal(observed[~mask], actual[~mask])
    untouched = boxes[1].clone(), task._design_saved_boxes[1].clone(), mask[1].clone()
    boxes[0] = torch.randn_like(boxes[0])
    next_sample = task._logical_box_values(boxes, torch.tensor([0])).clone()
    prepare_scene(task, torch.tensor([0]))
    assert torch.equal(observed_boxes(task, task._logical_box_values(boxes, torch.tensor([0])),
                                      torch.tensor([0])), next_sample)
    assert all(torch.equal(a, b) for a, b in zip(untouched,
        (boxes[1], task._design_saved_boxes[1], task._design_box_mask[1])))


def test_shared_push_markers_use_centers_without_changing_policy_goals():
    groups = torch.tensor([[0, 0, 2, 2, 4, 5, 6, 7]])
    templates = torch.full_like(groups, 5)
    graph = coupled.compose_graph(templates, groups, False, individual_goals=True, num_objects=16)
    center = torch.tensor([[[2., 5., .4], [2., 5., .4], [7., 5., .4], [7., 5., .4],
                            [0., 1., .4], [1., 2., .4], [2., 3., .4], [3., 4., .4]]])
    goals = center.clone()
    goals[0, :4, 0] += torch.tensor([-.5, .5, -.5, .5])
    task = SimpleNamespace(num_agents=8, num_envs=1, _groups=groups,
        _members=coupled.group_sizes(groups), _individual_push_goals=True,
        _center_tar_pos=center, _tar_pos=goals, _env_origins=torch.zeros(1, 3),
        _marker_pos=goals.clone(), relation_runtime=SimpleNamespace(graph=graph))
    original = goals.clone()
    result = push_goals(task, 0)
    assert len(result) == 6
    hide_goal_markers(task, torch.tensor([0]))
    for slot, owner, target in result:
        assert torch.equal(target, center[0, owner])
    assert (task._marker_pos >= 1000).all()
    assert torch.equal(goals, original)
    task._individual_push_goals = False
    assert len(push_goals(task, 0)) == 6
