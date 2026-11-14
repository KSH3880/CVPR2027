from types import SimpleNamespace

import numpy as np
import torch

from env.tasks.multi_agent.viewer_goal_markers import (
    draw_goal_markers, rectangle, rotation, surface_lines,
)
from env.tasks.multi_agent.push_goal_markers import draw_push_goal_markers
from utils import coupled_push_spec as coupled
from utils.edge_ontop_spec import ON_TOP
from utils.edge_interaction_spec import SIT, CLIMB


def test_surface_markers_follow_rotated_top_faces_and_payload_size():
    q = np.array([0., 0., np.sin(.7/2), np.cos(.7/2)])
    state = np.r_[np.array([3., 5., .6]), q, np.zeros(6)]
    size = np.array([2., 1., 1.2])
    payload = np.array([.4, .6, .5])
    for relation, count in ((ON_TOP, 11), (SIT, 11), (CLIMB, 7)):
        lines = surface_lines(state, size, relation, payload)
        assert lines.shape == (count, 2, 3)
        face = lines[:-3] if relation in (ON_TOP, CLIMB) else lines
        local = (face-state[:3])@rotation(q)
        np.testing.assert_allclose(local[..., 2], size[2]/2+.012)
        if relation == ON_TOP:
            np.testing.assert_allclose(np.max(np.abs(local[..., :2]), axis=(0, 1)), payload[:2]/2)
            assert lines[-3, 0, 2] > lines[-3, 1, 2]
        elif relation == CLIMB:
            assert lines[-3, 0, 2] < lines[-3, 1, 2]


def marker_task(templates, groups, push=False):
    m = len(templates)
    groups = torch.tensor([groups])
    graph = coupled.compose_graph(torch.tensor([templates]), groups, False,
                                   individual_goals=True, num_objects=2*m)
    captured = []
    boxes = torch.zeros(1, 2*m, 13)
    boxes[..., 6] = 1
    sizes = torch.full((1, 2*m, 3), .6)
    task = SimpleNamespace(num_agents=m, num_objects=2*m, num_envs=1,
        relation_runtime=SimpleNamespace(graph=graph), _box_states=boxes,
        _box_size=sizes, _tar_pos=torch.tensor([[[12., 23., .3]]]).expand(1, m, 3).clone(),
        _env_origins=torch.tensor([[10., 20., 0.]]), _design=True,
        _groups=groups, _members=coupled.group_sizes(groups), _individual_push_goals=True,
        _logical_box_values=lambda values: values, viewer=True, envs=[0],
        _agent_color=lambda owner: SimpleNamespace(x=.5, y=.8, z=.2),
        _viewer_agent_color=lambda env, owner: SimpleNamespace(x=.5, y=.8, z=.2),
        gym=SimpleNamespace(add_lines=lambda viewer, env, count, vertices, colors:
                            captured.append(vertices.copy())))
    task._center_tar_pos = task._tar_pos.clone()
    if push:
        task._joint = {}
    return task, captured


def test_carry_footprint_sits_on_ground_in_environment_local_coordinates():
    task, captured = marker_task([3, 3], [0, 1])
    goals = task._tar_pos.clone()
    draw_goal_markers(task)
    assert len(captured) == 2
    for lines in captured:
        assert lines.shape == (4, 2, 3)
        np.testing.assert_allclose(lines[..., 2], .012)
        np.testing.assert_allclose(lines.reshape(-1, 3).mean(0), [2., 3., .012], atol=1e-6)
    assert torch.equal(goals, task._tar_pos)


def test_push_design_draws_one_ring_and_stem_without_spheres():
    task, captured = marker_task([5, 5], [0, 0], push=True)
    draw_push_goal_markers(task)
    assert len(captured) == 1
    assert captured[0].shape == (33, 2, 3)
    np.testing.assert_allclose(captured[0][-1, 1], [2., 3., .3], atol=1e-6)
    task._design = False
    captured.clear()
    draw_push_goal_markers(task)
    assert len(captured) == 2
    assert all(lines.shape == (129, 2, 3) for lines in captured)


