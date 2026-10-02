"""CPU regressions for randomized AT destinations and shared viewer targets."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import yaml

from env.tasks.multi_agent.edge_ontop_task import SampledOnTopTaskMixin
from env.tasks.multi_agent.scene_features import (
    at_goal_marker_positions, object_task_owners, SHARED_TASK_COLOR,
    task_target_owners, task_marker_vertices,
)
from utils.edge_context_spec import EdgeContextGraph, AT, HOLDING
from utils.edge_interaction_spec import SIT, CLIMB
from utils.edge_ontop_spec import ON_TOP, expand_graph, select_graph
from utils.edge_stage2_spec import configure_evaluation_graph, sample_graph
from utils.stage2_shared_spec import PAIRS, ALIASES


def _graph(rows, agents=2, objects=4):
    # Rows are (owner, relation, source entity, destination entity, valid).
    owner, relation, src, dst, valid = zip(*rows)
    size = len(rows)
    return EdgeContextGraph(
        tuple(str(i) for i in range(size)), agents, objects,
        torch.tensor(src), torch.tensor(dst), torch.tensor(relation),
        torch.tensor(owner), torch.tensor(valid), torch.zeros(size, dtype=torch.bool),
        torch.zeros(size, size, dtype=torch.bool), torch.full((size,), -1),
    )


def test_at_marker_follows_shuffled_goal_not_owner_and_leaves_policy_targets_intact():
    # H0 carries O0 to G1, while H1 sits on O0. The unused G0 stays hidden.
    graph = _graph([(0, HOLDING, 0, 2, True), (0, AT, 2, 7, True),
                    (1, SIT, 1, 2, True), (1, AT, 3, 6, False)])
    targets = torch.tensor([[[4., 5., .3], [7., 8., .4]]])
    original = targets.clone()
    markers = at_goal_marker_positions(targets, graph)
    assert torch.equal(markers[0, 1], targets[0, 1])
    assert markers[0, 0, 2] == 20.
    assert torch.equal(markers[..., :2], targets[..., :2])
    assert torch.equal(targets, original)


def test_at_visibility_handles_expanded_humans_partial_resets_and_no_at():
    graph = _graph([(3, AT, 4, 10, True), (1, AT, 5, 13, True),
                    (0, CLIMB, 0, 4, True)], agents=4, objects=6)
    graph = expand_graph(graph, 3)
    relation = graph.edge_relation.clone()
    relation[2, :2] = SIT
    graph = replace(graph, edge_relation=relation)
    # A reset of envs 2 and 0 must use their own graphs in that order.
    ids = torch.tensor([2, 0])
    targets = torch.zeros(3, 4, 3)
    targets[..., 2] = .5
    markers = at_goal_marker_positions(targets[ids], select_graph(graph, ids))
    assert torch.equal(markers[0, :, 2], torch.full((4,), 20.))
    assert torch.equal(markers[1, :, 2], torch.tensor([.5, 20., 20., .5]))
    assert torch.equal(targets[..., 2], torch.full((3, 4), .5))


def test_object_sharing_counts_distinct_owners_including_ontop_support():
    graph = _graph([(0, HOLDING, 0, 2, True), (0, AT, 2, 6, True),
                    (1, HOLDING, 1, 3, True), (1, ON_TOP, 3, 2, True),
                    (1, SIT, 1, 4, False)])
    assert object_task_owners(graph, 0) == {0: {0, 1}, 1: {1}}


def _draw(graph):
    task = SampledOnTopTaskMixin()
    task._edge_ontop = task._edge_interaction = True
    task.viewer = object()
    task.envs = [object()]
    task.gym = Mock()
    task.relation_runtime = SimpleNamespace(graph=graph)
    task._agent_color = lambda owner: SimpleNamespace(
        x=.2 + .1 * owner, y=.3, z=.7)
    task._ontop_last_diag = {'target': torch.zeros(1, len(graph.ids), 3)}
    task._draw_ontop_context()
    task.gym.clear_lines.assert_called_once_with(task.viewer)
    return task.gym.add_lines.call_args_list


@pytest.mark.parametrize('preset', [*PAIRS, *ALIASES])
@pytest.mark.parametrize('agents,objects', [(2, 4), (4, 6)])
def test_shared9_marker_colors_require_same_action_and_same_object(preset, agents, objects):
    root = Path(__file__).resolve().parents[1]
    env = deepcopy(yaml.safe_load((root / 'data/cfg/multi_agent/'
        'approach_stage2_rescue_shared9.yaml').read_text())['env'])
    env.update(numAgents=agents, numObjects=objects)
    configure_evaluation_graph(env, True)
    graph = sample_graph(1, env['relationGraph'], preset=preset,
                         generator=torch.Generator().manual_seed(19))
    calls = _draw(graph)
    visible = graph.edge_valid[0] & (
        (graph.edge_relation[0] == SIT) | (graph.edge_relation[0] == CLIMB) |
        (graph.edge_relation[0] == ON_TOP))
    assert len(calls) == int(visible.sum()) > 0
    same_action = preset in ('sit_sit', 'climb_climb', 'ontop_ontop')
    for edge, call in zip(visible.nonzero(as_tuple=False).flatten().tolist(), calls):
        owner = int(graph.edge_owner[0, edge])
        color = SHARED_TASK_COLOR if same_action else (.2 + .1 * owner, .3, .7)
        relation = int(graph.edge_relation[0, edge])
        assert call.args[2] == {SIT: 24, ON_TOP: 12, CLIMB: 3}[relation]
        np.testing.assert_allclose(call.args[4], np.tile(color, (call.args[2], 1)))


def test_independent_sit_targets_keep_owner_colors_and_invalid_edges_are_not_drawn():
    graph = expand_graph(_graph([(0, SIT, 0, 2, True),
                                 (1, SIT, 1, 3, True),
                                 (1, CLIMB, 1, 2, False)]), 1)
    calls = _draw(graph)
    assert len(calls) == 2
    for owner, call in enumerate(calls):
        np.testing.assert_allclose(call.args[4], np.tile((.2 + .1 * owner, .3, .7), (24, 1)))


def test_same_box_sit_edges_are_both_drawn_yellow_at_the_same_point():
    graph = expand_graph(_graph([(0, SIT, 0, 2, True),
                                 (1, SIT, 1, 2, True)]), 1)
    calls = _draw(graph)
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0].args[3], calls[1].args[3])
    for call in calls:
        np.testing.assert_allclose(call.args[4], np.tile(SHARED_TASK_COLOR, (24, 1)))


def test_at_and_climb_on_one_box_do_not_make_climb_yellow():
    graph = expand_graph(_graph([(0, HOLDING, 0, 2, True),
                                 (0, AT, 2, 6, True),
                                 (1, CLIMB, 1, 2, True)]), 1)
    calls = _draw(graph)
    assert len(calls) == 1
    np.testing.assert_allclose(calls[0].args[4], np.tile((.3, .3, .7), (3, 1)))


def test_same_owner_duplicate_edges_do_not_make_a_shared_action():
    graph = _graph([(0, SIT, 0, 2, True), (0, SIT, 0, 2, True),
                    (1, SIT, 1, 2, False), (1, CLIMB, 1, 2, True)])
    assert task_target_owners(graph, 0) == {(SIT, 2): {0}, (CLIMB, 2): {1}}
    calls = _draw(expand_graph(graph, 1))
    assert len(calls) == 3
    np.testing.assert_allclose(calls[0].args[4], np.tile((.2, .3, .7), (24, 1)))


def test_marker_shapes_distinguish_sitting_stacking_and_climbing_at_same_point():
    target = np.array([1., 2., .7], dtype=np.float32)
    sit = task_marker_vertices(SIT, target)
    ontop = task_marker_vertices(ON_TOP, target)
    climb = task_marker_vertices(CLIMB, target)
    assert sit.shape == (48, 3) and ontop.shape == (24, 3) and climb.shape == (6, 3)
    np.testing.assert_allclose(sit[:, 2], target[2])
    np.testing.assert_allclose(np.linalg.norm(sit[:, :2] - target[:2], axis=-1), .18, atol=1e-6)
    np.testing.assert_allclose(sit[1::2], np.roll(sit[0::2], -1, axis=0), atol=1e-6)
    np.testing.assert_allclose(np.abs(ontop - target), .12, atol=1e-6)
    # Every cube edge varies along exactly one axis and has nonzero length.
    assert ((np.abs(ontop[1::2] - ontop[0::2]) > 1e-6).sum(-1) == 1).all()
    np.testing.assert_allclose((climb[0::2] + climb[1::2]) / 2, np.tile(target, (3, 1)))
    for vertices in (sit, ontop, climb):
        assert vertices.dtype == np.float32 and np.isfinite(vertices).all()
