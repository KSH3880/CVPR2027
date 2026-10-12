from pathlib import Path

import pytest
import torch
import yaml

from utils.edge_context_spec import HOLDING, AT, at_goal_marker_slots
from utils.edge_interaction_spec import SIT, CLIMB
from utils.edge_ontop_spec import ON_TOP
from utils.edge_scenario_spec import (CLIMB_TEMPLATES, agent_goal_indices, agent_object_indices,
    classify_templates, sample_graph)
from utils.edge_stage1_spec import validate_sampler
from utils.relation_task_spec import validate_relation_config


ROOT = Path(__file__).resolve().parents[1]
ENV = yaml.safe_load((ROOT / 'data/cfg/multi_agent/'
    'approach_scenario_independent_with_climb.yaml').read_text())['env']


def test_config_keeps_independent_binding_and_reward_contract():
    validate_sampler(ENV['relationGraph'])
    validate_relation_config(ENV['relationReward'])
    assert tuple(ENV['relationGraph']['template_probabilities']) == CLIMB_TEMPLATES
    assert ENV['relationGraph']['random_binding'] == {
        'objects': 'disjoint', 'goals': 'disjoint'}
    assert ENV['relationReward']['stage1_variant'] == 'scenario_independent_with_climb'
    assert ENV['relationReward']['task_sharing'] == {'self': .9, 'teammate': .1}
    assert all(sum(row) == pytest.approx(1.) for row in ENV['templateRsi'].values())


def check_independent_graph(graph):
    n = graph.edge_valid.shape[0]
    envs = torch.arange(n).repeat_interleave(2)
    agents = torch.arange(2).repeat(n)
    objects = agent_object_indices(graph, envs, agents).view(n, 2)
    assert torch.all(objects[:, 0] != objects[:, 1])

    top = graph.edge_valid & (graph.edge_relation == ON_TOP)
    assert torch.all(top.sum(-1) <= 1)
    support = graph.edge_dst[top] - 2
    if len(support):
        used = objects[top.any(-1)]
        assert torch.all((support != used[:, 0]) & (support != used[:, 1]))

    at = graph.edge_valid & (graph.edge_relation == AT)
    if at.any():
        goal = graph.edge_dst.masked_fill(~at, -1)
        both = at.sum(-1) == 2
        assert torch.all(goal[both].max(-1).values != goal[both].min(-1).values)


def test_at_goal_marker_uses_bound_goal_slot():
    graph = sample_graph(64, ENV['relationGraph'], preset='holding_at',
        generator=torch.Generator().manual_seed(27))
    at_edges, slots, active = at_goal_marker_slots(graph)
    rows, edges = at_edges.nonzero(as_tuple=True)
    assert (slots[rows, edges] != graph.edge_owner[rows, edges]).any()
    assert active.sum(-1).eq(1).all()
    assert active[rows, slots[rows, edges]].all()
    assert not active[rows, 1 - slots[rows, edges]].any()


def test_random_sampler_preserves_template_marginals_and_disjoint_objects():
    n = 8192
    graph = sample_graph(n, ENV['relationGraph'],
        generator=torch.Generator().manual_seed(27))
    check_independent_graph(graph)
    assert set(graph.edge_relation[graph.edge_valid].tolist()) == {
        HOLDING, AT, ON_TOP, SIT, CLIMB}
    envs = torch.arange(n).repeat_interleave(2)
    agents = torch.arange(2).repeat(n)
    kinds = classify_templates(graph, envs, agents, with_climb=True).view(n, 2)
    for agent in range(2):
        frequencies = torch.bincount(kinds[:, agent], minlength=5).float() / n
        assert torch.all((frequencies - .2).abs() < .02)
    assert not ((kinds[:, 0] == 4) & (kinds[:, 1] == 4)).any()
    objects = agent_object_indices(graph, envs, agents).view(n, 2)
    for agent in range(2):
        frequencies = torch.bincount(objects[:, agent], minlength=3).float() / n
        assert torch.all((frequencies - 1 / 3).abs() < .02)


def test_fixed_presets_and_role_swap_keep_objects_disjoint():
    for preset in ('holding', 'sit', 'climb', 'holding_at', 'holding_ontop'):
        for role_swap in (False, True):
            graph = sample_graph(64, ENV['relationGraph'], preset=preset,
                role_swap=role_swap, generator=torch.Generator().manual_seed(7))
            check_independent_graph(graph)


@pytest.mark.parametrize('role_swap', (False, True))
def test_at_goal_reset_index_matches_graph_target(role_swap):
    graph = sample_graph(128, ENV['relationGraph'], preset='holding_at',
        role_swap=role_swap, generator=torch.Generator().manual_seed(17))
    envs = torch.arange(128)
    agents = torch.full((128,), int(role_swap))
    at = graph.edge_valid & (graph.edge_relation == AT) & (
        graph.edge_owner == agents[:, None])
    expected = (graph.edge_dst.masked_fill(~at, 0).amax(-1)
                - graph.num_agents - graph.num_objects)
    assert set(expected.tolist()) == {0, 1}
    torch.testing.assert_close(agent_goal_indices(graph, envs, agents), expected)
