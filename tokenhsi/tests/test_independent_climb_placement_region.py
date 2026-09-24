from pathlib import Path

import pytest
import torch
import yaml

from env.tasks.multi_agent.edge_interaction_reward import interaction_own_success
from env.tasks.multi_agent.edge_ontop_reward import inner_xy_region_error
from env.tasks.multi_agent.edge_stage1_reward import Stage1ContextRuntime
from utils.edge_context_spec import AT, HOLDING
from utils.edge_interaction_spec import CLIMB
from utils.edge_ontop_spec import ON_TOP
from utils.edge_scenario_spec import (agent_object_indices, classify_templates,
    independent_logical_box_order, sample_graph)
from utils.relation_task_spec import (check_checkpoint_metadata,
    checkpoint_metadata, validate_relation_config)


ROOT = Path(__file__).resolve().parents[1]
ENV = yaml.safe_load((ROOT / 'data/cfg/multi_agent/approach_independent_climb_placement_region.yaml')
                     .read_text())['env']


def test_config_and_checkpoint_isolation():
    validate_relation_config(ENV['relationReward'])
    assert ENV['relationReward']['schema_version'] == 10
    assert ENV['relationReward']['task_sharing'] == {'self': 1., 'teammate': 0.}
    old = yaml.safe_load((ROOT / 'data/cfg/multi_agent/approach_scenario_with_climb.yaml')
                         .read_text())['env']
    with pytest.raises(ValueError, match='schema mismatch'):
        check_checkpoint_metadata(
            {'relation_metadata': checkpoint_metadata(old['relationReward'])},
            checkpoint_metadata(ENV['relationReward']))


def test_balanced_pairs_disjoint_objects_and_random_logical_slots():
    n = 12000
    graph = sample_graph(n, ENV['relationGraph'],
                         generator=torch.Generator().manual_seed(47))
    slots = torch.arange(n).repeat_interleave(2)
    agents = torch.arange(2).repeat(n)
    kind = classify_templates(graph, slots, agents, with_climb=True).view(n, 2)
    for agent in range(2):
        counts = torch.bincount(kind[:, agent], minlength=5).float() / n
        assert counts[:2].sum() == 0
        assert torch.all((counts[2:] - 1 / 3).abs() < .02)
    assert not ((kind == 4).all(-1)).any()
    sources = agent_object_indices(graph, slots, agents).view(n, 2)
    assert (sources[:, 0] != sources[:, 1]).all()
    for logical in range(3):
        assert torch.all((sources == logical).float().mean(0).sub(1 / 3).abs() < .02)
    support = graph.edge_dst - 2
    top = graph.edge_valid & (graph.edge_relation == ON_TOP)
    for env in range(n):
        if top[env].any():
            support_id = support[env, top[env]].item()
            assert support_id not in sources[env].tolist()
    assignments = torch.tensor([[2, 0], [1, 2]])
    order = independent_logical_box_order(graph, torch.tensor([3, 7]), assignments)
    chosen = sources[[3, 7]]
    assert torch.equal(order.gather(1, chosen), assignments)
    assert torch.equal(order.sort(-1).values, torch.arange(3).expand(2, -1))


def test_each_viewer_preset_and_distinct_at_goals():
    for preset in ('climb', 'holding_at', 'holding_ontop',
                   'climb_ontop', 'at_ontop', 'random_scenario'):
        for role_swap in (False, True):
            graph = sample_graph(64, ENV['relationGraph'], preset=preset,
                                 role_swap=role_swap,
                                 generator=torch.Generator().manual_seed(3))
            assert graph.edge_valid.any(-1).all()
            at = graph.edge_valid & (graph.edge_relation == AT)
            both_at = at.sum(-1) == 2
            if both_at.any():
                destinations = graph.edge_dst[both_at][at[both_at]].view(-1, 2)
                assert (destinations[:, 0] != destinations[:, 1]).all()


def test_region_geometry_uses_percent_margin_in_rotated_box_frame():
    yaw = torch.tensor(torch.pi / 4)
    q = torch.tensor([0., 0., torch.sin(yaw / 2), torch.cos(yaw / 2)])
    support = torch.tensor([[1., -2., .2, *q.tolist()]])
    size = torch.tensor([[.5, .4, .4]])
    local = torch.tensor([[.19, .15], [.205, .15]])
    c, s = torch.cos(yaw), torch.sin(yaw)
    world = torch.stack((c * local[:, 0] - s * local[:, 1] + 1,
                         s * local[:, 0] + c * local[:, 1] - 2), -1)
    error = inner_xy_region_error(world, support.expand(2, -1),
                                  size.expand(2, -1), .1)
    assert error[0] == pytest.approx(0, abs=1e-6)
    assert error[1] == pytest.approx(.005, abs=1e-6)


def test_region_success_keeps_center_state_shaping_and_pair_saturation():
    graph = sample_graph(1, ENV['relationGraph'], preset='climb_ontop',
                         generator=torch.Generator().manual_seed(2))
    relation = graph.edge_relation[0]
    climb = int((relation == CLIMB).nonzero()[0])
    top = int((relation == ON_TOP).nonzero()[0])
    hold = int(((relation == HOLDING) & (graph.edge_owner[0] == graph.edge_owner[0, top])).nonzero()[0])
    phi = torch.full((1, 4), .1)
    progress = torch.full((1, 4), .3)
    z = torch.zeros(1, 4)
    feet = torch.zeros(1, 4)
    region = torch.zeros(1, 4)
    z[0, climb] = .19
    feet[0, climb] = .06
    success = interaction_own_success(phi, z, feet, graph, ENV['relationReward'], region)
    assert success[0, climb] and success[0, top]
    runtime = Stage1ContextRuntime(1, graph, ENV['relationReward'], 'cpu')
    result = runtime.step(phi, progress, z, feet, region)
    assert result['total'][0, climb] == pytest.approx(.6)
    assert result['total'][0, top] == pytest.approx(.6)
    assert result['total'][0, hold] == pytest.approx(.6)
    assert torch.equal(result['agent_task_reward'], result['local_task_reward'])
    region[0, climb] = .001
    region[0, top] = .001
    result = runtime.step(phi, progress, z, feet, region)
    assert not result['own_success'][0, climb]
    assert not result['own_success'][0, top]
    assert result['state_component'][0, climb] == pytest.approx(.02)
    assert result['progress_component'][0, climb] == pytest.approx(.06)
    region.zero_()
    z[0, climb] = .201
    z[0, top] = .0011
    assert not interaction_own_success(phi, z, feet, graph,
                                       ENV['relationReward'], region)[0, [climb, top]].any()
    z.zero_()
    feet[0, climb] = .071
    assert not interaction_own_success(phi, z, feet, graph,
                                       ENV['relationReward'], region)[0, climb]
