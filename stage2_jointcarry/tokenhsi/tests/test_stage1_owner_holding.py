from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from learning.multi_agent.amp_network_builder_ma import EdgeEncoder, build_entity_type_ids
from learning.multi_agent.edge_context_encoder import PackedEdgeOwnerHoldingFusion
from utils.edge_context_spec import AT, HOLDING
from utils.edge_ontop_spec import ON_TOP
from utils.edge_scenario_spec import sample_graph, validate_sampler
from utils.edge_stage1_spec import (owner_holding_graph_packet,
    parse_owner_holding_packet)
from utils.relation_task_spec import (checkpoint_metadata,
    check_checkpoint_metadata, validate_relation_config)


CONFIG_DIR = Path(__file__).resolve().parents[1] / 'data/cfg/multi_agent'
BASE = yaml.safe_load((CONFIG_DIR /
    'approach_scenario_stage1_paired_placement_no_near.yaml').read_text())['env']
EXPERIMENT = yaml.safe_load((CONFIG_DIR /
    'approach_scenario_stage1_paired_placement_owner_holding.yaml').read_text())['env']


def test_config_is_paired_placement_with_only_owner_holding_observation_added():
    expected = deepcopy(BASE)
    expected['relationGraph']['owner_holding_state'] = True
    expected['relationReward']['stage1_variant'] = (
        'scenario_independent_stage1_paired_placement_owner_holding')
    expected['relationReward']['observation']['graph_packet_fields'].append(
        'owner_holding_state')
    assert EXPERIMENT == expected
    validate_sampler(EXPERIMENT['relationGraph'], 2, 4)
    validate_relation_config(EXPERIMENT['relationReward'])
    old = checkpoint_metadata(BASE['relationReward'])
    new = checkpoint_metadata(EXPERIMENT['relationReward'])
    assert new['graph_record_width'] == 6
    assert new['context_dim_per_edge'] == 1
    with pytest.raises(ValueError, match='relation schema mismatch: suffix_fields'):
        check_checkpoint_metadata({'relation_metadata': old}, new)


@pytest.mark.parametrize('preset,placement_relation',
                         [('holding_at', AT), ('holding_ontop', ON_TOP)])
def test_owner_holding_packet_uses_matching_prerequisite_after_edge_shuffle(
        preset, placement_relation):
    graph = sample_graph(3, EXPERIMENT['relationGraph'], preset=preset,
                         generator=torch.Generator().manual_seed(142))
    phi = torch.zeros(3, 4)
    for batch in range(3):
        for edge in range(4):
            if graph.edge_relation[batch, edge] == HOLDING:
                phi[batch, edge] = .2 if graph.edge_owner[batch, edge] == 0 else .8
    packet = owner_holding_graph_packet(graph, phi)
    valid, src, dst, relation, owner, state = parse_owner_holding_packet(packet)
    assert packet.shape == (3, 24)
    assert valid.all()
    for batch in range(3):
        for edge in range(4):
            if relation[batch, edge] == HOLDING:
                assert state[batch, edge] == 1
            else:
                assert relation[batch, edge] == placement_relation
                hold = (relation[batch] == HOLDING) & (owner[batch] == owner[batch, edge]) & \
                    (dst[batch] == src[batch, edge])
                assert hold.sum() == 1
                assert state[batch, edge] == phi[batch, hold].item()

    missing_hold = replace(graph, edge_valid=graph.edge_valid.clone())
    missing_hold.edge_valid[0, (graph.edge_relation[0] == HOLDING).nonzero()[0]] = False
    with pytest.raises(ValueError, match='requires one owner/source HOLDING edge'):
        owner_holding_graph_packet(missing_hold, phi)


def test_state_changes_only_placement_bias_on_its_existing_source_target_pair():
    graph = sample_graph(1, EXPERIMENT['relationGraph'], preset='holding_at',
                         generator=torch.Generator().manual_seed(81))
    first = torch.zeros(1, 4)
    second = first.clone()
    hold = (graph.edge_relation[0] == HOLDING) & (graph.edge_owner[0] == 0)
    second[0, hold] = .7
    packet1 = owner_holding_graph_packet(graph, first)
    packet2 = owner_holding_graph_packet(graph, second)

    encoder = EdgeEncoder(2, 2, num_relation_types=11)
    fusion = PackedEdgeOwnerHoldingFusion()
    with torch.no_grad():
        encoder.bias_projection.zero_()
        encoder.bias_projection[..., 0] = 1
        fusion.state_mlp[0].weight.zero_()
        fusion.state_mlp[0].bias.fill_(1)
        fusion.state_mlp[0].weight[0, -1] = 1
        fusion.state_mlp[-1].weight.zero_()
        fusion.state_mlp[-1].bias.zero_()
        fusion.state_mlp[-1].weight[0, 0] = 1
    types = build_entity_type_ids(2, 4)
    background = torch.zeros(2, 2, 8, 8)
    delta = fusion(packet2, encoder, types, background) - fusion(
        packet1, encoder, types, background)
    placement = (graph.edge_relation[0] == AT) & (graph.edge_owner[0] == 0)
    edge = placement.nonzero()[0].item()
    source = graph.edge_src[0, edge].item()
    target = graph.edge_dst[0, edge].item()
    assert torch.allclose(delta[:, 0, :, source, target], torch.full((2, 2), .7))
    delta[:, 0, :, source, target] = 0
    assert torch.count_nonzero(delta) == 0
