"""Four-object Stage-2 wiring against the live unified owner-HOLDING source."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml

from learning.multi_agent.amp_network_builder_ma import AMPMultiAgentBuilder, EdgeEncoder
from learning.multi_agent.coordination_head import GroundedEdgeCoordination
from learning.multi_agent.edge_context_encoder import PackedEdgeOwnerHoldingFusion
from learning.multi_agent.stage2_transfer import transfer_stage1_weights
from utils.edge_context_spec import AT, HOLDING
from utils.edge_interaction_spec import CLIMB, SIT
from utils.edge_ontop_spec import ON_TOP, permute_graph
from utils.edge_scenario_spec import classify_templates
from utils.edge_stage1_spec import owner_holding_graph_packet
from utils.edge_stage2_spec import (classify_family, sample_graph,
    validate_sampler)
from utils.relation_task_spec import checkpoint_metadata, validate_relation_config
from utils.unified_training import validate_unified_env

ROOT = Path(__file__).resolve().parents[1]
ENV = yaml.safe_load((ROOT / 'data/cfg/multi_agent/approach_stage2_unified_owner_holding.yaml').read_text())['env']
SOURCE = yaml.safe_load((ROOT / 'data/cfg/multi_agent/approach_scenario_stage1_unified_owner_holding.yaml').read_text())['env']


class Wrapper(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.a2c_network = network


def network(stage2):
    name = ('amp_ma_stage2_unified_owner_holding.yaml' if stage2 else
            'amp_ma_carry_relation.yaml')
    params = yaml.safe_load((ROOT / 'data/cfg/train/rlg' / name).read_text())['params']['network']
    builder = AMPMultiAgentBuilder()
    builder.load(params)
    env = ENV if stage2 else SOURCE
    return builder.build('stage2_unified_owner_test', actions_num=32,
        input_shape=(648,), amp_input_shape=(1320,), value_size=1,
        num_agents=2, num_objects=4, humanoid_obs_size=230, object_obs_size=39,
        goal_obs_size=6, observation_mode='clean_scene',
        scene_entity_sizes=[223, 30, 1], scene_kinematic_size=7,
        scene_arena_scale=5., relation_reward_mode=env['relationReward']['mode'],
        relation_graph_spec=env['relationGraph'], device='cpu')


def test_config_inherits_stage1_and_samples_canonical_cooperation():
    validate_unified_env(ENV)
    validate_relation_config(ENV['relationReward'])
    validate_sampler(ENV['relationGraph'], 2, 4)
    for field in ('templateRsi', 'skillDiscProb', 'ampTaskConditioning',
                  'goalRotation', 'box'):
        assert ENV[field] == SOURCE[field]
    for field in ('hard_skill_training', 'task_sharing', 'edge_aggregation',
                  'success', 'observation'):
        assert ENV['relationReward'][field] == SOURCE['relationReward'][field]
    graph = sample_graph(1000, ENV['relationGraph'],
                         generator=torch.Generator().manual_seed(29))
    family = classify_family(graph)
    proportions = torch.bincount(family, minlength=4).float() / len(family)
    torch.testing.assert_close(proportions, torch.tensor([.1, .3, .3, .3]),
                               atol=.04, rtol=0)
    for preset, downstream in (('place_climb', CLIMB), ('place_sit', SIT),
                               ('place_stack', ON_TOP)):
        graph = sample_graph(32, ENV['relationGraph'], preset=preset,
                             generator=torch.Generator().manual_seed(4))
        assert (classify_family(graph) == {CLIMB: 1, SIT: 2, ON_TOP: 3}[downstream]).all()
        for row in range(32):
            active = graph.edge_valid[row]
            relation = graph.edge_relation[row]
            at = (active & (relation == AT)).nonzero().item()
            hold = (active & (relation == HOLDING) &
                    (graph.edge_owner[row] == graph.edge_owner[row, at])).nonzero().item()
            carrier = graph.edge_owner[row, at].item()
            assert graph.edge_src[row, hold] == carrier
            assert graph.edge_dst[row, hold] == 2 + carrier
            assert graph.edge_src[row, at] == 2 + carrier
            assert graph.edge_dst[row, at] == 6 + carrier
            down = (active & (relation == downstream)).nonzero().item()
            assert graph.edge_owner[row, down] == 1 - carrier
            assert graph.edge_dst[row, down] == 2 + carrier
        assert owner_holding_graph_packet(graph, torch.rand(32, 4)).shape == (32, 24)
    independent = sample_graph(2000, ENV['relationGraph'], preset='independent',
                                generator=torch.Generator().manual_seed(5))
    env_ids = torch.arange(2000).repeat_interleave(2)
    owners = torch.arange(2).repeat(2000)
    templates = classify_templates(independent, env_ids, owners, True).reshape(2000, 2)
    frequencies = torch.bincount(templates.flatten(), minlength=5).float() / 4000
    torch.testing.assert_close(frequencies, torch.tensor([.05, .05, .2, .35, .35]),
                               atol=.025, rtol=0)
    assert (classify_family(independent) == 0).all()


def test_owner_source_checkpoint_transfers_without_changing_initial_action():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    source = network(False)
    target = network(True)
    checkpoint = {'model': Wrapper(source).state_dict(),
                  'relation_metadata': checkpoint_metadata(SOURCE['relationReward']),
                  'epoch': 2500}
    report = transfer_stage1_weights(Wrapper(target), checkpoint)
    assert report['source_variant'] == SOURCE['relationReward']['stage1_variant']
    assert all(not p.requires_grad for p in target.actor_encoder.parameters())
    graph = sample_graph(2, ENV['relationGraph'], preset='place_stack')
    packet = owner_holding_graph_packet(graph, torch.rand(2, 4))
    poses = torch.randn(2, 8, 7)
    poses[..., 3:] = torch.nn.functional.normalize(poses[..., 3:], dim=-1)
    obs = torch.cat((torch.randn(2, 568), poses.flatten(1), packet), -1)
    source.eval(); target.eval()
    torch.testing.assert_close(source.eval_actor(obs)[0], target.eval_actor(obs)[0],
                               atol=1e-5, rtol=1e-5)
    optimizer = torch.optim.Adam((p for p in target.parameters() if p.requires_grad), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad()
        target.eval_actor(obs)[0].square().mean().backward()
        optimizer.step()
    assert target.coordination.grounding[0].weight.grad.abs().sum() > 0
    wrong = deepcopy(checkpoint)
    wrong['relation_metadata']['packet_version'] = 3
    with pytest.raises(ValueError, match='matching Stage-1'):
        transfer_stage1_weights(Wrapper(network(True)), wrong)


def test_coordination_uses_owner_state_and_ignores_edge_slot_order():
    torch.manual_seed(13)
    graph = sample_graph(2, ENV['relationGraph'], preset='place_stack')
    hold = graph.edge_valid & (graph.edge_relation == HOLDING)
    phi0 = torch.zeros(2, 4)
    phi1 = phi0.clone()
    phi1[hold] = 1.
    encoder = EdgeEncoder(4, 2, num_relation_types=11)
    fusion = PackedEdgeOwnerHoldingFusion()
    with torch.no_grad():
        fusion.state_mlp[0].weight.zero_()
        fusion.state_mlp[0].bias.fill_(1.)
        fusion.state_mlp[0].weight[0, -1] = 1.
        fusion.state_mlp[2].weight.zero_()
        fusion.state_mlp[2].bias.zero_()
        fusion.state_mlp[2].weight[0, 0] = 1.
    module = GroundedEdgeCoordination()
    humans = torch.randn(2, 2, 64)
    nodes = torch.randn(2, 8, 64)
    types = torch.tensor([0, 0, 1, 1, 1, 1, 2, 2])
    result0 = module(humans, nodes, owner_holding_graph_packet(graph, phi0),
                     encoder, types, fusion)
    result1 = module(humans, nodes, owner_holding_graph_packet(graph, phi1),
                     encoder, types, fusion)
    assert (result1 - result0).abs().max() > 1e-5
    order = torch.tensor([[2, 0, 3, 1], [1, 3, 0, 2]])
    shuffled = permute_graph(graph, order)
    shuffled_phi = phi1.gather(1, order)
    same = module(humans, nodes, owner_holding_graph_packet(shuffled, shuffled_phi),
                  encoder, types, fusion)
    torch.testing.assert_close(same, result1, atol=1e-6, rtol=1e-6)
