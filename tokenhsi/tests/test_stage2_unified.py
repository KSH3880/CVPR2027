"""Semantic Stage-2 must preserve the non-shared unified Stage-1 policy."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml

from learning.multi_agent.amp_network_builder_ma import AMPMultiAgentBuilder
from learning.multi_agent.stage2_transfer import transfer_stage1_weights
from utils.edge_stage1_spec import compile_stage1_graph, semantic_graph_packet
from utils.edge_stage2_spec import classify_family, sample_graph, validate_sampler
from utils.edge_scenario_spec import classify_templates
from utils.relation_task_spec import checkpoint_metadata, validate_relation_config
from utils.unified_training import validate_unified_env

ROOT = Path(__file__).resolve().parents[1]
ENV = yaml.safe_load((ROOT / 'data/cfg/multi_agent/approach_stage2_unified.yaml').read_text())['env']
SOURCE = yaml.safe_load((ROOT / 'data/cfg/multi_agent/approach_scenario_stage1_unified.yaml').read_text())['env']


class Wrapper(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.a2c_network = network


def network(stage2):
    name = 'amp_ma_stage2_unified.yaml' if stage2 else 'amp_ma_carry_relation.yaml'
    params = yaml.safe_load((ROOT / 'data/cfg/train/rlg' / name).read_text())['params']['network']
    builder = AMPMultiAgentBuilder()
    builder.load(params)
    env = ENV if stage2 else SOURCE
    return builder.build('stage2_unified_test', actions_num=32,
        input_shape=(644,), amp_input_shape=(1320,), value_size=1,
        num_agents=2, num_objects=4, humanoid_obs_size=230, object_obs_size=39,
        goal_obs_size=6, observation_mode='clean_scene',
        scene_entity_sizes=[223, 30, 1], scene_kinematic_size=7,
        scene_arena_scale=5., relation_reward_mode=env['relationReward']['mode'],
        relation_graph_spec=env['relationGraph'], device='cpu')


def test_inherits_full_stage1_env_reward_and_network():
    validate_unified_env(ENV)
    validate_relation_config(ENV['relationReward'])
    validate_sampler(ENV['relationGraph'], 2, 4)
    assert {k: v for k, v in ENV.items() if k not in ('relationGraph', 'relationReward')} == {
        k: v for k, v in SOURCE.items() if k not in ('relationGraph', 'relationReward')}
    excluded = ('mode', 'schema_version', 'stage1_variant')
    assert {k: v for k, v in ENV['relationReward'].items() if k not in excluded} == {
        k: v for k, v in SOURCE['relationReward'].items() if k not in excluded}
    source_params = yaml.safe_load((ROOT / 'data/cfg/train/rlg/amp_ma_carry_relation.yaml').read_text())['params']
    target_params = yaml.safe_load((ROOT / 'data/cfg/train/rlg/amp_ma_stage2_unified.yaml').read_text())['params']
    assert {k: v for k, v in target_params['network'].items() if k != 'coordination'} == source_params['network']
    assert {k: v for k, v in target_params['config'].items() if k != 'name'} == {
        k: v for k, v in source_params['config'].items() if k != 'name'}
    meta = checkpoint_metadata(ENV['relationReward'])
    assert (meta['packet_version'], meta['graph_record_width'], meta['context_fusion']) == (3, 5, 'semantic_only')
    assert compile_stage1_graph(ENV['relationGraph'], 2, 4).num_objects == 4
    bad = deepcopy(ENV)
    bad['relationGraph']['sampler'] += '_owner'
    with pytest.raises(ValueError, match='own reward variant'):
        validate_unified_env(bad)


def test_cooperative_distribution_and_independent_canonical_bindings():
    graph = sample_graph(1000, ENV['relationGraph'], generator=torch.Generator().manual_seed(29))
    proportions = torch.bincount(classify_family(graph), minlength=4).float() / 1000
    torch.testing.assert_close(proportions, torch.tensor([.1, .3, .3, .3]), atol=.04, rtol=0)
    owner_env = yaml.safe_load((ROOT / 'data/cfg/multi_agent/approach_stage2_unified_owner_holding.yaml').read_text())['env']
    # Changing the packet contract must not change canonical graph semantics or role swaps.
    for preset in ('place_climb', 'place_sit', 'place_stack', 'independent'):
        for swapped in (False, True):
            a = sample_graph(32, ENV['relationGraph'], preset=preset, role_swap=swapped,
                             generator=torch.Generator().manual_seed(4))
            b = sample_graph(32, owner_env['relationGraph'], preset=preset, role_swap=swapped,
                             generator=torch.Generator().manual_seed(4))
            for key in ('edge_src', 'edge_dst', 'edge_owner', 'edge_relation', 'edge_valid', 'required_goal'):
                torch.testing.assert_close(getattr(a, key), getattr(b, key))
            assert semantic_graph_packet(a, 32).shape == (32, 20)
    independent = sample_graph(2000, ENV['relationGraph'], preset='independent',
                               generator=torch.Generator().manual_seed(5))
    templates = classify_templates(independent, torch.arange(2000).repeat_interleave(2),
                                   torch.arange(2).repeat(2000), True)
    frequencies = torch.bincount(templates, minlength=5).float() / 4000
    torch.testing.assert_close(frequencies, torch.tensor([.05, .05, .2, .35, .35]), atol=.025, rtol=0)


def test_transfer_initial_policy_identity_gradients_and_variant_rejection():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    source = network(False)
    target = network(True)
    checkpoint = {'model': Wrapper(source).state_dict(),
                  'relation_metadata': checkpoint_metadata(SOURCE['relationReward']), 'epoch': 7500}
    report = transfer_stage1_weights(Wrapper(target), checkpoint)
    assert len(report['copied_tensors']) == 169
    assert len(report['new_tensors']) == 8
    assert all(not p.requires_grad for p in target.actor_encoder.parameters())
    assert not target.actor_encoder.owner_holding_state
    source.eval(); target.eval()
    for preset in ('place_climb', 'place_sit', 'place_stack', 'independent'):
        graph = sample_graph(2, ENV['relationGraph'], preset=preset)
        poses = torch.randn(2, 8, 7)
        poses[..., 3:] = torch.nn.functional.normalize(poses[..., 3:], dim=-1)
        obs = torch.cat((torch.randn(2, 568), poses.flatten(1), semantic_graph_packet(graph, 2)), -1)
        for a, b in zip(source.eval_actor(obs), target.eval_actor(obs)):
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(source.eval_critic(obs), target.eval_critic(obs), atol=1e-5, rtol=1e-5)
    frozen = {k: v.clone() for k, v in target.actor_encoder.state_dict().items()}
    optimizer = torch.optim.Adam((p for p in target.parameters() if p.requires_grad), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad()
        target.eval_actor(obs)[0].square().mean().backward()
        optimizer.step()
    assert target.coordination.grounding[0].weight.grad.abs().sum() > 0
    assert target.action_head[0][0].weight[:, 64:].abs().sum() > 0
    for k, v in frozen.items():
        torch.testing.assert_close(target.actor_encoder.state_dict()[k], v, atol=0, rtol=0)
    for variant in ('scenario_independent_stage1_unified_owner_holding',
                    'scenario_independent_stage1_unified_shared_edge_encoder',
                    'scenario_independent_stage1_plane'):
        wrong = deepcopy(checkpoint)
        wrong['relation_metadata']['relation_reward_config']['stage1_variant'] = variant
        with pytest.raises(ValueError, match='matching Stage-1'):
            transfer_stage1_weights(Wrapper(target), wrong)
