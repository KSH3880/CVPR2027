from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from utils.before_spec import (environment_families, sample_graph, sample_sizes,
                               validate_env, family_probabilities)
from utils.edge_context_spec import AT, HOLDING
from utils.edge_ontop_spec import permute_graph
from utils.edge_stage2_spec import classify_family
from utils.task_role_spec import task_graph_packet
from env.tasks.multi_agent.before_reward import BeforeRuntime
from env.tasks.multi_agent.edge_stage1_reward import Stage1ContextRuntime
from env.tasks.multi_agent.before_reset import align_transform, transform_states, transform_amp
from learning.multi_agent.task_coordination import TaskCoordination, task_relations
from learning.multi_agent.task_role_encoder import TaskTypeEmbeddingBias

ROOT = Path(__file__).resolve().parents[1]
ENV = yaml.safe_load((ROOT/'data/cfg/multi_agent/approach_stage2_before_task_embedding.yaml').read_text())['env']
SPEC = ENV['relationGraph']


def test_sampling_sizes_families_and_partial_reset():
    torch.manual_seed(23)
    family = environment_families(2048, 'cpu')
    assert torch.bincount(family).tolist() == [410, 546, 546, 546]
    graph = sample_graph(2048, SPEC, families=family)
    assert torch.equal(classify_family(graph), family)
    packet = task_graph_packet(graph, 2048).reshape(-1, 2, 5)
    assert (packet[family > 0, 0, 1] == 2).all()
    assert (packet[family > 0, 1, 4] == 2).all()
    assert (graph.prereq_mask.sum((1, 2)) == (family > 0)).all()
    assert not (task_relations(packet, 2, before=True) == 2).any()
    ids = torch.arange(0, 2048, 3)
    reset = sample_graph(len(ids), SPEC, families=family[ids])
    assert torch.equal(classify_family(reset), family[ids])
    independent = task_graph_packet(sample_graph(30000, SPEC, preset='independent'), 30000).reshape(-1, 2, 5)
    for owner in (0, 1):
        rates = torch.bincount(independent[:, owner, 1].long(), minlength=4)/30000
        torch.testing.assert_close(rates, torch.tensor([.1, .25, .325, .325]), atol=.012, rtol=0)
    sizes = sample_sizes(2048, 'cpu')
    assert (sizes >= torch.tensor([.5, .5, .35])).all()
    assert (sizes <= torch.tensor([.6, .6, .45])).all()
    assert torch.unique((sizes*20).round().reshape(-1, 3), dim=0).shape[0] == 27
    expected = torch.tensor([.65, .1, .25])*(410/2048) + torch.tensor([2., .5, .5])*(546/2048)
    torch.testing.assert_close(family_probabilities(family, 'random_scenario'), expected)


def test_rsi_cache_uses_shared_source_and_support_sizes():
    from env.tasks.multi_agent.before_rsi_cache import BeforeSizeRsiCache
    from utils.size_rsi import size_key
    cache = BeforeSizeRsiCache.__new__(BeforeSizeRsiCache)
    cache.families = [0, 1, 2, 3]
    cache.env = SimpleNamespace(_before_families=torch.tensor(cache.families))
    cache.sizes = [[[.5, .5, .35], [.55, .55, .4], [.6, .6, .45], [.5, .6, .4]] for _ in range(4)]
    source, support = cache.objects(torch.arange(4), 1)
    assert source.tolist() == [1, 0, 0, 1]
    assert support.tolist() == [3, 3, 3, 0]
    for row in range(4):
        assert cache._key('sit', row, 1) == ('sit', size_key(cache.sizes[row][source[row]]))
        assert cache._key('putDownOnTop', row, 1) == ('putDownOnTop',
            size_key(cache.sizes[row][source[row]]), size_key(cache.sizes[row][support[row]]))


@pytest.mark.parametrize('preset', ['before_climb', 'before_sit', 'before_stack'])
def test_current_gate_no_latch_progress_holding_and_saturation(preset):
    graph = sample_graph(4, SPEC, preset=preset)
    runtime = BeforeRuntime(4, graph, deepcopy(ENV['relationReward']), 'cpu')
    phi = torch.ones(4, 4)
    progress = torch.full_like(phi, .37)
    zero = torch.zeros_like(phi)
    dependent = graph.prereq_mask.any(-1)
    at = (graph.edge_relation == AT) & graph.edge_valid
    hold_b = (graph.edge_relation == HOLDING) & (graph.edge_owner == 1) & graph.edge_valid
    for delivered in (False, True, False):
        phi[at] = 1. if delivered else 0.
        result = runtime.step(phi, progress, zero, zero, zero)
        assert (result['own_success'][dependent] == delivered).all()
        assert (result['reward_saturated'][dependent] == delivered).all()
        expected = .2 if delivered else 0.
        torch.testing.assert_close(result['state_component'][dependent], torch.full((4,), expected))
        torch.testing.assert_close(result['success_component'][dependent], torch.full((4,), expected))
        torch.testing.assert_close(result['progress_component'][dependent], torch.full((4,), .2 if delivered else .074))
        assert (result['total'][hold_b] > 0).all()
        assert (runtime.done[:, 1] == delivered).all()
    runtime.reset(torch.arange(4), phi, zero, zero, zero)
    assert not runtime.done.any()
    # AT remains sufficient when A releases its grasp.
    phi[at] = 1.
    phi[(graph.edge_relation == HOLDING) & (graph.edge_owner == 0)] = 0.
    result = runtime.step(phi, progress, zero, zero, zero)
    assert result['own_success'][dependent].all()


def test_independent_reward_is_stage1_and_edge_permutation():
    graph = sample_graph(40, SPEC, preset='independent')
    reward = deepcopy(ENV['relationReward'])
    phi, progress = torch.rand(40, 4), torch.rand(40, 4)
    zero = torch.zeros_like(phi)
    a = BeforeRuntime(40, graph, reward, 'cpu').step(phi, progress, zero, zero, zero)
    b = Stage1ContextRuntime(40, graph, reward, 'cpu').step(phi, progress, zero, zero, zero)
    for key in ('total', 'own_success', 'agent_task_reward', 'reward_saturated'):
        torch.testing.assert_close(a[key], b[key])
    graph = sample_graph(40, SPEC, preset='before_stack')
    order = torch.rand(40, 4).argsort(-1)
    a = BeforeRuntime(40, graph, reward, 'cpu').step(phi, progress, zero, zero, zero)
    b = BeforeRuntime(40, permute_graph(graph, order), reward, 'cpu').step(
        phi.gather(1, order), progress.gather(1, order), zero, zero, zero)
    torch.testing.assert_close(a['agent_task_reward'], b['agent_task_reward'])


def test_ca_direction_task_order_and_gradient():
    torch.manual_seed(7)
    graph = sample_graph(24, SPEC)
    packet = task_graph_packet(graph, 24).reshape(24, 2, 5)
    relations = task_relations(packet, 2, True)
    expected = torch.tensor([[1, 0], [3, 1]]).expand(24, -1, -1).clone()
    expected[classify_family(graph) == 0, 1, 0] = 0
    torch.testing.assert_close(relations, expected)
    module = TaskCoordination(before=True)
    embedding = TaskTypeEmbeddingBias(4, 2).requires_grad_(False)
    nodes = torch.randn(24, 8, 64, requires_grad=True)
    module.relation_bias.data.normal_()
    output = module(nodes[:, :2], nodes, packet, embedding)
    shuffled = module(nodes[:, :2], nodes, packet[:, [1, 0]], embedding)
    torch.testing.assert_close(output, shuffled)
    grad = torch.autograd.grad(output.square().sum(), (nodes, module.relation_bias), retain_graph=True)
    other = torch.autograd.grad(shuffled.square().sum(), (nodes, module.relation_bias))
    for x, y in zip(grad, other):
        torch.testing.assert_close(x, y)
    assert (grad[1][:, [0, 1, 3]].abs().sum(0) > 0).all()
    assert not grad[1][:, 2].any()


def test_shared_reset_transform_preserves_relative_geometry_velocities_and_amp():
    source = torch.tensor([[1., 2., .2, 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]])
    target = source.clone()
    target[:, :3] = torch.tensor([4., -1., .6])
    target[:, 5:7] = 2**-.5
    rotation, translation = align_transform(source, target)
    torch.testing.assert_close(transform_states(source, rotation, translation), target)
    body = source[:, None].expand(1, 3, -1).clone()
    body[:, :, 0] += torch.arange(3)
    body[..., 7] = 2.
    moved = transform_states(body, rotation, translation)
    torch.testing.assert_close(moved[..., 8], torch.full((1, 3), 2.))
    torch.testing.assert_close((moved[..., :3]-target[:, None, :3]).norm(dim=-1), torch.arange(3).float()[None])
    task = SimpleNamespace(_num_amp_obs_steps=4,
        _before_amp_rotation=rotation[:, None].expand(1, 2, -1),
        _before_amp_translation=translation[:, None].expand(1, 2, -1))
    slots = (torch.tensor([0]), torch.tensor([1]))
    transformed = transform_amp(task, slots, body[0, :, :3], body[0, :, 3:7],
        body[0, :, 7:10], body[0, :, 10:13], body[0, :, None, :3])
    for actual, expected in zip(transformed[:4], (moved[0, :, :3], moved[0, :, 3:7], moved[0, :, 7:10], moved[0, :, 10:13])):
        torch.testing.assert_close(actual, expected)


def build_network():
    from learning.multi_agent.amp_network_builder_ma import AMPMultiAgentBuilder
    train = yaml.safe_load((ROOT/'data/cfg/train/rlg/amp_ma_stage2_before_task_embedding.yaml').read_text())
    builder = AMPMultiAgentBuilder()
    builder.load(train['params']['network'])
    return builder.build('amp', actions_num=32, input_shape=(634,), amp_input_shape=(1320,),
        value_size=1, num_agents=2, num_objects=4, humanoid_obs_size=230,
        object_obs_size=39, goal_obs_size=6, observation_mode='clean_scene',
        scene_entity_sizes=[223, 30, 1], scene_kinematic_size=7, scene_arena_scale=5.,
        relation_reward_mode=ENV['relationReward']['mode'], relation_graph_spec=SPEC, device='cpu')


def test_config_checkpoint_transfer_and_source_isolation():
    from utils.relation_task_spec import validate_relation_config, checkpoint_metadata, check_checkpoint_metadata
    from learning.multi_agent.stage2_transfer import transfer_stage1_weights
    from test_joint_carry import build_network as joint_network
    from test_stage2_coordination import Wrapper
    from test_task_role_message import observations
    from utils.task_role_spec import CARRY_DISTILL_VARIANT
    validate_env(ENV)
    validate_relation_config(ENV['relationReward'])
    source, target = Wrapper(joint_network(False)), Wrapper(build_network())
    reward = yaml.safe_load((ROOT/'data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_distill.yaml').read_text())['env']['relationReward']
    checkpoint = dict(model=source.state_dict(), relation_metadata=checkpoint_metadata(reward))
    report = transfer_stage1_weights(target, checkpoint)
    assert len(report['frozen_parameters']) == 69
    obs = observations(sample_graph(4, SPEC))
    with torch.no_grad():
        expected = source.a2c_network.action_head(target.a2c_network.actor_encoder(obs)).flatten(0, 1)
        actual, _ = target.a2c_network.eval_actor(obs)
        torch.testing.assert_close(actual, expected)
    checkpoint['relation_metadata']['relation_reward_config']['stage1_variant'] = CARRY_DISTILL_VARIANT
    with pytest.raises(ValueError, match='four-task'):
        transfer_stage1_weights(target, checkpoint)
    joint_reward = yaml.safe_load((ROOT/'data/cfg/multi_agent/approach_stage2_joint_carry_mixed80_task_embedding.yaml').read_text())['env']['relationReward']
    with pytest.raises(ValueError):
        check_checkpoint_metadata({'relation_metadata': checkpoint_metadata(joint_reward)}, checkpoint_metadata(ENV['relationReward']))


def test_full_actor_entity_and_task_shuffle_gradient_reload():
    from test_task_role_message import observations
    net = build_network()
    obs = observations(sample_graph(8, SPEC))
    def action(observation, order):
        humans, nodes = net.actor_encoder(observation, return_all=True, token_order=order)
        context = net.coordination(humans, nodes, observation[:, -10:], net.actor_encoder.edge_encoder)
        return net.action_head(torch.cat((humans, context), -1))
    # Give the new branch nonzero influence; [W, 0] alone would hide CA bugs.
    net.action_head[0][0].weight.data.normal_(std=.01)
    net.coordination.relation_bias.data.normal_()
    original = action(obs, torch.arange(8))
    swapped = obs.clone()
    swapped[:, -10:] = obs[:, -10:].reshape(-1, 2, 5).flip(1).flatten(1)
    shuffled = action(swapped, torch.tensor([7, 2, 0, 4, 1, 6, 3, 5]))
    torch.testing.assert_close(original, shuffled, atol=1e-6, rtol=1e-5)
    original.square().mean().backward()
    gradients = {name: p.grad.clone() for name, p in net.named_parameters() if p.grad is not None}
    net.zero_grad(set_to_none=True)
    shuffled.square().mean().backward()
    for name, p in net.named_parameters():
        if name in gradients:
            torch.testing.assert_close(gradients[name], p.grad, atol=2e-6, rtol=2e-4)
    restored = build_network()
    restored.load_state_dict(net.state_dict(), strict=True)
    torch.testing.assert_close(net.eval_actor(obs)[0], restored.eval_actor(obs)[0], atol=1e-6, rtol=1e-5)
