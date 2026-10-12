from copy import deepcopy
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import pytest
import torch
import yaml

from env.tasks.multi_agent.edge_stage1_reward import Stage1ContextRuntime
from env.tasks.multi_agent.edge_ontop_reward import evaluate_ontop_edges
from learning.multi_agent.amp_network_builder_ma import AMPMultiAgentBuilder
from utils.edge_ontop_spec import permute_graph
from utils.edge_scenario_spec import (compose_canonical_graph, sample_graph,
    sample_size_conditioned_graph, classify_templates, agent_object_indices, compile_graph)
from utils.relation_task_spec import (validate_relation_config, checkpoint_metadata,
                                      check_checkpoint_metadata)
from utils.size_rsi import sample_sizes, task_probabilities
from utils.task_role_spec import task_graph_packet, task_size_probabilities, TASK_PROBS
from utils.unified_training import validate_unified_env, validate_typed_bias_config, family_from_templates
from test_typed_edge_message import CFG as OLD, TRAIN as OLD_TRAIN
from test_typed_edge_bias import observations as old_observations

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / 'data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_task_message.yaml').read_text())['env']
TRAIN = yaml.safe_load((ROOT / 'data/cfg/train/rlg/amp_ma_carry_relation_unified_size_rsi_task_message.yaml').read_text())


def network():
    torch.set_num_threads(1)
    builder = AMPMultiAgentBuilder()
    builder.load(TRAIN['params']['network'])
    net = builder.build('amp', actions_num=32, input_shape=(634,), amp_input_shape=(1320,),
        value_size=1, num_agents=2, num_objects=4, humanoid_obs_size=230,
        object_obs_size=39, goal_obs_size=6, observation_mode='clean_scene',
        scene_entity_sizes=[223, 30, 1], scene_kinematic_size=7, scene_arena_scale=5.,
        relation_reward_mode=CFG['relationReward']['mode'],
        relation_graph_spec=CFG['relationGraph'], device='cpu')
    for enc in (net.actor_encoder, net.critic_encoder):
        enc.gta_diagnostics_first_forward = False
    return net


def observations(graph):
    return torch.cat((old_observations(graph)[:, :-20],
                      task_graph_packet(graph, len(graph.edge_valid))), -1)


def all_pairs():
    return compose_canonical_graph(torch.cartesian_prod(torch.arange(1, 5), torch.arange(1, 5)))


def test_config_contract_unchanged_geometry_rsi_amp_and_checkpoint_isolation():
    expected = deepcopy(OLD)
    expected['relationGraph']['policy_task_roles'] = True
    expected['relationGraph']['template_probabilities'] = dict(zip(
        expected['relationGraph']['template_probabilities'], TASK_PROBS))
    expected['relationReward'].update(stage1_variant='scenario_independent_stage1_unified_size_rsi_task_message',
        task_sharing={'self': 1., 'teammate': 0.}, edge_aggregation='self_sum',
        observation={'graph_packet_fields': ['valid', 'task', 'actor', 'payload', 'target']})
    assert CFG == expected
    validate_relation_config(CFG['relationReward'])
    validate_unified_env(CFG)
    validate_typed_bias_config(CFG, TRAIN)
    compile_graph(CFG['relationGraph'], 2, 4)
    for env, train in ((CFG, OLD_TRAIN), (OLD, TRAIN)):
        with pytest.raises(ValueError, match='must be paired'):
            validate_typed_bias_config(env, train)
    for old, new in ((OLD, CFG), (CFG, OLD)):
        with pytest.raises(ValueError, match='mismatch'):
            check_checkpoint_metadata({'relation_metadata': checkpoint_metadata(old['relationReward'])},
                                       checkpoint_metadata(new['relationReward']))
    bad = deepcopy(CFG)
    bad['relationGraph'].pop('policy_task_roles')
    with pytest.raises(ValueError, match='must be paired'):
        validate_unified_env(bad)


def test_all_pairs_packet_binding_owner_and_edge_shuffle():
    graph = all_pairs()
    # Swap physical objects and goals. Actor/owner bindings remain explicit.
    ids = torch.tensor([1, 0, 5, 4, 3, 2, 7, 6])
    graph = replace(graph, edge_src=ids[graph.edge_src], edge_dst=ids[graph.edge_dst],
                    edge_owner=1-graph.edge_owner)
    packet = task_graph_packet(graph, 16).reshape(16, 2, 5)
    for b in range(16):
        for owner in range(2):
            edges = graph.edge_valid[b] & (graph.edge_owner[b] == owner)
            primary = edges & (graph.edge_src[b] < 2)
            p = primary.nonzero()[0, 0]
            secondary = edges & ~primary
            actor, target = int(graph.edge_src[b, p]), int(graph.edge_dst[b, p])
            relation = int(graph.edge_relation[b, p])
            if secondary.any():
                q = secondary.nonzero()[0, 0]
                expected = [1, int(graph.edge_relation[b, q])-5, actor, target, int(graph.edge_dst[b, q])]
            else:
                expected = [1, relation-9, actor, 0, target]
            assert packet[b, owner].tolist() == expected
    shuffled = permute_graph(graph, torch.rand(16, 4).argsort(-1))
    torch.testing.assert_close(task_graph_packet(shuffled, 16), packet.flatten(1), atol=0, rtol=0)


def test_task_bias_and_message_oracle_roles_are_distinct_and_reverse_is_background():
    enc = network().actor_encoder
    fusion = enc.context_fusion
    assert fusion.bias_table.shape == (4, 3, 3, 4, 2)
    assert fusion.message_table.shape == (4, 3, 3, 64)
    assert enc.edge_encoder.bias_table.shape == (3, 2, 3, 4, 2)
    with torch.no_grad():
        fusion.bias_table.copy_(torch.arange(288).reshape_as(fusion.bias_table))
        fusion.message_table.copy_(torch.arange(2304).reshape_as(fusion.message_table))
        enc.edge_encoder.bias_table.fill_(-1)
    packet = task_graph_packet(all_pairs(), 16)
    bias, (src, dst, values) = fusion(packet, enc.build_relation_bias())
    expected = torch.full_like(bias, -1)
    for b, tasks in enumerate(packet.reshape(16, 2, 5).long()):
        for _, task, actor, payload, target in tasks:
            edges = [(actor, target, 0, 2)]
            if task >= 2:
                edges += [(actor, payload, 0, 1), (payload, target, 1, 2)]
            for s, t, sr, tr in edges:
                expected[:, b, :, s, t] = fusion.bias_table[task, sr, tr]
                matches = (src[b] == s) & (dst[b] == t)
                torch.testing.assert_close(values[b, :, matches].sum(1),
                    fusion.message_table[task, sr, tr].reshape(2, 32))
                torch.testing.assert_close(bias[:, b, :, t, s], torch.full((4, 2), -1.))
    torch.testing.assert_close(bias, expected)
    assert not torch.equal(bias[:, 15, :, 0, 2], bias[:, 15, :, 0, 4])


@pytest.mark.parametrize('branch', ['actor', 'critic'])
def test_token_task_and_edge_permutations_preserve_outputs_gradients_and_active_rows(branch):
    torch.manual_seed(82)
    net = network()
    enc = getattr(net, branch + '_encoder')
    head = net.action_head if branch == 'actor' else net.value_head
    with torch.no_grad():
        enc.context_fusion.bias_table.uniform_(-1, 1)
    graph = all_pairs()
    obs = observations(graph)
    original = head(enc(obs, token_order=torch.arange(8)))
    original.square().mean().backward()
    grads = {name: p.grad.clone() for name, p in enc.named_parameters() if p.grad is not None}
    for task in range(4):
        for sr, tr in ((0, 2),) if task < 2 else ((0, 1), (0, 2), (1, 2)):
            assert enc.context_fusion.bias_table.grad[task, sr, tr].abs().sum() > 0
            assert enc.context_fusion.message_table.grad[task, sr, tr].count_nonzero() == 64
    assert enc.context_fusion.message_table.grad[:, 2].count_nonzero() == 0
    net.zero_grad(set_to_none=True)
    shuffled = permute_graph(graph, torch.rand(16, 4).argsort(-1))
    packet = task_graph_packet(shuffled, 16).reshape(16, 2, 5).flip(1).flatten(1)
    obs = torch.cat((obs[:, :-10], packet), -1)
    result = head(enc(obs, token_order=torch.tensor([6, 4, 1, 3, 7, 0, 5, 2])))
    torch.testing.assert_close(original, result, atol=1e-5, rtol=1e-5)
    result.square().mean().backward()
    for name, p in enc.named_parameters():
        if name in grads:
            torch.testing.assert_close(grads[name], p.grad, atol=5e-5, rtol=1e-3)


def test_size_sampling_no_standalone_holding_preserves_family_rsi_bindings():
    torch.manual_seed(172)
    sizes = sample_sizes(12000, 'cpu')
    weights = task_size_probabilities(sizes[:, :2])
    old = task_probabilities(sizes[:, :2])
    assert weights[..., 0].count_nonzero() == 0
    torch.testing.assert_close(weights[..., [0, 3, 4]].sum(-1), old[..., [0, 3, 4]].sum(-1))
    torch.testing.assert_close(weights[..., 1:3], old[..., 1:3], atol=0, rtol=0)
    torch.testing.assert_close(weights.mean((0, 1)), torch.tensor(TASK_PROBS), atol=.008, rtol=0)
    graph = sample_size_conditioned_graph(weights, CFG['relationGraph'], 'random_scenario')
    packet = task_graph_packet(graph, len(sizes)).reshape(-1, 5).long()
    rows = torch.arange(len(sizes)).repeat_interleave(2)
    actors = torch.arange(2).repeat(len(sizes))
    templates = classify_templates(graph, rows, actors, True)
    assert torch.equal(packet[:, 1], templates-1)
    assert torch.equal(packet[:, 2], actors)
    assert torch.equal(agent_object_indices(graph, rows, actors), actors)
    assert torch.equal(family_from_templates(templates),
                       torch.where(packet[:, 1] == 0, 1, torch.where(packet[:, 1] == 1, 2, 0)))
    for preset, task in (('sit', 0), ('climb', 1), ('carry_at', 2), ('carry_ontop', 3)):
        fixed = sample_size_conditioned_graph(weights[:2], CFG['relationGraph'], preset)
        assert (task_graph_packet(fixed, 2).reshape(2, 2, 5)[..., 1] == task).all()
    for sampler in (lambda: sample_graph(2, CFG['relationGraph'], preset='holding'),
                    lambda: sample_size_conditioned_graph(weights[:2], CFG['relationGraph'], 'holding')):
        with pytest.raises(ValueError, match='no standalone holding'):
            sampler()


def test_rewards_equal_old_local_sum_no_teammate_dependency_and_partial_suffix():
    graph = all_pairs()
    new = Stage1ContextRuntime(16, graph, CFG['relationReward'], 'cpu')
    old = Stage1ContextRuntime(16, graph, OLD['relationReward'], 'cpu')
    phi, progress, error = torch.rand(16, 4)*.8, torch.rand(16, 4), torch.ones(16, 4)
    result = new.step(phi, progress, error, error, error)
    expected = old.step(phi, progress, error, error, error)
    torch.testing.assert_close(result['agent_task_reward'], expected['local_task_reward'])
    assert result['teammate_task_reward'].count_nonzero() == 0
    changed = torch.where(graph.edge_owner == 1, .0, phi)
    again = new.step(changed, progress, error, error, error)
    torch.testing.assert_close(again['agent_task_reward'][:, 0], result['agent_task_reward'][:, 0])
    ids = torch.tensor([15, 0, 8, 2])
    torch.testing.assert_close(new.suffix(ids), task_graph_packet(graph, 16)[ids])
    full = new.step(torch.ones(16, 4), torch.ones(16, 4),
                    torch.zeros(16, 4), torch.zeros(16, 4), torch.zeros(16, 4))
    assert full['agent_task_reward'][15].tolist() == pytest.approx([1.2, 1.2])
    assert full['agent_task_reward'][0].tolist() == pytest.approx([.6, .6])


def test_carry_ontop_payload_above_support_and_at_goal_match_packet():
    graph = compose_canonical_graph(torch.tensor([[3, 4]]))
    packet = task_graph_packet(graph, 1).reshape(1, 2, 5).long()
    boxes = torch.zeros(1, 4, 13)
    boxes[..., 6] = 1
    boxes[0, 1, :3] = torch.tensor([2., 3., .7])
    boxes[0, 3, :3] = torch.tensor([2., 3., .25])
    sizes = torch.full((1, 4, 3), .4)
    sizes[0, 3, 2] = .5
    goals = torch.tensor([[[5., 6., .2], [-5., -6., .2]]])
    _, diag = evaluate_ontop_edges(torch.zeros(1, 2, 2, 3), torch.zeros(1, 2, 3),
                                  boxes, sizes, goals, graph, CFG['relationReward'])
    at = (graph.edge_relation[0] == 7).nonzero()[0, 0]
    top = (graph.edge_relation[0] == 8).nonzero()[0, 0]
    assert packet[0, 1].tolist() == [1, 3, 1, 3, 5]
    torch.testing.assert_close(diag['target'][0, at], goals[0, packet[0, 0, 4]-6])
    torch.testing.assert_close(diag['target'][0, top], boxes[0, packet[0, 1, 3]-2, :3])
    assert abs(float(diag['signed_gap'][0, top])) < 1e-6


def test_checkpoint_roundtrip_rebuild_and_actor_critic_separation():
    net = network().eval()
    assert net.actor_encoder.context_fusion is not net.critic_encoder.context_fusion
    obs = observations(all_pairs())
    expected = net.eval_actor(obs)[0]
    value = net.eval_critic(obs)
    stream = BytesIO()
    torch.save(net.state_dict(), stream)
    stream.seek(0)
    restored = network().eval()
    restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    restored.set_entity_counts(2, 4)
    torch.testing.assert_close(restored.eval_actor(obs)[0], expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(restored.eval_critic(obs), value, atol=1e-6, rtol=1e-5)
    with pytest.raises(ValueError, match='packet architecture'):
        restored.actor_encoder.set_task_graph(OLD['relationGraph'])
