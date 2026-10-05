from copy import deepcopy
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import pytest
import torch
import yaml

from learning.multi_agent.amp_network_builder_ma import AMPMultiAgentBuilder, EdgeEncoder
from learning.multi_agent.edge_context_encoder import PackedEdgeSemanticFusion
from utils.edge_ontop_spec import permute_graph
from utils.edge_scenario_spec import compose_canonical_graph
from utils.edge_stage1_spec import semantic_graph_packet
from utils.relation_task_spec import (validate_relation_config, checkpoint_metadata,
                                     check_checkpoint_metadata)
from utils.unified_training import validate_unified_env, validate_typed_bias_config

ROOT = Path(__file__).resolve().parents[1]
BASE = 'approach_scenario_stage1_unified_size_rsi'
CFG = yaml.safe_load((ROOT / f'data/cfg/multi_agent/{BASE}_typed_bias.yaml').read_text())['env']
TRAIN = yaml.safe_load((ROOT / 'data/cfg/train/rlg/amp_ma_carry_relation_unified_size_rsi_typed_bias.yaml').read_text())


def network(train=TRAIN, config=CFG):
    torch.set_num_threads(1)
    builder = AMPMultiAgentBuilder()
    builder.load(train['params']['network'])
    net = builder.build('amp', actions_num=32, input_shape=(644,), amp_input_shape=(1320,),
        value_size=1, num_agents=2, num_objects=4, humanoid_obs_size=230,
        object_obs_size=39, goal_obs_size=6, observation_mode='clean_scene',
        scene_entity_sizes=[223, 30, 1], scene_kinematic_size=7, scene_arena_scale=5.,
        relation_reward_mode=config['relationReward']['mode'],
        relation_graph_spec=config['relationGraph'], device='cpu')
    for enc in (net.actor_encoder, net.critic_encoder):
        enc.gta_diagnostics_first_forward = False
    return net


def observations(graph):
    n = graph.edge_valid.shape[0]
    kin = torch.randn(n, 8, 7)
    kin[..., 3:] = torch.nn.functional.normalize(kin[..., 3:], dim=-1)
    return torch.cat((torch.randn(n, 568), kin.flatten(1), semantic_graph_packet(graph, n)), -1)


def test_config_preserves_size_rsi_and_rejects_mismatched_architecture():
    base = yaml.safe_load((ROOT / f'data/cfg/multi_agent/{BASE}.yaml').read_text())['env']
    expected = deepcopy(base)
    expected['relationReward']['stage1_variant'] += '_typed_bias'
    assert CFG == expected
    train = yaml.safe_load((ROOT / 'data/cfg/train/rlg/amp_ma_carry_relation.yaml').read_text())
    expected_train = deepcopy(train)
    expected_train['params']['network']['transformer']['relation_bias_mode'] = 'typed_lookup'
    assert TRAIN == expected_train
    validate_relation_config(CFG['relationReward'])
    validate_unified_env(CFG)
    validate_typed_bias_config(CFG, TRAIN)
    for env, wrong_train in ((CFG, train), (base, TRAIN)):
        with pytest.raises(ValueError, match='must be paired'):
            validate_typed_bias_config(env, wrong_train)
    for flag, value in (('share_edge_encoder', True), ('relation_bias', False)):
        bad = deepcopy(TRAIN)
        bad['params']['network']['transformer'][flag] = value
        with pytest.raises(ValueError, match='separate actor/critic'):
            validate_typed_bias_config(CFG, bad)
    for old, new in ((base, CFG), (CFG, base)):
        with pytest.raises(ValueError):
            check_checkpoint_metadata({'relation_metadata': checkpoint_metadata(old['relationReward'])},
                                      checkpoint_metadata(new['relationReward']))


def test_all_task_pairs_directed_edges_background_and_goal_slot_swap():
    net = network()
    enc = net.actor_encoder
    table = enc.edge_encoder.bias_table
    assert table.shape == (3, 11, 3, 4, 2)
    assert table.numel() == 792
    assert torch.count_nonzero(table) == 0
    assert enc.edge_encoder is not net.critic_encoder.edge_encoder
    assert list(enc.edge_encoder.state_dict()) == ['bias_table']
    with torch.no_grad():
        table.copy_(torch.arange(792).reshape_as(table) / 100)
    graph = compose_canonical_graph(torch.cartesian_prod(torch.arange(5), torch.arange(5)))
    dst = graph.edge_dst.clone()
    at = graph.edge_valid & (graph.edge_relation == 7)
    dst[at] = 13 - dst[at]
    graph = replace(graph, edge_dst=dst)
    packet = semantic_graph_packet(graph, 25)
    actual = enc.context_fusion(packet, enc.edge_encoder, enc.entity_types, enc.build_relation_bias())
    expected = torch.empty_like(actual)
    for batch in range(25):
        for src in range(8):
            for dst in range(8):
                relation = 1 if src == dst else 0
                expected[:, batch, :, src, dst] = table[enc.entity_types[src], relation, enc.entity_types[dst]]
        for edge in range(4):
            if graph.edge_valid[batch, edge]:
                src, dst, relation = (int(getattr(graph, key)[batch, edge])
                                     for key in ('edge_src', 'edge_dst', 'edge_relation'))
                expected[:, batch, :, src, dst] = table[enc.entity_types[src], relation, enc.entity_types[dst]]
    torch.testing.assert_close(actual, expected)
    order = torch.rand(25, 4).argsort(-1)
    shuffled = semantic_graph_packet(permute_graph(graph, order), 25)
    reordered = enc.context_fusion(shuffled, enc.edge_encoder, enc.entity_types, enc.build_relation_bias())
    torch.testing.assert_close(actual, reordered, atol=0, rtol=0)


def test_table_can_reproduce_existing_mlp_bias():
    torch.manual_seed(41)
    enc = network().actor_encoder
    old = EdgeEncoder(4, 2, num_relation_types=11)
    with torch.no_grad():
        old.bias_projection.normal_(std=.2)
        for src in range(3):
            for relation in range(11):
                for dst in range(3):
                    x = torch.cat((old.source_type_embed.weight[src], old.relation_embed.weight[relation],
                                   old.target_type_embed.weight[dst]))
                    enc.edge_encoder.bias_table[src, relation, dst].copy_(old.bias_projection @ old.edge_mlp(x))
    graph = compose_canonical_graph(torch.cartesian_prod(torch.arange(5), torch.arange(5)))
    packet = semantic_graph_packet(graph, 25)
    expected = PackedEdgeSemanticFusion()(packet, old, enc.entity_types, old(enc.entity_types, enc.rel_matrix))
    actual = enc.context_fusion(packet, enc.edge_encoder, enc.entity_types, enc.build_relation_bias())
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('branch', ['actor', 'critic'])
def test_actions_values_and_gradients_survive_token_and_edge_permutations(branch):
    torch.manual_seed(17)
    net = network()
    enc, head = (net.actor_encoder, net.action_head) if branch == 'actor' else (net.critic_encoder, net.value_head)
    with torch.no_grad():
        enc.edge_encoder.bias_table.uniform_(-1, 1)
    graph = compose_canonical_graph(torch.cartesian_prod(torch.arange(5), torch.arange(5)))
    obs = observations(graph)
    identity = torch.arange(8)
    permutation = torch.tensor([6, 4, 1, 3, 7, 0, 5, 2])
    original = head(enc(obs, token_order=identity))
    original.square().mean().backward()
    grads = {name: p.grad.clone() for name, p in enc.named_parameters() if p.grad is not None}
    net.zero_grad(set_to_none=True)
    order = torch.rand(25, 4).argsort(-1)
    shuffled = torch.cat((obs[:, :-20], semantic_graph_packet(permute_graph(graph, order), 25)), -1)
    result = head(enc(shuffled, token_order=permutation))
    torch.testing.assert_close(original, result, atol=1e-5, rtol=1e-5)
    result.square().mean().backward()
    for name, p in enc.named_parameters():
        if name in grads:
            torch.testing.assert_close(grads[name], p.grad, atol=5e-5, rtol=1e-3)
    assert torch.isfinite(result).all()


def test_policy_gradient_updates_holding_without_touching_placement_rows():
    torch.manual_seed(71)
    net = network()
    actor = net.actor_encoder.edge_encoder.bias_table
    critic = net.critic_encoder.edge_encoder.bias_table
    graph = compose_canonical_graph(torch.zeros(4, 2, dtype=torch.long))
    obs = observations(graph)
    optimizer = torch.optim.Adam(net.parameters(), lr=.001)
    before = actor.detach().clone()
    net.action_head(net.actor_encoder(obs)).square().mean().backward()
    assert actor.grad[0, 6, 1].abs().sum() > 0
    assert torch.count_nonzero(actor.grad[:, 7:9]) == 0
    assert critic.grad is None
    optimizer.step()
    assert not torch.equal(before[0, 6, 1], actor[0, 6, 1])
    torch.testing.assert_close(before[:, 7:9], actor[:, 7:9], atol=0, rtol=0)


def test_checkpoint_roundtrip_and_graph_rebuild():
    torch.manual_seed(61)
    net = network()
    net.eval()
    for enc in (net.actor_encoder, net.critic_encoder):
        with torch.no_grad():
            enc.edge_encoder.bias_table.uniform_(-1, 1)
    graph = compose_canonical_graph(torch.tensor([[3, 4], [1, 2]]))
    obs = observations(graph)
    expected_mu = net.eval_actor(obs)[0]
    expected_value = net.eval_critic(obs)
    buffer = BytesIO()
    torch.save(net.state_dict(), buffer)
    buffer.seek(0)
    restored = network()
    restored.eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    for name, value in net.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, atol=0, rtol=0)
    torch.testing.assert_close(restored.eval_actor(obs)[0], expected_mu, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(restored.eval_critic(obs), expected_value, atol=1e-6, rtol=1e-5)
    old_table = restored.actor_encoder.edge_encoder.bias_table.detach().clone()
    restored.set_entity_counts(2, 4)
    assert restored.actor_encoder.build_relation_bias().shape == (4, 2, 8, 8)
    torch.testing.assert_close(restored.actor_encoder.edge_encoder.bias_table, old_table)
    torch.testing.assert_close(restored.eval_actor(obs)[0], expected_mu, atol=1e-6, rtol=1e-5)
