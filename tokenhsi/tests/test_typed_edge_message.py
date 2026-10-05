from copy import deepcopy
from dataclasses import replace
from io import BytesIO

import pytest
import torch
import yaml

from test_typed_edge_bias import (ROOT, CFG as BIAS_CFG, TRAIN as BIAS_TRAIN,
                                 network as bias_network, observations)
from learning.multi_agent.amp_network_builder_ma import RelationTransformerLayer, build_gta_transforms
from utils.edge_ontop_spec import permute_graph
from utils.edge_scenario_spec import compose_canonical_graph
from utils.edge_stage1_spec import semantic_graph_packet
from utils.relation_task_spec import (validate_relation_config, checkpoint_metadata,
                                     check_checkpoint_metadata)
from utils.unified_training import validate_unified_env, validate_typed_bias_config

CFG = yaml.safe_load((ROOT / 'data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_message.yaml').read_text())['env']
TRAIN = yaml.safe_load((ROOT / 'data/cfg/train/rlg/amp_ma_carry_relation_unified_size_rsi_typed_bias_message.yaml').read_text())


def network():
    return bias_network(TRAIN, CFG)


def test_config_only_adds_messages_and_isolates_checkpoints():
    expected = deepcopy(BIAS_CFG)
    expected['relationReward']['stage1_variant'] += '_message'
    assert CFG == expected
    expected = deepcopy(BIAS_TRAIN)
    expected['params']['network']['transformer']['relation_message'] = {
        'enable': True, 'alpha': 1., 'init_std': .02}
    assert TRAIN == expected
    validate_relation_config(CFG['relationReward'])
    validate_unified_env(CFG)
    validate_typed_bias_config(CFG, TRAIN)
    for env, train in ((CFG, BIAS_TRAIN), (BIAS_CFG, TRAIN)):
        with pytest.raises(ValueError, match='must be paired'):
            validate_typed_bias_config(env, train)
    for key, value in (('alpha', .5), ('init_std', .2)):
        wrong = deepcopy(TRAIN)
        wrong['params']['network']['transformer']['relation_message'][key] = value
        with pytest.raises(ValueError, match='alpha=1'):
            validate_typed_bias_config(CFG, wrong)
    for old, new in ((BIAS_CFG, CFG), (CFG, BIAS_CFG)):
        with pytest.raises(ValueError):
            check_checkpoint_metadata({'relation_metadata': checkpoint_metadata(old['relationReward'])},
                                      checkpoint_metadata(new['relationReward']))


def test_sparse_lookup_splits_heads_and_masks_non_task_edges():
    net = network()
    enc = net.actor_encoder
    table = enc.relation_message.message_table
    assert table.shape == (3, 11, 3, 64)
    assert table.numel() == 6336
    assert len([k for k in enc.state_dict() if k.endswith('message_table')]) == 1
    assert .018 < table[:, 6:].std().item() < .022
    assert table[:, :6].count_nonzero() == 0
    assert table is not net.critic_encoder.relation_message.message_table
    with torch.no_grad():
        table.copy_(torch.arange(table.numel()).reshape_as(table) / 1000)
    graph = compose_canonical_graph(torch.cartesian_prod(torch.arange(5), torch.arange(5)))
    dst = graph.edge_dst.clone()
    at = graph.edge_valid & (graph.edge_relation == 7)
    dst[at] = 13 - dst[at]
    graph = replace(graph, edge_dst=dst)
    src, dst, values = enc.relation_message(semantic_graph_packet(graph, 25), enc.entity_types)
    for batch in range(25):
        for edge in range(4):
            expected = torch.zeros(2, 32)
            if graph.edge_valid[batch, edge]:
                expected = table[enc.entity_types[src[batch, edge]], graph.edge_relation[batch, edge],
                                 enc.entity_types[dst[batch, edge]]].reshape(2, 32)
            torch.testing.assert_close(values[batch, :, edge], expected)
    relations = graph.edge_relation.clone()
    relations[:, :2] = torch.tensor([0, 1])
    graph = replace(graph, edge_relation=relations)
    values = enc.relation_message(semantic_graph_packet(graph, 25), enc.entity_types)[2]
    assert values[:, :, :2].count_nonzero() == 0


@pytest.mark.parametrize('gta', [False, True])
def test_message_uses_attention_and_receiver_frame_with_exact_gradients(gta):
    torch.manual_seed(9)
    layer = RelationTransformerLayer(64, 2, 128)
    with torch.no_grad():
        layer.qkv.weight.zero_()
        layer.qkv.bias.zero_()
        layer.qkv.weight[128:].copy_(torch.eye(64))
    x = torch.randn(2, 8, 64)
    attention = torch.softmax(torch.randn(2, 2, 8, 8), -1)
    src = torch.tensor([[0, 2, 0], [1, 3, 5]])
    dst = torch.tensor([[2, 6, 3], [3, 5, 1]])
    values = torch.randn(2, 2, 3, 32, requires_grad=True)
    args = {'rel_bias': attention.log()}
    if gta:
        kin = torch.randn(2, 8, 7)
        kin[..., 3:] = torch.nn.functional.normalize(kin[..., 3:], dim=-1)
        args['gta_g'], args['gta_ginv'] = build_gta_transforms(kin, 1.)
    outputs = []
    hook = layer.proj.register_forward_pre_hook(lambda module, inputs: outputs.append(inputs[0]))
    layer(x, **args)
    layer(x, **args, relation_message=(src, dst, values), collect_diagnostics=True)
    hook.remove()
    expected = torch.zeros(2, 8, 2, 32)
    for b in range(2):
        for e in range(3):
            expected[b, src[b, e]] += attention[b, :, src[b, e], dst[b, e], None] * values[b, :, e]
    delta = outputs[1] - outputs[0]
    torch.testing.assert_close(delta, expected.flatten(2), atol=2e-6, rtol=1e-5)
    probe = torch.randn_like(delta)
    actual_grad = torch.autograd.grad((delta * probe).sum(), values, retain_graph=True)[0]
    expected_grad = torch.autograd.grad((expected.flatten(2) * probe).sum(), values)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    assert layer.last_diagnostics['relation_message_rms'] > 0


@pytest.mark.parametrize('branch', ['actor', 'critic'])
def test_all_tasks_actions_values_and_gradients_survive_permutations(branch):
    torch.manual_seed(17)
    net = network()
    enc, head = (net.actor_encoder, net.action_head) if branch == 'actor' else (net.critic_encoder, net.value_head)
    with torch.no_grad():
        enc.edge_encoder.bias_table.uniform_(-1, 1)
    graph = compose_canonical_graph(torch.cartesian_prod(torch.arange(5), torch.arange(5)))
    dst = graph.edge_dst.clone()
    at = graph.edge_valid & (graph.edge_relation == 7)
    dst[at] = 13 - dst[at]
    graph = replace(graph, edge_dst=dst)
    obs = observations(graph)
    original = head(enc(obs, token_order=torch.arange(8)))
    original.square().mean().backward()
    grads = {name: p.grad.clone() for name, p in enc.named_parameters() if p.grad is not None}
    table_grad = enc.relation_message.message_table.grad
    for s, r, t in ((0, 6, 1), (1, 7, 2), (1, 8, 1), (0, 9, 1), (0, 10, 1)):
        assert table_grad[s, r, t].abs().sum() > 0
    assert table_grad[:, :6].count_nonzero() == 0
    net.zero_grad(set_to_none=True)
    order = torch.rand(25, 4).argsort(-1)
    shuffled = torch.cat((obs[:, :-20], semantic_graph_packet(permute_graph(graph, order), 25)), -1)
    result = head(enc(shuffled, token_order=torch.tensor([6, 4, 1, 3, 7, 0, 5, 2])))
    torch.testing.assert_close(original, result, atol=1e-5, rtol=1e-5)
    result.square().mean().backward()
    for name, p in enc.named_parameters():
        if name in grads:
            torch.testing.assert_close(grads[name], p.grad, atol=5e-5, rtol=1e-3)


def test_zero_alpha_matches_old_policy_and_actor_updates_only_active_rows():
    torch.manual_seed(21)
    net = network()
    old = bias_network()
    old.load_state_dict({k: v for k, v in net.state_dict().items() if '.relation_message.' not in k}, strict=True)
    graph = compose_canonical_graph(torch.zeros(4, 2, dtype=torch.long))
    obs = observations(graph)
    for branch in ('actor', 'critic'):
        enc = getattr(net, branch + '_encoder')
        old_enc = getattr(old, branch + '_encoder')
        enc.relation_message_alpha = 0.
        torch.testing.assert_close(enc(obs, token_order=torch.arange(8)),
                                   old_enc(obs, token_order=torch.arange(8)), atol=0, rtol=0)
        enc.relation_message_alpha = 1.
    table = net.actor_encoder.relation_message.message_table
    before = table.detach().clone()
    optimizer = torch.optim.Adam(net.parameters(), lr=2e-5)
    net.action_head(net.actor_encoder(obs)).square().mean().backward()
    assert table.grad[0, 6, 1].abs().sum() > 0
    assert table.grad[:, 7:].count_nonzero() == 0
    assert net.critic_encoder.relation_message.message_table.grad is None
    optimizer.step()
    assert not torch.equal(before[0, 6, 1], table[0, 6, 1])
    torch.testing.assert_close(before[:, 7:], table[:, 7:], atol=0, rtol=0)


def test_strict_checkpoint_roundtrip_and_graph_rebuild():
    net = network().eval()
    graph = compose_canonical_graph(torch.tensor([[3, 4], [1, 2]]))
    obs = observations(graph)
    expected = net.eval_actor(obs)[0]
    expected_value = net.eval_critic(obs)
    buffer = BytesIO()
    torch.save(net.state_dict(), buffer)
    buffer.seek(0)
    restored = network().eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    restored.set_entity_counts(2, 4)
    torch.testing.assert_close(restored.eval_actor(obs)[0], expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(restored.eval_critic(obs), expected_value, atol=1e-6, rtol=1e-5)
    with pytest.raises(RuntimeError, match='Unexpected key'):
        bias_network().load_state_dict(net.state_dict(), strict=True)
