"""Task-only embedding, physical-type projection, and unchanged runtime contracts."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

import test_task_role_message as reference
import test_task_role_mlp as shared
import test_task_role_mlp_split as split
from learning.multi_agent.task_role_encoder import TaskTypeEmbeddingBias, TaskTypeEmbeddingFusion
from utils.task_role_spec import TASK_EMBEDDING_VARIANT, task_graph_packet
from utils.relation_task_spec import checkpoint_metadata, check_checkpoint_metadata, validate_relation_config
from utils.unified_training import validate_unified_env, validate_typed_bias_config

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / 'data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding.yaml').read_text())['env']
TRAIN = yaml.safe_load((ROOT / 'data/cfg/train/rlg/amp_ma_carry_relation_unified_size_rsi_task_embedding.yaml').read_text())


@pytest.fixture(autouse=True)
def use_embedding_config(monkeypatch):
    monkeypatch.setattr(reference, 'CFG', CFG)
    monkeypatch.setattr(reference, 'TRAIN', TRAIN)


def test_config_parity_and_checkpoint_isolation():
    expected = deepcopy(shared.CFG)
    expected['relationReward']['stage1_variant'] = TASK_EMBEDDING_VARIANT
    assert CFG == expected  # RSI/AMP/sizes/reward/packet/shuffles unchanged.
    expected_train = deepcopy(shared.TRAIN)
    expected_train['params']['network']['transformer']['relation_bias_mode'] = 'task_type_embedding'
    assert TRAIN == expected_train
    validate_relation_config(CFG['relationReward'])
    validate_unified_env(CFG)
    validate_typed_bias_config(CFG, TRAIN)
    metadata = checkpoint_metadata(CFG['relationReward'])
    assert metadata['packet_version'] == 5
    assert metadata['context_fusion'] == 'task_category_embedding64_type_pair_projection'
    for old_env, old_train in ((shared.MESSAGE_CFG, shared.MESSAGE_TRAIN),
                               (shared.CFG, shared.TRAIN), (split.CFG, split.TRAIN)):
        for env, train in ((CFG, old_train), (old_env, TRAIN)):
            with pytest.raises(ValueError, match='must be paired'):
                validate_typed_bias_config(env, train)
        for a, b in ((CFG, old_env), (old_env, CFG)):
            with pytest.raises(ValueError, match='mismatch'):
                check_checkpoint_metadata({'relation_metadata': checkpoint_metadata(a['relationReward'])},
                                          checkpoint_metadata(b['relationReward']))
    for key, value in [('relation_message', {'enable': True}), ('relation_bias', False),
                       ('share_edge_encoder', True), ('num_layers', 3), ('layer_num_heads', 4)]:
        bad = deepcopy(TRAIN)
        bad['params']['network']['transformer'][key] = value
        with pytest.raises(ValueError):
            validate_typed_bias_config(CFG, bad)


@pytest.mark.parametrize('check', [
    reference.test_all_pairs_packet_binding_owner_and_edge_shuffle,
    reference.test_size_sampling_no_standalone_holding_preserves_family_rsi_bindings,
    reference.test_rewards_equal_old_local_sum_no_teammate_dependency_and_partial_suffix,
    reference.test_carry_ontop_payload_above_support_and_at_goal_match_packet,
    reference.test_checkpoint_roundtrip_rebuild_and_actor_critic_separation,
])
def test_runtime_contracts(check):
    check()


def prepare():
    enc = reference.network().actor_encoder
    with torch.no_grad():
        enc.edge_encoder.bias_projection.normal_()
    return enc


def fused(enc, packet, types=None):
    types = enc.entity_types if types is None else types
    bg = enc.edge_encoder(types, enc.rel_matrix)
    return enc.context_fusion(packet, enc.edge_encoder, types, bg)[0]


def test_all_pairs_manual_oracle_background_direction_and_invalid_records():
    enc = prepare()
    packet = task_graph_packet(reference.all_pairs(), 16)
    # Define expected categories from graph records, independently of task_edges.
    categories = torch.full((16, 8, 8), 4, dtype=torch.long)
    categories.diagonal(dim1=-2, dim2=-1).fill_(5)
    for b, records in enumerate(packet.reshape(16, 2, 5).long()):
        for valid, task, actor, payload, target in records:
            categories[b, actor, target] = task
            if task >= 2:
                categories[b, actor, payload] = task
                categories[b, payload, target] = task
    output = fused(enc, packet)
    expected = torch.empty_like(output)
    for b in range(16):
        for src in range(8):
            for dst in range(8):
                vector = enc.edge_encoder.category_embed.weight[categories[b, src, dst]]
                projection = enc.edge_encoder.bias_projection[enc.entity_types[src], enc.entity_types[dst]]
                expected[:, b, :, src, dst] = (projection * vector).sum(-1)
    torch.testing.assert_close(output, expected)
    empty = torch.zeros_like(packet)
    bg = enc.build_relation_bias()
    torch.testing.assert_close(fused(enc, empty), bg[:, None].expand_as(output))
    # Partial packet must not overwrite edges or diagonal with dummy padding.
    partial = packet.reshape(16, 2, 5).clone()
    partial[:, 1] = 0
    partial_out = fused(enc, partial.flatten(1))
    torch.testing.assert_close(partial_out[:, :, :, 1], bg[:, None, :, 1].expand(4, 16, 2, 8))


def test_type_pair_and_task_sharing_without_agent_or_role_parameters():
    enc = prepare()
    bias = fused(enc, task_graph_packet(reference.all_pairs(), 16))
    # Last scene is carry_ontop for both humans: payload and support are both O.
    torch.testing.assert_close(bias[:, 15, :, 0, 2], bias[:, 15, :, 0, 4], atol=0, rtol=0)
    torch.testing.assert_close(bias[:, 15, :, 0, 2], bias[:, 15, :, 1, 3], atol=0, rtol=0)
    # carry_at/carry_at scene: H->G and O->G use different directed type pairs.
    assert not torch.equal(bias[:, 10, :, 0, 6], bias[:, 10, :, 2, 6])
    before = enc.edge_encoder.bias_table().detach().clone()
    with torch.no_grad():
        enc.edge_encoder.bias_projection[0, 1, 0, 0].add_(1.)
    delta = enc.edge_encoder.bias_table().detach() - before
    # One H->O layer/head vector is shared by ALL six categories, including NONE/SELF.
    assert (delta[:, 0, 1, 0, 0].abs() > 0).all()
    delta[:, 0, 1, 0, 0] = 0
    assert delta.count_nonzero() == 0
    names = list(dict(enc.edge_encoder.named_parameters()))
    assert names == ['bias_projection', 'category_embed.weight']
    assert sum(p.numel() for p in enc.edge_encoder.parameters()) == 4992
    assert list(enc.context_fusion.parameters()) == []
    assert not any('message' in name for name, _ in enc.named_parameters())


def test_arbitrary_endpoint_and_type_permutation_remaps_bias():
    enc = prepare()
    packet = task_graph_packet(reference.all_pairs(), 16)
    base = fused(enc, packet)
    order = torch.tensor([6, 4, 1, 3, 7, 0, 5, 2])
    inverse = order.argsort()
    changed = packet.reshape(16, 2, 5).clone()
    changed[:, :, 2:] = inverse[changed[:, :, 2:].long()].float()
    actual = fused(enc, changed.flatten(1), enc.entity_types[order])
    torch.testing.assert_close(actual, base.index_select(-2, order).index_select(-1, order))


@pytest.mark.parametrize('branch', ['actor', 'critic'])
def test_network_token_edge_task_shuffle_outputs_and_gradients(branch):
    torch.manual_seed(82)
    net = reference.network()
    enc = getattr(net, branch + '_encoder')
    head = net.action_head if branch == 'actor' else net.value_head
    with torch.no_grad():
        enc.edge_encoder.bias_projection.uniform_(-.1, .1)
    graph = reference.all_pairs()
    obs = reference.observations(graph)
    original = head(enc(obs, token_order=torch.arange(8)))
    original.square().mean().backward()
    grads = {n: p.grad.clone() for n, p in enc.named_parameters() if p.grad is not None}
    assert (enc.edge_encoder.category_embed.weight.grad.abs().sum(-1) > 0).all()
    net.zero_grad(set_to_none=True)
    shuffled = reference.permute_graph(graph, torch.rand(16, 4).argsort(-1))
    packet = task_graph_packet(shuffled, 16).reshape(16, 2, 5).flip(1).flatten(1)
    changed = torch.cat((obs[:, :-10], packet), -1)
    result = head(enc(changed, token_order=torch.tensor([6, 4, 1, 3, 7, 0, 5, 2])))
    torch.testing.assert_close(original, result, atol=1e-5, rtol=1e-5)
    result.square().mean().backward()
    for n, p in enc.named_parameters():
        if n in grads:
            torch.testing.assert_close(grads[n], p.grad, atol=5e-5, rtol=1e-3)


def test_zero_projection_all_categories_learn_and_actor_critic_are_independent():
    net = reference.network()
    initial = {}
    for branch in ('actor', 'critic'):
        enc = getattr(net, branch + '_encoder')
        assert isinstance(enc.edge_encoder, TaskTypeEmbeddingBias)
        assert isinstance(enc.context_fusion, TaskTypeEmbeddingFusion)
        assert enc.edge_encoder.bias_projection.shape == (3, 3, 4, 2, 64)
        assert enc.edge_encoder.bias_projection.count_nonzero() == 0
        initial[branch] = enc.edge_encoder.category_embed.weight.detach().clone()
        assert fused(enc, task_graph_packet(reference.all_pairs(), 16)).count_nonzero() == 0
    obs = reference.observations(reference.all_pairs())
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        (net.eval_actor(obs)[0].square().mean() + net.eval_critic(obs).square().mean()).backward()
        optimizer.step()
    for branch in ('actor', 'critic'):
        enc = getattr(net, branch + '_encoder')
        assert enc.edge_encoder.bias_projection.count_nonzero() > 0
        assert ((initial[branch] - enc.edge_encoder.category_embed.weight).abs().sum(-1) > 0).all()
    assert net.actor_encoder.edge_encoder is not net.critic_encoder.edge_encoder
