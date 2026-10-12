"""GTA gate math, sparse/dense gradient oracle, runtime binding and regression."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

import test_task_role_message as reference
import test_task_type_embedding as baseline
from learning.multi_agent.task_role_gate import (entmax15, TaskRoleGate, BlockGate,
    gated_scores, gated_values)
from learning.multi_agent.amp_network_builder_ma import build_gta_transforms, apply_gta_transform
from utils.task_role_spec import TASK_GATE_VARIANT, task_graph_packet
from utils.relation_task_spec import checkpoint_metadata, check_checkpoint_metadata, validate_relation_config
from utils.unified_training import validate_unified_env, validate_typed_bias_config

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT/'data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_task_role_gate_entmax.yaml').read_text())['env']
TRAIN = yaml.safe_load((ROOT/'data/cfg/train/rlg/amp_ma_carry_relation_unified_size_rsi_task_role_gate_entmax.yaml').read_text())


@pytest.fixture(autouse=True)
def configuration(monkeypatch):
    monkeypatch.setattr(reference, 'CFG', CFG)
    monkeypatch.setattr(reference, 'TRAIN', TRAIN)


def test_config_parity_and_checkpoint_isolation():
    expected = deepcopy(baseline.CFG)
    expected['relationReward']['stage1_variant'] = TASK_GATE_VARIANT
    assert CFG == expected  # Exact reward/RSI/AMP/size/packet/shuffle parity.
    expected = deepcopy(baseline.TRAIN)
    a, b = expected['params']['network'], TRAIN['params']['network']
    a['transformer'] = b['transformer']
    assert expected == TRAIN
    validate_relation_config(CFG['relationReward'])
    validate_unified_env(CFG)
    validate_typed_bias_config(CFG, TRAIN)
    for old_env, old_train in [(baseline.CFG, baseline.TRAIN), (baseline.shared.CFG, baseline.shared.TRAIN),
                               (baseline.split.CFG, baseline.split.TRAIN)]:
        for env, train in [(CFG, old_train), (old_env, TRAIN)]:
            with pytest.raises(ValueError, match='must be paired'):
                validate_typed_bias_config(env, train)
        for x, y in [(CFG, old_env), (old_env, CFG)]:
            with pytest.raises(ValueError, match='mismatch'):
                check_checkpoint_metadata({'relation_metadata': checkpoint_metadata(x['relationReward'])},
                                          checkpoint_metadata(y['relationReward']))
    for key, value in [('relation_bias', True), ('relation_bias_mode', 'task_type_embedding'),
                       ('relation_message', {'enable': True}), ('share_edge_encoder', True),
                       ('num_layers', 3), ('layer_num_heads', 4), ('attention_normalizer', 'softmax'),
                       ('gta', {'enable': False}), ('task_role_gate', {'enable': True})]:
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


def bisect_reference(scores):
    x = (scores - scores.amax(-1, keepdim=True)) / 2
    lo, hi = x.amax(-1, keepdim=True) - 1, x.amax(-1, keepdim=True)
    for _ in range(80):
        mid = (lo + hi) / 2
        large = (x - mid).clamp_min(0).square().sum(-1, keepdim=True) > 1
        lo, hi = torch.where(large, mid, lo), torch.where(large, hi, mid)
    return (x - (lo + hi) / 2).clamp_min(0).square()


@pytest.mark.parametrize('width', [1, 8, 32])
def test_entmax_reference_gradient_and_extremes(width):
    torch.manual_seed(2)
    x = torch.randn(3, width, dtype=torch.double, requires_grad=True)
    y = entmax15(x)
    torch.testing.assert_close(y, bisect_reference(x), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(y.sum(-1), torch.ones(3, dtype=y.dtype))
    assert (y >= 0).all()
    assert torch.autograd.gradcheck(entmax15, (x,))
    for scores in [torch.zeros(3, width), torch.linspace(-1e6, 1e6, width).expand(3, -1)]:
        scores = scores.clone().requires_grad_()
        p = entmax15(scores)
        torch.testing.assert_close(p.sum(-1), torch.ones(3))
        p.square().sum().backward()
        assert torch.isfinite(scores.grad).all()
        if width > 1 and scores[0, 0] < 0:
            assert p[:, :-1].count_nonzero() == 0
    torch.testing.assert_close(entmax15(x.T, dim=0).T, y)


def test_gate_roles_padding_initialization_and_gta_commutation():
    encoder, gate = TaskRoleGate().double(), BlockGate().double()
    packet = task_graph_packet(reference.all_pairs(), 16)
    condition = encoder(packet)
    src, dst, valid, category, features = condition
    assert torch.equal(category[15, :3], torch.tensor([9, 10, 11]))
    assert category[0, 1] == 1 and category[5, 1] == 4  # sit/climb target role
    assert gate(condition)[3].count_nonzero() == 0
    with torch.no_grad():
        gate.mlp[-1].weight.normal_(std=.1)
    delta = gate(condition)[3]
    blocks = delta.reshape(16, 2, 6, 8, 4)
    torch.testing.assert_close(blocks, blocks[..., :1].expand_as(blocks), atol=0, rtol=0)
    assert (delta[0, :, ~valid[0]] == 0).all()
    assert not torch.equal(delta[15, :, 0], delta[15, :, 1])
    invalid = torch.tensor([[0, -999, -100, 999, -999]], dtype=torch.float)
    assert gate(encoder(invalid))[3].count_nonzero() == 0
    # One block factor commutes with arbitrary homogeneous rigid transforms.
    pose = torch.randn(16, 6, 7, dtype=torch.double)
    pose[..., 3:] = torch.nn.functional.normalize(pose[..., 3:], dim=-1)
    g, _ = build_gta_transforms(pose)
    v = torch.randn_like(delta)
    torch.testing.assert_close((1 + delta) * apply_gta_transform(g, v),
                               apply_gta_transform(g, (1 + delta) * v))
    arbitrary = torch.rand_like(delta)
    assert not torch.allclose(arbitrary * apply_gta_transform(g, v), apply_gta_transform(g, arbitrary * v))


def test_sparse_dense_forward_and_all_gradients_with_duplicates_and_padding():
    torch.manual_seed(31)
    b, h, n, e, d = 2, 2, 5, 4, 32
    q, k, v = [torch.randn(b, h, n, d, dtype=torch.double, requires_grad=True) for _ in range(3)]
    delta = (torch.randn(b, h, e, d, dtype=torch.double) * .2).requires_grad_()
    src = torch.tensor([[0, 0, 0, 3], [1, 2, 3, 0]])
    dst = torch.tensor([[2, 2, 0, 4], [2, 3, 4, 0]])
    valid = torch.tensor([[1, 1, 0, 1], [1, 1, 1, 0]], dtype=torch.bool)
    edges = src, dst, valid, delta * valid[:, None, :, None]
    score = gated_scores(q, k, edges, d ** -.5)
    sparse = gated_values(entmax15(score), v, edges)
    # Independent dense oracle, including additive duplicate corrections.
    pair_delta = torch.zeros(b, h, n, n, d, dtype=torch.double)
    for sample in range(b):
        for edge in range(e):
            if valid[sample, edge]:
                pair_delta[sample, :, src[sample, edge], dst[sample, edge]] += delta[sample, :, edge]
    keys = k[:, :, None] * (1 + pair_delta)
    expected_score = (q[:, :, :, None] * keys).sum(-1) * d ** -.5
    expected = (entmax15(expected_score)[..., None] * v[:, :, None] * (1 + pair_delta)).sum(3)
    torch.testing.assert_close(score, expected_score)
    torch.testing.assert_close(sparse, expected)
    weight = torch.randn_like(sparse)
    inputs = (q, k, v, delta)
    a = torch.autograd.grad((sparse * weight).sum(), inputs, retain_graph=True)
    z = torch.autograd.grad((expected * weight).sum(), inputs)
    for x, y in zip(a, z):
        torch.testing.assert_close(x, y, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize('branch', ['actor', 'critic'])
def test_token_task_edge_permutation_outputs_gradients(branch):
    torch.manual_seed(82)
    net = reference.network()
    enc = getattr(net, branch + '_encoder')
    head = net.action_head if branch == 'actor' else net.value_head
    with torch.no_grad():
        for layer in enc.layers:
            layer.block_gate.mlp[-1].weight.normal_(std=.02)
    graph = reference.all_pairs()
    obs = reference.observations(graph)
    original = head(enc(obs, token_order=torch.arange(8)))
    original.square().mean().backward()
    gradients = {name: p.grad.clone() for name, p in enc.named_parameters() if p.grad is not None}
    assert (enc.context_fusion.task_embed.weight.grad.abs().sum(-1) > 0).all()
    net.zero_grad(set_to_none=True)
    shuffled = reference.permute_graph(graph, torch.rand(16, 4).argsort(-1))
    packet = task_graph_packet(shuffled, 16).reshape(16, 2, 5).flip(1).flatten(1)
    changed = torch.cat((obs[:, :-10], packet), -1)
    result = head(enc(changed, token_order=torch.tensor([6, 4, 1, 3, 7, 0, 5, 2])))
    torch.testing.assert_close(original, result, atol=1e-5, rtol=1e-5)
    result.square().mean().backward()
    for name, p in enc.named_parameters():
        if name in gradients:
            torch.testing.assert_close(gradients[name], p.grad, atol=5e-5, rtol=1e-3)
    assert all(torch.isfinite(v) for v in enc.last_diagnostics.values())


def test_zero_start_two_step_learning_independent_branches_and_task_sensitivity():
    torch.manual_seed(26)
    net = reference.network()
    snapshots = {}
    for branch in ('actor', 'critic'):
        enc = getattr(net, branch + '_encoder')
        assert enc.edge_encoder is None and enc.task_action_fusion is None
        assert enc.build_relation_bias() is None
        snapshots[branch] = {n: p.detach().clone() for n, p in enc.named_parameters()
                             if 'block_gate' in n or 'task_embed' in n}
        for layer in enc.layers:
            assert layer.block_gate.mlp[-1].weight.count_nonzero() == 0
    obs = reference.observations(reference.all_pairs())
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        (net.eval_actor(obs)[0].square().mean() + net.eval_critic(obs).square().mean()).backward()
        for branch in ('actor', 'critic'):
            enc = getattr(net, branch + '_encoder')
            grad = enc.context_fusion.task_embed.weight.grad
            assert torch.isfinite(grad).all()
            assert (grad.count_nonzero() == 0) if step == 0 else (grad.abs().sum(-1) > 0).all()
        optimizer.step()
    for branch in ('actor', 'critic'):
        enc = getattr(net, branch + '_encoder')
        for n, p in enc.named_parameters():
            if n in snapshots[branch]:
                assert not torch.equal(p, snapshots[branch][n]), n
        before = enc(obs, token_order=torch.arange(8))
        changed = obs.clone()
        packet = changed[:, -10:].reshape(16, 2, 5)
        packet[..., 1] = torch.where(packet[..., 1] == 0, 1, packet[..., 1])
        assert not torch.equal(before, enc(changed, token_order=torch.arange(8)))
    assert net.actor_encoder.context_fusion is not net.critic_encoder.context_fusion
    assert sum(p.numel() for p in net.actor_encoder.context_fusion.parameters()) == 256
    assert sum(p.numel() for l in net.actor_encoder.layers for p in l.block_gate.parameters()) == 21568


def test_layer_gate_is_after_gta_and_before_return_with_dense_oracle():
    from learning.multi_agent.amp_network_builder_ma import RelationTransformerLayer
    torch.manual_seed(7)
    layer = RelationTransformerLayer(64, 2, 128, task_gate=True).double()
    fusion = TaskRoleGate().double()
    with torch.no_grad():
        layer.block_gate.mlp[-1].weight.normal_(std=.1)
    x = torch.randn(2, 8, 64, dtype=torch.double)
    pose = torch.randn(2, 8, 7, dtype=torch.double)
    pose[..., 3:] = torch.nn.functional.normalize(pose[..., 3:], dim=-1)
    g, inv = build_gta_transforms(pose)
    condition = fusion(task_graph_packet(reference.all_pairs(), 16)[-2:])
    captured = []
    hook = layer.proj.register_forward_pre_hook(lambda m, args: captured.append(args[0]))
    layer(x, gta_g=g, gta_ginv=inv, gate_condition=condition)
    hook.remove()
    q, k, v = layer.qkv(x).reshape(2, 8, 3, 2, 32).permute(2, 0, 3, 1, 4)
    q, k, v = apply_gta_transform(g.transpose(-1, -2), q), apply_gta_transform(inv, k), apply_gta_transform(inv, v)
    src, dst, valid, delta = layer.block_gate(condition)
    pair_gate = torch.ones(2, 2, 8, 8, 32, dtype=torch.double)
    for b in range(2):
        for e in range(6):
            if valid[b, e]:
                pair_gate[b, :, src[b, e], dst[b, e]] += delta[b, :, e]
    scores = (q[:, :, :, None] * k[:, :, None] * pair_gate).sum(-1) / 32 ** .5
    world = (entmax15(scores)[..., None] * v[:, :, None] * pair_gate).sum(3)
    expected = apply_gta_transform(g, world).transpose(1, 2).reshape(2, 8, 64)
    torch.testing.assert_close(captured[0], expected, atol=1e-12, rtol=1e-12)


def test_non_task_candidates_and_zero_support_distractor_do_not_require_task_edges():
    # Controlled scores: a non-task obstacle is the sole support, while task
    # targets can be zero. Adding low-score distractors leaves aggregation intact.
    q = torch.ones(1, 2, 1, 32, dtype=torch.double)
    k = torch.zeros(1, 2, 3, 32, dtype=torch.double)
    k[:, :, 2] = 10
    v = torch.randn_like(k)
    src, dst = torch.zeros(1, 1, dtype=torch.long), torch.ones(1, 1, dtype=torch.long)
    edges = src, dst, torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, 2, 1, 32, dtype=torch.double)
    scores = gated_scores(q, k, edges, 32 ** -.5)
    alpha = entmax15(scores)
    assert alpha[..., :2].count_nonzero() == 0
    torch.testing.assert_close(alpha[..., 2], torch.ones_like(alpha[..., 2]))
    result = gated_values(alpha, v, edges)
    larger_k = torch.cat((k, -torch.ones(1, 2, 13, 32, dtype=torch.double)), 2)
    larger_v = torch.cat((v, torch.randn(1, 2, 13, 32, dtype=torch.double)), 2)
    extended = gated_values(entmax15(gated_scores(q, larger_k, edges, 32 ** -.5)), larger_v, edges)
    torch.testing.assert_close(result, extended)


@pytest.mark.parametrize('branch', ['actor', 'critic'])
def test_physical_human_object_goal_rebinding_preserves_outputs_and_gradients(branch):
    torch.manual_seed(29)
    net = reference.network()
    enc = getattr(net, branch + '_encoder')
    head = net.action_head if branch == 'actor' else net.value_head
    with torch.no_grad():
        for layer in enc.layers:
            layer.block_gate.mlp[-1].weight.normal_(std=.02)
    obs = reference.observations(reference.all_pairs())
    order = torch.tensor([1, 0, 5, 3, 2, 4, 7, 6])
    inverse = order.argsort()
    humans = obs[:, :446].reshape(16, 2, 223)[:, order[:2]].flatten(1)
    objects = obs[:, 446:566].reshape(16, 4, 30)[:, order[2:6]-2].flatten(1)
    goals = obs[:, 566:568][:, order[6:]-6]
    poses = obs[:, 568:624].reshape(16, 8, 7)[:, order].flatten(1)
    packet = obs[:, -10:].reshape(16, 2, 5).clone()
    packet[..., 2:] = inverse[packet[..., 2:].long()].to(packet.dtype)
    changed = torch.cat((humans, objects, goals, poses, packet.flip(1).flatten(1)), -1)
    original = head(enc(obs, token_order=torch.arange(8)))[:, order[:2]]
    original.square().mean().backward()
    grads = {n: p.grad.clone() for n, p in enc.named_parameters() if p.grad is not None}
    net.zero_grad(set_to_none=True)
    actual = head(enc(changed, token_order=torch.arange(8)))
    torch.testing.assert_close(actual, original, atol=1e-5, rtol=1e-5)
    actual.square().mean().backward()
    for name, p in enc.named_parameters():
        if name in grads:
            torch.testing.assert_close(p.grad, grads[name], atol=5e-5, rtol=1e-3)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_optimized_gate_matches_legacy_lookup_outputs_and_all_gradients(dtype):
    from copy import deepcopy
    torch.manual_seed(41)
    gate = BlockGate().to(dtype=dtype)
    embedding = TaskRoleGate().to(dtype=dtype)
    with torch.no_grad():
        gate.mlp[-1].weight.normal_(std=.1)
    old_gate, old_embedding = deepcopy(gate), deepcopy(embedding)
    packet = task_graph_packet(reference.all_pairs(), 16).repeat(8, 1)
    # Include entirely invalid records and concentrated repeated category rows.
    packet[:16] = packet[-1]
    packet[16:32, 5:] = 0
    condition = embedding(packet)
    result = gate(condition)[3]
    src, dst, valid, category, inputs = old_embedding(packet)
    old_table = old_gate.mlp(inputs).tanh().reshape(12, 2, 8).repeat_interleave(4, -1)
    expected = old_table[category].permute(0, 2, 1, 3) * valid[:, None, :, None]
    torch.testing.assert_close(result, expected, atol=1e-7, rtol=1e-6)
    weights = torch.randn_like(result)
    (result * weights).sum().backward()
    (expected * weights).sum().backward()
    for new, old in zip(list(gate.parameters()) + list(embedding.parameters()),
                        list(old_gate.parameters()) + list(old_embedding.parameters())):
        torch.testing.assert_close(new.grad, old.grad, atol=1e-4, rtol=1e-4)
    # Strict legacy state loading, no-grad rollout, and frozen gate all agree.
    gate.load_state_dict(old_gate.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(gate(embedding(packet))[3], result)
    gate.requires_grad_(False); embedding.requires_grad_(False)
    torch.testing.assert_close(gate(embedding(packet))[3], result)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA precision fallback')
def test_gate_tf32_enabled_preserves_full_precision_lookup():
    torch.manual_seed(71)
    gate, embedding = BlockGate().cuda(), TaskRoleGate().cuda()
    with torch.no_grad():
        gate.mlp[-1].weight.normal_(std=.1)
    packet = task_graph_packet(reference.all_pairs(), 16).cuda()
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        condition = embedding(packet)
        src, dst, valid, category, inputs = condition
        table = gate.mlp(inputs).tanh().reshape(12, 2, 8).repeat_interleave(4, -1)
        expected = table[category].permute(0, 2, 1, 3) * valid[:, None, :, None]
        actual = gate(condition)[3]
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        actual.square().sum().backward()
        assert torch.isfinite(embedding.task_embed.weight.grad).all()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
