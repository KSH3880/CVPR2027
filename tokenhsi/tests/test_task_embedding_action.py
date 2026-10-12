"""Zero background and human-only, post-GTA residual task conditioning."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

import test_task_role_message as reference
import test_task_type_embedding as base
from learning.multi_agent.task_role_encoder import TaskActionFusion, task_action_condition
from learning.multi_agent.amp_network_builder_ma import RelationTransformerLayer, build_gta_transforms
from utils.task_role_spec import TASK_EMBEDDING_ACTION_VARIANT, task_graph_packet
from utils.relation_task_spec import checkpoint_metadata, check_checkpoint_metadata, validate_relation_config
from utils.unified_training import validate_unified_env, validate_typed_bias_config

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / 'data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_action.yaml').read_text())['env']
TRAIN = yaml.safe_load((ROOT / 'data/cfg/train/rlg/amp_ma_carry_relation_unified_size_rsi_task_embedding_action.yaml').read_text())


@pytest.fixture(autouse=True)
def use_config(monkeypatch):
    monkeypatch.setattr(reference, 'CFG', CFG)
    monkeypatch.setattr(reference, 'TRAIN', TRAIN)


def test_config_parity_and_exclusive_checkpoint_contract():
    expected = deepcopy(base.CFG)
    expected['relationReward']['stage1_variant'] = TASK_EMBEDDING_ACTION_VARIANT
    assert CFG == expected  # Exact AMP/RSI/reward/size/sampling/shuffle parity.
    expected = deepcopy(base.TRAIN)
    expected['params']['network']['transformer']['relation_bias_mode'] = 'task_type_embedding_action'
    assert TRAIN == expected
    validate_relation_config(CFG['relationReward'])
    validate_unified_env(CFG)
    validate_typed_bias_config(CFG, TRAIN)
    metadata = checkpoint_metadata(CFG['relationReward'])
    assert metadata['context_fusion'] == 'task_embedding4_zero_background_human_residual_mlp128_64_64_shared_layers'
    for old_env, old_train in [(base.CFG, base.TRAIN), (base.shared.CFG, base.shared.TRAIN),
                                (base.split.CFG, base.split.TRAIN),
                                (base.shared.MESSAGE_CFG, base.shared.MESSAGE_TRAIN)]:
        for a, b in [(CFG, old_env), (old_env, CFG)]:
            with pytest.raises(ValueError, match='mismatch'):
                check_checkpoint_metadata({'relation_metadata': checkpoint_metadata(a['relationReward'])},
                                          checkpoint_metadata(b['relationReward']))
        for env, train in [(CFG, old_train), (old_env, TRAIN)]:
            with pytest.raises(ValueError, match='must be paired'):
                validate_typed_bias_config(env, train)
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


def test_task_bias_oracle_and_exact_zero_background_after_updates():
    enc = reference.network().actor_encoder
    with torch.no_grad():
        enc.edge_encoder.category_embed.weight.normal_()
        enc.edge_encoder.bias_projection.normal_()
    packet = task_graph_packet(reference.all_pairs(), 16)
    actual = base.fused(enc, packet)
    expected = torch.zeros_like(actual)
    for b, records in enumerate(packet.reshape(16, 2, 5).long()):
        for valid, task, actor, payload, target in records:
            pairs = [(actor, target)] + ([(actor, payload), (payload, target)] if task >= 2 else [])
            for src, dst in pairs:
                vector = enc.edge_encoder.category_embed.weight[task]
                projection = enc.edge_encoder.bias_projection[enc.entity_types[src], enc.entity_types[dst]]
                expected[:, b, :, src, dst] = (projection * vector).sum(-1)
    torch.testing.assert_close(actual, expected)
    assert enc.edge_encoder.category_embed.weight.shape == (4, 64)
    assert enc.build_relation_bias().count_nonzero() == 0
    assert base.fused(enc, torch.zeros_like(packet)).count_nonzero() == 0
    # Type-pair sharing for payload and support is unchanged.
    torch.testing.assert_close(actual[:, 15, :, 0, 2], actual[:, 15, :, 0, 4], atol=0, rtol=0)
    base.test_arbitrary_endpoint_and_type_permutation_remaps_bias()


def test_human_only_conditioning_manual_oracle_padding_and_task_record_order():
    torch.manual_seed(15)
    fusion = TaskActionFusion()
    torch.nn.init.normal_(fusion.mlp[-1].weight, std=.05)
    embedding = torch.nn.Embedding(4, 64)
    packet = torch.tensor([[1, 2, 1, 2, 6, 1, 0, 0, 0, 3],
                           [1, 3, 0, 2, 4, 0, 0, 0, 0, 0]], dtype=torch.float)
    x = torch.randn(2, 8, 64)
    condition = task_action_condition(packet, embedding)
    actual = fusion(x, condition, True)
    expected = x.clone()
    for b, records in enumerate(packet.reshape(2, 2, 5).long()):
        for valid, task, actor, payload, target in records:
            if valid:
                expected[b, actor] += fusion.mlp(torch.cat((x[b, actor], embedding.weight[task])))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual[:, 2:], x[:, 2:], atol=0, rtol=0)
    torch.testing.assert_close(actual[1, 1], x[1, 1], atol=0, rtol=0)
    flipped = packet.reshape(2, 2, 5).flip(1).flatten(1)
    torch.testing.assert_close(fusion(x, task_action_condition(flipped, embedding)), actual)
    assert all(torch.isfinite(v) for v in fusion.last_diagnostics.values())


def test_post_gta_pre_output_projection_placement_and_attention_independence():
    torch.manual_seed(18)
    layer = RelationTransformerLayer(64, 2, 128)
    fusion = TaskActionFusion()
    torch.nn.init.normal_(fusion.mlp[-1].weight, std=.05)
    x = torch.randn(2, 8, 64)
    pose = torch.randn(2, 8, 7)
    pose[..., 3:] = torch.nn.functional.normalize(pose[..., 3:], dim=-1)
    g, ginv = build_gta_transforms(pose)
    condition = (torch.tensor([[1, 0], [0, 1]]), torch.randn(2, 2, 64), torch.ones(2, 2, dtype=torch.bool))
    captured = []
    hook = layer.proj.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
    base_out = layer(x, gta_g=g, gta_ginv=ginv)
    result = layer(x, gta_g=g, gta_ginv=ginv, task_condition=condition, task_action_fusion=fusion)
    hook.remove()
    torch.testing.assert_close(captured[1], fusion(captured[0], condition))
    assert not torch.equal(base_out, result)
    # No task edges/bias were needed for this direct injection. Task swap changes
    # output while QK, V and bias of this layer stay fixed.
    swapped = (condition[0], condition[1].flip(1), condition[2])
    assert not torch.equal(result, layer(x, gta_g=g, gta_ginv=ginv,
        task_condition=swapped, task_action_fusion=fusion))


@pytest.mark.parametrize('branch', ['actor', 'critic'])
def test_token_edge_task_shuffle_outputs_and_all_gradients(branch):
    torch.manual_seed(82)
    net = reference.network()
    enc = getattr(net, branch + '_encoder')
    head = net.action_head if branch == 'actor' else net.value_head
    with torch.no_grad():
        enc.edge_encoder.bias_projection.uniform_(-.1, .1)
        enc.task_action_fusion.mlp[-1].weight.normal_(std=.02)
    obs = reference.observations(reference.all_pairs())
    original = head(enc(obs, token_order=torch.arange(8)))
    original.square().mean().backward()
    grads = {n: p.grad.clone() for n, p in enc.named_parameters() if p.grad is not None}
    assert (enc.edge_encoder.category_embed.weight.grad.abs().sum(-1) > 0).all()
    assert all(p.grad.abs().sum() > 0 for p in enc.task_action_fusion.parameters())
    net.zero_grad(set_to_none=True)
    graph = reference.permute_graph(reference.all_pairs(), torch.rand(16, 4).argsort(-1))
    packet = task_graph_packet(graph, 16).reshape(16, 2, 5).flip(1).flatten(1)
    changed = torch.cat((obs[:, :-10], packet), -1)
    result = head(enc(changed, token_order=torch.tensor([6, 4, 1, 3, 7, 0, 5, 2])))
    torch.testing.assert_close(original, result, atol=1e-5, rtol=1e-5)
    result.square().mean().backward()
    for name, p in enc.named_parameters():
        if name in grads:
            torch.testing.assert_close(grads[name], p.grad, atol=5e-5, rtol=1e-3)


def test_zero_start_two_step_learning_shared_layers_and_independent_branches():
    torch.manual_seed(26)
    net = reference.network()
    initial = {}
    for branch in ('actor', 'critic'):
        enc = getattr(net, branch + '_encoder')
        assert enc.task_action_fusion.mlp[-1].weight.count_nonzero() == 0
        assert enc.task_action_fusion.mlp[-1].bias.count_nonzero() == 0
        assert enc.edge_encoder.bias_projection.count_nonzero() == 0
        assert sum(p.numel() for p in enc.task_action_fusion.parameters()) == 12416
        assert not any('task_action' in n for layer in enc.layers for n, _ in layer.named_parameters())
        initial[branch] = {n: p.detach().clone() for n, p in enc.named_parameters()
                          if 'task_action' in n or 'category_embed' in n}
        # Exactly four invocations of the same module, one per layer.
        calls = []
        hook = enc.task_action_fusion.register_forward_hook(lambda m, a, r: calls.append(m))
        obs = reference.observations(reference.all_pairs())
        enc(obs)
        hook.remove()
        assert len(calls) == 4 and all(m is enc.task_action_fusion for m in calls)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        (net.eval_actor(obs)[0].square().mean() + net.eval_critic(obs).square().mean()).backward()
        optimizer.step()
    for branch in ('actor', 'critic'):
        enc = getattr(net, branch + '_encoder')
        for n, p in enc.named_parameters():
            if n in initial[branch]:
                assert not torch.equal(initial[branch][n], p), n
        assert ((initial[branch]['edge_encoder.category_embed.weight'] -
                 enc.edge_encoder.category_embed.weight).abs().sum(-1) > 0).all()
        assert enc.build_relation_bias().count_nonzero() == 0
    assert net.actor_encoder.task_action_fusion is not net.critic_encoder.task_action_fusion
    assert net.actor_encoder.edge_encoder is not net.critic_encoder.edge_encoder
