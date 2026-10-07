"""Task-message environment parity and message-free MLP policy contracts."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

import test_task_role_message as reference
from learning.multi_agent.task_role_encoder import TaskRoleMLPFusion
from utils.relation_task_spec import validate_relation_config, checkpoint_metadata, check_checkpoint_metadata
from utils.task_role_spec import TASK_ROLE_MLP_VARIANT, task_graph_packet
from utils.unified_training import validate_unified_env, validate_typed_bias_config

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / 'data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp.yaml').read_text())['env']
TRAIN = yaml.safe_load((ROOT / 'data/cfg/train/rlg/amp_ma_carry_relation_unified_size_rsi_task_mlp.yaml').read_text())
MESSAGE_CFG, MESSAGE_TRAIN = reference.CFG, reference.TRAIN


@pytest.fixture(autouse=True)
def use_mlp_config(monkeypatch):
    monkeypatch.setattr(reference, 'CFG', CFG)
    monkeypatch.setattr(reference, 'TRAIN', TRAIN)


def test_complete_env_and_train_parity_and_checkpoint_isolation():
    expected = deepcopy(MESSAGE_CFG)
    expected['relationReward']['stage1_variant'] = TASK_ROLE_MLP_VARIANT
    assert CFG == expected  # Includes all RSI, AMP, sizes, sampling and reward settings.
    train = deepcopy(MESSAGE_TRAIN)
    train['params']['network']['transformer'].update(
        relation_bias_mode='task_role_mlp', relation_message={'enable': False})
    assert TRAIN == train
    validate_relation_config(CFG['relationReward'])
    validate_unified_env(CFG)
    validate_typed_bias_config(CFG, TRAIN)
    for env, config in ((CFG, MESSAGE_TRAIN), (MESSAGE_CFG, TRAIN)):
        with pytest.raises(ValueError, match='must be paired'):
            validate_typed_bias_config(env, config)
    for old, new in ((CFG, MESSAGE_CFG), (MESSAGE_CFG, CFG)):
        with pytest.raises(ValueError, match='mismatch'):
            check_checkpoint_metadata({'relation_metadata': checkpoint_metadata(old['relationReward'])},
                                      checkpoint_metadata(new['relationReward']))
    metadata = checkpoint_metadata(CFG['relationReward'])
    assert metadata['packet_version'] == 5
    assert metadata['context_fusion'] == 'task_role_mlp_bias64'
    bad = deepcopy(TRAIN)
    bad['params']['network']['transformer']['relation_message']['enable'] = True
    with pytest.raises(ValueError, match='must be paired'):
        validate_typed_bias_config(CFG, bad)


@pytest.mark.parametrize('check', [
    reference.test_all_pairs_packet_binding_owner_and_edge_shuffle,
    reference.test_size_sampling_no_standalone_holding_preserves_family_rsi_bindings,
    reference.test_rewards_equal_old_local_sum_no_teammate_dependency_and_partial_suffix,
    reference.test_carry_ontop_payload_above_support_and_at_goal_match_packet,
    reference.test_checkpoint_roundtrip_rebuild_and_actor_critic_separation,
])
def test_shared_task_rsi_amp_reward_and_checkpoint_contracts(check):
    check()


def test_mlp_bias_oracle_directed_roles_background_and_no_message():
    net = reference.network()
    assert not any('message' in name for name, _ in net.named_parameters())
    for enc in (net.actor_encoder, net.critic_encoder):
        assert isinstance(enc.context_fusion, TaskRoleMLPFusion)
        assert not enc.use_relation_message
    enc = net.actor_encoder
    fusion = enc.context_fusion
    packet = task_graph_packet(reference.all_pairs(), 16)
    bias, message = fusion(packet, enc.build_relation_bias())
    assert message is None
    assert bias.count_nonzero() == 0
    with torch.no_grad():
        fusion.bias_projection.normal_()
        enc.edge_encoder.bias_projection.normal_()
    background = enc.build_relation_bias()
    bias, message = fusion(packet, background)
    expected = background[:, None].expand_as(bias).clone()
    for b, tasks in enumerate(packet.reshape(16, 2, 5).long()):
        for _, task, actor, payload, target in tasks:
            edges = [(actor, target, 0, 2)]
            if task >= 2:
                edges += [(actor, payload, 0, 1), (payload, target, 1, 2)]
            for s, t, sr, tr in edges:
                # Independent per-edge oracle for the batched category encoding.
                semantic = fusion.edge_mlp(torch.cat((fusion.source_role_embed.weight[sr],
                    fusion.task_embed.weight[task], fusion.target_role_embed.weight[tr])))
                expected[:, b, :, s, t] = torch.einsum('d,lhd->lh', semantic, fusion.bias_projection)
                torch.testing.assert_close(bias[:, b, :, t, s], background[:, :, t, s])
    torch.testing.assert_close(bias, expected)
    assert message is None
    assert not torch.equal(bias[:, 15, :, 0, 2], bias[:, 15, :, 0, 4])


@pytest.mark.parametrize('branch', ['actor', 'critic'])
def test_permutation_outputs_gradients_and_optimizer_updates(branch):
    torch.manual_seed(82)
    net = reference.network()
    enc = getattr(net, branch + '_encoder')
    head = net.action_head if branch == 'actor' else net.value_head
    with torch.no_grad():
        enc.context_fusion.bias_projection.uniform_(-.1, .1)
        enc.edge_encoder.bias_projection.uniform_(-.1, .1)
    graph = reference.all_pairs()
    obs = reference.observations(graph)
    original = head(enc(obs, token_order=torch.arange(8)))
    original.square().mean().backward()
    grads = {name: p.grad.clone() for name, p in enc.named_parameters() if p.grad is not None}
    for name, p in enc.context_fusion.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name
    assert (enc.context_fusion.task_embed.weight.grad.abs().sum(-1) > 0).all()
    net.zero_grad(set_to_none=True)
    shuffled = reference.permute_graph(graph, torch.rand(16, 4).argsort(-1))
    packet = task_graph_packet(shuffled, 16).reshape(16, 2, 5).flip(1).flatten(1)
    changed = torch.cat((obs[:, :-10], packet), -1)
    result = head(enc(changed, token_order=torch.tensor([6, 4, 1, 3, 7, 0, 5, 2])))
    torch.testing.assert_close(original, result, atol=1e-5, rtol=1e-5)
    result.square().mean().backward()
    for name, p in enc.named_parameters():
        if name in grads:
            torch.testing.assert_close(grads[name], p.grad, atol=5e-5, rtol=1e-3)
    before = {name: p.detach().clone() for name, p in enc.context_fusion.named_parameters()}
    torch.optim.Adam(enc.parameters(), lr=1e-3).step()
    for name, p in enc.context_fusion.named_parameters():
        assert not torch.equal(before[name], p), name


def test_zero_projection_initialization_learns_in_two_steps():
    net = reference.network()
    obs = reference.observations(reference.all_pairs())
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    initial = {branch: getattr(net, branch + '_encoder').context_fusion.edge_mlp[0].weight.detach().clone()
               for branch in ('actor', 'critic')}
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        (net.eval_actor(obs)[0].square().mean() + net.eval_critic(obs).square().mean()).backward()
        optimizer.step()
    for branch in ('actor', 'critic'):
        fusion = getattr(net, branch + '_encoder').context_fusion
        assert fusion.bias_projection.count_nonzero() > 0
        assert not torch.equal(initial[branch], fusion.edge_mlp[0].weight)
    assert net.actor_encoder.context_fusion is not net.critic_encoder.context_fusion
