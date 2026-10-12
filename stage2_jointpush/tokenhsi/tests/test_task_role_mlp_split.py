"""Task-specific MLP/projection routing with shared role embeddings."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

import test_task_role_message as reference
import test_task_role_mlp as shared_tests
from learning.multi_agent.task_role_encoder import TaskRoleMLPFusion
from utils.relation_task_spec import validate_relation_config, checkpoint_metadata, check_checkpoint_metadata
from utils.task_role_spec import TASK_ROLE_SPLIT_MLP_VARIANT, task_graph_packet
from utils.unified_training import validate_unified_env, validate_typed_bias_config

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / 'data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_split.yaml').read_text())['env']
TRAIN = yaml.safe_load((ROOT / 'data/cfg/train/rlg/amp_ma_carry_relation_unified_size_rsi_task_mlp_split.yaml').read_text())


@pytest.fixture(autouse=True)
def use_split_config(monkeypatch):
    monkeypatch.setattr(reference, 'CFG', CFG)
    monkeypatch.setattr(reference, 'TRAIN', TRAIN)


def test_env_train_parity_and_strict_checkpoint_contract():
    expected = deepcopy(shared_tests.CFG)
    expected['relationReward']['stage1_variant'] = TASK_ROLE_SPLIT_MLP_VARIANT
    assert CFG == expected
    expected_train = deepcopy(shared_tests.TRAIN)
    expected_train['params']['network']['transformer']['relation_bias_mode'] = 'task_role_mlp_split'
    assert TRAIN == expected_train
    validate_relation_config(CFG['relationReward'])
    validate_unified_env(CFG)
    validate_typed_bias_config(CFG, TRAIN)
    for old_env, old_train in ((shared_tests.CFG, shared_tests.TRAIN),
                              (shared_tests.MESSAGE_CFG, shared_tests.MESSAGE_TRAIN)):
        for env, train in ((CFG, old_train), (old_env, TRAIN)):
            with pytest.raises(ValueError, match='must be paired'):
                validate_typed_bias_config(env, train)
        for old, new in ((CFG, old_env), (old_env, CFG)):
            with pytest.raises(ValueError, match='mismatch'):
                check_checkpoint_metadata({'relation_metadata': checkpoint_metadata(old['relationReward'])},
                                          checkpoint_metadata(new['relationReward']))
    assert checkpoint_metadata(CFG['relationReward'])['context_fusion'] == 'task_role_split_mlp_bias64'
    for key, value in [('relation_message', {'enable': True}), ('relation_bias', False),
                       ('share_edge_encoder', True)]:
        bad = deepcopy(TRAIN)
        bad['params']['network']['transformer'][key] = value
        with pytest.raises(ValueError):
            validate_typed_bias_config(CFG, bad)
    shared = TaskRoleMLPFusion(4, 2)
    split = TaskRoleMLPFusion(4, 2, split_tasks=True)
    for src, dst in ((shared, split), (split, shared)):
        with pytest.raises(RuntimeError):
            dst.load_state_dict(src.state_dict(), strict=True)


@pytest.mark.parametrize('check', [
    reference.test_all_pairs_packet_binding_owner_and_edge_shuffle,
    reference.test_size_sampling_no_standalone_holding_preserves_family_rsi_bindings,
    reference.test_rewards_equal_old_local_sum_no_teammate_dependency_and_partial_suffix,
    reference.test_carry_ontop_payload_above_support_and_at_goal_match_packet,
    reference.test_checkpoint_roundtrip_rebuild_and_actor_critic_separation,
])
def test_shared_task_rsi_amp_reward_and_checkpoint_contracts(check):
    check()


@pytest.mark.parametrize('branch', ['actor', 'critic'])
def test_network_permutations_gradients_and_optimizer_updates(branch):
    shared_tests.test_permutation_outputs_gradients_and_optimizer_updates(branch)


def test_tied_split_matches_shared_outputs_and_summed_gradients():
    torch.manual_seed(27)
    shared = TaskRoleMLPFusion(4, 2)
    split = TaskRoleMLPFusion(4, 2, split_tasks=True)
    with torch.no_grad():
        shared.bias_projection.normal_()
        split.bias_projection.copy_(shared.bias_projection[None].expand_as(split.bias_projection))
    for name in ('source_role_embed', 'task_embed', 'target_role_embed'):
        getattr(split, name).load_state_dict(getattr(shared, name).state_dict())
    for mlp in split.edge_mlps:
        mlp.load_state_dict(shared.edge_mlp.state_dict())
    packet = task_graph_packet(reference.all_pairs(), 16)
    background = torch.randn(4, 2, 8, 8)
    out_shared, msg_shared = shared(packet, background)
    out_split, msg_split = split(packet, background)
    assert msg_shared is msg_split is None
    torch.testing.assert_close(out_shared, out_split)
    probe = torch.randn_like(out_shared)
    (out_shared * probe).sum().backward()
    (out_split * probe).sum().backward()
    torch.testing.assert_close(shared.bias_projection.grad, split.bias_projection.grad.sum(0), atol=2e-5, rtol=2e-5)
    for name, p in shared.edge_mlp.named_parameters():
        expected = sum(dict(mlp.named_parameters())[name].grad for mlp in split.edge_mlps)
        torch.testing.assert_close(p.grad, expected, atol=2e-4, rtol=2e-5)
    for name in ('source_role_embed', 'task_embed', 'target_role_embed'):
        torch.testing.assert_close(getattr(shared, name).weight.grad,
                                   getattr(split, name).weight.grad, atol=2e-4, rtol=2e-5)


@pytest.mark.parametrize('task', range(4))
def test_task_local_weights_and_gradients_cannot_change_other_tasks(task):
    torch.manual_seed(61)
    fusion = TaskRoleMLPFusion(4, 2, split_tasks=True)
    with torch.no_grad():
        fusion.bias_projection.normal_()
    packets = torch.tensor([[1, t, 0, 2 if t >= 2 else 0, 6 if t == 2 else 4] for t in range(4)]).float()
    background = torch.zeros(4, 2, 8, 8)
    before, _ = fusion(packets, background)
    bias, _ = fusion(packets[task:task+1], background)
    bias.square().sum().backward()
    for t, mlp in enumerate(fusion.edge_mlps):
        magnitude = sum(p.grad.abs().sum() for p in mlp.parameters())
        assert (magnitude > 0) if t == task else (magnitude == 0)
        assert (fusion.bias_projection.grad[t].abs().sum() > 0) if t == task else (fusion.bias_projection.grad[t].count_nonzero() == 0)
    assert fusion.source_role_embed.weight.grad.abs().sum() > 0
    assert fusion.target_role_embed.weight.grad.abs().sum() > 0
    assert fusion.task_embed.weight.grad[task].abs().sum() > 0
    assert fusion.task_embed.weight.grad[torch.arange(4) != task].count_nonzero() == 0
    with torch.no_grad():
        fusion.edge_mlps[task][-1].bias.add_(1.)
        fusion.bias_projection[task].add_(.2)
    after, _ = fusion(packets, background)
    # Dense bias axes: layer, batch(task), head, source, target.
    others = torch.arange(4) != task
    torch.testing.assert_close(before[:, others], after[:, others], atol=0, rtol=0)
    assert not torch.equal(before[:, task], after[:, task])


def test_parameter_count_no_messages_and_all_tasks_learn_from_zero_projection():
    net = reference.network()
    assert sum(p.numel() for p in net.parameters()) == 4209730
    assert not any('message' in n for n, _ in net.named_parameters())
    initial = {}
    for branch in ('actor', 'critic'):
        enc = getattr(net, branch + '_encoder')
        f = enc.context_fusion
        assert f.split_tasks and not enc.use_relation_message
        assert sum(p.numel() for p in f.parameters()) == 35552
        assert f.bias_projection.shape == (4, 4, 2, 64)
        assert f.bias_projection.count_nonzero() == 0
        assert len({p.data_ptr() for mlp in f.edge_mlps for p in mlp.parameters()}) == 16
        initial[branch] = [mlp[0].weight.detach().clone() for mlp in f.edge_mlps]
        bias, msg = f(task_graph_packet(reference.all_pairs(), 16), enc.build_relation_bias())
        assert bias.count_nonzero() == 0 and msg is None
    obs = reference.observations(reference.all_pairs())
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        (net.eval_actor(obs)[0].square().mean() + net.eval_critic(obs).square().mean()).backward()
        optimizer.step()
    for branch in ('actor', 'critic'):
        f = getattr(net, branch + '_encoder').context_fusion
        for t in range(4):
            assert f.bias_projection[t].count_nonzero() > 0
            assert not torch.equal(initial[branch][t], f.edge_mlps[t][0].weight)
    assert net.actor_encoder.context_fusion is not net.critic_encoder.context_fusion
    assert net.actor_encoder.edge_encoder is not net.critic_encoder.edge_encoder
