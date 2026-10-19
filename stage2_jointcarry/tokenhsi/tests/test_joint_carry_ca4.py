from copy import deepcopy

import pytest
import torch

from learning.multi_agent.stage2_transfer import transfer_stage1_weights
from learning.multi_agent.task_coordination import TaskCoordination
from learning.multi_agent.task_role_encoder import TaskTypeEmbeddingBias
from utils.joint_carry_amp import check_ablation_checkpoint, validate_ablation
from utils.relation_task_spec import checkpoint_metadata
from test_joint_carry import build_network, joint_observations, ROOT
from test_joint_carry_amp_ablation import configs, network
from test_stage2_coordination import Wrapper
import yaml


def test_ca4_contract_changes_only_architecture():
    base_env, base_train = configs('locoamp_task_embedding')
    env, train = configs('locoamp_ca4_task_embedding')
    validate_ablation(env, train)
    restored = deepcopy(env)
    restored['jointCarryAblation']['policy'] = 'task_ca'
    assert restored == base_env
    restored = deepcopy(train)
    restored['params']['config']['name'] = base_train['params']['config']['name']
    coordination = restored['params']['network']['coordination']
    assert coordination.pop('num_layers') == 4
    assert coordination.pop('feedforward_dim') == 128
    assert restored == base_train
    for saved_env, target_env in ((env, base_env), (base_env, env)):
        with pytest.raises(ValueError, match='ablation differs'):
            check_ablation_checkpoint({'joint_carry_ablation':saved_env['jointCarryAblation']}, target_env)
    with pytest.raises(ValueError, match='coordination'):
        validate_ablation(env, base_train)


def test_stack_padding_task_and_entity_permutation_gradients():
    torch.manual_seed(29)
    torch.set_num_threads(1)
    module = TaskCoordination(num_layers=4, feedforward_dim=128)
    embed = TaskTypeEmbeddingBias(4, 2).requires_grad_(False)
    packet = torch.tensor([[[1, 2, 0, 3, 7], [1, 2, 1, 3, 7],
                            [1, 1, 2, 0, 4], [0, 0, 0, 0, 0]],
                           [[1, 2, 0, 3, 7], [1, 3, 1, 4, 6],
                            [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                           [[0, 0, 0, 0, 0]] * 4])
    nodes = torch.randn(3, 9, 64)
    with torch.no_grad():
        for block in module.blocks:
            block.relation_bias.normal_()
    output = module(nodes[:, :3], nodes, packet, embed)
    assert torch.isfinite(output).all() and output[2].count_nonzero() == 0
    task_order = torch.tensor([[2, 0, 3, 1], [1, 3, 0, 2], [3, 2, 1, 0]])
    shuffled = packet.gather(1, task_order[..., None].expand(-1, -1, 5))
    torch.testing.assert_close(output, module(nodes[:, :3], nodes, shuffled, embed))
    order = torch.tensor([2, 0, 1, 5, 3, 4, 8, 6, 7])
    remapped = packet.clone()
    remapped[..., 2:] = order.argsort()[packet[..., 2:]]
    new_nodes = nodes[:, order]
    torch.testing.assert_close(output[:, order[:3]], module(new_nodes[:, :3], new_nodes, remapped, embed))
    loss_weight = torch.randn_like(output)
    (output * loss_weight).sum().backward()
    gradients = {name:p.grad.clone() for name,p in module.named_parameters()}
    module.zero_grad()
    module.shuffle_task_order = True
    (module(nodes[:, :3], nodes, shuffled, embed) * loss_weight).sum().backward()
    for name, p in module.named_parameters():
        torch.testing.assert_close(p.grad, gradients[name], atol=2e-5, rtol=2e-4)
    for block in module.blocks:
        assert (block.relation_bias.grad.abs().sum(0) > 0).all()
    assert all(p.grad is None for p in embed.parameters())


def test_ca4_stage1_transfer_frozen_encoder_and_all_blocks_learn():
    torch.manual_seed(31)
    source = Wrapper(build_network(False))
    target = network('locoamp_ca4_task_embedding')
    reward = yaml.safe_load((ROOT/'data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding.yaml').read_text())['env']['relationReward']
    report = transfer_stage1_weights(target, dict(model=source.state_dict(), relation_metadata=checkpoint_metadata(reward)))
    net = target.a2c_network
    assert report['head_expansion'] == {'source_columns':64, 'context_columns':64}
    assert sum(p.numel() for p in net.coordination.parameters()) == 175064
    assert sum(p.numel() for p in net.parameters()) == 4371226
    assert net.action_head[0][0].weight[:, 64:].count_nonzero() == 0
    obs = joint_observations()
    with torch.no_grad():
        expected = source.a2c_network.action_head(net.actor_encoder(obs)).flatten(0, 1)
        for _ in range(3):
            torch.testing.assert_close(net.eval_actor(obs)[0], expected, atol=1e-6, rtol=1e-5)
    frozen = {name:p.clone() for name,p in net.actor_encoder.state_dict().items()}
    initial = {name:p.clone() for name,p in net.coordination.state_dict().items()}
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-3)
    for _ in range(3):
        optimizer.zero_grad()
        net.eval_actor(obs)[0].square().mean().backward()
        optimizer.step()
    assert all(not p.requires_grad and p.grad is None for p in net.actor_encoder.parameters())
    assert all(torch.equal(value, net.actor_encoder.state_dict()[name]) for name,value in frozen.items())
    for i in range(4):
        for key in ('attention.in_proj_weight', 'query_norm.weight', 'ffn.0.weight', 'ffn_norm.weight', 'relation_bias'):
            name = f'blocks.{i}.{key}'
            assert not torch.equal(initial[name], net.coordination.state_dict()[name])
    with torch.no_grad():
        expected = net.eval_actor(obs)[0]
        for _ in range(3):
            torch.testing.assert_close(net.eval_actor(obs)[0], expected, atol=1e-6, rtol=1e-5)
    assert all(torch.isfinite(p).all() for p in net.parameters())
