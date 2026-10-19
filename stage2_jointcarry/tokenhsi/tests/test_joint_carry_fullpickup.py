from copy import deepcopy

import pytest
import torch
import yaml

from learning.multi_agent.stage2_transfer import transfer_stage1_weights
from learning.multi_agent.task_coordination import TaskCoordination
from learning.multi_agent.task_role_encoder import TaskTypeEmbeddingBias
from utils.joint_carry_amp import check_ablation_checkpoint, validate_ablation
from utils.joint_carry_rsi import load_paired_snapshots
from utils.joint_carry_spec import validate_env
from utils.relation_task_spec import checkpoint_metadata
from test_joint_carry import build_network, joint_observations, ROOT
from test_joint_carry_amp_ablation import configs, network
from test_stage2_coordination import Wrapper


SUFFIX = 'locoamp_ca4_goal360_fullpickup_task_embedding'


def test_fullpickup_contract_and_data():
    base, base_train = configs('locoamp_ca4_goal360_task_embedding')
    env, train = configs(SUFFIX)
    validate_env(env); validate_ablation(env, train)
    restored = deepcopy(env)
    restored['jointCarryRsi'] = base['jointCarryRsi']
    restored['jointCarryAblation']['policy'] = base['jointCarryAblation']['policy']
    assert restored == base
    restored = deepcopy(train)
    restored['params']['config']['name'] = base_train['params']['config']['name']
    restored['params']['network']['coordination'] = base_train['params']['network']['coordination']
    assert restored == base_train
    rsi = env['jointCarryRsi']
    pools = load_paired_snapshots(rsi['directory'], 'cpu', True, rsi['dataset_sha256'])
    assert {k:len(v['source_frame']) for k,v in pools.items()} == dict(pickUp=100, carryWith=129, putDown=135)
    assert torch.equal(pools['pickUp']['source_frame'], torch.arange(100))
    old = load_paired_snapshots(base['jointCarryRsi']['directory'], 'cpu', True)
    for skill in ('carryWith', 'putDown'):
        for key in old[skill]:
            torch.testing.assert_close(old[skill][key], pools[skill][key], rtol=0, atol=0)
    with pytest.raises(ValueError, match='digest'):
        load_paired_snapshots(rsi['directory'], 'cpu', expected_digest='wrong')
    weights = dict(joint_carry_rsi_dataset=rsi['dataset_sha256'],
                   joint_carry_ablation=env['jointCarryAblation'],
                   joint_carry_goal_sampling=rsi['goalSampling'])
    check_ablation_checkpoint(weights, env)
    with pytest.raises(ValueError, match='RSI dataset'):
        check_ablation_checkpoint(weights, base)
    wrong = deepcopy(weights); wrong['joint_carry_ablation'] = base['jointCarryAblation']
    with pytest.raises(ValueError, match='ablation'):
        check_ablation_checkpoint(wrong, env)


def test_transfer_identity_independent_branches_and_learning():
    torch.manual_seed(123)
    torch.set_num_threads(1)
    source = Wrapper(build_network(False))
    target = network(SUFFIX)
    reward = yaml.safe_load((ROOT/'data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding.yaml').read_text())['env']['relationReward']
    report = transfer_stage1_weights(target, dict(model=source.state_dict(), relation_metadata=checkpoint_metadata(reward)))
    net = target.a2c_network
    assert report['head_expansion'] == dict(source_columns=64, context_columns=0)
    assert net.action_head[0][0].in_features == net.value_head[0][0].in_features == 64
    for branch in ('action_head', 'value_head', 'critic_encoder'):
        for key, value in getattr(source.a2c_network, branch).state_dict().items():
            torch.testing.assert_close(value, getattr(net, branch).state_dict()[key], rtol=0, atol=0)
    assert not ({id(p) for p in net.coordination.parameters()} & {id(p) for p in net.critic_coordination.parameters()})
    obs = joint_observations()
    with torch.no_grad():
        for _ in range(3):
            torch.testing.assert_close(net.eval_actor(obs)[0], source.a2c_network.eval_actor(obs)[0], atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(net.eval_critic(obs), source.a2c_network.eval_critic(obs), atol=1e-6, rtol=1e-5)
    frozen = {k:v.clone() for k,v in net.actor_encoder.state_dict().items()}
    initial = {k:v.clone() for k,v in net.state_dict().items()}
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-3)
    for _ in range(4):
        optimizer.zero_grad()
        ((net.eval_actor(obs)[0]-.2).square().mean()+(net.eval_critic(obs)-1.).square().mean()).backward()
        optimizer.step()
    assert all(not p.requires_grad and p.grad is None for p in net.actor_encoder.parameters())
    assert all(torch.equal(v,net.actor_encoder.state_dict()[k]) for k,v in frozen.items())
    for branch in ('critic_encoder','coordination','critic_coordination','action_head','value_head'):
        assert all(p.requires_grad for p in getattr(net,branch).parameters())
        assert any(not torch.equal(initial[branch+'.'+k],v) for k,v in getattr(net,branch).state_dict().items())
    for branch in ('coordination','critic_coordination'):
        for i in range(4):
            key=f'{branch}.blocks.{i}.relation_bias'
            assert not torch.equal(initial[key],net.state_dict()[key])
    with torch.no_grad():
        actor,critic=net.eval_actor(obs)[0],net.eval_critic(obs)
        for _ in range(3):
            torch.testing.assert_close(actor,net.eval_actor(obs)[0],atol=3e-6,rtol=1e-5)
            torch.testing.assert_close(critic,net.eval_critic(obs),atol=3e-6,rtol=1e-5)


def test_residual_padding_permutation_and_critic_embedding_gradient():
    torch.manual_seed(9)
    module=TaskCoordination(num_layers=4,feedforward_dim=128,residual_output=True,detach_task_embedding=False)
    embedding=TaskTypeEmbeddingBias(4,2)
    nodes=torch.randn(2,8,64)
    packet=torch.tensor([[[1,2,0,2,6],[1,2,1,2,6]],[[0,0,0,0,0],[0,0,0,0,0]]])
    module.init_identity()
    torch.testing.assert_close(module(nodes[:,:2],nodes,packet,embedding),nodes[:,:2],rtol=0,atol=0)
    with torch.no_grad():
        for block in module.blocks:
            block.attention.out_proj.weight.normal_(std=.02)
            block.ffn[-1].weight.normal_(std=.02)
            block.relation_bias.normal_(std=.1)
    expected=module(nodes[:,:2],nodes,packet,embedding)
    torch.testing.assert_close(expected[1],nodes[1,:2],rtol=0,atol=0)
    torch.testing.assert_close(expected,module(nodes[:,:2],nodes,packet[:,[1,0]],embedding))
    order=torch.tensor([1,0,4,3,2,5,7,6])
    moved=packet.clone();moved[...,2:]=order.argsort()[packet[...,2:]]
    changed=nodes[:,order]
    torch.testing.assert_close(expected[:,[1,0]],module(changed[:,:2],changed,moved,embedding),atol=2e-6,rtol=1e-5)
    expected.square().mean().backward()
    assert embedding.category_embed.weight.grad.abs().sum()>0
