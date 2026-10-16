from copy import deepcopy

import pytest
import torch
import yaml

from learning.multi_agent.human_coordination import HumanCoordination, human_relations
from learning.multi_agent.stage2_transfer import transfer_stage1_weights
from utils.joint_carry_amp import validate_ablation, check_ablation_checkpoint
from utils.joint_carry_spec import validate_env
from utils.relation_task_spec import checkpoint_metadata
from test_joint_carry_amp_ablation import configs, network, ROOT
from test_joint_carry import build_network, joint_observations
from test_stage2_coordination import Wrapper

BASE = 'locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_task_embedding'
NEW = BASE.replace('locoamp_ca4', 'locoamp_sa4')


def test_config_and_checkpoint_isolation():
    env, train = configs(NEW)
    base, old_train = configs(BASE)
    validate_env(env)
    validate_ablation(env, train)
    restored = deepcopy(env)
    restored['jointCarryAblation']['policy'] = base['jointCarryAblation']['policy']
    assert restored == base
    restored = deepcopy(train)
    restored['params']['network']['coordination'] = old_train['params']['network']['coordination']
    restored['params']['config']['name'] = old_train['params']['config']['name']
    assert restored == old_train
    weights = dict(joint_carry_loco_platform=env['jointCarryRsi']['locoPlatform'],
        joint_carry_rsi_dataset=env['jointCarryRsi']['dataset_sha256'],
        joint_carry_goal_sampling=env['jointCarryRsi']['goalSampling'],
        joint_carry_ablation=env['jointCarryAblation'])
    check_ablation_checkpoint(weights, env)
    with pytest.raises(ValueError, match='ablation differs'):
        check_ablation_checkpoint(weights, base)
    with pytest.raises(ValueError, match='coordination'):
        validate_ablation(env, old_train)
    slug = 'approach_stage2_joint_carry_mixed80_' + NEW
    for mode in ('train', 'test'):
        script = (ROOT/'scripts/multi_agent'/(slug+'_'+mode+'.sh')).read_text()
        assert slug+'.yaml' in script
        assert 'amp_ma_stage2_joint_carry_mixed80_'+NEW+'.yaml' in script
    assert slug+'_test.sh' in (ROOT/'scripts/multi_agent'/(slug+'_vnc.sh')).read_text()


def test_six_human_relations_and_permutation():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    sa = HumanCoordination()
    humans = torch.randn(3, 6, 64, requires_grad=True)
    packet = torch.tensor([[[1,2,0,6,14],[1,2,1,6,14],
                            [1,2,2,7,16],[1,2,3,7,16],
                            [1,2,4,8,18],[1,3,5,9,10]]]).expand(3,-1,-1).clone()
    expected = torch.eye(6, dtype=torch.long).expand(3,-1,-1).clone()
    expected[:,0,1] = expected[:,1,0] = expected[:,2,3] = expected[:,3,2] = 2
    torch.testing.assert_close(human_relations(packet, 6), expected)
    output = sa(humans, None, packet, None)
    shuffled = packet[:,[4,2,0,5,1,3]]
    torch.testing.assert_close(output, sa(humans,None,shuffled,None))
    sa.shuffle_human_order = True
    for _ in range(3):
        torch.testing.assert_close(output,sa(humans,None,shuffled,None),atol=2e-6,rtol=1e-5)
    order = torch.tensor([4,2,0,5,1,3])
    moved = packet.clone(); moved[...,2] = order.argsort()[packet[...,2]]
    torch.testing.assert_close(output[:,order],sa(humans[:,order],None,moved,None),atol=2e-6,rtol=1e-5)
    padded = torch.cat((packet, torch.zeros(3,2,5,dtype=torch.long)),1)
    torch.testing.assert_close(output,sa(humans,None,padded,None),atol=2e-6,rtol=1e-5)
    torch.testing.assert_close(humans,sa(humans,None,torch.zeros_like(packet),None),atol=0,rtol=0)
    output.square().mean().backward()
    for block in sa.blocks:
        assert (block.relation_bias.grad.abs().sum(0) > 0).all()
    assert torch.isfinite(humans.grad).all()


def test_all_layers_use_updated_humans_as_qkv_manual_oracle():
    torch.manual_seed(7)
    sa = HumanCoordination()
    humans = torch.randn(2,2,64)
    packet = torch.tensor([[[1,2,0,2,6],[1,2,1,2,6]],
                           [[1,2,0,2,6],[1,3,1,3,4]]])
    relations = human_relations(packet,2)
    context = humans
    for block in sa.blocks:
        with torch.no_grad():
            block.relation_bias.normal_(std=.3)
        normalized = block.query_norm(context)
        weights = block.attention.in_proj_weight.chunk(3)
        biases = block.attention.in_proj_bias.chunk(3)
        q,k,v = [torch.nn.functional.linear(normalized,w,b).reshape(2,2,2,32).transpose(1,2)
                 for w,b in zip(weights,biases)]
        logits = q @ k.transpose(-1,-2) / 32**.5
        probability = (logits + block.relation_bias[:,relations].permute(1,0,2,3)).softmax(-1)
        assert (probability > 0).all()
        update = (probability @ v).transpose(1,2).reshape(2,2,64)
        context = context + block.attention.out_proj(update)
        context = context + block.ffn(block.ffn_norm(context))
    torch.testing.assert_close(context,sa(humans,None,packet,None),atol=2e-6,rtol=1e-5)


def test_actor_critic_transfer_and_training():
    torch.manual_seed(123)
    torch.set_num_threads(1)
    source = Wrapper(build_network(False))
    target = network(NEW)
    reward = yaml.safe_load((ROOT/'data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding.yaml').read_text())['env']['relationReward']
    report = transfer_stage1_weights(target, dict(model=source.state_dict(), relation_metadata=checkpoint_metadata(reward)))
    net = target.a2c_network
    assert report['head_expansion'] == dict(source_columns=64, context_columns=0)
    for ca in (net.coordination, net.critic_coordination):
        assert isinstance(ca, HumanCoordination) and len(ca.blocks) == 4
        assert not hasattr(ca, 'grounding')
    obs = joint_observations()
    with torch.no_grad():
        torch.testing.assert_close(net.eval_actor(obs)[0], source.a2c_network.eval_actor(obs)[0], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(net.eval_critic(obs), source.a2c_network.eval_critic(obs), atol=1e-6, rtol=1e-5)
    initial = deepcopy(net.state_dict())
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-3)
    for _ in range(4):
        optimizer.zero_grad()
        ((net.eval_actor(obs)[0]-.2).square().mean()+(net.eval_critic(obs)-1).square().mean()).backward()
        optimizer.step()
    for key, value in net.actor_encoder.state_dict().items():
        assert torch.equal(initial['actor_encoder.'+key], value)
    assert all(not p.requires_grad and p.grad is None for p in net.actor_encoder.parameters())
    for branch in ('coordination','critic_coordination','critic_encoder','action_head','value_head'):
        assert any(not torch.equal(initial[branch+'.'+k], v) for k,v in getattr(net,branch).state_dict().items())
    for ca in (net.coordination, net.critic_coordination):
        assert all(block.relation_bias.count_nonzero() > 0 for block in ca.blocks)
