from copy import deepcopy

import pytest
import torch
import yaml

from learning.multi_agent.task_coordination import TaskCoordination
from learning.multi_agent.task_role_encoder import TaskTypeEmbeddingBias
from learning.multi_agent.ca_none_mask_inference import enable_ca_none_mask
from learning.multi_agent.stage2_transfer import transfer_stage1_weights
from utils.joint_carry_amp import validate_ablation, check_ablation_checkpoint
from utils.joint_carry_spec import validate_env
from utils.relation_task_spec import checkpoint_metadata
from test_joint_carry_amp_ablation import configs, network, ROOT
from test_joint_carry import build_network, joint_observations
from test_stage2_coordination import Wrapper

BASE = 'locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_task_embedding'
NEW = BASE.replace('_task_embedding', '_nonemask_task_embedding')


def test_config_and_checkpoint_isolation():
    env, train = configs(NEW)
    base, old_train = configs(BASE)
    validate_env(env)
    validate_ablation(env, train)
    restored = deepcopy(env)
    restored['jointCarryAblation']['policy'] = base['jointCarryAblation']['policy']
    assert restored == base
    restored = deepcopy(train)
    assert restored['params']['network']['coordination'].pop('mask_none') is True
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


def test_training_mask_matches_inference_and_zero_none_gradient():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    ca = TaskCoordination(num_layers=4, feedforward_dim=128, residual_output=True, mask_none=True)
    ref = deepcopy(ca)
    ref.mask_none = False
    enable_ca_none_mask(ref)
    embed = TaskTypeEmbeddingBias(4, 2)
    nodes = torch.randn(3, 8, 64, requires_grad=True)
    packet = torch.tensor([[[1,2,0,2,6],[1,2,1,2,6]],
                           [[1,2,0,2,6],[1,3,1,3,4]],
                           [[0,0,0,0,0],[0,0,0,0,0]]])
    actual = ca(nodes[:,:2], nodes, packet, embed)
    torch.testing.assert_close(actual, ref(nodes[:,:2], nodes, packet, embed), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(actual[2], nodes[2,:2], atol=0, rtol=0)
    ca.shuffle_task_order = True
    torch.testing.assert_close(actual, ca(nodes[:,:2], nodes, packet[:,[1,0]], embed), atol=1e-6, rtol=1e-5)
    order = torch.tensor([1,0,4,3,2,5,7,6])
    moved = packet.clone(); moved[...,2:] = order.argsort()[packet[...,2:]]
    changed = nodes[:,order]
    torch.testing.assert_close(actual[:,[1,0]], ca(changed[:,:2],changed,moved,embed),atol=2e-6,rtol=1e-5)
    actual.square().mean().backward()
    assert torch.isfinite(nodes.grad).all()
    for block in ca.blocks:
        assert block.relation_bias.grad[:,0].count_nonzero() == 0
        assert block.relation_bias.grad[:,1:].abs().sum() > 0


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
        assert ca.mask_none and ca.task_input == 'sa_features'
        assert ca.grounding[0].in_features == 256
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
        assert all(block.relation_bias[:,0].count_nonzero() == 0 for block in ca.blocks)
