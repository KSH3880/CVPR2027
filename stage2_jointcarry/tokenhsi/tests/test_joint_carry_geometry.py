from copy import deepcopy

import pytest
import torch
import yaml

from learning.multi_agent.task_geometry import canonical_pose, relative_pose
from learning.multi_agent.task_coordination import TaskCoordination
from learning.multi_agent.task_role_encoder import TaskTypeEmbeddingBias
from learning.multi_agent.amp_network_builder_ma import _quat_to_rotation_matrix
from learning.multi_agent.stage2_transfer import transfer_stage1_weights
from utils.joint_carry_amp import validate_ablation, check_ablation_checkpoint
from utils.joint_carry_spec import validate_env
from utils.relation_task_spec import checkpoint_metadata
from test_joint_carry_amp_ablation import configs, network, ROOT
from test_joint_carry import build_network, joint_observations
from test_stage2_coordination import Wrapper

BASE = 'locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_task_embedding'
NEW = BASE.replace('_task_embedding', '_geometry_task_embedding')


def test_config_and_checkpoint_isolation():
    env, train = configs(NEW)
    base, base_train = configs(BASE)
    validate_env(env)
    validate_ablation(env, train)
    restored = deepcopy(env)
    restored['jointCarryAblation']['policy'] = base['jointCarryAblation']['policy']
    assert restored == base
    restored = deepcopy(train)
    restored['params']['config']['name'] = base_train['params']['config']['name']
    assert restored['params']['network']['coordination'].pop('task_input') == 'geometry'
    assert restored == base_train
    weights = dict(joint_carry_loco_platform=env['jointCarryRsi']['locoPlatform'],
        joint_carry_rsi_dataset=env['jointCarryRsi']['dataset_sha256'],
        joint_carry_goal_sampling=env['jointCarryRsi']['goalSampling'],
        joint_carry_ablation=env['jointCarryAblation'])
    check_ablation_checkpoint(weights, env)
    with pytest.raises(ValueError, match='ablation differs'):
        check_ablation_checkpoint(weights, base)
    with pytest.raises(ValueError, match='coordination'):
        validate_ablation(env, base_train)
    basename = 'approach_stage2_joint_carry_mixed80_' + NEW
    for mode in ('train', 'test'):
        script = (ROOT/'scripts/multi_agent'/(basename+'_'+mode+'.sh')).read_text()
        assert 'amp_ma_stage2_joint_carry_mixed80_'+NEW+'.yaml' in script
        assert basename+'.yaml' in script
    assert basename+'_test.sh' in (ROOT/'scripts/multi_agent'/(basename+'_vnc.sh')).read_text()


def test_relative_pose_matrix_oracle_and_quaternion_sign():
    torch.manual_seed(25)
    source, target = canonical_pose(torch.randn(2,6,7)), canonical_pose(torch.randn(2,6,7))
    actual = relative_pose(source, target)
    rs, rt = _quat_to_rotation_matrix(source[...,3:]), _quat_to_rotation_matrix(target[...,3:])
    expected_pos = (rs.transpose(-1,-2) @ (target[...,:3]-source[...,:3]).unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(actual[...,:3], expected_pos, atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(_quat_to_rotation_matrix(actual[...,3:]), rs.transpose(-1,-2) @ rt, atol=2e-6, rtol=1e-5)
    source[...,3:] *= -1
    target[...,3:] *= -1
    torch.testing.assert_close(actual, relative_pose(source,target),atol=1e-6,rtol=1e-5)
    source[...,:3] += 15
    target[...,:3] += 15
    torch.testing.assert_close(actual,relative_pose(source,target),atol=2e-6,rtol=1e-5)


def test_geometry_slots_permutation_and_feature_independence():
    torch.manual_seed(18)
    torch.set_num_threads(1)
    ca = TaskCoordination(task_input='geometry',num_layers=4,feedforward_dim=128,residual_output=True)
    embed = TaskTypeEmbeddingBias(4,2).requires_grad_(False)
    nodes = torch.randn(2,8,64)
    poses = canonical_pose(torch.randn(2,8,7))
    packet = torch.tensor([[[1,2,0,2,6],[1,2,1,2,6],[0,0,0,0,0]],
                           [[1,2,0,2,6],[1,3,1,3,4],[0,0,0,0,0]]])
    seen=[]
    hook=ca.grounding.register_forward_pre_hook(lambda module,args: seen.append(args[0].detach().clone()))
    tokens,_=ca.task_tokens(nodes,packet,embed,poses)
    features=seen[-1]
    assert features.shape == (2,3,85)
    torch.testing.assert_close(features[:,:2,64:71],poses[:,:2])
    torch.testing.assert_close(features[1,1,78:85],relative_pose(poses[1,1],poses[1,4]))
    torch.testing.assert_close(tokens,ca.task_tokens(torch.randn_like(nodes),packet,embed,poses)[0])
    assert tokens[:,2].count_nonzero()==0
    hook.remove()
    expected=ca(nodes[:,:2],nodes,packet,embed,entity_poses=poses)
    ca.shuffle_task_order=True
    torch.testing.assert_close(expected,ca(nodes[:,:2],nodes,packet[:,[2,1,0]],embed,entity_poses=poses),atol=2e-6,rtol=1e-5)
    order=torch.tensor([1,0,5,3,2,4,7,6])
    moved=packet.clone();moved[...,2:]=order.argsort()[packet[...,2:]]
    changed=nodes[:,order]
    torch.testing.assert_close(expected[:,[1,0]],ca(changed[:,:2],changed,moved,embed,entity_poses=poses[:,order]),atol=2e-6,rtol=1e-5)
    empty=ca(nodes[:,:2],nodes,torch.zeros_like(packet),embed,entity_poses=poses)
    torch.testing.assert_close(empty,nodes[:,:2])
    with pytest.raises(ValueError,match='canonical entity poses'):
        ca(nodes[:,:2],nodes,packet,embed)
    expected.square().mean().backward()
    assert ca.grounding[0].weight.grad.abs().sum()>0
    assert embed.category_embed.weight.grad is None


def test_full_network_transfer_shuffle_and_learning():
    torch.manual_seed(123)
    torch.set_num_threads(1)
    source=Wrapper(build_network(False))
    target=network(NEW)
    reward=yaml.safe_load((ROOT/'data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding.yaml').read_text())['env']['relationReward']
    report=transfer_stage1_weights(target,dict(model=source.state_dict(),relation_metadata=checkpoint_metadata(reward)))
    net=target.a2c_network
    assert report['head_expansion']==dict(source_columns=64,context_columns=0)
    assert net.coordination.grounding[0].in_features==net.critic_coordination.grounding[0].in_features==85
    obs=joint_observations()
    with torch.no_grad():
        torch.testing.assert_close(net.eval_actor(obs)[0],source.a2c_network.eval_actor(obs)[0],atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(net.eval_critic(obs),source.a2c_network.eval_critic(obs),atol=1e-6,rtol=1e-5)
    frozen={k:v.clone() for k,v in net.actor_encoder.state_dict().items()}
    initial={k:v.clone() for k,v in net.state_dict().items()}
    optimizer=torch.optim.Adam([p for p in net.parameters() if p.requires_grad],lr=1e-3)
    for _ in range(4):
        optimizer.zero_grad()
        ((net.eval_actor(obs)[0]-.2).square().mean()+(net.eval_critic(obs)-1).square().mean()).backward()
        optimizer.step()
    assert all(torch.equal(v,net.actor_encoder.state_dict()[k]) for k,v in frozen.items())
    assert all(p.grad is None and not p.requires_grad for p in net.actor_encoder.parameters())
    for branch in ['coordination','critic_coordination','critic_encoder','action_head','value_head']:
        assert any(not torch.equal(initial[branch+'.'+k],v) for k,v in getattr(net,branch).state_dict().items())
    for branch in ['coordination','critic_coordination']:
        key=branch+'.grounding.0.weight'
        assert not torch.equal(initial[key],net.state_dict()[key])
    with torch.no_grad():
        expected_actor,expected_value=net.eval_actor(obs)[0],net.eval_critic(obs)
        for _ in range(3):
            torch.testing.assert_close(net.eval_actor(obs)[0],expected_actor,atol=3e-6,rtol=1e-5)
            torch.testing.assert_close(net.eval_critic(obs),expected_value,atol=3e-6,rtol=1e-5)
