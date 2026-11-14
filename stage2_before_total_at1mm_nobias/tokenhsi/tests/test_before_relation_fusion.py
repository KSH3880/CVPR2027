from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from learning.multi_agent.before_relation_fusion import BeforeRelationFusion, BeforeFusionCoordinationBlock
from learning.multi_agent.human_coordination import CoordinationBlock, HumanCoordination, stage2_human_relations
from learning.multi_agent.stage2_push_network import transfer_stage1
from test_stage2_push_common import build, _common_observation
from utils.before_object_alias import encoder_packet, goal_aligned_scene
from utils.before_spec import checkpoint_contract, check_checkpoint
from utils.experiments import native_config, validate_experiment
from utils.unified_training import validate_unified_env

BASE = 'stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2'
NAME = BASE + '_relfusion_zerobg'


def test_config_only_changes_fusion_and_experiment_name():
    cfg = native_config(NAME)
    validate_unified_env(cfg['env'])
    validate_experiment(NAME, cfg['env'], native_config(NAME, True))
    saved = {'before_contract': checkpoint_contract(cfg['env'])}
    check_checkpoint(saved, cfg['env'])
    with pytest.raises(ValueError, match='checkpoint contract'):
        check_checkpoint(saved, native_config(BASE)['env'])
    with pytest.raises(ValueError, match='checkpoint contract'):
        check_checkpoint({'before_contract': checkpoint_contract(native_config(BASE)['env'])}, cfg['env'])
    assert cfg['env']['relationGraph'].pop('human_relation_fusion') == 'before_zerobg'
    cfg['experiment']['name'] = BASE
    assert cfg == native_config(BASE)
    train = native_config(NAME, True)
    train['params']['config']['name'] = native_config(BASE, True)['params']['config']['name']
    assert train == native_config(BASE, True)


def test_unknown_fusion_rejected():
    env = native_config(NAME)['env']
    env['relationGraph']['human_relation_fusion'] = 'zerobg'
    with pytest.raises(ValueError, match='fusion'):
        validate_unified_env(env)


def test_attention_matches_legacy_output_and_input_gradient():
    torch.manual_seed(7)
    old = CoordinationBlock(64, 2, 128, 4).double()
    new = BeforeFusionCoordinationBlock(64, 2, 128).double()
    new.load_state_dict({k: v for k, v in old.state_dict().items() if k != 'relation_bias'}, strict=False)
    fusion = BeforeRelationFusion(64).double()
    with torch.no_grad():
        new.bias_projection.normal_()
        old.relation_bias.copy_(new.effective_relation_bias(fusion.embedding))
    relation = torch.tensor([[[1, 0, 0], [2, 1, 0], [0, 0, 1]]]).expand(2, -1, -1)
    padding = torch.zeros(2, 1, 3, 3, dtype=torch.bool)
    padding[0, :, :, 2] = True
    x = torch.randn(2, 3, 64, dtype=torch.double, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    expected, actual = old(x, None, relation, padding), new(y, None, relation, padding, fusion)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=1e-9, rtol=1e-9)


def test_only_follower_gets_fusion_before_projection():
    block = BeforeFusionCoordinationBlock(64, 2, 128)
    fusion = BeforeRelationFusion(64)
    with torch.no_grad():
        fusion.mlp[-1].bias.fill_(1.)
        block.attention.out_proj.weight.copy_(torch.eye(64)*2)
        block.attention.out_proj.bias.zero_()
        block.ffn[-1].weight.zero_()
        block.ffn[-1].bias.zero_()
    relation = torch.tensor([[[1, 0, 0, 0], [2, 1, 0, 0], [0, 0, 1, 3], [0, 0, 3, 1]]])
    x = torch.randn(1, 4, 64)
    actual = block(x, None, relation, torch.tensor(False), fusion)
    fusion.reset_output()
    baseline = block(x, None, relation, torch.tensor(False), fusion)
    torch.testing.assert_close(actual[:, 1]-baseline[:, 1], torch.full((1, 64), 2.))
    torch.testing.assert_close(actual[:, [0, 2, 3]], baseline[:, [0, 2, 3]], rtol=0, atol=0)


@pytest.mark.parametrize('kind', [0, 1, 3])
def test_two_pairs_packet_shuffle_identity_and_gradients(kind):
    torch.set_num_threads(1)
    torch.manual_seed(22)
    branch = HumanCoordination(relations=4, relation_fusion='before_zerobg')
    branch.init_identity()
    packet = torch.tensor([[[1, 2, 3, 4, 13], [1, kind, 0, 6, 4],
                            [1, 2, 1, 5, 15], [1, kind, 2, 7, 5]]]).reshape(1, -1)
    groups = torch.tensor([[3, 1, 1, 3]])
    relations = stage2_human_relations(packet, groups)
    assert (relations == 2).nonzero().tolist() == [[0, 0, 3], [0, 2, 1]]
    seen = []
    hook = branch.relation_fusion.register_forward_pre_hook(lambda module, args: seen.append(args[1].clone()))
    x = torch.randn(1, 4, 64)
    torch.testing.assert_close(branch(x, None, packet, None, groups=groups), x, atol=0, rtol=0)
    hook.remove()
    assert len(seen) == 4 and all(v.tolist() == [[True, False, True, False]] for v in seen)
    optimizer = torch.optim.Adam(branch.parameters(), lr=.001)
    target = torch.randn_like(x)
    for _ in range(5):
        optimizer.zero_grad()
        (branch(x, None, packet, None, groups=groups)-target).square().mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in branch.parameters() if p.grad is not None)
        optimizer.step()
    assert branch.relation_fusion.embedding.grad.abs().sum() > 0
    assert branch.relation_fusion.mlp[0].weight.grad.abs().sum() > 0
    assert branch.blocks[0].bias_projection.grad.abs().sum() > 0
    assert all(not b.effective_relation_bias(branch.relation_fusion.embedding)[:, [0, 1, 3]].any() for b in branch.blocks)
    expected = branch(x, None, packet, None, groups=groups)
    reordered = packet.reshape(1, 4, 5)[:, [2, 0, 3, 1]].reshape(1, -1)
    branch.shuffle_human_order = True
    with patch('torch.randperm', return_value=torch.tensor([3, 0, 2, 1])):
        actual = branch(x, None, reordered, None, groups=groups)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('case', ['before_sit', 'before_climb', 'before_ontop', 'two_pairs', 'solo', 'coupled'])
def test_actor_critic_keep_virtual_input_and_original_relation_packet(case):
    torch.set_num_threads(1)
    net = build(NAME).eval()
    net.coordination.shuffle_human_order = net.critic_coordination.shuffle_human_order = False
    obs = _common_observation(case)
    saved = obs.clone()
    scene, groups = obs[:, :-4], obs[:, -4:]
    packet = scene[:, -20:]
    virtual = goal_aligned_scene(scene, 4, 8)
    expected_scene = torch.cat((virtual[:, :-20], encoder_packet(packet, 4, 8)), -1)
    expected_active = (stage2_human_relations(packet, groups) == 2).any(-1)
    seen = []
    def encoder_hook(module, args):
        torch.testing.assert_close(args[0], expected_scene, rtol=0, atol=0)
    def fusion_hook(module, args):
        torch.testing.assert_close(args[1], expected_active, rtol=0, atol=0)
        seen.append(module)
    hooks = [e.register_forward_pre_hook(encoder_hook) for e in (net.actor_encoder, net.critic_encoder)]
    hooks += [c.relation_fusion.register_forward_pre_hook(fusion_hook) for c in (net.coordination, net.critic_coordination)]
    with torch.no_grad():
        net.eval_actor(obs)
        net.eval_critic(obs)
    for h in hooks:
        h.remove()
    assert len(seen) == 8 and torch.equal(obs, saved)
    assert net.coordination.relation_fusion is not net.critic_coordination.relation_fusion


def test_stage1_transfer_initial_outputs_and_strict_reload():
    torch.set_num_threads(1)
    checkpoint = torch.load(native_config(NAME)['experiment']['transfer']['checkpoint'], map_location='cpu', weights_only=False)
    model = torch.nn.Module()
    model.a2c_network = build(NAME)
    report = transfer_stage1(model, checkpoint)
    assert report['source_epoch'] == 26000
    assert all(torch.equal(model.state_dict()[key], value) for key, value in checkpoint['model'].items())
    net = model.a2c_network.eval()
    assert all(not p.requires_grad for p in net.actor_encoder.parameters())
    assert all(p.requires_grad for p in net.critic_encoder.parameters())
    assert all(p.requires_grad for p in net.coordination.parameters())
    net.relation_graph_spec['shuffle_token_order'] = False
    net.coordination.shuffle_human_order = net.critic_coordination.shuffle_human_order = False
    obs = _common_observation('two_pairs')
    with torch.no_grad():
        humans = net._humans(net.actor_encoder, obs[:, :-4])
        expected = net.action_head(humans).reshape(-1, 32)
        actual, sigma = net.eval_actor(obs)
        value = net.eval_critic(obs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.isfinite(value).all() and torch.isfinite(sigma).all()
    restored = deepcopy(net)
    restored.load_state_dict(net.state_dict(), strict=True)
    torch.testing.assert_close(restored.eval_actor(obs)[0], actual, rtol=0, atol=0)
