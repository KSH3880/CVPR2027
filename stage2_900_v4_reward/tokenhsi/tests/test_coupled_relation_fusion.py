from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from learning.multi_agent.coupled_relation_fusion import CoupledRelationFusion, FusionCoordinationBlock
from learning.multi_agent.human_coordination import CoordinationBlock, HumanCoordination
from learning.multi_agent.stage2_push_network import transfer_stage1
from test_stage2_push_common import build, _common_observation
from utils.coupled_push_spec import checkpoint_contract, check_checkpoint, compose_graph
from utils.experiments import native_config, validate_experiment
from utils.task_role_spec import task_graph_packet

BASE = 'stage2_coupled_push_individual_goal_solo5_carry5_ground'
MODES = ('learnedbg', 'zerobg')


@pytest.mark.parametrize('mode', MODES)
def test_config_changes_only_fusion_and_identity(mode):
    name = BASE+'_relfusion_'+mode
    cfg = native_config(name)
    validate_experiment(name, cfg['env'], native_config(name, True))
    saved = {'coupled_push_contract': checkpoint_contract(cfg['env'])}
    check_checkpoint(saved, cfg['env'])
    for other in (BASE, BASE+'_relfusion_'+('zerobg' if mode == 'learnedbg' else 'learnedbg')):
        with pytest.raises(ValueError, match='settings differ'):
            check_checkpoint(saved, native_config(other)['env'])
    assert cfg['env']['relationGraph'].pop('human_relation_fusion') == mode
    cfg['experiment']['name'] = BASE
    assert cfg == native_config(BASE)
    train = native_config(name, True)
    train['params']['config']['name'] = native_config(BASE, True)['params']['config']['name']
    assert train == native_config(BASE, True)


@pytest.mark.parametrize('mode', MODES)
def test_manual_attention_matches_legacy_outputs_and_input_gradients(mode):
    torch.manual_seed(12)
    old = CoordinationBlock(64, 2, 128, 4).double()
    new = FusionCoordinationBlock(64, 2, 128, 4, mode).double()
    new.load_state_dict({k: v for k, v in old.state_dict().items() if k != 'relation_bias'}, strict=False)
    fusion = CoupledRelationFusion(64).double()
    with torch.no_grad():
        new.bias_projection.normal_()
        if mode == 'learnedbg':
            new.background_bias.normal_()
        old.relation_bias.copy_(new.effective_relation_bias(fusion.embedding))
    relation = torch.tensor([[[1, 3, 0], [3, 1, 0], [0, 0, 1]]]).expand(2, -1, -1)
    padding = torch.zeros(2, 1, 3, 3, dtype=torch.bool)
    padding[0, :, :, 2] = True
    x = torch.randn(2, 3, 64, dtype=torch.double, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    expected = old(x, None, relation, padding)
    actual = new(y, None, relation, padding, fusion)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=1e-9, rtol=1e-9)


def test_fusion_is_before_output_projection_and_zero_for_solo():
    block = FusionCoordinationBlock(64, 2, 128, 4, 'zerobg')
    fusion = CoupledRelationFusion(64)
    with torch.no_grad():
        fusion.mlp[-1].bias.fill_(1.)
        block.attention.out_proj.weight.copy_(torch.eye(64)*2)
        block.attention.out_proj.bias.zero_()
        block.ffn[-1].weight.zero_()
        block.ffn[-1].bias.zero_()
    relations = torch.tensor([[[1, 3, 0], [3, 1, 0], [0, 0, 1]]])
    x = torch.randn(1, 3, 64)
    actual = block(x, None, relations, torch.tensor(False), fusion)
    fusion.reset_output()
    baseline = block(x, None, relations, torch.tensor(False), fusion)
    torch.testing.assert_close(actual[:, :2]-baseline[:, :2], torch.full((1, 2, 64), 2.))
    torch.testing.assert_close(actual[:, 2], baseline[:, 2], rtol=0, atol=0)


@pytest.mark.parametrize('people', [1, 2, 3, 4, 8])
def test_one_condition_per_person_independent_of_group_size(people):
    branch = HumanCoordination(relations=4, relation_fusion='zerobg')
    groups = torch.zeros(1, people, dtype=torch.long)
    packet = task_graph_packet(compose_graph(torch.full_like(groups, 5), groups, False), 1)
    seen = []
    handle = branch.relation_fusion.register_forward_pre_hook(lambda module, args: seen.append(args[1].clone()))
    branch(torch.randn(1, people, 64), None, packet, None, groups=groups)
    handle.remove()
    assert len(seen) == 4
    assert all(torch.equal(mask, torch.full((1, people), people > 1)) for mask in seen)


@pytest.mark.parametrize('mode', MODES)
def test_mixed_groups_shuffle_and_gradients(mode):
    torch.manual_seed(22)
    branch = HumanCoordination(relations=4, relation_fusion=mode)
    branch.init_identity()
    groups = torch.tensor([[0, 0, 0, 0, 4, 4, 6, 7]])
    packet = task_graph_packet(compose_graph(torch.full_like(groups, 5), groups, False), 1)
    x = torch.randn(1, 8, 64)
    torch.testing.assert_close(branch(x, None, packet, None, groups=groups), x, atol=0, rtol=0)
    optimizer = torch.optim.Adam(branch.parameters(), lr=.001)
    target = torch.randn_like(x)
    for _ in range(5):
        optimizer.zero_grad()
        out = branch(x, None, packet, None, groups=groups)
        (out-target).square().mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in branch.parameters() if p.grad is not None)
        optimizer.step()
    assert branch.relation_fusion.embedding.grad.abs().sum() > 0
    assert branch.relation_fusion.mlp[0].weight.grad.abs().sum() > 0
    assert branch.blocks[0].bias_projection.grad.abs().sum() > 0
    if mode == 'zerobg':
        assert not any('background_bias' in name for name, _ in branch.named_parameters())
        assert all(not block.effective_relation_bias(branch.relation_fusion.embedding)[:, :2].any() for block in branch.blocks)
    else:
        assert branch.blocks[0].background_bias.grad.abs().sum() > 0
    expected = branch(x, None, packet, None, groups=groups)
    branch.shuffle_human_order = True
    with patch('torch.randperm', return_value=torch.tensor([7, 0, 4, 2, 6, 1, 3, 5])):
        actual = branch(x, None, packet, None, groups=groups)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_ab_only_background_parameters_differ_and_initial_weights_match():
    torch.manual_seed(31)
    a = build(BASE+'_relfusion_learnedbg')
    torch.manual_seed(31)
    b = build(BASE+'_relfusion_zerobg')
    extra = set(a.state_dict())-set(b.state_dict())
    assert len(extra) == 8 and all(k.endswith('background_bias') for k in extra)
    assert set(b.state_dict()).issubset(a.state_dict())
    for key, value in b.state_dict().items():
        torch.testing.assert_close(value, a.state_dict()[key], atol=0, rtol=0)


@pytest.mark.parametrize('mode', MODES)
def test_stage1_transfer_initial_policy_and_strict_reload(mode):
    torch.set_num_threads(1)
    name = BASE+'_relfusion_'+mode
    checkpoint = torch.load(native_config(name)['experiment']['transfer']['checkpoint'], map_location='cpu', weights_only=False)
    model = torch.nn.Module()
    model.a2c_network = build(name)
    report = transfer_stage1(model, checkpoint)
    assert report['source_epoch'] == 26000
    assert all(torch.equal(model.state_dict()[key], value) for key, value in checkpoint['model'].items())
    assert all(not p.requires_grad for p in model.a2c_network.actor_encoder.parameters())
    assert all(p.requires_grad for p in model.a2c_network.coordination.parameters())
    net = model.a2c_network.eval()
    net.relation_graph_spec['shuffle_token_order'] = False
    net.coordination.shuffle_human_order = net.critic_coordination.shuffle_human_order = False
    obs = _common_observation('coupled')
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
