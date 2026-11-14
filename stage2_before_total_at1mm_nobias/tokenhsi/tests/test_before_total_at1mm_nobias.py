from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

from learning.multi_agent.before_relation_fusion import BeforeFusionCoordinationBlock, BeforeRelationFusion
from learning.multi_agent.human_coordination import HumanCoordination
from learning.multi_agent.stage2_push_network import transfer_stage1
from test_stage2_push_common import build, _common_observation
from utils.before_spec import check_checkpoint, checkpoint_contract
from utils.experiments import experiment_horizon, native_config, validate_experiment
from utils.unified_training import validate_unified_env

BASE = 'stage2_before_total_at1mm'
NAME = BASE + '_nobias'
MODE = 'before_zerobg_nobias'


def test_config_and_checkpoint_isolation():
    cfg, base = native_config(NAME), native_config(BASE)
    train = native_config(NAME, True)
    validate_unified_env(cfg['env'])
    validate_experiment(NAME, cfg['env'], deepcopy(train))
    assert experiment_horizon(NAME) == cfg['env']['episodeLength'] == 900
    assert not train['params']['load_checkpoint']
    for source, target in ((cfg, base), (base, cfg)):
        saved = {'before_contract': checkpoint_contract(source['env'])}
        check_checkpoint(saved, source['env'])
        with pytest.raises(ValueError, match='checkpoint contract'):
            check_checkpoint(saved, target['env'])
    assert cfg['env']['relationGraph']['human_relation_fusion'] == MODE
    cfg['env']['relationGraph']['human_relation_fusion'] = 'before_zerobg'
    cfg['experiment']['name'] = BASE
    assert cfg == base
    train['params']['config']['name'] = 'Stage2BeforeTotalAt1mm'
    assert train == native_config(BASE, True)


@pytest.mark.parametrize('masked', [False, True])
def test_matches_zero_bias_output_and_gradients(masked):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    base = BeforeFusionCoordinationBlock(64, 2, 128).double()
    block = BeforeFusionCoordinationBlock(64, 2, 128, relation_bias=False).double()
    block.load_state_dict({k: v for k, v in base.state_dict().items() if k != 'bias_projection'})
    fusion = BeforeRelationFusion(64).double()
    with torch.no_grad():
        fusion.mlp[-1].weight.normal_(std=.1)
    other_fusion = deepcopy(fusion)
    relations = torch.tensor([[[1, 0, 0], [2, 1, 0], [0, 0, 1]]]).expand(2, -1, -1)
    padding = torch.zeros(2, 1, 3, 3, dtype=torch.bool)
    if masked:
        padding[0, :, :, 2] = True
    x = torch.randn(2, 3, 64, dtype=torch.double, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    expected = base(x, None, relations, padding, fusion)
    with patch.object(block, 'effective_relation_bias', side_effect=RuntimeError('bias path called')):
        with patch.object(F, 'scaled_dot_product_attention', wraps=F.scaled_dot_product_attention) as attention:
            actual = block(y, None, relations, padding if masked else None, other_fusion)
            if not masked:
                assert attention.call_args.kwargs['attn_mask'] is None
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(fusion.embedding.grad, other_fusion.embedding.grad)
    assert other_fusion.embedding.grad.abs().sum() > 0


def test_fusion_learns_and_person_shuffle_preserves_outputs():
    torch.set_num_threads(1)
    torch.manual_seed(22)
    branch = HumanCoordination(relations=4, relation_fusion=MODE)
    branch.init_identity()
    packet = torch.tensor([[[1, 2, 3, 4, 13], [1, 0, 0, 6, 4],
                            [1, 2, 1, 5, 15], [1, 1, 2, 7, 5]]]).reshape(1, -1)
    groups = torch.tensor([[3, 1, 1, 3]])
    x, target = torch.randn(1, 4, 64), torch.randn(1, 4, 64)
    seen = []
    hook = branch.relation_fusion.register_forward_pre_hook(lambda module, args: seen.append(args[1].clone()))
    torch.testing.assert_close(branch(x, None, packet, None, groups=groups), x, atol=0, rtol=0)
    hook.remove()
    assert len(seen) == 4 and all(v.tolist() == [[True, False, True, False]] for v in seen)
    optimizer = torch.optim.Adam(branch.parameters(), lr=.001)
    for _ in range(5):
        optimizer.zero_grad()
        (branch(x, None, packet, None, groups=groups)-target).square().mean().backward()
        optimizer.step()
    assert branch.relation_fusion.embedding.grad.abs().sum() > 0
    assert branch.relation_fusion.mlp[0].weight.grad.abs().sum() > 0
    assert not any('bias_projection' in key for key in branch.state_dict())
    expected = branch(x, None, packet, None, groups=groups)
    branch.shuffle_human_order = True
    with patch('torch.randperm', return_value=torch.tensor([3, 0, 2, 1])):
        actual = branch(x, None, packet, None, groups=groups)
    torch.testing.assert_close(actual, expected)


def test_actor_critic_transfer_and_strict_reload():
    torch.set_num_threads(1)
    model = torch.nn.Module()
    model.a2c_network = build(NAME)
    checkpoint = torch.load(native_config(NAME)['experiment']['transfer']['checkpoint'], map_location='cpu', weights_only=False)
    assert transfer_stage1(model, checkpoint)['source_epoch'] == 26000
    assert all(torch.equal(model.state_dict()[key], value) for key, value in checkpoint['model'].items())
    net = model.a2c_network.eval()
    assert all(not p.requires_grad for p in net.actor_encoder.parameters())
    for branch in (net.coordination, net.critic_coordination):
        assert branch.relation_fusion is not None
        assert all(block.bias_projection is None for block in branch.blocks)
        branch.shuffle_human_order = False
    net.relation_graph_spec['shuffle_token_order'] = False
    obs = _common_observation('two_pairs')
    with patch.object(F, 'scaled_dot_product_attention', wraps=F.scaled_dot_product_attention) as attention:
        actor = net.eval_actor(obs)[0]
        critic = net.eval_critic(obs)
    assert attention.call_count == 8
    assert all(call.kwargs['attn_mask'] is None for call in attention.call_args_list)
    restored = build(NAME).eval()
    restored.load_state_dict(net.state_dict(), strict=True)
    restored.relation_graph_spec['shuffle_token_order'] = False
    restored.coordination.shuffle_human_order = restored.critic_coordination.shuffle_human_order = False
    torch.testing.assert_close(restored.eval_actor(obs)[0], actor, atol=0, rtol=0)
    torch.testing.assert_close(restored.eval_critic(obs), critic, atol=0, rtol=0)
