from copy import deepcopy

import pytest
import torch
from torch import nn

from learning.multi_agent.mean_cooperation import MeanCooperation
from learning.multi_agent.sct_coopft_network import SCTCoopFTBuilder
from learning.multi_agent.stage2_push_network import transfer_stage1
from utils.experiments import native_config, validate_experiment, ROOT
from test_stage2_push_common import _common_observation


def build():
    env = native_config('sct_coopft')['env']
    builder = SCTCoopFTBuilder()
    builder.load(native_config('sct_coopft', True)['params']['network'])
    return builder.build('amp', actions_num=32, input_shape=(1272,), amp_input_shape=(1330,),
        value_size=1, num_agents=4, num_objects=8, observation_mode='clean_scene',
        scene_entity_sizes=[223, 30, 1], scene_kinematic_size=7, scene_arena_scale=16.,
        relation_reward_mode=env['relationReward']['mode'],
        relation_graph_spec=env['relationGraph'], device='cpu')


def test_config_preserves_stage2_900_environment_and_ppo_amp():
    cfg = native_config('sct_coopft')
    original = native_config('stage2_900')
    assert cfg['env'] == original['env'] and cfg['sim'] == original['sim']
    assert cfg['experiment']['transfer'] == original['experiment']['transfer']
    train = native_config('sct_coopft', True)
    validate_experiment('sct_coopft', cfg['env'], deepcopy(train))
    previous = native_config('stage2_900', True)
    assert train['params']['algo'] == previous['params']['algo'] == {'name': 'ma'}
    assert 'distillation' not in train['params']['config']
    actual_config = dict(train['params']['config'], name=previous['params']['config']['name'])
    assert actual_config == previous['params']['config']
    for change in ({'algo': {'name': 'ma_distill'}}, {'distillation': {'version': 1}}):
        bad = deepcopy(train)
        if 'algo' in change:
            bad['params']['algo'] = change['algo']
        else:
            bad['params']['config'].update(change)
        with pytest.raises(ValueError, match='ordinary PPO'):
            validate_experiment('sct_coopft', cfg['env'], bad)


def test_partner_global_means_and_directed_relation_codes():
    module = MeanCooperation()
    packet = torch.tensor([[[1, 2, 0, 4, 12], [1, 0, 1, 4, 4],
                            [1, 4, 2, 6, 14], [1, 4, 3, 6, 15]]]).flatten(1)
    groups = torch.tensor([[0, 0, 2, 2]])
    humans = torch.randn(1, 4, 64)
    captured = []
    handle = module.blocks[0].mix.register_forward_pre_hook(lambda _, args: captured.append(args[0]))
    actual = module(humans, None, packet, None, groups=groups)
    handle.remove()
    values = module.blocks[0].query_norm(humans)
    expected_partner = torch.stack((torch.zeros_like(values[:, 0]), values[:, 0],
                                    values[:, 3], values[:, 2]), 1)
    expected_global = torch.stack([values[:, [j for j in range(4) if j != i]].mean(1)
                                   for i in range(4)], 1)
    torch.testing.assert_close(captured[0][..., 64:128], expected_partner)
    torch.testing.assert_close(captured[0][..., 128:192], expected_global)
    assert torch.equal(captured[0][..., 192:].argmax(-1), torch.tensor([[0, 1, 2, 2]]))
    assert torch.equal(actual, humans)


@pytest.mark.parametrize('people', [1, 2, 3, 4, 8, 16])
def test_scalability_zero_partners_and_permutation(people):
    module = MeanCooperation()
    assert sum(p.numel() for p in module.parameters()) == 150784
    assert not any(isinstance(m, (nn.Embedding, nn.MultiheadAttention)) for m in module.modules())
    for block in module.blocks:
        nn.init.normal_(block.mix[-1].weight, std=.02)
        nn.init.normal_(block.ffn[-1].weight, std=.02)
    humans = torch.randn(2, people, 64)
    ids = torch.arange(people)
    packet = torch.stack((torch.ones_like(ids), ids*0+4, ids, people+ids,
                          3*people+ids), -1)[None].expand(2, -1, -1).clone()
    groups = ids[None].expand(2, -1)
    observed = []
    handle = module.blocks[0].mix.register_forward_pre_hook(lambda _, args: observed.append(args[0]))
    actual = module(humans, None, packet.flatten(1), None, groups=groups)
    handle.remove()
    assert observed[0][..., 64:128].eq(0).all()
    if people == 1:
        assert observed[0][..., 128:192].eq(0).all()
    order = torch.randperm(people)
    packet[..., 2] = order.argsort()[packet[..., 2]]
    shuffled = module(humans[:, order], None, packet.flip(1).flatten(1), None,
                     groups=groups[:, order])
    torch.testing.assert_close(shuffled, actual[:, order], rtol=1e-5, atol=1e-6)
    assert torch.isfinite(actual).all()
    empty = packet.clone()
    empty[..., 0] = 0
    assert torch.equal(module(humans, None, empty.flatten(1), None, groups=groups), humans)


def test_actual_stage1_transfer_frozen_encoder_gradients_and_reload():
    torch.set_num_threads(1)
    checkpoint = torch.load(ROOT/native_config('sct_coopft')['experiment']['transfer']['checkpoint'],
                            map_location='cpu', weights_only=False)
    model = nn.Module()
    model.a2c_network = network = build()
    report = transfer_stage1(model, checkpoint)
    assert report['source_epoch'] == 26000 and len(report['copied']) == 165
    for key, value in checkpoint['model'].items():
        assert torch.equal(model.state_dict()[key], value)
    network.eval()
    network.actor_encoder.relation_graph_spec['shuffle_token_order'] = False
    obs = _common_observation('coupled')
    with torch.no_grad():
        humans = network._humans(network.actor_encoder, obs[:, :-4])
        torch.testing.assert_close(network.eval_actor(obs)[0], network.action_head(humans).reshape(4, -1),
                                   rtol=0, atol=0)
    loss = (network.eval_actor(obs)[0].square().mean()
            + network.eval_critic(obs).square().mean()
            + network.eval_disc(torch.randn(8, 1330)).square().mean())
    loss.backward()
    assert all(p.grad is None for p in network.actor_encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in network.critic_coordination.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in network.critic_encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in network._disc_mlp.parameters())
    for block in network.coordination.blocks:
        assert block.mix[-1].weight.grad.abs().sum() > 0
    assert network.action_head[-1].weight.grad.abs().sum() > 0
    restored = build()
    restored.load_state_dict(network.state_dict(), strict=True)
    restored.eval()
    restored.actor_encoder.relation_graph_spec['shuffle_token_order'] = False
    torch.testing.assert_close(restored.eval_actor(obs)[0], network.eval_actor(obs)[0], rtol=0, atol=0)
