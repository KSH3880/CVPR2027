from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from learning.multi_agent.stage2_jointpush_network import Stage2JointPushBuilder, transfer_stage1
from utils.stage2_jointpush import checkpoint_contract, check_checkpoint, compose_graph, validate_env
from utils.task_role_spec import task_graph_packet

BASE = 'stage2_jointpush_goal_saturation'
NAME = BASE + '_adapter'


def config(folder, name):
    return yaml.safe_load(Path('tokenhsi/data/cfg', folder, name + '.yaml').read_text())


def build():
    env = config('multi_agent', NAME)['env']
    builder = Stage2JointPushBuilder()
    builder.load(config('train/rlg', NAME)['params']['network'])
    model = nn.Module()
    model.a2c_network = builder.build('amp', actions_num=32, input_shape=(1272,), amp_input_shape=(1330,),
        value_size=1, num_agents=4, num_objects=8, observation_mode='clean_scene', scene_entity_sizes=[223,30,1],
        scene_kinematic_size=7, scene_arena_scale=16., relation_reward_mode=env['relationReward']['mode'],
        relation_graph_spec=env['relationGraph'], device='cpu')
    return model


def test_adapter_config_and_checkpoint_contract():
    original = config('multi_agent', BASE)
    variant = config('multi_agent', NAME)
    validate_env(variant['env'])
    check_checkpoint({'stage2_jointpush_contract': checkpoint_contract(variant['env'])}, variant['env'])
    with pytest.raises(ValueError):
        check_checkpoint({'stage2_jointpush_contract': checkpoint_contract(original['env'])}, variant['env'])
    assert variant['env']['stage2JointPush'].pop('action_adapter') == 'tokenhsi_internal'
    variant['experiment']['name'] = original['experiment']['name']
    assert variant == original
    original = config('train/rlg', BASE)
    variant = config('train/rlg', NAME)
    assert variant['params']['network'].pop('use_internal_adaptation') is True
    variant['params']['config']['name'] = original['params']['config']['name']
    assert variant == original


def test_adapter_stage1_identity_gradients_and_restore():
    checkpoint = Path('pth/Stage1PushFinetune_00014000.pth')
    if not checkpoint.exists():
        pytest.skip('User Stage1 14000 checkpoint is not distributed with the repository')
    model = build()
    weights = torch.load(checkpoint, map_location='cpu', weights_only=False)
    report = transfer_stage1(model, weights)
    assert report['source_epoch'] == 14000 and len(report['copied']) == 165
    net = model.a2c_network
    assert [(a.in_features, a.out_features) for a in net.action_adapters] == [(64,1024),(1024,512)]
    assert all(torch.count_nonzero(p) == 0 for p in net.action_adapters.parameters())
    assert all(not p.requires_grad for p in net.actor_encoder.parameters())
    assert all(not p.requires_grad for p in net.action_head.parameters())
    assert all(p.requires_grad for p in net.action_adapters.parameters())
    assert all(p.requires_grad for p in net.coordination.parameters())
    assert all(p.requires_grad for p in net.critic_encoder.parameters())
    obs = torch.zeros(2,1272)
    obs[:,1136:1248].reshape(2,16,7)[...,6] = 1.
    groups = torch.tensor([[0,0,2,3], [0,0,0,0]])
    graph = compose_graph(torch.full((2,4),5), groups)
    obs[:,1248:1268] = task_graph_packet(graph,2)
    obs[:,-4:] = groups
    net.eval()
    with torch.no_grad():
        reference = net.action_head(net.actor_encoder(obs[:,:-4])).reshape(8,32)
    action, _ = net.eval_actor(obs)
    torch.testing.assert_close(action, reference)
    (action * torch.randn_like(action)).sum().backward()
    assert all(p.grad is None for p in net.actor_encoder.parameters())
    assert all(p.grad is None for p in net.action_head.parameters())
    for adapter in net.action_adapters:
        assert torch.isfinite(adapter.weight.grad).all() and adapter.weight.grad.abs().sum() > 0
    assert net.coordination.blocks[-1].attention.out_proj.weight.grad.abs().sum() > 0
    with torch.no_grad():
        net.action_adapters[0].bias.add_(.1)
    changed, _ = net.eval_actor(obs)
    assert not torch.allclose(changed, reference)
    restored = build()
    restored.load_state_dict(model.state_dict(), strict=True)
    restored.a2c_network.eval()
    output, _ = restored.a2c_network.eval_actor(obs)
    torch.testing.assert_close(output, changed)
    assert all(not p.requires_grad for p in restored.a2c_network.action_head.parameters())
