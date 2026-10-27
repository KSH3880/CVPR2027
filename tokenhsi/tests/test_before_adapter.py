from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from learning.multi_agent.before_network import BeforeBuilder
from learning.multi_agent.stage2_network_builder import freeze_actor, transfer_stage1
from utils.before_spec import validate_env, checkpoint_contract, sample_graph
from utils.stage2_spec import check_checkpoint
from utils.task_role_spec import task_graph_packet
from utils.relation_task_spec import checkpoint_metadata

ROOT = Path(__file__).resolve().parents[2]
BASE = 'stage2_before_ground_mix226_atprogress_1200'
ADAPTER = 'stage2_before_ground_mix226_atprogress_adapter_1200'


def config(name, folder='multi_agent'):
    return yaml.safe_load((ROOT/f'tokenhsi/data/cfg/{folder}/{name}.yaml').read_text())


def build(name):
    torch.manual_seed(26)
    env = config(name)['env']
    builder = BeforeBuilder()
    builder.load(config(name, 'train/rlg')['params']['network'])
    net = builder.build('amp', actions_num=32, input_shape=(1268,), amp_input_shape=(1320,),
        value_size=1, num_agents=4, num_objects=8, observation_mode='clean_scene',
        scene_entity_sizes=[223, 30, 1], scene_kinematic_size=7, scene_arena_scale=16.,
        relation_reward_mode=env['relationReward']['mode'], relation_graph_spec=env['relationGraph'], device='cpu')
    net.actor_encoder.relation_graph_spec['shuffle_token_order'] = False
    net.coordination.shuffle_human_order = False
    net.critic_coordination.shuffle_human_order = False
    return net


def observations():
    env = config(ADAPTER)['env']
    graph, _, _ = sample_graph(torch.full((2, 4, 3), .4), env['box']['build']['taskSizeRanges'], 'before_stack')
    poses = torch.zeros(2, 16, 7); poses[..., 6] = 1
    return torch.cat((torch.randn(2, 1136), poses.flatten(1), task_graph_packet(graph, 2)), -1)


def test_config_only_changes_actor_adapter_and_rejects_mixed_checkpoints():
    old, new = config(BASE), config(ADAPTER)
    validate_env(new['env'])
    comparable = deepcopy(new)
    assert comparable['env']['relationGraph'].pop('actor_head_adapter') == 'tokenhsi_internal_v1'
    assert comparable['env'] == old['env'] and comparable['sim'] == old['sim']
    assert new['experiment']['transfer']['checkpoint'].endswith('00078000.pth')
    a, b = config(BASE, 'train/rlg'), config(ADAPTER, 'train/rlg')
    b['params']['config']['name'] = a['params']['config']['name']
    assert a == b
    check_checkpoint({'before_contract': checkpoint_contract(new['env'])}, new['env'])
    for src, dst in [(old, new), (new, old)]:
        with pytest.raises(ValueError, match='BEFORE checkpoint'):
            check_checkpoint({'before_contract': checkpoint_contract(src['env'])}, dst['env'])
    assert 'actor_head_adapter' not in checkpoint_contract(old['env'])
    new['env']['relationGraph']['actor_head_adapter'] = 'unknown'
    with pytest.raises(ValueError, match='actor_head_adapter'):
        validate_env(new['env'])


def test_initial_policy_identity_frozen_head_and_gradient_flow():
    torch.set_num_threads(1)
    baseline, net = build(BASE), build(ADAPTER)
    assert [(m.in_features, m.out_features) for m in net.action_adapters] == [(64, 1024), (1024, 512)]
    assert all(torch.count_nonzero(p) == 0 for p in net.action_adapters.parameters())
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(value, net.state_dict()[key], rtol=0, atol=0)
    obs = observations()
    torch.testing.assert_close(net.eval_actor(obs)[0], baseline.eval_actor(obs)[0], rtol=0, atol=0)
    torch.testing.assert_close(net.eval_critic(obs), baseline.eval_critic(obs), rtol=0, atol=0)
    model = torch.nn.Module(); model.a2c_network = net
    freeze_actor(model)
    snapshot = deepcopy(net.state_dict())
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=2e-5)
    for _ in range(2):
        opt.zero_grad()
        (net.eval_actor(obs)[0].square().mean() + net.eval_critic(obs).square().mean()).backward()
        assert all(p.grad is None and not p.requires_grad for p in net.actor_encoder.parameters())
        assert all(p.grad is None and not p.requires_grad for p in net.action_head.parameters())
        assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in net.action_adapters.parameters())
        assert net.coordination.blocks[0].attention.out_proj.weight.grad.abs().sum() > 0
        assert net.value_head[-1].weight.grad.abs().sum() > 0
        opt.step()
    for key, value in snapshot.items():
        if key.startswith(('actor_encoder.', 'action_head.')):
            torch.testing.assert_close(net.state_dict()[key], value, rtol=0, atol=0)
    assert all(torch.count_nonzero(p) > 0 for p in net.action_adapters.parameters())
    # Checkpoint restore keeps adapters and the opt-in freeze behavior.
    restored = build(ADAPTER); restored.load_state_dict(net.state_dict(), strict=True)
    wrapper = torch.nn.Module(); wrapper.a2c_network = restored; freeze_actor(wrapper)
    torch.testing.assert_close(restored.eval_actor(obs)[0], net.eval_actor(obs)[0], rtol=0, atol=0)
    assert all(not p.requires_grad for p in restored.action_head.parameters())
    wrapper.a2c_network = baseline; freeze_actor(wrapper)
    assert all(p.requires_grad for p in baseline.action_head.parameters())


def test_adapters_sum_before_relu_and_leave_output_layer_unadapted():
    net = build(ADAPTER)
    with torch.no_grad():
        net.action_adapters[0].weight.normal_()
        net.action_adapters[0].bias.fill_(-.5)
        net.action_adapters[1].weight.normal_()
        net.action_adapters[1].bias.fill_(-.5)
    x = torch.randn(2, 4, 64)
    h1 = torch.relu(net.action_head[0][0](x) + net.action_adapters[0](x))
    h2 = torch.relu(net.action_head[0][2](h1) + net.action_adapters[1](h1))
    torch.testing.assert_close(net._actor_actions(x), net.action_head[1](h2), rtol=0, atol=0)
    assert len(net.action_adapters) == 2


def test_stage1_transfer_preserves_all_source_tensors_and_zero_adapters():
    # A self-contained source with the exact transferable Stage1 tensor schema.
    base = torch.nn.Module(); base.a2c_network = build(BASE)
    source = {k: v for k, v in base.state_dict().items()
              if not k.startswith(('a2c_network.coordination.', 'a2c_network.critic_coordination.'))}
    env = config('stage1_v2_fixed')['env']
    checkpoint = dict(model=source, epoch=78000, relation_metadata=checkpoint_metadata(env['relationReward']),
                      relation_experiment_config={'env': env})
    target = torch.nn.Module(); target.a2c_network = build(ADAPTER)
    report = transfer_stage1(target, checkpoint)
    assert set(report['copied']) == set(source)
    for key, value in source.items():
        torch.testing.assert_close(target.state_dict()[key], value, rtol=0, atol=0)
    assert len([k for k in report['added'] if 'action_adapters.' in k]) == 4
    assert all(torch.count_nonzero(p) == 0 for p in target.a2c_network.action_adapters.parameters())
    assert all(not p.requires_grad for p in target.a2c_network.action_head.parameters())
