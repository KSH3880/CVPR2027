from copy import deepcopy

import pytest
import torch

from learning.multi_agent.bridge_waypoint_single_pass_network import BridgeWaypointSinglePassBuilder
from learning.multi_agent.sensor_stage2_network import SensorStage2Builder, freeze_backbone
from tests.test_sensor_stage2 import build
from utils.before_object_alias import alias_layout, encoder_packet
from utils.coupled_push_spec import checkpoint_contract, check_checkpoint
from utils.experiments import SENSOR_FOLLOW_CAP_SINGLE_PASS_NAMES, native_config, validate_experiment
from utils.stage2_spec import compose_graph
from utils.task_role_spec import task_graph_packet


@pytest.mark.parametrize('name', SENSOR_FOLLOW_CAP_SINGLE_PASS_NAMES)
def test_new_configs_change_only_policy_input_and_reject_old_checkpoints(name):
    old = name[:-len('_single_pass')]
    config = native_config(name)
    env = deepcopy(config['env'])
    validate_experiment(name, env, native_config(name, True))
    assert env['sensorScene']['bridge']['leader_waypoint'].pop('policy_input') == 'shared_gta_single_pass_v1'
    assert env == native_config(old)['env']
    assert config['experiment']['transfer'] == native_config(old)['experiment']['transfer']
    old_checkpoint = dict(coupled_push_contract=checkpoint_contract(env))
    with pytest.raises(ValueError):
        check_checkpoint(old_checkpoint, config['env'])
    with pytest.raises(ValueError):
        check_checkpoint(dict(coupled_push_contract=checkpoint_contract(config['env'])), env)


@pytest.mark.parametrize('family', [1, 2, 4])
@pytest.mark.parametrize('method', ['eval_actor', 'eval_critic'])
@pytest.mark.parametrize('active', [[True, False, True, False], [False]*4, [True]*4])
def test_one_forward_preserves_follower_goal_and_uses_shared_scene(family, method, active):
    torch.set_num_threads(1)
    torch.manual_seed(14)
    model = build(BridgeWaypointSinglePassBuilder, SENSOR_FOLLOW_CAP_SINGLE_PASS_NAMES[0], 10493)
    base = build(SensorStage2Builder, 'stage2_sensor_900_bridge_before_fixed_layout_cache', 10488)
    base.load_state_dict(model.state_dict(), strict=True)
    rows = torch.arange(4)
    leaders, followers = rows, (rows+1) % 4
    tasks = torch.full((4, 4), 5)
    tasks[rows, leaders], tasks[rows, followers] = 3, family
    groups = torch.arange(4).expand(4, -1).clone()
    for row in rows:
        pushers = (tasks[row] == 5).nonzero().flatten()
        groups[row, pushers] = pushers[0]
    pre = torch.full((4, 4), -1)
    pre[rows, followers] = leaders
    graph = compose_graph(tasks, groups, pre, 8)
    packet = task_graph_packet(graph, 4).float()
    poses = torch.randn(4, 16, 7)
    poses[..., 3:7] = torch.tensor([0., 0., 0., 1.])
    features = torch.randn(4, 1136)
    sensor = torch.randn(4, 9216)
    scene = torch.cat((features, poses.flatten(1), packet), -1)
    obs = torch.cat((scene, groups.float(), sensor), -1)
    goals = torch.randn(4, 3)+10
    active = torch.tensor(active)
    full = torch.cat((obs, leaders[:, None].float(), active[:, None].float(), goals), -1)
    untouched = full.clone()

    expected_poses = poses.clone()
    dependent, _, alias, _ = alias_layout(packet, 4, 8)
    r, record = dependent.nonzero(as_tuple=True)
    expected_poses[r, alias[r, record], :3] = poses[r, 12+leaders[r], :3]
    expected_poses[rows[active], 12+leaders[active], :3] = goals[active]
    expected_scene = torch.cat((features, expected_poses.flatten(1), encoder_packet(packet, 4, 8)), -1)
    branch = 'actor' if method == 'eval_actor' else 'critic'
    encoder = getattr(model.a2c_network, branch+'_encoder')
    coordination = getattr(model.a2c_network, 'coordination' if branch == 'actor' else 'critic_coordination')
    calls = []
    def encoder_input(module, args):
        calls.append('encoder')
        torch.testing.assert_close(args[0], expected_scene, rtol=0, atol=0)
    hooks = [encoder.register_forward_pre_hook(encoder_input),
             coordination.register_forward_pre_hook(lambda module, args: calls.append('coordination'))]
    def forward(net, values):
        result = getattr(net, method)(values)
        return result[0] if isinstance(result, tuple) else result
    got = forward(model.a2c_network, full)
    for hook in hooks:
        hook.remove()
    assert calls == ['encoder', 'coordination']
    assert torch.equal(full, untouched)
    base.a2c_network.relation_graph_spec['before_virtual_object'] = 'already_resolved'
    reference = torch.cat((features, expected_poses.flatten(1), packet, groups.float(), sensor), -1)
    torch.testing.assert_close(got, forward(base.a2c_network, reference), rtol=0, atol=0)
    order = torch.tensor([3, 0, 2, 1])
    torch.testing.assert_close(forward(model.a2c_network, full[order]).reshape(4, 4, -1),
                              got.reshape(4, 4, -1)[order], atol=2e-6, rtol=2e-6)
    freeze_backbone(model)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=2e-5)
    for _ in range(2):
        optimizer.zero_grad()
        forward(model.a2c_network, full).square().mean().backward()
        optimizer.step()
    assert getattr(model.a2c_network, branch+'_sensor')[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.a2c_network.actor_encoder.parameters())
