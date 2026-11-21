from copy import deepcopy

import pytest
import torch

from learning.multi_agent.bridge_waypoint_network import BridgeWaypointBuilder
from learning.multi_agent.sensor_stage2_network import SensorStage2Builder, freeze_backbone
from tests.test_sensor_stage2 import build
from utils.bridge_waypoint import staged_progress
from utils.experiments import SENSOR_FOLLOW_CAP_NAMES, native_config, validate_experiment
from utils.stage2_spec import compose_graph
from utils.task_role_spec import task_graph_packet


def test_configs_preserve_native_sampling_and_separate_runs():
    base = native_config('stage2_sensor_900_bridge_before_fixed_layout_cache')['env']
    for i, name in enumerate(SENSOR_FOLLOW_CAP_NAMES):
        config = native_config(name)
        env = config['env']
        validate_experiment(name, env, native_config(name, True))
        actual = deepcopy(env)
        bridge = actual['sensorScene']['bridge']
        assert bridge.pop('follow_cap') == {'lift': .05}
        assert bridge.pop('leader_waypoint') == dict(enabled=i > 0, switch_margin=1.,
                                                    reward='staged' if i == 2 else 'original')
        assert actual == base
        assert 'stage1_2_base_teacher' in config['experiment']['transfer']['checkpoint']


def test_staged_progress_is_continuous_at_switch_and_saturates():
    radius = torch.full((5,), 1.5)
    distance = torch.tensor([1.500001, 1.5, 8., 8., 8.])
    switched = torch.tensor([False, True, True, True, True])
    final = torch.tensor([.2, .2, .6, 1., .1])
    reference = torch.full((5,), .2)
    approach, finish = staged_progress(distance, radius, switched, final, reference, 1.)
    reward = .2*(approach+finish)
    torch.testing.assert_close(reward, torch.tensor([.2, .2, .3, .4, .2]), atol=1e-6, rtol=0)
    approach, finish = staged_progress(distance, radius, switched, torch.ones(5), torch.ones(5), 1.)
    assert torch.isfinite(finish).all()
    assert finish.tolist() == [0., 1., 1., 1., 1.]


@pytest.mark.parametrize('method', ['eval_actor', 'eval_critic'])
def test_guidance_changes_only_active_leaders_and_survives_minibatch_shuffle(method):
    torch.set_num_threads(1)
    torch.manual_seed(14)
    name = SENSOR_FOLLOW_CAP_NAMES[1]
    model = build(BridgeWaypointBuilder, name, 10493)
    base = build(SensorStage2Builder, 'stage2_sensor_900_bridge_before_fixed_layout_cache', 10488)
    base.load_state_dict(model.state_dict(), strict=True)
    tasks = torch.tensor([[3, 1, 5, 5]]).expand(4, -1)
    groups = torch.tensor([[0, 1, 2, 2]]).expand(4, -1)
    pre = torch.tensor([[-1, 0, -1, -1]]).expand(4, -1)
    graph = compose_graph(tasks, groups, pre, 8)
    poses = torch.randn(4, 16, 7)
    poses[..., 3:7] = torch.tensor([0., 0., 0., 1.])
    obs = torch.cat((torch.randn(4, 1136), poses.flatten(1), task_graph_packet(graph, 4).float(),
                     groups.float(), torch.randn(4, 9216)), -1)
    metadata = torch.tensor([[0., 1., 9., 8., 3.], [0., 0., 9., 8., 3.],
                             [0., 1., 8., 7., 2.], [0., 0., 9., 8., 3.]])
    full = torch.cat((obs, metadata), -1)
    def forward(net, values):
        result = getattr(net, method)(values)
        return result[0] if isinstance(result, tuple) else result
    got = forward(model.a2c_network, full).reshape(4, 4, -1)
    expected = forward(base.a2c_network, obs).reshape_as(got)
    mask = torch.ones(4, 4, dtype=torch.bool)
    mask[[0, 2], 0] = False
    torch.testing.assert_close(got[mask], expected[mask], rtol=0, atol=0)
    assert (got[~mask]-expected[~mask]).abs().max() > 1e-6
    order = torch.tensor([3, 0, 2, 1])
    shuffled = forward(model.a2c_network, full[order]).reshape_as(got)
    torch.testing.assert_close(shuffled, got[order], atol=2e-6, rtol=2e-6)
    freeze_backbone(model)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=2e-5)
    for _ in range(2):
        optimizer.zero_grad()
        forward(model.a2c_network, full).square().mean().backward()
        optimizer.step()
    prefix = 'actor' if method == 'eval_actor' else 'critic'
    assert getattr(model.a2c_network, prefix+'_sensor')[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.a2c_network.actor_encoder.parameters())
