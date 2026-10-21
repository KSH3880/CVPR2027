from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn

from env.tasks.multi_agent.cliff_geometry import (
    height_observation, sample_layout, sensor_grid, sensor_positions, stage_height, terrain_fall)
from learning.multi_agent.amp_network_builder_ma import AMPMultiAgentBuilder
from learning.multi_agent.cliff_sensor import check_sensor_checkpoint, transfer_sensor_weights
from learning.multi_agent.scene_normalizer import SceneRunningMeanStd
from learning.multi_agent.stage2_transfer import freeze_stage2_encoder
from utils.edge_scenario_spec import sample_graph, classify_templates
from utils.edge_stage1_spec import semantic_graph_packet
from utils.relation_task_spec import checkpoint_metadata


ROOT = Path(__file__).resolve().parents[1]
ENV = yaml.safe_load((ROOT / 'data/cfg/multi_agent/approach_stage2_cliff_sensor.yaml').read_text())['env']


def network(sensor):
    params = yaml.safe_load((ROOT / 'data/cfg/train/rlg/amp_ma_cliff_sensor.yaml').read_text())['params']['network']
    params['cliff_sensor'] = sensor
    params['transformer']['gta']['diagnostics_first_forward'] = False
    builder = AMPMultiAgentBuilder()
    builder.load(params)
    net = builder.build('cliff_test', actions_num=32, input_shape=(2655 if sensor else 607,),
        amp_input_shape=(1290,), value_size=1, num_agents=2, num_objects=3,
        observation_mode='clean_scene', scene_entity_sizes=[223, 30, 1],
        scene_kinematic_size=7, scene_arena_scale=8.,
        relation_reward_mode=ENV['relationReward']['mode'],
        relation_graph_spec=ENV['relationGraph'], sensor_points=1024 if sensor else 0, device='cpu')
    model = nn.Module()
    model.add_module('a2c_network', net)
    return model


def observation():
    graph = sample_graph(3, ENV['relationGraph'])
    obs = torch.randn(3, 607)
    poses = obs[:, 538:587].reshape(3, 7, 7)
    poses.zero_()
    poses[..., 6] = 1
    obs[:, 587:] = semantic_graph_packet(graph, 3)
    return obs


def test_sensor_feature_mask_removes_sensor_dependence():
    torch.manual_seed(31)
    net = network(True).a2c_network.eval()
    base = observation()
    sensor = torch.randn(len(base), 2048)
    first = torch.cat((base, sensor), -1)
    second = torch.cat((base, sensor + 2), -1)
    state_keys = tuple(net.state_dict())
    with torch.no_grad():
        normal_first = net.eval_actor(first)[0]
        normal_second = net.eval_actor(second)[0]
        net.mask_sensor = True
        masked_first = net.eval_actor(first)[0]
        masked_second = net.eval_actor(second)[0]
        masked_value = net.eval_critic(first)
        masked_value_second = net.eval_critic(second)
    assert not torch.allclose(normal_first, normal_second)
    torch.testing.assert_close(masked_first, masked_second, rtol=0, atol=0)
    torch.testing.assert_close(masked_value, masked_value_second, rtol=0, atol=0)
    assert tuple(net.state_dict()) == state_keys


def test_sensor_world_coordinates_and_drop():
    origins = torch.tensor([[0., 0., 0.], [20., -30., 0.]])
    root = origins[:, None] + torch.tensor([[[1.5, 0., 2.4], [0., 0., 2.4]]])
    half = torch.full((2, 2), 2.)
    grid = sensor_grid(32, 2., 'cpu')
    headings = torch.tensor([[0., torch.pi / 2]]).expand(2, -1)
    points = sensor_positions(root, headings, grid)
    torch.testing.assert_close(points[0, 1, 0], torch.tensor([2., -2.]))
    sensor = height_observation(root, headings, grid, origins, half, 1.5, 3.)
    assert sensor.shape == (2, 2, 1024)
    torch.testing.assert_close(sensor[0], sensor[1], atol=1e-5, rtol=0)
    torch.testing.assert_close(sensor[0, 0].min(), torch.tensor(-.8))
    torch.testing.assert_close(sensor[0, 0].max(), torch.tensor(-.3))


def test_stage_and_ground_falls_keep_contact_and_warmup():
    pos = torch.zeros(5, 2, 3, 3)
    pos[..., 2] = 2.4
    contact = torch.zeros_like(pos)
    pos[0, 0, 0, 2] = 1.6  # fallen on stage
    pos[1, 0, 0, :3] = torch.tensor([3., 0., .1])  # fallen outside
    pos[2, 0, 0, 2] = 1.6  # airborne: no contact
    pos[3, 0, 2, 2] = 1.55  # foot contact is allowed
    pos[4, 0, 0, 2] = 1.6  # reset grace period
    contact[[0, 1, 4], 0, 0, 2] = 10
    contact[3, 0, 2, 2] = 10
    ground = stage_height(pos[..., :2], torch.zeros(5, 3), torch.full((5, 2), 2.), 1.5)
    fallen = terrain_fall(contact, pos, torch.tensor([2]), ground, torch.full((3,), .15),
                          torch.tensor([5, 5, 5, 5, 1]), torch.zeros(5))
    assert fallen[:, 0].tolist() == [True, True, False, False, False]
    assert not fallen[:, 1].any()
    assert fallen.any(-1).tolist() == [True, True, False, False, False]


def test_floor_touch_includes_feet_and_matches_contact_body():
    origins = torch.tensor([[0., 0., 0.], [20., -30., 4.]])
    pos = torch.zeros(2, 6, 3, 3)
    pos[..., 2] = 2.4
    pos[:, :, 2] = torch.tensor([3., 0., .08])
    pos[:, 1, 2] = torch.tensor([0., 0., 1.58])  # stage foot
    pos[:, 3, 2, 2] = 1.0  # object/wall contact above the floor
    pos[:, 4, 0, 0] = 3.  # high contact on a different body
    pos[:, 5, 1] = torch.tensor([3., 0., .08])  # a hand also counts
    pos[:, 5, 2, 2] = 1.0
    pos += origins[:, None, None]
    contact = torch.zeros_like(pos)
    contact[:, [0, 1, 3], 2, 2] = 10
    contact[:, 5, 1, 2] = 10
    contact[:, 4, 0, 2] = 10
    ground = stage_height(pos[..., :2], origins, torch.full((2, 2), 2.), 1.5)
    failed = terrain_fall(contact, pos, torch.tensor([2]), ground, torch.full((3,), .15),
                          torch.full((2,), 5), origins[:, 2])
    assert failed.tolist() == [[True, False, False, False, False, True]] * 2
    warmup = terrain_fall(contact, pos, torch.tensor([2]), ground, torch.full((3,), .15),
                          torch.ones(2), origins[:, 2])
    assert not warmup.any()


def test_random_layout_objects_outside_and_humans_inside():
    torch.manual_seed(11)
    half = 2 + torch.rand(2048, 2) * 2
    boxes = .25 + torch.rand(2048, 3, 3) * .4
    human, obj = sample_layout(half, boxes, .8, [.8, 1.5])
    assert ((half[:, None] - human.abs()).amin(-1) >= .8 - 1e-6).all()
    radius = boxes[..., :2].norm(dim=-1) / 2
    separation = (obj.abs() - half[:, None]).amax(-1) - radius
    assert (separation >= .8 - 1e-6).all()
    assert ((human[:, 0] - human[:, 1]).norm(dim=-1) > 1.).all()


def test_loco_rsi_keeps_three_task_relations_and_amp_distribution():
    graph = sample_graph(500, ENV['relationGraph'])
    kinds = classify_templates(graph, torch.arange(500).repeat_interleave(2),
                               torch.arange(2).repeat(500), True)
    assert set(kinds.tolist()) == {0, 1, 2}
    for weights in ENV['templateRsi'].values():
        assert weights == [1.] + [0.] * 7
    original = yaml.safe_load((ROOT / 'data/cfg/multi_agent/approach_scenario_stage1_self_sum.yaml').read_text())['env']
    assert ENV['relationReward'] == original['relationReward']
    assert ENV['skillDiscProb'] == original['skillDiscProb']


def test_transfer_preserves_actions_values_and_trains_sensor_only_outside_encoder():
    torch.manual_seed(7)
    source, target = network(False), network(True)
    checkpoint = dict(model=source.state_dict(), relation_metadata=checkpoint_metadata(ENV['relationReward']), epoch=14500)
    transfer_sensor_weights(target, checkpoint)
    optimizer = torch.optim.Adam(target.parameters(), lr=1e-3)
    freeze_stage2_encoder(target, optimizer)
    source.eval(); target.eval()
    base = observation()
    obs = torch.cat((base, torch.randn(3, 2048)), -1)
    old, new = source.a2c_network, target.a2c_network
    torch.testing.assert_close(old.eval_actor(base)[0], new.eval_actor(obs)[0], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(old.eval_critic(base), new.eval_critic(obs), atol=1e-6, rtol=1e-5)
    frozen = deepcopy(new.actor_encoder.state_dict())
    for _ in range(2):
        optimizer.zero_grad()
        loss = new.eval_actor(obs)[0].square().mean() + new.eval_critic(obs).square().mean()
        loss.backward()
        optimizer.step()
    for module in (new.actor_sensor, new.critic_sensor, new.action_head):
        assert sum(p.grad.abs().sum() for p in module.parameters() if p.grad is not None) > 0
    for name, parameter in new.actor_encoder.named_parameters():
        assert parameter.grad is None
        torch.testing.assert_close(parameter, frozen[name], atol=0, rtol=0)
    before = new.eval_actor(obs)[0]
    changed = obs.clone(); changed[:, -2048:] = 0
    assert (before - new.eval_actor(changed)[0]).abs().max() > 0
    restored = network(True)
    restored.load_state_dict(target.state_dict(), strict=True)
    restored.eval()
    torch.testing.assert_close(before, restored.a2c_network.eval_actor(obs)[0])


def test_normalization_preserves_pose_graph_sensor_and_checkpoint_contract():
    base = SceneRunningMeanStd([223, 30, 1], [2, 3, 2], [223, 30, 0], 7, 20)
    sensor = SceneRunningMeanStd([223, 30, 1], [2, 3, 2], [223, 30, 0], 7, 2068)
    sensor.load_state_dict(base.state_dict(), strict=True)
    base.eval(); sensor.eval()
    obs = torch.cat((observation(), torch.randn(3, 2048)), -1)
    torch.testing.assert_close(base(obs[:, :607]), sensor(obs)[:, :607])
    torch.testing.assert_close(obs[:, 536:], sensor(obs)[:, 536:])
    task = SimpleNamespace(sensor_metadata={'version': 1, 'cliff': ENV['cliff']})
    check_sensor_checkpoint({'sensor_metadata': task.sensor_metadata}, task)
    with pytest.raises(ValueError, match='Sensor checkpoint'):
        check_sensor_checkpoint({}, task)
    with pytest.raises(ValueError, match='Sensor checkpoint'):
        check_sensor_checkpoint({'sensor_metadata': task.sensor_metadata}, SimpleNamespace())
