import math

import torch
import torch.nn as nn

from learning.multi_agent.amp_network_builder_ma import (
    RelationEncoder,
    _quat_mul,
    _quat_rotate,
    build_pairwise_geometry,
)


def _yaw(angle):
    return torch.tensor([0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)])


def _scene_obs(batch, num_agents, num_objects, entity_sizes):
    num_tokens = 2 * num_agents + num_objects
    node_width = (num_agents * entity_sizes[0]
                  + num_objects * entity_sizes[1]
                  + num_agents * entity_sizes[2])
    obs = torch.randn(batch, node_width + num_tokens * 13)
    obs[:, node_width:].view(batch, num_tokens, 13)[..., 3:7] = \
        torch.tensor([0.0, 0.0, 0.0, 1.0])
    return obs


def _encoder(num_agents, num_objects, score=True, message=True):
    sizes = [11, 7, 1]
    return RelationEncoder(
        sizes, num_agents, num_objects, 16, 2, 2, 32,
        lambda size: nn.Sequential(nn.Linear(size, 16), nn.ReLU()),
        observation_mode="clean_scene",
        kinematic_size=13,
        relation_bias=True,
        geometry_cfg={
            "enable": score or message,
            "use_score": score,
            "use_message": message,
            "input_size": 15,
            "embedding_dim": 16,
        })


def test_scene_forward_and_gradients_for_variable_entity_counts():
    parameter_counts = []
    for num_agents, num_objects in ((1, 1), (2, 2), (3, 4)):
        encoder = _encoder(num_agents, num_objects)
        obs = _scene_obs(4, num_agents, num_objects, [11, 7, 1])
        output = encoder(obs)

        assert output.shape == (4, num_agents, 16)
        assert encoder.forward_calls == 1
        assert encoder.last_shape_flow["geometry"] == (
            4, 2 * num_agents + num_objects, 2 * num_agents + num_objects, 15)

        output.square().mean().backward()
        assert encoder.rel_embed.grad.abs().sum() > 0
        assert sum(p.grad.abs().sum() for p in encoder.geometry_encoder.parameters()
                   if p.grad is not None) > 0
        parameter_counts.append(sum(p.numel() for p in encoder.parameters()))

    assert len(set(parameter_counts)) == 1


def test_geometry_ablation_switches():
    for score, message in ((True, False), (False, True), (False, False)):
        encoder = _encoder(2, 3, score, message)
        output = encoder(_scene_obs(2, 2, 3, [11, 7, 1]))
        assert output.shape == (2, 2, 16)
        assert (encoder.last_shape_flow["geometry_score"] is not None) == score
        assert (encoder.last_shape_flow["geometry_message"] is not None) == message


def test_shared_human_heads_produce_distinct_outputs_and_both_trunks_get_gradients():
    actor = _encoder(2, 3)
    critic = _encoder(2, 3)
    action_head = nn.Linear(16, 5)
    value_head = nn.Linear(16, 1)
    obs = _scene_obs(3, 2, 3, [11, 7, 1])

    actions = action_head(actor(obs))
    values = value_head(critic(obs))
    assert actions.shape == (3, 2, 5)
    assert values.shape == (3, 2, 1)
    assert not torch.allclose(actions[:, 0], actions[:, 1])

    (actions.square().mean() + values.square().mean()).backward()
    assert action_head.weight.grad.abs().sum() > 0
    assert value_head.weight.grad.abs().sum() > 0
    assert actor.rel_embed.grad.abs().sum() > 0
    assert critic.rel_embed.grad.abs().sum() > 0
    assert next(actor.geometry_encoder.parameters()).grad.abs().sum() > 0
    assert next(critic.geometry_encoder.parameters()).grad.abs().sum() > 0


def test_directed_geometry_is_global_se2_invariant():
    kinematics = torch.zeros(1, 3, 13)
    kinematics[0, :, 0:3] = torch.tensor([
        [0.0, 0.0, 0.0], [1.0, 2.0, 0.5], [-2.0, 1.0, 0.25]])
    kinematics[0, :, 3:7] = torch.stack([_yaw(0.2), _yaw(-0.7), _yaw(1.1)])
    kinematics[0, :, 7:10] = torch.tensor([
        [0.5, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.2, 0.3, 0.0]])
    kinematics[0, :, 10:13] = torch.tensor([
        [0.0, 0.0, 0.1], [0.0, 0.0, -0.2], [0.0, 0.0, 0.4]])

    geometry = build_pairwise_geometry(kinematics)
    global_yaw = _yaw(0.83).view(1, 1, 4).expand(1, 3, 4)
    transformed = kinematics.clone()
    transformed[..., 0:3] = (_quat_rotate(global_yaw, kinematics[..., 0:3])
                              + torch.tensor([[[4.0, -3.0, 2.0]]]))
    transformed[..., 3:7] = _quat_mul(global_yaw, kinematics[..., 3:7])
    transformed[..., 7:10] = _quat_rotate(global_yaw, kinematics[..., 7:10])
    transformed[..., 10:13] = _quat_rotate(global_yaw, kinematics[..., 10:13])

    transformed_geometry = build_pairwise_geometry(transformed)
    assert torch.allclose(geometry, transformed_geometry, atol=2e-6)
    assert not torch.allclose(geometry[0, 0, 1], geometry[0, 1, 0])
