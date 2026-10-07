import math
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from learning.multi_agent.amp_network_builder_ma import (
    ENTITY_HUMAN,
    ENTITY_OBJECT,
    ENTITY_TARGET,
    RELATION_BIAS_EDGE_MLP,
    RELATION_BIAS_LOOKUP,
    AMPMultiAgentBuilder,
    RelationEncoder,
    RelationTransformerLayer,
    apply_gta_transform,
    _quat_mul,
    _quat_rotate,
    build_gta_transforms,
    build_entity_type_ids,
    build_relation_matrix,
    build_pairwise_geometry,
    resolve_geometry_position_scale,
)
from learning.multi_agent.scene_normalizer import SceneRunningMeanStd


SCENE_ENTITY_SIZES = [223, 30, 1]
SCENE_NORMALIZED_SIZES = [223, 30, 0]


def _yaw(angle):
    return torch.tensor([0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)])


def _pitch(angle):
    return torch.tensor([0.0, math.sin(angle / 2), 0.0, math.cos(angle / 2)])


def _roll(angle):
    return torch.tensor([math.sin(angle / 2), 0.0, 0.0, math.cos(angle / 2)])


def _scene_obs(batch, num_agents, num_objects, entity_sizes, pose_size=7):
    num_tokens = 2 * num_agents + num_objects
    node_width = (num_agents * entity_sizes[0]
                  + num_objects * entity_sizes[1]
                  + num_agents * entity_sizes[2])
    obs = torch.randn(batch, node_width + num_tokens * pose_size)
    target_offset = num_agents * entity_sizes[0] + num_objects * entity_sizes[1]
    obs[:, target_offset:target_offset + num_agents * entity_sizes[2]] = 1.0
    obs[:, node_width:].view(batch, num_tokens, pose_size)[..., 3:7] = \
        torch.tensor([0.0, 0.0, 0.0, 1.0])
    return obs


def _encoder(num_agents, num_objects, relation_mode=RELATION_BIAS_EDGE_MLP,
             gta=True, old_geometry=False):
    assert not (gta and old_geometry)
    pose_size = 7 if gta else 13
    return RelationEncoder(
        SCENE_ENTITY_SIZES, num_agents, num_objects, 16, 2, 2, 32,
        lambda size: nn.Sequential(nn.Linear(size, 16), nn.ReLU()),
        observation_mode="clean_scene",
        kinematic_size=pose_size,
        relation_bias=True,
        relation_bias_mode=relation_mode,
        geometry_cfg={
            "enable": old_geometry,
            "use_score": old_geometry,
            "use_message": old_geometry,
            "mode": "full15",
            "input_size": 15,
            "embedding_dim": 16,
            "position_scale": [0.2, 0.2, 1.0],
        },
        gta_cfg={
            "enable": gta,
            "translation_scale": 1.0,
            "representation": "se3_direct_sum",
        })


def test_scene_forward_and_gradients_for_variable_entity_counts():
    parameter_counts = []
    for num_agents, num_objects in ((1, 1), (2, 2), (3, 4)):
        encoder = _encoder(num_agents, num_objects)
        obs = _scene_obs(4, num_agents, num_objects, SCENE_ENTITY_SIZES)
        assert obs.shape == (4, 238 * num_agents + 37 * num_objects)
        output = encoder(obs)

        assert output.shape == (4, num_agents, 16)
        assert encoder.forward_calls == 1
        assert encoder.last_shape_flow["geometry"] is None
        assert encoder.last_shape_flow["gta_g"] == (
            4, 2 * num_agents + num_objects, 4, 4)

        output.square().mean().backward()
        assert encoder.edge_encoder.bias_projection.grad.abs().sum() > 0
        assert encoder.layers[0].qkv.weight.grad.abs().sum() > 0
        parameter_counts.append(sum(p.numel() for p in encoder.parameters()))

    assert len(set(parameter_counts)) == 1


def test_production_config_builds_gta_actor_and_critic():
    config_path = Path(__file__).resolve().parents[1] \
        / "data/cfg/train/rlg/amp_ma_carry.yaml"
    with config_path.open() as stream:
        network_config = yaml.safe_load(stream)["params"]["network"]

    num_agents, num_objects, batch = 2, 3, 3
    obs_width = 238 * num_agents + 37 * num_objects
    builder = AMPMultiAgentBuilder()
    builder.load(network_config)
    network = builder.build(
        "gta_config_test",
        actions_num=28,
        input_shape=(obs_width,),
        amp_input_shape=(64,),
        value_size=1,
        num_agents=num_agents,
        num_objects=num_objects,
        humanoid_obs_size=230,
        object_obs_size=39,
        goal_obs_size=6,
        observation_mode="clean_scene",
        scene_entity_sizes=SCENE_ENTITY_SIZES,
        scene_kinematic_size=7,
        scene_arena_scale=5.0,
        device="cpu")

    obs = _scene_obs(batch, num_agents, num_objects, SCENE_ENTITY_SIZES)
    mu, sigma = network.eval_actor(obs)
    value = network.eval_critic(obs)
    assert mu.shape == (batch * num_agents, 28)
    assert sigma.shape == (batch * num_agents, 28)
    assert value.shape == (batch * num_agents, 1)
    assert network.actor_encoder.gta_enabled
    assert network.critic_encoder.gta_enabled
    assert not network.actor_encoder.geometry_enabled
    assert network.actor_encoder.last_shape_flow["gta_g"] == (batch, 7, 4, 4)


def test_gta_and_old_geometry_are_mutually_exclusive():
    try:
        RelationEncoder(
            SCENE_ENTITY_SIZES, 2, 3, 16, 2, 2, 32,
            lambda size: nn.Sequential(nn.Linear(size, 16), nn.ReLU()),
            observation_mode="clean_scene", kinematic_size=7,
            relation_bias_mode=RELATION_BIAS_EDGE_MLP,
            geometry_cfg={"enable": True}, gta_cfg={"enable": True})
    except AssertionError as exc:
        assert "cannot be enabled together" in str(exc)
    else:
        raise AssertionError("GTA and old geometry must be mutually exclusive")


def test_geometry_modes_mask_components_without_changing_architecture():
    kinematics = torch.zeros(1, 2, 13)
    kinematics[..., 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    kinematics[0, 1, 0:3] = torch.tensor([1.0, 2.0, 3.0])
    kinematics[0, 1, 7:10] = torch.tensor([0.5, 0.25, -0.5])
    physical_types = torch.tensor([ENTITY_HUMAN, ENTITY_OBJECT])
    geo3 = build_pairwise_geometry(
        kinematics, entity_types=physical_types, mode="position3")
    geo9 = build_pairwise_geometry(
        kinematics, entity_types=physical_types, mode="pose9")
    geo15 = build_pairwise_geometry(
        kinematics, entity_types=physical_types, mode="full15")
    assert torch.count_nonzero(geo3[..., 3:]) == 0
    assert torch.count_nonzero(geo9[..., 9:]) == 0
    assert torch.count_nonzero(geo15[..., 9:]) > 0


def test_geometry_fixed_scales_and_pairwise_velocity_definition():
    assert resolve_geometry_position_scale(None, 5.0) == [0.2, 0.2, 1.0]
    assert resolve_geometry_position_scale(0.5, 5.0) == 0.5

    kinematics = torch.zeros(1, 2, 13)
    kinematics[..., 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    kinematics[0, 0, 3:7] = _yaw(math.pi / 2)
    kinematics[0, 0, 7:10] = torch.tensor([1.0, 0.0, 0.0])
    kinematics[0, 1, 7:10] = torch.tensor([1.0, 4.0, 0.0])
    kinematics[0, 0, 10:13] = torch.tensor([0.0, 0.0, 1.0])
    kinematics[0, 1, 10:13] = torch.tensor([0.0, 0.0, 5.0])
    geometry = build_pairwise_geometry(
        kinematics,
        scales=([0.2, 0.2, 1.0], 0.25, 0.25),
        entity_types=torch.tensor([ENTITY_HUMAN, ENTITY_OBJECT]))

    # World +Y is source-forward after a +90 degree heading; both differences are 4.
    assert torch.allclose(geometry[0, 0, 1, 9:12], torch.tensor([1.0, 0.0, 0.0]),
                          atol=2e-6)
    assert torch.allclose(geometry[0, 0, 1, 12:15], torch.tensor([0.0, 0.0, 1.0]),
                          atol=2e-6)


def test_legacy_multirow_layout_is_unchanged():
    entity_sizes = [230, 39, 6]
    num_agents, num_objects = 2, 3
    encoder = RelationEncoder(
        entity_sizes, num_agents, num_objects, 16, 2, 2, 32,
        lambda size: nn.Sequential(nn.Linear(size, 16), nn.ReLU()),
        observation_mode="legacy_multirow",
        relation_bias=True,
        relation_bias_mode=RELATION_BIAS_LOOKUP,
        geometry_cfg={"enable": True})
    width = num_agents * 230 + num_objects * 39 + num_agents * 6
    output = encoder(torch.randn(4, width))

    assert output.shape == (4, 16)
    assert encoder.last_shape_flow["geometry"] is None


def test_shared_human_heads_produce_distinct_outputs_and_both_trunks_get_gradients():
    actor = _encoder(2, 3)
    critic = _encoder(2, 3)
    action_head = nn.Linear(16, 5)
    value_head = nn.Linear(16, 1)
    obs = _scene_obs(3, 2, 3, SCENE_ENTITY_SIZES)

    actions = action_head(actor(obs))
    values = value_head(critic(obs))
    assert actions.shape == (3, 2, 5)
    assert values.shape == (3, 2, 1)
    assert not torch.allclose(actions[:, 0], actions[:, 1])

    (actions.square().mean() + values.square().mean()).backward()
    assert action_head.weight.grad.abs().sum() > 0
    assert value_head.weight.grad.abs().sum() > 0
    assert actor.edge_encoder.bias_projection.grad.abs().sum() > 0
    assert critic.edge_encoder.bias_projection.grad.abs().sum() > 0
    assert actor.layers[0].qkv.weight.grad.abs().sum() > 0
    assert critic.layers[0].qkv.weight.grad.abs().sum() > 0


def test_a2_semantic_edges_are_embedded_and_separate_from_geometry():
    encoder = _encoder(2, 3)
    edges = encoder.build_edge_embeddings()
    bias = encoder.build_relation_bias()
    assert edges.shape == (7, 7, 64)
    assert bias.shape == (2, 2, 7, 7)
    assert torch.count_nonzero(bias) == 0  # zero-projected plain-attention start

    # Same relation ID can produce different directed A2 embeddings through H/O/T types.
    relation = encoder.rel_matrix
    none_pairs = torch.nonzero(relation == 0)
    pair_a = tuple(none_pairs[0].tolist())
    pair_b = next(tuple(pair.tolist()) for pair in none_pairs[1:]
                  if encoder.entity_types[pair[0]] != encoder.entity_types[pair_a[0]]
                  or encoder.entity_types[pair[1]] != encoder.entity_types[pair_a[1]])
    assert not torch.allclose(edges[pair_a], edges[pair_b])

    # The zero projection intentionally gates the embeddings on step one. Once it has
    # moved, gradients reach the A2 embeddings/MLP while GTA stays algebraic.
    obs = _scene_obs(4, 2, 3, SCENE_ENTITY_SIZES)
    encoder(obs).square().mean().backward()
    projection_grad = encoder.edge_encoder.bias_projection.grad
    assert projection_grad is not None and projection_grad.abs().sum() > 0
    with torch.no_grad():
        encoder.edge_encoder.bias_projection.add_(projection_grad)
    encoder.zero_grad(set_to_none=True)
    encoder(obs).square().mean().backward()
    for parameter in (
            encoder.edge_encoder.source_type_embed.weight,
            encoder.edge_encoder.relation_embed.weight,
            encoder.edge_encoder.target_type_embed.weight,
            encoder.edge_encoder.edge_mlp[0].weight,
            encoder.edge_encoder.edge_mlp[2].weight):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    assert encoder.layers[0].qkv.weight.grad.abs().sum() > 0


def test_clean_a1_lookup_remains_available_for_ablation():
    encoder = _encoder(2, 3, relation_mode=RELATION_BIAS_LOOKUP)
    assert hasattr(encoder, "rel_embed")
    assert not hasattr(encoder, "edge_encoder")
    assert encoder.build_relation_bias().shape == (2, 2, 7, 7)
    output = encoder(_scene_obs(2, 2, 3, SCENE_ENTITY_SIZES))
    output.square().mean().backward()
    assert encoder.rel_embed.grad.abs().sum() > 0


def test_a2_checkpoint_and_parameters_are_entity_count_independent():
    source = _encoder(1, 1)
    target = _encoder(3, 4)
    assert sum(p.numel() for p in source.parameters()) == \
        sum(p.numel() for p in target.parameters())
    target.load_state_dict(source.state_dict(), strict=True)

    source.set_entity_counts(3, 4)
    output = source(_scene_obs(2, 3, 4, SCENE_ENTITY_SIZES))
    assert output.shape == (2, 3, 16)
    assert source.build_edge_embeddings().shape == (10, 10, 64)
    assert source.build_relation_bias().shape == (2, 2, 10, 10)


def test_scene_normalizer_only_changes_local_prefixes():
    num_agents, num_objects = 2, 3
    counts = [num_agents, num_objects, num_agents]
    obs = _scene_obs(4, num_agents, num_objects, SCENE_ENTITY_SIZES)
    normalizer = SceneRunningMeanStd(
        SCENE_ENTITY_SIZES, counts,
        normalized_sizes=SCENE_NORMALIZED_SIZES,
        kinematic_size=7)
    assert [tuple(rms.running_mean.shape) for rms in normalizer.running_mean_std] == [
        (223,), (30,)]
    normalized = normalizer(obs)

    offset = 0
    for size, count, normalized_size in zip(
            SCENE_ENTITY_SIZES, counts, SCENE_NORMALIZED_SIZES):
        width = size * count
        before = obs[:, offset:offset + width].view(4, count, size)
        after = normalized[:, offset:offset + width].view(4, count, size)
        assert torch.equal(after[..., normalized_size:], before[..., normalized_size:])
        offset += width

    # The raw 7-D pose records are copied byte-for-byte for GTA construction.
    assert torch.equal(normalized[:, offset:], obs[:, offset:])


def test_gta_transform_convention_and_analytic_inverse():
    poses = torch.zeros(1, 2, 7)
    poses[..., 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    poses[0, 1, 0:3] = torch.tensor([2.0, 0.0, 0.0])

    g, ginv = build_gta_transforms(poses)
    identity = torch.eye(4).view(1, 1, 4, 4)
    assert torch.allclose(torch.matmul(g, ginv), identity.expand_as(g), atol=1e-6)
    relative = torch.matmul(g[:, 0], ginv[:, 1])
    assert torch.allclose(relative[0, 0:3, 3], torch.tensor([2.0, 0.0, 0.0]))

    poses[0, 0, 3:7] = _yaw(math.pi / 2)
    g, ginv = build_gta_transforms(poses)
    relative = torch.matmul(g[:, 0], ginv[:, 1])
    assert torch.allclose(relative[0, 0:3, 3], torch.tensor([0.0, -2.0, 0.0]),
                          atol=2e-6)


def test_target_owner_heading_recovers_heading_local_goal_vector():
    heading = _yaw(math.pi / 2)
    poses = torch.zeros(1, 2, 7)
    poses[0, 0, 0:3] = torch.tensor([1.0, 2.0, 0.8])
    poses[0, 1, 0:3] = torch.tensor([1.0, 4.0, 0.8])
    poses[0, :, 3:7] = heading

    g, ginv = build_gta_transforms(poses)
    human_from_target = torch.matmul(g[:, 0], ginv[:, 1])
    assert torch.allclose(human_from_target[0, 0:3, 0:3], torch.eye(3), atol=1e-6)
    assert torch.allclose(human_from_target[0, 0:3, 3],
                          torch.tensor([2.0, 0.0, 0.0]), atol=2e-6)


def test_gta_direct_sum_factorization_matches_slow_pairwise_reference():
    torch.manual_seed(7)
    batch, heads, tokens, head_dim = 2, 2, 3, 8
    poses = torch.randn(batch, tokens, 7)
    poses[..., 3:7] = torch.randn(batch, tokens, 4)
    g, ginv = build_gta_transforms(poses, translation_scale=0.4)

    q = torch.randn(batch, heads, tokens, head_dim)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    bias = torch.randn(heads, tokens, tokens)

    q_shared = apply_gta_transform(g.transpose(-1, -2), q)
    k_shared = apply_gta_transform(ginv, k)
    v_shared = apply_gta_transform(ginv, v)
    logits = torch.matmul(q_shared, k_shared.transpose(-2, -1)) / math.sqrt(head_dim)
    attention = torch.softmax(logits + bias.unsqueeze(0), dim=-1)
    factorized = apply_gta_transform(g, torch.matmul(attention, v_shared))

    relative = torch.matmul(g.unsqueeze(2), ginv.unsqueeze(1))
    num_blocks = head_dim // 4
    q_blocks = q.reshape(batch, heads, tokens, num_blocks, 4)
    k_blocks = k.reshape(batch, heads, tokens, num_blocks, 4)
    v_blocks = v.reshape(batch, heads, tokens, num_blocks, 4)
    pair_k = torch.einsum("bijmn,bhjrn->bhijrm", relative, k_blocks)
    pair_v = torch.einsum("bijmn,bhjrn->bhijrm", relative, v_blocks)
    slow_logits = torch.einsum("bhirm,bhijrm->bhij", q_blocks, pair_k) \
        / math.sqrt(head_dim)
    slow_attention = torch.softmax(slow_logits + bias.unsqueeze(0), dim=-1)
    slow = torch.einsum("bhij,bhijrm->bhirm", slow_attention, pair_v).reshape_as(q)

    assert torch.allclose(factorized, slow, atol=2e-5, rtol=2e-5)


def test_identity_gta_equals_semantic_only_attention():
    torch.manual_seed(11)
    layer = RelationTransformerLayer(16, 2, 32)
    x = torch.randn(3, 5, 16)
    relation_bias = torch.randn(2, 5, 5)
    identity = torch.eye(4).view(1, 1, 4, 4).expand(3, 5, 4, 4)

    semantic_only = layer(x, rel_bias=relation_bias)
    gta = layer(x, rel_bias=relation_bias, gta_g=identity, gta_ginv=identity)
    assert torch.allclose(gta, semantic_only, atol=1e-6)


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


def test_object_source_position_uses_yaw_but_rotation_uses_full_orientation():
    kinematics = torch.zeros(1, 2, 13)
    kinematics[..., 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    kinematics[0, 1, 0:3] = torch.tensor([2.0, -1.0, 0.75])
    entity_types = torch.tensor([ENTITY_OBJECT, ENTITY_HUMAN])

    yaw_only = kinematics.clone()
    yaw_only[0, 0, 3:7] = _yaw(0.7)
    tilted = kinematics.clone()
    tilted[0, 0, 3:7] = _quat_mul(_yaw(0.7), _pitch(0.9))

    yaw_geo = build_pairwise_geometry(yaw_only, entity_types=entity_types)
    tilted_geo = build_pairwise_geometry(tilted, entity_types=entity_types)
    assert torch.allclose(yaw_geo[0, 0, 1, 0:3], tilted_geo[0, 0, 1, 0:3], atol=2e-6)
    assert not torch.allclose(yaw_geo[0, 0, 1, 3:9], tilted_geo[0, 0, 1, 3:9])


def test_relative_rotation_uses_full_human_orientation():
    kinematics = torch.zeros(1, 2, 13)
    kinematics[..., 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    kinematics[0, 1, 0:3] = torch.tensor([1.0, 0.5, 0.25])
    entity_types = torch.tensor([ENTITY_HUMAN, ENTITY_HUMAN])

    level = kinematics.clone()
    level[0, 0, 3:7] = _yaw(-0.4)
    rolled = kinematics.clone()
    rolled[0, 0, 3:7] = _quat_mul(_yaw(-0.4), _roll(0.8))
    level_geo = build_pairwise_geometry(level, entity_types=entity_types)
    rolled_geo = build_pairwise_geometry(rolled, entity_types=entity_types)

    assert torch.allclose(level_geo[0, 0, 1, 0:3], rolled_geo[0, 0, 1, 0:3], atol=2e-6)
    assert not torch.allclose(level_geo[0, 0, 1, 3:9], rolled_geo[0, 0, 1, 3:9])


def test_target_pairs_are_position_only_and_target_sources_use_scene_axes():
    kinematics = torch.zeros(1, 4, 13)
    kinematics[..., 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    kinematics[0, :, 0:3] = torch.tensor([
        [0.0, 0.0, 0.9], [2.0, -1.0, 0.4], [1.0, 3.0, 0.2], [-2.0, 4.0, 0.7]])
    kinematics[0, 0, 3:7] = _quat_mul(_yaw(0.5), _roll(0.3))
    kinematics[0, 1, 3:7] = _quat_mul(_yaw(-0.8), _pitch(0.6))
    kinematics[0, 0:2, 7:13] = 1.0
    entity_types = torch.tensor([
        ENTITY_HUMAN, ENTITY_OBJECT, ENTITY_TARGET, ENTITY_TARGET])

    geometry = build_pairwise_geometry(
        kinematics,
        scales=([0.2, 0.2, 1.0], 0.25, 0.25),
        entity_types=entity_types)
    target = entity_types == ENTITY_TARGET
    target_pair = target.view(4, 1) | target.view(1, 4)
    assert torch.count_nonzero(geometry[0][target_pair][..., 3:]) == 0

    expected_t0_to_t1 = (kinematics[0, 3, 0:3] - kinematics[0, 2, 0:3]) \
        * torch.tensor([0.2, 0.2, 1.0])
    assert torch.allclose(geometry[0, 2, 3, 0:3], expected_t0_to_t1)
    assert torch.count_nonzero(torch.diagonal(geometry[0], dim1=0, dim2=1)) == 0


def test_relation_none_still_has_physical_geometry():
    num_agents, num_objects = 1, 2
    relation = build_relation_matrix(num_agents, num_objects)
    types = build_entity_type_ids(num_agents, num_objects)
    # H0 -> O1 is an unassigned semantic relation but remains physically located.
    assert relation[0, 2].item() == 0
    kinematics = torch.zeros(1, 4, 13)
    kinematics[..., 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    kinematics[0, 2, 0] = 2.0
    geometry = build_pairwise_geometry(kinematics, entity_types=types)
    assert torch.equal(geometry[0, 0, 2, 0:3], torch.tensor([2.0, 0.0, 0.0]))
