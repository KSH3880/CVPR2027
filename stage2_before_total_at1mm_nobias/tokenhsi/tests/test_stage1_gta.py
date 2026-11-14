import math
import torch
from learning.multi_agent.amp_network_builder_ma import (
    RelationTransformerLayer,
    build_gta_transforms,
    apply_gta_transform,
)
from learning.multi_agent.scene_normalizer import SceneRunningMeanStd


def test_scene_normalizer_only_changes_local_prefixes():
    num_agents, num_objects = (2, 3)
    counts = [num_agents, num_objects, num_agents]
    obs = _scene_obs(4, num_agents, num_objects, SCENE_ENTITY_SIZES)
    normalizer = SceneRunningMeanStd(
        SCENE_ENTITY_SIZES,
        counts,
        normalized_sizes=SCENE_NORMALIZED_SIZES,
        kinematic_size=7,
    )
    assert [tuple(rms.running_mean.shape) for rms in normalizer.running_mean_std] == [
        (223,),
        (30,),
    ]
    normalized = normalizer(obs)
    offset = 0
    for size, count, normalized_size in zip(
        SCENE_ENTITY_SIZES, counts, SCENE_NORMALIZED_SIZES
    ):
        width = size * count
        before = obs[:, offset : offset + width].view(4, count, size)
        after = normalized[:, offset : offset + width].view(4, count, size)
        assert torch.equal(after[..., normalized_size:], before[..., normalized_size:])
        offset += width
    assert torch.equal(normalized[:, offset:], obs[:, offset:])


def test_gta_transform_convention_and_analytic_inverse():
    poses = torch.zeros(1, 2, 7)
    poses[..., 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    poses[0, 1, 0:3] = torch.tensor([2.0, 0.0, 0.0])
    g, ginv = build_gta_transforms(poses)
    identity = torch.eye(4).view(1, 1, 4, 4)
    assert torch.allclose(torch.matmul(g, ginv), identity.expand_as(g), atol=1e-06)
    relative = torch.matmul(g[:, 0], ginv[:, 1])
    assert torch.allclose(relative[0, 0:3, 3], torch.tensor([2.0, 0.0, 0.0]))
    poses[0, 0, 3:7] = _yaw(math.pi / 2)
    g, ginv = build_gta_transforms(poses)
    relative = torch.matmul(g[:, 0], ginv[:, 1])
    assert torch.allclose(
        relative[0, 0:3, 3], torch.tensor([0.0, -2.0, 0.0]), atol=2e-06
    )


def test_gta_direct_sum_factorization_matches_slow_pairwise_reference():
    torch.manual_seed(7)
    batch, heads, tokens, head_dim = (2, 2, 3, 8)
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
    slow_logits = torch.einsum("bhirm,bhijrm->bhij", q_blocks, pair_k) / math.sqrt(
        head_dim
    )
    slow_attention = torch.softmax(slow_logits + bias.unsqueeze(0), dim=-1)
    slow = torch.einsum("bhij,bhijrm->bhirm", slow_attention, pair_v).reshape_as(q)
    assert torch.allclose(factorized, slow, atol=2e-05, rtol=2e-05)


def test_identity_gta_equals_semantic_only_attention():
    torch.manual_seed(11)
    layer = RelationTransformerLayer(16, 2, 32)
    x = torch.randn(3, 5, 16)
    relation_bias = torch.randn(2, 5, 5)
    identity = torch.eye(4).view(1, 1, 4, 4).expand(3, 5, 4, 4)
    semantic_only = layer(x, rel_bias=relation_bias)
    gta = layer(x, rel_bias=relation_bias, gta_g=identity, gta_ginv=identity)
    assert torch.allclose(gta, semantic_only, atol=1e-06)


SCENE_ENTITY_SIZES = [223, 30, 1]
SCENE_NORMALIZED_SIZES = [223, 30, 0]


def _yaw(angle):
    return torch.tensor([0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)])


def _scene_obs(batch, num_agents, num_objects, entity_sizes, pose_size=7):
    num_tokens = 2 * num_agents + num_objects
    node_width = (
        num_agents * entity_sizes[0]
        + num_objects * entity_sizes[1]
        + num_agents * entity_sizes[2]
    )
    obs = torch.randn(batch, node_width + num_tokens * pose_size)
    target_offset = num_agents * entity_sizes[0] + num_objects * entity_sizes[1]
    obs[:, target_offset : target_offset + num_agents * entity_sizes[2]] = 1.0
    obs[:, node_width:].view(batch, num_tokens, pose_size)[..., 3:7] = torch.tensor(
        [0.0, 0.0, 0.0, 1.0]
    )
    return obs
