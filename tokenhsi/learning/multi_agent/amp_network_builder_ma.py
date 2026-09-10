# Multi-agent relation/geometry-transformer policy.
#
# legacy_multirow keeps the original obs row (one per env/agent, ego-first):
#       [ humanoid_0..M-1 | object_0..M-1 | goal_0..M-1 ]
#            |                  |               |
#           T_h                T_o             T_g          <- one tokenizer per entity TYPE
#            +E_humanoid        +E_object       +E_goal      <- learnable type embeddings
#            \__________________|_______________/
#                               |
#                  (2M + O) tokens, 64-d each
#                               |
#              Transformer with a static task-relation matrix R
#              turned into a learnable per-head additive attention bias
#                               |
# clean_scene instead builds one intrinsic entity set and one 7-D pose per token. GTA
# aligns Q/K/V with token-wise SE(3) transforms without pairwise K/V materialization,
# updates all tokens, and gathers all M Human outputs together. It uses the A2 typed
# semantic edge encoder; A1 remains available and is forced for the exact legacy baseline.
#                               |
#                 shared action head -> mu     (actor encoder)
#                 shared value head  -> V^i    (critic encoder, separate weights)
#
# There is no self/other distinction and no agent-ID embedding: the relation matrix
# already carries "who owns what", and the ego is defined by the readout position, so
# the network stays permutation-equivariant and independent of the agent count. The only
# parameters that see a new value when M goes 1 -> 2 are the relation-bias entries for
# the cross-agent relation types, which are zero-initialised (= plain attention).
#
# Actor and critic use two separate encoders (matching `separate: True`), so the policy
# trunk never feeds the value function. Because a row already contains every agent's
# state, the critic is a centralized per-agent critic V^i = V(s^1..M, o^1..M, g^1..M; ego=i).
#
# Every parameter is independent of M and O -- the entity counts only change the number
# of tokens and the size of the (non-learnable) relation matrix. A checkpoint trained
# with M agents / M objects therefore loads into a network built for M' agents / O'
# objects without any weight surgery.

import math

import torch
import torch.nn as nn

from learning.amp_network_builder import AMPBuilder

# relation types between two tokens (non-learnable, derived from the assignment)
REL_NONE = 0          # nothing ties the two entities together
REL_SELF = 1          # a token with itself
REL_TEAMMATE = 2      # humanoid <-> humanoid
REL_OWN_OBJECT = 3    # humanoid <-> the box it owns
REL_OWN_GOAL = 4      # humanoid <-> the target of the box it owns
REL_OBJECT_GOAL = 5   # box <-> its own target
NUM_REL_TYPES = 6

ENTITY_HUMAN = 0
ENTITY_OBJECT = 1
ENTITY_TARGET = 2
NUM_ENTITY_TYPES = 3

GEOMETRY_MODES = ("position3", "pose9", "full15")
GTA_REPRESENTATION_SE3_DIRECT_SUM = "se3_direct_sum"

RELATION_BIAS_LOOKUP = "lookup"
RELATION_BIAS_EDGE_MLP = "edge_mlp"
RELATION_BIAS_MODES = (RELATION_BIAS_LOOKUP, RELATION_BIAS_EDGE_MLP)


def build_entity_type_ids(num_agents, num_objects, device=None):
    """Entity kinds for the canonical [M humans | O objects | M targets] order."""
    return torch.cat([
        torch.full((num_agents,), ENTITY_HUMAN, dtype=torch.long, device=device),
        torch.full((num_objects,), ENTITY_OBJECT, dtype=torch.long, device=device),
        torch.full((num_agents,), ENTITY_TARGET, dtype=torch.long, device=device),
    ])


def resolve_geometry_position_scale(configured_scale, arena_scale):
    """Resolve automatic XY scaling while keeping env-local Z in metres."""
    if configured_scale is not None:
        return configured_scale
    assert arena_scale > 0.0
    xy_scale = 1.0 / float(arena_scale)
    return [xy_scale, xy_scale, 1.0]


def build_relation_matrix(num_agents, num_objects=None):
    """Build R for [M humanoids | O objects | M goals].

    Object slots [0, M) are the assigned objects, in owner order.  Any remaining
    object slots are unassigned distractors and therefore only have REL_SELF on their
    diagonal; every cross-entity edge involving them is REL_NONE.
    """
    M = num_agents
    O = M if num_objects is None else num_objects
    assert M >= 1
    assert O >= M, "num_objects must be at least num_agents"

    L = 2 * M + O
    R = torch.full((L, L), REL_NONE, dtype=torch.long)

    def humanoid_idx(k):
        return k

    def object_idx(k):
        return M + k

    def goal_idx(k):
        return M + O + k

    for i in range(L):
        R[i, i] = REL_SELF

    for a in range(M):
        for b in range(M):
            if a == b:
                R[humanoid_idx(a), object_idx(b)] = REL_OWN_OBJECT
                R[object_idx(b), humanoid_idx(a)] = REL_OWN_OBJECT
                R[humanoid_idx(a), goal_idx(b)] = REL_OWN_GOAL
                R[goal_idx(b), humanoid_idx(a)] = REL_OWN_GOAL
                R[object_idx(a), goal_idx(b)] = REL_OBJECT_GOAL
                R[goal_idx(b), object_idx(a)] = REL_OBJECT_GOAL
            else:
                R[humanoid_idx(a), humanoid_idx(b)] = REL_TEAMMATE

    return R


from utils.relation_task_spec import (
    LEGACY_MODE, STATE_MODE, compile_carry_subgoal, build_state_relation_matrix)


class EdgeEncoder(nn.Module):
    """A2 semantic edge encoder: typed relations -> layer/head scalar biases.

    This branch is deliberately independent of continuous geometry. Its input is only
    (source entity type, A1 relation type, target entity type), so the resulting
    (L,L,edge_dim) tensor has no batch dimension and all pairs share the same MLP.
    """

    def __init__(self, num_layers, num_heads, source_type_dim=16, relation_dim=32,
                 target_type_dim=16, edge_dim=64, activation=nn.ReLU, num_relation_types=NUM_REL_TYPES):
        super().__init__()
        input_dim = source_type_dim + relation_dim + target_type_dim
        self.source_type_embed = nn.Embedding(NUM_ENTITY_TYPES, source_type_dim)
        self.relation_embed = nn.Embedding(num_relation_types, relation_dim)
        self.target_type_embed = nn.Embedding(NUM_ENTITY_TYPES, target_type_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_dim, edge_dim),
            activation(),
            nn.Linear(edge_dim, edge_dim),
        )

        # Zero projection makes the initial semantic bias exactly zero. On the first
        # optimizer step only this projection learns; gradients then reach embeddings
        # and the edge MLP from the following step onward.
        self.bias_projection = nn.Parameter(torch.zeros(num_layers, num_heads, edge_dim))
        for embed in (self.source_type_embed, self.relation_embed, self.target_type_embed):
            nn.init.trunc_normal_(embed.weight, std=0.02)

    def build_edge_embeddings(self, entity_types, relation_matrix):
        L = entity_types.shape[0]
        assert relation_matrix.shape == (L, L)
        source = self.source_type_embed(entity_types).unsqueeze(1).expand(-1, L, -1)
        relation = self.relation_embed(relation_matrix)
        target = self.target_type_embed(entity_types).unsqueeze(0).expand(L, -1, -1)
        return self.edge_mlp(torch.cat([source, relation, target], dim=-1))

    def project_bias(self, edge_embeddings):
        """(L,L,D) -> (layers,heads,L,L)."""
        return torch.einsum("ijd,lhd->lhij", edge_embeddings, self.bias_projection)

    def forward(self, entity_types, relation_matrix):
        return self.project_bias(self.build_edge_embeddings(entity_types, relation_matrix))


class RelationTransformerLayer(nn.Module):
    """Post-LN attention with semantic bias and optional GTA or historical Geo.

    Written by hand instead of nn.TransformerEncoderLayer so the bias can broadcast over
    the batch: R is identical for every env, so the bias is (num_heads, L, L) and costs
    nothing, whereas nn.MultiheadAttention would require a materialised (B*H, L, L) mask.
    """

    def __init__(self, d_model, num_heads, dim_feedforward, activation=nn.ReLU):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.act = activation()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, rel_bias=None, geo_score=None, geo_message=None,
                gta_g=None, gta_ginv=None, diagnostics_label=None, collect_diagnostics=False):
        # x: (B,L,d), rel_bias: (H,L,L), geo_score: (B,H,L,L),
        # geo_message: (B,H,L,L,head_dim), gta matrices: (B,L,4,4)
        B, L, d = x.shape

        qkv = self.qkv(x).view(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                                    # (B, H, L, hd)

        gta_active = gta_g is not None or gta_ginv is not None
        if gta_active:
            assert gta_g is not None and gta_ginv is not None
            assert geo_score is None and geo_message is None, \
                "GTA and pairwise Geo must not execute in the same attention layer"
            q = apply_gta_transform(gta_g.transpose(-1, -2), q)
            k = apply_gta_transform(gta_ginv, k)
            v = apply_gta_transform(gta_ginv, v)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale            # (B, H, L, L)
        if collect_diagnostics:
            with torch.no_grad():
                sampled = attn[:32].detach().float()
                qk_rms = (sampled - sampled.mean(-1, keepdim=True)).square().mean().sqrt()
        if rel_bias is not None:
            if rel_bias.ndim == 3:
                assert rel_bias.shape == (self.num_heads, L, L)
                attn = attn + rel_bias.unsqueeze(0)
            else:
                assert rel_bias.shape == (B, self.num_heads, L, L)
                attn = attn + rel_bias
        if geo_score is not None:
            attn = attn + geo_score
        if diagnostics_label is not None:
            with torch.no_grad():
                flat_logits = attn.detach().float().reshape(-1)
                q_norm = torch.linalg.vector_norm(q.detach().float(), dim=-1).mean()
                k_norm = torch.linalg.vector_norm(k.detach().float(), dim=-1).mean()
                v_norm = torch.linalg.vector_norm(v.detach().float(), dim=-1).mean()
                p95 = torch.quantile(flat_logits, 0.95)
                p99 = torch.quantile(flat_logits, 0.99)
        attn = torch.softmax(attn, dim=-1)
        if collect_diagnostics:
            with torch.no_grad():
                p = attn[:32].detach().float()
                self.last_diagnostics = {'qk_row_rms': qk_rms,
                    'entropy': -(p * p.clamp_min(1e-12).log()).sum(-1).mean()}
        if diagnostics_label is not None:
            with torch.no_grad():
                probs = attn.detach().float()
                entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean()
                print("[MA][GTA diagnostics] {} q/k/v_norm={:.4f}/{:.4f}/{:.4f} "
                      "logit_p95/p99={:.4f}/{:.4f} attention_entropy={:.4f}".format(
                          diagnostics_label, q_norm.item(), k_norm.item(), v_norm.item(),
                          p95.item(), p99.item(), entropy.item()))

        if geo_message is None:
            out = torch.matmul(attn, v)                                     # (B,H,L,hd)
        else:
            # sum_j a_ij(V_j + g_ij), written without materialising a second
            # (B,H,L,L,hd) V broadcast.
            out = torch.matmul(attn, v)
            out = out + torch.sum(attn.unsqueeze(-1) * geo_message, dim=3)   # (B,H,L,hd)
        if gta_active:
            out = apply_gta_transform(gta_g, out)
        out = out.transpose(1, 2).reshape(B, L, d)

        x = self.norm1(x + self.proj(out))
        x = self.norm2(x + self.linear2(self.act(self.linear1(x))))
        return x


def _quat_conjugate(q):
    out = q.clone()
    out[..., :3] = -out[..., :3]
    return out


def _quat_mul(q, r):
    """Quaternion product for xyzw tensors with arbitrary leading dimensions."""
    qx, qy, qz, qw = q.unbind(-1)
    rx, ry, rz, rw = r.unbind(-1)
    return torch.stack([
        qw * rx + qx * rw + qy * rz - qz * ry,
        qw * ry - qx * rz + qy * rw + qz * rx,
        qw * rz + qx * ry - qy * rx + qz * rw,
        qw * rw - qx * rx - qy * ry - qz * rz,
    ], dim=-1)


def _quat_rotate(q, v):
    """Rotate vectors by xyzw quaternions without an IsaacGym dependency."""
    q_xyz = q[..., :3]
    uv = torch.cross(q_xyz, v, dim=-1)
    uuv = torch.cross(q_xyz, uv, dim=-1)
    return v + 2.0 * (q[..., 3:4] * uv + uuv)


def _normalize_quaternion(q, eps=1e-8):
    """Normalize xyzw quaternions without modifying the stored simulator state."""
    norm = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    return q / norm.clamp_min(eps)


def _quat_to_rotation_matrix(q):
    """Convert normalized xyzw quaternions to local-to-env rotation matrices."""
    q = _normalize_quaternion(q)
    x, y, z, w = q.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return torch.stack([
        1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy),
        2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx),
        2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy),
    ], dim=-1).reshape(q.shape[:-1] + (3, 3))


def build_gta_transforms(entity_poses, translation_scale=1.0):
    """Construct token-wise ``g=T^-1`` and ``g^-1=T`` from 7-D poses.

    ``entity_poses`` stores env-local position and a local-to-env xyzw quaternion.
    Translation is scaled uniformly before it enters the homogeneous SE(3) matrix.
    """
    assert entity_poses.ndim == 3 and entity_poses.shape[-1] == 7
    translation_scale = float(translation_scale)
    assert translation_scale > 0.0

    p = entity_poses[..., 0:3] * translation_scale
    rot = _quat_to_rotation_matrix(entity_poses[..., 3:7])
    rot_t = rot.transpose(-1, -2)

    shape = entity_poses.shape[:-1] + (4, 4)
    ginv = torch.zeros(shape, device=entity_poses.device, dtype=entity_poses.dtype)
    ginv[..., 0:3, 0:3] = rot
    ginv[..., 0:3, 3] = p
    ginv[..., 3, 3] = 1.0

    g = torch.zeros_like(ginv)
    g[..., 0:3, 0:3] = rot_t
    g[..., 0:3, 3] = -torch.matmul(rot_t, p.unsqueeze(-1)).squeeze(-1)
    g[..., 3, 3] = 1.0
    return g, ginv


def apply_gta_transform(matrix, features):
    """Apply a 4x4 representation directly to every 4-D head-feature block.

    Args:
        matrix: (B,L,4,4), mathematical column-vector transform.
        features: (B,H,L,D), where D is divisible by four.
    """
    assert matrix.ndim == 4 and matrix.shape[-2:] == (4, 4)
    assert features.ndim == 4 and features.shape[0] == matrix.shape[0]
    assert features.shape[2] == matrix.shape[1]
    assert features.shape[-1] % 4 == 0
    blocks = features.reshape(features.shape[:-1] + (features.shape[-1] // 4, 4))
    matrix = matrix.to(dtype=features.dtype)
    transformed = torch.einsum("blij,bhlrj->bhlri", matrix, blocks)
    return transformed.reshape_as(features)


def _calc_heading_quat(q):
    """Extract TokenHSI-compatible yaw from a full xyzw orientation."""
    ref_dir = torch.zeros_like(q[..., 0:3])
    ref_dir[..., 0] = 1.0
    facing = _quat_rotate(q, ref_dir)
    heading = torch.atan2(facing[..., 1], facing[..., 0])
    half_heading = 0.5 * heading
    heading_q = torch.zeros_like(q)
    heading_q[..., 2] = torch.sin(half_heading)
    heading_q[..., 3] = torch.cos(half_heading)
    return heading_q


def _as_xyz_scale(value, reference):
    """Turn a scalar or XYZ sequence into a device/dtype-matched scale tensor."""
    scale = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if scale.numel() == 1:
        return scale
    assert scale.numel() == 3, "geometry component scale must be scalar or XYZ"
    return scale.reshape(1, 1, 1, 3)


def build_pairwise_geometry(entity_kinematics, scales=None, entity_types=None,
                            mode="full15"):
    """Build one shared 15-D geometry schema for every directed entity pair.

    Args:
        entity_kinematics: (B,L,13), quaternion convention xyzw.
            H/O carry full physical orientation. Target carries identity orientation
            and zero velocities.
        scales: optional (position, linear velocity, angular velocity) multipliers;
            each multiplier may be scalar or XYZ.
        entity_types: optional (L,) IDs from build_entity_type_ids. If provided,
            every target-involving pair keeps only relative position.
        mode: position3, pose9, or full15. The output width always stays 15.
    Returns:
        (B,L,L,15): dp(3), dR tangent/normal(6), dv(3), dw(3).

    Position and velocities use only the source H/O heading, preventing object tilt
    from tilting XYZ. Relative rotation uses both entities' full orientations. A
    target source has no heading, so its outgoing positions stay in shared scene axes.
    Self edges are all-zero; REL_SELF supplies their semantic signal separately.
    """
    assert mode in GEOMETRY_MODES, "unknown geometry mode: {}".format(mode)
    pos = entity_kinematics[..., 0:3]
    rot = entity_kinematics[..., 3:7]
    vel = entity_kinematics[..., 7:10]
    ang = entity_kinematics[..., 10:13]

    B, L = pos.shape[:2]
    if entity_types is not None:
        assert entity_types.shape == (L,)
        entity_types = entity_types.to(device=pos.device)

    src_heading = _calc_heading_quat(rot)
    if entity_types is not None:
        target_sources = entity_types == ENTITY_TARGET
        identity = torch.zeros_like(src_heading)
        identity[..., 3] = 1.0
        src_heading = torch.where(target_sources.view(1, L, 1), identity, src_heading)

    src_heading_inv = _quat_conjugate(src_heading).unsqueeze(2).expand(B, L, L, 4)
    src_full_inv = _quat_conjugate(rot).unsqueeze(2).expand(B, L, L, 4)
    tgt_rot = rot.unsqueeze(1).expand(B, L, L, 4)

    rel_pos = _quat_rotate(src_heading_inv, pos.unsqueeze(1) - pos.unsqueeze(2))
    rel_vel = _quat_rotate(src_heading_inv, vel.unsqueeze(1) - vel.unsqueeze(2))
    rel_ang = _quat_rotate(src_heading_inv, ang.unsqueeze(1) - ang.unsqueeze(2))
    rel_rot = _quat_mul(src_full_inv, tgt_rot)

    ref_tan = torch.zeros_like(rel_pos)
    ref_tan[..., 0] = 1.0
    ref_norm = torch.zeros_like(rel_pos)
    ref_norm[..., 2] = 1.0
    rel_rot_6d = torch.cat([_quat_rotate(rel_rot, ref_tan),
                            _quat_rotate(rel_rot, ref_norm)], dim=-1)

    if scales is not None:
        rel_pos = rel_pos * _as_xyz_scale(scales[0], rel_pos)
        rel_vel = rel_vel * _as_xyz_scale(scales[1], rel_vel)
        rel_ang = rel_ang * _as_xyz_scale(scales[2], rel_ang)

    if entity_types is not None:
        target = entity_types == ENTITY_TARGET
        target_pair = target.view(1, L, 1) | target.view(1, 1, L)
        rel_rot_6d = rel_rot_6d.masked_fill(target_pair.unsqueeze(-1), 0.0)
        rel_vel = rel_vel.masked_fill(target_pair.unsqueeze(-1), 0.0)
        rel_ang = rel_ang.masked_fill(target_pair.unsqueeze(-1), 0.0)

    if mode == "position3":
        rel_rot_6d = torch.zeros_like(rel_rot_6d)
        rel_vel = torch.zeros_like(rel_vel)
        rel_ang = torch.zeros_like(rel_ang)
    elif mode == "pose9":
        rel_vel = torch.zeros_like(rel_vel)
        rel_ang = torch.zeros_like(rel_ang)

    geometry = torch.cat([rel_pos, rel_rot_6d, rel_vel, rel_ang], dim=-1)
    self_pair = torch.eye(L, dtype=torch.bool, device=geometry.device).view(1, L, L, 1)
    return geometry.masked_fill(self_pair, 0.0)


class RelationEncoder(nn.Module):
    """Legacy row or clean scene -> contextualised humanoid embeddings.

    Holds one tokenizer and one type embedding per entity type, plus the relation-biased
    transformer stack. None of its parameters depend on the number of agents.
    """

    def __init__(self, entity_sizes, num_agents, num_objects, d_model, num_heads,
                 num_layers, dim_feedforward, tokenizer_builder,
                 observation_mode="legacy_multirow", kinematic_size=13,
                 relation_bias=True, relation_bias_mode=RELATION_BIAS_LOOKUP,
                 geometry_cfg=None, gta_cfg=None, diagnostics_name="encoder",
                 relation_reward_mode=LEGACY_MODE, diagnostics_interval=100):
        super().__init__()
        if relation_reward_mode not in (LEGACY_MODE, STATE_MODE):
            raise ValueError('Unsupported relation reward mode')
        self.state_relation = relation_reward_mode == STATE_MODE
        self.suffix_width = 9 * num_agents if self.state_relation else 0
        self.diagnostics_interval = max(1, int(diagnostics_interval))
        self.last_diagnostics = {}
        if self.state_relation and (observation_mode != 'clean_scene' or
                relation_bias_mode != RELATION_BIAS_EDGE_MLP or not relation_bias):
            raise ValueError('state_relation_v0 requires clean_scene and enabled edge_mlp relation bias')
        if relation_bias_mode not in RELATION_BIAS_MODES:
            raise ValueError("unknown relation_bias_mode {!r}; expected one of {}".format(
                relation_bias_mode, RELATION_BIAS_MODES))
        self.entity_sizes = list(entity_sizes)
        self.num_agents = num_agents
        self.num_objects = num_objects
        self.entity_counts = [num_agents, num_objects, num_agents]
        self.observation_mode = observation_mode
        self.kinematic_size = kinematic_size
        self.use_relation_bias = relation_bias
        self.relation_bias_mode = relation_bias_mode

        self.tokenizers = nn.ModuleList([tokenizer_builder(sz) for sz in self.entity_sizes])

        self.type_embed = nn.Parameter(torch.zeros(len(self.entity_sizes), d_model))
        nn.init.trunc_normal_(self.type_embed, std=0.02)

        if self.relation_bias_mode == RELATION_BIAS_LOOKUP:
            # Preserve A1's parameter name and exact scalar-lookup implementation for
            # legacy checkpoints and clean-scene ablations.
            self.rel_embed = nn.Parameter(torch.zeros(num_layers, num_heads, NUM_REL_TYPES))
        else:
            self.edge_encoder = EdgeEncoder(num_layers, num_heads,
                                            num_relation_types=8 if self.state_relation else NUM_REL_TYPES)

        # non-persistent: derived from entity counts, so it must NOT end up in the
        # checkpoint -- otherwise loading M=2,O=2 into M=2,O=3 would fail on shape.
        self.register_buffer("rel_matrix", build_relation_matrix(num_agents, num_objects), persistent=False)
        self.register_buffer("entity_types", build_entity_type_ids(num_agents, num_objects),
                             persistent=False)
        if self.state_relation:
            self.rel_matrix = build_state_relation_matrix(num_agents, num_objects)
            graph = compile_carry_subgoal(num_agents, num_objects)
            self.register_buffer('state_edge_src', graph.edge_src, persistent=False)
            self.register_buffer('state_edge_dst', graph.edge_dst, persistent=False)
            self.register_buffer('state_edge_owner', graph.edge_owner, persistent=False)
            self.dynamic_edge_mlp = nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 64))
            # Parameter (not Linear): the builder's global Linear init cannot overwrite zero init.
            self.dynamic_bias_projection = nn.Parameter(torch.zeros(num_layers, num_heads, 64))

        self.layers = nn.ModuleList([
            RelationTransformerLayer(d_model, num_heads, dim_feedforward) for _ in range(num_layers)
        ])

        geometry_cfg = {} if geometry_cfg is None else geometry_cfg
        self.geometry_enabled = bool(geometry_cfg.get("enable", False)) \
            and self.observation_mode == "clean_scene"
        self.geometry_use_score = self.geometry_enabled and bool(geometry_cfg.get("use_score", True))
        self.geometry_use_message = self.geometry_enabled and bool(geometry_cfg.get("use_message", True))
        self.geometry_size = int(geometry_cfg.get("input_size", 15))
        assert self.geometry_size == 15, "shared pairwise geometry schema must stay 15-D"
        self.geometry_mode = geometry_cfg.get("mode", "full15")
        assert self.geometry_mode in GEOMETRY_MODES, \
            "geometry.mode must be one of {}".format(GEOMETRY_MODES)
        geo_dim = int(geometry_cfg.get("embedding_dim", d_model))

        def component_scale(name, default):
            value = geometry_cfg.get(name, default)
            if isinstance(value, (list, tuple)):
                assert len(value) == 3, "{} must be scalar or XYZ".format(name)
                return tuple(float(x) for x in value)
            return float(value)

        self.geometry_scales = (component_scale("position_scale", 1.0),
                                component_scale("velocity_scale", 0.25),
                                component_scale("angular_velocity_scale", 0.25))

        if self.geometry_enabled:
            self.geometry_encoder = nn.Sequential(
                nn.Linear(self.geometry_size, geo_dim),
                nn.ReLU(),
                nn.Linear(geo_dim, geo_dim),
            )
            self.geometry_score = nn.Linear(geo_dim, num_heads) if self.geometry_use_score else None
            self.geometry_message = nn.Linear(geo_dim, d_model) if self.geometry_use_message else None
        else:
            self.geometry_encoder = None
            self.geometry_score = None
            self.geometry_message = None

        gta_cfg = {} if gta_cfg is None else gta_cfg
        self.gta_enabled = bool(gta_cfg.get("enable", False)) \
            and self.observation_mode == "clean_scene"
        self.gta_translation_scale = float(gta_cfg.get("translation_scale", 1.0))
        self.gta_representation = gta_cfg.get(
            "representation", GTA_REPRESENTATION_SE3_DIRECT_SUM)
        self.gta_diagnostics_first_forward = bool(
            gta_cfg.get("diagnostics_first_forward", False))
        self.diagnostics_name = diagnostics_name
        assert self.gta_translation_scale > 0.0
        assert self.gta_representation == GTA_REPRESENTATION_SE3_DIRECT_SUM, \
            "unsupported GTA representation: {}".format(self.gta_representation)
        assert not (self.gta_enabled and self.geometry_enabled), \
            "GTA and old pairwise geometry cannot be enabled together"
        if self.gta_enabled:
            assert self.kinematic_size == 7, \
                "GTA clean-scene observations require 7-D pose records"
            assert all(layer.head_dim % 4 == 0 for layer in self.layers), \
                "GTA se3_direct_sum requires head_dim divisible by four"
        if self.geometry_enabled:
            assert self.kinematic_size == 13, \
                "historical pairwise Geo requires 13-D kinematic records"

        self.forward_calls = 0
        self.last_shape_flow = None

    def set_entity_counts(self, num_agents, num_objects=None):
        """Re-target the encoder at different entity counts (weights are unchanged)."""
        if num_objects is None:
            num_objects = num_agents
        self.num_agents = num_agents
        self.num_objects = num_objects
        self.entity_counts = [num_agents, num_objects, num_agents]
        device = self._relation_device()
        self.rel_matrix = build_relation_matrix(num_agents, num_objects).to(device)
        self.entity_types = build_entity_type_ids(
            num_agents, num_objects, device=device)
        if self.state_relation:
            self.rel_matrix = build_state_relation_matrix(num_agents, num_objects, device)
            graph = compile_carry_subgoal(num_agents, num_objects, device)
            self.state_edge_src, self.state_edge_dst = graph.edge_src, graph.edge_dst
            self.state_edge_owner = graph.edge_owner
            self.suffix_width = 9 * num_agents

    def set_num_agents(self, num_agents):
        """Backward-compatible shorthand for the old 1:1:1 entity layout."""
        self.set_entity_counts(num_agents, num_agents)

    def _relation_device(self):
        if self.relation_bias_mode == RELATION_BIAS_LOOKUP:
            return self.rel_embed.device
        return self.edge_encoder.bias_projection.device

    def build_edge_embeddings(self):
        """Expose A2's batch-independent semantic edges for inspection/tests."""
        if self.relation_bias_mode != RELATION_BIAS_EDGE_MLP:
            raise RuntimeError("edge embeddings are only available in edge_mlp mode")
        return self.edge_encoder.build_edge_embeddings(self.entity_types, self.rel_matrix)

    def build_relation_bias(self):
        """Return semantic bias for every layer/head as (layers,heads,L,L)."""
        if self.relation_bias_mode == RELATION_BIAS_LOOKUP:
            return self.rel_embed[:, :, self.rel_matrix]
        return self.edge_encoder(self.entity_types, self.rel_matrix)

    def build_dynamic_relation_bias(self, suffix):
        """Stored rollout history -> (layers,batch,heads,L,L), directed edges only."""
        B, E, L = suffix.shape[0], 2 * self.num_agents, sum(self.entity_counts)
        assert suffix.shape[1] == self.suffix_width
        state = suffix[:, :4 * E].reshape(B, E, 4)
        done = suffix[:, 4 * E:][:, self.state_edge_owner].unsqueeze(-1)
        edges = self.dynamic_edge_mlp(torch.cat([state, done], -1))
        values = torch.einsum('bed,lhd->lbhe', edges, self.dynamic_bias_projection)
        dense = values.new_zeros(*values.shape[:-1], L * L)
        dense = dense.index_copy(-1, self.state_edge_src * L + self.state_edge_dst, values)
        return dense.reshape(*values.shape[:-1], L, L)

    def forward(self, obs):
        self.forward_calls += 1
        B = obs.shape[0]

        tokens = []
        offset = 0
        for i, (tok, size, count) in enumerate(zip(self.tokenizers, self.entity_sizes,
                                                   self.entity_counts)):
            width = count * size
            block = obs[:, offset:offset + width].view(B, count, size)
            tokens.append(tok(block) + self.type_embed[i])
            offset += width

        x = torch.cat(tokens, dim=1)                                # (B, 2M+O, d)

        geometry = None
        geo_score = None
        geo_message = None
        gta_g = None
        gta_ginv = None
        if self.observation_mode == "clean_scene":
            L = sum(self.entity_counts)
            kin_width = L * self.kinematic_size
            expected = offset + kin_width + self.suffix_width
            assert expected == obs.shape[1], \
                "scene obs is {} wide, nodes+kinematics cover {}".format(obs.shape[1], expected)
            entity_kinematics = obs[:, offset:offset + kin_width].view(B, L, self.kinematic_size)
            if self.gta_enabled:
                if self.gta_diagnostics_first_forward and self.forward_calls == 1:
                    with torch.no_grad():
                        quat_norm = torch.linalg.vector_norm(
                            entity_kinematics[..., 3:7].detach().float(), dim=-1)
                        pos_norm = torch.linalg.vector_norm(
                            entity_kinematics[..., 0:3].detach().float(), dim=-1)
                        print("[MA][GTA diagnostics] {} quaternion_norm(min/mean/max)="
                              "{:.6f}/{:.6f}/{:.6f} position_norm_p95={:.4f}".format(
                                  self.diagnostics_name,
                                  quat_norm.min().item(), quat_norm.mean().item(),
                                  quat_norm.max().item(),
                                  torch.quantile(pos_norm.reshape(-1), 0.95).item()))
                gta_g, gta_ginv = build_gta_transforms(
                    entity_kinematics, self.gta_translation_scale)
            elif self.geometry_enabled:
                geometry = build_pairwise_geometry(
                    entity_kinematics,
                    scales=self.geometry_scales,
                    entity_types=self.entity_types,
                    mode=self.geometry_mode)
                z_geo = self.geometry_encoder(geometry)
                if self.geometry_use_score:
                    geo_score = self.geometry_score(z_geo).permute(0, 3, 1, 2).contiguous()
                if self.geometry_use_message:
                    geo_message = self.geometry_message(z_geo).view(
                        B, L, L, self.layers[0].num_heads, self.layers[0].head_dim)
                    geo_message = geo_message.permute(0, 3, 1, 2, 4).contiguous()
        else:
            assert offset == obs.shape[1], \
                "obs row is {} wide, entity blocks cover {}".format(obs.shape[1], offset)

        relation_bias = self.build_relation_bias() if self.use_relation_bias else None
        collect = self.state_relation and (self.forward_calls == 1 or
                                           self.forward_calls % self.diagnostics_interval == 0)
        if self.state_relation:
            dynamic = self.build_dynamic_relation_bias(obs[:, -self.suffix_width:])
            if collect:
                with torch.no_grad():
                    static = relation_bias.detach().float()
                    sampled = dynamic[:, :32].detach().float()
                    self.last_diagnostics = {
                        'static_row_rms': (static - static.mean(-1, keepdim=True)).square().mean().sqrt(),
                        'dynamic_row_rms': (sampled - sampled.mean(-1, keepdim=True)).square().mean().sqrt()}
            relation_bias = relation_bias.unsqueeze(1) + dynamic
        for i, layer in enumerate(self.layers):
            rel_bias = None if relation_bias is None else relation_bias[i]
            diagnostics_label = None
            if (self.gta_enabled and self.gta_diagnostics_first_forward
                    and self.forward_calls == 1):
                diagnostics_label = "{} layer{}".format(self.diagnostics_name, i)
            x = layer(x, rel_bias, geo_score, geo_message, gta_g, gta_ginv,
                      diagnostics_label=diagnostics_label, collect_diagnostics=collect)
            if collect:
                self.last_diagnostics.update({'layer{}/{}'.format(i, k): v
                                               for k, v in layer.last_diagnostics.items()})

        self.last_shape_flow = {
            "obs": tuple(obs.shape),
            "tokens": tuple(x.shape),
            "geometry": None if geometry is None else tuple(geometry.shape),
            "geometry_mode": self.geometry_mode if self.geometry_enabled else None,
            "geometry_score": None if geo_score is None else tuple(geo_score.shape),
            "geometry_message": None if geo_message is None else tuple(geo_message.shape),
            "gta": self.gta_enabled,
            "gta_g": None if gta_g is None else tuple(gta_g.shape),
            "gta_ginv": None if gta_ginv is None else tuple(gta_ginv.shape),
            "gta_translation_scale": self.gta_translation_scale,
            "relation_bias_mode": self.relation_bias_mode,
            "humans": (B, self.num_agents, x.shape[-1]),
        }

        if self.observation_mode == "clean_scene":
            # Read out only after every H/O/T token has been updated by every layer.
            return x[:, :self.num_agents]

        # Legacy rows remain ego-first and preserve the exact A1 baseline.
        return x[:, 0]


class AMPMultiAgentBuilder(AMPBuilder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def build(self, name, **kwargs):
        return AMPMultiAgentBuilder.Network(self.params, **kwargs)

    class Network(AMPBuilder.Network):

        def __init__(self, params, **kwargs):
            self.num_agents = kwargs["num_agents"]
            self.num_objects = kwargs.get("num_objects", self.num_agents)
            self.observation_mode = kwargs.get("observation_mode", "legacy_multirow")
            if self.observation_mode == "clean_scene":
                self.entity_sizes = list(kwargs["scene_entity_sizes"])
            else:
                self.entity_sizes = [kwargs["humanoid_obs_size"],
                                     kwargs["object_obs_size"],
                                     kwargs["goal_obs_size"]]
            self.scene_kinematic_size = kwargs.get("scene_kinematic_size", 13)
            self.relation_reward_mode = kwargs.get('relation_reward_mode', LEGACY_MODE)
            self.scene_arena_scale = float(kwargs.get("scene_arena_scale", 1.0))
            assert self.scene_arena_scale > 0.0

            super().__init__(params, **kwargs)

            # both trunks of the base A2C network are replaced by relation transformers
            del self.actor_mlp, self.mu, self.mu_act
            del self.critic_mlp, self.value, self.value_act

            self.device = kwargs.get("device", "cuda:0")
            self._build_transformer(params, **kwargs)
            return

        def _build_transformer(self, params, **kwargs):
            tp = params["transformer"]
            d_model = tp["num_features"]
            num_heads = tp["layer_num_heads"]
            num_layers = tp["num_layers"]
            dim_ff = tp["layer_dim_feedforward"]
            tokenizer_units = tp["tokenizer_units"]
            relation_bias = bool(tp.get("relation_bias", True))
            configured_relation_bias_mode = tp.get(
                "relation_bias_mode", RELATION_BIAS_LOOKUP)
            # legacy_multirow remains the exact A1 baseline even when the clean-scene
            # training config selects A2.
            relation_bias_mode = (RELATION_BIAS_LOOKUP
                                  if self.observation_mode == "legacy_multirow"
                                  else configured_relation_bias_mode)
            geometry_cfg = dict(tp.get("geometry", {}))
            # None means "follow the task arena": XY maps to arena-radius units while
            # Z stays in metres. This avoids a stale hard-coded 0.2 if envSpacing changes.
            geometry_cfg["position_scale"] = resolve_geometry_position_scale(
                geometry_cfg.get("position_scale"), self.scene_arena_scale)
            gta_cfg = dict(tp.get("gta", {}))
            if self.observation_mode == "clean_scene":
                assert not (gta_cfg.get("enable", False)
                            and geometry_cfg.get("enable", False)), \
                    "transformer.gta and transformer.geometry are mutually exclusive"

            def tokenizer(input_size):
                return self._build_mlp(input_size=input_size,
                                       units=tokenizer_units + [d_model],
                                       activation=self.activation,
                                       norm_func_name=self.normalization,
                                       dense_func=torch.nn.Linear,
                                       d2rl=self.is_d2rl,
                                       norm_only_first_layer=self.norm_only_first_layer)

            print("[MA] {} mode, {} agents, {} objects, {} tokens, entity sizes {} -> {}-d, relation {}".format(
                self.observation_mode,
                self.num_agents, self.num_objects, 2 * self.num_agents + self.num_objects,
                self.entity_sizes, d_model, relation_bias_mode))
            if self.observation_mode == "clean_scene" and geometry_cfg.get("enable", False):
                print("[MA] geometry mode={}, scales: position={}, velocity={}, angular={}".format(
                    geometry_cfg.get("mode", "full15"),
                    geometry_cfg.get("position_scale", 1.0),
                    geometry_cfg.get("velocity_scale", 0.25),
                    geometry_cfg.get("angular_velocity_scale", 0.25)))
            if self.observation_mode == "clean_scene" and gta_cfg.get("enable", False):
                print("[MA] GTA representation={}, translation_scale={}".format(
                    gta_cfg.get("representation", GTA_REPRESENTATION_SE3_DIRECT_SUM),
                    gta_cfg.get("translation_scale", 1.0)))

            def encoder(diagnostics_name):
                return RelationEncoder(self.entity_sizes, self.num_agents, self.num_objects, d_model,
                                       num_heads, num_layers, dim_ff, tokenizer,
                                       observation_mode=self.observation_mode,
                                       kinematic_size=self.scene_kinematic_size,
                                       relation_bias=relation_bias,
                                       relation_bias_mode=relation_bias_mode,
                                       geometry_cfg=geometry_cfg,
                                       gta_cfg=gta_cfg,
                                       relation_reward_mode=self.relation_reward_mode,
                                       diagnostics_interval=tp.get('relation_diagnostics_interval', 100),
                                       diagnostics_name=diagnostics_name)

            self.actor_encoder = encoder("actor")
            self.critic_encoder = encoder("critic")

            def head(output_size, units):
                return nn.Sequential(
                    self._build_mlp(input_size=d_model, units=units,
                                    activation=self.activation, dense_func=torch.nn.Linear),
                    torch.nn.Linear(units[-1], output_size),
                )

            self.action_head = head(kwargs['actions_num'], tp["extra_mlp_units"])
            self.value_head = head(self.value_size, tp["extra_mlp_units"])

            mlp_init = self.init_factory.create(**{"name": "default"})
            for net in [self.actor_encoder, self.critic_encoder, self.action_head, self.value_head]:
                for m in net.modules():
                    if isinstance(m, nn.Linear):
                        mlp_init(m.weight)
                        if getattr(m, "bias", None) is not None:
                            torch.nn.init.zeros_(m.bias)
            return

        def set_entity_counts(self, num_agents, num_objects=None):
            """Run a trained policy with different entity counts (weights unchanged)."""
            if num_objects is None:
                num_objects = num_agents
            self.num_agents = num_agents
            self.num_objects = num_objects
            self.actor_encoder.set_entity_counts(num_agents, num_objects)
            self.critic_encoder.set_entity_counts(num_agents, num_objects)
            return

        def set_num_agents(self, num_agents):
            """Backward-compatible shorthand for the old 1:1:1 entity layout."""
            return self.set_entity_counts(num_agents, num_agents)

        def eval_actor(self, obs):
            if self.is_continuous and self.space_config['fixed_sigma']:
                encoded = self.actor_encoder(obs)
                mu = self.action_head(encoded)
                if self.observation_mode == "clean_scene":
                    mu = mu.reshape(obs.shape[0] * self.num_agents, -1)
                sigma = mu * 0.0 + self.sigma_act(self.sigma)
                return mu, sigma
            raise NotImplementedError

        def eval_critic(self, obs):
            encoded = self.critic_encoder(obs)
            value = self.value_head(encoded)
            if self.observation_mode == "clean_scene":
                value = value.reshape(obs.shape[0] * self.num_agents, -1)
            return value
