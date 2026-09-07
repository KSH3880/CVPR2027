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
# clean_scene instead builds one intrinsic entity set and compact kinematics per env,
# constructs directed LxL geometry on forward, updates all tokens, and gathers all M
# humanoid outputs together. The A1 scalar relation lookup is identical in both modes.
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


class RelationTransformerLayer(nn.Module):
    """Post-LN attention with A1 scalar bias and optional geometry score/message.

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

    def forward(self, x, rel_bias=None, geo_score=None, geo_message=None):
        # x: (B,L,d), rel_bias: (H,L,L), geo_score: (B,H,L,L),
        # geo_message: (B,H,L,L,head_dim)
        B, L, d = x.shape

        qkv = self.qkv(x).view(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                                    # (B, H, L, hd)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale            # (B, H, L, L)
        if rel_bias is not None:
            attn = attn + rel_bias.unsqueeze(0)
        if geo_score is not None:
            attn = attn + geo_score
        attn = torch.softmax(attn, dim=-1)

        if geo_message is None:
            out = torch.matmul(attn, v)                                     # (B,H,L,hd)
        else:
            # sum_j a_ij(V_j + g_ij), written without materialising a second
            # (B,H,L,L,hd) V broadcast.
            out = torch.matmul(attn, v)
            out = out + torch.sum(attn.unsqueeze(-1) * geo_message, dim=3)   # (B,H,L,hd)
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


def build_pairwise_geometry(entity_kinematics, scales=None):
    """Build source-frame directed geometry from (p, q, v, w) entity states.

    Args:
        entity_kinematics: (B,L,13), quaternion convention xyzw.
        scales: optional (position, linear velocity, angular velocity) multipliers.
    Returns:
        (B,L,L,15): dp(3), dR tangent/normal(6), dv(3), dw(3).
    """
    pos = entity_kinematics[..., 0:3]
    rot = entity_kinematics[..., 3:7]
    vel = entity_kinematics[..., 7:10]
    ang = entity_kinematics[..., 10:13]

    B, L = pos.shape[:2]
    src_inv = _quat_conjugate(rot).unsqueeze(2).expand(B, L, L, 4)
    tgt_rot = rot.unsqueeze(1).expand(B, L, L, 4)

    rel_pos = _quat_rotate(src_inv, pos.unsqueeze(1) - pos.unsqueeze(2))
    rel_vel = _quat_rotate(src_inv, vel.unsqueeze(1) - vel.unsqueeze(2))
    rel_ang = _quat_rotate(src_inv, ang.unsqueeze(1) - ang.unsqueeze(2))
    rel_rot = _quat_mul(src_inv, tgt_rot)

    ref_tan = torch.zeros_like(rel_pos)
    ref_tan[..., 0] = 1.0
    ref_norm = torch.zeros_like(rel_pos)
    ref_norm[..., 2] = 1.0
    rel_rot_6d = torch.cat([_quat_rotate(rel_rot, ref_tan),
                            _quat_rotate(rel_rot, ref_norm)], dim=-1)

    if scales is not None:
        rel_pos = rel_pos * scales[0]
        rel_vel = rel_vel * scales[1]
        rel_ang = rel_ang * scales[2]
    return torch.cat([rel_pos, rel_rot_6d, rel_vel, rel_ang], dim=-1)


class RelationEncoder(nn.Module):
    """Legacy row or clean scene -> contextualised humanoid embeddings.

    Holds one tokenizer and one type embedding per entity type, plus the relation-biased
    transformer stack. None of its parameters depend on the number of agents.
    """

    def __init__(self, entity_sizes, num_agents, num_objects, d_model, num_heads,
                 num_layers, dim_feedforward, tokenizer_builder,
                 observation_mode="legacy_multirow", kinematic_size=13,
                 relation_bias=True, geometry_cfg=None):
        super().__init__()
        self.entity_sizes = list(entity_sizes)
        self.num_agents = num_agents
        self.num_objects = num_objects
        self.entity_counts = [num_agents, num_objects, num_agents]
        self.observation_mode = observation_mode
        self.kinematic_size = kinematic_size
        self.use_relation_bias = relation_bias

        self.tokenizers = nn.ModuleList([tokenizer_builder(sz) for sz in self.entity_sizes])

        self.type_embed = nn.Parameter(torch.zeros(len(self.entity_sizes), d_model))
        nn.init.trunc_normal_(self.type_embed, std=0.02)

        # zero init: a relation type never seen during training behaves as plain
        # attention, which is what makes M=1 -> M=2 transfer cheap
        self.rel_embed = nn.Parameter(torch.zeros(num_layers, num_heads, NUM_REL_TYPES))

        # non-persistent: derived from entity counts, so it must NOT end up in the
        # checkpoint -- otherwise loading M=2,O=2 into M=2,O=3 would fail on shape.
        self.register_buffer("rel_matrix", build_relation_matrix(num_agents, num_objects), persistent=False)

        self.layers = nn.ModuleList([
            RelationTransformerLayer(d_model, num_heads, dim_feedforward) for _ in range(num_layers)
        ])

        geometry_cfg = {} if geometry_cfg is None else geometry_cfg
        self.geometry_enabled = bool(geometry_cfg.get("enable", False)) \
            and self.observation_mode == "clean_scene"
        self.geometry_use_score = self.geometry_enabled and bool(geometry_cfg.get("use_score", True))
        self.geometry_use_message = self.geometry_enabled and bool(geometry_cfg.get("use_message", True))
        self.geometry_size = int(geometry_cfg.get("input_size", 15))
        geo_dim = int(geometry_cfg.get("embedding_dim", d_model))
        self.geometry_scales = (float(geometry_cfg.get("position_scale", 1.0)),
                                float(geometry_cfg.get("velocity_scale", 0.25)),
                                float(geometry_cfg.get("angular_velocity_scale", 0.25)))

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

        self.forward_calls = 0
        self.last_shape_flow = None

    def set_entity_counts(self, num_agents, num_objects=None):
        """Re-target the encoder at different entity counts (weights are unchanged)."""
        if num_objects is None:
            num_objects = num_agents
        self.num_agents = num_agents
        self.num_objects = num_objects
        self.entity_counts = [num_agents, num_objects, num_agents]
        self.rel_matrix = build_relation_matrix(num_agents, num_objects).to(self.rel_embed.device)

    def set_num_agents(self, num_agents):
        """Backward-compatible shorthand for the old 1:1:1 entity layout."""
        self.set_entity_counts(num_agents, num_agents)

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
        if self.observation_mode == "clean_scene":
            L = sum(self.entity_counts)
            kin_width = L * self.kinematic_size
            expected = offset + kin_width
            assert expected == obs.shape[1], \
                "scene obs is {} wide, nodes+kinematics cover {}".format(obs.shape[1], expected)
            entity_kinematics = obs[:, offset:].view(B, L, self.kinematic_size)
            if self.geometry_enabled:
                geometry = build_pairwise_geometry(entity_kinematics, self.geometry_scales)
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

        for i, layer in enumerate(self.layers):
            rel_bias = self.rel_embed[i][:, self.rel_matrix] if self.use_relation_bias else None
            x = layer(x, rel_bias, geo_score, geo_message)

        self.last_shape_flow = {
            "obs": tuple(obs.shape),
            "tokens": tuple(x.shape),
            "geometry": None if geometry is None else tuple(geometry.shape),
            "geometry_score": None if geo_score is None else tuple(geo_score.shape),
            "geometry_message": None if geo_message is None else tuple(geo_message.shape),
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
            geometry_cfg = tp.get("geometry", {})

            def tokenizer(input_size):
                return self._build_mlp(input_size=input_size,
                                       units=tokenizer_units + [d_model],
                                       activation=self.activation,
                                       norm_func_name=self.normalization,
                                       dense_func=torch.nn.Linear,
                                       d2rl=self.is_d2rl,
                                       norm_only_first_layer=self.norm_only_first_layer)

            print("[MA] {} mode, {} agents, {} objects, {} tokens, entity sizes {} -> {}-d".format(
                self.observation_mode,
                self.num_agents, self.num_objects, 2 * self.num_agents + self.num_objects,
                self.entity_sizes, d_model))

            def encoder():
                return RelationEncoder(self.entity_sizes, self.num_agents, self.num_objects, d_model,
                                       num_heads, num_layers, dim_ff, tokenizer,
                                       observation_mode=self.observation_mode,
                                       kinematic_size=self.scene_kinematic_size,
                                       relation_bias=relation_bias,
                                       geometry_cfg=geometry_cfg)

            self.actor_encoder = encoder()
            self.critic_encoder = encoder()

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
