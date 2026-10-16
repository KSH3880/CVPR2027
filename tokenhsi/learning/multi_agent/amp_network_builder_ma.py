import math

import torch
from torch import nn

from learning.amp_network_builder import AMPBuilder
from learning.multi_agent.human_coordination import HumanCoordination
from learning.multi_agent.task_role_encoder import (
    TaskTypeEmbeddingBias,
    TaskTypeEmbeddingFusion,
)
from utils.edge_stage2_spec import STAGE2_CONTEXT_MODE


def build_entity_type_ids(num_agents, num_objects, device=None):
    return torch.cat(
        [
            torch.full((n,), kind, dtype=torch.long, device=device)
            for kind, n in enumerate((num_agents, num_objects, num_agents))
        ]
    )


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
    return torch.stack(
        [
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy - wz),
            2.0 * (xz + wy),
            2.0 * (xy + wz),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz - wx),
            2.0 * (xz - wy),
            2.0 * (yz + wx),
            1.0 - 2.0 * (xx + yy),
        ],
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


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


class RelationTransformerLayer(nn.Module):
    """Post-LN attention with task bias and GTA.

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

    def forward(self, x, rel_bias, gta_g, gta_ginv, collect_diagnostics=False):
        B, L, d = x.shape
        qkv = (
            self.qkv(x)
            .view(B, L, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = apply_gta_transform(gta_g.transpose(-1, -2), q)
        k = apply_gta_transform(gta_ginv, k)
        v = apply_gta_transform(gta_ginv, v)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if collect_diagnostics:
            sampled = attn[:32].detach().float()
            qk_rms = (sampled - sampled.mean(-1, keepdim=True)).square().mean().sqrt()
        attn = torch.softmax(attn + rel_bias, dim=-1)
        if collect_diagnostics:
            p = attn[:32].detach().float()
            self.last_diagnostics = {
                "qk_row_rms": qk_rms,
                "entropy": -(p * p.clamp_min(1e-12).log()).sum(-1).mean(),
            }
        out = apply_gta_transform(gta_g, torch.matmul(attn, v))
        out = out.transpose(1, 2).reshape(B, L, d)
        x = self.norm1(x + self.proj(out))
        return self.norm2(x + self.linear2(self.act(self.linear1(x))))


class RelationEncoder(nn.Module):
    def __init__(
        self,
        entity_sizes,
        num_agents,
        num_objects,
        d_model,
        num_heads,
        num_layers,
        dim_feedforward,
        tokenizer_builder,
        graph_spec,
        stage2,
        diagnostics_interval=100,
    ):
        super().__init__()
        self.entity_sizes = entity_sizes
        self.kinematic_size = 7
        self.relation_graph_spec = graph_spec
        self.task_role_input = True
        self.gta_diagnostics_first_forward = False
        self.gta_translation_scale = 1.0
        self.forward_calls = 0
        self.diagnostics_interval = diagnostics_interval
        self.last_diagnostics = {}
        self.tokenizers = nn.ModuleList([tokenizer_builder(sz) for sz in entity_sizes])
        self.type_embed = nn.Parameter(torch.zeros(len(entity_sizes), d_model))
        nn.init.trunc_normal_(self.type_embed, std=0.02)
        self.edge_encoder = TaskTypeEmbeddingBias(num_layers, num_heads)
        self.context_fusion = TaskTypeEmbeddingFusion(merge_shared=stage2)
        self.layers = nn.ModuleList(
            [
                RelationTransformerLayer(d_model, num_heads, dim_feedforward)
                for _ in range(num_layers)
            ]
        )
        self.register_buffer(
            "rel_matrix", torch.empty(0, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "entity_types", torch.empty(0, dtype=torch.long), persistent=False
        )
        self.set_entity_counts(num_agents, num_objects)

    def set_entity_counts(self, num_agents, num_objects):
        self.num_agents, self.num_objects = num_agents, num_objects
        self.entity_counts = [num_agents, num_objects, num_agents]
        self.suffix_width = 5 * num_agents
        device = self.edge_encoder.bias_projection.device
        self.entity_types = build_entity_type_ids(num_agents, num_objects, device)
        self.rel_matrix = torch.eye(
            sum(self.entity_counts), dtype=torch.long, device=device
        )

    def set_task_graph(self, spec):
        self.relation_graph_spec = spec

    def build_relation_bias(self):
        return self.edge_encoder(self.entity_types, self.rel_matrix)

    def forward(self, obs, return_all=False, token_order=None):
        self.forward_calls += 1
        B = len(obs)
        tokens, offset = [], 0
        for i, (tok, size, count) in enumerate(
            zip(self.tokenizers, self.entity_sizes, self.entity_counts)
        ):
            width = count * size
            tokens.append(
                tok(obs[:, offset : offset + width].reshape(B, count, size))
                + self.type_embed[i]
            )
            offset += width
        x = torch.cat(tokens, dim=1)
        length = sum(self.entity_counts)
        poses = obs[:, offset : offset + length * 7].reshape(B, length, 7)
        gta_g, gta_ginv = build_gta_transforms(poses, self.gta_translation_scale)
        bias, _ = self.context_fusion(
            obs[:, -self.suffix_width :],
            self.edge_encoder,
            self.entity_types,
            self.build_relation_bias(),
        )
        if self.relation_graph_spec.get('mask_unused_entities', False):
            from utils.group_carry_spec import entity_validity
            valid = entity_validity(obs[:, -self.suffix_width:], self.num_agents, self.num_objects)
            bias = bias.masked_fill(~valid[None, :, None, None, :], float('-inf'))
        inverse = None
        if (
            self.relation_graph_spec.get("shuffle_token_order", False)
            or token_order is not None
        ):
            if token_order is None:
                token_order = torch.randperm(length, device=x.device)
            elif not torch.equal(
                token_order.sort().values, torch.arange(length, device=x.device)
            ):
                raise ValueError("token_order must permute all entity slots")
            inverse = token_order.argsort()
            x = x[:, token_order]
            bias = bias.index_select(-2, token_order).index_select(-1, token_order)
            gta_g, gta_ginv = gta_g[:, token_order], gta_ginv[:, token_order]
        collect = (
            self.forward_calls == 1
            or self.forward_calls % self.diagnostics_interval == 0
        )
        if collect:
            self.last_diagnostics = {}
        for i, layer in enumerate(self.layers):
            x = layer(x, bias[i], gta_g, gta_ginv, collect)
            if collect:
                self.last_diagnostics.update(
                    {
                        "layer{}/{}".format(i, k): v
                        for k, v in layer.last_diagnostics.items()
                    }
                )
        if inverse is not None:
            x = x[:, inverse]
        return (x[:, : self.num_agents], x) if return_all else x[:, : self.num_agents]


class AMPMultiAgentBuilder(AMPBuilder):
    def build(self, name, **kwargs):
        return self.Network(self.params, **kwargs)

    class Network(AMPBuilder.Network):
        def __init__(self, params, **kwargs):
            self.num_agents = kwargs["num_agents"]
            self.num_objects = kwargs["num_objects"]
            self.observation_mode = "clean_scene"
            self.entity_sizes = list(kwargs["scene_entity_sizes"])
            self.scene_kinematic_size = 7
            self.relation_reward_mode = kwargs["relation_reward_mode"]
            self.stage2 = self.relation_reward_mode == STAGE2_CONTEXT_MODE
            self.relation_graph_spec = kwargs["relation_graph_spec"]
            super().__init__(params, **kwargs)
            del self.actor_mlp, self.mu, self.mu_act
            del self.critic_mlp, self.value, self.value_act
            tp = params["transformer"]
            cc = params.get("coordination", {})
            self.grasp_view = cc.get("grasp_view", False)
            self.ca_only = self.stage2
            if self.stage2 and (
                cc.get("mode") != "human_coupled"
                or cc.get("num_layers") != 4
                or cc.get("head_input") != "ca_only"
                or cc.get("critic_ca") is not True
            ):
                raise ValueError("Stage2 requires human SA4 actor/critic")
            if self.grasp_view and (self.num_agents, self.num_objects) not in ((2, 4), (4, 8)):
                raise ValueError("Grasp views require 2 humans / 4 objects")
            width = tp["num_features"]

            def tokenizer(size):
                return self._build_mlp(
                    input_size=size,
                    units=tp["tokenizer_units"] + [width],
                    activation=self.activation,
                    norm_func_name=self.normalization,
                    dense_func=nn.Linear,
                    d2rl=self.is_d2rl,
                    norm_only_first_layer=self.norm_only_first_layer,
                )

            def encoder():
                return RelationEncoder(
                    self.entity_sizes,
                    self.num_agents,
                    self.num_objects,
                    width,
                    tp["layer_num_heads"],
                    tp["num_layers"],
                    tp["layer_dim_feedforward"],
                    tokenizer,
                    self.relation_graph_spec,
                    self.stage2,
                    tp.get("relation_diagnostics_interval", 100),
                )

            self.actor_encoder, self.critic_encoder = encoder(), encoder()
            if self.stage2:

                def coordination():
                    return HumanCoordination(
                        width=width,
                        heads=cc["num_heads"],
                        num_layers=4,
                        feedforward_dim=128,
                        shuffle_human_order=cc.get("shuffle_human_order", False),
                    )

                self.coordination, self.critic_coordination = (
                    coordination(),
                    coordination(),
                )

            def head(size):
                units = tp["extra_mlp_units"]
                return nn.Sequential(
                    self._build_mlp(
                        input_size=width,
                        units=units,
                        activation=self.activation,
                        dense_func=nn.Linear,
                    ),
                    nn.Linear(units[-1], size),
                )

            self.action_head, self.value_head = (
                head(kwargs["actions_num"]),
                head(self.value_size),
            )
            modules = [
                self.actor_encoder,
                self.critic_encoder,
                self.action_head,
                self.value_head,
            ]
            if self.stage2:
                modules.append(self.coordination)
                modules.append(self.critic_coordination)
            initialize = self.init_factory.create(name="default")
            for module in modules:
                for layer in module.modules():
                    if isinstance(layer, nn.Linear):
                        initialize(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
            if self.stage2:
                self.coordination.init_identity()
                self.critic_coordination.init_identity()

        def set_entity_counts(self, num_agents, num_objects=None):
            self.num_agents = num_agents
            self.num_objects = num_agents if num_objects is None else num_objects
            self.actor_encoder.set_entity_counts(self.num_agents, self.num_objects)
            self.critic_encoder.set_entity_counts(self.num_agents, self.num_objects)

        def set_task_graph(self, spec):
            self.actor_encoder.set_task_graph(spec)
            self.critic_encoder.set_task_graph(spec)
            self.relation_graph_spec = spec

        def _encode(self, encoder, coordination, obs):
            if self.grasp_view:
                from learning.multi_agent.grasp_view import encode_grasp_views

                humans = encode_grasp_views(encoder, obs)
            else:
                humans = encoder(obs)
            return coordination(humans, None, obs[:, -encoder.suffix_width :], None)

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            actor = self.eval_actor(obs, obs_dict.get('actor_features'))
            return actor + (self.eval_critic(obs), obs_dict.get('rnn_states'))

        @torch.no_grad()
        def encode_actor(self, obs):
            if self.grasp_view:
                from learning.multi_agent.grasp_view import encode_grasp_views

                return encode_grasp_views(self.actor_encoder, obs)
            return self.actor_encoder(obs)

        def eval_actor(self, obs, actor_features=None):
            if self.stage2:
                humans = self.encode_actor(obs) if actor_features is None else actor_features
                encoded = self.coordination(
                    humans, None, obs[:, -self.actor_encoder.suffix_width :], None
                )
            else:
                encoded = self.actor_encoder(obs)
            mu = self.action_head(encoded).reshape(len(obs) * self.num_agents, -1)
            return mu, mu * 0.0 + self.sigma_act(self.sigma)

        def eval_critic(self, obs):
            encoded = (
                self._encode(self.critic_encoder, self.critic_coordination, obs)
                if self.stage2
                else self.critic_encoder(obs)
            )
            return self.value_head(encoded).reshape(len(obs) * self.num_agents, -1)
