"""Stage1 scene policy: shared task bias, human residual MLP and GTA."""

import math
import torch
from torch import nn
from learning.amp_network_builder import AMPBuilder
from learning.multi_agent.task_role_encoder import (
    TaskTypeEmbeddingBias,
    TaskTypeEmbeddingFusion,
    TaskActionFusion,
    task_action_condition,
)
from utils.edge_stage1_spec import STAGE1_CONTEXT_MODE, compile_stage1_graph
from utils.unified_training import validate_transformer

ENTITY_HUMAN, ENTITY_OBJECT, ENTITY_TARGET = (0, 1, 2)
GTA_REPRESENTATION_SE3_DIRECT_SUM = "se3_direct_sum"
RELATION_BIAS_STAGE1 = "task_shared_embedding_action"


def build_entity_type_ids(num_agents, num_objects, device=None):
    """Entity kinds for the canonical [M humans | O objects | M targets] order."""
    return torch.cat(
        [
            torch.full((num_agents,), ENTITY_HUMAN, dtype=torch.long, device=device),
            torch.full((num_objects,), ENTITY_OBJECT, dtype=torch.long, device=device),
            torch.full((num_agents,), ENTITY_TARGET, dtype=torch.long, device=device),
        ]
    )


def build_relation_matrix(num_agents, num_objects=None):
    num_objects = num_agents if num_objects is None else num_objects
    if num_agents < 1 or num_objects < num_agents:
        raise ValueError("Stage1 requires M>=1 and O>=M")
    return torch.eye(2 * num_agents + num_objects, dtype=torch.long)


class RelationTransformerLayer(nn.Module):
    """Post-LN softmax attention with GTA and human task conditioning."""

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

    def forward(
        self,
        x,
        rel_bias=None,
        gta_g=None,
        gta_ginv=None,
        diagnostics_label=None,
        collect_diagnostics=False,
        task_condition=None,
        task_action_fusion=None,
    ):
        B, L, d = x.shape
        qkv = (
            self.qkv(x)
            .view(B, L, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (qkv[0], qkv[1], qkv[2])
        gta_active = gta_g is not None or gta_ginv is not None
        if gta_active:
            assert gta_g is not None and gta_ginv is not None
            q = apply_gta_transform(gta_g.transpose(-1, -2), q)
            k = apply_gta_transform(gta_ginv, k)
            v = apply_gta_transform(gta_ginv, v)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if collect_diagnostics:
            with torch.no_grad():
                sampled = attn[:32].detach().float()
                qk_rms = (
                    (sampled - sampled.mean(-1, keepdim=True)).square().mean().sqrt()
                )
        if rel_bias is not None:
            if rel_bias.ndim == 3:
                assert rel_bias.shape == (self.num_heads, L, L)
                attn = attn + rel_bias.unsqueeze(0)
            else:
                assert rel_bias.shape == (B, self.num_heads, L, L)
                attn = attn + rel_bias
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
                self.last_diagnostics = {
                    "qk_row_rms": qk_rms,
                    "entropy": -(p * p.clamp_min(1e-12).log()).sum(-1).mean(),
                }
        if diagnostics_label is not None:
            with torch.no_grad():
                probs = attn.detach().float()
                entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean()
                print(
                    "[MA][GTA diagnostics] {} q/k/v_norm={:.4f}/{:.4f}/{:.4f} logit_p95/p99={:.4f}/{:.4f} attention_entropy={:.4f}".format(
                        diagnostics_label,
                        q_norm.item(),
                        k_norm.item(),
                        v_norm.item(),
                        p95.item(),
                        p99.item(),
                        entropy.item(),
                    )
                )
        out = torch.matmul(attn, v)
        if gta_active:
            out = apply_gta_transform(gta_g, out)
        out = out.transpose(1, 2).reshape(B, L, d)
        if task_action_fusion is not None:
            out = task_action_fusion(out, task_condition, collect_diagnostics)
            if collect_diagnostics:
                self.last_diagnostics.update(task_action_fusion.last_diagnostics)
        x = self.norm1(x + self.proj(out))
        x = self.norm2(x + self.linear2(self.act(self.linear1(x))))
        return x


def _normalize_quaternion(q, eps=1e-08):
    """Normalize xyzw quaternions without modifying the stored simulator state."""
    norm = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    return q / norm.clamp_min(eps)


def _quat_to_rotation_matrix(q):
    """Convert normalized xyzw quaternions to local-to-env rotation matrices."""
    q = _normalize_quaternion(q)
    x, y, z, w = q.unbind(-1)
    xx, yy, zz = (x * x, y * y, z * z)
    xy, xz, yz = (x * y, x * z, y * z)
    wx, wy, wz = (w * x, w * y, w * z)
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
    return (g, ginv)


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


class RelationEncoder(nn.Module):
    """Stage1 scene tokens to per-human embeddings, independent of entity counts."""

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
        observation_mode="clean_scene",
        kinematic_size=7,
        relation_bias=True,
        relation_bias_mode="task_shared_embedding_action",
        gta_cfg=None,
        diagnostics_name="encoder",
        relation_reward_mode="state_relation_edge_stage1_v1",
        diagnostics_interval=100,
        relation_graph_spec=None,
    ):
        super().__init__()
        if (
            observation_mode != "clean_scene"
            or relation_reward_mode != STAGE1_CONTEXT_MODE
        ):
            raise ValueError("Stage1 requires clean_scene and the Stage1 reward mode")
        if relation_bias_mode != "task_shared_embedding_action" or not relation_bias:
            raise ValueError("Stage1 requires shared task bias")
        compile_stage1_graph(relation_graph_spec, num_agents, num_objects)
        self.relation_graph_spec = relation_graph_spec
        self.diagnostics_interval = max(1, int(diagnostics_interval))
        self.last_diagnostics = {}
        self.entity_sizes = list(entity_sizes)
        self.num_agents = num_agents
        self.num_objects = num_objects
        self.entity_counts = [num_agents, num_objects, num_agents]
        self.observation_mode = observation_mode
        self.kinematic_size = kinematic_size
        self.use_relation_bias = relation_bias
        self.relation_bias_mode = relation_bias_mode
        self.tokenizers = nn.ModuleList(
            [tokenizer_builder(sz) for sz in self.entity_sizes]
        )
        self.type_embed = nn.Parameter(torch.zeros(len(self.entity_sizes), d_model))
        nn.init.trunc_normal_(self.type_embed, std=0.02)
        self.edge_encoder = TaskTypeEmbeddingBias(num_layers, num_heads)
        self.register_buffer(
            "rel_matrix",
            build_relation_matrix(num_agents, num_objects),
            persistent=False,
        )
        self.register_buffer(
            "entity_types",
            build_entity_type_ids(num_agents, num_objects),
            persistent=False,
        )
        self.context_fusion = TaskTypeEmbeddingFusion()
        self.suffix_width = 5 * num_agents
        self._set_context_background()
        self.task_action_fusion = TaskActionFusion()
        self.layers = nn.ModuleList(
            [
                RelationTransformerLayer(d_model, num_heads, dim_feedforward)
                for _ in range(num_layers)
            ]
        )
        gta_cfg = {} if gta_cfg is None else gta_cfg
        self.gta_translation_scale = float(gta_cfg.get("translation_scale", 1.0))
        self.gta_representation = gta_cfg.get(
            "representation", GTA_REPRESENTATION_SE3_DIRECT_SUM
        )
        self.gta_diagnostics_first_forward = bool(
            gta_cfg.get("diagnostics_first_forward", False)
        )
        self.diagnostics_name = diagnostics_name
        assert self.gta_translation_scale > 0.0
        assert self.gta_representation == GTA_REPRESENTATION_SE3_DIRECT_SUM
        assert self.kinematic_size == 7
        assert all((layer.head_dim % 4 == 0 for layer in self.layers))
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
            num_agents, num_objects, device=device
        )
        self.set_task_graph(self.relation_graph_spec)

    def _set_context_background(self):
        self.rel_matrix.zero_()
        self.rel_matrix.fill_diagonal_(1)

    def set_task_graph(self, spec):
        if (
            spec.get("semantic_only") is not True
            or spec.get("policy_task_roles") is not True
        ):
            raise ValueError("Cannot change task packet architecture")
        compile_stage1_graph(
            spec, self.num_agents, self.num_objects, self._relation_device()
        )
        self.relation_graph_spec = spec
        self.suffix_width = 5 * self.num_agents
        self._set_context_background()

    def set_num_agents(self, num_agents):
        """Backward-compatible shorthand for the old 1:1:1 entity layout."""
        self.set_entity_counts(num_agents, num_agents)

    def _relation_device(self):
        return self.edge_encoder.bias_projection.device

    def build_relation_bias(self):
        return self.edge_encoder(self.entity_types, self.rel_matrix)

    def forward(self, obs, return_all=False, token_order=None, object_valid=None):
        self.forward_calls += 1
        B = obs.shape[0]
        tokens = []
        offset = 0
        for i, (tok, size, count) in enumerate(
            zip(self.tokenizers, self.entity_sizes, self.entity_counts)
        ):
            width = count * size
            block = obs[:, offset : offset + width].view(B, count, size)
            tokens.append(tok(block) + self.type_embed[i])
            offset += width
        x = torch.cat(tokens, dim=1)
        gta_g = None
        gta_ginv = None
        L = sum(self.entity_counts)
        kin_width = L * self.kinematic_size
        expected = offset + kin_width + self.suffix_width
        assert (
            expected == obs.shape[1]
        ), "scene obs is {} wide, nodes+kinematics cover {}".format(
            obs.shape[1], expected
        )
        entity_kinematics = obs[:, offset : offset + kin_width].view(
            B, L, self.kinematic_size
        )
        if self.gta_diagnostics_first_forward and self.forward_calls == 1:
            with torch.no_grad():
                quat_norm = torch.linalg.vector_norm(
                    entity_kinematics[..., 3:7].detach().float(), dim=-1
                )
                pos_norm = torch.linalg.vector_norm(
                    entity_kinematics[..., 0:3].detach().float(), dim=-1
                )
                print(
                    "[MA][GTA diagnostics] {} quaternion_norm(min/mean/max)={:.6f}/{:.6f}/{:.6f} position_norm_p95={:.4f}".format(
                        self.diagnostics_name,
                        quat_norm.min().item(),
                        quat_norm.mean().item(),
                        quat_norm.max().item(),
                        torch.quantile(pos_norm.reshape(-1), 0.95).item(),
                    )
                )
        gta_g, gta_ginv = build_gta_transforms(
            entity_kinematics, self.gta_translation_scale
        )
        relation_bias = (
            self.build_relation_bias()
            if self.use_relation_bias and (not False)
            else None
        )
        collect = (
            self.forward_calls == 1
            or self.forward_calls % self.diagnostics_interval == 0
        )
        relation_bias, relation_message = self.context_fusion(
            obs[:, -self.suffix_width :],
            self.edge_encoder,
            self.entity_types,
            relation_bias,
        )
        if collect:
            self.last_diagnostics = {}
        task_condition = (
            task_action_condition(
                obs[:, -self.suffix_width :], self.edge_encoder.category_embed
            )
            if self.task_action_fusion is not None
            else None
        )
        if object_valid is not None:
            valid = torch.ones(B, L, device=obs.device, dtype=torch.bool)
            valid[:, self.num_agents:self.num_agents+self.num_objects] = object_valid
            relation_bias = relation_bias.masked_fill(~valid[None, :, None, None, :], float("-inf"))
        inverse_order = None
        if (
            self.relation_graph_spec.get("shuffle_token_order", False)
            or token_order is not None
        ):
            if token_order is None:
                token_order = torch.randperm(x.shape[1], device=x.device)
            elif not torch.equal(
                token_order.sort().values, torch.arange(x.shape[1], device=x.device)
            ):
                raise ValueError("token_order must permute all entity slots")
            inverse_order = token_order.argsort()
            if task_condition is not None:
                actors, vectors, valid = task_condition
                task_condition = (inverse_order[actors], vectors, valid)
            x = x.index_select(1, token_order)
            if relation_bias is not None:
                relation_bias = relation_bias.index_select(
                    -2, token_order
                ).index_select(-1, token_order)
            if gta_g is not None:
                gta_g = gta_g.index_select(1, token_order)
                gta_ginv = gta_ginv.index_select(1, token_order)
        for i, layer in enumerate(self.layers):
            rel_bias = None if relation_bias is None else relation_bias[i]
            diagnostics_label = None
            if self.gta_diagnostics_first_forward and self.forward_calls == 1:
                diagnostics_label = "{} layer{}".format(self.diagnostics_name, i)
            x = layer(
                x,
                rel_bias,
                gta_g=gta_g,
                gta_ginv=gta_ginv,
                diagnostics_label=diagnostics_label,
                collect_diagnostics=collect,
                task_condition=task_condition,
                task_action_fusion=self.task_action_fusion,
            )
            if collect:
                self.last_diagnostics.update(
                    {
                        "layer{}/{}".format(i, k): v
                        for k, v in layer.last_diagnostics.items()
                    }
                )
        if inverse_order is not None:
            x = x.index_select(1, inverse_order)
        self.last_shape_flow = {
            "obs": tuple(obs.shape),
            "tokens": tuple(x.shape),
            "gta": True,
            "gta_g": None if gta_g is None else tuple(gta_g.shape),
            "gta_ginv": None if gta_ginv is None else tuple(gta_ginv.shape),
            "gta_translation_scale": self.gta_translation_scale,
            "relation_bias_mode": "task_shared_embedding_action",
            "humans": (B, self.num_agents, x.shape[-1]),
        }
        return (x[:, : self.num_agents], x) if return_all else x[:, : self.num_agents]


class AMPMultiAgentBuilder(AMPBuilder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def build(self, name, **kwargs):
        return AMPMultiAgentBuilder.Network(self.params, **kwargs)

    class Network(AMPBuilder.Network):

        def __init__(self, params, **kwargs):
            validate_transformer(params)
            if kwargs.get("observation_mode") != "clean_scene":
                raise ValueError("Stage1 requires clean_scene observations")
            if kwargs.get("relation_reward_mode") != STAGE1_CONTEXT_MODE:
                raise ValueError("Only Stage1 reward mode is supported")
            self.num_agents = kwargs["num_agents"]
            self.num_objects = kwargs.get("num_objects", self.num_agents)
            self.observation_mode = "clean_scene"
            self.entity_sizes = list(kwargs["scene_entity_sizes"])
            self.scene_kinematic_size = kwargs.get("scene_kinematic_size", 7)
            self.relation_reward_mode = kwargs["relation_reward_mode"]
            self.relation_graph_spec = kwargs["relation_graph_spec"]
            self.scene_arena_scale = float(kwargs.get("scene_arena_scale", 1.0))
            assert self.scene_arena_scale > 0.0
            super().__init__(params, **kwargs)
            del self.actor_mlp, self.mu, self.mu_act
            del self.critic_mlp, self.value, self.value_act
            self.device = kwargs.get("device", "cuda:0")
            self._build_transformer(params, **kwargs)

        def _build_transformer(self, params, **kwargs):
            tp = params["transformer"]
            d_model = tp["num_features"]
            num_heads, num_layers = (tp["layer_num_heads"], tp["num_layers"])
            dim_ff, tokenizer_units = (
                tp["layer_dim_feedforward"],
                tp["tokenizer_units"],
            )
            gta_cfg = dict(tp["gta"])

            def tokenizer(input_size):
                return self._build_mlp(
                    input_size=input_size,
                    units=tokenizer_units + [d_model],
                    activation=self.activation,
                    norm_func_name=self.normalization,
                    dense_func=torch.nn.Linear,
                    d2rl=self.is_d2rl,
                    norm_only_first_layer=self.norm_only_first_layer,
                )

            print(
                "[MA] Stage1: {} agents, {} objects, shared task bias + Action MLP + GTA".format(
                    self.num_agents, self.num_objects
                )
            )

            def encoder(label):
                return RelationEncoder(
                    self.entity_sizes,
                    self.num_agents,
                    self.num_objects,
                    d_model,
                    num_heads,
                    num_layers,
                    dim_ff,
                    tokenizer,
                    kinematic_size=self.scene_kinematic_size,
                    gta_cfg=gta_cfg,
                    relation_graph_spec=self.relation_graph_spec,
                    diagnostics_interval=tp.get("relation_diagnostics_interval", 100),
                    diagnostics_name=label,
                )

            self.actor_encoder = encoder("actor")
            self.critic_encoder = encoder("critic")

            def head(output_size, units):
                return nn.Sequential(
                    self._build_mlp(
                        input_size=d_model,
                        units=units,
                        activation=self.activation,
                        dense_func=torch.nn.Linear,
                    ),
                    nn.Linear(units[-1], output_size),
                )

            self.action_head = head(kwargs["actions_num"], tp["extra_mlp_units"])
            self.value_head = head(self.value_size, tp["extra_mlp_units"])
            mlp_init = self.init_factory.create(name="default")
            for branch in (
                self.actor_encoder,
                self.critic_encoder,
                self.action_head,
                self.value_head,
            ):
                for module in branch.modules():
                    if isinstance(module, nn.Linear):
                        mlp_init(module.weight)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)
            for encoder in (self.actor_encoder, self.critic_encoder):
                encoder.task_action_fusion.reset_output()

        def set_entity_counts(self, num_agents, num_objects=None):
            """Run a trained policy with different entity counts (weights unchanged)."""
            if num_objects is None:
                num_objects = num_agents
            self.num_agents = num_agents
            self.num_objects = num_objects
            self.actor_encoder.set_entity_counts(num_agents, num_objects)
            self.critic_encoder.set_entity_counts(num_agents, num_objects)
            return

        def set_task_graph(self, spec):
            self.actor_encoder.set_task_graph(spec)
            self.critic_encoder.set_task_graph(spec)
            self.relation_graph_spec = spec

        def set_num_agents(self, num_agents):
            """Backward-compatible shorthand for the old 1:1:1 entity layout."""
            return self.set_entity_counts(num_agents, num_agents)

        def eval_actor(self, obs):
            if self.is_continuous and self.space_config["fixed_sigma"]:
                encoded = self.actor_encoder(obs)
                mu = self.action_head(encoded)
                mu = mu.reshape(obs.shape[0] * self.num_agents, -1)
                sigma = mu * 0.0 + self.sigma_act(self.sigma)
                return (mu, sigma)
            raise NotImplementedError

        def eval_critic(self, obs):
            encoded = self.critic_encoder(obs)
            value = self.value_head(encoded)
            value = value.reshape(obs.shape[0] * self.num_agents, -1)
            return value
