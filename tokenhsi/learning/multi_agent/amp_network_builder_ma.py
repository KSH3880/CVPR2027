# Multi-agent relation-transformer policy.
#
#   obs row (one per (env, agent), ego-first)
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
#                       token 0 = ego humanoid
#                               |
#                 shared action head -> mu     (actor encoder)
#                 shared value head  -> V^i    (critic encoder, separate weights)
#
# There is no self/other distinction and no agent-ID embedding: the relation matrix
# already carries "who owns what", and the ego is defined by the readout position, so
# the network stays permutation-equivariant and independent of the agent count.
#
# Two interchangeable relation-bias mechanisms are supported:
#   lookup   (A1): one scalar per (layer, head, relation type)
#   edge_mlp (A2): [source entity type, relation type, target entity type] -> edge
#                  embedding -> one scalar per (layer, head)
# Both mechanisms only add a scalar to the QK attention logit.  They never modify Q/K/V,
# node features, values, or the token layout.
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

# entity types used only by the A2 edge encoder.  These embeddings are deliberately
# separate from RelationEncoder.type_embed, which belongs to the node/token stream.
ENTITY_HUMANOID = 0
ENTITY_OBJECT = 1
ENTITY_GOAL = 2
NUM_ENTITY_TYPES = 3

RELATION_BIAS_LOOKUP = "lookup"
RELATION_BIAS_EDGE_MLP = "edge_mlp"
RELATION_BIAS_MODES = (RELATION_BIAS_LOOKUP, RELATION_BIAS_EDGE_MLP)


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


def build_entity_type_ids(num_agents, num_objects=None):
    """Build token entity types for [M humanoids | O objects | M goals]."""
    M = num_agents
    O = M if num_objects is None else num_objects
    assert M >= 1
    assert O >= M, "num_objects must be at least num_agents"

    return torch.cat([
        torch.full((M,), ENTITY_HUMANOID, dtype=torch.long),
        torch.full((O,), ENTITY_OBJECT, dtype=torch.long),
        torch.full((M,), ENTITY_GOAL, dtype=torch.long),
    ])


class EdgeEncoder(nn.Module):
    """A2 relation encoder: typed token pairs -> layer/head attention biases.

    The edge representation is intentionally limited to source entity type, the existing
    relation taxonomy, and target entity type.  Geometry and current token features are
    not inputs.  Since entity_type_ids and relation_matrix are shared by the whole batch,
    this module materialises only (L, L, edge_dim) and (layers, heads, L, L), never a
    batch-sized edge tensor.
    """

    def __init__(self, num_layers, num_heads, source_type_dim=16, relation_dim=32,
                 target_type_dim=16, edge_dim=64, activation=nn.ReLU):
        super().__init__()
        input_dim = source_type_dim + relation_dim + target_type_dim

        self.source_type_embed = nn.Embedding(NUM_ENTITY_TYPES, source_type_dim)
        self.relation_embed = nn.Embedding(NUM_REL_TYPES, relation_dim)
        self.target_type_embed = nn.Embedding(NUM_ENTITY_TYPES, target_type_dim)

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_dim, edge_dim),
            activation(),
            nn.Linear(edge_dim, edge_dim),
        )

        # b_ij^(layer,head) = w_(layer,head)^T e_ij.  Exact zero initialisation keeps
        # the first forward identical to plain attention, matching A1's initial state.
        self.bias_projection = nn.Parameter(torch.zeros(num_layers, num_heads, edge_dim))

        for embed in [self.source_type_embed, self.relation_embed, self.target_type_embed]:
            nn.init.trunc_normal_(embed.weight, std=0.02)

    def build_edge_embeddings(self, entity_type_ids, relation_matrix):
        """Return E with shape (L, L, edge_dim), without a batch dimension."""
        L = entity_type_ids.shape[0]
        assert relation_matrix.shape == (L, L)

        source = self.source_type_embed(entity_type_ids).unsqueeze(1).expand(-1, L, -1)
        relation = self.relation_embed(relation_matrix)
        target = self.target_type_embed(entity_type_ids).unsqueeze(0).expand(L, -1, -1)
        return self.edge_mlp(torch.cat([source, relation, target], dim=-1))

    def project_bias(self, edge_embeddings):
        """Project (L, L, edge_dim) edges to (layers, heads, L, L) biases."""
        return torch.einsum("ijd,lhd->lhij", edge_embeddings, self.bias_projection)

    def forward(self, entity_type_ids, relation_matrix):
        edge_embeddings = self.build_edge_embeddings(entity_type_ids, relation_matrix)
        return self.project_bias(edge_embeddings)


class RelationTransformerLayer(nn.Module):
    """Post-LN encoder layer whose attention logits get an additive per-head relation bias.

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

    def forward(self, x, rel_bias):
        # x: (B, L, d), rel_bias: (num_heads, L, L)
        B, L, d = x.shape

        qkv = self.qkv(x).view(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                                    # (B, H, L, hd)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale            # (B, H, L, L)
        attn = attn + rel_bias.unsqueeze(0)
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)                                         # (B, H, L, hd)
        out = out.transpose(1, 2).reshape(B, L, d)

        x = self.norm1(x + self.proj(out))
        x = self.norm2(x + self.linear2(self.act(self.linear1(x))))
        return x


class RelationEncoder(nn.Module):
    """obs row -> contextualised ego-token embedding.

    Holds one tokenizer and one type embedding per entity type, plus the relation-biased
    transformer stack. None of its parameters depend on the number of agents.
    """

    def __init__(self, entity_sizes, num_agents, num_objects, d_model, num_heads,
                 num_layers, dim_feedforward, tokenizer_builder,
                 relation_bias_mode=RELATION_BIAS_LOOKUP):
        super().__init__()
        if relation_bias_mode not in RELATION_BIAS_MODES:
            raise ValueError("unknown relation_bias_mode {!r}; expected one of {}".format(
                relation_bias_mode, RELATION_BIAS_MODES))

        self.entity_sizes = list(entity_sizes)
        self.num_agents = num_agents
        self.num_objects = num_objects
        self.entity_counts = [num_agents, num_objects, num_agents]
        self.relation_bias_mode = relation_bias_mode

        self.tokenizers = nn.ModuleList([tokenizer_builder(sz) for sz in self.entity_sizes])

        self.type_embed = nn.Parameter(torch.zeros(len(self.entity_sizes), d_model))
        nn.init.trunc_normal_(self.type_embed, std=0.02)

        if self.relation_bias_mode == RELATION_BIAS_LOOKUP:
            # A1 is deliberately left unchanged, including the parameter name, so old
            # lookup-mode checkpoints continue to load strictly.
            self.rel_embed = nn.Parameter(torch.zeros(num_layers, num_heads, NUM_REL_TYPES))
        else:
            self.edge_encoder = EdgeEncoder(num_layers, num_heads)

        # non-persistent: derived from entity counts, so it must NOT end up in the
        # checkpoint -- otherwise loading M=2,O=2 into M=2,O=3 would fail on shape.
        self.register_buffer("rel_matrix", build_relation_matrix(num_agents, num_objects), persistent=False)
        self.register_buffer("entity_type_ids", build_entity_type_ids(num_agents, num_objects),
                             persistent=False)

        self.layers = nn.ModuleList([
            RelationTransformerLayer(d_model, num_heads, dim_feedforward) for _ in range(num_layers)
        ])

    def set_entity_counts(self, num_agents, num_objects=None):
        """Re-target the encoder at different entity counts (weights are unchanged)."""
        if num_objects is None:
            num_objects = num_agents
        self.num_agents = num_agents
        self.num_objects = num_objects
        self.entity_counts = [num_agents, num_objects, num_agents]
        device = self._relation_device()
        self.rel_matrix = build_relation_matrix(num_agents, num_objects).to(device)
        self.entity_type_ids = build_entity_type_ids(num_agents, num_objects).to(self.rel_matrix.device)

    def set_num_agents(self, num_agents):
        """Backward-compatible shorthand for the old 1:1:1 entity layout."""
        self.set_entity_counts(num_agents, num_agents)

    def _relation_device(self):
        if self.relation_bias_mode == RELATION_BIAS_LOOKUP:
            return self.rel_embed.device
        return self.edge_encoder.bias_projection.device

    def build_edge_embeddings(self):
        """Expose A2's batch-independent (L, L, 64) edges for inspection/tests."""
        if self.relation_bias_mode != RELATION_BIAS_EDGE_MLP:
            raise RuntimeError("edge embeddings are only available in edge_mlp mode")
        return self.edge_encoder.build_edge_embeddings(self.entity_type_ids, self.rel_matrix)

    def build_relation_bias(self):
        """Return all layer/head biases with shape (layers, heads, L, L)."""
        if self.relation_bias_mode == RELATION_BIAS_LOOKUP:
            return self.rel_embed[:, :, self.rel_matrix]
        return self.edge_encoder(self.entity_type_ids, self.rel_matrix)

    def forward(self, obs):
        B = obs.shape[0]

        tokens = []
        offset = 0
        for i, (tok, size, count) in enumerate(zip(self.tokenizers, self.entity_sizes,
                                                   self.entity_counts)):
            width = count * size
            block = obs[:, offset:offset + width].view(B, count, size)
            tokens.append(tok(block) + self.type_embed[i])
            offset += width

        assert offset == obs.shape[1], \
            "obs row is {} wide, entity blocks cover {}".format(obs.shape[1], offset)

        x = torch.cat(tokens, dim=1)                                # (B, 2M+O, d)

        if self.relation_bias_mode == RELATION_BIAS_LOOKUP:
            # Keep the original A1 lookup in the layer loop exactly as before.
            for i, layer in enumerate(self.layers):
                x = layer(x, self.rel_embed[i][:, self.rel_matrix])  # bias: (H, L, L)
        else:
            relation_bias = self.build_relation_bias()              # (layers, H, L, L)
            for i, layer in enumerate(self.layers):
                x = layer(x, relation_bias[i])                       # bias: (H, L, L)

        # token 0 is always the ego humanoid (obs rows are built ego-first)
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
            self.entity_sizes = [kwargs["humanoid_obs_size"],
                                 kwargs["object_obs_size"],
                                 kwargs["goal_obs_size"]]

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
            relation_bias_mode = tp.get("relation_bias_mode", RELATION_BIAS_LOOKUP)

            def tokenizer(input_size):
                return self._build_mlp(input_size=input_size,
                                       units=tokenizer_units + [d_model],
                                       activation=self.activation,
                                       norm_func_name=self.normalization,
                                       dense_func=torch.nn.Linear,
                                       d2rl=self.is_d2rl,
                                       norm_only_first_layer=self.norm_only_first_layer)

            print("[MA] {} agents, {} objects, {} tokens, entity sizes {} -> {}-d, relation bias {}".format(
                self.num_agents, self.num_objects, 2 * self.num_agents + self.num_objects,
                self.entity_sizes, d_model, relation_bias_mode))

            def encoder():
                return RelationEncoder(self.entity_sizes, self.num_agents, self.num_objects, d_model,
                                       num_heads, num_layers, dim_ff, tokenizer, relation_bias_mode)

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
                mu = self.action_head(self.actor_encoder(obs))
                sigma = mu * 0.0 + self.sigma_act(self.sigma)
                return mu, sigma
            raise NotImplementedError

        def eval_critic(self, obs):
            return self.value_head(self.critic_encoder(obs))
