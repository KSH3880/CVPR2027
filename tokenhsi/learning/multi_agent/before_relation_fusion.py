import torch
from torch import nn
from torch.nn import functional as F

from learning.multi_agent.human_coordination import CoordinationBlock


class BeforeRelationFusion(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(width))
        nn.init.trunc_normal_(self.embedding, std=0.02)
        self.mlp = nn.Sequential(nn.Linear(2*width, width), nn.ReLU(), nn.Linear(width, width))
        self.reset_output()

    def reset_output(self):
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, context, active):
        condition = self.embedding.expand_as(context)
        delta = self.mlp(torch.cat((context, condition), dim=-1))
        return context + delta * active[..., None]


class BeforeFusionCoordinationBlock(CoordinationBlock):
    def __init__(self, width, heads, feedforward_dim, relation_bias=True):
        super().__init__(width, heads, feedforward_dim, relations=4)
        del self.relation_bias
        self.bias_projection = nn.Parameter(torch.zeros(heads, width)) if relation_bias else None

    def effective_relation_bias(self, embedding):
        before = self.bias_projection @ embedding
        return torch.stack((torch.zeros_like(before), torch.zeros_like(before),
                            before, torch.zeros_like(before)), dim=-1)

    def forward(self, humans, tokens, relations, padding, fusion):
        query = self.query_norm(humans)
        batch, people, width = query.shape
        heads = self.attention.num_heads
        depth = width // heads
        qkv = F.linear(query, self.attention.in_proj_weight, self.attention.in_proj_bias)
        q, k, v = qkv.reshape(batch, people, 3, heads, depth).permute(2, 0, 3, 1, 4).unbind(0)
        bias = None
        if self.bias_projection is not None:
            bias = self.effective_relation_bias(fusion.embedding)[:, relations].permute(1, 0, 2, 3)
        if padding is not None:
            if bias is None:
                bias = torch.zeros_like(padding, dtype=q.dtype)
            bias = bias.masked_fill(padding, float('-inf'))
        context = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.0)
        context = context.transpose(1, 2).reshape(batch, people, width)
        active = (relations == 2).any(-1) if padding is None else ((relations[:, None] == 2) & ~padding).any(-1).any(1)
        context = fusion(context, active)
        humans = humans + self.attention.out_proj(context)
        return humans + self.ffn(self.ffn_norm(humans))
