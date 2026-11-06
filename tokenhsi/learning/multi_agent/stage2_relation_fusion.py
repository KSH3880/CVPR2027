import torch
from torch import nn

from learning.multi_agent.coupled_relation_fusion import CoupledRelationFusion, FusionCoordinationBlock


class Stage2RelationFusion(CoupledRelationFusion):
    def __init__(self, width):
        super().__init__(width)
        self.embedding = nn.Parameter(torch.empty(2, width))
        nn.init.trunc_normal_(self.embedding, std=.02)

    def forward(self, context, active):
        values = context[..., None, :].expand(*context.shape[:-1], 2, context.shape[-1])
        conditions = self.embedding.expand_as(values)
        delta = self.mlp(torch.cat((values, conditions), -1))
        return context + (delta * active[..., None]).sum(-2)


class Stage2FusionCoordinationBlock(FusionCoordinationBlock):
    def __init__(self, width, heads, feedforward_dim):
        super().__init__(width, heads, feedforward_dim, 4, 'zerobg')

    def effective_relation_bias(self, embedding):
        return torch.cat((self.background_bias, self.bias_projection @ embedding.T), -1)

    def active_relations(self, relations, padding):
        return torch.stack([((relations[:, None] == kind) & ~padding).any(-1).any(1)
                            for kind in (2, 3)], -1)
