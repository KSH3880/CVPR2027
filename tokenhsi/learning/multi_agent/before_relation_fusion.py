import torch

from learning.multi_agent.coupled_relation_fusion import FusionCoordinationBlock


class BeforeFusionCoordinationBlock(FusionCoordinationBlock):
    def __init__(self, width, heads, feedforward_dim):
        super().__init__(width, heads, feedforward_dim, 4, 'zerobg')

    def effective_relation_bias(self, embedding):
        before = self.bias_projection @ embedding
        zero = torch.zeros_like(before)
        return torch.stack((zero, zero, before, zero), -1)

    def active_relations(self, relations, padding):
        return ((relations[:, None] == 2) & ~padding).any(-1).any(1)
