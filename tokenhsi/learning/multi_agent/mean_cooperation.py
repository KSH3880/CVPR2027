import torch
from torch import nn
from torch.nn import functional as F

from learning.multi_agent.human_coordination import stage2_human_relations


class MeanCooperationBlock(nn.Module):
    def __init__(self, width, mixer_hidden, feedforward_dim):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.mix = nn.Sequential(nn.Linear(3*width+3, mixer_hidden), nn.ReLU(),
                                 nn.Linear(mixer_hidden, width))
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, feedforward_dim), nn.ReLU(),
                                 nn.Linear(feedforward_dim, width))

    def forward(self, humans, partners, relation):
        values = self.query_norm(humans)
        weights = partners.to(values.dtype)
        partner_mean = weights @ values / weights.sum(-1, keepdim=True).clamp_min(1)
        other_mean = (values.sum(1, keepdim=True)-values) / max(values.shape[1]-1, 1)
        inputs = torch.cat((values, partner_mean, other_mean, relation), -1)
        humans = humans + self.mix(inputs)
        return humans + self.ffn(self.ffn_norm(humans))


class MeanCooperation(nn.Module):
    def __init__(self, width=64, num_layers=4, mixer_hidden=80, feedforward_dim=128):
        super().__init__()
        self.coupled_peer_mask = None
        self.blocks = nn.ModuleList([
            MeanCooperationBlock(width, mixer_hidden, feedforward_dim) for _ in range(num_layers)
        ])
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.bias)
        self.init_identity()

    def init_identity(self):
        for block in self.blocks:
            for branch in (block.mix, block.ffn):
                nn.init.zeros_(branch[-1].weight)
                nn.init.zeros_(branch[-1].bias)

    def forward(self, humans, nodes, packet, edge_encoder, groups=None, selected_peers=None):
        relations = stage2_human_relations(packet, groups)
        selected = self.coupled_peer_mask if selected_peers is None else selected_peers
        if selected is not None:
            relations = torch.where((relations == 3) & ~selected, 0, relations)
        partners = (relations == 2) | (relations == 3)
        kind = torch.where((relations == 2).any(-1), 1, (relations == 3).any(-1).long()*2)
        relation = F.one_hot(kind, 3).to(humans.dtype)
        context = humans
        for block in self.blocks:
            context = block(context, partners, relation)
        has_task = packet.reshape(packet.shape[0], -1, 5)[..., 0].bool().any(-1)
        return torch.where(has_task[:, None, None], context, humans)
