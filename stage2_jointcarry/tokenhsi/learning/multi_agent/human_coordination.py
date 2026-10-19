import torch
import torch.nn as nn

from learning.multi_agent.task_coordination import CoordinationBlock, task_relations


def human_relations(packet, num_agents):
    packet = packet.reshape(packet.shape[0], -1, 5).long()
    ids = torch.arange(num_agents, device=packet.device)
    owners = packet[..., 0].bool()[..., None] & (packet[..., 2, None] == ids)
    coupled = ((task_relations(packet, num_agents) == 2)[..., None]
               & owners[:, None]).any(2)
    return torch.where(torch.eye(num_agents, device=packet.device, dtype=torch.bool),
                       1, torch.where(coupled, 2, 0))


class HumanCoordination(nn.Module):
    def __init__(self, width=64, heads=2, num_layers=4, feedforward_dim=128,
                 shuffle_human_order=False):
        super().__init__()
        self.shuffle_human_order = shuffle_human_order
        self.blocks = nn.ModuleList([
            CoordinationBlock(width, heads, feedforward_dim, 3)
            for _ in range(num_layers)])

    def init_identity(self):
        for block in self.blocks:
            nn.init.zeros_(block.attention.out_proj.weight)
            nn.init.zeros_(block.attention.out_proj.bias)
            nn.init.zeros_(block.ffn[-1].weight)
            nn.init.zeros_(block.ffn[-1].bias)

    def forward(self, humans, nodes, packet, edge_encoder, entity_types=None, entity_poses=None):
        relations = human_relations(packet, humans.shape[1])
        context = humans
        if self.shuffle_human_order:
            order = torch.randperm(humans.shape[1], device=humans.device)
            context = context[:, order]
            relations = relations[:, order][:, :, order]
        padding = torch.zeros((), device=humans.device, dtype=torch.bool)
        for block in self.blocks:
            context = block(context, None, relations, padding)
        if self.shuffle_human_order:
            context = context[:, order.argsort()]
        has_task = packet.reshape(packet.shape[0], -1, 5)[..., 0].bool().any(-1)
        return torch.where(has_task[:, None, None], context, humans)
