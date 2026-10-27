import torch
import torch.nn as nn


def task_relations(packet, num_agents):
    valid, task, actor, payload, target = (
        packet.reshape(packet.shape[0], -1, 5).long().unbind(-1)
    )
    valid = valid.bool()
    own = valid[:, None] & (
        actor[:, None] == torch.arange(num_agents, device=packet.device)[None, :, None]
    )
    coupled = (
        valid[:, :, None]
        & valid[:, None, :]
        & (task[:, :, None] >= 2)
        & (task[:, :, None] == task[:, None, :])
        & (payload[:, :, None] == payload[:, None, :])
        & (target[:, :, None] == target[:, None, :])
        & (actor[:, :, None] != actor[:, None, :])
    )
    partner = (own[:, :, :, None] & coupled[:, None]).any(-2)
    relations = torch.where(own, 1, torch.where(partner, 2, 0))
    return relations


class CoordinationBlock(nn.Module):
    def __init__(self, width, heads, feedforward_dim, relations):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            width, heads, dropout=0.0, batch_first=True
        )
        self.query_norm = nn.LayerNorm(width)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, width),
        )
        self.relation_bias = nn.Parameter(torch.zeros(heads, relations))

    def forward(self, humans, tokens, relations, padding):
        bias = self.relation_bias[:, relations].permute(1, 0, 2, 3)
        bias = bias.masked_fill(padding, float("-inf")).flatten(0, 1)
        query = self.query_norm(humans)
        if tokens is None:
            tokens = query
        context, _ = self.attention(
            query, tokens, tokens, attn_mask=bias, need_weights=False
        )
        humans = humans + context
        return humans + self.ffn(self.ffn_norm(humans))


def human_relations(packet, num_agents, before=False):
    packet = packet.reshape(packet.shape[0], -1, 5).long()
    ids = torch.arange(num_agents, device=packet.device)
    owners = packet[..., 0].bool()[..., None] & (packet[..., 2, None] == ids)
    if before:
        valid, task, actor, payload, target = packet.unbind(-1)
        dependency = (valid.bool()[:, :, None] & valid.bool()[:, None, :]
            & (task[:, :, None] == 2)
            & ((task[:, None, :] == 0) | (task[:, None, :] == 1) | (task[:, None, :] == 3))
            & (payload[:, :, None] == target[:, None, :])
            & (actor[:, :, None] != actor[:, None, :]))
        predecessor = torch.einsum('nqi,nji,njk->nqk', owners.transpose(1, 2).float(),
                                   dependency.float(), owners.float()).bool()
        return torch.where(torch.eye(num_agents, device=packet.device, dtype=torch.bool),
                           1, torch.where(predecessor, 2, 0))
    coupled = (
        (task_relations(packet, num_agents) == 2)[..., None] & owners[:, None]
    ).any(2)
    return torch.where(
        torch.eye(num_agents, device=packet.device, dtype=torch.bool),
        1,
        torch.where(coupled, 2, 0),
    )


class HumanCoordination(nn.Module):
    def __init__(
        self,
        width=64,
        heads=2,
        num_layers=4,
        feedforward_dim=128,
        shuffle_human_order=False,
        before=False,
    ):
        super().__init__()
        self.shuffle_human_order = shuffle_human_order
        self.before = before
        self.blocks = nn.ModuleList(
            [
                CoordinationBlock(width, heads, feedforward_dim, 3)
                for _ in range(num_layers)
            ]
        )

    def init_identity(self):
        for block in self.blocks:
            nn.init.zeros_(block.attention.out_proj.weight)
            nn.init.zeros_(block.attention.out_proj.bias)
            nn.init.zeros_(block.ffn[-1].weight)
            nn.init.zeros_(block.ffn[-1].bias)

    def forward(
        self, humans, nodes, packet, edge_encoder, entity_types=None, entity_poses=None,
        relations=None,
    ):
        if relations is None:
            relations = human_relations(packet, humans.shape[1], self.before)
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
