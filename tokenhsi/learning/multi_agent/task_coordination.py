import torch
import torch.nn as nn


def task_relations(packet, num_agents, before=False):
    valid, task, actor, payload, target = packet.reshape(packet.shape[0], -1, 5).long().unbind(-1)
    valid = valid.bool()
    own = valid[:, None] & (actor[:, None] ==
        torch.arange(num_agents, device=packet.device)[None, :, None])
    # This module's joint-carry contract declares matching carry bundles coupled.
    coupled = (valid[:, :, None] & valid[:, None, :] & (task[:, :, None] >= 2)
        & (task[:, :, None] == task[:, None, :])
        & (payload[:, :, None] == payload[:, None, :])
        & (target[:, :, None] == target[:, None, :])
        & (actor[:, :, None] != actor[:, None, :]))
    partner = (own[:, :, :, None] & coupled[:, None]).any(-2)
    relations = torch.where(own, 1, torch.where(partner, 2, 0))
    if before:
        dependency = (valid[:, :, None] & valid[:, None, :]
            & (task[:, :, None] == 2)
            & ((task[:, None, :] == 0) | (task[:, None, :] == 1) | (task[:, None, :] == 3))
            & (payload[:, :, None] == target[:, None, :])
            & (actor[:, :, None] != actor[:, None, :]))
        predecessor = (own[:, :, None, :] & dependency[:, None]).any(-1)
        relations = torch.where(predecessor, 3, relations)
    return relations


class TaskCoordination(nn.Module):
    def __init__(self, width=64, hidden=128, heads=2, shuffle_task_order=False, before=False):
        super().__init__()
        self.shuffle_task_order = shuffle_task_order
        self.before = before
        self.grounding = nn.Sequential(nn.Linear(4 * width, hidden), nn.ReLU(),
                                       nn.Linear(hidden, width))
        self.attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.relation_bias = nn.Parameter(torch.zeros(heads, 4 if before else 3))

    def task_tokens(self, nodes, packet, edge_encoder):
        valid, task, actor, payload, target = packet.reshape(packet.shape[0], -1, 5).long().unbind(-1)
        valid = valid.bool()
        task, actor, payload, target = [x.masked_fill(~valid, 0)
                                       for x in (task, actor, payload, target)]
        row = torch.arange(nodes.shape[0], device=nodes.device)[:, None]
        embedding = edge_encoder.category_embed(task).detach()
        carried = nodes[row, payload] * (task >= 2)[..., None]
        tokens = self.grounding(torch.cat((embedding, nodes[row, actor],
                                          carried, nodes[row, target]), -1))
        return tokens.masked_fill(~valid[..., None], 0), valid

    def forward(self, humans, nodes, packet, edge_encoder, entity_types=None):
        if self.shuffle_task_order:
            packet = packet.reshape(packet.shape[0], -1, 5)
            packet = packet[:, torch.randperm(packet.shape[1], device=packet.device)]
        tokens, valid = self.task_tokens(nodes, packet, edge_encoder)
        relations = task_relations(packet, humans.shape[1], self.before)
        bias = self.relation_bias[:, relations].permute(1, 0, 2, 3)
        has_task = valid.any(-1)
        safe_valid = valid.clone()
        safe_valid[~has_task, 0] = True
        bias = bias.masked_fill(~safe_valid[:, None, None], float('-inf'))
        context, _ = self.attention(humans, tokens, tokens,
            attn_mask=bias.flatten(0, 1), need_weights=False)
        return context * has_task[:, None, None]
