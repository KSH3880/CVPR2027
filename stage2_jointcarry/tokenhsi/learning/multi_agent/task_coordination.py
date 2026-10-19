import torch
import torch.nn as nn

from learning.multi_agent.task_geometry import canonical_pose, relative_pose


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


class CoordinationBlock(nn.Module):
    def __init__(self, width, heads, feedforward_dim, relations):
        super().__init__()
        self.attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.query_norm = nn.LayerNorm(width)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, feedforward_dim), nn.ReLU(),
                                 nn.Linear(feedforward_dim, width))
        self.relation_bias = nn.Parameter(torch.zeros(heads, relations))

    def forward(self, humans, tokens, relations, padding):
        bias = self.relation_bias[:, relations].permute(1, 0, 2, 3)
        bias = bias.masked_fill(padding, float('-inf')).flatten(0, 1)
        query = self.query_norm(humans)
        if tokens is None:
            tokens = query
        context, _ = self.attention(query, tokens, tokens,
                                    attn_mask=bias, need_weights=False)
        humans = humans + context
        return humans + self.ffn(self.ffn_norm(humans))


class TaskCoordination(nn.Module):
    def __init__(self, width=64, hidden=128, heads=2, shuffle_task_order=False, before=False,
                 num_layers=1, feedforward_dim=None, residual_output=False,
                 detach_task_embedding=True, task_input='sa_features', mask_none=False):
        super().__init__()
        self.shuffle_task_order = shuffle_task_order
        self.before = before
        self.residual_output = residual_output
        self.detach_task_embedding = detach_task_embedding
        self.mask_none = mask_none
        if mask_none and feedforward_dim is None:
            raise ValueError('NONE masking requires stacked task CA')
        if task_input not in ('sa_features', 'geometry'):
            raise ValueError('Unknown task token input: ' + task_input)
        self.task_input = task_input
        input_width = width + 21 if task_input == 'geometry' else 4 * width
        self.grounding = nn.Sequential(nn.Linear(input_width, hidden), nn.ReLU(),
                                       nn.Linear(hidden, width))
        if feedforward_dim is None:
            if num_layers != 1:
                raise ValueError('Stacked task CA requires a feedforward dimension')
            self.attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
            self.relation_bias = nn.Parameter(torch.zeros(heads, 4 if before else 3))
        else:
            if num_layers < 1 or feedforward_dim < 1:
                raise ValueError('Task CA block dimensions must be positive')
            self.blocks = nn.ModuleList([
                CoordinationBlock(width, heads, feedforward_dim, 4 if before else 3)
                for _ in range(num_layers)])

    def init_identity(self):
        for block in self.blocks:
            nn.init.zeros_(block.attention.out_proj.weight)
            nn.init.zeros_(block.attention.out_proj.bias)
            nn.init.zeros_(block.ffn[-1].weight)
            nn.init.zeros_(block.ffn[-1].bias)

    def task_tokens(self, nodes, packet, edge_encoder, entity_poses=None):
        valid, task, actor, payload, target = packet.reshape(packet.shape[0], -1, 5).long().unbind(-1)
        valid = valid.bool()
        task, actor, payload, target = [x.masked_fill(~valid, 0)
                                       for x in (task, actor, payload, target)]
        row = torch.arange(nodes.shape[0], device=nodes.device)[:, None]
        embedding = edge_encoder.category_embed(task)
        if self.detach_task_embedding:
            embedding = embedding.detach()
        if self.task_input == 'geometry':
            if entity_poses is None or entity_poses.shape != (*nodes.shape[:2], 7):
                raise ValueError('Geometry task tokens require canonical entity poses [B,L,7]')
            human = canonical_pose(entity_poses[row, actor])
            carried = relative_pose(human, entity_poses[row, payload]) * (task >= 2)[..., None]
            target_feature = relative_pose(human, entity_poses[row, target])
        else:
            human = nodes[row, actor]
            carried = nodes[row, payload] * (task >= 2)[..., None]
            target_feature = nodes[row, target]
        tokens = self.grounding(torch.cat((embedding, human, carried, target_feature), -1))
        return tokens.masked_fill(~valid[..., None], 0), valid

    def forward(self, humans, nodes, packet, edge_encoder, entity_types=None, entity_poses=None):
        if self.shuffle_task_order:
            packet = packet.reshape(packet.shape[0], -1, 5)
            packet = packet[:, torch.randperm(packet.shape[1], device=packet.device)]
        tokens, valid = self.task_tokens(nodes, packet, edge_encoder, entity_poses)
        relations = task_relations(packet, humans.shape[1], self.before)
        has_task = valid.any(-1)
        safe_valid = valid.clone()
        safe_valid[~has_task, 0] = True
        if hasattr(self, 'blocks'):
            padding = ~safe_valid[:, None, None]
            if self.mask_none:
                padding = padding | (relations == 0)[:, None]
                has_related = (~padding).any(-1).squeeze(1)
                padding[..., 0] &= has_related[:, None]
            context = humans
            for block in self.blocks:
                context = block(context, tokens, relations, padding)
            if self.mask_none:
                context = torch.where(has_related[..., None], context,
                                      humans if self.residual_output else torch.zeros_like(context))
            if self.residual_output:
                return torch.where(has_task[:, None, None], context, humans)
            return context * has_task[:, None, None]
        bias = self.relation_bias[:, relations].permute(1, 0, 2, 3)
        bias = bias.masked_fill(~safe_valid[:, None, None], float('-inf'))
        context, _ = self.attention(humans, tokens, tokens,
            attn_mask=bias.flatten(0, 1), need_weights=False)
        return context * has_task[:, None, None]
