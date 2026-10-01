import torch
import torch.nn as nn

from utils.edge_stage1_spec import (parse_semantic_packet,
    parse_owner_holding_packet)
from utils.edge_context_spec import AT
from utils.edge_ontop_spec import ON_TOP


class GroundedEdgeCoordination(nn.Module):
    def __init__(self, width=64, hidden=128, heads=2):
        super().__init__()
        self.grounding = nn.Sequential(nn.Linear(3 * width, hidden), nn.ReLU(),
                                       nn.Linear(hidden, width))
        self.attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)

    def forward(self, humans, nodes, packet, edge_encoder, entity_types,
                owner_fusion=None):
        if owner_fusion is None:
            valid, src, dst, relation, _ = parse_semantic_packet(packet)
        else:
            valid, src, dst, relation, _, state = parse_owner_holding_packet(packet)
        batch, edges = valid.shape
        if edges == 0:
            return humans.new_zeros(humans.shape)
        count = nodes.shape[1]
        if ((src[valid] < 0) | (src[valid] >= count) |
                (dst[valid] < 0) | (dst[valid] >= count)).any():
            raise ValueError('Stage-2 task edge endpoint is outside the scene')
        src = src.masked_fill(~valid, 0)
        dst = dst.masked_fill(~valid, 0)
        relation = relation.masked_fill(~valid, 0)
        row = torch.arange(batch, device=nodes.device)[:, None]
        semantics = edge_encoder.edge_mlp(torch.cat((
            edge_encoder.source_type_embed(entity_types[src]),
            edge_encoder.relation_embed(relation),
            edge_encoder.target_type_embed(entity_types[dst])), -1))
        if owner_fusion is not None:
            placement = valid & ((relation == AT) | (relation == ON_TOP))
            semantics = semantics + placement[..., None] * owner_fusion.state_mlp(
                torch.cat((semantics, state[..., None]), -1))
        grounded = self.grounding(torch.cat((semantics, nodes[row, src], nodes[row, dst]), -1))
        grounded = grounded.masked_fill(~valid[..., None], 0.)
        has_edge = valid.any(-1)
        safe_mask = ~valid.clone()
        safe_mask[~has_edge, 0] = False
        context, _ = self.attention(humans, grounded, grounded,
                                    key_padding_mask=safe_mask, need_weights=False)
        return context * has_edge[:, None, None]
