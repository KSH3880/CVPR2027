"""Shared scalar-context fusion into existing entity attention (no edge tokens)."""
import torch
from torch import nn


class EdgeContextFusion(nn.Module):
    def __init__(self, graph):
        super().__init__()
        self.context_encoder = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 64))
        self.fusion_mlp = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 64))
        for field in ('edge_src', 'edge_dst', 'edge_relation', 'edge_valid'):
            self.register_buffer(field, getattr(graph, field), persistent=False)

    def set_graph(self, graph):
        for field in ('edge_src', 'edge_dst', 'edge_relation', 'edge_valid'):
            setattr(self, field, getattr(graph, field).to(self.edge_src.device))

    def forward(self, suffix, semantic_encoder, entity_types, background_bias):
        E, L = self.edge_src.numel(), entity_types.numel()
        if suffix.ndim != 2 or suffix.shape[1] != 2 * E:
            raise ValueError('Expected stored context [N,2E]')
        sem = semantic_encoder.edge_mlp(torch.cat([
            semantic_encoder.source_type_embed(entity_types[self.edge_src]),
            semantic_encoder.relation_embed(self.edge_relation),
            semantic_encoder.target_type_embed(entity_types[self.edge_dst])], -1))
        ctx = self.context_encoder(suffix.reshape(-1, E, 2))
        fused = self.fusion_mlp(torch.cat([sem[None].expand(ctx.shape[0], -1, -1), ctx], -1))
        values = torch.einsum('bed,lhd->lbhe', fused, semantic_encoder.bias_projection)
        values = values * self.edge_valid[None, None, None]
        indices = self.edge_src * L + self.edge_dst
        dense = values.new_zeros(*values.shape[:-1], L * L).scatter_add(
            -1, indices[None, None, None].expand_as(values), values)
        # Task semantics appear only in fused task edges. Keep SELF/NONE background
        # on other pairs; multiple valid task edges on one pair explicitly SUM.
        occupied = torch.zeros(L * L, dtype=torch.long, device=values.device).scatter_add(
            0, indices, self.edge_valid.long()).bool().reshape(L, L)
        background = background_bias.masked_fill(occupied[None, None], 0.)
        return dense.reshape(*dense.shape[:-1], L, L) + background[:, None]


class PackedEdgeContextFusion(nn.Module):
    """Semantic bindings and context come exclusively from each stored PPO sample."""
    def __init__(self):
        super().__init__()
        self.context_encoder = nn.Sequential(nn.Linear(2,32),nn.ReLU(),nn.Linear(32,64))
        self.fusion_mlp = nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Linear(64,64))

    def forward(self, suffix, semantic_encoder, entity_types, background_bias):
        from utils.edge_ontop_spec import parse_packet
        valid,src,dst,relation,owner,context=parse_packet(suffix)
        n,e=src.shape;l=entity_types.numel()
        sem=semantic_encoder.edge_mlp(torch.cat([
            semantic_encoder.source_type_embed(entity_types[src]),
            semantic_encoder.relation_embed(relation),
            semantic_encoder.target_type_embed(entity_types[dst])],-1))
        fused=self.fusion_mlp(torch.cat([sem,self.context_encoder(context)],-1))
        values=torch.einsum('bed,lhd->lbhe',fused,semantic_encoder.bias_projection)*valid[None,:,None]
        indices=src*l+dst
        dense=values.new_zeros(*values.shape[:-1],l*l).scatter_add(-1,indices[None,:,None].expand_as(values),values)
        occupied=torch.zeros(n,l*l,dtype=torch.long,device=src.device).scatter_add(1,indices,valid.long()).bool().reshape(n,l,l)
        background=background_bias[:,None].masked_fill(occupied[None,:,None],0.)
        return dense.reshape(*dense.shape[:-1],l,l)+background
