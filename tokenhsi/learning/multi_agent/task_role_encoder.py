import torch
from torch import nn

from utils.task_role_spec import task_edges


class TaskTypeEmbeddingBias(nn.Module):
    """Six category vectors, projected by directed physical type pair/layer/head.

    Categories 0..3 are sit/climb/carry_at/carry_ontop; 4=NONE, 5=SELF.
    Both background and task edges use the same projection parameters. Entity
    IDs and actor/payload/target roles never enter the embedding or projection.
    """

    def __init__(self, num_layers, num_heads):
        super().__init__()
        self.category_embed = nn.Embedding(6, 64)
        nn.init.trunc_normal_(self.category_embed.weight, std=0.02)
        self.bias_projection = nn.Parameter(
            torch.zeros(3, 3, num_layers, num_heads, 64)
        )

    def bias_table(self):
        # Only 6 * 3 * 3 categories, independent of the rollout batch size.
        return torch.einsum(
            "cd,stlhd->cstlh", self.category_embed.weight, self.bias_projection
        )

    def forward(self, entity_types, relations):
        # Background relations contain only NONE=0 and SELF=1.
        table = self.bias_table()
        values = table[relations + 4, entity_types[:, None], entity_types[None, :]]
        return values.permute(2, 3, 0, 1)


class TaskTypeEmbeddingFusion(nn.Module):
    """Replace background on task edges before the common token/GTA permutation."""

    def __init__(self, merge_shared=False):
        super().__init__()
        self.merge_shared = merge_shared

    def forward(self, suffix, edge_encoder, entity_types, background_bias):
        valid, src, dst, task, _, _ = task_edges(suffix)
        table = edge_encoder.bias_table()
        values = table[task, entity_types[src], entity_types[dst]].permute(2, 0, 3, 1)
        values = values * valid[None, :, None]
        length = len(entity_types)
        indices = src * length + dst
        dense = values.new_zeros(*values.shape[:-1], length * length).scatter_add(
            -1, indices[None, :, None].expand_as(values), values
        )
        counts = torch.zeros(
            suffix.shape[0], length * length, device=src.device, dtype=torch.long
        ).scatter_add(1, indices, valid.long())
        if self.merge_shared:
            dense = dense / counts.clamp_min(1)[None, :, None]
        occupied = counts.bool().reshape(-1, length, length)
        background = background_bias[:, None].masked_fill(occupied[None, :, None], 0.0)
        return dense.reshape(*dense.shape[:-1], length, length) + background, None
