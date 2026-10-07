import torch
from torch import nn

from utils.task_role_spec import task_edges


class TaskRoleMLPFusion(nn.Module):
    """Task and directed endpoint roles -> MLP -> layer/head attention bias only."""

    def __init__(self, num_layers, num_heads, split_tasks=False):
        super().__init__()
        self.split_tasks = split_tasks
        self.source_role_embed = nn.Embedding(3, 16)
        self.task_embed = nn.Embedding(4, 32)
        self.target_role_embed = nn.Embedding(3, 16)
        if split_tasks:
            self.edge_mlps = nn.ModuleList([
                nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64))
                for _ in range(4)])
            self.bias_projection = nn.Parameter(torch.zeros(4, num_layers, num_heads, 64))
        else:
            self.edge_mlp = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64))
            self.bias_projection = nn.Parameter(torch.zeros(num_layers, num_heads, 64))
        for embed in (self.source_role_embed, self.task_embed, self.target_role_embed):
            nn.init.trunc_normal_(embed.weight, std=.02)

    def forward(self, suffix, background_bias):
        valid, src, dst, task, src_role, dst_role = task_edges(suffix)
        # Encode 4*3*3 categories once, then gather across the environment batch.
        tasks, sources, targets = torch.meshgrid(
            torch.arange(4, device=suffix.device), torch.arange(3, device=suffix.device),
            torch.arange(3, device=suffix.device), indexing='ij')
        features = torch.cat((self.source_role_embed(sources),
            self.task_embed(tasks), self.target_role_embed(targets)), -1)
        if self.split_tasks:
            # Each task encodes only its nine role pairs, independent of batch size.
            semantic = torch.stack([mlp(features[t]) for t, mlp in enumerate(self.edge_mlps)])
            table = torch.einsum('tsrd,tlhd->tsrlh', semantic, self.bias_projection)
        else:
            semantic = self.edge_mlp(features)
            table = torch.einsum('tsrd,lhd->tsrlh', semantic, self.bias_projection)
        bias = table[task, src_role, dst_role].permute(2, 0, 3, 1)
        bias = bias * valid[None, :, None]
        n = suffix.shape[0]
        length = background_bias.shape[-1]
        indices = src * length + dst
        dense = bias.new_zeros(*bias.shape[:-1], length * length).scatter_add(
            -1, indices[None, :, None].expand_as(bias), bias)
        occupied = torch.zeros(n, length * length, device=src.device,
            dtype=torch.long).scatter_add(1, indices, valid.long()).bool().reshape(n, length, length)
        background = background_bias[:, None].masked_fill(occupied[None, :, None], 0.)
        return dense.reshape(*dense.shape[:-1], length, length) + background, None


class TaskRoleFusion(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, init_std):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.bias_table = nn.Parameter(torch.zeros(4, 3, 3, num_layers, num_heads))
        self.message_table = nn.Parameter(torch.empty(4, 3, 3, d_model))
        nn.init.normal_(self.message_table, std=init_std)

    def forward(self, suffix, background_bias):
        valid, src, dst, task, src_role, dst_role = task_edges(suffix)
        n, e = src.shape
        l = background_bias.shape[-1]
        bias = self.bias_table[task, src_role, dst_role].permute(2, 0, 3, 1)
        bias = bias * valid[None, :, None]
        indices = src * l + dst
        dense = bias.new_zeros(*bias.shape[:-1], l*l).scatter_add(
            -1, indices[None, :, None].expand_as(bias), bias)
        occupied = torch.zeros(n, l*l, device=src.device, dtype=torch.long).scatter_add(
            1, indices, valid.long()).bool().reshape(n, l, l)
        background = background_bias[:, None].masked_fill(occupied[None, :, None], 0.)
        values = self.message_table[task, src_role, dst_role] * valid[..., None]
        values = values.reshape(n, e, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        return dense.reshape(*dense.shape[:-1], l, l) + background, (src, dst, values)
