import torch
from torch import nn

from utils.task_role_spec import task_edges


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
