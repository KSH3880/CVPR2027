import torch
from torch import nn

from utils.task_role_spec import task_edges


class TaskTypeEmbeddingBias(nn.Module):
    """Category vectors, projected by directed physical type pair/layer/head.

    Categories 0..3 are sit/climb/carry_at/carry_ontop. With zero_background=False,
    4=NONE and 5=SELF share the task projection. Otherwise only four task vectors
    exist and background bias is exactly zero. Entity IDs and actor/payload/target
    roles never enter the embedding or projection. With shared_types=True,
    one projection per layer/head is expanded across all directed type pairs.
    """
    def __init__(self, num_layers, num_heads, zero_background=False, shared_types=False):
        super().__init__()
        self.zero_background = zero_background
        self.shared_types = shared_types
        self.category_embed = nn.Embedding(4 if zero_background else 6, 64)
        nn.init.trunc_normal_(self.category_embed.weight, std=.02)
        shape = (num_layers, num_heads, 64) if shared_types else (3, 3, num_layers, num_heads, 64)
        self.bias_projection = nn.Parameter(torch.zeros(shape))

    def bias_table(self):
        # Only 6 * 3 * 3 categories, independent of the rollout batch size.
        if self.shared_types:
            table = torch.einsum('cd,lhd->clh', self.category_embed.weight, self.bias_projection)
            return table[:, None, None].expand(-1, 3, 3, -1, -1)
        return torch.einsum('cd,stlhd->cstlh', self.category_embed.weight, self.bias_projection)

    def forward(self, entity_types, relations):
        # Background relations contain only NONE=0 and SELF=1.
        if self.zero_background:
            layers, heads = self.bias_projection.shape[-3:-1]
            return self.bias_projection.new_zeros(layers, heads, len(entity_types), len(entity_types))
        table = self.bias_table()
        values = table[relations + 4, entity_types[:, None], entity_types[None, :]]
        return values.permute(2, 3, 0, 1)


class TaskActionFusion(nn.Module):
    """Shared across layers: z_H += MLP([z_H, e_task]), without attention weighting.

    Explicit actor indices follow token permutations; packet record order never
    determines which human receives the task. Only valid human records are added.
    """
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 64))
        self.reset_output()

    def reset_output(self):
        # Also called after the network builder's global Linear initialization.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, tokens, condition, collect_diagnostics=False):
        actors, task_vectors, valid = condition
        indices = actors[..., None].expand(-1, -1, tokens.shape[-1])
        humans = tokens.gather(1, indices)
        delta = self.mlp(torch.cat((humans, task_vectors), -1)) * valid[..., None]
        if collect_diagnostics:
            with torch.no_grad():
                mask = valid[:32, :, None].to(humans.dtype)
                count = (mask.sum() * humans.shape[-1]).clamp_min(1)
                state_rms = (humans[:32].detach().float().square() * mask).sum().div(count).sqrt()
                delta_rms = delta[:32].detach().float().square().sum().div(count).sqrt()
                self.last_diagnostics = dict(task_action_input_rms=state_rms,
                    task_action_delta_rms=delta_rms,
                    task_action_to_input_rms=delta_rms / state_rms.clamp_min(1e-8))
        return tokens.scatter_add(1, indices, delta)


def task_action_condition(packet, embedding):
    valid, task, actor, _, _ = packet.reshape(packet.shape[0], -1, 5).long().unbind(-1)
    valid = valid.bool()
    actors = actor.masked_fill(~valid, 0)
    vectors = embedding(task.masked_fill(~valid, 0)) * valid[..., None]
    return actors, vectors, valid


class TaskTypeEmbeddingFusion(nn.Module):
    """Replace background on task edges before the common token/GTA permutation."""
    def forward(self, suffix, edge_encoder, entity_types, background_bias):
        valid, src, dst, task, _, _ = task_edges(suffix)
        table = edge_encoder.bias_table()
        values = table[task, entity_types[src], entity_types[dst]].permute(2, 0, 3, 1)
        values = values * valid[None, :, None]
        length = len(entity_types)
        indices = src * length + dst
        dense = values.new_zeros(*values.shape[:-1], length * length).scatter_add(
            -1, indices[None, :, None].expand_as(values), values)
        occupied = torch.zeros(suffix.shape[0], length * length, device=src.device,
            dtype=torch.long).scatter_add(1, indices, valid.long()).bool().reshape(-1, length, length)
        background = background_bias[:, None].masked_fill(occupied[None, :, None], 0.)
        return dense.reshape(*dense.shape[:-1], length, length) + background, None


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
