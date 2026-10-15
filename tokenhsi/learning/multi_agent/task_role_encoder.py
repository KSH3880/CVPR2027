import torch
from torch import nn
from utils.task_role_spec import task_edges


class TaskTypeEmbeddingBias(nn.Module):
    """Four task vectors and one projection per layer/head; background is zero."""

    def __init__(self, num_layers, num_heads):
        super().__init__()
        self.category_embed = nn.Embedding(4, 64)
        nn.init.trunc_normal_(self.category_embed.weight, std=0.02)
        self.bias_projection = nn.Parameter(torch.zeros(num_layers, num_heads, 64))

    def bias_table(self):
        table = torch.einsum(
            "cd,lhd->clh", self.category_embed.weight, self.bias_projection
        )
        return table[:, None, None].expand(-1, 3, 3, -1, -1)

    def forward(self, entity_types, relations):
        layers, heads = self.bias_projection.shape[:2]
        return self.bias_projection.new_zeros(
            layers, heads, len(entity_types), len(entity_types)
        )


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
                state_rms = (
                    (humans[:32].detach().float().square() * mask)
                    .sum()
                    .div(count)
                    .sqrt()
                )
                delta_rms = delta[:32].detach().float().square().sum().div(count).sqrt()
                self.last_diagnostics = dict(
                    task_action_input_rms=state_rms,
                    task_action_delta_rms=delta_rms,
                    task_action_to_input_rms=delta_rms / state_rms.clamp_min(1e-08),
                )
        return tokens.scatter_add(1, indices, delta)


def task_action_condition(packet, embedding):
    valid, task, actor, _, _ = packet.reshape(packet.shape[0], -1, 5).long().unbind(-1)
    valid = valid.bool()
    actors = actor.masked_fill(~valid, 0)
    vectors = embedding(task.masked_fill(~valid, 0)) * valid[..., None]
    return (actors, vectors, valid)


class TaskTypeEmbeddingFusion(nn.Module):
    """Replace background on task edges before the common token/GTA permutation."""

    def forward(self, suffix, edge_encoder, entity_types, background_bias):
        valid, src, dst, task = task_edges(suffix)
        table = edge_encoder.bias_table()
        values = table[task, entity_types[src], entity_types[dst]].permute(2, 0, 3, 1)
        values = values * valid[None, :, None]
        length = len(entity_types)
        indices = src * length + dst
        dense = values.new_zeros(*values.shape[:-1], length * length).scatter_add(
            -1, indices[None, :, None].expand_as(values), values
        )
        occupied = (
            torch.zeros(
                suffix.shape[0], length * length, device=src.device, dtype=torch.long
            )
            .scatter_add(1, indices, valid.long())
            .bool()
            .reshape(-1, length, length)
        )
        background = background_bias[:, None].masked_fill(occupied[None, :, None], 0.0)
        return (dense.reshape(*dense.shape[:-1], length, length) + background, None)
