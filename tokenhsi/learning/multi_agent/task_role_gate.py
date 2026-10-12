"""Task-conditioned block gates and exact 1.5-entmax for GTA attention.

Entmax uses the threshold/Jacobian equations of Peters et al. (2019),
https://arxiv.org/abs/1905.05702. Full sorting, no top-k approximation.
"""
import torch
from torch import nn

from utils.task_role_spec import task_edges


class _Entmax15(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, dim):
        # Center before sorting; accumulate low-precision inputs in float32.
        x = scores.float() if scores.dtype in (torch.float16, torch.bfloat16) else scores
        x = (x - x.amax(dim=dim, keepdim=True)) * .5
        ordered = x.sort(dim=dim, descending=True).values
        shape = [1] * x.ndim
        shape[dim] = x.shape[dim]
        count = torch.arange(1, x.shape[dim] + 1, device=x.device, dtype=x.dtype).reshape(shape)
        mean = ordered.cumsum(dim) / count
        variance_sum = ordered.square().cumsum(dim) - count * mean.square()
        thresholds = mean - ((1 - variance_sum) / count).clamp_min(0).sqrt()
        support = (ordered >= thresholds).sum(dim=dim, keepdim=True)
        threshold = thresholds.gather(dim, support - 1)
        probabilities = (x - threshold).clamp_min(0).square()
        ctx.save_for_backward(probabilities)
        ctx.dim, ctx.input_dtype = dim, scores.dtype
        return probabilities.to(scores.dtype)

    @staticmethod
    def backward(ctx, gradient):
        probabilities, = ctx.saved_tensors
        root = probabilities.sqrt()
        weighted = gradient.to(root.dtype) * root
        correction = weighted.sum(ctx.dim, keepdim=True) / root.sum(ctx.dim, keepdim=True)
        return (weighted - correction * root).to(ctx.input_dtype), None


def entmax15(scores, dim=-1):
    return _Entmax15.apply(scores, dim)


class TaskRoleGate(nn.Module):
    """One task embedding table per encoder; 12 shared task/role inputs per forward."""
    def __init__(self):
        super().__init__()
        self.task_embed = nn.Embedding(4, 64)
        nn.init.trunc_normal_(self.task_embed.weight, std=.02)

    def forward(self, packet):
        valid, src, dst, task, source_role, target_role = task_edges(packet)
        # (actor,payload)/(actor,target)/(payload,target) -> 0/1/2.
        role = source_role + target_role - 1
        category = (task * 3 + role).masked_fill(~valid, 0)
        src, dst = src.masked_fill(~valid, 0), dst.masked_fill(~valid, 0)
        onehot = torch.eye(3, device=packet.device, dtype=self.task_embed.weight.dtype)
        inputs = torch.cat((self.task_embed.weight[:, None].expand(4, 3, 64),
                            onehot[None].expand(4, 3, 3)), -1).reshape(12, 67)
        return src, dst, valid, category, inputs


class BlockGate(nn.Module):
    """Layer-specific MLP; K and V share its 2 heads x 8 block factors."""
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(67, 64), nn.SiLU(), nn.Linear(64, 16))
        self.reset_output()

    def reset_output(self):
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, condition):
        src, dst, valid, category, inputs = condition
        # Work directly with (gate - 1) to keep tiny corrections accurate.
        table_delta = self.mlp(inputs).tanh()
        reduced_precision_gemm = (table_delta.is_cuda and table_delta.dtype == torch.float32
                                  and torch.backends.cuda.matmul.allow_tf32)
        if torch.is_grad_enabled() and table_delta.requires_grad and not reduced_precision_gemm:
            # Many scene edges select the same 12 rows. Advanced-index backward
            # atomically accumulates into those hot rows; a fixed one-hot selector
            # gives the same lookup and sums gradients with a matrix multiply.
            selector = torch.nn.functional.one_hot(category, 12).to(table_delta.dtype)
            selected = selector @ table_delta
        else:
            # Rollout/evaluation has no backward reduction to optimize.
            selected = table_delta[category]
        # Expand blocks after selection, so the shared-table gradient has only
        # 16 columns rather than 64. No parameters/checkpoint keys change.
        delta = selected.reshape(*category.shape, 2, 8).repeat_interleave(4, -1).permute(0, 2, 1, 3)
        return src, dst, valid, delta * valid[:, None, :, None]


def gated_scores(q, k, edges, scale):
    src, dst, valid, delta = edges
    q_edge = q.gather(2, src[:, None, :, None].expand_as(delta))
    k_edge = k.gather(2, dst[:, None, :, None].expand_as(delta))
    scores = (q @ k.transpose(-1, -2)) * scale
    correction = (q_edge * delta * k_edge).sum(-1) * scale
    indices = (src * k.shape[2] + dst)[:, None].expand_as(correction)
    return scores.flatten(2).scatter_add(2, indices, correction).reshape_as(scores)


def gated_values(alpha, v, edges):
    src, dst, valid, delta = edges
    weights = alpha.flatten(2).gather(2, (src * v.shape[2] + dst)[:, None].expand(-1, v.shape[1], -1))
    values = v.gather(2, dst[:, None, :, None].expand_as(delta))
    return (alpha @ v).scatter_add(2, src[:, None, :, None].expand_as(delta),
                                   weights[..., None] * delta * values)


def gate_diagnostics(alpha, edges, category):
    src, dst, valid, delta = edges
    with torch.no_grad():
        weights = alpha.flatten(2).gather(2, (src * alpha.shape[-1] + dst)[:, None].expand(-1, alpha.shape[1], -1))
        mask = valid[:, None].expand_as(weights)
        count = mask.sum().clamp_min(1)
        result = {
            'task_edge_zero_rate': ((weights == 0) & mask).sum() / count,
            'task_edge_attention_mean': (weights * mask).sum() / count,
            'attention_support_count': (alpha > 0).float().sum(-1).mean(),
            'gate_delta_rms': (delta.square().sum() / (count * delta.shape[-1])).sqrt(),
        }
        # Mass per receiver with at least one task edge (not per edge).
        mass = torch.zeros_like(alpha[..., 0]).scatter_add(2, src[:, None].expand_as(weights), weights * mask)
        receivers = torch.zeros_like(mass).scatter_add(2, src[:, None].expand_as(weights), mask.to(mass.dtype)) > 0
        result['task_attention_mass'] = mass.sum() / receivers.sum().clamp_min(1)
        for task, name in enumerate(('sit', 'climb', 'carry_at', 'carry_ontop')):
            selected = mask & (category[:, None] // 3 == task)
            result[name + '_edge_zero_rate'] = ((weights == 0) & selected).sum() / selected.sum().clamp_min(1)
        return result
