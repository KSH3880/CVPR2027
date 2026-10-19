import math
import types

import torch
import torch.nn.functional as F


def entmax15(scores):
    values = (scores - scores.amax(-1, keepdim=True)) / 2
    ordered = values.sort(-1, descending=True).values
    count = torch.arange(1, scores.shape[-1] + 1, device=scores.device)
    mean = ordered.cumsum(-1) / count
    variance = ordered.square().cumsum(-1) / count - mean.square()
    threshold = mean - ((1 - count * variance) / count).clamp_min(0).sqrt()
    support = (threshold <= ordered).sum(-1, keepdim=True)
    threshold = threshold.gather(-1, support - 1)
    return (values - threshold).clamp_min(0).square()


def ca_forward(block, humans, tokens, relations, padding):
    attention = block.attention
    width, heads = attention.embed_dim, attention.num_heads
    batch, people, _ = humans.shape
    tasks = tokens.shape[1]
    qw, kw, vw = attention.in_proj_weight.chunk(3)
    qb, kb, vb = attention.in_proj_bias.chunk(3)
    query = F.linear(block.query_norm(humans), qw, qb)
    key, value = F.linear(tokens, kw, kb), F.linear(tokens, vw, vb)
    query = query.reshape(batch, people, heads, width // heads).transpose(1, 2)
    key = key.reshape(batch, tasks, heads, width // heads).transpose(1, 2)
    value = value.reshape(batch, tasks, heads, width // heads).transpose(1, 2)
    scores = query @ key.transpose(-1, -2) / math.sqrt(width // heads)
    bias = block.relation_bias[:, relations].permute(1, 0, 2, 3)
    probability = entmax15((scores + bias).masked_fill(padding, -torch.inf))
    context = (probability @ value).transpose(1, 2).reshape(batch, people, width)
    humans = humans + attention.out_proj(context)
    return humans + block.ffn(block.ffn_norm(humans))


def enable_ca_entmax(coordination):
    for block in coordination.blocks:
        block.forward = types.MethodType(ca_forward, block)
