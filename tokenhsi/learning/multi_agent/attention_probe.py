"""Read-only diagnostics for the existing Stage-2 multihead attention.

Reconstructs per-head weights without changing the production forward path.
Counterfactuals keep the encoder/query/values fixed and only alter CA routing.
"""
import math
import torch
import torch.nn.functional as F


@torch.no_grad()
def alter_context(attention, humans, edges, valid, owner, mode):
    """Inference-only interventions; leave the actor encoder unchanged."""
    if mode == 'ca_off':
        return torch.zeros_like(humans)
    n, m, width = humans.shape
    e, h = edges.shape[1], attention.num_heads
    d = width // h
    wq, wk, wv = attention.in_proj_weight.chunk(3)
    bq, bk, bv = attention.in_proj_bias.chunk(3)
    v = F.linear(edges, wv, bv).reshape(n, e, h, d).transpose(1, 2)
    if mode == 'uniform':
        weights = (valid.float() / valid.sum(-1, keepdim=True))[:, None, None].expand(n, h, m, e)
    elif mode == 'peer_removed':
        own = valid[:, None] & (owner[:, None] == torch.arange(m, device=humans.device)[None, :, None])
        if not own.any(-1).all():
            raise ValueError('Every human must have an owned edge for peer removal')
        q = F.linear(humans, wq, bq).reshape(n, m, h, d).transpose(1, 2)
        k = F.linear(edges, wk, bk).reshape(n, e, h, d).transpose(1, 2)
        weights = ((q @ k.transpose(-1, -2))/math.sqrt(d)).masked_fill(~own[:, None], -torch.inf).softmax(-1)
    else:
        raise ValueError(mode)
    return attention.out_proj((weights @ v).transpose(1, 2).reshape(n, m, width))


@torch.no_grad()
def inspect_coordination(attention, action_head, humans, edges, valid, context):
    n, m, width = humans.shape
    e, h = edges.shape[1], attention.num_heads
    d = width // h
    if not valid.any(-1).all():
        raise ValueError('Probe requires at least one valid edge per scene')
    wq, wk, wv = attention.in_proj_weight.chunk(3)
    bq, bk, bv = attention.in_proj_bias.chunk(3)
    q = F.linear(humans, wq, bq).reshape(n, m, h, d).transpose(1, 2)
    k = F.linear(edges, wk, bk).reshape(n, e, h, d).transpose(1, 2)
    v = F.linear(edges, wv, bv).reshape(n, e, h, d).transpose(1, 2)
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(d)
    weights = scores.masked_fill(~valid[:, None, None], -torch.inf).softmax(-1)

    def project(a):
        z = (a @ v).transpose(1, 2).reshape(n, m, width)
        return attention.out_proj(z)

    rebuilt = project(weights)
    error = (rebuilt - context).abs().max().item()
    variants = []
    for edge in range(e):
        mask = valid.clone()
        mask[:, edge] = False
        # Sole-edge removal is undefined; exclude it with NaN below.
        mask[~mask.any(-1), edge] = True
        variants.append(project(scores.masked_fill(~mask[:, None, None], -torch.inf).softmax(-1)))
    variants.append(torch.zeros_like(context))
    uniform = valid[:, None, None].to(weights.dtype) / valid.sum(-1)[:, None, None, None]
    variants.append(project(uniform.expand_as(weights)))
    for head in range(h):
        a = weights.clone()
        a[:, head] = 0
        variants.append(project(a))
    contexts = torch.stack(variants, 0)
    original = action_head(torch.cat((humans, context), -1))
    changed = action_head(torch.cat((humans.expand(len(variants), -1, -1, -1), contexts), -1))
    raw = ((changed - original[None]) ** 2).mean(-1).sqrt()
    clipped = ((changed.clamp(-1, 1) - original[None].clamp(-1, 1)) ** 2).mean(-1).sqrt()
    defined = valid & (valid.sum(-1, keepdim=True) > 1)
    raw[:e] = raw[:e].masked_fill(~defined.T[:, :, None], float('nan'))
    clipped[:e] = clipped[:e].masked_fill(~defined.T[:, :, None], float('nan'))
    entropy = -(weights * weights.clamp_min(1e-30).log()).sum(-1)
    entropy = entropy / valid.sum(-1).float().log().clamp_min(1e-12)[:, None, None]
    return dict(attention=weights, entropy=entropy,
                edge_delta_raw=raw[:e].permute(1, 2, 0),
                edge_delta=clipped[:e].permute(1, 2, 0),
                ca_off_delta=clipped[e], uniform_delta=clipped[e + 1],
                head_delta=clipped[e + 2:].permute(1, 0, 2),
                reconstruction_error=error)
