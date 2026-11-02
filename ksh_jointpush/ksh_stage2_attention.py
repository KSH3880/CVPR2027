"""Read-only canonical replay of actor human-coordination attention."""
from copy import deepcopy
import math

import torch
import torch.nn.functional as F


def mha_components(module, query, key, mask=None, include_raw=False):
    """Scaled QK, bias, probabilities; optionally append unscaled raw QK."""
    assert module.batch_first and module._qkv_same_embed_dim and module.dropout == 0
    batch, count, width = query.shape
    heads = module.num_heads
    depth = width // heads
    bias = module.in_proj_bias
    q = F.linear(query, module.in_proj_weight[:width], None if bias is None else bias[:width])
    k = F.linear(key, module.in_proj_weight[width:2*width], None if bias is None else bias[width:2*width])
    q = q.reshape(batch, count, heads, depth).transpose(1, 2)
    k = k.reshape(batch, key.shape[1], heads, depth).transpose(1, 2)
    scores = (q / math.sqrt(depth)) @ k.transpose(-1, -2)
    if mask is None:
        additive = torch.zeros_like(scores)
    else:
        additive = mask.reshape(batch, heads, count, key.shape[1]) if mask.ndim == 3 else mask[None, None].expand_as(scores)
        if additive.dtype == torch.bool:
            additive = torch.zeros_like(scores).masked_fill(additive, float('-inf'))
    result = (scores, additive, (scores + additive).softmax(-1))
    return result + (q @ k.transpose(-1, -2),) if include_raw else result


@torch.no_grad()
def canonical_coordination_maps(coordination, humans, packet, groups):
    """Replay real canonical inputs on a copy; leave rollout order/RNG untouched.

    Returns batch, layer, head, query agent, key agent. The recorded original
    coordination output can be compared with `output` to validate equivariance.
    """
    branch = deepcopy(coordination).eval()
    branch.shuffle_human_order = False
    records = []
    errors = []
    hooks = []

    def observe(module, args, kwargs):
        query, key, value = args[:3]
        assert kwargs.get('key_padding_mask') is None
        mask = kwargs.get('attn_mask')
        scores, additive, probability, raw = mha_components(module, query, key, mask, include_raw=True)
        # Calling forward directly bypasses module hooks and uses PyTorch's
        # independent per-head weight return for numerical validation.
        _, reference = module.forward(query, key, value, attn_mask=mask,
                                      need_weights=True, average_attn_weights=False)
        errors.append(float((probability-reference).abs().max()))
        records.append(tuple(x.detach().clone() for x in (scores, additive, probability, raw)))

    for block in branch.blocks:
        hooks.append(block.attention.register_forward_pre_hook(observe, with_kwargs=True))
    try:
        output = branch(humans, None, packet, None, groups=groups)
    finally:
        for hook in hooks:
            hook.remove()
    assert len(records) == len(branch.blocks)
    assert max(errors) < 2e-6, errors
    result = {name: torch.stack([row[i] for row in records], 1)
              for i, name in enumerate(('qk_scores', 'relation_bias', 'attention', 'raw_qk_scores'))}
    result['output'] = output
    result['mha_reference_max_error'] = max(errors)
    return result
