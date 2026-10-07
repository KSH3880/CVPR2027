import torch

from utils.edge_ontop_spec import batched
from utils.size_rsi import SIZE_RSI_VARIANT, task_probabilities

TASK_ROLE_VARIANT = SIZE_RSI_VARIANT + '_task_message'
TASK_ROLE_MLP_VARIANT = SIZE_RSI_VARIANT + '_task_mlp'
TASK_ROLE_SPLIT_MLP_VARIANT = SIZE_RSI_VARIANT + '_task_mlp_split'
TASK_EMBEDDING_VARIANT = SIZE_RSI_VARIANT + '_task_embedding'
TASK_ROLE_VARIANTS = (TASK_ROLE_VARIANT, TASK_ROLE_MLP_VARIANT, TASK_ROLE_SPLIT_MLP_VARIANT, TASK_EMBEDDING_VARIANT)
TASKS = ('sit', 'climb', 'carry_at', 'carry_ontop')
TASK_FIELDS = ('valid', 'task', 'actor', 'payload', 'target')
TASK_PROBS = (0., .10, .25, .325, .325)


def task_preset(preset):
    if preset == 'holding':
        raise ValueError('Task-message experiment has no standalone holding task')
    return {'carry_at': 'holding_at', 'carry_ontop': 'holding_ontop'}.get(preset, preset)


def task_size_probabilities(sizes):
    # All three carry templates have identical size distributions. Their total
    # prior stays .65, preserving the asset distribution and existing RSI cache.
    probs = task_probabilities(sizes)
    probs[..., 3:] += probs[..., :1] / 2
    probs[..., 0] = 0
    return probs


def task_graph_packet(graph, batch_size):
    valid, src, dst, relation, owner = [batched(getattr(graph, key), batch_size)
        for key in ('edge_valid', 'edge_src', 'edge_dst', 'edge_relation', 'edge_owner')]
    owned = valid[:, None] & (owner[:, None] ==
        torch.arange(graph.num_agents, device=src.device)[None, :, None])
    primary = owned & ((relation[:, None] == 6) | (relation[:, None] == 9)
                       | (relation[:, None] == 10))
    placement = owned & ((relation[:, None] == 7) | (relation[:, None] == 8))
    actor = (src[:, None] * primary).sum(-1)
    primary_target = (dst[:, None] * primary).sum(-1)
    carry = placement.any(-1)
    target = torch.where(carry, (dst[:, None] * placement).sum(-1), primary_target)
    payload = torch.where(carry, primary_target, 0)
    task = torch.where(carry, (relation[:, None] * placement).sum(-1) - 5,
                       (relation[:, None] * primary).sum(-1) - 9)
    return torch.stack((primary.any(-1), task, actor, payload, target), -1).float().flatten(1)


def task_edges(packet):
    valid, task, actor, payload, target = packet.reshape(packet.shape[0], -1, 5).long().unbind(-1)
    carry = task >= 2
    active = torch.stack((carry, valid.bool(), carry), -1) & valid.bool()[..., None]
    src = torch.stack((actor, actor, payload), -1)
    dst = torch.stack((payload, target, target), -1)
    src_role = torch.tensor([0, 0, 1], device=packet.device).expand_as(src)
    dst_role = torch.tensor([1, 2, 2], device=packet.device).expand_as(dst)
    task = task[..., None].expand_as(src)
    return tuple(x.flatten(1) for x in (active, src, dst, task, src_role, dst_role))
