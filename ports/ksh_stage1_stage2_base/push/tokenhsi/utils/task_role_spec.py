"""Stage1 task packets and canonical actor/payload/target connections."""

import torch
from utils.edge_ontop_spec import batched
from utils.size_rsi import SIZE_RANGES, task_probabilities

STAGE1_VARIANT = "stage1"
STAGE1_V2_VARIANT = "stage1_v2"
STAGE1_VARIANTS = (STAGE1_VARIANT, STAGE1_V2_VARIANT)
LEGACY_TASKS = ("sit", "climb", "carry_at", "carry_ontop")
TASKS = LEGACY_TASKS + ("push_at",)
TASK_FIELDS = ("valid", "task", "actor", "payload", "target")
TASK_PROBS = (0.0, 0.10, 0.25, 0.325, 0.325)


def task_preset(preset):
    if preset == "holding":
        raise ValueError("Stage1 has no standalone holding task")
    return {"carry_at": "holding_at", "carry_ontop": "holding_ontop"}.get(
        preset, preset
    )


def task_size_probabilities(sizes, ranges=SIZE_RANGES):
    # All three carry templates have identical size distributions. Their total
    # prior stays .65, preserving the asset distribution and existing RSI cache.
    probs = task_probabilities(sizes, ranges)
    probs[..., 3:] += probs[..., :1] / 2
    probs[..., 0] = 0
    return probs


def task_graph_packet(graph, batch_size):
    valid, src, dst, relation, owner = [
        batched(getattr(graph, key), batch_size)
        for key in ("edge_valid", "edge_src", "edge_dst", "edge_relation", "edge_owner")
    ]
    owned = valid[:, None] & (
        owner[:, None]
        == torch.arange(graph.num_agents, device=src.device)[None, :, None]
    )
    primary = owned & (
        (relation[:, None] == 6) | (relation[:, None] == 9) | (relation[:, None] == 10)
        | (relation[:, None] == 11)
    )
    placement = owned & ((relation[:, None] == 7) | (relation[:, None] == 8))
    actor = (src[:, None] * primary).sum(-1)
    primary_target = (dst[:, None] * primary).sum(-1)
    carry = placement.any(-1)
    target = torch.where(carry, (dst[:, None] * placement).sum(-1), primary_target)
    payload = torch.where(carry, primary_target, 0)
    task = torch.where(
        carry,
        (relation[:, None] * placement).sum(-1) - 5,
        (relation[:, None] * primary).sum(-1) - 9,
    )
    task = torch.where((owned & (relation[:, None] == 11)).any(-1), 4, task)
    return (
        torch.stack((primary.any(-1), task, actor, payload, target), -1)
        .float()
        .flatten(1)
    )


def task_edges(packet):
    valid, task, actor, payload, target = (
        packet.reshape(packet.shape[0], -1, 5).long().unbind(-1)
    )
    carry = task >= 2
    active = torch.stack((carry, valid.bool(), carry), -1) & valid.bool()[..., None]
    src = torch.stack((actor, actor, payload), -1)
    dst = torch.stack((payload, target, target), -1)
    task = task[..., None].expand_as(src)
    return tuple(x.flatten(1) for x in (active, src, dst, task))
