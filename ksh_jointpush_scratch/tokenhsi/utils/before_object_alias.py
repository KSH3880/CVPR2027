"""Two role tokens for one physical BEFORE support, within a fixed object budget.

The environment keeps the original task packet for rewards and human relations.
Only the scene encoder receives the packet with follower targets redirected.
"""
import torch


MODE = 'replace_distractor_v1'


def alias_layout(packet, num_agents, num_objects):
    """Return dependent records, original targets, alias targets and task count.

    Canonical task objects occupy a compact prefix. Aliases replace the following
    distractors in actor order, independent of task-record or edge permutation.
    """
    valid, task, actor, payload, target = packet.reshape(len(packet), -1, 5).long().unbind(-1)
    valid = valid.bool()
    dependency = (valid[:, :, None] & valid[:, None, :]
        & (task[:, :, None] == 2)
        & ((task[:, None, :] == 0) | (task[:, None, :] == 1) | (task[:, None, :] == 3))
        & (payload[:, :, None] == target[:, None, :])
        & (actor[:, :, None] != actor[:, None, :]))
    dependent = dependency.any(1)
    objects = torch.stack((payload, target), -1)
    used = torch.stack((valid & (task >= 2), valid), -1)
    used &= (objects >= num_agents) & (objects < num_agents + num_objects)
    slots = (objects - num_agents).clamp(0, num_objects - 1).flatten(1)
    counts = torch.zeros(len(packet), num_objects, device=packet.device, dtype=torch.long)
    counts.scatter_add_(1, slots, used.long().flatten(1))
    task_count = (counts > 0).sum(-1)
    rank = (dependent[:, None, :] & (actor[:, None, :] < actor[:, :, None])).sum(-1)
    alias = num_agents + task_count[:, None] + rank
    return dependent, target, alias, task_count


def replace_distractors(mapping, packet, num_agents):
    """Replace physical mappings, preserving all canonical task-token indices."""
    dependent, target, alias, count = alias_layout(packet, num_agents, mapping.shape[1])
    total = count + dependent.sum(-1)
    if (total > mapping.shape[1]).any():
        raise ValueError('BEFORE object aliases exceed the object token budget')
    row, record = dependent.nonzero(as_tuple=True)
    result = mapping.clone()
    result[row, alias[row, record] - num_agents] = mapping[row, target[row, record] - num_agents]
    return result, total


def encoder_packet(packet, num_agents, num_objects):
    """Redirect follower targets only; retain actor task conditioning exactly once."""
    dependent, target, alias, _ = alias_layout(packet, num_agents, num_objects)
    result = packet.reshape(len(packet), -1, 5).clone()
    result[..., 4] = torch.where(dependent, alias, target).to(result.dtype)
    return result.reshape_as(packet)
