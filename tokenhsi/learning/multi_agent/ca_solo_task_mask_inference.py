import types

from learning.multi_agent.task_coordination import CoordinationBlock


def block_solo_for_joint(block, humans, tokens, relations, padding):
    joint_queries = (relations == 2).any(-1)
    solo_tasks = ~(relations == 2).any(-2)
    blocked = joint_queries[:, None, :, None] & solo_tasks[:, None, None, :]
    return CoordinationBlock.forward(block, humans, tokens, relations, padding | blocked)


def enable_solo_task_mask(coordination):
    for block in coordination.blocks:
        block.forward = types.MethodType(block_solo_for_joint, block)
