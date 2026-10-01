"""Logical assignment changes without writing simulator state."""

from contextlib import contextmanager
import torch


def assignment_rows(assignment):
    if assignment.ndim != 2 or assignment.shape[1] != 2 or assignment.dtype != torch.long:
        raise ValueError("assignment must be long [envs,2]")
    if not torch.equal(assignment.sort(-1).values, torch.tensor([0, 1], device=assignment.device).expand_as(assignment)):
        raise ValueError("each environment must assign each box exactly once")
    return (2 * torch.arange(len(assignment), device=assignment.device)[:, None] + assignment).flatten()


@contextmanager
def assigned_inputs(task, goals_follow_box=True):
    """Present gathered, read-only task inputs; restore every physical alias."""
    index = assignment_rows(task.box_assignment)
    physical = task._box_states
    goals, previous = task._box_tar_pos, task._prev_box_pos
    size, bps = task._box_lib._box_size, task._box_lib._box_bps
    try:
        task._box_states = task.humanoid_rows(physical)[index].reshape(task.num_envs, 2, -1)
        task._box_lib._box_size = size[index]
        task._box_lib._box_bps = bps[index]
        task._prev_box_pos = previous[index]
        if goals_follow_box:
            task._box_tar_pos = goals[index]
        yield
    finally:
        task._box_states, task._box_tar_pos, task._prev_box_pos = physical, goals, previous
        task._box_lib._box_size, task._box_lib._box_bps = size, bps
