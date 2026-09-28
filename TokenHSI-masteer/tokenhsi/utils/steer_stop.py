"""Pure helpers for flag-free stop commands encoded by steering geometry."""

import torch


def cosine_transition(elapsed, steps, rising=True):
    """Return a clamped cosine transition in ``[0, 1]``.

    ``rising=True`` contracts a normal steering window into a hold anchor.
    ``rising=False`` expands it again. This value is controller state only;
    the policy observes the resulting points, never the transition value.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    phase = (elapsed.float() / float(steps)).clamp(0.0, 1.0)
    rising_value = 0.5 * (1.0 - torch.cos(torch.pi * phase))
    return rising_value if rising else 1.0 - rising_value


def contract_to_anchor(points, anchor, contraction):
    """Contract ``K`` world-space points toward one fixed world anchor."""
    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError("points must have shape (N, K, 2)")
    if anchor.shape != (points.shape[0], 2):
        raise ValueError("anchor must have shape (N, 2)")
    if contraction.shape != (points.shape[0],):
        raise ValueError("contraction must have shape (N,)")
    weight = contraction[:, None, None].to(dtype=points.dtype)
    return points * (1.0 - weight) + anchor[:, None, :] * weight


def stop_aware_speed_reward(command, progress_speed, planar_velocity, gain=4.0):
    """Track positive progress while making physical stillness optimal at stop.

    The positive-progress gate is useful for locomotion but is discontinuous at
    ``command == 0``.  Stop commands therefore use planar physical speed and do
    not pass through that gate.
    """
    if command.shape != progress_speed.shape:
        raise ValueError("command and progress_speed must have the same shape")
    if planar_velocity.shape != command.shape + (2,):
        raise ValueError("planar_velocity must have shape command.shape + (2,)")
    tracking = torch.exp(-gain * (command - progress_speed).square())
    tracking = torch.where(progress_speed > 0.0, tracking, torch.zeros_like(tracking))
    stillness = torch.exp(-gain * planar_velocity.square().sum(dim=-1))
    return torch.where(command < 0.05, stillness, tracking)


def blend_goal_to_anchor(final_goal, anchor_xy, root_height, contraction):
    """Blend a goal toward a fixed stop anchor without adding a flag channel."""
    if final_goal.ndim != 2 or final_goal.shape[-1] != 3:
        raise ValueError("final_goal must have shape (N, 3)")
    if anchor_xy.shape != (final_goal.shape[0], 2):
        raise ValueError("anchor_xy must have shape (N, 2)")
    if root_height.shape != (final_goal.shape[0],):
        raise ValueError("root_height must have shape (N,)")
    if contraction.shape != (final_goal.shape[0],):
        raise ValueError("contraction must have shape (N,)")
    target = torch.cat([anchor_xy, root_height[:, None]], dim=-1)
    source = final_goal.clone()
    source[:, 2] = root_height
    weight = contraction[:, None].to(dtype=final_goal.dtype)
    return source * (1.0 - weight) + target * weight
