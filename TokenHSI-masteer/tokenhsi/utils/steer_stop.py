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
