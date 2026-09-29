"""Per-environment first-episode timeout staggering for Carry training."""

from __future__ import annotations

import torch


def initial_timeout_deadlines(num_envs: int, episode_length: int,
                              device: torch.device) -> torch.Tensor:
    """Sample first timeout over the full episode, without changing task age.

    Later episodes use the ordinary full-length timeout. A minimum of two
    steps avoids a zero-length first episode and preserves fall detection.
    """
    if num_envs < 1 or episode_length < 3:
        raise ValueError("num_envs must be positive and episode_length >= 3")
    return torch.randint(
        3, episode_length + 1, (num_envs,), device=device,
        dtype=torch.long,
    )


__all__ = ["initial_timeout_deadlines"]
