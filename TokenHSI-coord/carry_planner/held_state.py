"""Observed Carry pickup state, excluding elevated but untouched boxes."""

from __future__ import annotations

import torch


def observed_box_held(
    root_xy: torch.Tensor,
    box_xyz: torch.Tensor,
    box_size_z: torch.Tensor,
    hands_xyz: torch.Tensor,
) -> torch.Tensor:
    """Return [B,2] grasp proxy from the same live geometry as the executor."""
    if root_xy.shape != box_xyz.shape[:-1] + (2,):
        raise ValueError("root_xy must match box_xyz batch and agent axes")
    if hands_xyz.shape != box_xyz.shape or box_size_z.shape != box_xyz.shape[:-1]:
        raise ValueError("hands_xyz/box_size_z shape mismatch")
    lifted = box_xyz[..., 2] > box_size_z * 0.5 + 0.2
    root_near = (root_xy - box_xyz[..., :2]).norm(dim=-1) <= 0.7
    hand_near = (hands_xyz - box_xyz).norm(dim=-1) <= 0.25
    return lifted & root_near & hand_near


__all__ = ["observed_box_held"]
