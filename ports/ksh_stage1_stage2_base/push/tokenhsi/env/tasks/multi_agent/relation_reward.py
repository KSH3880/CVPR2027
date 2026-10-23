"""Graph-bound object speed penalty used by Stage1."""

import torch


def box_speed_penalty(previous, current, dt, coefficient=1.0, threshold=2.5):
    speed = torch.linalg.vector_norm((current - previous) / dt, dim=-1)
    return -coefficient * (
        1 - torch.exp(-2 * (speed.clamp_min(threshold) - threshold).square())
    )
