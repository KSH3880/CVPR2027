"""Synthetic carried-box crossing states for the first C5 teacher oracle."""

import math

import torch

from coordinator.geometry import rotate_xy
from coordinator.schema import CoordinatorState


def sample_crossing_states(count, generator=None):
    if count <= 0:
        raise ValueError("count must be positive")
    center = 2.0 * torch.rand(count, 2, generator=generator) - 1.0
    yaw = 2.0 * math.pi * torch.rand(count, generator=generator) - math.pi
    crossing = math.pi / 3.0 + math.pi / 3.0 * torch.rand(
        count, generator=generator
    )
    directions = torch.stack(
        (
            torch.stack((torch.ones_like(yaw), torch.zeros_like(yaw)), dim=-1),
            torch.stack((torch.cos(crossing), torch.sin(crossing)), dim=-1),
        ),
        dim=1,
    )
    directions = rotate_xy(directions, yaw[:, None])
    shared_distance = 2.0 + 2.0 * torch.rand(count, 1, generator=generator)
    distance_jitter = 0.2 * torch.rand(count, 2, generator=generator) - 0.1
    distance = shared_distance + distance_jitter
    root = center[:, None] - distance[..., None] * directions
    goal = center[:, None] + distance[..., None] * directions
    heading = torch.atan2(directions[..., 1], directions[..., 0])
    size = 0.4 + 0.3 * torch.rand(count, 2, 2, generator=generator)
    return CoordinatorState(
        root_xy=root,
        heading=heading,
        root_vel_xy=torch.zeros(count, 2, 2),
        box_xyz=torch.cat((root, torch.full((count, 2, 1), 0.8)), dim=-1),
        box_heading=heading.clone(),
        box_vel_xy=torch.zeros(count, 2, 2),
        box_size_xy=size,
        goal_xy=goal,
        held=torch.ones(count, 2),
        phase=torch.full((count, 2), 2.0),
    )
