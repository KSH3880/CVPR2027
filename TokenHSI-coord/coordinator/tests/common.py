from __future__ import annotations

import torch

from coordinator.schema import CoordinatorState


def make_state(batch: int = 2, held: bool = False) -> CoordinatorState:
    root = torch.tensor([[[-4.0, 0.0], [0.0, -4.0]]]).repeat(batch, 1, 1)
    box_xy = root.clone() if held else torch.tensor([[[-2.5, 0.0], [0.0, -2.5]]]).repeat(batch, 1, 1)
    box = torch.cat((box_xy, torch.full((batch, 2, 1), 0.8 if held else 0.3)), dim=-1)
    return CoordinatorState(
        root_xy=root,
        heading=torch.zeros(batch, 2),
        root_vel_xy=torch.tensor([[[0.8, 0.0], [0.0, 0.8]]]).repeat(batch, 1, 1),
        box_xyz=box,
        box_heading=torch.zeros(batch, 2),
        box_vel_xy=torch.zeros(batch, 2, 2),
        box_size_xy=torch.full((batch, 2, 2), 0.5),
        goal_xy=torch.tensor([[[4.0, 0.0], [0.0, 4.0]]]).repeat(batch, 1, 1),
        held=torch.full((batch, 2), float(held)),
        phase=torch.full((batch, 2), 2.0 if held else 0.0),
    )
