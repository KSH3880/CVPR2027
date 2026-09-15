"""Pure execution-time views of the planner's single end-to-end path."""

from __future__ import annotations

import torch

from coordinator.schema import CoordinatorState

from .constraints import (
    ordered_box_goal_visit, project_points_to_segments, retreat_box_clearance,
)


def execution_view(
    path: torch.Tensor,
    box_xy: torch.Tensor,
    goal_xy: torch.Tensor,
    retreat: torch.Tensor,
) -> torch.Tensor:
    """Interpolate carry prefix or attach retreat suffix without moving its end.

    The planner itself always emits the same complete path.  This function is
    only the frozen Carry compatibility boundary.  It finds the goal visit
    ordered after the box visit without assigning either target a point index.
    """
    if path.ndim != 4 or path.shape[1] != 2 or path.shape[-1] != 2:
        raise ValueError("expected selected model path [B,2,P,2]")
    expected = path.shape[:2] + (2,)
    if box_xy.shape != expected or goal_xy.shape != expected:
        raise ValueError("box/goal must match [B,2,2]")
    if retreat.shape != path.shape[:2] or retreat.dtype != torch.bool:
        raise ValueError("retreat must be bool [B,2]")

    visit = ordered_box_goal_visit(path, box_xy, goal_xy, tolerance=0.0)
    goal_projection = project_points_to_segments(path, goal_xy)
    segment = visit["goal_segment"]
    fraction = goal_projection["fraction"].gather(
        -1, segment[..., None]
    ).squeeze(-1)

    lengths = (path[..., 1:, :] - path[..., :-1, :]).norm(dim=-1)
    cumulative = torch.cat((
        torch.zeros_like(lengths[..., :1]), lengths.cumsum(dim=-1),
    ), dim=-1)
    goal_arc = (
        cumulative.gather(-1, segment[..., None]).squeeze(-1)
        + fraction * lengths.gather(-1, segment[..., None]).squeeze(-1)
    )
    end_arc = cumulative[..., -1]
    unit = torch.linspace(
        0.0, 1.0, path.shape[-2], device=path.device, dtype=path.dtype,
    )
    start = torch.where(retreat, goal_arc, torch.zeros_like(goal_arc))
    end = torch.where(retreat, end_arc, goal_arc)
    query = start[..., None] + unit * (end - start)[..., None]

    flat_arc = cumulative.reshape(-1, cumulative.shape[-1]).contiguous()
    flat_query = query.reshape(-1, query.shape[-1]).contiguous()
    lower = (torch.searchsorted(flat_arc, flat_query, right=True) - 1).clamp(
        0, path.shape[-2] - 2
    )
    ds = flat_arc.gather(1, lower + 1) - flat_arc.gather(1, lower)
    t = (
        (flat_query - flat_arc.gather(1, lower)) / ds.clamp(min=1e-8)
    ).clamp(0, 1)
    values = path.reshape(-1, path.shape[-2], 2)
    row = torch.arange(values.shape[0], device=path.device)[:, None]
    sampled = values[row, lower] + t[..., None] * (
        values[row, lower + 1] - values[row, lower]
    )
    sampled = sampled.reshape_as(path)

    # Attach the first suffix segment to the current root. All remaining
    # suffix samples stay in their predicted world coordinates: translating
    # the whole suffix also moved the virtual box, sometimes onto the root.
    sampled[..., 0, :] = path[..., 0, :]
    sampled[..., -1, :] = torch.where(
        retreat[..., None], path[..., -1, :], goal_xy,
    )
    return sampled


def retreat_box_geometry(
    path: torch.Tensor,
    state: CoordinatorState,
    env_mask: torch.Tensor,
):
    """Measure committed A1 retreat paths against their real bottom boxes."""
    if path.ndim != 4 or path.shape[:2] != (state.batch_size, 2):
        raise ValueError("path must be [B,2,P,2]")
    if env_mask.shape != (state.batch_size,) or env_mask.dtype != torch.bool:
        raise ValueError("env_mask must be bool [B]")
    return retreat_box_clearance(
        path[env_mask, 0],
        state.box_xyz[env_mask, 0, :2],
        state.box_heading[env_mask, 0],
        state.box_size_xy[env_mask, 0],
    )


__all__ = ["execution_view", "retreat_box_geometry"]
