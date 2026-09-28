"""Remaining-suffix spline decoder for plain simultaneous Carry."""

from __future__ import annotations

from typing import Tuple

import torch

from .schema import (
    CARRY_LEARNED_PATH_POINTS, CARRY_PATH_KNOTS, MAX_SPEED, MIN_SPEED,
    STACK_PATH_POINTS,
)


def _hermite(knots: torch.Tensor, subdivisions: int) -> torch.Tensor:
    tangent = torch.empty_like(knots)
    tangent[..., 0, :] = knots[..., 1, :] - knots[..., 0, :]
    tangent[..., -1, :] = knots[..., -1, :] - knots[..., -2, :]
    if knots.shape[-2] > 2:
        tangent[..., 1:-1, :] = 0.5 * (
            knots[..., 2:, :] - knots[..., :-2, :]
        )
    t = torch.linspace(
        0.0, 1.0, subdivisions + 1,
        device=knots.device, dtype=knots.dtype,
    )
    shape = (1,) * (knots.ndim - 2) + (1, subdivisions + 1, 1)
    t = t.reshape(shape)
    t2, t3 = t.square(), t.square() * t
    p0, p1 = knots[..., :-1, None, :], knots[..., 1:, None, :]
    m0, m1 = tangent[..., :-1, None, :], tangent[..., 1:, None, :]
    segments = (
        (2.0 * t3 - 3.0 * t2 + 1.0) * p0
        + (t3 - 2.0 * t2 + t) * m0
        + (-2.0 * t3 + 3.0 * t2) * p1
        + (t3 - t2) * m1
    )
    return torch.cat((
        segments[..., :-1, :].flatten(start_dim=-3, end_dim=-2),
        segments[..., -1, -1:, :],
    ), dim=-2)


def _monotone(knots: torch.Tensor, subdivisions: int) -> torch.Tensor:
    slope = knots[..., 1:] - knots[..., :-1]
    tangent = torch.empty_like(knots)
    tangent[..., 0] = slope[..., 0]
    tangent[..., -1] = slope[..., -1]
    if knots.shape[-1] > 2:
        left, right = slope[..., :-1], slope[..., 1:]
        same = left * right > 0.0
        denominator = left + right
        safe = torch.where(
            denominator.abs() > 1e-7, denominator,
            torch.ones_like(denominator),
        )
        harmonic = 2.0 * left * right / safe
        tangent[..., 1:-1] = torch.where(
            same, harmonic, torch.zeros_like(harmonic),
        )
    t = torch.linspace(
        0.0, 1.0, subdivisions + 1,
        device=knots.device, dtype=knots.dtype,
    )
    shape = (1,) * (knots.ndim - 1) + (1, subdivisions + 1)
    t = t.reshape(shape)
    t2, t3 = t.square(), t.square() * t
    p0, p1 = knots[..., :-1, None], knots[..., 1:, None]
    m0, m1 = tangent[..., :-1, None], tangent[..., 1:, None]
    segments = (
        (2.0 * t3 - 3.0 * t2 + 1.0) * p0
        + (t3 - 2.0 * t2 + t) * m0
        + (-2.0 * t3 + 3.0 * t2) * p1
        + (t3 - t2) * m1
    )
    return torch.cat((
        segments[..., :-1].flatten(start_dim=-2, end_dim=-1),
        segments[..., -1, -1:],
    ), dim=-1)


def _resample_at(
    curve: torch.Tensor, value: torch.Tensor, target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    length = (curve[..., 1:, :] - curve[..., :-1, :]).norm(dim=-1)
    arc = torch.cat((torch.zeros_like(length[..., :1]), length.cumsum(-1)), -1)
    upper = torch.searchsorted(
        arc.contiguous(), target.contiguous(), right=True,
    ).clamp(1, curve.shape[-2] - 1)
    lower = upper - 1
    arc0, arc1 = arc.gather(-1, lower), arc.gather(-1, upper)
    blend = ((target - arc0) / (arc1 - arc0).clamp(min=1e-7)).clamp(0.0, 1.0)
    lower_xy = lower[..., None].expand(*lower.shape, 2)
    upper_xy = upper[..., None].expand(*upper.shape, 2)
    point0 = curve.gather(-2, lower_xy)
    point1 = curve.gather(-2, upper_xy)
    sampled = point0 + blend[..., None] * (point1 - point0)
    value0, value1 = value.gather(-1, lower), value.gather(-1, upper)
    sampled_value = value0 + blend * (value1 - value0)
    sampled[..., 0, :] = curve[..., 0, :]
    sampled[..., -1, :] = curve[..., -1, :]
    sampled_value[..., 0] = value[..., 0]
    sampled_value[..., -1] = value[..., -1]
    return sampled, sampled_value


def _uniform_targets(curve: torch.Tensor) -> torch.Tensor:
    length = (curve[..., 1:, :] - curve[..., :-1, :]).norm(dim=-1)
    total = length.sum(dim=-1, keepdim=True)
    fraction = torch.linspace(
        0.0, 1.0, STACK_PATH_POINTS,
        device=curve.device, dtype=curve.dtype,
    )
    return total * fraction


def decode_carry_suffix(
    root: torch.Tensor,
    box: torch.Tensor,
    goal: torch.Tensor,
    held: torch.Tensor,
    path_raw: torch.Tensor,
    speed_raw: torch.Tensor,
    control_scale: float,
):
    """Decode current-state-anchored remaining paths.

    Inputs use [B,C,A,...]. The returned box index is dynamic for unheld
    agents and -1 once pickup has completed.
    """
    if path_raw.shape[-2:] != (CARRY_LEARNED_PATH_POINTS, 2):
        raise ValueError("path_raw must end in [4,2]")
    if speed_raw.shape != path_raw.shape[:-2] + (CARRY_PATH_KNOTS,):
        raise ValueError("speed_raw must end in [7]")
    if root.shape != path_raw.shape[:-2] + (2,):
        raise ValueError("root/path shape mismatch")
    if box.shape != root.shape or goal.shape != root.shape:
        raise ValueError("root/box/goal shape mismatch")
    if held.shape != root.shape[:-1]:
        raise ValueError("held shape mismatch")

    raw = torch.tanh(path_raw)
    speed_knots = MIN_SPEED + (MAX_SPEED - MIN_SPEED) * torch.sigmoid(speed_raw)

    approach_delta = box - root
    carry_delta = goal - box
    direct_delta = goal - root
    approach_scale = torch.minimum(
        approach_delta.norm(dim=-1), path_raw.new_full((), control_scale),
    )
    carry_scale = torch.minimum(
        carry_delta.norm(dim=-1), path_raw.new_full((), control_scale),
    )
    direct_scale = torch.minimum(
        direct_delta.norm(dim=-1), path_raw.new_full((), control_scale),
    )

    approach_mid = root + 0.5 * approach_delta
    carry_fraction = path_raw.new_tensor((0.25, 0.50, 0.75))
    carry_base = box[..., None, :] + (
        carry_fraction.reshape((1,) * (box.ndim - 1) + (3, 1))
        * carry_delta[..., None, :]
    )
    pre_learned = torch.cat((
        approach_mid[..., None, :]
        + approach_scale[..., None, None] * raw[..., :1, :],
        carry_base + carry_scale[..., None, None] * raw[..., 1:, :],
    ), dim=-2)
    pre_control = torch.cat((
        root[..., None, :], pre_learned[..., :1, :], box[..., None, :],
        pre_learned[..., 1:, :], goal[..., None, :],
    ), dim=-2)

    approach_curve = _hermite(pre_control[..., :3, :], 32)
    approach_speed = _monotone(speed_knots[..., :3], 32)
    carry_curve = _hermite(pre_control[..., 2:, :], 16)
    carry_speed = _monotone(speed_knots[..., 2:], 16)
    combined_curve = torch.cat((approach_curve, carry_curve[..., 1:, :]), -2)
    combined_speed = torch.cat((approach_speed, carry_speed[..., 1:]), -1)
    approach_length = (
        approach_curve[..., 1:, :] - approach_curve[..., :-1, :]
    ).norm(dim=-1).sum(dim=-1)
    total_length = (
        combined_curve[..., 1:, :] - combined_curve[..., :-1, :]
    ).norm(dim=-1).sum(dim=-1)
    target = _uniform_targets(combined_curve)
    box_index = torch.round(
        (STACK_PATH_POINTS - 1) * approach_length
        / total_length.clamp(min=1e-7)
    ).long().clamp(0, STACK_PATH_POINTS - 1)
    target = target.scatter(-1, box_index[..., None], approach_length[..., None])
    pre_path, pre_speed = _resample_at(combined_curve, combined_speed, target)
    pre_path.scatter_(
        -2, box_index[..., None, None].expand(*box_index.shape, 1, 2),
        box[..., None, :],
    )

    direct_fraction = path_raw.new_tensor((0.20, 0.40, 0.60, 0.80))
    direct_base = root[..., None, :] + (
        direct_fraction.reshape((1,) * (root.ndim - 1) + (4, 1))
        * direct_delta[..., None, :]
    )
    direct_learned = (
        direct_base + direct_scale[..., None, None] * raw
    )
    direct_control = torch.cat((
        root[..., None, :], direct_learned, goal[..., None, :],
    ), dim=-2)
    direct_speed_knots = speed_knots[..., (0, 1, 3, 4, 5, 6)]
    direct_curve = _hermite(direct_control, 12)
    direct_speed_curve = _monotone(direct_speed_knots, 12)
    direct_target = _uniform_targets(direct_curve)
    direct_path, direct_speed = _resample_at(
        direct_curve, direct_speed_curve, direct_target,
    )

    use_direct = held >= 0.5
    path = torch.where(use_direct[..., None, None], direct_path, pre_path)
    speed = torch.where(use_direct[..., None], direct_speed, pre_speed)
    dynamic_box_index = torch.where(
        use_direct, torch.full_like(box_index, -1), box_index,
    )
    path[..., 0, :] = root
    path[..., -1, :] = goal
    return path, speed, dynamic_box_index


__all__ = ["decode_carry_suffix"]
