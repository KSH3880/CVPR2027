"""Continuous, queryable Carry suffix with no fixed control-point allocation."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .carry_suffix import _resample_at, _uniform_targets
from .schema import (
    CARRY_LEARNED_PATH_POINTS, CARRY_PATH_KNOTS, MAX_SPEED, MIN_SPEED,
    STACK_PATH_POINTS,
)

IMPLICIT_FEATURES = CARRY_LEARNED_PATH_POINTS * 2 + CARRY_PATH_KNOTS + 2 + 1 + 3
IMPLICIT_COEFFICIENTS = 2 * CARRY_LEARNED_PATH_POINTS + CARRY_PATH_KNOTS


def decode_carry_implicit_suffix(root, box, goal, held, path_raw, speed_raw,
                                 control_scale: float, coefficient_net):
    """Query one learned continuous curve per task leg, then resample to 33.

    The policy's four XY action pairs are latent variables, not points tied to
    fixed positions. A shared learned network converts them and scene geometry
    into smooth sine/cosine coefficients for each leg. Exact root, box and goal
    anchors are imposed after evaluating the continuous curves.
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
    if control_scale <= 0:
        raise ValueError("control_scale must be positive")

    unit = torch.linspace(0.0, 1.0, 33, device=root.device, dtype=root.dtype)
    order = torch.arange(
        1, CARRY_LEARNED_PATH_POINTS + 1, device=root.device, dtype=root.dtype,
    )
    sine = torch.sin(math.pi * order[:, None] * unit[None])
    speed_order = torch.arange(
        CARRY_PATH_KNOTS, device=root.device, dtype=root.dtype,
    )
    cosine = torch.cos(math.pi * speed_order[:, None] * unit[None])
    latent = torch.cat((path_raw.flatten(start_dim=-2), speed_raw), dim=-1)

    def leg(start, end, phase):
        delta = end - start
        length = delta.norm(dim=-1, keepdim=True)
        direction = delta / length.clamp(min=1e-6)
        phase_code = F.one_hot(
            torch.full_like(held, phase, dtype=torch.long), num_classes=3,
        ).to(root.dtype)
        features = torch.cat((
            latent, direction, length / control_scale, phase_code,
        ), dim=-1)
        coefficients = coefficient_net(features)
        xy_coefficients = coefficients[..., :8].reshape(
            *coefficients.shape[:-1], 2, CARRY_LEARNED_PATH_POINTS,
        )
        displacement = torch.einsum("...ck,kn->...nc", xy_coefficients, sine)
        baseline = start[..., None, :] + unit[None, None, None, :, None] * delta[..., None, :]
        curve = baseline + control_scale * torch.tanh(displacement)
        curve[..., 0, :] = start
        curve[..., -1, :] = end
        speed_coefficients = coefficients[..., 8:]
        speed_logits = (
            speed_raw[..., :1]
            + torch.einsum("...k,kn->...n", speed_coefficients, cosine)
        )
        speed = MIN_SPEED + (MAX_SPEED - MIN_SPEED) * torch.sigmoid(speed_logits)
        return curve, speed

    approach, approach_speed = leg(root, box, 0)
    carry, carry_speed = leg(box, goal, 1)
    pre_curve = torch.cat((approach, carry[..., 1:, :]), dim=-2)
    pre_speed = torch.cat((approach_speed, carry_speed[..., 1:]), dim=-1)
    approach_length = (approach[..., 1:, :] - approach[..., :-1, :]).norm(dim=-1).sum(-1)
    total_length = (pre_curve[..., 1:, :] - pre_curve[..., :-1, :]).norm(dim=-1).sum(-1)
    box_index = torch.round(
        (STACK_PATH_POINTS - 1) * approach_length
        / total_length.clamp(min=1e-7)
    ).long().clamp(0, STACK_PATH_POINTS - 1)
    target = _uniform_targets(pre_curve)
    target = target.scatter(-1, box_index[..., None], approach_length[..., None])
    pre_path, pre_velocity = _resample_at(pre_curve, pre_speed, target)
    pre_path.scatter_(
        -2, box_index[..., None, None].expand(*box_index.shape, 1, 2),
        box[..., None, :],
    )

    direct_curve, direct_speed = leg(root, goal, 2)
    direct_path, direct_velocity = _resample_at(
        direct_curve, direct_speed, _uniform_targets(direct_curve),
    )
    use_direct = held >= 0.5
    path = torch.where(use_direct[..., None, None], direct_path, pre_path)
    velocity = torch.where(use_direct[..., None], direct_velocity, pre_velocity)
    dynamic_box_index = torch.where(
        use_direct, torch.full_like(box_index, -1), box_index,
    )
    path[..., 0, :] = root
    path[..., -1, :] = goal
    return path, velocity, dynamic_box_index


__all__ = [
    "IMPLICIT_FEATURES", "IMPLICIT_COEFFICIENTS",
    "decode_carry_implicit_suffix",
]
