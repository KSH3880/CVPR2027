"""Random PUSH approaches measured from full collider bounds to box surfaces."""
import math
import torch
from utils.door_reference import rotate

MODE = 'random_azimuth_surface_clearance_v1'


def bounds_clearance(lo, hi, sizes, xy_only=True):
    """Conservative minimum distance over all sphere/capsule/mesh bounds."""
    axes = 2 if xy_only else 3
    half = sizes[:, None, :axes]/2
    delta = torch.maximum(lo[..., :axes]-half, -half-hi[..., :axes]).clamp_min(0)
    return delta.norm(dim=-1).amin(-1)


def body_clearance(geometry, body, rotations, boxes, sizes, xy_only=True):
    lo, hi = geometry.bounds(body, rotations, boxes)
    return bounds_clearance(lo, hi, sizes, xy_only)


def place_at_surface_gap(geometry, relative, rotations, boxes, sizes, root_z,
                         bearings, desired_gap):
    """Rigid placement only: uniform azimuth, specified full-body surface gap.

    Bounds include every collider and complete foot mesh. Binary search moves
    the whole person; no joint, velocity or reference posture is modified.
    """
    root = boxes[:, :3].clone()
    root[:, 2] = root_z
    body = relative + root[:, None]
    body[..., 2] += geometry.floor_shift(body, rotations)[:, None]
    lo, hi = geometry.bounds(body, rotations, boxes)
    direction = torch.stack((bearings.cos(), bearings.sin(), torch.zeros_like(bearings)), -1)
    inverse = boxes[:, 3:7].clone()
    inverse[:, :3] *= -1
    local_direction = rotate(inverse, direction)
    radius = torch.maximum(lo.abs(), hi.abs())[..., :2].norm(dim=-1).amax(-1)
    lower = torch.zeros_like(desired_gap)
    upper = sizes[:, :2].norm(dim=-1)/2+radius+desired_gap+1.
    for _ in range(28):
        middle = (lower+upper)/2
        delta = local_direction[:, None]*middle[:, None, None]
        gap = bounds_clearance(lo+delta, hi+delta, sizes)
        near = gap < desired_gap
        lower = torch.where(near, middle, lower)
        upper = torch.where(near, upper, middle)
    body = body+direction[:, None]*upper[:, None, None]
    gap = body_clearance(geometry, body, rotations, boxes, sizes)
    if (not torch.isfinite(body).all() or (gap < desired_gap-.002).any()
            or (gap > desired_gap+.002).any()):
        raise RuntimeError('Cannot place PUSH at the requested collider surface gap')
    return body


def random_approach(geometry, relative, rotations, boxes, sizes, root_z, spec):
    n = len(boxes)
    bearing = torch.rand(n, device=boxes.device)*2*math.pi
    lower, upper = spec['loco_surface_clearance_range_m']
    gap = lower+(upper-lower)*torch.rand(n, device=boxes.device)
    return place_at_surface_gap(geometry, relative, rotations, boxes, sizes, root_z, bearing, gap)


def unsafe_initial_rows(geometry, poses, boxes, sizes, minimum_gap):
    """Reject contact with any physical box, including unassigned distractors."""
    unsafe = ~torch.isfinite(poses).flatten(1).all(-1)
    for object_id in range(boxes.shape[1]):
        clearance = body_clearance(geometry, poses[..., :3], poses[..., 3:7],
                                   boxes[:, object_id], sizes[:, object_id], xy_only=False)
        unsafe |= clearance < minimum_gap
        unsafe |= geometry.invalid(poses[..., :3], poses[..., 3:7], boxes[:, object_id], sizes[:, object_id])
    return unsafe
