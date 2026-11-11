"""Opt-in Stage1 v2 geometry contracts; reference motions are never retargeted."""
from copy import deepcopy
import torch
from push_reference.utils.size_rsi import SCREEN, STAGE1_V2_RANGES

RSI_SPEC = dict(cacheDirectory='output/rsi_cache_stage1_v2', lateProbability=.7,
                lateFraction=.3, screen=deepcopy(SCREEN),
                geometryFilter='body_points_v1', phaseMode='sampled_box')


def max_bound_stack_height(sizes):
    """Canonical four-object graph: only payload i -> support i+2 can stack.

    A tall Climb-only asset cannot become a carry payload. Also check each
    standalone box so a large object is never hidden by the binding bound.
    """
    if sizes.shape[-2:] != (4, 3):
        raise ValueError('Stage1 v2 requires four canonically bound objects')
    bounds = torch.tensor(STAGE1_V2_RANGES['HOLDING_ON_TOP'], device=sizes.device)
    payload = sizes[..., :2, :]
    eligible = ((payload >= bounds[:, 0]-1e-6) & (payload <= bounds[:, 1]+1e-6)).all(-1)
    stack = (sizes[..., :2, 2]+sizes[..., 2:, 2])*eligible
    return torch.maximum(sizes[..., 2].amax(-1), stack.amax(-1))


def pose_geometry_valid(body, feet, boxes, sizes):
    """Reject deep joint-center penetration and below-floor feet, not task failure.

    This conservative geometric screen supplements physical screening; it is
    not an exact mesh-intersection test. Prefix is rows, boxes is rows x K x 13.
    """
    relative = body[:, None]-boxes[:, :, None, :3]
    quat = boxes[:, :, None, 3:7]
    quat = quat/quat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    xyz = quat[..., :3].expand_as(relative)
    cross = 2*torch.cross(xyz, relative, dim=-1)
    local = relative-quat[..., 3:4]*cross+torch.cross(xyz, cross, dim=-1)
    penetration = (local.abs() < sizes[:, :, None]/2-.06).all(-1).any(-1).any(-1)
    return (~penetration & (feet[..., 2].amin(-1) >= -.02)
            & torch.isfinite(body).all(-1).all(-1))


def sampled_phase(skill, roots, feet, box_xy, sizes):
    """One phase mask per actual grounded SIT/CLIMB size, not origin-Z metadata."""
    near = (roots[None, :, :2]-box_xy[None, None]).norm(dim=-1) <= sizes[:, None, :2].norm(dim=-1)/2+.3
    top = sizes[:, None, 2]
    if skill == 'sit':
        return near & (roots[None, :, 2] <= top+.35)
    if skill == 'climb':
        return near & (feet[None, :, :, 2].mean(-1) >= top-.10)
    raise ValueError('Sampled interaction phase requires SIT or CLIMB')
