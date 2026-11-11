"""Stage1 26k per-owner CONTACT+AT. Single-humanoid scene adapter."""
import torch
from utils.jointpush_geometry import push_rear_xy, push_face_geometry


def contact(hands, boxes, goals, sizes, hand_force, box_force, spec):
    distance = push_rear_xy(hands, boxes, goals, sizes).amax(-1)
    phi = torch.exp(-spec['near_distance_scale']*distance.square())
    normal = push_face_geometry(hands, boxes, goals, sizes)[1]
    reaction = -(hand_force[..., :2]*normal[..., None, :]).sum(-1)
    force = (box_force[..., :2]*normal).sum(-1)
    hit = ((phi > spec['satisfaction_threshold']) &
           (reaction >= spec['min_hand_force']).all(-1) & (force >= spec['min_box_force']))
    return phi, hit


def reward(root, hands, boxes, goals, sizes, hit, spec):
    distance = (boxes[..., :3]-goals).norm(dim=-1)
    xy = (boxes[..., :2]-goals[..., :2]).norm(dim=-1)
    phi_c = torch.exp(-spec['near_distance_scale']*push_rear_xy(hands, boxes, goals, sizes).amax(-1).square())
    p_c = 1/(1+((root[..., :2]-boxes[..., :2]).norm(dim=-1)-spec['progress_delta']).clamp_min(0)/spec['progress_sigma'])
    phi_a = torch.exp(-10*distance.square())
    p_a = 1/(1+(xy-.1).clamp_min(0))
    # The 26k checkpoint overrides the generic phi >= .9 threshold for AT.
    at = (phi_a >= torch.exp(phi_a.new_tensor(-10*.2**2))) & ((boxes[..., 2]-goals[..., 2]).abs() <= .001)
    saturated = hit | at
    rc = .2*torch.where(saturated, 1., phi_c)+.2*torch.where(saturated, 1., p_c)+.2*saturated.float()
    ra = .2*torch.where(at, 1., phi_a)+.2*torch.where(at, 1., p_a)+.2*at.float()
    return rc+ra, at, xy
