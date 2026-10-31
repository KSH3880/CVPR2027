import torch

from utils.jointpush_geometry import yaw_from_quat


def group_face_geometry(hands, boxes, sizes, center_goals, groups, members, regions, tolerance=.18):
    yaw = yaw_from_quat(boxes[..., 3:7])
    c, s = yaw.cos(), yaw.sin()
    local_normals = boxes.new_tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    local_tangents = boxes.new_tensor([[0., 1.], [0., 1.], [1., 0.], [1., 0.]])
    normals = torch.stack((c[..., None]*local_normals[:, 0]-s[..., None]*local_normals[:, 1],
                           s[..., None]*local_normals[:, 0]+c[..., None]*local_normals[:, 1]), -1)
    tangents = torch.stack((c[..., None]*local_tangents[:, 0]-s[..., None]*local_tangents[:, 1],
                            s[..., None]*local_tangents[:, 0]+c[..., None]*local_tangents[:, 1]), -1)
    half = sizes[..., [0, 0, 1, 1]]/2
    widths = sizes[..., [1, 1, 0, 0]]/2
    centers = boxes[..., None, :2]-normals*half[..., None]
    slot_half = widths/members[..., None]
    slot_offset = (2*regions[..., None]+1)*slot_half-widths
    slot_centers = centers+tangents*slot_offset[..., None]
    offset = hands[..., None, :, :2]-slot_centers[..., None, :]
    along = (offset*tangents[..., None, :]).sum(-1)
    across = (offset*normals[..., None, :]).sum(-1)
    over = (along.abs()-slot_half[..., None]).clamp_min(0)
    worst = (over.square()+across.square()).amax(-1)
    same = groups[:, :, None] == groups[:, None, :]
    group_worst = worst[:, None].masked_fill(~same[..., None], -torch.inf).amax(2)
    delta = center_goals[..., :2]-boxes[..., :2]
    direction = delta/delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    direction = torch.where((delta.norm(dim=-1) > 1e-6)[..., None], direction, normals[..., 2, :])
    allowed = (normals*direction[..., None, :]).sum(-1) >= .5-1e-6
    face = group_worst.masked_fill(~allowed, torch.inf).argmin(-1)

    def select(values):
        return values.gather(-2, face[..., None, None].expand(*face.shape, 1, 2)).squeeze(-2)

    normal, tangent, center = select(normals), select(tangents), select(centers)
    width = widths.gather(-1, face[..., None]).squeeze(-1)
    offset = hands[..., :2]-center[..., None, :]
    lateral = ((offset*tangent[..., None, :]).sum(-1).abs()-width[..., None]).clamp_min(0)
    depth = (offset*normal[..., None, :]).sum(-1)
    vertical = ((hands[..., 2]-boxes[..., None, 2]).abs()-sizes[..., None, 2]/2).clamp_min(0)
    face_distance = (lateral.square()+depth.square()+vertical.square()).sqrt()
    behind = ((hands[..., :2]-boxes[..., None, :2])*normal[..., None, :]).sum(-1) <= 0
    on_face = ((face_distance <= tolerance) & behind).all(-1)
    distance = worst.gather(-1, face[..., None]).squeeze(-1).sqrt()
    return dict(face=face, distance=distance, normal=normal, on_face=on_face,
                slot_center=select(slot_centers), slot_half=slot_half.gather(-1, face[..., None]).squeeze(-1))


def individual_goal_gate(on_face, tokens, goals, groups, goal_saturation=True,
                         success_distance=.2, z_tolerance=.001):
    same = groups[:, :, None] == groups[:, None, :]
    gate = (on_face[:, None, :] | ~same).all(-1)
    delta = goals-tokens[..., :3]
    arrived = (delta.norm(dim=-1) <= success_distance) & (delta[..., 2].abs() <= z_tolerance)
    if goal_saturation:
        gate |= arrived
    return torch.where(same.sum(-1) > 1, gate, True)
