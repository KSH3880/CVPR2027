import torch
import torch.nn.functional as F


def canonical_pose(pose):
    q = F.normalize(pose[..., 3:7], dim=-1)
    q = torch.where(q[..., 3:] < 0, -q, q)
    return torch.cat((pose[..., :3], q), -1)


def relative_pose(reference, target):
    reference, target = canonical_pose(reference), canonical_pose(target)
    inverse = torch.cat((-reference[..., 3:6], reference[..., 6:7]), -1)
    delta = target[..., :3] - reference[..., :3]
    cross = torch.cross(inverse[..., :3], delta, dim=-1)
    position = delta + 2 * (inverse[..., 3:] * cross
                            + torch.cross(inverse[..., :3], cross, dim=-1))
    q = target[..., 3:7]
    vector = inverse[..., 3:] * q[..., :3] + q[..., 3:] * inverse[..., :3]
    vector = vector + torch.cross(inverse[..., :3], q[..., :3], dim=-1)
    scalar = inverse[..., 3:] * q[..., 3:] - (inverse[..., :3] * q[..., :3]).sum(-1, keepdim=True)
    return canonical_pose(torch.cat((position, vector, scalar), -1))
