import torch
from utils.shared_support import rotate


def transform(states, q, shift):
    while q.ndim < states.ndim:
        q, shift = q.unsqueeze(-2), shift.unsqueeze(-2)
    result = states.clone()
    result[..., :3] = rotate(q, states[..., :3])+shift
    b = states[..., 3:7]
    result[..., 3:7] = torch.cat((q[..., 3:]*b[..., :3]+b[..., 3:]*q[..., :3]
        +torch.cross(q[..., :3].expand_as(b[..., :3]), b[..., :3], dim=-1),
        q[..., 3:]*b[..., 3:]-(q[..., :3]*b[..., :3]).sum(-1, keepdim=True)), -1)
    result[..., 7:10] = rotate(q, states[..., 7:10])
    result[..., 10:13] = rotate(q, states[..., 10:13])
    return result
