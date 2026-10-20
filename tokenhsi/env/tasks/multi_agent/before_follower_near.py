import torch

from utils.before_layout import transform
from utils.shared_support import rotate as quat_rotate


def pair_invalid(roots, bodies, boxes, sizes, stack):
    bad = (roots[:, 0, :2]-roots[:, 1, :2]).norm(dim=-1) < .8
    relative = bodies[:, :, None, :, :3]-boxes[:, None, :, None, :3]
    q = boxes[:, None, :, None, 3:7].expand(*relative.shape[:-1], 4).clone()
    q[..., :3] *= -1
    inside = (quat_rotate(q, relative).abs() < sizes[:, None, :, None]/2+.03).all(-1).any(-1)
    active = torch.stack((torch.ones_like(stack), stack), -1)
    bad |= (inside & active[:, None]).flatten(1).any(-1)
    separation = (boxes[:, 0, :2]-boxes[:, 1, :2]).norm(dim=-1)
    bad |= stack & (separation < sizes[..., :2].norm(dim=-1).sum(-1)/2+.1)
    return bad


def apply_layout(t, ids):
    row, follower = (t._before_predecessor[ids] >= 0).nonzero(as_tuple=True)
    if not len(row):
        return
    envs = ids[row]
    first = t._before_predecessor[envs, follower]
    humans = torch.stack((first, follower), -1)
    slots = (envs[:, None], humans)
    roots = t._humanoid_root_states[slots].clone()
    bodies = t._kinematic_humanoid_rigid_body_states[slots].clone()
    boxes = t._box_states[slots].clone()
    sizes = t._box_size[slots]
    stack = t._templates[envs, follower] == 4
    pending = torch.arange(len(envs), device=t.device)
    for _ in range(128):
        distance = (1.+torch.rand(len(pending), 2, device=t.device)).sort(-1, descending=True).values
        angle = torch.rand(len(pending), 2, device=t.device)*2*torch.pi
        target = roots[pending].clone()
        target[..., :2] = boxes[pending, 0, None, :2]+distance[..., None]*torch.stack((angle.cos(), angle.sin()), -1)
        yaw = angle+torch.pi
        yaw[:, 1] = torch.where(stack[pending], torch.rand(len(pending), device=t.device)*2*torch.pi, yaw[:, 1])
        x, y, z, w = roots[pending, ..., 3:7].unbind(-1)
        yaw -= torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        q = torch.zeros_like(roots[pending, ..., 3:7])
        q[..., 2], q[..., 3] = (yaw/2).sin(), (yaw/2).cos()
        shift = target[..., :3]-quat_rotate(q, roots[pending, ..., :3])
        moved_roots = transform(roots[pending], q, shift)
        moved_bodies = transform(bodies[pending], q, shift)
        moved_boxes = boxes[pending].clone()
        moved_boxes[:, 1] = transform(moved_boxes[:, 1], q[:, 1], shift[:, 1])
        bad = pair_invalid(moved_roots, moved_bodies, moved_boxes, sizes[pending], stack[pending])
        actual = (moved_roots[..., :2]-moved_boxes[:, 0, None, :2]).norm(dim=-1)
        bad |= actual[:, 1] >= actual[:, 0]
        accepted = pending[~bad]
        dest = (envs[accepted, None], humans[accepted])
        t._humanoid_root_states[dest] = moved_roots[~bad]
        t._kinematic_humanoid_rigid_body_states[dest] = moved_bodies[~bad]
        top = ~bad & stack[pending]
        t._box_states[envs[pending[top]], follower[pending[top]]] = moved_boxes[top, 1]
        pending = pending[bad]
        if not len(pending):
            return
    raise RuntimeError('BEFORE follower-near placement failed after 128 attempts')
