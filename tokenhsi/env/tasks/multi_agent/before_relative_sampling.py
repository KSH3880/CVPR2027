import torch
from utils.before_relative_sampling import sample_layout, pair_distances
from utils.before_layout import transform
from utils.shared_support import rotate


def apply_layout(t, ids):
    row, follower = (t._before_predecessor[ids] >= 0).nonzero(as_tuple=True)
    if not len(row):
        return
    es = ids[row]; leader = t._before_predecessor[es, follower]
    first = t._humanoid_root_states[es, leader, :2]
    root = t._humanoid_root_states[es, follower, :2]
    advanced = t._before_carry_mask[es, follower]
    support, desired, payload = sample_layout(first, t._box_states[es, leader, :2],
        root, t._box_states[es, follower, :2], t._templates[es, follower],
        t._before_near_pairs[es, follower], advanced)
    delta = desired-root
    t._box_states[es, leader, :2] = support
    t._humanoid_root_states[es, follower, :2] += delta
    t._kinematic_humanoid_rigid_body_states[es, follower, :, :2] += delta[:, None]
    top = t._templates[es, follower] == 4
    t._box_states[es[top], follower[top], :2] = payload[top]
    ce, ch = es[advanced], follower[advanced]
    if len(ce):
        angle = torch.rand(len(ce), device=t.device)*2*torch.pi
        q = torch.zeros(len(ce), 4, device=t.device)
        q[:, 2], q[:, 3] = torch.sin(angle/2), torch.cos(angle/2)
        root = t._humanoid_root_states[ce, ch]
        shift = root[:, :3]-rotate(q, root[:, :3])
        t._humanoid_root_states[ce, ch] = transform(root, q, shift)
        t._kinematic_humanoid_rigid_body_states[ce, ch] = transform(
            t._kinematic_humanoid_rigid_body_states[ce, ch], q, shift)
        t._box_states[ce, ch] = transform(t._box_states[ce, ch], q, shift)


def invalid(t, ids):
    row, follower = (t._before_near_pairs[ids]).nonzero(as_tuple=True)
    es = ids[row]; leader = t._before_predecessor[es, follower]
    first, second = pair_distances(t._humanoid_root_states[es, leader, :2],
        t._box_states[es, leader, :2], t._humanoid_root_states[es, follower, :2],
        t._box_states[es, follower, :2], t._templates[es, follower])
    bad = torch.zeros(len(ids), device=t.device, dtype=torch.bool)
    bad[row[second >= first]] = True
    return bad
