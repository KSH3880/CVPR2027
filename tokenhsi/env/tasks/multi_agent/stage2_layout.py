import torch

from utils.before_goal_sampling import route_distances, sample_layout


def resample_before_references(task, pending, saved):
    ids = pending[task._scene_domains[pending] != 1]
    if not len(ids):
        return saved
    task._reset_ref_slots, task._reset_ref_motion_ids, task._reset_ref_motion_times = {}, {}, {}
    task._reset_actors(ids)
    fresh = (task._reset_ref_slots, task._reset_ref_motion_ids, task._reset_ref_motion_times)
    merged = ({}, {}, {})
    for skill in saved[0].keys() | fresh[0].keys():
        if skill in saved[0]:
            keep = ~torch.isin(saved[0][skill][0], ids)
            merged[0][skill] = tuple(x[keep] for x in saved[0][skill])
            merged[1][skill], merged[2][skill] = saved[1][skill][keep], saved[2][skill][keep]
        if skill in fresh[0]:
            if skill not in merged[0]:
                for index in range(3):
                    merged[index][skill] = fresh[index][skill]
            else:
                merged[0][skill] = tuple(torch.cat((a, b)) for a, b in zip(merged[0][skill], fresh[0][skill]))
                for index in (1, 2):
                    merged[index][skill] = torch.cat((merged[index][skill], fresh[index][skill]))
    return merged


def pair_values(task, ids):
    rows, follower = (task._before_predecessor[ids] >= 0).nonzero(as_tuple=True)
    env = ids[rows]
    leader = task._before_predecessor[env, follower]
    source = task._scenario_physical_object(env, leader)
    goal = task._scenario_goal(env, leader)
    return rows, env, follower, leader, source, goal


def apply_before_layout(task, ids):
    _, env, follower, leader, source, goal_index = pair_values(task, ids)
    if not len(env):
        return
    roots = task._humanoid_root_states
    goal = task._tar_pos[env, goal_index, :2]
    leader_route = ((roots[env, leader, :2]-task._box_states[env, source, :2]).norm(dim=-1)
                    +(task._box_states[env, source, :2]-goal).norm(dim=-1))
    desired, payload = sample_layout(goal, leader_route, task._templates[env, follower],
                                    task._before_near_pairs[env, follower])
    delta = desired-roots[env, follower, :2]
    roots[env, follower, :2] += delta
    task._kinematic_humanoid_rigid_body_states[env, follower, :, :2] += delta[:, None]
    top = task._templates[env, follower] == 4
    box = task._scenario_physical_object(env[top], follower[top])
    task._box_states[env[top], box, :2] = payload[top]


def invalid_before_layout(task, ids):
    rows, env, follower, leader, source, goal_index = pair_values(task, ids)
    payload = task._scenario_physical_object(env, follower)
    goal = task._tar_pos[env, goal_index, :2]
    roots = task._humanoid_root_states
    first, second = route_distances(roots[env, leader, :2], task._box_states[env, source, :2],
        goal, roots[env, follower, :2], task._box_states[env, payload, :2], task._templates[env, follower])
    failed = task._before_near_pairs[env, follower] & (second+.0005 >= first)
    failed |= (roots[env, follower, :2]-goal).norm(dim=-1) < 1.
    bad = torch.zeros(len(ids), dtype=torch.bool, device=ids.device)
    bad[rows[failed]] = True
    return bad
