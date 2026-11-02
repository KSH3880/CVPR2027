import torch
from utils.before_goal_sampling import route_distances, sample_layout


def pair_values(task, ids):
    row, follower = (task._before_predecessor[ids] >= 0).nonzero(as_tuple=True)
    env = ids[row]
    leader = task._before_predecessor[env, follower]
    source = task._scenario_physical_object(env, leader)
    goal = task._scenario_goal(env, leader)
    return row, env, follower, leader, source, goal


def apply_layout(task, ids):
    _, env, follower, leader, source, goal_index = pair_values(task, ids)
    if not len(env):
        return
    goal = task._tar_pos[env, goal_index, :2]
    roots = task._humanoid_root_states
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


def invalid(task, ids, margin=0.):
    row, env, follower, leader, source, goal_index = pair_values(task, ids)
    payload = task._scenario_physical_object(env, follower)
    goal = task._tar_pos[env, goal_index, :2]
    root = task._humanoid_root_states[env, follower, :2]
    first, second = route_distances(task._humanoid_root_states[env, leader, :2],
        task._box_states[env, source, :2], goal, root,
        task._box_states[env, payload, :2], task._templates[env, follower])
    failed = task._before_near_pairs[env, follower] & (second+margin >= first)
    failed |= (root-goal).norm(dim=-1) < 1.
    bad = torch.zeros(len(ids), dtype=torch.bool, device=ids.device)
    bad[row[failed]] = True
    return bad


def static_infeasible(task, ids):
    from env.tasks.multi_agent.scene_validation import scene_infeasible
    mapping = task._logical_box_order[ids]
    slots = torch.arange(task.num_objects, device=ids.device)[None] < task._task_token_count[ids, None]
    occupied = torch.zeros(len(ids), task.num_physical_objects, dtype=torch.long, device=ids.device)
    occupied.scatter_add_(1, mapping, slots.long())
    active = occupied > 0
    templates = task._templates[ids]
    row, owner = (templates == 5).nonzero(as_tuple=True)
    active[row, 4*task.num_agents+owner] = False
    row, owner = ((templates == 4) & (task._before_predecessor[ids] < 0)).nonzero(as_tuple=True)
    active[row, 3*task.num_agents+owner] = False
    return scene_infeasible(task, ids, margin=.001, active=active) | invalid(task, ids, margin=.0005)
