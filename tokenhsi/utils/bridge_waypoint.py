import torch

from utils.before_object_alias import goal_aligned_scene


def bridge_goal(bridge, size, payload_size):
    x, y, z, w = bridge[:, 3:7].unbind(-1)
    vertical = torch.stack((2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1)
    goal = bridge[:, :3].clone()
    goal[:, 2] += (vertical.abs()*size/2).sum(-1) + payload_size[:, 2]/2
    return goal


def update_switch(payload_xy, goal_xy, payload_size, margin, switched):
    distance = (payload_xy-goal_xy).norm(dim=-1)
    radius = payload_size[:, :2].norm(dim=-1)/2 + margin
    return switched | (distance <= radius), distance, radius


def waypoint_scene(scene, leader, goal_local, num_agents=4, num_objects=8, active=None):
    # Resolve the follower alias against the original goal before moving the leader Target.
    result = goal_aligned_scene(scene, num_agents, num_objects)
    end = result.shape[1] - 5*num_agents
    poses = result[:, end-7*(2*num_agents+num_objects):end].reshape(len(scene), -1, 7)
    rows = torch.arange(len(scene), device=scene.device)
    targets = num_agents+num_objects+leader
    if active is not None:
        goal_local = torch.where(active[:, None], goal_local, poses[rows, targets, :3])
    poses[rows, targets, :3] = goal_local
    return result


def staged_progress(distance, radius, switched, final_progress, reference, sigma):
    approach = 1 / (1 + (distance-radius).clamp_min(0)/sigma)
    approach = torch.where(switched, 1., approach)
    remaining = 1-reference
    final = ((final_progress-reference)/remaining.clamp_min(1e-6)).clamp(0, 1)
    final = torch.where(remaining <= 1e-6, (final_progress >= 1).float(), final)
    return approach, torch.where(switched, final, 0.)
