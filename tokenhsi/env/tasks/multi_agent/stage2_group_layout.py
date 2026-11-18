import math

import torch

from utils.stage2_spec import before_members


def translate_people(task, ids):
    groups = task._groups[ids]
    same = groups[:, :, None] == groups[:, None, :]
    centers = torch.bmm(same.float(), task._humanoid_root_states[ids, :, :2])/same.sum(-1, keepdim=True)
    leader = groups == torch.arange(task.num_agents, device=task.device)[None]
    count = leader.sum(-1)
    order = torch.rand(groups.shape, device=task.device).masked_fill(~leader, float('inf'))
    rank = order.argsort(-1).argsort(-1).gather(1, groups)
    angle = torch.rand(len(ids), 1, device=task.device)*2*math.pi+2*math.pi*rank/count[:, None]
    radius = torch.where(count > 1, 5./(2*torch.sin(math.pi/count)), 0.)
    targets = task._env_origins[ids, None, :2]+radius[:, None, None]*torch.stack((angle.cos(), angle.sin()), -1)
    task._group_spawn_targets[ids] = targets
    move_people(task, ids, targets-centers)


def move_people(task, ids, delta):
    task._humanoid_root_states[ids, :, :2] += delta
    task._kinematic_humanoid_rigid_body_states[ids, :, :, :2] += delta[:, :, None]


def recenter_before(task, ids):
    member = before_members(task._before_predecessor[ids])
    groups = task._groups[ids]
    same = groups[:, :, None] == groups[:, None, :]
    centers = torch.bmm(same.float(), task._humanoid_root_states[ids, :, :2])/same.sum(-1, keepdim=True)
    delta = (task._group_spawn_targets[ids]-centers)*member[..., None]
    move_people(task, ids, delta)
    task._tar_pos[ids, :, :2] += delta
    physical = task._agent_box_assignment[ids]
    count = torch.zeros(len(ids), task.num_physical_objects, device=task.device)
    count.scatter_add_(1, physical, member.float())
    shift = torch.zeros(len(ids), task.num_physical_objects, 2, device=task.device)
    shift.scatter_add_(1, physical[..., None].expand(-1, -1, 2), delta)
    task._box_states[ids, :, :2] += shift/count.clamp_min(1)[..., None]


def park_unused(task, ids):
    mapping = task._logical_box_order[ids]
    retained = torch.zeros(len(ids), task.num_physical_objects, dtype=torch.long, device=task.device)
    retained.scatter_add_(1, mapping, task._stage2_task_slots[ids].long())
    row, physical = (retained == 0).nonzero(as_tuple=True)
    env = ids[row]
    task._box_states[env, physical] = 0.
    task._box_states[env, physical, :2] = task._env_origins[env, :2]+1000.
    task._box_states[env, physical, 0] += 12*physical
    task._box_states[env, physical, 2] = task._box_size[env, physical, 2]/2
    task._box_states[env, physical, 6] = 1.


def clean_observation(task, obs, ids):
    valid = task._stage2_task_slots[ids]
    m, o = task.num_agents, task.num_objects
    start = m*task.get_clean_humanoid_obs_size()
    width = task.get_clean_object_obs_size()
    obs[:, start:start+o*width].view(-1, o, width).masked_fill_(~valid[..., None], 0.)
    pose_start = start+o*width+m*task.get_clean_goal_obs_size()
    poses = obs[:, pose_start:pose_start+(2*m+o)*7].view(-1, 2*m+o, 7)[:, m:m+o]
    poses.masked_fill_(~valid[..., None], 0.)
    poses[..., 6].masked_fill_(~valid, 1.)
    return obs


def camera(task):
    from isaacgym import gymapi
    valid = task._stage2_task_slots[0]
    boxes = task._logical_box_values(task._box_states)[0, valid, :3]
    points = torch.cat((task._humanoid_root_states[0, :, :3], boxes, task._tar_pos[0]), 0)
    low, high = points.amin(0), points.amax(0)
    center = (low+high)/2
    center[2] = max(float(center[2]), .8)
    radius = max(float((high-low)[:2].norm())*.8, 3.)
    if task._camera_follow_flag:
        task.gym.viewer_camera_look_at(task.viewer, None,
            gymapi.Vec3(float(center[0])+radius*.65, float(center[1])-radius, 2.8+radius*.5),
            gymapi.Vec3(*center.tolist()))
    task._cam_prev_char_pos[:] = task._humanoid_root_states[0, 0, :3].cpu().numpy()
