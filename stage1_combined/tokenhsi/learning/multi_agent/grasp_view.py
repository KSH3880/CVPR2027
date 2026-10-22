import torch
from utils.grasp_geometry import grasp_y


GRASP_VIEW = {
    "assignment": "nearest_opposite_at_reset",
    "payload_center": "holding_anchor",
    "size": "physical",
    "velocity": "rigid_point",
}


def joint_rows(packet):
    tasks = packet.reshape(-1, 2, 5)
    return (
        (tasks[..., 0] > 0).all(-1)
        & (tasks[..., 1] >= 2).all(-1)
        & (tasks[:, 0, 1] == tasks[:, 1, 1])
        & (tasks[:, 0, 3:] == tasks[:, 1, 3:]).all(-1)
    )


def make_grasp_views(obs, sides, inset):
    if sides.shape[1] != 2 or obs.shape[-1] != 634:
        return group_grasp_views(obs, sides, inset, sides.shape[1], (obs.shape[-1]-243*sides.shape[1])//37)
    views = obs[:, None].expand(-1, 2, -1).clone()
    n = len(obs)
    tasks = obs[:, -10:].reshape(n, 2, 5)
    payload = (tasks[:, 0, 3].long() - 2).clamp(0, 3)
    rows = torch.arange(n, device=obs.device)[:, None]
    owners = torch.arange(2, device=obs.device)[None]
    objects = views[:, :, 446:566].reshape(n, 2, 4, 30)
    poses = views[:, :, 568:624].reshape(n, 2, 8, 7)
    box = objects[rows, owners, payload[:, None]]
    pose = poses[rows, owners, payload[:, None] + 2]
    half_y = box[..., 6:].reshape(n, 2, 8, 3)[..., 1].abs().amax(-1)
    delta = torch.zeros_like(pose[..., :3])
    delta[..., 1] = (half_y - inset).clamp_min(0) * sides * joint_rows(tasks)[:, None]
    q = pose[..., 3:7]
    cross = 2 * torch.cross(q[..., :3], delta, dim=-1)
    pose[..., :3] += delta + q[..., 3:] * cross + torch.cross(q[..., :3], cross, dim=-1)
    box[..., :3] += torch.cross(box[..., 3:6], delta, dim=-1)
    poses[rows, owners, payload[:, None] + 2] = pose
    objects[rows, owners, payload[:, None]] = box
    return views.flatten(1)


def encode_grasp_views(encoder, obs):
    if encoder.num_agents != 2 or encoder.num_objects != 4:
        m = encoder.num_agents
        humans = encoder(obs.reshape(-1, obs.shape[-1]//m)).reshape(len(obs), m, m, -1)
        ids = torch.arange(m, device=obs.device)
        return humans[:, ids, ids]
    views = obs.reshape(len(obs), 2, -1)
    joint = joint_rows(views[:, 0, -10:])
    paired = joint.nonzero().flatten()
    solo = (~joint).nonzero().flatten()
    indices = torch.cat((solo * 2, paired * 2, paired * 2 + 1))
    humans = encoder(views.flatten(0, 1)[indices])
    result = humans.new_empty(len(obs), 2, humans.shape[-1])
    result[solo] = humans[: len(solo)]
    result[paired, 0] = humans[len(solo) : len(solo) + len(paired), 0]
    result[paired, 1] = humans[len(solo) + len(paired) :, 1]
    return result


def initial_grasp_sides(obs, num_agents=2, num_objects=4):
    if (num_agents, num_objects) != (2, 4):
        return group_initial_sides(obs, num_agents, num_objects)
    poses = obs[:, 568:624].reshape(-1, 8, 7)
    payload = obs[:, -10:].reshape(-1, 2, 5)[:, 0, 3].long()
    box = poses[torch.arange(len(obs), device=obs.device), payload]
    relative = poses[:, :2, :3] - box[:, None, :3]
    q = box[:, None, 3:7].expand(-1, 2, -1).clone()
    q[..., :3] *= -1
    cross = 2 * torch.cross(q[..., :3], relative, dim=-1)
    local = relative + q[..., 3:] * cross + torch.cross(q[..., :3], cross, dim=-1)
    first = torch.where(local[:, 0, 1] <= local[:, 1, 1], -1.0, 1.0)
    return torch.stack((first, -first), -1)


def group_grasp_views(obs, sides, inset, m, o):
    from utils.stage2_spec import owner_tasks
    from learning.multi_agent.human_coordination import human_relations
    tasks = owner_tasks(obs[:, -5*m:], m)
    coupled = (human_relations(obs[:, -5*m:], m) == 2).any(-1)
    views = obs[:, None].expand(-1, m, -1).clone()
    start = m*223
    pose_start = start + o*30 + m
    objects = views[:, :, start:start+o*30].reshape(len(obs), m, o, 30)
    poses = views[:, :, pose_start:pose_start+(2*m+o)*7].reshape(len(obs), m, 2*m+o, 7)
    row = torch.arange(len(obs), device=obs.device)[:, None]
    human = torch.arange(m, device=obs.device)[None]
    payload = tasks[..., 3].long()
    box = objects[row, human, payload-m]
    pose = poses[row, human, payload]
    half_y = box[..., 6:].reshape(len(obs), m, 8, 3)[..., 1].abs().amax(-1)
    delta = torch.zeros_like(pose[..., :3])
    delta[..., 1] = grasp_y(half_y, sides, inset)*coupled
    q = pose[..., 3:7]
    cross = 2*torch.cross(q[..., :3], delta, dim=-1)
    pose[..., :3] += delta + q[..., 3:]*cross + torch.cross(q[..., :3], cross, dim=-1)
    box[..., :3] += torch.cross(box[..., 3:6], delta, dim=-1)
    poses[row, human, payload] = pose
    objects[row, human, payload-m] = box
    return views.flatten(1)


def group_initial_sides(obs, m, o):
    from utils.stage2_spec import owner_tasks
    from learning.multi_agent.human_coordination import human_relations
    start = m*223+o*30+m
    poses = obs[:, start:start+(2*m+o)*7].reshape(len(obs), 2*m+o, 7)
    tasks = owner_tasks(obs[:, -5*m:], m)
    coupled = human_relations(obs[:, -5*m:], m) == 2
    box = poses[torch.arange(len(obs), device=obs.device)[:, None], tasks[..., 3].long()]
    relative = poses[:, :m, :3] - box[..., :3]
    q = box[..., 3:7].clone()
    q[..., :3] *= -1
    cross = 2*torch.cross(q[..., :3], relative, dim=-1)
    y = (relative + q[..., 3:]*cross + torch.cross(q[..., :3], cross, dim=-1))[..., 1]
    partner = (coupled*y[:, None]).sum(-1)
    ids = torch.arange(m, device=obs.device)[None]
    partner_id = (coupled*ids[:, None]).sum(-1)
    negative = (y < partner) | ((y == partner) & (ids < partner_id))
    return torch.where(negative, -1., 1.)*coupled.any(-1)
