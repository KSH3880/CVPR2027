import torch


GRASP_VIEW = {'assignment': 'nearest_opposite_at_reset', 'payload_center': 'holding_anchor',
              'size': 'physical', 'velocity': 'rigid_point'}


def joint_rows(packet):
    tasks = packet.reshape(-1, 2, 5)
    return ((tasks[..., 0] > 0).all(-1) & (tasks[..., 1] >= 2).all(-1)
            & (tasks[:, 0, 1] == tasks[:, 1, 1])
            & (tasks[:, 0, 3:] == tasks[:, 1, 3:]).all(-1))


def scene_parts(obs, num_agents):
    m = num_agents
    o = (obs.shape[-1] - 243 * m) // 37
    node_end = 224 * m + 30 * o
    objects = obs[..., 223*m:223*m+30*o].reshape(*obs.shape[:-1], o, 30)
    poses = obs[..., node_end:-5*m].reshape(*obs.shape[:-1], 2*m+o, 7)
    return objects, poses


def owner_tasks(obs, num_agents):
    packet = obs[:, -5*num_agents:].reshape(len(obs), num_agents, 5)
    owners = torch.arange(num_agents, device=obs.device)
    matches = packet[..., 0].bool()[..., None] & (packet[..., 2, None] == owners)
    indices = matches.long().argmax(1)
    return packet.gather(1, indices[..., None].expand(-1, -1, 5)) * matches.any(1)[..., None]


def initial_grasp_sides(obs, num_agents=2):
    from learning.multi_agent.human_coordination import human_relations
    m = num_agents
    tasks = owner_tasks(obs, m)
    coupled = human_relations(obs[:, -5*m:], m) == 2
    if (coupled.sum(-1) > 1).any():
        raise ValueError('Opposite grasp anchors require two people per coupled payload')
    _, poses = scene_parts(obs, m)
    payload = tasks[..., 3].long().clamp(m, poses.shape[1] - m - 1)
    box = poses.gather(1, payload[..., None].expand(-1, -1, 7))
    relative = poses[:, :m, :3] - box[..., :3]
    q = box[..., 3:7].clone()
    q[..., :3] *= -1
    cross = 2 * torch.cross(q[..., :3], relative, dim=-1)
    local = relative + q[..., 3:] * cross + torch.cross(q[..., :3], cross, dim=-1)
    partner = coupled.long().argmax(-1)
    y, other_y = local[..., 1], local[..., 1].gather(1, partner)
    first = (y < other_y) | ((y == other_y) & (torch.arange(m, device=obs.device) < partner))
    return torch.where(first, -1., 1.) * coupled.any(-1)


def make_grasp_views(obs, sides, inset, anchor_z=0.):
    from learning.multi_agent.human_coordination import human_relations
    n, m = sides.shape
    views = obs[:, None].expand(-1, m, -1).clone()
    tasks = owner_tasks(obs, m)
    objects, poses = scene_parts(views, m)
    payload = (tasks[..., 3].long() - m).clamp(0, objects.shape[-2] - 1)
    rows = torch.arange(n, device=obs.device)[:, None]
    owners = torch.arange(m, device=obs.device)[None]
    box = objects[rows, owners, payload]
    pose = poses[rows, owners, payload + m]
    half_y = box[..., 6:].reshape(n, m, 8, 3)[..., 1].abs().amax(-1)
    coupled = (human_relations(obs[:, -5*m:], m) == 2).any(-1)
    delta = torch.zeros_like(pose[..., :3])
    delta[..., 1] = (half_y - inset).clamp_min(0) * sides * coupled
    delta[..., 2] = anchor_z * coupled
    q = pose[..., 3:7]
    cross = 2 * torch.cross(q[..., :3], delta, dim=-1)
    pose[..., :3] += delta + q[..., 3:] * cross + torch.cross(q[..., :3], cross, dim=-1)
    box[..., :3] += torch.cross(box[..., 3:6], delta, dim=-1)
    poses[rows, owners, payload + m] = pose
    objects[rows, owners, payload] = box
    return views.flatten(1)


def encode_grasp_views(encoder, obs):
    m = encoder.entity_counts[0]
    views = obs.reshape(len(obs), m, -1)
    if m != 2:
        from learning.multi_agent.human_coordination import human_relations
        coupled = (human_relations(views[:, 0, -5*m:], m) == 2).any(-1)
        owners = torch.arange(m, device=obs.device)[None].expand(len(obs), -1)
        first_solo = (~coupled).long().argmax(-1, keepdim=True)
        indices = torch.arange(len(obs), device=obs.device)[:, None] * m + torch.where(coupled, owners, first_solo)
        unique, inverse = indices.flatten().unique(return_inverse=True)
        humans = encoder(views.flatten(0, 1)[unique])
        return humans[inverse, owners.flatten()].reshape(len(obs), m, -1)
    joint = joint_rows(views[:, 0, -10:])
    paired = joint.nonzero().flatten()
    solo = (~joint).nonzero().flatten()
    indices = torch.cat((solo * 2, paired * 2, paired * 2 + 1))
    humans = encoder(views.flatten(0, 1)[indices])
    result = humans.new_empty(len(obs), 2, humans.shape[-1])
    result[solo] = humans[:len(solo)]
    result[paired, 0] = humans[len(solo):len(solo) + len(paired), 0]
    result[paired, 1] = humans[len(solo) + len(paired):, 1]
    return result
