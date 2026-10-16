import torch


def shift_payload_observation(obs, num_agents, num_objects, fraction):
    if not torch.as_tensor(fraction).abs().le(1).all():
        raise ValueError('SHIFT must be between -1 and 1 (fraction of box half-width)')
    m, o = num_agents, num_objects
    node_end = m * 223 + o * 30 + m
    pose_end = node_end + (2 * m + o) * 7
    if obs.shape[-1] != pose_end + m * 5:
        raise ValueError('Expected Stage1 scene223/object30/pose7/task5 observation')
    if torch.as_tensor(fraction).eq(0).all():
        return obs
    result = obs.clone()
    objects = result[..., m * 223:m * 223 + o * 30].reshape(-1, o, 30)
    poses = result[..., node_end:pose_end].reshape(-1, 2 * m + o, 7)[:, m:m + o]
    packet = obs[..., pose_end:].reshape(-1, m, 5)
    object_ids = torch.arange(o, device=obs.device) + m
    selected = ((packet[..., 0] > 0) & (packet[..., 1] >= 2))[..., None]
    selected = (selected & (packet[..., 3, None] == object_ids)).any(1)
    half_y = objects[..., 6:].reshape(-1, o, 8, 3)[..., 1].abs().amax(-1)
    delta = torch.zeros_like(objects[..., :3])
    fraction = torch.as_tensor(fraction, device=obs.device, dtype=obs.dtype)
    if fraction.ndim:
        fraction = fraction.reshape(-1, 1)
    delta[..., 1] = fraction * half_y * selected
    q = poses[..., 3:7]
    cross = 2 * torch.cross(q[..., :3], delta, dim=-1)
    world_delta = delta + q[..., 3:] * cross + torch.cross(q[..., :3], cross, dim=-1)
    poses[..., :3] += world_delta
    objects[..., :3] += torch.cross(objects[..., 3:6], delta, dim=-1)
    return result


def shared_payload_views(obs, signs, strength, inset=.15):
    # Two policy views of one physical scene, ordered [scene0 A, scene0 B, ...].
    packet = obs[:, -10:].reshape(-1, 2, 5)
    if not ((packet[..., 0] == 1).all() and (packet[..., 1] >= 2).all()
            and (packet[:, 0, 1] == packet[:, 1, 1]).all()
            and (packet[:, 0, 3:] == packet[:, 1, 3:]).all()):
        raise ValueError('Both humans must share one carry payload and target')
    payload = packet[:, 0, 3].long() - 2
    objects = obs[:, 446:566].reshape(-1, 4, 30)
    half_y = objects[torch.arange(len(obs), device=obs.device), payload, 6:].reshape(-1, 8, 3)[..., 1].abs().amax(-1)
    fraction = strength * (half_y - inset).clamp_min(0) / half_y
    views = obs[:, None].expand(-1, 2, -1).reshape(-1, obs.shape[-1])
    return shift_payload_observation(views, 2, 4, (fraction[:, None] * signs).flatten())


def initial_grasp_sides(obs):
    poses = obs[:, 568:624].reshape(-1, 8, 7)
    payload = obs[:, -10:].reshape(-1, 2, 5)[:, 0, 3].long()
    box = poses[torch.arange(len(obs), device=obs.device), payload]
    relative = poses[:, :2, :3] - box[:, None, :3]
    q = box[:, None, 3:7].expand(-1, 2, -1).clone()
    q[..., :3] *= -1
    cross = 2 * torch.cross(q[..., :3], relative, dim=-1)
    local = relative + q[..., 3:] * cross + torch.cross(q[..., :3], cross, dim=-1)
    first = torch.where(local[:, 0, 1] <= local[:, 1, 1], -1., 1.)
    return torch.stack((first, -first), -1)
