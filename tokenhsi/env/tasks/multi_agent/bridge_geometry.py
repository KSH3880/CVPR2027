import torch


@torch.jit.script
def compute_agent_cpa_collision_penalty(root_pos, root_vel, min_dist, dt, discount):
    # type: (Tensor, Tensor, float, float, float) -> Tensor
    rel_pos = root_pos[..., :2].unsqueeze(2) - root_pos[..., :2].unsqueeze(1)
    rel_vel = root_vel[..., :2].unsqueeze(2) - root_vel[..., :2].unsqueeze(1)
    dot = torch.sum(rel_pos * rel_vel, dim=-1)
    distance = torch.linalg.norm(rel_pos, dim=-1)
    speed = torch.linalg.norm(rel_vel, dim=-1)
    closing = torch.clamp(-dot / torch.clamp_min(distance * speed, 1e-6), 0.0, 1.0)
    t_cpa = torch.clamp_min(-dot / torch.clamp_min(speed * speed, 1e-6), 0.0)
    cpa_distance = torch.linalg.norm(rel_pos + rel_vel * t_cpa.unsqueeze(-1), dim=-1)
    distance_risk = torch.clamp_min(min_dist - cpa_distance, 0.0) / min_dist
    urgency = torch.pow(torch.full_like(t_cpa, discount), t_cpa / dt)
    risk = closing * distance_risk * urgency
    diagonal = torch.eye(root_pos.shape[1], device=root_pos.device, dtype=torch.bool).unsqueeze(0)
    return risk.masked_fill(diagonal, 0.0).max(dim=-1)[0]


def _uniform(bounds, *shape, device):
    return bounds[0] + (bounds[1] - bounds[0]) * torch.rand(*shape, device=device)


def sample_bridge_layout(half_a, half_b, gap, bridge_half, box_size, cfg, margin,
                         near_probability, agent_distance=1., far_offset=None):
    # Env-local frame: stage A at the origin, stage B across a +x gap, task boxes on stage B.
    n, device = len(gap), gap.device
    need = bridge_half[:, 1] + cfg['landing_margin']
    offset = ((torch.rand(n, device=device) * 2 - 1) * (half_a[:, 1] + half_b[:, 1] - 2 * need)
              if far_offset is None else far_offset)
    overlap_low = torch.maximum(-half_a[:, 1], offset - half_b[:, 1])
    overlap_high = torch.minimum(half_a[:, 1], offset + half_b[:, 1])
    low, high = overlap_low + need, overlap_high - need
    landing = torch.stack((half_a[:, 0] + gap / 2, low + torch.rand(n, device=device) * (high - low)), -1)
    far_stage = torch.stack((half_a[:, 0] + gap + half_b[:, 0], offset), -1)
    # The bridge slides along the channel from beyond both stage ends.
    sign = torch.where(torch.rand(n, device=device) < .5, 1., -1.)
    end = torch.maximum(half_a[:, 1], sign * offset + half_b[:, 1]) + bridge_half[:, 1]
    start = torch.stack((landing[:, 0], sign * (end + _uniform(cfg['approach_distance'], n, device=device))), -1)

    radius = box_size[..., :2].norm(dim=-1) / 2
    near_edge = (half_a[:, 0] + gap)[:, None]
    band = cfg['object_band']
    x_low = near_edge + band[0] + radius
    x_high = torch.minimum(near_edge + band[1] + radius, near_edge + 2 * half_b[:, :1] - margin - radius)
    lane = 2 * half_b[:, 1:] / 3
    order = torch.rand(n, 3, device=device).argsort(-1)
    jitter = (lane / 2 - radius - .05).clamp_min(0)
    objects = torch.stack((x_low + torch.rand(n, 3, device=device) * (x_high - x_low),
                           offset[:, None] - half_b[:, 1:] + (order + .5) * lane
                           + (torch.rand(n, 3, device=device) * 2 - 1) * jitter), -1)
    goal = torch.stack((near_edge[:, 0] + sum(band) / 2, offset), -1)

    # Normal starts stay near the gap edge of stage A, laterally within the stage overlap.
    low_a, high_a = -half_a + margin, half_a - margin
    depth = cfg['spawn_depth']
    x_low_a = torch.maximum(half_a[:, :1] - depth[1], low_a[:, :1])
    x_high_a = half_a[:, :1] - depth[0]
    y_low_a = torch.maximum(overlap_low, low_a[:, 1])[:, None]
    y_high_a = torch.minimum(overlap_high, high_a[:, 1])[:, None]
    spec = cfg['near_start']
    near = torch.rand(n, 2, device=device) < near_probability

    def draw():
        xy = torch.stack((x_low_a + torch.rand(n, 2, device=device) * (x_high_a - x_low_a),
                          y_low_a + torch.rand(n, 2, device=device) * (y_high_a - y_low_a)), -1)
        edge = half_a[:, :1] - _uniform(spec['edge_distance'], n, 2, device=device)
        lateral = landing[:, 1:] + (torch.rand(n, 2, device=device) * 2 - 1) * spec['lateral_range']
        xy[..., 0] = torch.where(near, edge, xy[..., 0])
        xy[..., 1] = torch.where(near, torch.minimum(torch.maximum(lateral, low_a[:, 1:]), high_a[:, 1:]), xy[..., 1])
        return xy

    humans = draw()
    for _ in range(16):
        close = (humans[:, 0] - humans[:, 1]).norm(dim=-1) < agent_distance
        if not close.any():
            break
        humans[close, 1] = draw()[close, 1]
    close = (humans[:, 0] - humans[:, 1]).norm(dim=-1) < agent_distance
    humans[close, 1, 0] = humans[close, 0, 0]
    humans[close, 1, 1] = humans[close, 0, 1] - torch.where(humans[close, 0, 1] > 0, 1.2, -1.2)
    return {'humans': humans, 'near': near, 'objects': objects, 'goal': goal, 'far_stage': far_stage,
            'landing': landing, 'start': start}


def bridge_schedule(distance, arrival, speed_range, minimum):
    # Move continuously when possible; wait at the start if even the slowest speed is early,
    # or start part-way along the path (negative wait) if the fastest speed is late.
    speed = (distance / arrival).clamp(*speed_range)
    duration = (distance / speed).clamp_min(minimum)
    return duration, arrival - duration


def bridge_trajectory(start, goal, elapsed, duration):
    phase = (elapsed / duration).clamp(0, 1)
    blend = phase ** 3 * (10 + phase * (6 * phase - 15))
    rate = 30 * phase ** 2 * (1 - phase) ** 2 / duration
    return start + blend[:, None] * (goal - start), rate[:, None] * (goal - start)


def bridge_control(target_pos, target_vel, state, mass, frequency, damping, max_acceleration):
    omega = 2 * torch.pi * frequency
    acceleration = (omega ** 2 * (target_pos - state[:, :3])
                    + 2 * damping * omega * (target_vel - state[:, 7:10]))
    norm = acceleration.norm(dim=-1, keepdim=True)
    acceleration = acceleration * (max_acceleration / norm.clamp_min(1e-6)).clamp_max(1)
    return mass[:, None] * acceleration
