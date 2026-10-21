import torch

from env.tasks.multi_agent.bridge_geometry import compute_agent_cpa_collision_penalty


@torch.jit.script
def compute_wall_cpa_collision_penalty(root_pos, root_vel, starts, ends, min_dist, dt, discount):
    # type: (Tensor, Tensor, Tensor, Tensor, float, float, float) -> Tensor
    # Each wall is a stationary agent at its point nearest to the root; the agent CPA does the rest.
    segment = ends - starts
    along = (((root_pos[:, :, None, :2] - starts[:, None]) * segment[:, None]).sum(-1)
             / segment.square().sum(-1)[:, None]).clamp(0., 1.)
    nearest = starts[:, None] + along.unsqueeze(-1) * segment[:, None]
    n, agents, walls = nearest.shape[0], nearest.shape[1], nearest.shape[2]
    pos = torch.cat((root_pos[:, :, None, :2], nearest), 2).view(n * agents, walls + 1, 2)
    vel = torch.cat((root_vel[:, :, None, :2], torch.zeros_like(nearest)), 2).view(n * agents, walls + 1, 2)
    return compute_agent_cpa_collision_penalty(pos, vel, min_dist, dt, discount)[:, 0].view(n, agents)


def room_walls(half_size, thickness, height, passage_width=None):
    n = len(half_size)
    centers = half_size.new_zeros(n, 4, 3)
    centers[:, :2, 0] = (half_size[:, :1] + thickness / 2) * half_size.new_tensor([1., -1.])
    centers[:, 2:, 1] = (half_size[:, 1:] + thickness / 2) * half_size.new_tensor([1., -1.])
    centers[..., 2] = height / 2
    half = torch.full_like(centers, height / 2)
    half[:, :2, 0], half[:, :2, 1] = thickness / 2, half_size[:, 1:]
    half[:, 2:, 0], half[:, 2:, 1] = half_size[:, :1] + thickness, thickness / 2
    starts = torch.stack((torch.stack((half_size[:, 0], -half_size[:, 1]), -1),
                          -half_size, torch.stack((-half_size[:, 0], half_size[:, 1]), -1),
                          -half_size), 1)
    ends = torch.stack((half_size, torch.stack((-half_size[:, 0], half_size[:, 1]), -1),
                        half_size, torch.stack((half_size[:, 0], -half_size[:, 1]), -1)), 1)
    if passage_width is not None:
        partition_half = torch.stack((torch.full_like(half_size[:, 0], thickness / 2),
            half_size[:, 1] - passage_width, torch.full_like(half_size[:, 0], height / 2)), -1)
        center = torch.zeros_like(partition_half)
        center[:, 2] = height / 2
        corners = partition_half[:, None, :2] * half_size.new_tensor([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
        centers = torch.cat((centers, center[:, None]), 1)
        half = torch.cat((half, partition_half[:, None]), 1)
        starts = torch.cat((starts, corners), 1)
        ends = torch.cat((ends, corners.roll(-1, 1)), 1)
    return centers, half, starts, ends


def sample_room_layout(half_size, box_size, human_margin, object_margin):
    n, objects = box_size.shape[:2]
    radii = torch.cat((box_size[..., :2].norm(p=2, dim=-1) / 2,
                       half_size.new_full((n, 2), .4)), -1)
    margins = radii + object_margin
    margins[:, objects:] = human_margin
    positions = half_size.new_zeros(n, objects + 2, 2)
    for i in range(objects + 2):
        pending = torch.arange(n, device=half_size.device)
        for _ in range(256):
            positions[pending, i] = (torch.rand(len(pending), 2, device=half_size.device) * 2 - 1) * (
                half_size[pending] - margins[pending, i, None])
            distance = (positions[pending, i, None] - positions[pending, :i]).norm(p=2, dim=-1)
            overlap = (distance < radii[pending, i, None] + radii[pending, :i] + object_margin).any(-1)
            pending = pending[overlap]
            if not len(pending):
                break
        if len(pending):
            raise ValueError('Room is too small for non-overlapping starts')
    return positions[:, objects:], positions[:, :objects]


def sample_partition_layout(half_size, box_size, assigned, human_margin, object_margin, thickness, passage_width,
                            pickup_cross=None):
    n, objects = box_size.shape[:2]
    device = half_size.device
    box_radius = box_size[..., :2].norm(dim=-1) / 2
    goal_radius = box_radius.gather(1, assigned)
    radii = torch.cat((box_radius, half_size.new_full((n, 2), .4), goal_radius), -1)
    margins = radii + object_margin
    margins[:, objects:objects + 2] = human_margin
    margins[:, objects + 2:] = torch.maximum(margins[:, objects + 2:], half_size.new_tensor(human_margin))
    side = torch.where(torch.rand(n, 1, device=device) < .5, -1., 1.) * half_size.new_tensor([1., -1.])
    box_side = torch.where(torch.rand(n, objects, device=device) < .5, -1., 1.)
    box_side.scatter_(1, assigned, side)
    human_side = side if pickup_cross is None else torch.where(pickup_cross, -side, side)
    sides = torch.cat((box_side, human_side, -side), 1)
    low_x = thickness / 2 + margins
    high_x = half_size[:, :1] - margins
    y_extent = (half_size[:, 1:] - passage_width - object_margin).expand_as(radii).clone()
    bound = torch.zeros(n, objects, device=device, dtype=torch.bool).scatter_(1, assigned, True)
    y_extent[:, :objects] = torch.where(bound, y_extent[:, :objects], half_size[:, 1:] - margins[:, :objects])
    human_extent = half_size[:, 1:] - human_margin
    if pickup_cross is not None:
        human_extent = torch.where(pickup_cross, y_extent[:, objects:objects + 2], human_extent)
    y_extent[:, objects:objects + 2] = human_extent
    positions = half_size.new_zeros(n, objects + 4, 2)
    pending = torch.arange(n, device=device)
    diagonal = torch.eye(objects + 4, device=device, dtype=torch.bool)[None]
    # Goal positions may be occupied by a humanoid that will move during the episode.
    diagonal[:, objects:objects + 2, objects + 2:] = True
    diagonal[:, objects + 2:, objects:objects + 2] = True
    for _ in range(4096):
        r = torch.rand(len(pending), objects + 4, 2, device=device)
        positions[pending, :, 0] = sides[pending] * (low_x[pending] + r[..., 0] * (high_x[pending] - low_x[pending]))
        positions[pending, :, 1] = (2 * r[..., 1] - 1) * y_extent[pending]
        p = positions[pending]
        distance = (p[:, :, None] - p[:, None]).norm(dim=-1).masked_fill(diagonal, torch.inf)
        overlap = (distance < radii[pending, :, None] + radii[pending, None, :] + object_margin).any(-1).any(-1)
        pending = pending[overlap]
        if not len(pending):
            break
    if len(pending):
        raise ValueError('Partition room is too small for non-overlapping starts and goals')
    return positions[:, objects:objects + 2], positions[:, :objects], positions[:, objects + 2:]
