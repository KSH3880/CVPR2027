import torch


def stage_height(xy, origins, half_size, height):
    shape = (len(origins),) + (1,) * (xy.ndim - 2) + (2,)
    inside = ((xy - origins[:, :2].reshape(shape)).abs()
              <= half_size.reshape(shape)).all(-1)
    return inside.to(xy.dtype) * height + origins[:, 2].reshape(shape[:-1])


def sensor_grid(resolution, extent, device):
    axis = torch.linspace(-extent, extent, resolution, device=device)
    x, y = torch.meshgrid(axis, axis, indexing='ij')
    return torch.stack((x.flatten(), y.flatten()), -1)


def sensor_positions(root, heading, grid):
    c, s = heading.cos()[..., None], heading.sin()[..., None]
    x = c * grid[:, 0] - s * grid[:, 1]
    y = s * grid[:, 0] + c * grid[:, 1]
    return root[..., None, :2] + torch.stack((x, y), -1)


def height_observation(root, heading, grid, origins, half_size, height, clip):
    xy = sensor_positions(root, heading, grid)
    ground = stage_height(xy, origins, half_size, height)
    return (ground - root[..., 2, None]).clamp(-clip, clip) / clip


def terrain_fall(contact, body_pos, contact_ids, ground, thresholds, progress, floor_height):
    contact_mask = contact.abs().sum(-1) > 0.1
    low = body_pos[..., 2] - ground < thresholds
    outside = ground <= floor_height[:, None, None] + 1e-4
    floor_contact = (contact_mask & low & outside).any(-1)
    contact_mask[..., contact_ids] = False
    low[..., contact_ids] = False
    fallen = contact_mask.any(-1) & low.any(-1)
    return (fallen | floor_contact) & (progress[:, None] > 1)


def sample_layout(half_size, box_size, margin, gap):
    n, objects = box_size.shape[:2]
    device = half_size.device
    side = torch.randint(4, (n,), device=device)
    sides = (side[:, None] + torch.tensor([0, 2, 1], device=device)) % 4
    axes = sides % 2
    sign = torch.where(sides < 2, 1., -1.)
    normal_half = half_size.gather(1, axes)
    tangent_half = half_size.gather(1, 1 - axes)
    radius = box_size[..., :2].norm(dim=-1) / 2
    outside = normal_half + radius + gap[0] + torch.rand(n, objects, device=device) * (gap[1] - gap[0])
    tangent = (torch.rand(n, objects, device=device) * 2 - 1) * (tangent_half - margin)
    boxes = torch.zeros(n, objects, 2, device=device)
    boxes.scatter_(2, axes[..., None], (sign * outside)[..., None])
    boxes.scatter_(2, (1 - axes)[..., None], tangent[..., None])
    distance = margin + torch.rand(n, 2, device=device) * (normal_half[:, :2] - margin - 0.3)
    humans = torch.zeros(n, 2, 2, device=device)
    humans.scatter_(2, axes[:, :2, None], (sign[:, :2] * (normal_half[:, :2] - distance))[..., None])
    # Separate the two starts even when both normal coordinates are near the centre.
    human_tangent = torch.tensor([-1., 1.], device=device) * (tangent_half[:, :2] - margin) * 0.7
    humans.scatter_(2, (1 - axes[:, :2, None]), human_tangent[..., None])
    return humans, boxes
