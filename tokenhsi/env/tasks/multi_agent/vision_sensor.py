import torch

from env.tasks.multi_agent.cliff_geometry import sensor_positions


def fan_grid(resolution, distance, fov, minimum, power, device):
    angles = torch.linspace(-fov / 2, fov / 2, resolution, device=device) * (torch.pi / 180)
    radii = minimum + (distance - minimum) * torch.linspace(0, 1, resolution, device=device).pow(power)
    return torch.stack((angles.cos()[:, None] * radii,
                        angles.sin()[:, None] * radii), -1).flatten(0, 1)


def uniform_fan_grid(rings, distance, fov, device):
    points, successors = [], []
    for ring in range(rings):
        count = 2 * ring + 1
        fraction = (torch.arange(count) + .5) / count
        angle = (fraction - .5) * (fov * torch.pi / 180)
        radius = (ring + .5) * distance / rings
        points.append(torch.stack((angle.cos(), angle.sin()), -1) * radius)
        if ring + 1 < rings:
            successors.append((fraction * (count + 2)).floor().long() + (ring + 1)**2)
        else:
            successors.append(torch.arange(ring**2, rings**2))
    return torch.cat(points).to(device), torch.cat(successors).to(device)


def rotation_matrix(q):
    x, y, z, w = q.unbind(-1)
    return torch.stack((1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w),
                        2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w),
                        2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)), -1).reshape(-1, 3, 3)


def ray_box_interval(start, direction, center, rotation, half):
    local_start = torch.matmul(start - center[:, None, None], rotation[:, None])
    local_direction = torch.matmul(direction, rotation[:, None])
    return _slab_interval(local_start, local_direction, half[:, None, None])


def _slab_interval(local_start, local_direction, half):
    parallel = local_direction.abs() < 1e-8
    denominator = torch.where(parallel, torch.ones_like(local_direction), local_direction)
    lo = (-half - local_start) / denominator
    hi = (half - local_start) / denominator
    near, far = torch.minimum(lo, hi), torch.maximum(lo, hi)
    inside = local_start.abs() <= half + 1e-6
    near = torch.where(parallel, torch.where(inside, -torch.inf, torch.inf), near)
    far = torch.where(parallel, torch.where(inside, torch.inf, -torch.inf), far)
    return near.amax(-1), far.amin(-1)


def _group_interval(start, direction, centers, rotations, half):
    local_start = torch.einsum('eapoj,eojk->eapok',
        start[..., None, :] - centers[:, None, None], rotations)
    local_direction = torch.einsum('eapj,eojk->eapok', direction, rotations)
    return _slab_interval(local_start, local_direction, half[:, None, None])


def fill_occluded_rings(surface, visible, successors, rings):
    indices = torch.arange(surface.shape[-1], device=surface.device)
    source = torch.where(visible, indices, successors)
    # Visible samples are absorbing nodes; doubling follows at most rings - 1 links.
    for _ in range((rings - 1).bit_length()):
        source = source.gather(-1, source)
    known = visible.gather(-1, source)
    return surface.gather(-1, source), torch.where(visible, 1, torch.where(known, 2, 0))


def fill_occluded(surface, visible):
    count = surface.shape[-1]
    indices = torch.arange(count, device=surface.device).expand_as(surface)
    next_visible = torch.where(visible, indices, count).flip(-1).cummin(-1).values.flip(-1)
    known = next_visible < count
    filled = surface.gather(-1, next_visible.clamp_max(count - 1))
    # 0: unknown, 1: directly visible, 2: filled from the nearest farther visible sample.
    status = torch.where(visible, 1, torch.where(known, 2, 0))
    return filled, status


def vertical_height(xy, centers, quaternions, half_sizes, floor):
    # Top-down lookup: each box is its yaw footprint at the height of its highest corner.
    rotation = rotation_matrix(quaternions).view(*centers.shape[:2], 3, 3)
    shape = (len(centers),) + (1,) * (xy.ndim - 2) + (centers.shape[1],)
    yaw = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
    top = centers[..., 2] + (rotation[..., 2, :].abs() * half_sizes).sum(-1)
    offset = xy[..., None, :] - centers[..., :2].view(*shape, 2)
    c, s = yaw.cos().view(shape), yaw.sin().view(shape)
    x = c * offset[..., 0] + s * offset[..., 1]
    y = c * offset[..., 1] - s * offset[..., 0]
    inside = (x.abs() <= half_sizes[..., 0].view(shape)) & (y.abs() <= half_sizes[..., 1].view(shape))
    height = torch.where(inside, top.view(shape), -torch.inf).amax(-1)
    return torch.maximum(height, floor.view(shape[:-1]))


def fan_height_observation(eye, root, heading, grid, centers, quaternions, half_sizes, floor, clip):
    xy = sensor_positions(eye, heading, grid)
    surface = vertical_height(xy, centers, quaternions, half_sizes, floor)
    observation = (surface - root[..., 2, None]).clamp(-clip, clip) / clip
    return {'observation': observation, 'xy': xy, 'surface': surface, 'filled': surface,
            'status': torch.ones_like(surface, dtype=torch.long), 'eye': eye}


def fan_observation(eye, root, heading, grid, centers, quaternions, half_sizes,
                    floor, resolution, clip, successors=None, object_batch_size=1):
    xy = sensor_positions(eye, heading, grid)
    ceiling = (centers[..., 2] + half_sizes.norm(dim=-1)).amax(-1) + 1
    top = torch.cat((xy, ceiling[:, None, None, None].expand(*xy.shape[:-1], 1)), -1)
    down = top.new_tensor([0., 0., -1.]).view(1, 1, 1, 3).expand(len(eye), 1, 1, 3)
    surface = floor[:, None, None].expand(xy.shape[:-1]).clone()
    rotations = rotation_matrix(quaternions).view(*centers.shape[:2], 3, 3)
    groups = range(0, centers.shape[1], object_batch_size)

    def intervals(start, direction, i):
        if object_batch_size == 1:
            near, far = ray_box_interval(start, direction, centers[:, i], rotations[:, i], half_sizes[:, i])
            return near[..., None], far[..., None]
        group = slice(i, i + object_batch_size)
        return _group_interval(start, direction, centers[:, group], rotations[:, group], half_sizes[:, group])

    for i in groups:
        near, far = intervals(top, down, i)
        hit = (far >= near) & (far >= 0)
        height = ceiling[:, None, None, None] - near
        surface = torch.maximum(surface, torch.where(hit, height, -torch.inf).amax(-1))

    targets = torch.cat((xy, surface[..., None]), -1)
    direction = targets - eye[..., None, :]
    visible = torch.ones_like(surface, dtype=torch.bool)
    for i in groups:
        near, far = intervals(eye[..., None, :], direction, i)
        blocked = (far >= near) & (far > 1e-5) & (near < 1 - 1e-5)
        visible &= ~blocked.any(-1)
    if successors is None:
        shape = (*surface.shape[:-1], resolution, resolution)
        filled, status = fill_occluded(surface.reshape(shape), visible.reshape(shape))
        filled, status = filled.flatten(-2), status.flatten(-2)
    else:
        filled, status = fill_occluded_rings(surface, visible, successors, resolution)
    observation = (filled - root[..., 2, None]).clamp(-clip, clip) / clip
    observation = torch.where(status != 0, observation, 2.)
    return {'observation': observation, 'xy': xy, 'surface': surface,
            'filled': filled, 'status': status, 'eye': eye}
