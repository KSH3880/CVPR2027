import torch


def sample_box_properties(count, options, device):
    if 'size_ranges' in options:
        if options['mass_mode'] != 'volume_linear':
            raise ValueError('PUSH cuboids require volume_linear mass')
        ranges = torch.tensor(options['size_ranges'], device=device)
        sizes = ranges[:, 0]+torch.rand(count, 3, device=device)*(ranges[:, 1]-ranges[:, 0])
        if options['y_ge_z']:
            sizes[:, 1:] = sizes[:, 1:].sort(descending=True).values
        volume_fraction = (sizes.prod(-1)-ranges[:, 0].prod())/(ranges[:, 1].prod()-ranges[:, 0].prod())
        lo, hi = options['mass_range']
        masses = lo+(hi-lo)*volume_fraction
    elif 'size_range' in options:
        lo, hi = options['size_range']
        size_fraction = torch.rand(count, 1, device=device)
        edge = lo+(hi-lo)*size_fraction
        sizes = edge.expand(-1, 3).clone()
        lo, hi = options['mass_range']
        if options.get('mass_mode') == 'size_linear':
            masses = lo+(hi-lo)*size_fraction[:, 0]
        else:
            masses = lo+(hi-lo)*torch.rand(count, device=device)
    else:
        sizes = torch.tensor(options['size'], device=device).expand(count, -1).clone()
        masses = torch.full((count,), options['mass'], device=device)
    return sizes, masses


def approach_distance(roots, centers, normals, half_size, stand_off, clearance):
    delta = roots[..., :2]-centers[..., :2]
    tangent = torch.stack((normals[..., 1], -normals[..., 0]), -1)
    start = torch.stack(((delta*tangent).sum(-1), (delta*normals[..., :2]).sum(-1)), -1)
    extent = half_size[..., :2]+clearance
    target = torch.zeros_like(start)
    target[..., 1] = -half_size[..., 1]-stand_off
    left = torch.stack((-extent[..., 0], -extent[..., 1]), -1)
    right = torch.stack((extent[..., 0], -extent[..., 1]), -1)
    back_left = left.clone()
    back_right = right.clone()
    back_left[..., 1] *= -1
    back_right[..., 1] *= -1
    destinations = torch.stack((target, left, right, back_left, back_right), -2)
    remaining = torch.stack((torch.zeros_like(start[..., 0]), (left-target).norm(dim=-1),
        (right-target).norm(dim=-1), 2*extent[..., 1]+(left-target).norm(dim=-1),
        2*extent[..., 1]+(right-target).norm(dim=-1)), -1)
    segment = destinations-start[..., None, :]
    parallel = segment.abs() < 1e-7
    denominator = torch.where(parallel, torch.ones_like(segment), segment)
    lo = (-extent[..., None, :]+1e-4-start[..., None, :])/denominator
    hi = (extent[..., None, :]-1e-4-start[..., None, :])/denominator
    inside_axis = start[..., None, :].abs() < extent[..., None, :]-1e-4
    enter = torch.where(parallel, torch.where(inside_axis, -torch.inf, torch.inf), torch.minimum(lo, hi)).amax(-1)
    leave = torch.where(parallel, torch.where(inside_axis, torch.inf, -torch.inf), torch.maximum(lo, hi)).amin(-1)
    blocked = torch.maximum(enter, torch.zeros_like(enter)) < torch.minimum(leave, torch.ones_like(leave))
    paths = (segment.norm(dim=-1)+remaining).masked_fill(blocked, torch.inf)
    distance = paths.amin(-1)
    return torch.where(torch.isfinite(distance), distance, (start-target).norm(dim=-1))
