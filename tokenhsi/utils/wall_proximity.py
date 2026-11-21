import itertools
from pathlib import Path
import xml.etree.ElementTree as ET

import torch


def load_capsules(asset, names, device):
    xml = ET.parse(Path(asset['assetRoot']) / asset['assetFileName']).getroot()
    endpoints, radii, owners = [], [], []
    for owner, name in enumerate(names):
        for geom in xml.find(".//body[@name='" + name + "']").findall('geom'):
            if geom.get('fromto'):
                points = list(map(float, geom.get('fromto').split()))
            else:
                points = list(map(float, geom.get('pos', '0 0 0').split())) * 2
            endpoints.append(points)
            radii.append(float(geom.get('size').split()[0]))
            owners.append(owner)
    return (torch.tensor(endpoints, device=device).reshape(-1, 2, 3),
            torch.tensor(radii, device=device), torch.tensor(owners, device=device))


def segment_box_distance(start, end, half):
    # Squared point-to-box distance is quadratic between crossings of the six face planes.
    delta = end - start
    divisor = torch.where(delta.abs() > 1e-12, delta, torch.ones_like(delta))
    cuts = torch.cat(((-half - start) / divisor, (half - start) / divisor), -1)
    cuts = torch.where(torch.cat((delta.abs(), delta.abs()), -1) > 1e-12, cuts, 0.)
    cuts = torch.cat((torch.zeros_like(cuts[..., :1]), cuts.clamp(0., 1.), torch.ones_like(cuts[..., :1])), -1).sort(-1).values
    low, high = cuts[..., :-1], cuts[..., 1:]
    middle = start[..., None, :] + ((low + high) / 2)[..., None] * delta[..., None, :]
    active = middle.abs() > half[..., None, :]
    offset = start[..., None, :] - middle.sign() * half[..., None, :]
    slope = delta[..., None, :]
    t = -(offset * slope * active).sum(-1) / (slope.square() * active).sum(-1).clamp_min(1e-20)
    t = torch.maximum(low, torch.minimum(high, t))
    points = start[..., None, :] + t[..., None] * slope
    return (points.abs() - half[..., None, :]).clamp_min(0.).norm(dim=-1).amin(-1)


def capsule_wall_gap(body_pos, body_rot, shapes, centers, rotations, halves):
    endpoints, radii, owners = shapes
    points = body_pos[..., owners, None, :] + torch.einsum('nmsij,skj->nmski', body_rot[..., owners, :, :], endpoints)
    gaps = []
    for wall in range(centers.shape[1]):
        local = torch.einsum('nmski,nij->nmskj', points - centers[:, wall, None, None, None], rotations[:, wall])
        gap = segment_box_distance(local[..., 0, :], local[..., 1, :], halves[:, wall, None, None]) - radii
        gaps.append(gap.amin(-1))
    return torch.stack(gaps, -1).amin(-1)


def box_vertices(center, rotation, half):
    signs = center.new_tensor(list(itertools.product((-1., 1.), repeat=3)))
    return center[..., None, :] + (signs * half[..., None, :]) @ rotation.transpose(-1, -2)


def box_box_gap(center_a, rotation_a, half_a, center_b, rotation_b, half_b):
    a, b = box_vertices(center_a, rotation_a, half_a), box_vertices(center_b, rotation_b, half_b)
    pairs = [(i, j) for i in range(8) for j in range(i + 1, 8) if (i ^ j) in (1, 2, 4)]
    first, second = zip(*pairs)
    a_in_b = (a - center_b[..., None, :]) @ rotation_b
    b_in_a = (b - center_a[..., None, :]) @ rotation_a
    a_gap = segment_box_distance(a_in_b[..., first, :], a_in_b[..., second, :], half_b[..., None, :]).amin(-1)
    b_gap = segment_box_distance(b_in_a[..., first, :], b_in_a[..., second, :], half_a[..., None, :]).amin(-1)
    return torch.minimum(a_gap, b_gap)


def proximity(gap, margin):
    return (1. - gap / margin).clamp(0., 1.)
