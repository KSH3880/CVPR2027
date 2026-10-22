"""Spatial/force proxies for non-hand contact, without GPU collision pairs."""
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import torch
from utils.door_reference import rotate


def validate_contact(spec):
    if set(spec) != {'body_force_threshold', 'body_distance_margin', 'open_credit_tolerance'}:
        raise ValueError('Invalid manipulationContact fields')
    for value in spec.values():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('Contact thresholds must be finite and positive')
    if spec['body_distance_margin'] > .1 or spec['open_credit_tolerance'] > .05:
        raise ValueError('Contact margins must remain small')


def collision_samples(asset_file, include_hands=False):
    """Cover spheres/capsules and foot mesh AABBs with local bounding spheres."""
    asset_file = Path(asset_file)
    tree = ET.parse(asset_file)
    meshes = {m.attrib['name']: asset_file.parent / m.attrib['file']
              for m in tree.findall('./asset/mesh')}
    result = []
    for body in tree.findall('.//body'):
        name = body.attrib['name']
        if not include_hands and name in ('right_hand', 'left_hand'):
            continue
        for geom in body.findall('./geom'):
            kind = geom.get('type', 'capsule')
            if kind == 'mesh':
                vertices = [list(map(float, line.split()[1:4]))
                            for line in meshes[geom.attrib['mesh']].read_text().splitlines()
                            if line.startswith('v ')]
                points = torch.tensor(vertices)
                lo, hi = points.amin(0), points.amax(0)
                center = (lo + hi) / 2
                center += torch.tensor(list(map(float, geom.get('pos', '0 0 0').split())))
                result.append((name, center.tolist(), float((hi - lo).norm() / 2)))
            elif 'fromto' in geom.attrib:
                ends = torch.tensor(list(map(float, geom.attrib['fromto'].split()))).reshape(2, 3)
                radius = float(geom.attrib['size'].split()[0])
                length = float((ends[1] - ends[0]).norm())
                count = max(2, math.ceil(length / radius) + 1)
                cover = math.sqrt(radius ** 2 + (length / (count - 1) / 2) ** 2)
                for t in torch.linspace(0, 1, count):
                    result.append((name, ((1 - t) * ends[0] + t * ends[1]).tolist(), cover))
            elif kind == 'sphere':
                result.append((name, list(map(float, geom.get('pos', '0 0 0').split())),
                               float(geom.attrib['size'])))
            else:
                raise ValueError('Unsupported contact geometry: ' + kind)
    if not result:
        raise ValueError('No non-hand contact geometry')
    return result


def nonhand_contact(points, radii, forces, boxes, sizes, object_forces, spec):
    """Points are world-space collision samples; vertical ground support is excluded.

    Shapes: points/forces [env, owner, samples, 3], boxes [env, owner, 13].
    This does not claim pair-specific forces or a hand/body force fraction.
    """
    inv = boxes[..., 3:7].clone()
    inv[..., :3] *= -1
    local = rotate(inv[:, :, None].expand(*points.shape[:-1], 4),
                   points - boxes[..., :3][:, :, None])
    half = torch.as_tensor(sizes, device=points.device, dtype=points.dtype) / 2
    outside = (local.abs() - half[..., None, :]).clamp_min(0).norm(dim=-1)
    near = outside <= radii + spec['body_distance_margin']
    # Ground normal forces alone cannot trigger a body/object contact.
    horizontal = forces[..., :2].norm(dim=-1)
    active = horizontal >= spec['body_force_threshold']
    target_active = object_forces[..., :2].norm(dim=-1) >= spec['body_force_threshold']
    per_sample = near & active & target_active[..., None]
    peak = torch.where(per_sample, horizontal, 0.).amax(-1)
    return per_sample.any(-1), peak
