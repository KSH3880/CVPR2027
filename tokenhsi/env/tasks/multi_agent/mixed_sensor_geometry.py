from pathlib import Path
import xml.etree.ElementTree as ET

import torch

from env.tasks.multi_agent.vision_sensor import rotation_matrix


def load_foot_vertices(asset, device):
    path = Path(asset['assetRoot']) / asset['assetFileName']
    xml = ET.parse(path).getroot()
    meshes = {mesh.get('name'): mesh for mesh in xml.findall('asset/mesh')}
    feet = []
    for name in ('right_foot', 'left_foot'):
        geom = xml.find(".//body[@name='" + name + "']/geom")
        mesh = meshes[geom.get('mesh')]
        with (path.parent / mesh.get('file')).open() as source:
            vertices = [[float(v) for v in line.split()[1:4]] for line in source if line.startswith('v ')]
        vertices = torch.tensor(vertices, device=device)
        vertices *= vertices.new_tensor([float(v) for v in mesh.get('scale', '1 1 1').split()])
        quat = vertices.new_tensor([float(v) for v in geom.get('quat', '1 0 0 0').split()]).roll(-1)
        vertices = vertices @ rotation_matrix(quat).squeeze(0).T
        vertices += vertices.new_tensor([float(v) for v in geom.get('pos', '0 0 0').split()])
        feet.append(vertices)
    return feet


def lowest_foot_height(body, body_ids, vertices):
    heights = []
    for index, points in zip(body_ids, vertices):
        foot = body[..., index, :]
        rotation = rotation_matrix(foot[..., 3:7]).reshape(*foot.shape[:-1], 3, 3)
        heights.append((rotation[..., 2, None, :] * points).sum(-1).amin(-1) + foot[..., 2])
    return torch.stack(heights, -1).amin(-1)


def sample_stage_goals(half_size, own_box_size, margin):
    radius = own_box_size[..., :2].norm(dim=-1) / 2
    inset = torch.maximum(radius + .2, torch.full_like(radius, margin))
    available = half_size[:, None] - inset[..., None]
    goals = (torch.rand_like(available) * 2 - 1) * available
    pending = torch.arange(len(half_size), device=half_size.device)
    for _ in range(256):
        overlap = (goals[pending, 0] - goals[pending, 1]).norm(dim=-1) < radius[pending].sum(-1) + .2
        pending = pending[overlap]
        if not len(pending):
            return goals
        goals[pending, 1] = (torch.rand(len(pending), 2, device=half_size.device) * 2 - 1) * available[pending, 1]
    raise ValueError('Stage is too small for separate transport goals')
