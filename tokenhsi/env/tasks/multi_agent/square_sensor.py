"""Root-centred square height sensor, RSI foot clearance and the viewer overlay."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch


def rotation_matrix(q):
    x, y, z, w = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(-1, 3, 3)


def sensor_grid(resolution, half_extent, device):
    axis = torch.linspace(-half_extent, half_extent, resolution, device=device)
    x, y = torch.meshgrid(axis, axis, indexing='ij')
    return torch.stack((x.flatten(), y.flatten()), -1)


def sensor_positions(root, heading, grid):
    c, s = heading.cos()[..., None], heading.sin()[..., None]
    x = c*grid[:, 0]-s*grid[:, 1]
    y = s*grid[:, 0]+c*grid[:, 1]
    return root[..., None, :2]+torch.stack((x, y), -1)


def vertical_height(xy, centers, quaternions, half_sizes, floor, chamfer=None, exclude=None):
    """Top-down lookup: each box is its yaw footprint at the height of its highest corner.
    exclude: optional (n, boxes) mask of boxes this lookup ignores."""
    rotation = rotation_matrix(quaternions).view(*centers.shape[:2], 3, 3)
    shape = (len(centers),)+(1,)*(xy.ndim-2)+(centers.shape[1],)
    yaw = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
    top = centers[..., 2]+(rotation[..., 2, :].abs()*half_sizes).sum(-1)
    offset = xy[..., None, :]-centers[..., :2].view(*shape, 2)
    c, s = yaw.cos().view(shape), yaw.sin().view(shape)
    x = c*offset[..., 0]+s*offset[..., 1]
    y = c*offset[..., 1]-s*offset[..., 0]
    inside = (x.abs() <= half_sizes[..., 0].view(shape)) & (y.abs() <= half_sizes[..., 1].view(shape))
    if chamfer is not None:
        side, width, depth = [chamfer[..., i].view(shape) for i in range(3)]
        cut = width*((y.abs()-half_sizes[..., 1].view(shape)+depth)/depth.clamp_min(1e-6)).clamp(0,1)
        inside &= (side == 0) | (side*x <= half_sizes[..., 0].view(shape)-cut)
    if exclude is not None:
        inside &= ~exclude.view(shape)
    height = torch.where(inside, top.view(shape), -torch.inf).amax(-1)
    return torch.maximum(height, floor.view(shape[:-1]))


def height_observation(roots, heading, grid, centers, quaternions, half_sizes, floor, clip, chamfer=None, exclude=None):
    """(n, people, points) surface height relative to each root; one person at a time saves memory.
    exclude: optional (n, people, boxes) mask of boxes hidden from that person's sensor."""
    rows = []
    for person in range(roots.shape[1]):
        xy = sensor_positions(roots[:, person], heading[:, person], grid)
        surface = vertical_height(xy, centers, quaternions, half_sizes, floor, chamfer,
                                  None if exclude is None else exclude[:, person])
        rows.append((surface-roots[:, person, 2, None]).clamp(-clip, clip)/clip)
    return torch.stack(rows, 1)


def load_foot_vertices(asset, device):
    path = Path(asset['assetRoot'])/asset['assetFileName']
    xml = ET.parse(path).getroot()
    meshes = {mesh.get('name'): mesh for mesh in xml.findall('asset/mesh')}
    feet = []
    for name in ('right_foot', 'left_foot'):
        geom = xml.find(".//body[@name='"+name+"']/geom")
        mesh = meshes[geom.get('mesh')]
        with (path.parent/mesh.get('file')).open() as source:
            vertices = [[float(v) for v in line.split()[1:4]] for line in source if line.startswith('v ')]
        vertices = torch.tensor(vertices, device=device)
        vertices *= vertices.new_tensor([float(v) for v in mesh.get('scale', '1 1 1').split()])
        quat = vertices.new_tensor([float(v) for v in geom.get('quat', '1 0 0 0').split()]).roll(-1)
        vertices = vertices@rotation_matrix(quat).squeeze(0).T
        vertices += vertices.new_tensor([float(v) for v in geom.get('pos', '0 0 0').split()])
        feet.append(vertices)
    return feet


def lowest_foot_height(body, body_ids, vertices):
    heights = []
    for index, points in zip(body_ids, vertices):
        foot = body[..., index, :]
        rotation = rotation_matrix(foot[..., 3:7]).reshape(*foot.shape[:-1], 3, 3)
        heights.append((rotation[..., 2, None, :]*points).sum(-1).amin(-1)+foot[..., 2])
    return torch.stack(heights, -1).amin(-1)


def draw_square_sensor(gym, viewer, env, points, resolution, colors):
    """points: (people, grid points, 3) in the viewer env frame."""
    size = .055
    for person, grid in enumerate(points):
        lifted = grid+grid.new_tensor([0., 0., .04])
        dx, dy = grid.new_tensor([size, 0., 0.]), grid.new_tensor([0., size, 0.])
        crosses = torch.cat((torch.stack((lifted-dx, lifted+dx), 1), torch.stack((lifted-dy, lifted+dy), 1)))
        rgb = np.tile(np.asarray(colors[person % len(colors)], dtype=np.float32), (len(crosses), 1))
        gym.add_lines(viewer, env, len(crosses), crosses.cpu().numpy(), rgb)
        square = lifted.reshape(resolution, resolution, 3)
        outline = torch.cat((square[0], square[1:, -1], square[-1, :-1].flip(0),
                             square[1:-1, 0].flip(0), square[:1, 0]))
        lines = torch.stack((outline[:-1], outline[1:]), 1)
        rgb = np.tile(np.array([[.3, .5, 1.]], dtype=np.float32), (len(lines), 1))
        gym.add_lines(viewer, env, len(lines), lines.cpu().numpy(), rgb)
