import torch
from isaacgym.torch_utils import quat_rotate


def collision_shapes(asset, names, device):
    from pathlib import Path
    import xml.etree.ElementTree as ET
    asset = Path(asset)
    xml = ET.parse(asset).getroot()
    meshes = {mesh.attrib['name']: asset.parent/mesh.attrib['file'] for mesh in xml.findall('asset/mesh')}
    ids, points, radii = [], [], []
    for body in xml.findall('.//body'):
        for geom in body.findall('geom'):
            spec = geom.attrib
            radius = 0.
            offset = [float(v) for v in spec.get('pos', '0 0 0').split()]
            if spec.get('type') == 'mesh':
                vertices = [[float(v) for v in line.split()[1:4]] for line in meshes[spec['mesh']].read_text().splitlines() if line.startswith('v ')]
                vertices = [[v+a for v, a in zip(point, offset)] for point in vertices]
            elif 'fromto' in spec:
                values = [float(v) for v in spec['fromto'].split()]
                vertices = [values[:3], values[3:]]
                radius = float(spec['size'].split()[0])
            else:
                vertices = [offset]
                radius = float(spec['size'].split()[0])
            ids.extend([names.index(body.attrib['name'])]*len(vertices))
            points.extend(vertices)
            radii.extend([radius]*len(vertices))
    return (torch.tensor(ids, device=device), torch.tensor(points, device=device),
            torch.tensor(radii, device=device))


def face_retreat(positions, rotations, face, normal, shapes, clearance):
    ids, points, radii = shapes
    q = rotations[:, ids]
    world = positions[:, ids]+quat_rotate(q.reshape(-1, 4), points[None].expand(len(positions), -1, -1).reshape(-1, 3)).reshape(len(positions), -1, 3)
    support = ((world-face[:, None])*normal[:, None]).sum(-1)+radii
    return (support.amax(-1)+clearance).clamp_min(0)


def reference_state(lib, ids, times):
    # Raw PUSH RSI: the reference pose without the reset-only arm height correction.
    _, _, dof, _, _, dof_vel, _ = lib.get_motion_state(ids, times)
    positions, rotations, velocities, angular = lib.get_motion_state_max(ids, times)
    return positions, rotations, velocities, angular, dof, dof_vel
