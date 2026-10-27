import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation as R
from joint_carry.correction import ASSET

def geometry(data):
    """Signed distances for actual asset spheres and sampled capsule centre lines.

    Feet use OBJ vertices for floor checks. Capsule centre samples are <=5mm
    apart; intersection at any sample proves overlap of the actual capsule.
    Mesh-vs-box and humanoid self-collision are left to PhysX.
    """
    xml = ET.parse(ASSET).getroot()
    meshes = {m.get('name'): ASSET.parent/m.get('file') for m in xml.findall('asset/mesh')}
    names = data['joint_names'].tolist()
    frames, agents = data['body_position'].shape[:2]
    hand_depth, other_depth, floor_depth = [np.zeros(frames) for _ in range(3)]
    worst_name = np.full(frames, '', dtype='<U60')
    all_clouds = [[] for _ in range(agents)]
    inverse = R.from_quat(data['box_rotation_xyzw']).inv()
    for body in xml.findall('.//worldbody//body'):
        if body.get('name') not in names:
            continue
        j = names.index(body.get('name'))
        for geom in body.findall('geom'):
            typ = geom.get('type', 'capsule')
            if typ == 'mesh':
                points = np.array([[float(v) for v in line.split()[1:4]]
                                   for line in meshes[geom.get('mesh')].read_text().splitlines()
                                   if line.startswith('v ')])
                radius = 0.
            elif geom.get('fromto'):
                ends = np.fromstring(geom.get('fromto'), sep=' ').reshape(2, 3)
                points = np.linspace(*ends, max(2, int(np.ceil(np.linalg.norm(ends[1]-ends[0])/.005))+1))
                radius = float(geom.get('size').split()[0])
            else:
                assert typ == 'sphere', typ
                points = np.fromstring(geom.get('pos', '0 0 0'), sep=' ')[None]
                radius = float(geom.get('size').split()[0])
            for a in range(agents):
                rot = R.from_quat(data['body_rotation_xyzw'][:, a, j])
                world = np.stack([rot.apply(point) for point in points], axis=1) + data['body_position'][:, a, j, None]
                floor_depth = np.maximum(floor_depth, np.maximum(0, radius-world[..., 2].min(1)))
                if typ == 'mesh':
                    continue
                local = np.stack([inverse.apply(world[:, k]-data['box_position']) for k in range(len(points))], axis=1)
                delta = np.abs(local)-data['box_size']/2
                sdf = np.linalg.norm(np.maximum(delta, 0), axis=-1) + np.minimum(delta.max(-1), 0)
                depth = np.maximum(0, radius-sdf.min(1))
                if 'hand' in body.get('name'):
                    hand_depth = np.maximum(hand_depth, depth)
                else:
                    bigger = depth > other_depth
                    worst_name[bigger] = body.get('name') + '/' + geom.get('name', '')
                    other_depth = np.maximum(other_depth, depth)
                # Thinner cloud sufficient to establish inter-person penetration.
                all_clouds[a].append((world[:, ::max(1, len(points)//12)], radius))
    peer_depth = np.zeros(frames)
    if agents == 2:
        for x, rx in all_clouds[0]:
            for y, ry in all_clouds[1]:
                distance = np.linalg.norm(x[:, :, None]-y[:, None], axis=-1).min((1, 2))
                peer_depth = np.maximum(peer_depth, np.maximum(0, rx+ry-distance))
    return dict(hand_box_penetration_m=hand_depth, nonhand_box_penetration_m=other_depth,
                floor_penetration_m=floor_depth, interperson_penetration_lower_bound_m=peer_depth,
                worst_nonhand_shape=worst_name)
