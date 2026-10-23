from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from joint_carry.kinematics import forward_kinematics, pack
from joint_carry.correction import ASSET

def foot_vertices(name):
    path = ASSET.parent / '../smpl_foot_meshes' / (name + '.obj')
    return np.array([[float(v) for v in line.split()[1:4]]
                     for line in path.read_text().splitlines() if line.startswith('v ')])

def lowest_vertex(quaternions, vertices):
    matrices = R.from_quat(quaternions).as_matrix()
    return np.einsum('fij,vj->fvi', matrices, vertices)[:, :, 2].min(1)

def retarget_legs(data):
    q = data['local_rotation_xyzw'].copy()
    roots = data['root_position'].copy()
    names = data['joint_names'].tolist()
    offsets, parents = data['local_translation'], data['parent_indices']
    asset = Path('tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml')
    bodies = {b.get('name'): b for b in ET.parse(asset).findall('.//worldbody//body')}
    errors = []
    for hip, knee, foot in ((9, 10, 11), (12, 13, 14)):
        joints = [j for index in (hip, knee, foot) for j in bodies[names[index]].findall('joint')]
        bounds = np.deg2rad([np.fromstring(j.get('range'), sep=' ') for j in joints])
        lower, upper = bounds[:, 0] + .002, bounds[:, 1] - .002
        reference = np.concatenate([R.from_quat(q[:, 0, j]).as_rotvec() for j in (hip, knee, foot)], axis=1)
        previous = np.clip(reference[0], lower, upper)
        for frame in range(len(q)):
            root_rotation = R.from_quat(q[frame, 0, 0])
            hip_position = roots[frame, 0] + root_rotation.apply(offsets[hip])
            target = data['body_position'][frame, 0, foot]
            target_rotation = R.from_quat(data['body_rotation_xyzw'][frame, 0, foot])

            def residual(x):
                local = R.from_rotvec(x.reshape(3, 3))
                thigh = root_rotation * local[0]
                shin = thigh * local[1]
                position = hip_position + thigh.apply(offsets[knee]) + shin.apply(offsets[foot])
                orientation = (target_rotation.inv() * shin * local[2]).as_rotvec()
                return np.concatenate(((position-target)/.003, orientation/.03,
                                       .2*(x-reference[frame]), .1*(x-previous)))

            fit = least_squares(residual, previous, bounds=(lower, upper), max_nfev=100,
                                ftol=1e-8, xtol=1e-8, gtol=1e-8)
            q[frame, 0, [hip, knee, foot]] = R.from_rotvec(fit.x.reshape(3, 3)).as_quat()
            previous = fit.x
            errors.append(float(np.linalg.norm(residual(fit.x)[:3])*.003))

    if q.shape[1] == 2:
        q[:, 1, 1:] = q[:, 0, 1:]
    positions, world = forward_kinematics(q[:, 0], roots[:, 0], parents, offsets)
    soles = np.stack([positions[:, j, 2]+lowest_vertex(world[:, j], foot_vertices(names[j]))
                      for j in (11, 14)], axis=1)
    # A constant lift preserves the source velocities without per-frame vertical jitter.
    lift = max(0., .003-float(soles.min()))
    roots[:, :, 2] += lift
    box = np.concatenate((data['box_position'], data['box_rotation_xyzw']), axis=-1)
    candidate = pack(q, roots, parents, offsets, box, data['box_size'], int(data['fps']), names)
    candidate['skill'] = data['skill']
    candidate['grip_inset_m'] = data['grip_inset_m']
    candidate['source_rsi_allowed'] = np.ones(len(q), dtype=bool)
    return candidate
