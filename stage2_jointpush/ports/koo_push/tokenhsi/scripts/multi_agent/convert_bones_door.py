"""SOMA -> phys_humanoid_v3, constrained position/orientation IK, CPU only.

Keep the source read-only. Save Poselib SkeletonMotion and an audit/preview.
The door overlay is a geometric compatibility diagnostic, not object mocap.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp
import torch
from lpanlib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from tokenhsi.utils.bones_bvh import read_bvh

SOURCE_NAME = 'inside_door_handle_right_side_open_walk_R_001__A515_M'
XML = ROOT / 'tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml'
MAPPING = {'pelvis': 'Hips', 'torso': 'Spine2', 'head': 'Head',
           'right_upper_arm': 'RightArm', 'right_lower_arm': 'RightForeArm', 'right_hand': 'RightHand',
           'left_upper_arm': 'LeftArm', 'left_lower_arm': 'LeftForeArm', 'left_hand': 'LeftHand',
           'right_thigh': 'RightLeg', 'right_shin': 'RightShin', 'right_foot': 'RightFoot',
           'left_thigh': 'LeftLeg', 'left_shin': 'LeftShin', 'left_foot': 'LeftFoot'}
DOF_BODIES = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
DOF_OFFSETS = [0, 3, 6, 9, 10, 13, 14, 17, 20, 23, 26, 29, 32]


def unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-9)


def frame(up, left):
    z = unit(up)
    y = unit(left - np.dot(left, z) * z)
    return np.column_stack((unit(np.cross(y, z)), y, z))


def limb_frame(upper, lower, fallback):
    z = -unit(upper)
    # phys humanoid elbows bend about negative local Y.
    normal = -np.cross(unit(upper), unit(lower))
    y = unit(normal if np.linalg.norm(normal) > .02 else fallback)
    y = unit(y - np.dot(y, z) * z)
    return np.column_stack((unit(np.cross(y, z)), y, z))


def joint_bounds(tree):
    bodies = {b.attrib['name']: b for b in ET.parse(XML).iter('body')}
    bounds = []
    for body in DOF_BODIES:
        joints = bodies[tree.node_names[body]].findall('joint')
        bounds.extend([np.deg2rad([float(x) for x in j.attrib['range'].split()]) for j in joints])
    return np.array(bounds).T


def unpack(q):
    local = np.tile(np.eye(3), (15, 1, 1))
    for b, lo, hi in zip(DOF_BODIES, DOF_OFFSETS[:-1], DOF_OFFSETS[1:]):
        vector = q[lo:hi] if hi-lo == 3 else np.array([0., q[lo], 0.])
        local[b] = Rotation.from_rotvec(vector).as_matrix()
    return local


def fk(local, root_rotation, offsets, parents):
    pos = np.zeros((15, 3))
    rot = np.zeros((15, 3, 3))
    rot[0] = root_rotation
    for j in range(1, 15):
        p = parents[j]
        pos[j] = pos[p] + rot[p] @ offsets[j]
        rot[j] = rot[p] @ local[j]
    return pos, rot


PUSH_NAMES = tuple(json.loads((ROOT / 'tokenhsi/data/bones_push180_selection.json').read_text())['clips'])


def convert(source, output, start=0., end=3.4, fps=30, profile='door'):
    torch.set_num_threads(1)
    nodes, source_pos, source_rot, dt = read_bvh(source)
    ids = {n['name']: i for i, n in enumerate(nodes)}
    allowed = (SOURCE_NAME,) if profile == 'door' else PUSH_NAMES if profile == 'push' else ()
    if len(nodes) != 78 or source.stem not in allowed:
        raise ValueError('Converter requires a selected Uniform source and matching door/push profile')
    duration = (len(source_pos)-1)*dt
    if not 0 <= start < end <= duration:
        raise ValueError('Invalid crop')
    times = np.arange(start, end + 1e-8, 1/fps)
    original_times = np.arange(len(source_pos))*dt
    src_pos = np.stack([np.stack([np.interp(times, original_times, source_pos[:, j, k]) for k in range(3)], -1)
                        for j in range(len(nodes))], axis=1)
    src_rot = np.stack([Slerp(original_times, Rotation.from_matrix(source_rot[:, j]))(times).as_matrix()
                        for j in range(len(nodes))], axis=1)
    tree = SkeletonTree.from_mjcf(str(XML))
    parents = tree.parent_indices.numpy()
    offsets = tree.local_translation.numpy().astype(float)
    target_ids = {n: i for i, n in enumerate(tree.node_names)}
    mapped = [ids[MAPPING[n]] for n in tree.node_names]
    # Anatomical initial forward -> +X, left -> +Y, world Y up -> Z.
    left = source_pos[0, ids['LeftArm']] - source_pos[0, ids['RightArm']]
    left[1] = 0
    left = unit(left)
    up = np.array([0., 1., 0.])
    forward = unit(np.cross(left, up))
    basis = np.stack((forward, left, up))
    pos = src_pos @ basis.T
    hips = pos[:, ids['Hips']]
    source_leg = sum(np.linalg.norm(nodes[ids[n]]['offset'])/100 for n in ('LeftShin', 'LeftFoot'))
    target_leg = sum(np.linalg.norm(offsets[target_ids[n]]) for n in ('left_shin', 'left_foot'))
    scale = target_leg / source_leg
    root = hips * scale
    root[:, :2] -= root[0, :2]
    targets = (pos[:, mapped] - hips[:, None]) * scale
    bounds = joint_bounds(tree)
    joint_values, all_rotations, all_positions = [], [], []
    root_rotations = []
    previous = np.zeros(32)
    residuals = []
    for f in range(len(times)):
        p = pos[f]
        root_r = frame(p[ids['Chest']]-p[ids['Hips']], p[ids['LeftLeg']]-p[ids['RightLeg']])
        desired = np.tile(root_r, (15, 1, 1))
        desired[1] = frame(p[ids['Neck1']]-p[ids['Spine2']], p[ids['LeftArm']]-p[ids['RightArm']])
        # SOMA head local +X is up, +Y forward, +Z left in this clip.
        desired[2] = basis @ src_rot[f, ids['Head']] @ np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        for side in ('left', 'right'):
            prefix = side.title()
            arm = target_ids[side+'_upper_arm']
            elbow, hand = arm+1, arm+2
            upper = p[ids[prefix+'ForeArm']]-p[ids[prefix+'Arm']]
            lower = p[ids[prefix+'Hand']]-p[ids[prefix+'ForeArm']]
            desired[arm] = limb_frame(upper, lower, root_r[:, 1])
            angle = np.arccos(np.clip(np.dot(unit(upper), unit(lower)), -1, 1))
            desired[elbow] = desired[arm] @ Rotation.from_rotvec([0, -angle, 0]).as_matrix()
            desired[hand] = desired[elbow]
            thigh, shin, foot = [target_ids[side+'_'+n] for n in ('thigh', 'shin', 'foot')]
            for b, nxt, this in ((thigh, prefix+'Shin', prefix+'Leg'), (shin, prefix+'Foot', prefix+'Shin')):
                direction = unit(p[ids[nxt]]-p[ids[this]])
                child = shin if b == thigh else foot
                desired[b] = Rotation.align_vectors([direction, root_r[:, 1]],
                                                    [unit(offsets[child]), [0, 1, 0]], weights=[10, 1])[0].as_matrix()
            toe_dir = unit(p[ids[prefix+'ToeBase']]-p[ids[prefix+'Foot']])
            # Preserve foot pitch while selecting the most upward source basis axis.
            source_foot = basis @ src_rot[f, ids[prefix+'Foot']]
            initial_foot = basis @ source_rot[0, ids[prefix+'Foot']]
            axis = int(np.argmax(np.abs(initial_foot[2])))
            foot_up = source_foot[:, axis] * np.sign(initial_foot[2, axis])
            z = unit(foot_up - toe_dir*np.dot(foot_up, toe_dir))
            desired[foot] = np.column_stack((toe_dir, unit(np.cross(z, toe_dir)), z))
        qref = []
        for b, lo, hi in zip(DOF_BODIES, DOF_OFFSETS[:-1], DOF_OFFSETS[1:]):
            rv = Rotation.from_matrix(desired[parents[b]].T @ desired[b]).as_rotvec()
            qref.extend(rv if hi-lo == 3 else [rv[1]])
        qref = np.clip(qref, bounds[0]+1e-6, bounds[1]-1e-6)
        weights = np.ones((15, 1))
        weights[[5, 8, 11, 14]] = 3
        # SOMA pivot locations differ from this humanoid. Fitting its shoulder
        # and head pivots directly can create a spurious torso bend to shorten
        # the trunk; preserve anatomical orientation and fit end effectors.
        weights[[1, 2, 3, 6, 9, 12]] = 0
        weights[[4, 7]] = .5

        def residual(q):
            actual, rotations = fk(unpack(q), root_r, offsets, parents)
            orientation = Rotation.from_matrix(np.swapaxes(desired[[1, 2, 11, 14]], -1, -2) @ rotations[[1, 2, 11, 14]]).as_rotvec()
            orientation *= np.array([.2, .08, .06, .06])[:, None]
            return np.concatenate(((weights*(actual-targets[f])).ravel(), orientation.ravel(),
                                   .008*(q-qref), .035*(q-previous)))

        initial = qref if f == 0 else previous
        fit = least_squares(residual, initial, bounds=bounds, max_nfev=70, ftol=1e-6, xtol=1e-6)
        previous = fit.x
        joint_values.append(fit.x)
        local = unpack(fit.x)
        local[0] = root_r
        actual, rotations = fk(local, root_r, offsets, parents)
        root_rotations.append(root_r)
        all_rotations.append(local)
        all_positions.append(actual)
        residuals.append(np.linalg.norm(actual[[5, 8, 11, 14]]-targets[f, [5, 8, 11, 14]], axis=-1))
        if f % 30 == 0:
            print('IK frame', f, '/', len(times), flush=True)
    local_rot = np.stack(all_rotations)
    quats = Rotation.from_matrix(local_rot.reshape(-1, 3, 3)).as_quat().reshape(-1, 15, 4)
    # Continuous quaternion signs avoid artificial derivative flips.
    for f in range(1, len(quats)):
        flip = (quats[f]*quats[f-1]).sum(-1) < 0
        quats[f, flip] *= -1
    motion = SkeletonMotion.from_skeleton_state(SkeletonState.from_rotation_and_root_translation(
        tree, torch.tensor(quats, dtype=torch.float32), torch.tensor(root, dtype=torch.float32), is_local=True), fps=fps)
    mesh_min = []
    for side in ('right', 'left'):
        vertices = np.array([[float(x) for x in line.split()[1:4]] for line in
                            (XML.parent.parent/'smpl_foot_meshes'/f'{side}_foot.obj').read_text().splitlines() if line.startswith('v ')])
        b = target_ids[side+'_foot']
        matrices = Rotation.from_quat(motion.global_rotation[:, b].numpy()).as_matrix()
        zs = np.einsum('fi,vi->fv', matrices[:, 2], vertices) + motion.global_translation[:, b, 2].numpy()[:, None]
        mesh_min.append(zs.min(axis=1))
    floor = np.min(mesh_min, axis=0)
    # One constant shift, preserving vertical movement and root velocity.
    ground_shift = .005-float(floor.min())
    root[:, 2] += ground_shift
    motion = SkeletonMotion.from_skeleton_state(SkeletonState.from_rotation_and_root_translation(
        tree, torch.tensor(quats, dtype=torch.float32), torch.tensor(root, dtype=torch.float32), is_local=True), fps=fps)
    output.mkdir(parents=True, exist_ok=True)
    motion.to_file(str(output/'ref_motion.npy'))
    positions = motion.global_translation.numpy()
    joint_values = np.stack(joint_values)
    audit = dict(source=str(source.relative_to(ROOT)), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                 source_frames=len(source_pos), source_fps=1/dt, source_skeleton_joints=77,
                 source_crop_start_s=start, source_crop_end_s=end,
                 source_crop_frame_range_inclusive=[int(np.ceil(start/dt)), int(np.floor(end/dt))],
                 source_interpolation_support_frames_inclusive=[int(np.floor(start/dt)), int(np.ceil(end/dt))],
                 fps=fps, frames=len(times), duration_s=(len(times)-1)/fps, rotation_shape=list(quats.shape),
                 root_translation_shape=list(root.shape), target_skeleton=tree.node_names,
                 joint_mapping=MAPPING, scale=scale, ground_shift_m=ground_shift,
                 foot_mesh_floor_min_m=float((floor+ground_shift).min()),
                 foot_mesh_floor_quantiles_m=np.quantile(floor+ground_shift, [0, .1, .5, .9, 1]).tolist(),
                 key_body_fit_error_mean_m=np.mean(residuals, axis=0).tolist(),
                 key_body_fit_error_max_m=np.max(residuals, axis=0).tolist(),
                 key_body_error_order=['right_hand', 'left_hand', 'right_foot', 'left_foot'],
                 joint_limit_violation_max_rad=float(np.maximum(bounds[0]-joint_values, joint_values-bounds[1]).clip(0).max()),
                 max_dof_speed_rad_s=float(np.abs(np.diff(joint_values, axis=0)*fps).max()),
                 root_net_horizontal_m=float(np.linalg.norm(root[-1, :2]-root[0, :2])),
                 left_hand_open_height_m={'min': float(positions[times>=1.25932, 8, 2].min()),
                                          'median': float(np.median(positions[times>=1.25932, 8, 2])),
                                          'max': float(positions[times>=1.25932, 8, 2].max())},
                 notes=['No measured door pose/contact in source.', 'IK follows scaled source endpoints with humanoid DOF limits.',
                        'Crop includes opening and early forward motion; doorway crossing is not verified.',
                        'Door compatibility must be checked separately; no learned-policy success claim.'])
    if profile == 'push':
        audit.pop('left_hand_open_height_m')
        audit['profile'] = 'push'
        audit['hand_height_median_m'] = np.median(positions[:, [5, 8], 2], axis=0).tolist()
        audit['notes'] = ['No measured box pose/contact in source.',
                          'Human reference only; box state comes from simulation.',
                          'IK follows scaled endpoints with humanoid DOF limits.']
    (output/'conversion_report.json').write_text(json.dumps(audit, indent=2)+'\n')
    # Y-up canvas coordinates, consistent with the existing offline viewer.
    preview = dict(name='TokenHSI retarget · left hand / left opening', display_fps=fps,
                   nodes=[dict(name='Hips' if n=='pelvis' else n, parent=int(p)) for n,p in zip(tree.node_names, parents)],
                   positions=positions[:, :, [0, 2, 1]].round(5).tolist(),
                   events=[dict(start_time=0., end_time=1.25932-start, description='왼손을 뻗어 손잡이에 접근'),
                           dict(start_time=1.25932-start, end_time=end-start, description='왼손으로 왼쪽에 밀며 전진 — 문 접촉은 별도 확인 필요')])
    if profile == 'push':
        preview['name'] = source.stem + ' · humanoid push reference'
        preview['events'] = [dict(start_time=0., end_time=end-start, description='양손으로 앞 물체를 밀며 전진; 물체 접촉은 별도 검증')]
    (output/'preview_data.json').write_text(json.dumps([preview]))
    print(json.dumps(audit, indent=2))
    return motion, audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT/'bones_seed/soma_uniform/bvh/231115'/f'{SOURCE_NAME}.bvh')
    parser.add_argument('--output', type=Path, default=ROOT/'tokenhsi/data/dataset_bones_door/motions'/SOURCE_NAME/'phys_humanoid_v3')
    parser.add_argument('--start', type=float, default=0.)
    parser.add_argument('--end', type=float, default=3.4)
    args = parser.parse_args()
    convert(args.source.resolve(), args.output.resolve(), args.start, args.end)
