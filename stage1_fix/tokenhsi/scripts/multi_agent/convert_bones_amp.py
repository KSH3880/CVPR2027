"""BONES Uniform -> phys_humanoid_v3, constrained IK, CPU only.

Adapted from koo_cvpr/stage1_fix convert_bones_door.py (2026-10-07).

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

CONVERTER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
PARSER_SHA256 = hashlib.sha256((ROOT/'tokenhsi/utils/bones_bvh.py').read_bytes()).hexdigest()
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



def convert(source, output, start=0., end=None, fps=30, profile='door'):
    torch.set_num_threads(1)
    nodes, source_pos, source_rot, dt = read_bvh(source)
    ids = {n['name']: i for i, n in enumerate(nodes)}
    prefix = {'door': ('inside_door_handle_left_side_open_walk_R_', 'inside_door_handle_right_side_open_walk_R_'), 'push': 'push_obstacle_180'}[profile]
    if len(nodes) != 78 or not source.stem.startswith(prefix):
        raise ValueError('Expected supported Uniform skeleton and task prefix')
    missing = set(MAPPING.values()) - set(ids)
    if missing:
        raise ValueError('Missing source joints: ' + str(sorted(missing)))
    if fps <= 0:
        raise ValueError('FPS must be positive')
    duration = (len(source_pos)-1)*dt
    if end is None:
        end = duration
    if not 0 <= start < end <= duration:
        raise ValueError('Invalid crop')
    times = np.arange(start, end + 1e-8, 1/fps)
    if len(times) < 2:
        raise ValueError('Crop must contain at least two output frames')
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
    left = (source_pos[0, ids['LeftArm']] - source_pos[0, ids['RightArm']]).copy()
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
    previous = np.zeros(32)
    residuals = []
    fit_success = []
    fit_nfev = []
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

        # Bound successive DOF steps to prevent equivalent IK branches from snapping.
        frame_bounds = bounds if f == 0 else np.stack((
            np.maximum(bounds[0], previous-12/fps), np.minimum(bounds[1], previous+12/fps)))
        initial = np.clip(qref if f == 0 else previous, frame_bounds[0]+1e-7, frame_bounds[1]-1e-7)
        fit = least_squares(residual, initial, bounds=frame_bounds, max_nfev=70, ftol=1e-6, xtol=1e-6)
        actual, _ = fk(unpack(fit.x), root_r, offsets, parents)
        endpoint_error = np.linalg.norm(actual[[5,8,11,14]]-targets[f,[5,8,11,14]], axis=-1).max()
        # Anatomical orientation seeds can start in a constrained local minimum.
        # Compare independent feasible seeds at the first frame or a poor fit.
        if f == 0 or endpoint_error > .05:
            neutral = np.clip(np.zeros(32), frame_bounds[0]+1e-7, frame_bounds[1]-1e-7)
            seeds = [neutral] if f == 0 else [qref, neutral]
            for seed in seeds:
                seed = np.clip(seed, frame_bounds[0]+1e-7, frame_bounds[1]-1e-7)
                candidate = least_squares(residual, seed, bounds=frame_bounds,
                                          max_nfev=100, ftol=1e-6, xtol=1e-6)
                if np.dot(candidate.fun,candidate.fun) < np.dot(fit.fun,fit.fun):
                    fit = candidate
        fit_success.append(bool(fit.success))
        fit_nfev.append(int(fit.nfev))
        previous = fit.x
        joint_values.append(fit.x)
        local = unpack(fit.x)
        local[0] = root_r
        actual, rotations = fk(local, root_r, offsets, parents)
        all_rotations.append(local)
        all_positions.append(actual)
        residuals.append(np.linalg.norm(actual[[5, 8, 11, 14]]-targets[f, [5, 8, 11, 14]], axis=-1))
        if f % 120 == 0:
            print(source.stem, 'IK frame', f, '/', len(times), flush=True)
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
    for data in (motion.local_rotation, motion.root_translation, motion.global_velocity, motion.global_angular_velocity):
        if not torch.isfinite(data).all():
            raise ValueError('Retarget produced nonfinite pose or velocity')
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
                 converter_sha256=CONVERTER_SHA256,
                 parser_sha256=PARSER_SHA256,
                 ik_dof_step_limit_rad_s=12.,
                 profile=profile, ik_converged_frames=sum(fit_success),
                 ik_total_frames=len(times), ik_max_nfev=max(fit_nfev),
                 hand_height_median_m=np.median(positions[:, [5, 8], 2], axis=0).tolist(),
                 notes=['Full clip retained by default; no task phase annotation inferred.',
                        'No measured object pose/contact; human reference only.',
                        'IK follows scaled source endpoints under humanoid DOF limits.',
                        'Geometry/physics and skill reward integration are separate.'])
    audit['review_flags'] = []
    if max(audit['key_body_fit_error_max_m']) > .05:
        audit['review_flags'].append('key-body fit error exceeds 5 cm')
    if audit['max_dof_speed_rad_s'] > 15:
        audit['review_flags'].append('DOF speed exceeds 15 rad/s')
    if audit['foot_mesh_floor_quantiles_m'][2] > .1:
        audit['review_flags'].append('median lowest foot clearance exceeds 10 cm')
    (output/'conversion_report.json').write_text(json.dumps(audit, indent=2)+'\n')
    preview = dict(name=source.stem, fps=fps, parents=parents.tolist(),
                   nodes=tree.node_names, positions=positions.round(5).tolist(),
                   source_positions=(targets+root[:, None]).round(5).tolist())
    (output/'preview_data.json').write_text(json.dumps(preview))
    print(source.stem, 'saved', len(times), 'frames; max endpoint error',
          round(max(audit['key_body_fit_error_max_m']), 4), 'm;', audit['review_flags'], flush=True)
    return motion, audit


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['all', 'door', 'push'], default='all')
    parser.add_argument('--clip', action='append', help='Exact BVH stem; repeat to select clips')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--skip-existing', action='store_true', help='Reuse only matching source/code/FPS/full-clip outputs')
    args = parser.parse_args()
    datasets = [('door', 'dataset_bones_dooropen', 'doorOpen'), ('push', 'dataset_bones_push', 'push')]
    for profile, folder, skill in datasets:
        if args.dataset not in ('all', profile):
            continue
        base = ROOT/'tokenhsi/data'/folder
        files = sorted(base.glob('*.bvh'))
        if args.clip:
            files = [p for p in files if p.stem in args.clip]
        if not files:
            raise ValueError('No selected BVH files in ' + str(base))
        entries, reports = [], []
        for source in files:
            output = base/'motions'/source.stem/'phys_humanoid_v3'
            report_path = output/'conversion_report.json'
            reuse = False
            if args.skip_existing and report_path.exists() and (output/'ref_motion.npy').exists():
                audit = json.loads(report_path.read_text())
                reuse = (audit.get('source_sha256') == hashlib.sha256(source.read_bytes()).hexdigest()
                    and audit.get('converter_sha256') == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                    and audit.get('parser_sha256') == hashlib.sha256((ROOT/'tokenhsi/utils/bones_bvh.py').read_bytes()).hexdigest()
                    and audit.get('fps') == args.fps and audit.get('source_crop_start_s') == 0)
            if not reuse:
                _, audit = convert(source, output, fps=args.fps, profile=profile)
            else:
                print('Reusing', source.stem, flush=True)
            reports.append(audit)
            entries.append(dict(file=str((output/'ref_motion.npy').relative_to(ROOT/'tokenhsi/data')),
                                weight=0.0 if audit['review_flags'] else 1.0))
        # A subset run must not replace the complete dataset manifest.
        suffix = '_selected' if args.clip else ''
        manifest = ROOT/'tokenhsi/data'/f'dataset_bones_{profile}_amp{suffix}.yaml'
        manifest.write_text('# Human references only; task/RSI object integration is separate.\n# Flagged clips are retained with weight 0 pending review; see conversion_summary.json.\n'
                            + yaml.safe_dump({'motions': {skill: entries}}, sort_keys=False))
        summary = base/('conversion_summary'+suffix+'.json')
        summary.write_text(json.dumps(dict(clips=len(reports), fps=args.fps, manifest=str(manifest.relative_to(ROOT)),
            flagged_clips=[r['source'] for r in reports if r['review_flags']], reports=reports), indent=2)+'\n')
        print('Manifest:', manifest, 'clips:', len(entries), flush=True)


if __name__ == '__main__':
    main()
