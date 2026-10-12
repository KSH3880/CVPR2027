import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation as R

from carry_joint_preview import forward_kinematics, angular_velocity

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'joint_carry/teamhoi_reference'
OUTPUT = ROOT / 'joint_carry/teamhoi_retarget_pilot'
ASSET = ROOT / 'tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml'
CLIPS = {
    'backward': 'loco_reverse/ACCAD_Male1Walking_c3d_Walk_B10_-_Walk_turn_left_45_stageii.npy',
    'sideways': 'sidewalk/cmu_sidewalk_141_33_poses.npy',
}


def select_clips(all_clips, posture_first=False):
    if posture_first:
        if all_clips:
            records = json.loads((SOURCE/'manifest.json').read_text())['clips']
            return ROOT/'joint_carry/teamhoi_retarget_posture_all', {Path(c['file']).stem:c['file'] for c in records}
        return ROOT/'joint_carry/teamhoi_retarget_posture_pilot', CLIPS
    if all_clips:
        records = json.loads((SOURCE/'manifest.json').read_text())['clips']
        return ROOT/'joint_carry/teamhoi_retarget_all', {Path(c['file']).stem:c['file'] for c in records}
    return OUTPUT, CLIPS


def foot_vertices(name):
    path = ASSET.parent / '../smpl_foot_meshes' / (name + '.obj')
    return np.array([[float(v) for v in line.split()[1:4]]
                     for line in path.read_text().splitlines() if line.startswith('v ')])


def lowest_vertex(quaternions, vertices):
    matrices = R.from_quat(quaternions).as_matrix()
    return np.einsum('fij,vj->fvi', matrices, vertices)[:, :, 2].min(1)


def midpoint_motion(rotations, roots):
    a = R.from_quat(rotations[:-1].reshape(-1, 4))
    b = R.from_quat(rotations[1:].reshape(-1, 4))
    mid = a*R.from_rotvec((a.inv()*b).as_rotvec()*.5)
    return mid.as_quat().reshape(rotations[:-1].shape), (roots[:-1]+roots[1:])*.5


def retarget(kind, filename, output_dir=OUTPUT, posture_first=False):
    raw = np.load(SOURCE / filename, allow_pickle=True).item()
    tree = raw['skeleton_tree']
    names = tree['node_names']
    parents = tree['parent_indices']['arr']
    xml = ET.parse(ASSET).getroot()
    bodies = {b.get('name'): b for b in xml.findall('.//worldbody//body')}
    offsets = np.array([np.fromstring(bodies[n].get('pos', '0 0 0'), sep=' ') for n in names])
    rotations = raw['rotation']['arr'].astype(np.float64)
    roots = raw['root_translation']['arr'].astype(np.float64)
    source_pos, source_q = forward_kinematics(rotations, roots, parents, tree['local_translation']['arr'])
    fps = raw['fps']
    frames = len(roots)
    source_roots = roots.copy()
    source_knees = R.from_quat(rotations[:, [10, 13]].reshape(-1, 4)).as_rotvec()[:, 1]
    joint_limits = {}
    upper_clip = 0.
    for j, name in enumerate(names):
        joints = bodies[name].findall('joint')
        if j == 0:
            continue
        if not joints:
            rotations[:, j] = [0, 0, 0, 1]
            continue
        bounds = np.deg2rad([np.fromstring(joint.get('range'), sep=' ') for joint in joints])
        joint_limits[j] = bounds
        rv = R.from_quat(rotations[:, j]).as_rotvec()
        if len(joints) == 1:
            rv[:, [0, 2]] = 0
            rv[:, 1] = np.clip(rv[:, 1], *bounds[0])
        elif j < 9:
            clipped = np.clip(rv, bounds[:, 0], bounds[:, 1])
            upper_clip = max(upper_clip, float(np.abs(clipped-rv).max()))
            rv = clipped
        rotations[:, j] = R.from_rotvec(rv).as_quat()

    if posture_first:
        positions, world_q = forward_kinematics(rotations, roots, parents, offsets)
        soles = np.stack([positions[:, j, 2]+lowest_vertex(world_q[:, j], foot_vertices(names[j]))
                          for j in (11, 14)], axis=1)
        roots[:, 2] += .005+.008*logsumexp(-soles/.008, axis=1)

    box = np.array([[x, y, z] for x in (-.0435, .1335)
                    for y in (-.045, .045) for z in (-.05, .005)])
    source_floor = np.stack([source_pos[:, j, 2]+lowest_vertex(source_q[:, j], box)
                             for j in (11, 14)], axis=1)
    floor_offset = float(np.quantile(source_floor.min(1), .02))
    source_speed = np.linalg.norm(np.gradient(source_pos[:, [11, 14], :2], axis=0)*fps, axis=-1)
    clearance = np.maximum(gaussian_filter1d(source_floor-floor_offset, 1., axis=0), .005)
    swing = np.clip((source_speed-source_speed[:, ::-1]-.15)/.35, 0, 1)
    swing *= np.clip((source_speed-.35)/.25, 0, 1)
    raised = clearance+swing*np.maximum(clearance[:, ::-1]+.035-clearance, 0)
    swing_raise = float((raised-clearance).max())
    clearance = gaussian_filter1d(raised, 1., axis=0)
    if posture_first:
        clearance -= (-.008*logsumexp(-clearance/.008, axis=1)-.005)[:, None]
    contacts = np.zeros((frames, 2), dtype=bool)
    targets = np.zeros((frames, 2, 3))
    errors = []
    for side, (hip, knee, foot) in enumerate(((9, 10, 11), (12, 13, 14))):
        target = source_pos[:, foot].copy()
        target[:, 2] = clearance[:, side]-lowest_vertex(source_q[:, foot], foot_vertices(names[foot]))
        speed = source_speed[:, side]
        contacts[:, side] = (source_floor[:, side]-floor_offset < .035) & (speed < .35)
        targets[:, side] = target
        bounds = np.concatenate([joint_limits[j] for j in (hip, knee, foot)])
        margin = .001 if posture_first else 1e-6
        lower, upper = bounds[:, 0]+margin, bounds[:, 1]-margin
        reference = np.concatenate([R.from_quat(rotations[:, j]).as_rotvec() for j in (hip, knee, foot)], axis=1)
        previous = np.clip(reference[0], lower, upper)
        for f in range(frames):
            root_rotation = R.from_quat(rotations[f, 0])
            hip_position = roots[f]+root_rotation.apply(offsets[hip])
            target_rotation = R.from_quat(source_q[f, foot])

            def residual(x):
                local = R.from_rotvec(x.reshape(3, 3))
                world_hip = root_rotation*local[0]
                world_knee = world_hip*local[1]
                position = hip_position+world_hip.apply(offsets[knee])+world_knee.apply(offsets[foot])
                orientation_error = (target_rotation.inv()*world_knee*local[2]).as_rotvec()
                if posture_first:
                    return np.concatenate(((position-target[f])/.005, orientation_error/.03,
                                           3.*(x-reference[f]), .08*(x-previous),
                                           [(x[4]-reference[f, 4])/np.deg2rad(5)]))
                return np.concatenate(((position-target[f])/.003, orientation_error/.03,
                                       .12*(x-reference[f]), .08*(x-previous)))

            fit = least_squares(residual, previous, bounds=(lower, upper), max_nfev=60,
                                ftol=1e-7, xtol=1e-7, gtol=1e-7)
            rotations[f, [hip, knee, foot]] = R.from_rotvec(fit.x.reshape(3, 3)).as_quat()
            previous = fit.x
            if f % 150 == 0:
                print(kind, names[hip], f, '/', frames, flush=True)
        positions, world_q = forward_kinematics(rotations, roots, parents, offsets)
        errors.append(np.linalg.norm(positions[:, foot]-target, axis=-1))

    positions, world_q = forward_kinematics(rotations, roots, parents, offsets)
    sole_heights = np.stack([positions[:, j, 2]+lowest_vertex(world_q[:, j], foot_vertices(names[j]))
                            for j in (11, 14)], axis=1)
    ground_shift = -.008*logsumexp(-sole_heights/.008, axis=1)-.005
    roots[:, 2] -= ground_shift
    targets[:, :, 2] -= ground_shift[:, None]
    positions, world_q = forward_kinematics(rotations, roots, parents, offsets)
    source_frames, source_fps = frames, fps
    mid_q, mid_root = midpoint_motion(rotations, roots)
    mid_pos, _ = forward_kinematics(mid_q, mid_root, parents, offsets)
    interpolation_error = np.linalg.norm(mid_pos[:, [5, 8, 11, 14]]-
        (positions[:-1, [5, 8, 11, 14]]+positions[1:, [5, 8, 11, 14]])*.5, axis=-1).max()
    if interpolation_error > .005:
        dense_q = np.empty((frames*2-1, 15, 4));dense_root = np.empty((frames*2-1, 3))
        dense_q[::2] = rotations;dense_q[1::2] = mid_q
        dense_root[::2] = roots;dense_root[1::2] = mid_root
        rotations, roots = dense_q, dense_root
        frames, fps = len(roots), fps*2
        positions, world_q = forward_kinematics(rotations, roots, parents, offsets)
    raw['rotation']['arr'] = rotations.astype(np.float32)
    raw['root_translation']['arr'] = roots.astype(np.float32)
    tree['local_translation']['arr'] = offsets.astype(np.float32)
    linear_velocity = np.diff(positions, axis=0)*fps
    raw['global_velocity']['arr'] = np.concatenate((linear_velocity, linear_velocity[-1:])).astype(np.float32)
    raw['global_angular_velocity']['arr'] = angular_velocity(world_q, fps).astype(np.float32)
    raw['fps'] = fps
    output = output_dir / (kind + '.npy')
    np.save(output, raw)
    np.savez_compressed(output_dir / (kind + '_targets.npz'), contacts=contacts, targets=targets,
                        source_positions=source_pos, source_quaternions=source_q, fps=source_fps)
    floors = np.stack([positions[:, j, 2]+lowest_vertex(world_q[:, j], foot_vertices(names[j]))
                       for j in (11, 14)], axis=1)
    source_grid = positions[::fps//source_fps, [11, 14], :2]
    knees = R.from_quat(rotations[::fps//source_fps, [10, 13]].reshape(-1, 4)).as_rotvec()[:, 1]
    foot_speed = np.linalg.norm(np.gradient(source_grid, axis=0)*source_fps, axis=-1)
    return dict(kind=kind, source=filename, source_sha256=hashlib.sha256((SOURCE/filename).read_bytes()).hexdigest(),
                output=str(output.relative_to(ROOT)), output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                frames=frames, fps=fps, source_frames=source_frames, source_fps=source_fps,
                pre_densification_key_interpolation_error_m=float(interpolation_error), root_xy_path_changed=False,
                root_height_adjustment_max_m=float(np.abs(roots[::fps//source_fps, 2]-source_roots[:, 2]).max()),
                knee_median_deg=float(np.rad2deg(np.median(knees))),
                source_knee_median_deg=float(np.rad2deg(np.median(source_knees))),
                knee_change_p95_deg=float(np.rad2deg(np.quantile(np.abs(knees-source_knees), .95))),
                swing_clearance_adjustment_max_m=swing_raise,
                source_floor_offset_m=floor_offset, upper_joint_clipping_max_rad=upper_clip,
                foot_target_error_max_m=float(np.max(errors)), floor_min_m=float(floors.min()),
                source_contact_samples=int(contacts.sum()),
                contact_foot_origin_speed_mean_mps=float(foot_speed[contacts].mean()),
                contact_foot_origin_speed_p95_mps=float(np.quantile(foot_speed[contacts], .95)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true', help='Convert all 12 clips into a separate directory')
    parser.add_argument('--posture-first', action='store_true')
    args = parser.parse_args()
    output_dir, clips = select_clips(args.all, args.posture_first)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {'asset':str(ASSET.relative_to(ROOT)), 'asset_sha256':hashlib.sha256(ASSET.read_bytes()).hexdigest(),
              'method':'Bounded leg IK preserves source foot XY/orientation and root XY; velocity-based swing clearance; smooth minimum sole height grounds root Z; recomputed FK velocities',
              'posture_first':args.posture_first,
              'clips':[retarget(kind, path, output_dir, args.posture_first) for kind, path in clips.items()],
              'training_connected':False, 'validation_status':'not_run'}
    if args.posture_first:
        report['method'] = 'Ground source posture and foot targets before bounded IK; preserve knee pose with a 5-degree residual scale; recompute target-body FK and velocities'
    (output_dir/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    if args.all:
        motions = {group:[{'file':Path(c['output']).name, 'weight':1.} for c in report['clips']
                          if c['source'].startswith(prefix)]
                   for group, prefix in (('backward', 'loco_reverse/'), ('sideways', 'sidewalk/'))}
        (output_dir/'amp_motions.yaml').write_text(yaml.safe_dump({'motions':motions}, sort_keys=False))
    print(json.dumps(report, indent=2), flush=True)
