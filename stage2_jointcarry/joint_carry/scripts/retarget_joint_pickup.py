import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

from carry_joint_preview import forward_kinematics, pack
from correct_carry_candidates import correct
from retarget_teamhoi_reference import foot_vertices, lowest_vertex


def retarget(source, output):
    data = dict(np.load(source, allow_pickle=False))
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
        print(names[hip], 'complete', flush=True)

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
    candidate, arm_stats = correct(candidate)
    candidate['original_source_rsi_allowed'] = data['source_rsi_allowed']
    candidate['render_caption'] = np.asarray('Bounded leg IK + arm refit | all 100 source frames')
    candidate['render_footer'] = np.asarray('Kinematic reference | physical screening reported separately')
    output.mkdir(parents=True, exist_ok=True)
    target = output/'cooperative_rsi_candidate.npz'
    np.savez_compressed(target, **candidate)
    bounds = np.deg2rad([np.fromstring(j.get('range'), sep=' ')
                        for name in names[1:] for j in bodies[name].findall('joint')])
    excess = np.maximum(np.maximum(bounds[:, 0]-candidate['dof_position'],
                                   candidate['dof_position']-bounds[:, 1]), 0.)
    report = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  output_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                  frames=len(q), fps=int(data['fps']), box_size=data['box_size'].tolist(),
                  leg_foot_target_error_max_m=max(errors), extra_root_lift_m=lift,
                  maximum_joint_limit_violation_rad=float(excess.max()),
                  arm_stats=arm_stats, training_connected=False, physics_validated=False)
    (output/'retarget_report.json').write_text(json.dumps(report, indent=2)+'\n')
    print({k: v for k, v in report.items() if k != 'arm_stats'}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('joint_carry/carry_joint_corrected/pickUp/cooperative_rsi_candidate.npz'))
    parser.add_argument('--output', type=Path, default=Path('output/joint_pickup_full_20261012/pickUp'))
    args = parser.parse_args()
    retarget(args.source, args.output)
