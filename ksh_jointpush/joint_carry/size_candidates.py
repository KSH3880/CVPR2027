from pathlib import Path
import hashlib
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation as R

from joint_carry.correction import correct, physical_rotations
from joint_carry.kinematics import pack, forward_kinematics
from joint_carry.leg_correction import retarget_legs
from tokenhsi.utils.grasp_geometry import ANCHOR_INSET
from joint_carry.single_source import paired_single, source_paths


def make_candidates(skill, size, source='single', inset=ANCHOR_INSET):
    size = np.round(np.asarray(size, dtype=float)*20)/20
    if source == 'single':
        data = paired_single(skill, size, inset)
        candidate, stats = correct(data)
        candidate['source_frame'] = np.arange(len(candidate['time']))
        candidate['terminal_box_xy'] = np.broadcast_to(candidate['box_position'][-1, :2],
                                                       (len(candidate['time']), 2)).copy()
        candidate['source_sha256'] = np.asarray(hashlib.sha256(b''.join(path.read_bytes() for path in source_paths(skill))).hexdigest())
        return candidate, stats
    path = Path('joint_carry/source_rsi')/skill/'cooperative_rsi_candidate.npz'
    files = [path, Path('joint_carry/leg_correction.py'), Path('joint_carry/kinematics.py'),
             Path('tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml')]
    signature = hashlib.sha256(b''.join(file.read_bytes() for file in files)).hexdigest()
    cached = Path('joint_carry/source_corrected')/skill/(signature+'.npz')
    if not cached.exists():
        with np.load(path, allow_pickle=False) as archive:
            original = dict(archive)
        repaired = retarget_legs(original)
        cached.parent.mkdir(parents=True, exist_ok=True)
        temporary = cached.with_name(str(os.getpid())+'.npz')
        np.savez_compressed(temporary, **repaired)
        os.replace(temporary, cached)
    with np.load(cached, allow_pickle=False) as archive:
        data = dict(archive)
    old_size = data['box_size'].copy()
    box_rotation = R.from_quat(data['box_rotation_xyzw'])
    inverse = box_rotation.inv()
    local_root = inverse.apply(data['root_position'][:, 0]-data['box_position'])
    side = np.sign(np.median(local_root[:, 1]))
    local_shift = np.zeros((len(local_root), 3))
    inward = min(.08, max(0., size[0]-old_size[0])*.12)
    inset_change = float(data['grip_inset_m']) - inset
    local_shift[:, 1] = side*((size[1]-old_size[1])/2 + inset_change-inward)
    first_shift = box_rotation.apply(local_shift)
    for human, shift in enumerate((first_shift, -first_shift)):
        data['root_position'][:, human] += shift
        data['body_position'][:, human] += shift[:, None]
    bottom = data["box_position"][:, 2]-old_size[2]/2
    carried = np.clip((bottom-.05)/.3, 0., 1.)
    floor_delta = (size[2]-old_size[2])/2
    height_delta = floor_delta + carried*(max(0., size[0]-old_size[0])*.33-floor_delta)
    data['box_position'][:, 2] += height_delta
    hands = [data['joint_names'].tolist().index(name) for name in ('left_hand', 'right_hand')]
    data['body_position'][:, :, hands, 2] += height_delta[:, None, None]
    anchor_shift = np.zeros((len(local_root), 3))
    anchor_shift[:, 1] = side*inward
    anchor_shift = box_rotation.apply(anchor_shift)
    data['body_position'][:, 0, hands] += anchor_shift[:, None]
    data['body_position'][:, 1, hands] -= anchor_shift[:, None]
    data['box_size'] = size
    data['grip_inset_m'] = np.asarray(inset)
    if size[0] > old_size[0]:
        targets = data['body_position'][:, :, hands].copy()
        q = physical_rotations(data)
        positions, world = forward_kinematics(q[:, 0], data['root_position'][:, 0], data['parent_indices'], data['local_translation'])
        line = inverse.apply(positions[:, 3]-positions[:, 6])
        angle = -np.arctan2(line[:, 1], line[:, 0])
        yaw = R.from_rotvec(np.stack((angle*0, angle*0, angle), -1))
        torso = box_rotation*yaw*inverse*R.from_quat(world[:, 1])
        q[:, 0, 1] = (R.from_quat(q[:, 0, 0]).inv()*torso).as_quat()
        q[:, 1, 1:] = q[:, 0, 1:]
        revised = pack(q, data['root_position'], data['parent_indices'], data['local_translation'],
                       np.concatenate((data['box_position'], data['box_rotation_xyzw']), -1), size, int(data['fps']), data['joint_names'].tolist())
        for key in ('skill', 'source_rsi_allowed', 'grip_inset_m'):
            revised[key] = data[key]
        revised['body_position'][:, :, hands] = targets
        data = revised
    # Bounded knee/foot IK is cached once per original clip, then preserved
    # while the arms are fitted against each actual box size.
    candidate, stats = correct(data)
    candidate['source_frame'] = np.arange(len(candidate['time']))
    candidate['terminal_box_xy'] = np.broadcast_to(candidate['box_position'][-1, :2],
                                                   (len(candidate['time']), 2)).copy()
    candidate['source_sha256'] = np.asarray(__import__('hashlib').sha256(path.read_bytes()).hexdigest())
    return candidate, stats
