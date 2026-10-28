from functools import lru_cache
from pathlib import Path
import hashlib
import os

import numpy as np
from scipy.spatial.transform import Rotation as R
import yaml

from tokenhsi.utils.grasp_geometry import grasp_y
from joint_carry.kinematics import pack
from joint_carry.leg_correction import retarget_legs


@lru_cache(None)
def source_paths(skill):
    manifest = Path('tokenhsi/data/dataset_loco_sit_carry_climb.yaml')
    tag = {'pickUp': 'B19_', 'carryWith': 'B20_', 'putDown': 'B21_'}[skill]
    entry = next(value for value in yaml.safe_load(manifest.read_text())['motions'][skill] if tag in value['file'])
    return manifest.parent / entry['file'], manifest.parent / entry['obj_file']


def load_single(skill):
    motion_path, box_path = source_paths(skill)
    raw = np.load(motion_path, allow_pickle=True).item()
    tree = raw['skeleton_tree']
    result = pack(raw['rotation']['arr'][:, None].astype(float),
                  raw['root_translation']['arr'][:, None].astype(float),
                  tree['parent_indices']['arr'], tree['local_translation']['arr'],
                  np.load(box_path).astype(float), [.4, .4, .4], int(raw['fps']), tree['node_names'])
    result['skill'] = np.asarray(skill)
    result['grip_inset_m'] = np.asarray(0.)
    result['source_rsi_allowed'] = np.ones(len(result['time']), dtype=bool)
    return result


def corrected_single(skill):
    paths = list(source_paths(skill)) + [Path(__file__), Path('joint_carry/leg_correction.py'),
             Path('joint_carry/kinematics.py'), Path('tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml')]
    signature = hashlib.sha256(b''.join(path.read_bytes() for path in paths)).hexdigest()
    cache = Path('joint_carry/source_corrected/single') / skill / (signature + '.npz')
    if not cache.exists():
        data = retarget_legs(load_single(skill))
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_name(str(os.getpid()) + '.npz')
        np.savez_compressed(temporary, **data)
        os.replace(temporary, cache)
    with np.load(cache, allow_pickle=False) as archive:
        return dict(archive)


def paired_single(skill, size, inset):
    single = corrected_single(skill)
    q = single['local_rotation_xyzw'][:, 0]
    root = single['root_position'][:, 0]
    box_rotation = R.from_quat(single['box_rotation_xyzw'])
    box_position = single['box_position'].copy()
    bottom = box_position[:, 2] - .2
    carried = np.clip((bottom - .05) / .3, 0., 1.)
    height_delta = (size[2] - .4) / 2 * (1 - carried)
    box_position[:, 2] += height_delta
    shift = np.zeros((len(q), 3))
    shift[:, 1] = grasp_y(size[1] / 2, -1., inset)
    first = root + box_rotation.apply(shift)
    half_turn = R.from_rotvec([0, 0, np.pi])
    second = box_position + half_turn.apply(first - box_position)
    clone = q.copy()
    clone[:, 0] = (half_turn * R.from_quat(q[:, 0])).as_quat()
    paired = pack(np.stack((q, clone), 1), np.stack((first, second), 1),
                  single['parent_indices'], single['local_translation'],
                  np.concatenate((box_position, single['box_rotation_xyzw']), -1),
                  size, int(single['fps']), single['joint_names'].tolist())
    hands = [single['joint_names'].tolist().index(name) for name in ('left_hand', 'right_hand')]
    paired['body_position'][:, :, hands, 2] += height_delta[:, None, None]
    paired['skill'] = np.asarray(skill)
    paired['grip_inset_m'] = np.asarray(inset)
    paired['source_rsi_allowed'] = single['source_rsi_allowed']
    return paired


def source_manifest():
    records = {}
    for skill in ('pickUp', 'carryWith', 'putDown'):
        records[skill] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in source_paths(skill)}
    return records
