"""Select sustained arms-forward native PUSH RSI, preserving an existing safe cache.

CPU only. Source poses, AMP motions, loco sampling and existing caches are untouched.
This is a posture proxy: the source dataset has no recorded box/contact trajectory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import yaml
from scipy.spatial.transform import Rotation


def forward_kinematics(motion):
    tree = motion['skeleton_tree']
    parents = tree['parent_indices']['arr']
    local = motion['rotation']['arr']
    positions = np.zeros((*local.shape[:2], 3))
    rotations = np.zeros_like(local, dtype=float)
    for joint, parent in enumerate(parents):
        if parent < 0:
            positions[:, joint] = motion['root_translation']['arr']
            rotations[:, joint] = local[:, joint]
        else:
            parent_rotation = Rotation.from_quat(rotations[:, parent])
            positions[:, joint] = positions[:, parent] + parent_rotation.apply(
                np.broadcast_to(tree['local_translation']['arr'][joint], (len(local), 3)))
            rotations[:, joint] = (parent_rotation * Rotation.from_quat(local[:, joint])).as_quat()
    return positions, rotations


def sustained_windows(mask, fps, minimum_seconds=.5, trim_seconds=.2):
    """Require >=.5 s after trimming .2 s at BOTH ends of each run."""
    trim = int(np.ceil(trim_seconds * fps))
    minimum = int(np.ceil(minimum_seconds * fps))
    result = np.zeros_like(mask, dtype=bool)
    intervals = []
    for start, stop in np.flatnonzero(np.diff(np.r_[False, mask, False])).reshape(-1, 2):
        start, stop = int(start) + trim, int(stop) - trim
        if stop - start >= minimum:
            result[start:stop] = True
            intervals.append([start, stop])  # exclusive stop
    return result, intervals


def select(source, manifest, output):
    output.mkdir(parents=True, exist_ok=True)
    with np.load(source, allow_pickle=False) as archive:
        payload = {key: archive[key].copy() for key in archive.files}
    original = payload['valid'].copy()
    offsets = payload['clip_offsets']
    motions = yaml.safe_load(manifest.read_text())['motions']['push']
    assert len(motions) == len(offsets) - 1
    selected = np.zeros(original.shape[1], dtype=bool)
    all_metrics, rows = [], []
    for clip, item in enumerate(motions):
        path = manifest.parent / item['file']
        motion = np.load(path, allow_pickle=True).item()
        assert motion['is_local']
        assert len(motion['rotation']['arr']) == offsets[clip+1] - offsets[clip]
        body, rotation = forward_kinematics(motion)
        names = motion['skeleton_tree']['node_names']
        hands = [names.index(side+'_hand') for side in ('right', 'left')]
        shoulders = [names.index(side+'_upper_arm') for side in ('right', 'left')]
        forward = Rotation.from_quat(rotation[:, names.index('torso')]).apply(
            np.broadcast_to([1., 0., 0.], (len(body), 3)))
        forward[:, 2] = 0.
        forward /= np.linalg.norm(forward, axis=1)[:, None]
        reach = ((body[:, hands]-body[:, shoulders])*forward[:, None]).sum(-1).min(-1)
        relative_height = (body[:, hands, 2]-body[:, shoulders, 2]).min(-1)
        metrics = np.stack((reach, relative_height), -1)
        candidate = np.isfinite(metrics).all(-1) & (reach >= .25) & (relative_height >= -.25)
        # Screen the intersection, so a narrow height-limit crossing cannot
        # reintroduce a brief transition even inside a long arms-forward phase.
        candidate &= original[:, offsets[clip]:offsets[clip+1]].all(0)
        mask, windows = sustained_windows(candidate, motion['fps'])
        selected[offsets[clip]:offsets[clip+1]] = mask
        all_metrics.append(metrics)
        rows.append(dict(clip=clip, file=item['file'], source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            fps=motion['fps'], original_frames=int(original[:,offsets[clip]:offsets[clip+1]].any(0).sum()),
            selected_frames=int(mask.sum()), windows_frames_exclusive=windows,
            windows_seconds_inclusive=[[a/motion['fps'], (b-1)/motion['fps']] for a,b in windows]))
    payload['valid'] &= selected[None]
    assert payload['valid'].any(1).all(), 'No fallback to preparation poses is allowed'
    contract = dict(kind='sustained_bilateral_arms_forward', min_each_hand_forward_of_shoulder_m=.25,
        min_each_hand_z_relative_to_shoulder_m=-.25, minimum_kept_run_seconds=.5,
        trim_each_end_seconds=.2, all_size_profiles_required=True, retarget=False)
    meta = json.loads(str(payload['metadata']))
    meta['posture_selection'] = contract
    meta['posture_filter_source_cache_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    meta['phase'] = 'sustained bilateral arms-forward frames only; posture proxy, no recorded box/contact'
    payload['metadata'] = np.array(json.dumps(meta, sort_keys=True))
    payload['posture_eligible'] = selected
    payload['posture_metrics'] = np.concatenate(all_metrics)
    target = output/('stage1_push_depth180_rsi_le1m_push_phase_cache.npz' if 'height_selection' in meta else 'stage1_push_depth180_native_push_phase_cache.npz')
    if target.exists():
        raise FileExistsError(target)
    np.savez_compressed(target, **payload)
    report = dict(contract=contract, source_cache=str(source), source_cache_sha256=meta['posture_filter_source_cache_sha256'],
        selected_cache=str(target), selected_cache_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        source_size_frames=int(original.sum()), selected_size_frames=int(payload['valid'].sum()),
        unique_selected_frames=int(selected.sum()), per_profile=payload['valid'].sum(1).tolist(),
        selected_clips=[row['clip'] for row in rows if row['selected_frames']], motions=rows,
        loco_changed=False, source_motion_changed=False, physics_screen='Inherited from unchanged source cache; not rerun by this CPU selector',
        limitation='Posture selection, not measured contact or policy stability; source motions have no box trajectories')
    (output/'push_phase_selection.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='motions'}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    select(args.source, args.manifest, args.output)
