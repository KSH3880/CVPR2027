"""SOMA input adapter for TokenHSI's rotation-transfer preprocessing.

No IK, interaction crop, height target, clip filtering, or RSI pose edits.
The original project_joints_simple function is imported unchanged.
"""
from pathlib import Path
import argparse
import hashlib
import inspect
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from isaacgym import gymapi  # noqa: F401; before torch
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import yaml
from lpanlib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from tokenhsi.data.data_utils import project_joints_simple
from tokenhsi.utils.bones_bvh import read_bvh

MAPPING = {
    'pelvis': 'Hips', 'torso': 'Spine2', 'head': 'Head',
    'right_upper_arm': 'RightArm', 'right_lower_arm': 'RightForeArm', 'right_hand': 'RightHand',
    'left_upper_arm': 'LeftArm', 'left_lower_arm': 'LeftForeArm', 'left_hand': 'LeftHand',
    'right_thigh': 'RightLeg', 'right_shin': 'RightShin', 'right_foot': 'RightFoot',
    'left_thigh': 'LeftLeg', 'left_shin': 'LeftShin', 'left_foot': 'LeftFoot',
}
# SOMA rest: forward +Z, left +X, up +Y -> target forward +X, left +Y, up +Z.
BASIS = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transfer_rotations(source_global, source_root, tree):
    """Same global-rotation/T-pose transfer as original generate_motion.py.

    Only the source-coordinate adapter differs: SOMA is not SMPL. Target
    arm T-pose rotations, floor rule, +0.07m and projection match upstream.
    """
    tpose = SkeletonState.zero_pose(tree)
    local = tpose.local_rotation.clone()
    for side, angle in (('left', np.pi / 2), ('right', -np.pi / 2)):
        i = tree.index(side + '_upper_arm')
        local[i] = torch.tensor([np.sin(angle / 2), 0., 0., np.cos(angle / 2)], dtype=torch.float32)
    tpose = SkeletonState.from_rotation_and_root_translation(tree, local, tpose.root_translation, is_local=True)
    target_rest = Rotation.from_quat(tpose.global_rotation.numpy()).as_matrix()
    global_rotation = BASIS @ source_global @ BASIS.T @ target_rest[None]
    quats = Rotation.from_matrix(global_rotation.reshape(-1, 3, 3)).as_quat().reshape(len(source_root), 15, 4)
    # Quaternion signs affect representation only, not the source motion.
    for f in range(1, len(quats)):
        flip = (quats[f] * quats[f - 1]).sum(-1) < 0
        quats[f, flip] *= -1
    root = source_root @ BASIS.T
    state = SkeletonState.from_rotation_and_root_translation(
        tree, torch.tensor(quats, dtype=torch.float32), torch.tensor(root, dtype=torch.float32), is_local=False).local_repr()
    motion = SkeletonMotion.from_skeleton_state(state, fps=30)
    minimum = motion.global_translation[:, :, 2].min(dim=-1)[0].mean()
    translation = motion.root_translation.clone()
    translation[:, 2] += -minimum + .07
    adjusted = SkeletonMotion.from_skeleton_state(SkeletonState.from_rotation_and_root_translation(
        tree, motion.local_rotation.clone(), translation, is_local=True), fps=30)
    result = project_joints_simple(adjusted)
    return result, adjusted, float(-minimum + .07)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'tokenhsi/data/dataset_bones_push_tokenhsi')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'tokenhsi/data/dataset_push_tokenhsi.yaml')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.output.exists() or args.manifest.exists():
        raise FileExistsError('Choose new output/manifest paths; existing datasets are immutable')
    original = ROOT / 'tokenhsi/data/dataset_amass_loco/generate_motion.py'
    helpers = ROOT / 'tokenhsi/data/data_utils.py'
    upstream = Path('/home/hwanhee/CVPR2027/TokenHSI')
    for file in (original, helpers):
        assert sha(file) == sha(upstream / file.relative_to(ROOT)), file
    tree = SkeletonTree.from_mjcf(str(ROOT / 'tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml'))
    selected = json.loads((ROOT / 'tokenhsi/data/bones_push180_selection.json').read_text())['clips']
    manifest = yaml.safe_load((ROOT / 'tokenhsi/data/dataset_push_only_push180.yaml').read_text())
    source_manifest_hash = sha(ROOT / 'tokenhsi/data/dataset_push_only_push180.yaml')
    ordered_names = [Path(row['file']).parents[1].name for row in manifest['motions']['push']]
    assert set(ordered_names) == set(selected) and len(ordered_names) == 21
    reports = []
    manifest['motions']['push'] = []
    for name in ordered_names:
        sources = list((ROOT / 'bones_seed/soma_uniform/bvh').rglob(name + '.bvh'))
        assert len(sources) == 1, name
        source = sources[0]
        nodes, pos, rot, dt = read_bvh(source)
        ids = {n['name']: i for i, n in enumerate(nodes)}
        skip = int((1 / dt) // 30)  # Exact upstream process_amass_seq rule.
        assert skip >= 1
        mapped = [ids[MAPPING[n]] for n in tree.node_names]
        rows = np.arange(0, len(pos), skip)
        motion, before_projection, floor_shift = transfer_rotations(rot[rows][:, mapped], pos[rows, ids['Hips']], tree)
        assert torch.isfinite(motion.global_translation).all() and torch.isfinite(motion.global_rotation).all()
        destination = args.output / 'motions' / name / 'phys_humanoid_v3'
        destination.mkdir(parents=True)
        motion.to_file(str(destination / 'ref_motion.npy'))
        # Audit the actual serialized data, not only its in-memory representation.
        loaded = SkeletonMotion.from_file(str(destination / 'ref_motion.npy'))
        torch.testing.assert_close(loaded.global_rotation, motion.global_rotation)
        torch.testing.assert_close(loaded.root_translation, motion.root_translation)
        assert int(loaded.fps) == 30 and len(loaded.root_translation) == len(rows)
        unchanged = [tree.index(n) for n in tree.node_names if 'arm' not in n and 'hand' not in n]
        torch.testing.assert_close(loaded.local_rotation[:, unchanged], before_projection.local_rotation[:, unchanged])
        for side in ('right', 'left'):
            q = loaded.local_rotation[:, tree.index(side + '_lower_arm')]
            assert q[:, [0, 2]].abs().max() < 1e-5
        report = dict(clip=name, source=str(source), source_sha256=sha(source), source_frames=len(pos),
            source_fps=1 / dt, stride=skip, selected_source_first=0, selected_source_last=int(rows[-1]),
            full_sequence=True, interaction_crop=False, ik=False, root_scale=1.,
            source_basis=BASIS.tolist(), mapping=MAPPING, fps=30, frames=len(rows),
            source_duration_s=(len(pos)-1)*dt, output_duration_s=(len(rows)-1)/30,
            floor_rule='mean(frame_min_joint_z), then +0.07m; original TokenHSI loco rule',
            root_z_shift_m=floor_shift, project_joints_simple_sha256=hashlib.sha256(inspect.getsource(project_joints_simple).encode()).hexdigest(),
            projection_non_arm_rotations_unchanged=True, serialized_reload_equal=True,
            foot_joint_min_z_m=float(loaded.global_translation[:, [tree.index('right_foot'), tree.index('left_foot')], 2].min()))
        (destination / 'conversion_report.json').write_text(json.dumps(report, indent=2) + '\n')
        preview = dict(name=name + ' TokenHSI rotation transfer', display_fps=30,
            nodes=[dict(name=n, parent=int(p)) for n,p in zip(tree.node_names, tree.parent_indices)],
            positions=loaded.global_translation.numpy()[:, :, [0,2,1]].round(5).tolist())
        (destination / 'preview_data.json').write_text(json.dumps([preview]))
        manifest['motions']['push'].append(dict(file=str((destination / 'ref_motion.npy').relative_to(args.manifest.parent)), weight=1.))
        reports.append(report)
        print('CONVERTED', name, len(rows), flush=True)
    args.manifest.write_text(yaml.safe_dump(manifest, sort_keys=False))
    summary = dict(method='tokenhsi_rotation_transfer_soma_adapter_v1',
        original_generator_sha256=sha(original), original_data_utils_sha256=sha(helpers),
        upstream_files_identical=True, source_manifest_sha256=source_manifest_hash,
        adaptation='SOMA BVH reader, cm-to-m, source joints and rest-coordinate basis only',
        absent=['IK', 'semantic_crop', 'hand_height_target', 'RSI_PD_filter'],
        clips=reports, manifest=str(args.manifest), manifest_sha256=sha(args.manifest))
    (args.output / 'conversion_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('DONE', args.manifest, flush=True)


if __name__ == '__main__':
    main()
