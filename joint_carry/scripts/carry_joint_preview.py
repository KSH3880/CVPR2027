"""CPU joint-motion preview and paired RSI *candidate* export; no physics rollout.

The clone faces the first person across an elongated box. Both retain the source
local joint rotations. A moving box-centred transform preserves the hand layout,
but does not preserve stance-foot velocities: this is deliberately not approved
RSI data or a two-person expert motion. Quaternions use xyzw, positions metres.
"""
import argparse
import itertools
import json
from pathlib import Path
import subprocess
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial.transform import Rotation as R
import yaml


DOF_BODIES = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
DOF_SIZES = [3, 3, 3, 1, 3, 1, 3, 3, 3, 3, 3, 3]
SKILL_TAGS = {'pickUp': 'B19_', 'carryWith': 'B20_', 'putDown': 'B21_'}
SKILL_TITLES = {'pickUp': 'PICK UP', 'carryWith': 'CARRY WITH', 'putDown': 'PUT DOWN'}


def forward_kinematics(local_q, root, parents, offsets):
    xyz = np.empty(local_q.shape[:-1] + (3,))
    rotations = []
    for j, parent in enumerate(parents):
        local = R.from_quat(local_q[:, j])
        rotations.append(local if parent < 0 else rotations[parent] * local)
        xyz[:, j] = root if parent < 0 else xyz[:, parent] + rotations[parent].apply(offsets[j])
    return xyz, np.stack([r.as_quat() for r in rotations], axis=1)


def angular_velocity(q, fps, local=False):
    """Quaternion finite differences; local=True matches MotionLib DOF convention."""
    shape = q.shape
    a, b = R.from_quat(q[:-1].reshape(-1, 4)), R.from_quat(q[1:].reshape(-1, 4))
    delta = a.inv() * b if local else b * a.inv()
    v = delta.as_rotvec().reshape((shape[0] - 1,) + shape[1:-1] + (3,)) * fps
    return np.concatenate([v, v[-1:]], axis=0)


def pack(local_q, roots, parents, offsets, box, size, fps, names):
    positions, rotations = zip(*(forward_kinematics(local_q[:, a], roots[:, a], parents, offsets)
                                for a in range(roots.shape[1])))
    positions, rotations = np.stack(positions, 1), np.stack(rotations, 1)
    angles = R.from_quat(local_q.reshape(-1, 4)).as_rotvec().reshape(local_q.shape[:-1] + (3,))
    local_vel = angular_velocity(local_q, fps, local=True)
    dof = np.concatenate([angles[:, :, j] if n == 3 else angles[:, :, j, 1:2]
                          for j, n in zip(DOF_BODIES, DOF_SIZES)], axis=-1)
    dof_vel = np.concatenate([local_vel[:, :, j] if n == 3 else local_vel[:, :, j, 1:2]
                              for j, n in zip(DOF_BODIES, DOF_SIZES)], axis=-1)
    return dict(fps=np.asarray(fps), time=np.arange(len(box)) / fps,
                joint_names=np.asarray(names), parent_indices=parents, local_translation=offsets,
                local_rotation_xyzw=local_q, root_position=roots,
                root_rotation_xyzw=local_q[:, :, 0], dof_position=dof, dof_velocity=dof_vel,
                body_position=positions, body_rotation_xyzw=rotations,
                body_linear_velocity=np.gradient(positions, 1 / fps, axis=0),
                body_angular_velocity=angular_velocity(rotations, fps),
                box_position=box[:, :3], box_rotation_xyzw=box[:, 3:], box_size=np.asarray(size),
                box_linear_velocity=np.gradient(box[:, :3], 1 / fps, axis=0),
                box_angular_velocity=angular_velocity(box[:, 3:], fps),
                physics_validated=np.asarray(False), rsi_ready=np.asarray(False))


def diagnostics(data):
    p = data['body_position']
    names = data['joint_names'].tolist()
    hands = p[:, :, [names.index('left_hand'), names.index('right_hand')]]
    inverse = R.from_quat(data['box_rotation_xyzw']).inv()
    relative = np.stack([inverse.apply(hands[:, a, h] - data['box_position'])
                         for a in range(p.shape[1]) for h in range(2)], axis=1)
    delta = np.abs(relative) - data['box_size'] / 2
    surface_distance = np.linalg.norm(np.maximum(delta, 0), axis=-1) + np.minimum(delta.max(-1), 0)
    feet = p[:, :, [names.index('left_foot'), names.index('right_foot')]]
    foot_v = data['body_linear_velocity'][:, :, [names.index('left_foot'), names.index('right_foot')]]
    low = feet[..., 2] < feet[..., 2].min(axis=0, keepdims=True) + .04
    speeds = np.linalg.norm(foot_v[..., :2], axis=-1)
    return dict(hand_joint_to_box_surface_abs_max_m=float(np.abs(surface_distance).max()),
                low_foot_horizontal_speed_mean_mps=[float(speeds[:, a][low[:, a]].mean())
                                                   for a in range(p.shape[1])],
                low_foot_note='Foot-origin height within 4cm of its per-clip minimum; proxy, not contact detection.',
                finite=all(bool(np.isfinite(v).all()) for v in data.values()
                           if isinstance(v, np.ndarray) and v.dtype.kind in 'fc'))


def box_vertices(position, quaternion, size):
    local = np.asarray(list(itertools.product([-1, 1], repeat=3))) * size / 2
    return R.from_quat(quaternion).apply(local) + position


def render(original, paired, output):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': '#dce5ef'})
    fig = plt.figure(figsize=(16, 10), facecolor='#111b29')
    axes = [fig.add_subplot(2, 2, i + 1, projection='3d') for i in range(4)]
    fig.subplots_adjust(left=.02, right=.98, bottom=.07, top=.89, wspace=.01, hspace=.04)
    title = SKILL_TITLES[str(original['skill'])]
    fig.text(.25, .955, str(original.get('render_title','ORIGINAL / ' + title)), ha='center', fontsize=23, weight='bold')
    fig.text(.75, .955, str(paired.get('render_title','COOPERATIVE / ' + title)), ha='center', fontsize=23, weight='bold')
    fig.text(.25, .92, str(original.get('render_caption','Recorded joints | 1 person | 0.40 m cube')), ha='center', fontsize=12)
    caption=paired.get('render_caption', 'Same joint angles | grip inset {:g} cm | 0.52 x 0.80 x 0.40 m box'.format(
        float(paired['grip_inset_m']) * 100))
    fig.text(.75, .92, str(caption), ha='center', fontsize=12)
    fig.text(.25, .045, str(original.get('render_footer','Separate source clip | recorded joints | no policy rollout')), ha='center', fontsize=12, color='#ffce78')
    fig.text(.75, .045, str(paired.get('render_footer','Candidate only | no IK / physics | clone feet may slide')),
             ha='center', fontsize=12, color='#ffce78')
    timers = [fig.text(x, .014, '', ha='center', fontsize=12) for x in (.25, .75)]
    faces = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4],
             [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    artists = []
    span = max(1.3, max(float(np.abs(d['body_position'][..., :2] -
        d['box_position'][:, None, None, :2]).max()) + .2 for d in (original, paired)))
    for index, ax in enumerate(axes):
        data = original if index % 2 == 0 else paired
        top = index >= 2
        ax.set_facecolor('#111b29')
        ax.set_proj_type('ortho')
        ax.view_init(elev=90 if top else 20, azim=-60 if top else -100)
        ax.set_box_aspect((2*span, 2*span, 2.1), zoom=1.25 if top else 1.45)
        ax.set_axis_off()
        ax.text2D(.03, .94, 'TOP VIEW' if top else 'PERSPECTIVE', transform=ax.transAxes,
                  color='#8195aa', fontsize=11)
        humans = []
        for a in range(data['body_position'].shape[1]):
            color = ['#5dbbff', '#ff9478'][a]
            lines = [ax.plot([], [], [], color=color, lw=4.5, solid_capstyle='round')[0]
                     for _ in data['parent_indices'][1:]]
            joints = ax.plot([], [], [], 'o', color=color, ms=4)[0]
            head = ax.plot([], [], [], 'o', color=color, ms=12)[0]
            humans.append((lines, joints, head))
        mesh = Poly3DCollection([], facecolors='#f2c862', edgecolors='#ffe5a2', alpha=.42, linewidths=1.2)
        ax.add_collection3d(mesh)
        grid = [ax.plot([], [], [], color='#334252', lw=.6)[0] for _ in range(30)]
        artists.append((data, humans, mesh, grid))
    writer = FFMpegWriter(fps=int(original['fps']), codec='libx264', bitrate=4500,
                         extra_args=['-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
    with writer.saving(fig, str(output / 'comparison.mp4'), dpi=100):
        for f in range(len(original['time'])):
            for ax, (data, humans, mesh, grid) in zip(axes, artists):
                centre = data['box_position'][f]
                ax.set_xlim(centre[0]-span, centre[0]+span)
                ax.set_ylim(centre[1]-span, centre[1]+span)
                ax.set_zlim(0, 2.1)
                base = np.floor(centre[:2] * 2) / 2
                for k in range(15):
                    g = (k-7)*.5
                    grid[k].set_data_3d([base[0]+g]*2, [base[1]-3.5, base[1]+3.5], [0, 0])
                    grid[k+15].set_data_3d([base[0]-3.5, base[0]+3.5], [base[1]+g]*2, [0, 0])
                for a, (lines, joints, head) in enumerate(humans):
                    p = data['body_position'][f, a]
                    for j, line in enumerate(lines, start=1):
                        line.set_data_3d(*p[[data['parent_indices'][j], j]].T)
                    joints.set_data_3d(*p.T)
                    head.set_data_3d(*p[2:3].T)
                v = box_vertices(centre, data['box_rotation_xyzw'][f], data['box_size'])
                mesh.set_verts([v[face] for face in faces])
            for timer in timers:
                timer.set_text('30 FPS   |   frame {:03d} / {}   |   source {:.2f} s'.format(
                    f + 1, len(original['time']), original['time'][f]))
            if f in (0, len(original['time']) // 2, len(original['time']) - 1):
                fig.savefig(output / ('frame_{:03d}.png'.format(f)), dpi=100, facecolor=fig.get_facecolor())
            writer.grab_frame(facecolor=fig.get_facecolor())
    plt.close(fig)
    for name, crop in [('original', 'crop=800:1000:0:0'), ('cooperative', 'crop=800:1000:800:0')]:
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(output/'comparison.mp4'),
                        '-vf', crop, '-c:v', 'libx264', '-crf', '19', '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart', str(output/(name+'.mp4'))], check=True)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(output/'comparison.mp4'),
                    '-vf', 'setpts=2*PTS', '-an', '-c:v', 'libx264', '-crf', '19',
                    '-movflags', '+faststart', str(output/'comparison_half_speed.mp4')], check=True)


def generate(skill, output, grip_inset):
    manifest = Path('tokenhsi/data/dataset_carry/dataset_carry.yaml')
    entries = yaml.safe_load(manifest.read_text())['motions'][skill]
    entry = next(e for e in entries if SKILL_TAGS[skill] in e['file'])
    source = manifest.parent / entry['file']
    raw = np.load(source, allow_pickle=True).item()
    assert raw['is_local'], 'Expected local joint quaternions'
    box = np.load(manifest.parent / entry['obj_file']).astype(float)
    q, root = raw['rotation']['arr'].astype(float), raw['root_translation']['arr'].astype(float)
    tree = raw['skeleton_tree']
    parents, offsets = tree['parent_indices']['arr'], tree['local_translation']['arr']
    fps, names = raw['fps'], tree['node_names']
    assert len(q) == len(box) and len(q) > 2
    original = pack(q[:, None], root[:, None], parents, offsets, box, [.4, .4, .4], fps, names)
    # The original box local +Y faces away from the carrier. Separate gripping
    # planes by 0.8m minus twice the inset. Rotate the clone 180 degrees about
    # world Z (no mirroring). Inset translates the whole body, with no IK.
    grip_offset = .4 - grip_inset
    shift = R.from_quat(box[:, 3:]).apply(np.tile([0., -grip_offset, 0.], (len(box), 1)))
    half_turn = R.from_rotvec([0, 0, np.pi])
    root_a = root + shift
    root_b = box[:, :3] + half_turn.apply(root_a - box[:, :3])
    clone_q = q.copy()
    clone_q[:, 0] = (half_turn * R.from_quat(q[:, 0])).as_quat()
    paired = pack(np.stack([q, clone_q], axis=1), np.stack([root_a, root_b], axis=1),
                  parents, offsets, box, [.52, .8, .4], fps, names)
    paired['grip_inset_m'] = np.asarray(grip_inset)
    # Preserve the source loader's RSI exclusions, without implying that any
    # synthetic frame passed contact or physics validation.
    skipped = entry.get('rsi_skipped_range', [])
    excluded_times = np.asarray(skipped) / len(q) * ((len(q)-1) / fps)
    allowed = np.ones(len(q), dtype=bool)
    if len(excluded_times):
        allowed &= ~((original['time'] >= excluded_times[0]) & (original['time'] <= excluded_times[1]))
    for data in (original, paired):
        data['skill'] = np.asarray(skill)
        data['source_rsi_allowed'] = allowed
        data['physics_valid_frame_mask'] = np.zeros(len(q), dtype=bool)
    # Verify the expected rigid transform and identical articulated joint angles.
    target = box[:, None, :3] + half_turn.apply(
        (paired['body_position'][:, 0] - box[:, None, :3]).reshape(-1, 3)).reshape(len(box), -1, 3)
    np.testing.assert_allclose(paired['body_position'][:, 1], target, atol=1e-7)
    np.testing.assert_allclose(paired['dof_position'][:, 0], paired['dof_position'][:, 1], atol=1e-7)
    assert diagnostics(paired)['finite']
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/'original_joints.npz', **original)
    np.savez_compressed(output/'cooperative_rsi_candidate.npz', **paired)
    metadata = dict(skill=skill, source=str(source), fps=fps, frames=len(q), duration_s=len(q)/fps,
                    source_manifest=str(manifest), source_rsi_skipped_frame_range=skipped,
                    source_rsi_skipped_time_range_s=excluded_times.tolist(),
                    quaternion_order='xyzw', position_unit='metre', angle_unit='radian',
                    shape_convention='frame, agent, joint/body, component; box shared by both agents',
                    method='Identical local joints; first carrier shifted -{:g}m along box Y; second rotated pi about moving box world Z.'.format(grip_offset),
                    grip_inset_m=grip_inset,
                    physics_validated=False, rsi_ready=False, ik_applied=False, simulator_pose_exact=False,
                    velocity_method='Linear: numpy gradient; angular: quaternion differences; last angular sample repeats previous.',
                    limitations=['No contact/force/collision validation; not connected to training reset.',
                                 'Preview uses full source quaternions. Physical 1-DOF elbows omit source elbow X/Z; exported 32-DOF reset can shift hand positions.',
                                 'Opposite-facing time-synchronous clone does not preserve planted feet.',
                                 'Separate source clips are not a stitched continuous trajectory.',
                                 'Source box heights are preserved, including any floor clearance.',
                                 'Hand distance uses joint origins, not actual hand collision geometry.'],
                    original=diagnostics(original), cooperative=diagnostics(paired))
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata, indent=2), flush=True)
    render(original, paired, output)
    print('Saved videos and joint candidates to', output, flush=True)
    return metadata


def bundle_candidates(output):
    instructions = '''Cooperative carry joint candidates (not physics-validated RSI)

Each skill directory contains original_joints.npz, cooperative_rsi_candidate.npz
and metadata.json. Clips are separate: do not join their positions or velocities
at the boundaries. Video montages are only a sequence of previews.

Load with numpy.load(path, allow_pickle=False). Frame times are in seconds,
positions are metres, joint angles are radians, quaternions are xyzw.
Agent index: 0 = shifted original, 1 = opposite-facing clone.

To initialise the 32-DOF approximation of a displayed frame f use all of:
  root_position[f], root_rotation_xyzw[f], dof_position[f]
  box_position[f], box_rotation_xyzw[f], box_size
Root linear/angular velocity is body_linear_velocity[f, :, 0] /
body_angular_velocity[f, :, 0]. Joint velocities are dof_velocity[f].
The articulated DOF layout is phys_humanoid_v3 (32 DOFs).
body_position/body_rotation_xyzw are the full source FK shown in the videos.
These are not exactly the simulator FK: physical elbows have only one DOF,
so source elbow X/Z rotations are omitted and hand positions can differ.
Moving inward changes whole-body position, never the internal joint angles.

source_rsi_allowed preserves the original YAML exclusion using MotionLib's
time conversion. It is not a physics approval. physics_valid_frame_mask is
all false, rsi_ready and physics_validated are false. No IK was applied.
Clone foot sliding, body/box collisions, original box floor clearance and
contact stability require checks before training. A paired reset loader is
not implemented by this preview script.
'''
    (output/'README.txt').write_text(instructions)
    with zipfile.ZipFile(output/'joint_candidates_all_skills.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in ('README.txt', 'all_skills.json'):
            archive.write(output/name, name)
        for skill in SKILL_TAGS:
            for name in ('original_joints.npz', 'cooperative_rsi_candidate.npz', 'metadata.json'):
                relative = Path(skill)/name
                archive.write(output/relative, str(relative))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('joint_carry/archive/carry_joint_preview'))
    parser.add_argument('--grip-inset', type=float, default=0.,
                        help='Move each complete skeleton inward from the box ends, in metres; local joints unchanged.')
    parser.add_argument('--skill', choices=['all'] + list(SKILL_TAGS), default='carryWith')
    args = parser.parse_args()
    if not 0 <= args.grip_inset < .4:
        parser.error('--grip-inset must be >= 0 and < 0.4 metres')
    if args.skill != 'all':
        generate(args.skill, args.output, args.grip_inset)
        return
    clips = [generate(skill, args.output / skill, args.grip_inset) for skill in SKILL_TAGS]
    summary = dict(clips=clips, continuous_trajectory=False, physics_validated=False,
                   note='Video montage only; source clips and RSI candidates remain separate.')
    (args.output/'all_skills.json').write_text(json.dumps(summary, indent=2)+'\n')
    for video in ('comparison', 'cooperative', 'original', 'comparison_half_speed'):
        playlist = args.output / (video + '_playlist.txt')
        playlist.write_text(''.join("file '{}/{}.mp4'\n".format(skill, video) for skill in SKILL_TAGS))
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0',
                        '-i', str(playlist), '-c', 'copy', '-movflags', '+faststart',
                        str(args.output / (video + '_all_skills.mp4'))], check=True)
    bundle_candidates(args.output)
    print('Saved all three skills and video montages to', args.output, flush=True)


if __name__ == '__main__':
    main()
