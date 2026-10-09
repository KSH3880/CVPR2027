import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

from carry_joint_preview import forward_kinematics


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'joint_carry/teamhoi_reference'
OUTPUT = ROOT / 'output_etc/teamhoi_reference_preview'


def load_clip(record):
    raw = np.load(DATA / record['file'], allow_pickle=True).item()
    tree = raw['skeleton_tree']
    parents = tree['parent_indices']['arr']
    positions, rotations = forward_kinematics(raw['rotation']['arr'],
        raw['root_translation']['arr'], parents, tree['local_translation']['arr'])
    q = rotations[:, 0]
    x, y, z, w = q.T
    yaw = np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    velocity = np.gradient(positions[:, 0], axis=0) * raw['fps']
    name = Path(record['file']).stem
    name = name.replace('ACCAD_', '').replace('Walking_c3d_', ' / ')
    name = name.replace('_stageii', '').replace('_poses', '').replace('_', ' ')
    return dict(name=name, positions=positions, parents=parents, yaw=yaw,
                velocity=velocity, fps=raw['fps'])


def render(records, filename, columns, retargeted=False):
    clips = [load_clip(record) for record in records]
    rows = (len(clips) + columns - 1) // columns
    fig = plt.figure(figsize=(6.4 * columns, 4.5 * rows + 1.3), facecolor='#101b2b')
    fig.subplots_adjust(left=.01, right=.99, bottom=.12 if rows == 1 else .055,
                        top=.82 if rows == 1 else .90, wspace=.01, hspace=.12)
    title = 'BACKWARD WALKING' if 'backward' in filename else 'SIDEWAYS WALKING'
    fig.text(.5, .975, ('Retargeted reference / ' if retargeted else 'TeamHOI reference / ') + title, ha='center', va='top',
             color='white', fontsize=24, weight='bold')
    fig.text(.5, .02, 'Yellow: facing direction    Pink: movement direction    ' +
             ('Retargeted motion, real-time playback' if retargeted else 'Original motion, real-time playback') +
             ' | Camera follows pelvis | No physics / no policy',
             ha='center', color='#bdd1e5', fontsize=12)
    panels = []
    for i, clip in enumerate(clips):
        ax = fig.add_subplot(rows, columns, i + 1, projection='3d')
        ax.set_facecolor('#101b2b')
        ax.set_proj_type('ortho')
        ax.view_init(elev=22, azim=-65)
        ax.set(xlim=(-1.25, 1.25), ylim=(-1.25, 1.25), zlim=(-.05, 2.05))
        ax.set_box_aspect((2.5, 2.5, 2.1), zoom=1.25)
        ax.set_axis_off()
        ax.set_title(clip['name'], fontsize=10, color='#e4edf8', pad=0)
        parents = clip['parents']
        edges = np.array([(int(parent), j) for j, parent in enumerate(parents) if parent >= 0])
        bones = Line3DCollection([], colors='#64c5ff', linewidths=5)
        grid = Line3DCollection([], colors='#34475c', linewidths=.7)
        arrows = Line3DCollection([], colors=['#ffcf55']*3 + ['#ff74b3']*3, linewidths=2.5)
        for artist in (grid, bones, arrows):
            ax.add_collection3d(artist)
        head, = ax.plot([], [], [], 'o', color='#9cddff', markersize=12)
        trail, = ax.plot([], [], [], color='#537a91', linewidth=1.2)
        caption = ax.text2D(.05, .02, '', transform=ax.transAxes, color='#bdcfe3', fontsize=10)
        panels.append((clip, edges, bones, grid, arrows, head, trail, caption))
    fps = 30
    frames = max(round(len(c['positions']) / c['fps'] * fps) for c in clips)
    writer = FFMpegWriter(fps=fps, codec='libx264', bitrate=4500,
                         extra_args=['-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-threads', '2'])
    with writer.saving(fig, str(OUTPUT / filename), dpi=100):
        for frame in range(frames):
            for clip, edges, bones, grid, arrows, head, trail, caption in panels:
                index = min(round(frame / fps * clip['fps']), len(clip['positions']) - 1)
                pos = clip['positions'][index].copy()
                center = pos[0].copy(); center[2] = 0
                pos -= center
                bones.set_segments(pos[edges])
                head.set_data_3d(*pos[2, :, None])
                segments = []
                for axis in (0, 1):
                    for tick in np.arange(-2., 2.01, .5):
                        endpoints = np.zeros((2, 3))
                        endpoints[:, axis] = tick - center[axis] % .5
                        endpoints[:, 1-axis] = [-2., 2.]
                        segments.append(endpoints)
                grid.set_segments(segments)
                facing = np.array([np.cos(clip['yaw'][index]), np.sin(clip['yaw'][index]), 0.])
                velocity = clip['velocity'][index].copy(); velocity[2] = 0
                vectors = [facing * .65, velocity * .55]
                arrow_lines = []
                for direction in vectors:
                    start = np.array([0., 0., .06])
                    end = start + direction
                    length = np.linalg.norm(direction)
                    unit = direction / max(length, 1e-9)
                    side = np.array([-unit[1], unit[0], 0.])
                    for a, b in [(start, end), (end, end-.12*unit+.055*side),
                                 (end, end-.12*unit-.055*side)]:
                        arrow_lines.append(np.stack((a, b)))
                arrows.set_segments(arrow_lines)
                path = clip['positions'][max(0, index-60):index+1, 0].copy() - center
                path[:, 2] = .015
                trail.set_data_3d(*path.T)
                local_forward = np.dot(velocity, facing)
                local_side = np.dot(velocity, [-facing[1], facing[0], 0.])
                caption.set_text(f'{index/clip["fps"]:.1f}s  '
                    f'forward {local_forward:+.2f} m/s | side {local_side:+.2f} m/s')
            if frame == min(60, frames-1):
                fig.savefig(OUTPUT / filename.replace('.mp4', '.png'), dpi=100, facecolor=fig.get_facecolor())
            writer.grab_frame()
            if frame % 150 == 0:
                print(filename, frame, '/', frames, flush=True)
    plt.close(fig)


if __name__ == '__main__':
    OUTPUT.mkdir(parents=True, exist_ok=True)
    clips = json.loads((DATA / 'manifest.json').read_text())['clips']
    render([c for c in clips if c['file'].startswith('loco_reverse/')], 'backward.mp4', 3)
    render([c for c in clips if c['file'].startswith('sidewalk/')], 'sideways.mp4', 3)
