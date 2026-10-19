from isaacgym import gymapi
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R
import torch

from carry_joint_preview import box_vertices
from retarget_teamhoi_reference import foot_vertices
from validate_carry_physics import Scene


def physics_frames(data):
    scene = Scene(data)
    bodies, boxes, frames, elapsed = [], [], [], []
    try:
        for frame in range(len(data['time'])):
            scene.reset(frame)
            for step in range(7):
                if step:
                    scene.step()
                bodies.append(scene.body[:30].numpy().reshape(2, 15, 13).copy())
                boxes.append(scene.root[-1].numpy().copy())
                frames.append(frame)
                elapsed.append(step/60)
    finally:
        scene.close()
    bodies, boxes = np.asarray(bodies), np.asarray(boxes)
    return dict(data, body_position=bodies[..., :3], body_rotation_xyzw=bodies[..., 3:7],
                box_position=boxes[:, :3], box_rotation_xyzw=boxes[:, 3:7],
                source_frame=np.asarray(frames), elapsed=np.asarray(elapsed))


def render(before, after, output, physics=False):
    colors = ['#66baff', '#72db95']
    faces = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4],
             [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    meshes = [foot_vertices(name) for name in ('right_foot', 'left_foot')]
    triangles = [ConvexHull(mesh).simplices for mesh in meshes]
    fig = plt.figure(figsize=(16, 10), facecolor='#101b2b')
    fig.subplots_adjust(left=0, right=1, top=.88, bottom=.10, wspace=0, hspace=0)
    heading = 'ALL 100 PICKUP FRAMES / BEFORE vs AFTER'
    fig.text(.5, .965, heading, ha='center', color='white', fontsize=20, weight='bold')
    fig.text(.25, .925, 'Existing paired reference', ha='center', color='#ffad85', fontsize=16)
    fig.text(.75, .925, 'Bounded leg IK + hand refit', ha='center', color='#72db95', fontsize=16)
    footer = ('PhysX reset audit: each source frame independently reset, then simulated for 0.1 s. Half-speed playback. No policy.'
              if physics else 'Kinematic reference playback with physical-body FK and foot meshes. No physics / no policy.')
    fig.text(.5, .055, footer, ha='center', color='#bdcddd', fontsize=11)
    caption = fig.text(.5, .02, '', ha='center', color='white', fontsize=13)
    panels = []
    for index in range(4):
        data = before if index % 2 == 0 else after
        ax = fig.add_subplot(2, 2, index+1, projection='3d')
        ax.set_facecolor('#101b2b'); ax.set_proj_type('ortho')
        ax.view_init(elev=85 if index >= 2 else 18, azim=-70 if index >= 2 else -100)
        ax.set_box_aspect((2.6, 2.6, 2.0), zoom=1.2); ax.set_axis_off()
        edges = np.asarray([(p, j) for j, p in enumerate(data['parent_indices']) if p >= 0])
        people = []
        for color in colors:
            bones = Line3DCollection([], colors=color, linewidths=5)
            feet = Poly3DCollection([], facecolors=color, edgecolors='#203344', linewidths=.25)
            head, = ax.plot([], [], [], 'o', color=color, markersize=12)
            ax.add_collection3d(bones); ax.add_collection3d(feet)
            people.append((bones, feet, head))
        box = Poly3DCollection([], facecolors='#efc452', edgecolors='#ffdf85', alpha=.4)
        grid = Line3DCollection([], colors='#34475b', linewidths=.6)
        ax.add_collection3d(box); ax.add_collection3d(grid)
        panels.append((data, ax, edges, people, box, grid))
    writer = FFMpegWriter(fps=30, codec='libx264', bitrate=4500,
                         extra_args=['-pix_fmt', 'yuv420p', '-threads', '2', '-movflags', '+faststart'])
    with writer.saving(fig, str(output), dpi=90):
        for f in range(len(after['body_position'])):
            for data, ax, edges, people, box, grid in panels:
                center = data['box_position'][f].copy(); center[2] = 0
                ax.set(xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), zlim=(-.05, 1.95))
                segments = []
                for axis in (0, 1):
                    for tick in np.arange(-2, 2.1, .25):
                        line = np.zeros((2, 3)); line[:, axis] = tick-center[axis] % .25
                        line[:, 1-axis] = [-2, 2]; segments.append(line)
                grid.set_segments(segments)
                for agent, (bones, feet, head) in enumerate(people):
                    positions = data['body_position'][f, agent]-center
                    bones.set_segments(positions[edges]); head.set_data_3d(*positions[2, :, None])
                    polygons = []
                    for j, mesh, tris in zip((11, 14), meshes, triangles):
                        vertices = R.from_quat(data['body_rotation_xyzw'][f, agent, j]).apply(mesh)+positions[j]
                        polygons.extend(vertices[tris])
                    feet.set_verts(polygons)
                vertices = box_vertices(data['box_position'][f]-center, data['box_rotation_xyzw'][f], data['box_size'])
                box.set_verts([vertices[face] for face in faces])
            frame = int(after['source_frame'][f]) if physics else f
            text = f'Source frame {frame:03d} / 099 | {frame/30:.2f} s | box 52 x 80 x 40 cm'
            text += ' | PREVIOUSLY EXCLUDED' if 30 <= frame <= 74 else ''
            if physics:
                text += f' | simulation +{after["elapsed"][f]:.3f} s'
            caption.set_text(text)
            if f in (30, 50, 70) and not physics:
                fig.savefig(output.with_name(f'frame_{f:03d}.png'), dpi=110)
            writer.grab_frame(facecolor=fig.get_facecolor())
            if f % 100 == 0:
                print(output.name, f, '/', len(after['body_position']), flush=True)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, default=Path('output/joint_pickup_full_20261012'))
    parser.add_argument('--physics', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    before = dict(np.load('joint_carry/carry_joint_corrected/pickUp/cooperative_rsi_candidate.npz'))
    after = dict(np.load(args.directory/'pickUp/cooperative_rsi_candidate.npz'))
    if args.physics:
        before, after = physics_frames(before), physics_frames(after)
        np.savez_compressed(args.directory/'physics_video_states.npz',
                            before=before['body_position'], after=after['body_position'],
                            source_frame=after['source_frame'], elapsed=after['elapsed'])
    render(before, after, args.directory/('physics_all_frames.mp4' if args.physics else 'comparison.mp4'), args.physics)
