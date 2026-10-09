import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np
import yaml
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R

from carry_joint_preview import forward_kinematics
from retarget_teamhoi_reference import ROOT, SOURCE, OUTPUT, select_clips, foot_vertices


def render(kind, source, data_dir=OUTPUT, out=None, posture_first=False):
    original = np.load(SOURCE/source, allow_pickle=True).item()
    corrected = np.load(data_dir/(kind+'.npy'), allow_pickle=True).item()
    tree = corrected['skeleton_tree']
    parents = tree['parent_indices']['arr']
    offsets = tree['local_translation']['arr']
    raws = [original, corrected]
    titles = ['Raw motion on our body', 'Retargeted + ground correction']
    colors = ['#ff8e85', '#6ad6bc']
    if posture_first:
        baseline = yaml.safe_load((ROOT/'tokenhsi/data/dataset_loco_sit_carry_climb.yaml').read_text())['motions']['loco'][0]['file']
        previous_dir = OUTPUT if kind in ('backward', 'sideways') else select_clips(True)[0]
        raws = [original, np.load(previous_dir/(kind+'.npy'), allow_pickle=True).item(), corrected,
                np.load(ROOT/'tokenhsi/data'/baseline, allow_pickle=True).item()]
        titles = ['Raw on our body', 'Previous correction', 'Posture-first correction', 'Existing walk (loop)']
        colors = ['#ff8e85', '#ffcb77', '#6ad6bc', '#8cbaff']
    clips = []
    knees = []
    for raw in raws:
        stride = raw['fps']//original['fps']
        clips.append(tuple(values[::stride] for values in forward_kinematics(
            raw['rotation']['arr'], raw['root_translation']['arr'], parents, offsets)))
        knees.append(np.rad2deg(R.from_quat(raw['rotation']['arr'][::stride, [10, 13]].reshape(-1, 4)).as_rotvec()[:, 1]).reshape(-1, 2))
    edges = np.array([(p, j) for j, p in enumerate(parents) if p >= 0])
    meshes = [foot_vertices(tree['node_names'][j]) for j in (11, 14)]
    triangles = [ConvexHull(mesh).simplices for mesh in meshes]
    fig = plt.figure(figsize=(6*len(clips), 6), facecolor='#101b2b')
    fig.subplots_adjust(left=0, right=1, top=.85, bottom=.10, wspace=0)
    title = kind.replace('ACCAD_', '').replace('Walking_c3d_', ' ').replace('_stageii', '').replace('_poses', '')
    fig.text(.5, .96, title+' / phys_humanoid_v3', ha='center', color='white', fontsize=13)
    fig.text(.5, .035, 'Kinematic comparison on our body + foot meshes | Not a trained policy / not physics playback',
             ha='center', color='#becbdd', fontsize=11)
    panels = []
    for i, ((positions, rotations), title, color) in enumerate(zip(clips, titles, colors)):
        ax = fig.add_subplot(1, len(clips), i+1, projection='3d')
        ax.set_facecolor('#101b2b');ax.set_proj_type('ortho')
        ax.view_init(elev=15, azim=-65)
        ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-.18, 1.9))
        ax.set_box_aspect((2, 2, 2.08));ax.set_axis_off()
        ax.set_title(title, color=color, fontsize=15)
        bones = Line3DCollection([], colors=color, linewidths=4)
        grid = Line3DCollection([], colors='#506078', linewidths=.7)
        feet = Poly3DCollection([], facecolors=color, edgecolors='#153e49', linewidths=.25, alpha=.9)
        for artist in (grid, bones, feet):
            ax.add_collection3d(artist)
        head, = ax.plot([], [], [], 'o', color=color, markersize=10)
        caption = ax.text2D(.1, .05, '', transform=ax.transAxes, color='white', fontsize=11)
        panels.append((positions, rotations, bones, grid, feet, head, caption))
    out = out or ROOT/'output_etc/teamhoi_retarget_pilot_check'
    out.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=30, codec='libx264', bitrate=2600,
                         extra_args=['-pix_fmt', 'yuv420p', '-threads', '2', '-movflags', '+faststart'])
    with writer.saving(fig, str(out/(kind+'.mp4')), dpi=90):
        for f in range(len(clips[0][0])):
            for panel_index, (positions, rotations, bones, grid, feet, head, caption) in enumerate(panels):
                frame = f % len(positions)
                center = positions[frame, 0].copy();center[2] = 0
                pos = positions[frame]-center
                bones.set_segments(pos[edges]);head.set_data_3d(*pos[2, :, None])
                lines = []
                for axis in (0, 1):
                    for tick in np.arange(-1.5, 1.6, .25):
                        line = np.zeros((2, 3));line[:, axis] = tick-center[axis] % .25
                        line[:, 1-axis] = [-1.5, 1.5];lines.append(line)
                grid.set_segments(lines)
                polygons = []
                for j, mesh, tris in zip((11, 14), meshes, triangles):
                    vertices = R.from_quat(rotations[frame, j]).apply(mesh)+pos[j]
                    polygons.extend(vertices[tris])
                feet.set_verts(polygons)
                angles = knees[panel_index][frame]
                caption.set_text(f'{f/30:.2f} s | knees {angles[0]:.0f} / {angles[1]:.0f} deg')
            if f == 60:
                fig.savefig(out/(kind+'.png'), dpi=110)
            writer.grab_frame()
            if f % 150 == 0:
                print(kind, f, '/', len(clips[0][0]), flush=True)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--posture-first', action='store_true')
    parser.add_argument('--montage', action='store_true', help='Render backward/sideways overviews instead of individual comparisons')
    args = parser.parse_args()
    data_dir, clips = select_clips(args.all, args.posture_first)
    out = ROOT/('output_etc/teamhoi_retarget_all_check' if args.all else 'output_etc/teamhoi_retarget_pilot_check')
    if args.posture_first:
        out = ROOT/('output/teamhoi_retarget_posture_all_check' if args.all else 'output_etc/teamhoi_retarget_posture_pilot_check')
    if args.montage:
        import preview_teamhoi_reference as overview
        overview.OUTPUT = out
        out.mkdir(parents=True, exist_ok=True)
        for group, prefix in (('backward', 'loco_reverse/'), ('sideways', 'sidewalk/')):
            records = [{'file':str(data_dir/(kind+'.npy'))} for kind, source in clips.items() if source.startswith(prefix)]
            overview.render(records, group+'_all.mp4', min(3, len(records)), retargeted=True)
    else:
        for kind, source in clips.items():
            render(kind, source, data_dir, out, args.posture_first)
