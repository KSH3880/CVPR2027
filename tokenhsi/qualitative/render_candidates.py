"""Rank bridge_before_search candidates and write good_XX folders with a contact sheet and preview video."""
import argparse
import glob
import json
import os
import shutil
import subprocess
from multiprocessing import Pool

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

EDGES = [('pelvis', 'torso'), ('torso', 'head'), ('torso', 'right_upper_arm'), ('right_upper_arm', 'right_lower_arm'),
         ('right_lower_arm', 'right_hand'), ('torso', 'left_upper_arm'), ('left_upper_arm', 'left_lower_arm'),
         ('left_lower_arm', 'left_hand'), ('pelvis', 'right_thigh'), ('right_thigh', 'right_shin'),
         ('right_shin', 'right_foot'), ('pelvis', 'left_thigh'), ('left_thigh', 'left_shin'), ('left_shin', 'left_foot')]
FACES = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]


def corners(center, size, q):
    x, y, z, w = q
    rot = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                    [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                    [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    signs = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
    return center+(signs*size/2)@rot.T


def draw_box(ax, center, size, q, color, alpha):
    c = corners(center, size, q)
    ax.add_collection3d(Poly3DCollection([c[list(f)] for f in FACES], facecolor=color, edgecolor='#555555',
                                         linewidths=.3, alpha=alpha))


def draw_frame(ax, d, t, meta, title):
    ax.cla()
    for center, size, q in zip(d['platform_centers'], d['platform_sizes'], d['platform_quaternions']):
        draw_box(ax, center, size, q, '#b8c2cc', .35)
    for slot, color in ((int(d['bridge_slot']), '#9bd17a'), (int(d['payload_slot']), '#4aa3df')):
        box = d['boxes'][t, slot]
        draw_box(ax, box[:3], d['object_sizes'][slot], box[3:7], color, .8)
    names = list(d['body_names'])
    roles = meta['role_agents']
    follower = [k for k in roles if k not in ('pushers', 'AT')]
    colors = {a: '#3fa34d' for a in roles['pushers']}
    colors[roles['AT']] = '#1f6fd1'
    for k in follower:
        colors[roles[k]] = '#a43fbf'
    for agent in range(d['positions'].shape[1]):
        p = d['positions'][t, agent]
        for a, b in EDGES:
            i, j = names.index(a), names.index(b)
            ax.plot(*zip(p[i], p[j]), color=colors.get(agent, 'k'), lw=1.6)
    lo = (d['platform_centers']-d['platform_sizes']/2).min(0)-.5
    hi = (d['platform_centers']+d['platform_sizes']/2).max(0)+.5
    people = d['positions'][0, :, 0]
    lo[:2], hi[:2] = np.minimum(lo[:2], people[:, :2].min(0)-1), np.maximum(hi[:2], people[:, :2].max(0)+1)
    lo[2], hi[2] = 0, 4.
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi-lo, zoom=1.1)
    side = np.sign(np.mean([d['positions'][0, a, 0, 1] for a in roles['pushers']])-d['platform_centers'][:, 1].mean())
    ax.view_init(elev=30, azim=65 if side > 0 else -65)
    ax.set_axis_off()
    ax.set_title(title, fontsize=9)


def render(job):
    folder, step = job
    d = dict(np.load(os.path.join(folder, 'trajectory.npz')))
    meta = json.load(open(os.path.join(folder, 'metadata.json')))
    first = meta['first']
    keys = [('start', 0), ('connected', first['arrived']), ('crossed', first['crossed']), ('AT', first['at']),
            (meta['follower_task'], first['climb']), ('end', d['positions'].shape[0]-1)]
    fig = plt.figure(figsize=(18, 6.5))
    for k, (name, t) in enumerate(keys):
        ax = fig.add_subplot(2, 3, k+1, projection='3d')
        draw_frame(ax, d, max(t, 0), meta, f'{name} {max(t, 0)/30:.1f}s')
    fig.subplots_adjust(0, 0, 1, .95, 0, .05)
    fig.savefig(os.path.join(folder, 'contact_sheet.jpg'), dpi=110)
    plt.close(fig)
    frames = os.path.join(folder, '_frames')
    os.makedirs(frames, exist_ok=True)
    fig = plt.figure(figsize=(9.6, 5.4))
    ax = fig.add_subplot(projection='3d')
    for index, t in enumerate(range(0, d['positions'].shape[0], step)):
        draw_frame(ax, d, t, meta, f'{t/30:.1f}s')
        fig.savefig(os.path.join(frames, f'{index:05d}.png'), dpi=100)
    plt.close(fig)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(30//step), '-i',
                    os.path.join(frames, '%05d.png'), '-pix_fmt', 'yuv420p', '-vf', 'scale=960:540',
                    os.path.join(folder, 'preview.mp4')], check=True)
    shutil.rmtree(frames)
    return folder


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pools', nargs='+', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--top', type=int, default=8)
    p.add_argument('--frame_step', type=int, default=2)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--max_at_distance', type=float, default=2.0)
    opts = p.parse_args()
    found = []
    for pool in opts.pools:
        for meta_path in glob.glob(os.path.join(pool, '*', 'metadata.json')):
            meta = json.load(open(meta_path))
            m = meta['metrics']
            if m['d_at'] > opts.max_at_distance:
                continue
            m['direct'] = (m['initial_distance']-m['d_at'])/max(m['walked'], 1e-6)
            found.append((m['score']+2*m['direct'], os.path.dirname(meta_path), meta))
    found.sort(key=lambda item: -item[0])
    os.makedirs(opts.out, exist_ok=True)
    jobs, table = [], []
    for rank, (score, source, meta) in enumerate(found[:opts.top], 1):
        name = f"good_{rank:02d}_batch{meta['batch']:04d}_env{meta['env']:03d}"
        target = os.path.join(opts.out, name)
        os.makedirs(target, exist_ok=True)
        for file in ('trajectory.npz', 'metadata.json'):
            shutil.copy(os.path.join(source, file), target)
        jobs.append((target, opts.frame_step))
        m = meta['metrics']
        table.append(dict(name=name, seed=meta['seed'], score=round(score, 2), wait_s=round(m['wait']/30, 1),
            walked_m=round(m['walked'], 2), direct=round(m['direct'], 2), at_distance_m=round(m['d_at'], 2),
            initial_distance_m=round(m['initial_distance'], 2),
            times_s={k: round(v, 1) for k, v in meta['event_times_seconds'].items()}))
    with open(os.path.join(opts.out, 'ranking.json'), 'w') as f:
        json.dump(dict(candidates=len(found), selected=table), f, indent=2)
    with Pool(min(opts.workers, max(len(jobs), 1))) as pool:
        for folder in pool.imap_unordered(render, jobs):
            print('rendered', folder, flush=True)


if __name__ == '__main__':
    main()
