"""CPU skeleton video: full source, cropped source, and existing AMP retarget."""
from pathlib import Path
import hashlib
import json
import math
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tokenhsi.utils.bones_bvh import read_bvh


def main():
    name = 'push_obstacle_180_001__A344_M'
    converted = ROOT / 'tokenhsi/data/dataset_bones_push180/motions' / name / 'phys_humanoid_v3'
    report = json.loads((converted / 'conversion_report.json').read_text())
    source = ROOT / report['source']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == report['source_sha256']
    nodes, pos, _, dt = read_bvh(source)
    ids = {n['name']: i for i, n in enumerate(nodes)}
    left = pos[0, ids['LeftArm']] - pos[0, ids['RightArm']]
    left[1] = 0
    left /= np.linalg.norm(left)
    up = np.array([0., 1., 0.])
    forward = np.cross(left, up)
    forward /= np.linalg.norm(forward)
    pos = pos @ np.stack((forward, left, up)).T
    # BVH has a stationary synthetic Root above the moving Hips.
    hips = ids['Hips']
    pos[:, :, :2] -= pos[0, hips, :2].copy()
    preview = json.loads((converted / 'preview_data.json').read_text())[0]
    amp = np.asarray(preview['positions'])[:, :, [0, 2, 1]]
    amp_nodes = preview['nodes']
    fps = 30
    duration = (len(pos) - 1) * dt
    crop = report['source_crop_end_s']
    out = ROOT / 'output/push_crop_comparison'
    out.mkdir(parents=True, exist_ok=True)
    video = out / (name + '_full_vs_crop_vs_amp.mp4')
    regular = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(regular, 20)
    small = ImageFont.truetype(regular, 16)
    width, height = 1440, 680
    nframes = math.ceil(duration * fps) + 1
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
           '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', str(fps), '-i', '-',
           '-an', '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p',
           '-movflags', '+faststart', str(video)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    titles = ['1  ORIGINAL FULL CLIP', '2  ORIGINAL: KEPT INTERVAL', '3  CURRENT AMP RETARGET']

    def skeleton(draw, data, ns, panel, stopped):
        # Root-following, common metric scale. Global displacement is shown separately.
        root_index = hips if panel < 2 else 0
        center = data[root_index, :2].copy()
        def project(p):
            x, y = p[:2] - center
            return (panel * 480 + 240 + 175 * (.88 * x - .48 * y),
                    545 - 175 * (p[2] + .16 * (.48 * x + .88 * y)))
        for k in np.arange(-1.5, 1.51, .5):
            for a, b in (([-1.5, k, 0], [1.5, k, 0]), ([k, -1.5, 0], [k, 1.5, 0])):
                a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
                a[:2] += center; b[:2] += center
                draw.line([project(a), project(b)], fill='#dce3ec', width=1)
        for i, node in enumerate(ns):
            parent = node['parent']
            if parent < 0 or (panel < 2 and i == hips):
                continue
            label = node['name'].lower()
            color = '#2776bd' if 'left' in label else '#d84e62' if 'right' in label else '#455369'
            if stopped:
                color = '#a3aab4'
            draw.line([project(data[parent]), project(data[i])], fill=color, width=4 if len(ns) < 20 else 3)
        draw.text((panel * 480 + 24, 550), f'Root XY displacement: {np.linalg.norm(data[root_index,:2]):.2f} m', font=small, fill='#374151')

    try:
        for frame in range(nframes):
            t = min(frame / fps, duration)
            source_idx = min(round(t / dt), len(pos) - 1)
            kept_idx = min(round(min(t, crop) / dt), len(pos) - 1)
            amp_idx = min(round(t * fps), len(amp) - 1)
            im = Image.new('RGB', (width, height), '#f6f8fc')
            draw = ImageDraw.Draw(im)
            for panel in range(3):
                x = panel * 480
                if panel:
                    draw.line([(x, 15), (x, 590)], fill='#c6cedb', width=2)
                draw.text((x + 24, 22), titles[panel], font=font, fill='#15243c')
                draw.text((x + 24, 55), name, font=small, fill='#374151')
                draw.text((x + 24, 82), f'Time {t:.2f}s / source {duration:.2f}s', font=small, fill='#374151')
                stopped = (panel == 1 and t > crop) or (panel == 2 and t > (len(amp)-1)/fps)
                status = ('REMOVED TAIL: original keeps playing' if panel == 0 and t > crop else
                          'CLIP ENDED - held final pose' if stopped else 'Playing')
                draw.text((x + 24, 112), status, font=small, fill='#b54720' if t > crop else '#168059')
                data = pos[source_idx] if panel == 0 else pos[kept_idx] if panel == 1 else amp[amp_idx]
                skeleton(draw, data, nodes if panel < 2 else amp_nodes, panel, stopped)
            draw.text((24, 597), f'Crop end: {crop:.2f}s | Blue: left limbs; red: right limbs | Root-following view, meters', font=small, fill='#15243c')
            draw.text((24, 624), 'Skeleton playback only: not a policy rollout; no measured box/contact forces.', font=small, fill='#374151')
            draw.text((24, 650), 'Motion Data by Bones Studio | https://bones.studio/ | Internal research preview', font=small, fill='#374151')
            if frame == round((crop + 1) * fps):
                im.save(out / 'comparison_after_crop.png')
            proc.stdin.write(im.tobytes())
    finally:
        proc.stdin.close()
        rc = proc.wait()
    if rc:
        raise RuntimeError(f'ffmpeg exit {rc}')
    metadata = dict(source=str(source), source_sha256=report['source_sha256'],
                    amp_preview=str(converted / 'preview_data.json'), video=str(video),
                    source_duration_s=duration, crop_end_s=crop, amp_frames=len(amp),
                    fps=fps, frames=nframes, view='root_following_same_metric_scale',
                    note='Original BVH sampled for visualization; frozen final poses after cropped clips end. No simulation.')
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
