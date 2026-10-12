import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import yaml

from carry_joint_preview import forward_kinematics
from retarget_teamhoi_reference import ROOT, SOURCE, OUTPUT, CLIPS, select_clips, foot_vertices, lowest_vertex

DATA = ROOT/'joint_carry/teamhoi_retarget_posture_pilot'
OUT = ROOT/'output_etc/teamhoi_retarget_posture_pilot_check'


def measure(path, offsets):
    raw = np.load(path, allow_pickle=True).item()
    q, roots, fps = raw['rotation']['arr'], raw['root_translation']['arr'], raw['fps']
    tree = raw['skeleton_tree']
    positions, world_q = forward_kinematics(q, roots, tree['parent_indices']['arr'], offsets)
    knees = np.rad2deg(R.from_quat(q[:, [10, 13]].reshape(-1, 4)).as_rotvec()[:, 1]).reshape(-1, 2)
    soles = np.stack([positions[:, j, 2]+lowest_vertex(world_q[:, j], foot_vertices(tree['node_names'][j]))
                      for j in (11, 14)], axis=1)
    result = dict(path=str(path.relative_to(ROOT)), frames=len(q), fps=fps,
                  knee_p10_median_p90_deg=np.quantile(knees, [.1, .5, .9]).tolist(),
                  knee_frame_change_p95_deg=float(np.quantile(np.abs(np.diff(knees, axis=0)), .95)),
                  pelvis_height_median_m=float(np.median(roots[:, 2])),
                  pelvis_height_span_m=float(np.ptp(roots[:, 2])),
                  pelvis_vertical_speed_p95_mps=float(np.quantile(np.abs(np.diff(roots[:, 2])*fps), .95)),
                  support_height_min_max_m=[float(soles.min()), float(soles.min(1).max())],
                  foot_height_difference_p50_p90_m=np.quantile(np.ptp(soles, axis=1), [.5, .9]).tolist())
    return result, knees, roots[:, 2], fps


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()
    if args.all:
        DATA, CLIPS = select_clips(True, True)
        OUTPUT, _ = select_clips(True)
        OUT = ROOT/'output/teamhoi_retarget_posture_all_check'
    OUT.mkdir(parents=True, exist_ok=True)
    offsets = np.load(DATA/(next(iter(CLIPS))+'.npy'), allow_pickle=True).item()['skeleton_tree']['local_translation']['arr']
    baseline = yaml.safe_load((ROOT/'tokenhsi/data/dataset_loco_sit_carry_climb.yaml').read_text())['motions']['loco'][:5]
    report = {'existing_walks':[measure(ROOT/'tokenhsi/data'/c['file'], offsets)[0] for c in baseline],
              'clips':{}, 'note':'Kinematic comparison; distributions are descriptive, not a naturalness certificate.'}
    for kind, source in CLIPS.items():
        paths = [SOURCE/source, OUTPUT/(kind+'.npy'), DATA/(kind+'.npy')]
        labels = ['Raw', 'Previous correction', 'Posture-first']
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        results = []
        for path, label in zip(paths, labels):
            result, knees, height, fps = measure(path, offsets)
            results.append(result)
            time = np.arange(len(knees))/fps
            for side in range(2):
                axes[side].plot(time, knees[:, side], label=label, linewidth=1.3)
            axes[2].plot(time, height, label=label, linewidth=1.3)
        raw = np.load(paths[0], allow_pickle=True).item()
        new = np.load(paths[2], allow_pickle=True).item()
        a = R.from_quat(raw['rotation']['arr'][:, [10, 13]].reshape(-1, 4)).as_rotvec()[:, 1]
        b = R.from_quat(new['rotation']['arr'][::new['fps']//raw['fps'], [10, 13]].reshape(-1, 4)).as_rotvec()[:, 1]
        report['clips'][kind] = {'variants':results, 'knee_change_p95_deg':float(np.rad2deg(np.quantile(np.abs(a-b), .95)))}
        axes[0].set_ylabel('Right knee (deg)');axes[1].set_ylabel('Left knee (deg)')
        axes[2].set_ylabel('Pelvis height (m)');axes[2].set_xlabel('Time (s)')
        axes[0].legend();axes[0].set_title(kind+' / same motion timing')
        for ax in axes:
            ax.grid(alpha=.25)
        fig.tight_layout();fig.savefig(OUT/(kind+'_gait.png'), dpi=130);plt.close(fig)
    (OUT/'gait_comparison.json').write_text(json.dumps(report, indent=2)+'\n')
