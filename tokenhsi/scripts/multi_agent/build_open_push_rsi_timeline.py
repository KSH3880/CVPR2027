"""Export complete AMP clips with RSI candidate markers to an offline frame viewer."""
from isaacgym import gymapi  # Import before torch in the runtime environment.
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tokenhsi'))
import torch
import yaml
from lpanlib.poselib.skeleton.skeleton3d import SkeletonMotion
from utils.open_push_rsi import lower_push_arms, bank_contract


def build(config, output, motion_file='tokenhsi/data/dataset_open_push.yaml'):
    env = yaml.safe_load(config.read_text())['env']
    manifest_path = (ROOT / motion_file).resolve()
    manifest = yaml.safe_load(manifest_path.read_text())
    bank = json.loads((ROOT / env['openPushRSI']['bank_file']).read_text())
    if bank['contract'] != bank_contract(dict(env, motion_file=str(manifest_path.relative_to(ROOT)))):
        raise ValueError('RSI bank does not match config/reference files')
    clips = []
    for skill in ('doorOpen', 'push'):
        for mid, row in enumerate(manifest['motions'][skill]):
            motion = SkeletonMotion.from_file(str(manifest_path.parent / row['file']))
            body = motion.global_translation.clone()
            names = list(motion.skeleton_tree.node_names)
            parents = motion.skeleton_tree.parent_indices.tolist()
            modified = body.clone()
            if skill == 'push':
                n = len(names)
                # DOFs are unused in this position-only export; the same reset arm
                # transform is applied to the complete clip, including non-candidates.
                dummy = body.new_zeros(len(body), n * 3)
                modified, *_ = lower_push_arms(
                    body, motion.global_rotation, motion.global_velocity,
                    motion.global_angular_velocity, dummy, dummy,
                    parents, [names.index('right_hand'), names.index('left_hand')],
                    list(range(n)), list(range(0, (n + 1) * 3, 3)),
                    env['relationReward']['push']['hand_target_height'])
            candidates = sorted((r for r in bank['candidates'][skill] if r['motion_id'] == mid), key=lambda r:r['time'])
            tail = int(len(candidates) * (1 - env['openPushRSI']['late_fraction']))
            for i, r in enumerate(candidates):
                r = dict(r)
                r['frame'] = round(r['time'] * motion.fps)
                r['late'] = i >= tail
                if skill == 'push':
                    r['initial_forward_m'] = r['box_progress'] * env['relationReward']['push']['target_distance']
                candidates[i] = r
            if not torch.isfinite(modified).all():
                raise ValueError('Nonfinite reference positions')
            clips.append(dict(skill=skill, motion_id=mid, name=Path(row['file']).parts[-3],
                fps=float(motion.fps), frames=len(body), nodes=[dict(name=n,parent=p) for n,p in zip(names,parents)],
                positions=body.numpy().round(5).tolist(), reset_positions=modified.numpy().round(5).tolist(),
                candidates=candidates))
    data=dict(config=str(config.relative_to(ROOT)), push_distance=env['relationReward']['push']['target_distance'], clips=clips)
    output.mkdir(parents=True,exist_ok=True)
    template=(Path(__file__).parent/'viewer_assets/open_push_rsi_timeline.html').read_text()
    html=template.replace('__MOTION_DATA__',json.dumps(data,ensure_ascii=False))
    (output/'rsi_timeline.html').write_text(html)
    (output/'rsi_timeline_data.json').write_text(json.dumps(data,ensure_ascii=False))
    print(json.dumps(dict(html=str(output/'rsi_timeline.html'), clips=[dict(skill=c['skill'],frames=c['frames'],candidates=len(c['candidates'])) for c in clips]),ensure_ascii=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg-env',default='tokenhsi/data/cfg/multi_agent/open_push_late_rsi_hinge_friction_push2m.yaml')
    parser.add_argument('--output-path',default='output/open_push_push2m_rsi_timeline')
    parser.add_argument('--motion-file', default='tokenhsi/data/dataset_open_push.yaml')
    args=parser.parse_args()
    build((ROOT/args.cfg_env).resolve(),(ROOT/args.output_path).resolve(),args.motion_file)
