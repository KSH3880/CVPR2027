import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.spatial.transform import Rotation as R

from joint_carry.artifacts import contract_digest
from tokenhsi.utils.grasp_geometry import ANCHOR_INSET


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True)
    parser.add_argument('--requests', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--source', choices=('single', 'stage2'), default='single')
    parser.add_argument('--inset', type=float, default=ANCHOR_INSET)
    args = parser.parse_args()
    signature = contract_digest(args.source, args.inset)
    requests = json.loads(Path(args.requests).read_text())
    xml = ET.parse('tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml')
    bodies = {body.get('name'): body for body in xml.findall('.//worldbody//body')}
    with np.load('joint_carry/source_rsi/pickUp/cooperative_rsi_candidate.npz', allow_pickle=False) as source:
        names = source['joint_names'].tolist()
    bounds = np.deg2rad([np.fromstring(joint.get('range'), sep=' ')
                        for name in names[1:] for joint in bodies[name].findall('joint')])
    phases = {name: [] for name in ('pickUp', 'carryWith', 'putDown')}
    cases, issues, total_bytes = [], [], 0
    for request in requests:
        size = request['size']
        key = '_'.join(str(round(value*20)) for value in size)
        case = dict(size=size, skills={})
        for skill in phases:
            path = Path(args.directory) / key / (skill + '.npz')
            metadata = path.with_suffix('.json')
            if not path.exists() or not metadata.exists():
                issues.append(str(path) + ': missing')
                continue
            saved = json.loads(metadata.read_text())
            if saved['contract_digest'] != signature:
                issues.append(str(path) + ': stale recipe')
            if hashlib.sha256(path.read_bytes()).hexdigest() != saved['sha256']:
                issues.append(str(path) + ': SHA mismatch')
            with np.load(path, allow_pickle=False) as data:
                n = len(data['source_frame'])
                if not n:
                    issues.append(str(path) + ': empty')
                    continue
                if n != saved['accepted'] or len(np.unique(data['source_frame'])) != n:
                    issues.append(str(path) + ': invalid frame count')
                if not all(np.isfinite(data[name]).all() for name in data.files):
                    issues.append(str(path) + ': nonfinite')
                if not np.allclose(data['box_size'], size):
                    issues.append(str(path) + ': size mismatch')
                if data['dof_position'].shape != (n, 2, 32) or data['root_state'].shape != (n, 2, 13):
                    issues.append(str(path) + ': invalid state shape')
                joints = data['dof_position']
                excess = float(np.maximum(np.maximum(bounds[:, 0]-joints, joints-bounds[:, 1]), 0).max())
                if excess > 1e-5:
                    issues.append(str(path) + ': joint limit violation')
                if not np.allclose(data['root_state'][..., :3], data['body_state'][..., 0, :3], atol=1e-6):
                    issues.append(str(path) + ': root/body mismatch')
                for state in (data['root_state'], data['box_state']):
                    if not np.allclose(np.linalg.norm(state[..., 3:7], axis=-1), 1., atol=1e-6):
                        issues.append(str(path) + ': invalid quaternion')
                box = data['box_state']
                hands = data['body_state'][:, :, [5, 8], :3].mean(2)
                inverse = R.from_quat(np.repeat(box[:, None, 3:7], 2, axis=1).reshape(-1, 4)).inv()
                local = inverse.apply((hands-box[:, None, :3]).reshape(-1, 3)).reshape(n, 2, 3)
                local_root = inverse.apply((data['root_state'][..., :3]-box[:, None, :3]).reshape(-1, 3)).reshape(n, 2, 3)
                side = np.sign(np.median(local_root[:, 0, 1])) * np.array([1., -1.])
                target = np.zeros((2, 3))
                target[:, 1] = side*(size[1]/2-args.inset)
                anchor_error = float(np.linalg.norm(local-target, axis=-1).max())
            if skill == 'carryWith' and saved['minimum_accepted_simultaneous_contact_fraction'] < .5:
                issues.append(str(path) + ': missing simultaneous contact')
            total_bytes += path.stat().st_size
            record = dict(accepted=n, total=saved['frames'], joint_limit_excess_rad=excess,
                          max_hand_target_error_m=saved['maximum_accepted_hand_target_error_m'],
                          max_anchor_error_m=anchor_error,
                          simultaneous_contact_min=saved['minimum_accepted_simultaneous_contact_fraction'])
            case['skills'][skill] = record
            phases[skill].append(record)
        cases.append(case)
    summary = {skill: dict(profiles=len(values), minimum_frames=min((v['accepted'] for v in values), default=0),
                          maximum_hand_target_error_m=max((v['max_hand_target_error_m'] for v in values), default=0.),
                          maximum_anchor_error_m=max((v['max_anchor_error_m'] for v in values), default=0.))
               for skill, values in phases.items()}
    result = dict(source=args.source, inset=args.inset, requested_sizes=len(requests),
                  contract_digest=signature, passed=not issues, issues=issues,
                  phases=summary, npz_bytes=total_bytes, cases=cases)
    Path(args.report).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'cases'}, indent=2))
    if issues:
        raise SystemExit('RSI audit failed; see report')


if __name__ == '__main__':
    main()
