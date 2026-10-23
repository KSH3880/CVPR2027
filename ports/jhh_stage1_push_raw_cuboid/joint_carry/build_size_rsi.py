import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tokenhsi'))
import isaacgym
import numpy as np

from joint_carry.size_candidates import make_candidates
from joint_carry.artifacts import contract_digest
from joint_carry.size_screen import screen
from tokenhsi.utils.grasp_geometry import ANCHOR_INSET


def profile_name(size, support):
    values = list(size)+(list(support) if support is not None else [])
    return '_'.join(str(round(float(value)*20)) for value in values)


def build(size, support, directory, source, inset):
    build_digest = contract_digest(source, inset)
    directory = Path(directory)/profile_name(size, support)
    directory.mkdir(parents=True, exist_ok=True)
    summary = {}
    skills = ('pickUp', 'carryWith', 'putDown') if support is None else ('putDown',)
    pending = []
    candidates = {}
    for skill in skills:
        target = directory/(skill+'.npz')
        metadata = directory/(skill+'.json')
        if target.exists() and metadata.exists():
            saved = json.loads(metadata.read_text())
            if (hashlib.sha256(target.read_bytes()).hexdigest() == saved.get('sha256')
                    and saved.get('contract_digest') == build_digest
                    and saved.get('accepted', 0) > 0
                    and np.allclose(saved.get('size', []), size, atol=1e-5)):
                summary[skill] = saved
                continue
        candidates[skill] = make_candidates(skill, size, source, inset)
        pending.append(skill)
    if pending:
        first = candidates[pending[0]][0]
        batch = dict(first)
        for key, value in first.items():
            if isinstance(value, np.ndarray) and value.ndim and len(value) == len(first['time']):
                batch[key] = np.concatenate([candidates[skill][0][key] for skill in pending])
        batch['skill'] = np.asarray('mixed')
        batch['carry_mask'] = np.concatenate([np.full(len(candidates[skill][0]['time']), skill == 'carryWith') for skill in pending])
        valid, batch_report = screen(batch, support)
        geometry_mask = np.asarray(batch_report.pop('geometry_mask'))
        contact_fraction = np.asarray(batch_report.pop('contact_fraction_by_frame'))
        simultaneous = np.asarray(batch_report.pop('simultaneous_contact_fraction_by_frame'))
        offset = 0
        for skill in pending:
            candidate, correction = candidates[skill]
            count = len(candidate['time'])
            selected = np.flatnonzero(valid[offset:offset+count])
            report = dict(batch_report, frames=count, accepted=len(selected),
                          geometry_accepted=int(geometry_mask[offset:offset+count].sum()),
                          mean_contact_fraction=contact_fraction[offset:offset+count].mean(0).tolist(),
                          minimum_accepted_simultaneous_contact_fraction=float(simultaneous[offset+selected].min()) if len(selected) else 0.)
            if support is None:
                report.pop("support_geometry_accepted", None)
            offset += count
            root = np.concatenate((candidate['root_position'], candidate['root_rotation_xyzw'],
                                   candidate['body_linear_velocity'][:, :, 0], candidate['body_angular_velocity'][:, :, 0]), -1)
            body = np.concatenate([candidate[key] for key in ('body_position', 'body_rotation_xyzw', 'body_linear_velocity', 'body_angular_velocity')], -1)
            box = np.concatenate([candidate[key] for key in ('box_position', 'box_rotation_xyzw', 'box_linear_velocity', 'box_angular_velocity')], -1)
            target = directory/(skill+'.npz')
            metadata = directory/(skill+'.json')
            temporary = target.with_name(str(os.getpid())+'_'+target.name)
            np.savez_compressed(temporary, root_state=root[selected], dof_position=candidate['dof_position'][selected],
                                dof_velocity=candidate['dof_velocity'][selected], body_state=body[selected], box_state=box[selected],
                                terminal_box_xy=candidate['terminal_box_xy'][selected], source_frame=selected,
                                box_size=np.asarray(size), support_size=np.asarray(support or []))
            os.replace(temporary, target)
            selected_set = set(selected.tolist())
            accepted_errors = [row['target_error_m'] for row in correction['arm_fit'] if row['frame'] in selected_set]
            saved = dict(report, size=list(size), support=support, contract_digest=build_digest,
                         source_mode=source, anchor_inset=inset,
                         source_sha256=str(candidate['source_sha256']),
                         maximum_hand_target_error_m=correction['maximum_hand_target_error_m'],
                         maximum_accepted_hand_target_error_m=max(accepted_errors, default=0.),
                         sha256=hashlib.sha256(target.read_bytes()).hexdigest())
            temporary_metadata = metadata.with_name(str(os.getpid())+'_'+metadata.name)
            temporary_metadata.write_text(json.dumps(saved, indent=2)+'\n')
            os.replace(temporary_metadata, metadata)
            summary[skill] = saved
            print(profile_name(size, support), skill, report, flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--requests', required=True)
    parser.add_argument('--single', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--directory', default='joint_carry/stage2_size_rsi/inset15_single')
    parser.add_argument('--source', choices=('single', 'stage2'), default='single')
    parser.add_argument('--inset', type=float, default=ANCHOR_INSET)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    build_digest = contract_digest(args.source, args.inset)
    requests = json.loads(Path(args.requests).read_text())
    if args.single:
        if len(requests) != 1:
            raise ValueError('Each physics worker handles one asset profile')
        request = requests[0]
        results = [build(request['size'], request.get('support'), args.directory, args.source, args.inset)]
    else:
        results = []
        jobs = Path(args.report).parent/'rsi_jobs'
        jobs.mkdir(parents=True, exist_ok=True)
        def run_request(index_request):
            index, request = index_request
            folder = Path(args.directory)/profile_name(request['size'], request.get('support'))
            skills = ('putDown',) if request.get('support') is not None else ('pickUp', 'carryWith', 'putDown')
            saved = {}
            for skill in skills:
                path, meta = folder/(skill+'.npz'), folder/(skill+'.json')
                if not path.exists() or not meta.exists():
                    break
                record = json.loads(meta.read_text())
                if (record.get('contract_digest') != build_digest or not record.get('accepted')
                        or hashlib.sha256(path.read_bytes()).hexdigest() != record.get('sha256')):
                    break
                saved[skill] = record
            if len(saved) == len(skills):
                for record in saved.values():
                    record['worker_attempts'] = 0
                return saved
            source = jobs/(str(os.getpid())+'_'+str(index)+'.json')
            report = source.with_suffix('.report.json')
            source.write_text(json.dumps([request])+'\n')
            command = [sys.executable, __file__, '--single', '--requests', str(source),
                       '--directory', args.directory, '--report', str(report),
                       '--source', args.source, '--inset', str(args.inset)]
            for attempt in range(2):
                result = subprocess.run(command)
                if result.returncode == 0:
                    records = json.loads(report.read_text())[0]
                    for record in records.values():
                        record['worker_attempts'] = attempt + 1
                    return records
                print('Profile worker failed:', request, 'attempt', attempt + 1, 'exit', result.returncode, flush=True)
            raise subprocess.CalledProcessError(result.returncode, command)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for index, result in enumerate(executor.map(run_request, enumerate(requests)), 1):
                results.append(result)
                print('Completed', index, '/', len(requests), flush=True)
    Path(args.report).write_text(json.dumps(results, indent=2)+'\n')
    if any(not skill['accepted'] for result in results for skill in result.values()):
        raise SystemExit('Some size/skill profiles have no physically screened RSI; see report')


if __name__ == '__main__':
    main()
