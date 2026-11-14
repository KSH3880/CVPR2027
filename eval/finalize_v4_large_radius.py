import contextlib
import fcntl
import json
from pathlib import Path
import time

import run_v4_large_radius as run


def main():
    output = run.OUTPUT
    run.existing.stage2_xy100.install()
    checked = {}
    while True:
        states = run.existing.read(output/'status.json')
        for key, state in states.items():
            if key in checked or state['phase'] not in ('complete', 'failed'):
                continue
            destination = run.existing.destination(state)
            if not (destination/'result.json').exists():
                continue
            with (output/f'{key}_revalidation.log').open('w') as log:
                try:
                    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                        checked[key] = dict(valid=True, proof=run.existing.validate(state))
                except Exception as error:
                    checked[key] = dict(valid=False, error=str(error))
            run.write(output/'revalidation.json', checked)
        lock = (output/'queue.lock').open()
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            time.sleep(10)
            continue
        break
    states = run.existing.read(output/'status.json')
    for name in ('status.json', 'summary.json', 'report.md', 'completion_verification.json'):
        source = output/name
        backup = output/('before_revalidation_'+name)
        if source.exists() and not backup.exists():
            backup.write_bytes(source.read_bytes())
    for key, result in checked.items():
        if result['valid']:
            states[key]['phase'] = 'complete'
            states[key]['source'] = str(run.existing.destination(states[key]))
            states[key].pop('error', None)
    run.STATES = states
    run.report()
    manifest = run.existing.read(output/'manifest.json')
    run.write(output/'completion_verification.json', dict(
        completed=sum(s['phase']=='complete' for s in states.values()), expected=4,
        checkpoint_unchanged=run.existing.sha(run.CHECKPOINT)==manifest['checkpoint_sha256'],
        revalidation=checked,
        validator_change='Accept recorded all-participant falls before step900; use actual frame counts',
        simulation_sources_unchanged={p: run.existing.sha(Path(p))==digest
            for p,digest in manifest['source_sha256'].items()
            if not p.endswith('run_v4_group_radius_eval.py')},
        current_validator_sha256={str(p):run.existing.sha(p) for p in
            (run.ROOT/'eval/run_stage2_scalability.py', run.ROOT/'eval/run_v4_group_radius_eval.py')}))


if __name__ == '__main__':
    main()
