"""One-shot, PID-checked handover of specified learners at a saved epoch.

The manifest is explicit: original PID/start ticks, checkpoint, wrapper, output,
and log. A ready marker gates all signals until validation has passed.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import torch


def identity(pid):
    try:
        stat = Path('/proc/{}/stat'.format(pid)).read_text().split(') ', 1)[1].split()
        return stat[19] if stat[0] != 'Z' else None
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    lock = open(str(manifest) + '.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    data = json.loads(manifest.read_text())
    state_path = manifest.with_name('status.json')
    if state_path.exists():
        raise RuntimeError('Existing handover status; refusing duplicate controller')
    states = {run['name']: {'state': 'waiting_checkpoint'} for run in data['runs']}

    def persist():
        temporary = state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(states, indent=2))
        temporary.replace(state_path)

    persist()
    while any(state['state'] == 'waiting_checkpoint' for state in states.values()):
        for run in data['runs']:
            state = states[run['name']]
            if state['state'] != 'waiting_checkpoint':
                continue
            if identity(run['pid']) != run['start_ticks']:
                raise RuntimeError('Original learner vanished or PID changed: ' + run['name'])
            checkpoint = Path(run['checkpoint'])
            if not Path(data['ready_marker']).exists() or not checkpoint.is_file():
                continue
            try:
                weights = torch.load(str(checkpoint), map_location='cpu', weights_only=False)
            except (EOFError, RuntimeError, OSError):
                continue  # Writer may still be completing this numbered file.
            if weights['epoch'] != data['epoch']:
                raise RuntimeError('Wrong checkpoint epoch: ' + str(checkpoint))
            for key in ('model', 'optimizer', 'running_mean_std', 'amp_input_mean_std',
                        'relation_task_instance', 'relation_metadata'):
                if key not in weights:
                    raise RuntimeError('Missing resume state: ' + key)
            state.update(checkpoint_epoch=weights['epoch'], checkpoint_frame=weights.get('frame'),
                         checkpoint_bytes=checkpoint.stat().st_size)
            del weights
            # Validate identity again just before signalling the explicitly named learner.
            if identity(run['pid']) != run['start_ticks']:
                raise RuntimeError('PID changed before handover')
            os.kill(run['pid'], signal.SIGTERM)
            deadline = time.monotonic() + 60
            while identity(run['pid']) is not None:
                if time.monotonic() > deadline:
                    raise RuntimeError('Original learner did not exit; no replacement launched')
                time.sleep(.5)
            env = os.environ.copy()
            for key in ('CHECK', 'MAX_ITERATIONS', 'TRANSFER_CHECKPOINT', 'CUDA_VISIBLE_DEVICES',
                        'CUDA_MPS_PIPE_DIRECTORY', 'CUDA_MPS_LOG_DIRECTORY'):
                env.pop(key, None)
            env.update(TOKENHSI_GPU=str(data['gpu']), TOKENHSI_CONDA_ENV='tokenhsi',
                       RESUME_CHECKPOINT=str(checkpoint), OUTPUT_PATH=run['output_path'],
                       SEED='42', PYTHONUNBUFFERED='1')
            with open(run['log'], 'ab', buffering=0) as log:
                proc = subprocess.Popen(['bash', run['wrapper'], '2', '2048', '4'],
                                        cwd=data['cwd'], env=env, stdout=log,
                                        stderr=subprocess.STDOUT, start_new_session=True)
            state.update(state='launched', wrapper_pid=proc.pid, log=run['log'],
                         launched_at=time.strftime('%Y-%m-%d %H:%M:%S'))
            print('RESUMED', run['name'], state, flush=True)
            persist()
        time.sleep(2)


if __name__ == '__main__':
    main()
