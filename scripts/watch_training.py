"""Restart only CUDA launch-timeout failures. STOP marker disables the watcher."""
import datetime
import fcntl
import os
from pathlib import Path
import time

from resume_after_limit import ROOT, read, write, identity, continue_training

STATE = ROOT / 'output/training_watchdog.json'
STOP = ROOT / 'output/training_watchdog.stop'


def eligible(old, current, log, stopped):
    return (not stopped and old['pid'] == current['pid']
            and old['output_path'] == current['output_path']
            and current['status'] == 'running'
            and 'RuntimeError: CUDA error: the launch timed out and was terminated' in log
            and 'MAX EPOCHS NUM!' not in log)


def candidates(active):
    files = list(Path(active['output_path']).glob('*/nn/TokenHSIPushKD_????????.pth'))
    fallback = Path(active['resume_checkpoint'])
    if fallback.is_file():
        files.append(fallback)
    return sorted(set(files), key=lambda p: int(p.stem.rsplit('_', 1)[1]), reverse=True)


def checkpoint(active):
    import torch
    contract = read(active['contract'])
    for path in candidates(active):
        try:
            state = torch.load(path, map_location='cpu', weights_only=True)
            epoch = int(path.stem.rsplit('_', 1)[1])
            assert state['epoch'] == epoch and state['optimizer']['state']
            assert state['push_kd_contract'] == contract
            return path, epoch
        except Exception as error:
            print(f'Skipping invalid checkpoint {path}: {error}', flush=True)
    raise RuntimeError('No valid checkpoint; refusing fresh training')


def main():
    os.chdir(ROOT)
    with (ROOT / 'output/training_watchdog.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        restarts = []
        while not STOP.exists():
            old = read(ROOT / 'output/active_training.json')
            start = identity(old['pid'])
            if start is None:
                raise RuntimeError('Arm only while training is alive')
            status = dict(status='watching', watcher_pid=os.getpid(), training_pid=old['pid'],
                          training_start_identity=start, output_path=old['output_path'],
                          restarts=restarts, reboot_persistent=False, stop_file=str(STOP))
            write(STATE, status)
            while identity(old['pid']) == start:
                if STOP.exists():
                    write(STATE, dict(status, status='disabled'))
                    return
                time.sleep(10)
            current = read(ROOT / 'output/active_training.json')
            log = Path(old['log']).read_text()
            if not eligible(old, current, log, STOP.exists()):
                write(STATE, dict(status, status='stopped_without_restart'))
                return
            # At most three restarts in a rolling hour; repeated immediate failures stop.
            now = time.time()
            if sum(now-r['time'] < 3600 for r in restarts) >= 3:
                write(STATE, dict(status, status='retry_limit_reached'))
                return
            write(STATE, dict(status, status='cooldown'))
            for _ in range(6):
                time.sleep(10)
                if STOP.exists():
                    write(STATE, dict(status, status='disabled'))
                    return
            if not eligible(old, read(ROOT / 'output/active_training.json'), log, STOP.exists()):
                write(STATE, dict(status, status='cancelled_active_changed'))
                return
            path, epoch = checkpoint(old)
            plan = dict(training=old, tensorboard=read(ROOT / 'output/active_tensorboard.json'),
                        expected_epoch=epoch, checkpoint=str(path), state_path=str(STATE),
                        target_max_epochs=-1, status='restarting')
            restarts.append(dict(time=now, epoch=epoch, checkpoint=str(path),
                                 at=datetime.datetime.now().astimezone().isoformat()))
            print(f'Restarting CUDA timeout from {path}', flush=True)
            # Startup failure is fail-closed: never spawn a second unverified learner.
            continue_training(plan)
        write(STATE, dict(status='disabled', watcher_pid=os.getpid()))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        write(STATE, dict(status='failed', watcher_pid=os.getpid(), error=repr(error)))
        raise
