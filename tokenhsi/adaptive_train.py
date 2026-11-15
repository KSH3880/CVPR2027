"""Supervise Stage1 PUSH physical-asset rebuilds; never retry a failed process."""
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from utils.experiments import ROOT


def main():
    environment = os.environ.copy()
    run_name = 'Stage1PushAdaptive_'+datetime.now().strftime('%d-%H-%M-%S')
    # A manual resume starts a new run directory, as other experiment wrappers
    # do. Only our verified in-loop handoffs reuse a directory; resuming an older
    # PTH must never overwrite later checkpoints from the source run.
    directory = ROOT/environment['OUTPUT_PATH']/run_name
    directory.mkdir(parents=True, exist_ok=True)
    marker = ROOT/'output_etc/check/stage1_push_adaptive'/run_name/('restart_'+uuid.uuid4().hex+'.json')
    environment.update(TOKENHSI_ADAPTIVE_RUN_NAME=run_name,
                       TOKENHSI_ADAPTIVE_RESTART_FILE=str(marker))
    with (directory/'adaptive_train.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            marker.unlink(missing_ok=True)
            print('[adaptive] launching '+run_name+' checkpoint='+environment.get('RESUME_CHECKPOINT', 'scratch'), flush=True)
            result = subprocess.run(['bash', str(ROOT/'tokenhsi/scripts/multi_agent/launch.sh'),
                                     'stage1_push_adaptive', 'train', *sys.argv[1:]],
                                    cwd=ROOT, env=environment)
            if result.returncode:
                raise SystemExit(result.returncode)
            if not marker.exists():
                return
            handoff = json.loads(marker.read_text())
            checkpoint = Path(handoff['checkpoint'])
            if checkpoint.parent.resolve() != (directory/'nn').resolve():
                raise ValueError('Adaptive handoff points outside the current run')
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != handoff['sha256']:
                raise ValueError('Adaptive checkpoint changed before rebuild')
            environment['RESUME_CHECKPOINT'] = str(checkpoint)
            marker.unlink()


if __name__ == '__main__':
    main()
