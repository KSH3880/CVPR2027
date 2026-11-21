import argparse
import fcntl
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import psutil

from suite import ROOT, OUT, OLD, SETTINGS, MODELS, SEEDS, jobs, key, command, cpu_command, source_hashes, read, write, sha

GPU_CAPS = {6: 88000, 7: 90000}


class ExistingProcess:
    def __init__(self, record, task):
        self.pid = record['pid']
        self.created = record['created']
        self.outputs = [task['marker']] if record['cpu'] or task['kind'] == 'reference' else [
            task['path']/'result.json', task['path']/'policy_verification.json']

    def poll(self):
        try:
            process = psutil.Process(self.pid)
            if process.create_time() == self.created and process.status() != psutil.STATUS_ZOMBIE:
                return None
        except psutil.NoSuchProcess:
            pass
        return 0 if all(p.exists() for p in self.outputs) else 1

    def wait(self, timeout):
        deadline = time.monotonic()+timeout
        while self.poll() is None:
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(str(self.pid), timeout)
            time.sleep(.1)
        return self.poll()


def memory():
    totals = subprocess.check_output(['nvidia-smi', '-i', '6,7', '--query-gpu=index,memory.used',
        '--format=csv,noheader,nounits'], text=True)
    used = dict(tuple(map(int, line.split(','))) for line in totals.splitlines())
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
        '--format=csv,noheader,nounits'], text=True)
    usage = {}
    for line in raw.splitlines():
        try:
            pid, value = map(int, line.split(','))
            usage[pid] = value
        except ValueError:
            pass
    return used, usage


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    for p, h in read(OLD/'formal_sources.json').items():
        if sha(ROOT/p) != h:
            raise ValueError('Original pilot source changed: '+p)
    for setting in ('enable_sit', 'coupled_3'):
        link = OUT/f'reference/{setting}/general_seed1'
        link.parent.mkdir(parents=True, exist_ok=True)
        target = OLD/f'reference/{setting}/general_seed1'
        if not link.exists():
            link.symlink_to(target, target_is_directory=True)
        if link.resolve() != target.resolve():
            raise ValueError('Reference link mismatch')
    expected = dict(gpus=list(GPU_CAPS), memory_caps_mib=GPU_CAPS, envs=64, seeds=SEEDS, steps=900,
        models={k: str(v) for k, v in MODELS.items()}, settings=SETTINGS, delays_seconds=[0, 2, 4],
        total_conditions=270, reused=30, new_evaluations=240, jobs=jobs(),
        source_sha256=source_hashes(), reuse_rule='first900 frames of existing1050step seed1 records',
        aggregation='within each seed, equal mean of three settings; then mean/sample std across seeds')
    manifest = OUT/'manifest.json'
    if manifest.exists():
        if read(manifest)['source_sha256'] != expected['source_sha256']:
            raise ValueError('Suite code/checkpoint changed since prepare')
    else:
        write(manifest, expected)
    checks = []
    for job in jobs():
        cmd, env = command(job, Path(job['source']))
        if any(cmd[cmd.index(flag)+1] != value for flag, value in (
                ('--seed', str(job['seed'])), ('--num_envs', '64'),
                ('--num_agents', str(SETTINGS[job['setting']])))):
            raise ValueError('Command mismatch')
        if env['TOKENHSI_GPU'] != '6' or env['DCT_STEPS'] != '900':
            raise ValueError('Runtime mismatch')
        checks.append(key(job))
        cmd7, env7 = command(job, Path(job['source']), gpu=7)
        if cmd7 != cmd or env7 != dict(env, TOKENHSI_GPU='7'):
            raise ValueError('GPU7 changed evaluation contract')
    write(OUT/'launch_verification.json', dict(passed=True, commands=len(checks)*2,
        new=240, reused=30, gpus=list(GPU_CAPS), memory_caps_mib=GPU_CAPS))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--workers', type=int, default=13)
    parser.add_argument('--gpu7-workers', type=int, default=6)
    parser.add_argument('--resume-active', type=Path)
    args = parser.parse_args()
    limits = {6: args.workers, 7: args.gpu7_workers}
    prepare()
    if args.prepare_only:
        print('270 commands verified; 240 new, 30 reused', flush=True)
        return
    lock = (OUT/'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (OUT/'supervisor.pid').write_text(str(os.getpid()))
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    tasks = []
    for seed in SEEDS:
        for setting in SETTINGS:
            job = dict(setting=setting, model='ours', delay=0, seed=seed)
            path = OUT/f'reference/{setting}/general_seed{seed}'
            tasks.append(dict(kind='reference', job=job, path=path, marker=path/'reference_info.json'))
    for setting in SETTINGS:
        if setting in ('enable_sit', 'coupled_3'):
            continue
        for model in MODELS:
            job = dict(setting=setting, model=model, delay=4, seed=1)
            path = OUT/'development'/key(job)
            tasks.append(dict(kind='development', job=job, path=path,
                marker=OUT/'development_analysis'/f'{key(job)}.json'))
    for job in jobs():
        tasks.append(dict(kind='formal', job=job, path=Path(job['source']),
            marker=OUT/'analysis'/f'{key(job)}.json'))
    running, finished, failed = {}, [], []
    start_time = time.time()
    last_launch = last_report = 0.
    state = dict(completed=False, gpus=list(GPU_CAPS), memory_caps_mib=GPU_CAPS,
        worker_limits=limits, pid=os.getpid(), started_unix=start_time)
    if args.resume_active:
        handoff = read(args.resume_active)
        finished = handoff['state']['finished']
        start_time = handoff['state']['started_unix']
        state['started_unix'] = start_time
        for record in handoff['running']:
            task = next(t for t in tasks if t['kind'] == record['kind'] and key(t['job']) == record['job'])
            process = ExistingProcess(record, task)
            log = task['marker'].with_suffix('.log') if record['cpu'] else task['path']/'run.log'
            running[process.pid] = dict(process=process, task=task, cpu=record['cpu'],
                gpu=record.get('gpu', 6) if not record['cpu'] else None,
                start=record['created'], log=str(log))
            print('ADOPT', record['job'], process.pid, flush=True)
    cpu_env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')

    def launch(task, cpu=False, gpu=None):
        job, path = task['job'], task['path']
        if cpu:
            cmd = cpu_command(job, path, task['marker'], 135 if task['kind'] == 'development' else 900)
            env = cpu_env
            logpath = task['marker'].with_suffix('.log')
        else:
            if path.exists():
                raise ValueError('Incomplete output needs inspection: '+str(path))
            path.mkdir(parents=True)
            cmd, env = command(job, path, task['kind'] == 'reference',
                135 if task['kind'] == 'development' else 900, gpu=gpu)
            write(path/'launch.json', dict(gpu=gpu, job=key(job), started_unix=time.time()))
            logpath = path/'run.log'
        logpath.parent.mkdir(parents=True, exist_ok=True)
        with logpath.open('w') as log:
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        running[process.pid] = dict(process=process, task=task, cpu=cpu, gpu=gpu, start=time.time(), log=str(logpath))
        print('START', 'CPU' if cpu else f'GPU{gpu}', task['kind'], key(job), process.pid, flush=True)

    try:
        while True:
            now = time.time()
            for pid, entry in list(running.items()):
                code = entry['process'].poll()
                if code is None:
                    continue
                del running[pid]
                record = dict(kind=entry['task']['kind'], job=key(entry['task']['job']), cpu=entry['cpu'],
                    gpu=entry['gpu'], seconds=now-entry['start'], finished_unix=now, returncode=code)
                (failed if code else finished).append(record)
                print('FAIL' if code else 'DONE', record, flush=True)
            if stopping or (OUT/'STOP').exists():
                state['stopped'] = True
                break
            if failed:
                state['failed'] = failed
                break
            pending = [t for t in tasks if not t['marker'].exists()]
            if not pending and not running:
                state.update(completed=True, verified_formal=sum(t['kind'] == 'formal' for t in tasks),
                    pending=0, updated_unix=now, elapsed_s=now-start_time)
                break
            used, usage = memory()
            active_paths = {str(e['task']['path']) for e in running.values()}
            gpu_entries = [e for e in running.values() if not e['cpu']]
            reserve = dict.fromkeys(GPU_CAPS, 0)
            for pid, entry in running.items():
                if entry['cpu']:
                    continue
                measured = usage.get(pid, 0)
                warm = now-entry['start'] >= 30 and measured >= 4000
                reserve[entry['gpu']] += 0 if warm else max(0, 6000-measured)
            available = [gpu for gpu, cap in GPU_CAPS.items()
                if sum(e['gpu'] == gpu for e in gpu_entries) < limits[gpu]
                and used[gpu]+reserve[gpu]+6000 <= cap]
            cpu_count = sum(e['cpu'] for e in running.values())
            for task in pending:
                if str(task['path']) in active_paths:
                    continue
                path, job = task['path'], task['job']
                if task['kind'] != 'reference' and (path/'policy_verification.json').exists() and (path/'result.json').exists():
                    if cpu_count < 2:
                        launch(task, cpu=True)
                        cpu_count += 1
                    continue
                if (OUT/'PAUSE_QUEUE').exists() or not available or now-last_launch < 3:
                    continue
                reference = OUT/f"reference/{job['setting']}/general_seed{job['seed']}/reference_info.json"
                if task['kind'] != 'reference' and not reference.exists():
                    continue
                if task['kind'] == 'formal' and job['setting'] not in ('enable_sit', 'coupled_3'):
                    gate = OUT/f"development_analysis/{job['setting']}/{job['model']}_d4_seed1.json"
                    if not gate.exists():
                        continue
                launch(task, gpu=available[0])
                last_launch = now
                break
            if now-last_report >= 30:
                subprocess.run([sys.executable, str(ROOT/'FSM/dct/analyze_suite.py'), '--summarize'],
                    cwd=ROOT, env=cpu_env, check=True, stdout=subprocess.DEVNULL)
                last_report = now
            formal_done = sum(t['marker'].exists() for t in tasks if t['kind'] == 'formal')
            progress = []
            for pid, entry in running.items():
                step = 0
                if not entry['cpu'] and entry['task']['kind'] != 'reference':
                    with open(entry['log'], 'rb') as handle:
                        handle.seek(max(0, os.fstat(handle.fileno()).st_size-12000))
                        matches = re.findall(rb'\[formal eval\] step=(\d+)/(\d+)', handle.read())
                    if matches:
                        step = int(matches[-1][0])
                progress.append(dict(pid=pid, cpu=entry['cpu'], kind=entry['task']['kind'],
                    gpu=entry['gpu'],
                    job=key(entry['task']['job']), elapsed_s=now-entry['start'], step=step,
                    gpu_memory_mib=usage.get(pid, 0)))
            state.update(gpu_used_mib=used[6], gpu_usage_mib=used,
                pending_reservation_mib=reserve, elapsed_s=now-start_time,
                verified_formal=formal_done, total_formal=270, pending=len(pending), running=progress,
                finished=finished, failed=failed, updated_unix=now)
            write(OUT/'status.json', state)
            time.sleep(3)
    finally:
        for pid, entry in running.items():
            if entry['process'].poll() is None:
                os.killpg(pid, signal.SIGTERM)
        for entry in running.values():
            try:
                entry['process'].wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(entry['process'].pid, signal.SIGKILL)
        state.update(running=[], finished=finished, failed=failed, ended_unix=time.time())
        write(OUT/'status.json', state)
        subprocess.run([sys.executable, str(ROOT/'FSM/dct/analyze_suite.py'), '--summarize'], cwd=ROOT, env=cpu_env, check=True)
    if failed:
        raise RuntimeError('Suite stopped after failed job; inspect status/log')


if __name__ == '__main__':
    main()
