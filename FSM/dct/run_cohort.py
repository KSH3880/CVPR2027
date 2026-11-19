import argparse
import fcntl
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from cohort import ROOT, OUT, SETTINGS, SEEDS, CONDITION, GPU_CAPS, PROBES
from cohort import jobs, key, command, cpu_command, prepare, read, write
from run_suite import ExistingProcess


def eligible_gpu_caps(requested):
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.total',
        '--format=csv,noheader,nounits'], text=True)
    totals = dict(tuple(map(int, line.split(','))) for line in raw.splitlines())
    caps = {gpu: min(cap, totals[gpu] - 1024) for gpu, cap in requested.items()
            if gpu in totals and totals[gpu] > 6524}
    if not caps:
        raise RuntimeError(f'Evaluation requires an eligible physical GPU from {tuple(requested)} with more than 6524 MiB')
    return caps


def memory(gpus):
    selected = ','.join(map(str, gpus))
    used = subprocess.check_output(['nvidia-smi', '-i', selected, '--query-gpu=index,memory.used',
        '--format=csv,noheader,nounits'], text=True)
    processes = subprocess.check_output(['nvidia-smi', '-i', selected,
        '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], text=True)
    usage = {}
    for row in processes.splitlines():
        try:
            pid, amount = map(int, row.split(','))
            usage[pid] = amount
        except ValueError:
            continue
    return dict(tuple(map(int, row.split(','))) for row in used.splitlines()), usage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume-active', type=Path)
    parser.add_argument('--local-gpu0', action='store_true', help='Use only physical GPU 0 on the local PC')
    args = parser.parse_args()
    prepare()
    if args.prepare_only:
        return
    gpu_caps = eligible_gpu_caps({0: 88000} if args.local_gpu0 else GPU_CAPS)
    lock = (OUT/'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (OUT/'supervisor.pid').write_text(str(os.getpid()))
    tasks = []
    for seed in SEEDS:
        for setting in SETTINGS:
            job = dict(setting=setting, seed=seed, model='ours', delay=0)
            path = OUT/f'reference/{setting}/general_seed{seed}'
            tasks.append(dict(kind='reference', job=job, path=path, marker=path/'reference_info.json', steps=900))
    for setting, model in PROBES:
        for delay in (0, CONDITION):
            job = dict(setting=setting, model=model, delay=delay, seed=1)
            tasks.append(dict(kind='development', job=job, path=OUT/'development'/key(job),
                marker=OUT/'development_analysis'/f'{key(job)}.json', steps=210))
    for job in jobs():
        tasks.append(dict(kind='formal', job=job, path=OUT/'rollouts'/key(job),
            marker=OUT/'analysis'/f'{key(job)}.json', steps=900))
    running, finished, failed = {}, [], []
    started = time.time()
    last_launch = last_report = 0
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    cpu_env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')

    def summarize():
        subprocess.run([sys.executable, str(ROOT/'FSM/dct/analyze_cohort.py'), '--summarize'],
            cwd=ROOT, env=cpu_env, check=True)

    def launch(task, cpu=False, gpu=None):
        path, job = task['path'], task['job']
        if cpu:
            cmd = cpu_command(job, path, task['marker'], task['steps'])
            env, logpath = cpu_env, task['marker'].with_suffix('.log')
        else:
            path.mkdir(parents=True, exist_ok=False)
            cmd, env = command(job, path, task['kind'] == 'reference', task['steps'], gpu)
            logpath = path/'run.log'
            write(path/'launch.json', dict(gpu=gpu, job=job, started_unix=time.time()))
        logpath.parent.mkdir(parents=True, exist_ok=True)
        with logpath.open('w') as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        running[proc.pid] = dict(proc=proc, task=task, cpu=cpu, gpu=gpu, start=time.time(), log=logpath)
        print('START', 'CPU' if cpu else f'GPU{gpu}', task['kind'], key(job), proc.pid, flush=True)

    state = dict(pid=os.getpid(), gpus=list(gpu_caps), memory_caps_mib=gpu_caps,
        completed=False, started_unix=started, total_formal=180, logical_evaluations=60)
    if args.resume_active:
        handoff = read(args.resume_active)
        started = handoff['state']['started_unix']
        state['started_unix'] = started
        finished = handoff['state']['finished']
        for record in handoff['running']:
            task = next(t for t in tasks if t['kind'] == record['kind'] and key(t['job']) == record['job'])
            proc = ExistingProcess(record, task)
            log = task['marker'].with_suffix('.log') if record['cpu'] else task['path']/'run.log'
            running[proc.pid] = dict(proc=proc, task=task, cpu=record['cpu'], gpu=record['gpu'],
                start=record['created'], log=log)
            print('ADOPT', record['job'], proc.pid, flush=True)
    try:
        while True:
            now = time.time()
            for pid, entry in list(running.items()):
                result = entry['proc'].poll()
                if result is None:
                    continue
                record = dict(job=key(entry['task']['job']), kind=entry['task']['kind'], cpu=entry['cpu'],
                    gpu=entry['gpu'], seconds=now-entry['start'], returncode=result)
                if result == 0:
                    task = entry['task']
                    outputs = [task['marker']] if entry['cpu'] or task['kind'] == 'reference' else [
                        task['path']/'result.json', task['path']/'policy_verification.json']
                    if not all(p.exists() for p in outputs):
                        record['returncode'] = result = 1
                (failed if result else finished).append(record)
                del running[pid]
                print('FAIL' if result else 'DONE', record, flush=True)
            formal_done = sum(t['marker'].exists() for t in tasks if t['kind'] == 'formal')
            dev_done = all(t['marker'].exists() for t in tasks if t['kind'] == 'development')
            state.update(verified_formal=formal_done, development_passed=dev_done,
                updated_unix=now, elapsed_s=now-started)
            if failed or stopping or (OUT/'STOP').exists():
                break
            if formal_done == 180 and not running:
                state['completed'] = True
                break
            if dev_done and not (OUT/'development_verification.json').exists():
                rows = [read(t['marker']) for t in tasks if t['kind'] == 'development']
                for setting in SETTINGS:
                    selected = [r for r in rows if r['setting'] == setting]
                    for field in ('initial_signature', 'input_signature'):
                        if len({r[field] for r in selected}) > 1:
                            raise ValueError('Development initial state/input mismatch')
                write(OUT/'development_verification.json', dict(passed=True, runs=len(rows),
                    paired_conditions=True, shared_initial_input=True, steps=210))
            used, usage = memory(gpu_caps)
            active = {str(e['task']['path']) for e in running.values()}
            reserve = {gpu: sum(0 if now-e['start'] >= 30 and usage.get(pid, 0) >= 4000 else
                max(0, 5500-usage.get(pid, 0)) for pid,e in running.items() if not e['cpu'] and e['gpu'] == gpu)
                for gpu in gpu_caps}
            available = sorted([g for g,cap in gpu_caps.items() if used[g]+reserve[g]+5500 <= cap],
                               key=lambda g: used[g]+reserve[g])
            cpu_count = sum(e['cpu'] for e in running.values())
            for task in tasks:
                if task['marker'].exists() or str(task['path']) in active:
                    continue
                path, job = task['path'], task['job']
                if task['kind'] != 'reference' and (path/'result.json').exists() and (path/'policy_verification.json').exists():
                    baseline = task['marker'].parent/f"{job['model']}_d0_seed{job['seed']}.json"
                    if cpu_count < 2 and (job['delay'] == 0 or baseline.exists()):
                        launch(task, cpu=True)
                        cpu_count += 1
                    continue
                if task['kind'] != 'reference':
                    reference = OUT/f"reference/{job['setting']}/general_seed{job['seed']}/reference_info.json"
                    if not reference.exists():
                        continue
                if task['kind'] == 'formal' and not dev_done:
                    continue
                if (OUT/'PAUSE_QUEUE').exists() or not available or now-last_launch < 3:
                    continue
                launch(task, gpu=available[0])
                last_launch = now
                break
            if now-last_report >= 60:
                summarize()
                last_report = now
            progress = []
            for pid,e in running.items():
                step = 0
                if not e['cpu'] and e['task']['kind'] != 'reference':
                    with e['log'].open('rb') as handle:
                        handle.seek(max(0, os.fstat(handle.fileno()).st_size-10000))
                        matches = re.findall(rb'\[formal eval\] step=(\d+)/(\d+)', handle.read())
                    if matches:
                        step = int(matches[-1][0])
                progress.append(dict(pid=pid, job=key(e['task']['job']), kind=e['task']['kind'],
                    cpu=e['cpu'], gpu=e['gpu'], elapsed_s=now-e['start'], step=step,
                    gpu_memory_mib=usage.get(pid, 0)))
            state.update(gpu_usage_mib=used, pending_reservation_mib=reserve,
                running=progress, finished=finished, failed=failed)
            write(OUT/'status.json', state)
            time.sleep(3)
    finally:
        for pid,e in running.items():
            if e['proc'].poll() is None:
                os.killpg(pid, signal.SIGTERM)
        for pid,e in running.items():
            try:
                e['proc'].wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(pid, signal.SIGKILL)
        state.update(running=[], finished=finished, failed=failed, ended_unix=time.time())
        write(OUT/'status.json', state)
        summarize()
    if failed:
        raise RuntimeError('Cohort evaluation failed; inspect status and logs')


if __name__ == '__main__':
    main()
