import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time

from random_delay import ROOT, OUT, jobs, key, command, cpu_command, prepare, write


def memory():
    used = int(subprocess.check_output(['nvidia-smi', '-i', '6', '--query-gpu=memory.used',
        '--format=csv,noheader,nounits'], text=True).strip())
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
        '--format=csv,noheader,nounits'], text=True)
    usage = {}
    for row in raw.splitlines():
        try:
            pid, value = map(int, row.split(','))
            usage[pid] = value
        except ValueError:
            pass
    return used, usage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    prepare()
    if args.prepare_only:
        return
    lock = (OUT/'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (OUT/'supervisor.pid').write_text(str(os.getpid()))
    tasks = []
    probes = {('enable_sit', 'fsm'), ('enable_sit', 'tokenhsi'), ('coupled_4', 'fsm'), ('coupled_3', 'ours')}
    for kind in ('development', 'formal'):
        for job in jobs():
            if kind == 'development' and (job['setting'], job['model']) not in probes:
                continue
            folder = 'development' if kind == 'development' else 'rollouts'
            analysis = 'development_analysis' if kind == 'development' else 'analysis'
            tasks.append(dict(kind=kind, job=job, path=OUT/folder/key(job),
                marker=OUT/analysis/f'{key(job)}.json', steps=210 if kind == 'development' else 900))
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
        subprocess.run([sys.executable, str(ROOT/'FSM/dct/analyze_random.py'), '--summarize'],
            cwd=ROOT, env=cpu_env, check=True)

    def launch(task, cpu=False):
        path, job = task['path'], task['job']
        if cpu:
            cmd = cpu_command(job, path, task['marker'], task['steps'])
            env, logpath = cpu_env, task['marker'].with_suffix('.log')
        else:
            path.mkdir(parents=True, exist_ok=False)
            cmd, env = command(job, path, task['steps'])
            logpath = path/'run.log'
        logpath.parent.mkdir(parents=True, exist_ok=True)
        with logpath.open('w') as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        running[proc.pid] = dict(proc=proc, task=task, cpu=cpu, start=time.time())
        print('START', 'CPU' if cpu else 'GPU6', task['kind'], key(job), proc.pid, flush=True)

    state = dict(pid=os.getpid(), gpu=6, memory_cap_mib=85000, completed=False, started_unix=started)
    try:
        while True:
            now = time.time()
            for pid, entry in list(running.items()):
                result = entry['proc'].poll()
                if result is None:
                    continue
                record = dict(job=key(entry['task']['job']), kind=entry['task']['kind'], cpu=entry['cpu'],
                    seconds=now-entry['start'], returncode=result)
                (failed if result else finished).append(record)
                del running[pid]
                print('FAIL' if result else 'DONE', record, flush=True)
            formal_done = sum(t['marker'].exists() for t in tasks if t['kind'] == 'formal')
            dev_done = all(t['marker'].exists() for t in tasks if t['kind'] == 'development')
            state.update(verified_formal=formal_done, total_formal=30, development_passed=dev_done,
                updated_unix=now, elapsed_s=now-started)
            if failed or stopping or (OUT/'STOP').exists():
                break
            if formal_done == 30 and not running:
                state['completed'] = True
                break
            used, usage = memory()
            active = {str(e['task']['path']) for e in running.values()}
            reserve = sum(0 if now-e['start'] >= 30 and usage.get(pid, 0) >= 4000 else
                max(0, 6000-usage.get(pid, 0)) for pid,e in running.items() if not e['cpu'])
            cpu_count = sum(e['cpu'] for e in running.values())
            gpu_count = len(running)-cpu_count
            for task in tasks:
                if task['marker'].exists() or str(task['path']) in active:
                    continue
                path = task['path']
                if (path/'result.json').exists() and (path/'policy_verification.json').exists():
                    if cpu_count < 2:
                        launch(task, cpu=True)
                        cpu_count += 1
                    continue
                if task['kind'] == 'formal' and not dev_done:
                    continue
                if (OUT/'PAUSE_QUEUE').exists() or gpu_count >= 16 or now-last_launch < 3 or used+reserve+6000 > 85000:
                    continue
                launch(task)
                last_launch = now
                break
            if now-last_report >= 30:
                summarize()
                last_report = now
            state.update(gpu_used_mib=used, pending_reservation_mib=reserve,
                running=[dict(pid=pid, job=key(e['task']['job']), kind=e['task']['kind'],
                    cpu=e['cpu'], elapsed_s=now-e['start'], gpu_memory_mib=usage.get(pid, 0)) for pid,e in running.items()],
                finished=finished, failed=failed)
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
        raise RuntimeError('Random delay evaluation failed; inspect status and logs')


if __name__ == '__main__':
    main()
