import fcntl
import os
from pathlib import Path
import subprocess
import time

from run_v4_formal import (OUTPUT, GPUS, jobs, destination, proof_path, ready, prepare_reference,
                           launch, validate, partial, read, write, sha, torch, stage2_xy100)


RESERVATIONS = {2: 6000, 3: 6500, 4: 7000, 8: 8500, 16: 11000}


def memory():
    rows = subprocess.check_output(['nvidia-smi', '-i', '5,6,7',
        '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
    result = {}
    for line in rows.splitlines():
        index, uuid, free = [x.strip() for x in line.split(',')]
        result[int(index)] = dict(uuid=uuid, free=int(free), processes={})
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
        '--format=csv,noheader,nounits'], text=True)
    for line in rows.splitlines():
        uuid, pid, used = [x.strip() for x in line.split(',')]
        for row in result.values():
            if row['uuid'] == uuid and used.isdigit():
                row['processes'][int(pid)] = int(used)
    return result


def alive(pid):
    try:
        state = (Path('/proc')/str(pid)/'stat').read_text().split(') ', 1)[1].split()[0]
        return state != 'Z'
    except FileNotFoundError:
        return False


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU supervisor requires CUDA_VISIBLE_DEVICES=')
    torch.set_num_threads(1)
    stage2_xy100.install()
    lock = (OUTPUT/'supervisor_fast.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (OUTPUT/'supervisor.pid').write_text(str(os.getpid())+'\n')
    all_jobs = jobs()
    state = read(OUTPUT/'acceleration_adoption.json')
    state.update(completed=False, phase='accelerated_evaluation', failed=[], workers_per_gpu=10,
                 reserve_mib=10000, startup_reservations_mib=RESERVATIONS)
    hashes = read(OUTPUT/'source_hashes.json')
    extra_hashes = {str(Path(__file__).resolve()): sha(Path(__file__).resolve())}
    write(OUTPUT/'acceleration_source_hashes.json', extra_hashes)
    references = read(OUTPUT/'reference_hashes.json')
    manifest = read(OUTPUT/'manifest.json')
    manifest.update(scheduler='accelerate_v4_formal.py', workers_per_gpu=10, reserve_mib=10000,
                    startup_reservations_mib=RESERVATIONS, adopted_children_not_restarted=True)
    write(OUTPUT/'manifest.json', manifest)
    running, pending = {}, []
    for job in all_jobs:
        key = job['key']
        if key in state['verified']:
            continue
        if key in state['running']:
            running[key] = None, job, state['running'][key]['gpu'], state['running'][key]['pid']
        elif proof_path(job).exists():
            state['verified'].append(key)
        elif (destination(job)/'result.json').exists():
            validate(job)
            state['verified'].append(key)
        else:
            pending.append(job)
    while pending or running:
        snapshot = memory()
        for gpu in sorted((5, 6), key=lambda g: -snapshot[g]['free']):
            active = [row for row in running.values() if row[2] == gpu]
            while pending and len(active) < 10 and not state['failed']:
                reserved = sum(max(RESERVATIONS[j['people']]-snapshot[gpu]['processes'].get(pid, 0), 0)
                               for _, j, _, pid in active)
                job = next((j for j in pending if ready(j) and
                    snapshot[gpu]['free']-reserved-10000 >= RESERVATIONS[j['people']]), None)
                if job is None:
                    break
                prepare_reference(job, references)
                process = launch(job, gpu)
                pending.remove(job)
                row = process, job, gpu, process.pid
                running[job['key']] = row
                active.append(row)
                state['running'][job['key']] = dict(pid=process.pid, gpu=gpu, started_unix=time.time())
                print('START accelerated', job['key'], 'GPU'+str(gpu), 'PID', process.pid, flush=True)
        for key, (process, job, gpu, worker_pid) in list(running.items()):
            if (process.poll() is None) if process is not None else alive(worker_pid):
                continue
            try:
                if process is not None and process.returncode:
                    raise RuntimeError(f'Worker exited {process.returncode}')
                validate(job)
                state['verified'].append(key)
                print('VERIFIED', key, 'GPU'+str(gpu), flush=True)
            except Exception as error:
                state['failed'].append(dict(key=key, error=str(error)))
                print('FAILED', key, str(error), flush=True)
            del running[key]
            del state['running'][key]
        state.update(pending=len(pending), updated_unix=time.time(),
            gpu_memory={g:dict(free=row['free'], active=sum(x[2] == g for x in running.values()))
                        for g, row in snapshot.items()})
        write(OUTPUT/'status.json', state)
        partial(all_jobs, state)
        if state['failed'] and not running:
            raise RuntimeError('Queue stopped after failure')
        if pending or running:
            time.sleep(5)
    if len(state['verified']) != 42 or any(sha(Path(p)) != value for p, value in
                                         {**hashes, **references, **extra_hashes}.items()):
        raise ValueError('Source changed or evaluation incomplete')
    from summarize_v4_formal import summarize
    summarize(OUTPUT)
    write(OUTPUT/'completion_verification.json', dict(completed=True, runs=42, table1=24, table2=18,
        envs=512, seeds=[1, 6, 7], gpus=GPUS, first_actual_policy_input_exact=True, policy_preserved=True,
        cpu_sr_counts_errors_verified=True, table2_cpu_replay_verified=True, group_radii_verified=True,
        latest20cm_cooperation_verified=True, source_reference_hashes_unchanged=True,
        adopted_children_not_restarted=True, acceleration_source_hashes=extra_hashes))
    state['completed'] = manifest['completed'] = True
    state['phase'] = 'complete'
    write(OUTPUT/'manifest.json', manifest)
    write(OUTPUT/'status.json', state)


if __name__ == '__main__':
    main()
