import argparse
import concurrent.futures
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ind_mechanisms_common import ROOT, OUT, OURS, SCT, command, digest, write


def jobs(phase):
    result = []
    for experiment, conditions in [('a', ('solo_a','solo_b')), ('b', ('separated','cross'))]:
        if phase not in ('all', experiment):
            continue
        for seed in (1,6,7):
            for index, condition in enumerate(conditions):
                for model in ('sct','ours'):
                    preferred = 6 if (model == 'sct') == ((seed != 6) == (index == 0)) else 7
                    result.append(dict(experiment=experiment, condition=condition, model=model,
                        seed=seed, preferred_gpu=preferred, key=f'{condition}/{model}/seed{seed}'))
    return result


def memory():
    output = subprocess.check_output(['nvidia-smi','-i','6,7','--query-gpu=index,memory.used','--format=csv,noheader,nounits'], text=True)
    return {int(row.split(',')[0]):int(row.split(',')[1]) for row in output.strip().splitlines()}


def process_memory():
    output = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'], text=True)
    return {int(row.split(',')[0]):int(row.split(',')[1]) for row in output.strip().splitlines() if row.split(',')[1].strip().isdigit()}


def fits(used, reservations, reserve, cap):
    return used+reservations+reserve <= cap


def validate(path, scene, horizon):
    with (path/'cpu_validation.log').open('w') as log:
        subprocess.run([sys.executable, str(ROOT/'eval/ind_mechanisms_metrics.py'), str(path), str(scene), str(horizon)],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def run(args):
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT/'.scheduler.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = OUT/'status.json'
    planned = jobs(args.phase)
    state = dict(started_utc=datetime.now(timezone.utc).isoformat(), phase=args.phase, gpus=[6,7],
        cap_mib=args.cap_mib, max_workers_per_gpu=args.workers, expected_runs=len(planned),
        running={}, validating=[], completed=[], failed=[], waiting=[], finished=False)
    code_paths = [ROOT/'eval'/name for name in ('ind_mechanisms_eval.py','ind_mechanisms_scenes.py',
        'ind_mechanisms_metrics.py','run_ind_mechanisms.py','summarize_ind_mechanisms.py')]
    manifest = dict(checkpoints={str(p):digest(p) for p in (OURS,SCT)},
        code_sha256={str(p):digest(p) for p in code_paths if p.exists()},
        envs=512, horizon=900, seeds=[1,6,7], gpu_cap_mib=args.cap_mib,
        collision_metric='geometric body-point/surface proximity; no contact-pair attribution', jobs=planned)
    write(OUT/f'manifest_{args.phase}.json', manifest)
    pending, running, futures = [], {}, {}
    for job in planned:
        if (OUT/'runs'/job['key']/'verification.json').exists():
            proof = json.loads((OUT/'runs'/job['key']/'verification.json').read_text())
            scene = OUT/f'scenes/{job["condition"]}_seed{job["seed"]}.pt'
            result = json.loads((OUT/'runs'/job['key']/'result.json').read_text())
            if proof['scene_sha256'] != digest(scene) or result['checkpoint_sha256'] != digest(SCT if job['model']=='sct' else OURS):
                raise ValueError('Existing result provenance changed')
            state['completed'].append(job['key'])
        else:
            pending.append(job)
    cpu = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    halted = False
    last_report = 0
    while pending or running or futures:
        updated = False
        used, processes = memory(), process_memory()
        for key, active in list(running.items()):
            proc, job = active['process'], active['job']
            live = processes.get(proc.pid, 0)
            active['peak_mib'] = max(active['peak_mib'], live)
            state['running'][key]['peak_process_mib'] = active['peak_mib']
            if proc.poll() is None:
                continue
            active['log'].close()
            elapsed = time.monotonic()-active['start']
            write(active['path']/'execution.json', dict(gpu=active['gpu'], pid=proc.pid,
                elapsed_seconds=elapsed, peak_process_mib=active['peak_mib'], returncode=proc.returncode))
            del running[key]
            del state['running'][key]
            if proc.returncode:
                text = (active['path']/'run.log').read_text(errors='replace')[-16000:]
                if ('out of memory' in text.lower()) and job.get('attempt',0) == 0:
                    job['attempt'] = 1
                    job['reserve_mib'] = max(8192, active['peak_mib']+2048)
                    pending.insert(0, job)
                else:
                    state['failed'].append(dict(key=key, path=str(active['path']), reason='simulation failed'))
                    halted = True
                continue
            scene = OUT/f'scenes/{job["condition"]}_seed{job["seed"]}.pt'
            future = cpu.submit(validate, active['path'], scene, 900)
            futures[future] = (job, active['path'])
            state['validating'].append(key)
        for future, (job,path) in list(futures.items()):
            if not future.done():
                continue
            key = job['key']
            state['validating'].remove(key)
            del futures[future]
            try:
                future.result()
                canonical = OUT/'runs'/key
                if path != canonical:
                    if canonical.exists():
                        canonical.rename(canonical.with_name(canonical.name+'_failed_attempt0'))
                    path.rename(canonical)
                state['completed'].append(key)
                updated = True
            except Exception as error:
                state['failed'].append(dict(key=key, path=str(path), reason=str(error)))
                halted = True
        for gpu in (6,7):
            if halted:
                break
            current = [x for x in running.values() if x['gpu']==gpu]
            if len(current) >= args.workers:
                continue
            reservations = sum(max(0, x['reserve_mib']-processes.get(x['process'].pid,0)) for x in current)
            choices = sorted(pending, key=lambda j:(j['experiment'], j['preferred_gpu']!=gpu, j['seed']))
            if not choices:
                continue
            job = choices[0]
            if not (OUT/(job['experiment']+'_ready.json')).exists():
                continue
            if job['experiment']=='b' and any(x['experiment']=='a' for x in pending):
                continue
            reserve = job.get('reserve_mib', 6144)
            if not fits(used[gpu], reservations, reserve, args.cap_mib):
                continue
            path = OUT/'runs'/job['key']
            if path.exists():
                attempt = 1
                while path.with_name(path.name+f'_attempt{attempt}').exists():
                    attempt += 1
                path = path.with_name(path.name+f'_attempt{attempt}')
            path.mkdir(parents=True)
            scene = OUT/f'scenes/{job["condition"]}_seed{job["seed"]}.pt'
            cmd, env = command('ind_mechanisms_eval.py', path, job['seed'],
                1 if job['experiment']=='a' else 2, gpu,
                dict(IND_SCENE=str(scene), IND_MODEL=job['model']))
            log = (path/'run.log').open('w')
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            running[job['key']] = dict(process=proc, job=job, path=path, log=log, gpu=gpu,
                start=time.monotonic(), peak_mib=0, reserve_mib=reserve)
            state['running'][job['key']] = dict(pid=proc.pid,gpu=gpu,path=str(path),reserve_mib=reserve)
            pending.remove(job)
        state['waiting'] = [j['key'] for j in pending]
        state['gpu_used_mib'] = used
        state['halted'] = halted
        write(state_path, state)
        if updated:
            subprocess.run([sys.executable,str(ROOT/'eval/summarize_ind_mechanisms.py')],cwd=ROOT,check=True)
        if time.monotonic()-last_report > 60:
            print(json.dumps(dict(completed=len(state['completed']),running=state['running'],waiting=len(pending),gpu_used_mib=used)), flush=True)
            last_report = time.monotonic()
        if halted and not running and not futures:
            break
        time.sleep(3)
    cpu.shutdown(wait=True)
    state['finished'] = not pending and not state['failed']
    write(state_path, state)
    if (ROOT/'eval/summarize_ind_mechanisms.py').exists():
        subprocess.run([sys.executable,str(ROOT/'eval/summarize_ind_mechanisms.py')],cwd=ROOT,check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('a','b','all'), default='all')
    parser.add_argument('--cap-mib', type=int, default=70000)
    parser.add_argument('--workers', type=int, choices=(1,2,3), default=3)
    run(parser.parse_args())
