import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ['SINGLE_CARRY'] = 'table1'
import torch
import run_single_size as base
from single_size_results import geometry, read, write, verify
from single_common_seed_results import OUTPUT, PREVIOUS, REUSED, NEW_SEEDS, ALL_SEEDS, NEW_RUNS, TOTAL_RUNS, summarize, result_path

base.OUTPUT = OUTPUT
base.GPUS = (5, 6, 7)


def job(model, seed, phase='formal', steps=900):
    return dict(variant='common', model=model, seed=seed, phase=phase, steps=steps,
                key=f'common/{model}/seed{seed}')


def run_phase(jobs, state):
    pending = list(jobs)
    running = {}
    while pending or running:
        for pid, (proc, item, gpu) in list(running.items()):
            if proc.poll() is None:
                continue
            path = base.destination(item)
            logfile = OUTPUT/'logs'/(item['phase']+'_'+item['key'].replace('/', '_')+'.log')
            if proc.returncode and 'daemon could not be verified' in logfile.read_text() and not (path/'first_policy_input.json').exists() and item.get('retry', 0) < 2:
                archive = OUTPUT/'startup_failures'/str(pid)
                archive.parent.mkdir(exist_ok=True)
                path.rename(archive)
                logfile.rename(archive/'startup.log')
                state['startup_retries'].append(dict(job=item, archive=str(archive)))
                pending.append(dict(item, retry=item.get('retry', 0)+1))
            elif proc.returncode:
                state['failed'].append(dict(job=item, exit_code=proc.returncode))
            else:
                try:
                    reference = OUTPUT/f'reference/common/general_seed{item["seed"]}'
                    proof = dict(geometry=geometry(path, 'common')) if item['phase']=='reference' else verify(path, reference, 'common', item['model'], item['steps'])
                    write(path/'verification.json', proof)
                    state['done'].append(item['phase']+'/'+item['key'])
                    if item['phase']=='formal':
                        state['verified'].append(item['key'])
                        summarize(state['verified'])
                except Exception as error:
                    state['failed'].append(dict(job=item, error=repr(error)))
            del running[pid]
        if state['failed']:
            pending.clear()
        while pending:
            ready = {pid for pid,(_,item,_) in running.items() if (base.destination(item)/'first_policy_input.json').exists()}
            used, allocations = base.memory()
            choices = []
            for gpu in base.GPUS:
                assigned = [(pid, item) for pid, (_,item,g) in running.items() if g==gpu]
                reserve = sum(512 if pid in ready else max(base.RESERVE-allocations.get(pid, 0), 0) for pid,item in assigned)
                if used[gpu]+reserve+base.RESERVE+1024 <= base.LIMIT:
                    choices.append((len(assigned), used[gpu]+reserve, gpu))
            if not choices:
                break
            gpu = min(choices)[2]
            item = pending.pop(0)
            path = base.destination(item)
            path.mkdir(parents=True, exist_ok=False)
            cmd, env = base.command(item, gpu)
            logfile = OUTPUT/'logs'/(item['phase']+'_'+item['key'].replace('/', '_')+'.log')
            with logfile.open('w') as log:
                proc = subprocess.Popen(cmd, cwd=base.ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT)
            running[proc.pid] = (proc, item, gpu)
            write(path/'execution.json', dict(pid=proc.pid, gpu=gpu, command=cmd, started_unix=time.time(), log=str(logfile)))
            print('START', item['phase'], item['key'], 'GPU', gpu, 'PID', proc.pid, flush=True)
            time.sleep(2)
        used, _ = base.memory()
        with (OUTPUT/'memory_observations.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(unix=time.time(), memory_mib=used))+'\n')
        state.update(gpu_memory_mib=used, running={str(pid):dict(job=j,gpu=g) for pid,(_,j,g) in running.items()}, pending=len(pending))
        write(OUTPUT/'status.json', state)
        if pending or running:
            time.sleep(5)
    if state['failed']:
        raise RuntimeError('Evaluation failed; inspect status and logs')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT/'logs').mkdir(exist_ok=True)
    lock = (OUTPUT/'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUTPUT/'status.json').exists():
        raise ValueError('Existing run must be preserved')
    jobs = [job(model, seed) for seed in NEW_SEEDS for model in base.MODELS]
    previous_manifest = read(PREVIOUS/'manifest.json')
    sources = set(previous_manifest['source_sha256'])
    sources.update(str(base.ROOT/'eval'/name) for name in ('run_single_common_seeds.py', 'single_common_seed_results.py', 'single_carry_sizes.py'))
    hashes = {p:base.sha(p) for p in sorted(sources)}
    reused_hashes = {str(PREVIOUS/name):base.sha(PREVIOUS/name) for name in ('summary.json', 'report.md', 'manifest.json')}
    for model in base.MODELS:
        checkpoint = str(base.CHECKPOINTS[model])
        if hashes[checkpoint] != previous_manifest['source_sha256'][checkpoint]:
            raise ValueError('Checkpoint differs from original comparison')
        for seed in REUSED:
            path = result_path(model, seed)
            proof = read(path.parent/'verification.json')
            if not all(proof[k] for k in ('initial_state_exact', 'first_input_exact', 'cpu_sr_error_exact', 'policy_preserved')):
                raise ValueError('Reused rollout was not verified')
            for name in ('result.json', 'verification.json', 'policy_verification.json'):
                reused_hashes[str(path.parent/name)] = base.sha(path.parent/name)
    for seed in REUSED:
        source = PREVIOUS/f'reference/common/general_seed{seed}'
        target = OUTPUT/f'reference/common/general_seed{seed}'
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.symlink_to(source, target_is_directory=True)
        for name in ('initial.pt', 'observed_scene.pt'):
            reused_hashes[str(source/name)] = base.sha(source/name)
    manifest = dict(condition='② existing common sizes', carry_sizes='table1', seeds=ALL_SEEDS,
        new_seeds=NEW_SEEDS, reused_seeds=REUSED, envs=512, agents=1, objects=2, steps=900,
        gpus=base.GPUS, gpu_total_limit_mib=base.LIMIT, startup_reservation_mib=base.RESERVE,
        jobs=jobs, expected_new_runs=NEW_RUNS, expected_total_runs=TOTAL_RUNS, source_sha256=hashes,
        reused_sha256=reused_hashes, previous_output=str(PREVIOUS), started_unix=time.time())
    write(OUTPUT/'manifest.json', manifest)
    if args.prepare_only:
        print(f'Prepared {NEW_RUNS} new runs + {len(REUSED)*4} verified reused runs', flush=True)
        return
    (OUTPUT/'supervisor.pid').write_text(str(os.getpid())+'\n')
    state = dict(completed=False, done=[], verified=[], failed=[], startup_retries=[], phase='reference', started_unix=time.time())
    summarize([])
    run_phase([job('ours', seed, 'reference') for seed in NEW_SEEDS], state)
    reference_hashes = {str(OUTPUT/f'reference/common/general_seed{seed}'/name):base.sha(OUTPUT/f'reference/common/general_seed{seed}'/name)
                        for seed in NEW_SEEDS for name in ('initial.pt', 'observed_scene.pt')}
    write(OUTPUT/'reference_hashes.json', reference_hashes)
    state['phase'] = 'development'
    run_phase([job(model, NEW_SEEDS[0], 'probe', 60) for model in base.MODELS]+[job('ours', 1, 'probe', 60)], state)
    state['phase'] = 'formal'
    run_phase(jobs, state)
    for path, digest in {**hashes, **reused_hashes, **reference_hashes}.items():
        if base.sha(path) != digest:
            raise ValueError('Tracked file changed: '+path)
    summarize(state['verified'])
    if len(state['verified']) != NEW_RUNS or not read(OUTPUT/'summary.json')['completed']:
        raise ValueError('Incomplete coverage')
    state.update(completed=True, phase='complete', finished_unix=time.time())
    write(OUTPUT/'status.json', state)
    write(OUTPUT/'completion_verification.json', dict(new_runs=NEW_RUNS, reused_runs=len(REUSED)*4, total_runs=TOTAL_RUNS,
        new_episodes=NEW_RUNS*512, total_episodes=TOTAL_RUNS*512, new_shared_references=len(NEW_SEEDS), development_runs=5,
        development_seed=NEW_SEEDS[0], all_tracked_files_preserved=True, task_counts_match_between_models=True,
        startup_retries=len(state['startup_retries']), elapsed_minutes=(state['finished_unix']-state['started_unix'])/60))
    print(f'COMPLETE {NEW_RUNS} new / {TOTAL_RUNS} total', flush=True)


if __name__ == '__main__':
    main()
