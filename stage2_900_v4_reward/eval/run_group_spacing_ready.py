from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import os
from pathlib import Path
import time
import traceback

from run_group_spacing import Suite, LOCK, VERIFY_LOCK, memory, write, sha, summarize, ROOT


def wait_reference(suite, job, pid):
    proc = Path('/proc')/str(pid)
    while (proc/'stat').exists():
        try:
            state = (proc/'stat').read_text().split(') ',1)[1].split()[0]
        except FileNotFoundError:
            break
        if state == 'Z':
            break
        time.sleep(2)
    with VERIFY_LOCK:
        suite.validate(job)
    print('VERIFIED adopted',job['key'],flush=True)


def run(suite, previous):
    jobs = suite.references+suite.jobs
    pending = [j for j in jobs if not suite.proof(j).exists() and j['key'] not in previous['running']]
    for job in jobs:
        if suite.proof(job).exists():
            suite.state['references_verified' if job['model']=='reference' else 'verified'].append(dict(key=job['key'],reused=True))
    suite.state['phase']='references_and_ready_evaluations'
    suite.manifest['scheduler']='run_group_spacing_ready.py'
    suite.manifest['scheduler_sha256']=sha(__file__)
    write(suite.output/'manifest.json',suite.manifest)
    futures={}
    with ThreadPoolExecutor(max_workers=2*suite.args.workers_per_gpu) as pool:
        for job in jobs:
            if job['key'] not in previous['running'] or suite.proof(job).exists():
                continue
            state = previous['running'][job['key']]
            suite.state['running'][job['key']]=dict(state,adopted=True)
            future=pool.submit(wait_reference,suite,job,state['pid'])
            futures[future]=state['gpu'],job
        while pending or futures:
            snapshot=memory()
            for gpu in sorted((6,7),key=lambda g:-snapshot[g]['free']):
                active=[(f,j) for f,(g,j) in futures.items() if g==gpu]
                while len(active)<suite.args.workers_per_gpu and not suite.state['failed']:
                    reserved=sum(max(suite.state['running'][j['key']]['reservation_mib']-
                        snapshot[gpu]['processes'].get(suite.state['running'][j['key']].get('pid'),0),0) for _,j in active)
                    ready=[j for j in pending if j['model']=='reference' or suite.proof(dict(j,model='reference')).exists()]
                    ready.sort(key=lambda j:(j['model']!='reference',j['mode']!='mixed',j['seed']!=1,j['people']))
                    job=next((j for j in ready if snapshot[gpu]['free']-reserved-suite.args.reserve_mib >= (6000 if j['people']==8 else 8000)),None)
                    if job is None:
                        break
                    pending.remove(job)
                    with LOCK:
                        suite.state['running'][job['key']]=dict(gpu=gpu,reservation_mib=6000 if job['people']==8 else 8000,phase='starting',started_unix=time.time())
                        suite.save()
                    future=pool.submit(suite.run_job,job,gpu)
                    futures[future]=gpu,job
                    active.append((future,job))
            done,_=wait(futures,timeout=5,return_when=FIRST_COMPLETED) if futures else (set(),set())
            for future in done:
                gpu,job=futures.pop(future)
                with LOCK:
                    detail=suite.state['running'].pop(job['key'])
                    try:
                        future.result()
                        suite.state['references_verified' if job['model']=='reference' else 'verified'].append(dict(key=job['key'],**detail))
                    except Exception:
                        suite.state['failed'].append(dict(key=job['key'],error=traceback.format_exc()))
                        print(suite.state['failed'][-1]['error'],flush=True)
                    suite.save()
            suite.state['pending']=len(pending)
            suite.state['gpu_memory']={g:{k:v for k,v in row.items() if k!='processes'} for g,row in snapshot.items()}
            suite.save()
            if suite.state['failed'] and not futures:
                raise RuntimeError('Queue stopped after failure; existing processes were not interrupted')
            if not futures and pending:
                time.sleep(5)
    if len(suite.state['verified'])!=len(suite.jobs) or len(suite.state['references_verified'])!=len(suite.references):
        raise ValueError('Incomplete evaluation')
    summarize(suite.output,suite.jobs)
    if any(sha(p)!=value for p,value in suite.hashes.items()) or sha(__file__)!=suite.manifest['scheduler_sha256']:
        raise ValueError('Sources or checkpoints changed')
    write(suite.output/'completion_verification.json',dict(completed=True,new_runs=len(suite.jobs),references=len(suite.references),
        envs=64,seeds=sorted({j['seed'] for j in suite.jobs}),horizon=900,initial_scenes_exact=True,cpu_sr_and_error_verified=True,
        configured_group_spacing_verified=True,checkpoint_hashes_unchanged=True,fsm_legacy_first_observation_difference_disclosed=True))
    suite.manifest['completed']=True
    write(suite.output/'manifest.json',suite.manifest)
    suite.state.update(completed=True,phase='complete',pending=0)
    suite.save()


if __name__=='__main__':
    from argparse import Namespace
    output=ROOT/'output_etc/check/table1_no_group_spacing_env64_seeds167_20261111_retry1024'
    previous=json.loads((output/'adoption_status.json').read_text())
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('CPU supervisor requires CUDA_VISIBLE_DEVICES=')
    run(Suite(Namespace(output=output,workers_per_gpu=10,reserve_mib=10000)),previous)
