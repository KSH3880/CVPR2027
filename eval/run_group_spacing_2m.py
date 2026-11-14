from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import os
import subprocess
import time
import traceback

import run_group_spacing_1m as one
from run_group_spacing import ROOT, LOCK, write, sha, summarize
from run_coopft_tables import memory

OUTPUT=ROOT/'output_etc/check/table1_group_spacing_env64_seed1_20261111/2m'
one.OUTPUT=OUTPUT


def launch(command,*args,**kwargs):
    if isinstance(command,list):
        command=[str(ROOT/'eval/group_spacing_2m_eval.py') if item==str(ROOT/'eval/group_spacing_eval.py')
                 else item for item in command]
    return one.original_popen(command,*args,**kwargs)


subprocess.Popen=launch


class TwoMeter(one.OneMeter):
    def __init__(self):
        super().__init__()
        self.jobs=[dict(j,gap=2.) for j in self.jobs]
        self.references=[dict(j,gap=2.) for j in self.references]
        self.state['gpus']=[5,6,7]
        self.extra_hashes.update({str(p):sha(p) for p in (ROOT/'eval/run_group_spacing_2m.py',
                                                        ROOT/'eval/group_spacing_2m_eval.py')})
        self.manifest.update(protocol='table1_group_spacing_2m_pilot',group_spawn_gap=2.,gpus=[5,6,7],
            jobs=self.jobs,extra_source_hashes=self.extra_hashes,
            coopft_gpu_guard='GPU5 in-memory adapter; GPUs6/7 native')
        write(self.output/'manifest.json',self.manifest)
        self.save()

    def partial(self):
        super().partial()
        path=self.output/'partial_report.md'
        path.write_text(path.read_text().replace('그룹 간격1m','그룹 간격2m',1))

    def run(self):
        jobs=self.references+self.jobs
        pending=[j for j in jobs if not self.proof(j).exists()]
        for job in jobs:
            if self.proof(job).exists():
                self.state['references_verified' if job['model']=='reference' else 'verified'].append(dict(key=job['key'],reused=True))
        self.state['phase']='references_and_ready_evaluations'
        reference_gpu={j['name']:gpu for j,gpu in zip(self.references,(6,7,5,6))}
        futures={}
        self.partial()
        with ThreadPoolExecutor(max_workers=30) as pool:
            while pending or futures:
                snapshot=memory()
                snapshot[5]=one.memory5()
                external={5:0,6:0,7:0}
                other=OUTPUT.parent/'1m/status.json'
                if other.exists():
                    state=json.loads(other.read_text())
                    external[5]=sum(max(v['reservation_mib']-snapshot[5]['processes'].get(v.get('pid'),0),0)
                                    for v in state['running'].values())
                for gpu in sorted((5,6,7),key=lambda g:-snapshot[g]['free']):
                    active=[j for _,(g,j) in futures.items() if g==gpu]
                    while len(active)<10 and not self.state['failed']:
                        reserved=sum(max(self.state['running'][j['key']]['reservation_mib']-
                            snapshot[gpu]['processes'].get(self.state['running'][j['key']].get('pid'),0),0) for j in active)
                        ready=[j for j in pending if (j['model']=='reference' and reference_gpu[j['name']]==gpu)
                               or (j['model']!='reference' and self.proof(dict(j,model='reference')).exists())]
                        ready.sort(key=lambda j:(j['model']!='reference',j['mode']!='mixed',j['people']))
                        job=next((j for j in ready if snapshot[gpu]['free']-reserved-external[gpu]-10000 >=
                                  (6000 if j['people']==8 else 8000)),None)
                        if job is None:
                            break
                        pending.remove(job)
                        with LOCK:
                            self.state['running'][job['key']]=dict(gpu=gpu,
                                reservation_mib=6000 if job['people']==8 else 8000,phase='starting',started_unix=time.time())
                            self.save()
                        futures[pool.submit(self.run_job,job,gpu)]=gpu,job
                        active.append(job)
                done,_=wait(futures,timeout=5,return_when=FIRST_COMPLETED) if futures else (set(),set())
                for future in done:
                    gpu,job=futures.pop(future)
                    with LOCK:
                        detail=self.state['running'].pop(job['key'])
                        try:
                            future.result()
                            self.state['references_verified' if job['model']=='reference' else 'verified'].append(dict(key=job['key'],**detail))
                        except Exception:
                            self.state['failed'].append(dict(key=job['key'],error=traceback.format_exc()))
                            print(self.state['failed'][-1]['error'],flush=True)
                        self.save()
                    self.partial()
                self.state['pending']=len(pending)
                self.state['gpu_memory']={g:{k:v for k,v in row.items() if k!='processes'} for g,row in snapshot.items()}
                self.save()
                if self.state['failed'] and not futures:
                    raise RuntimeError('Queue stopped after failure; other processes were not interrupted')
                if not futures and pending:
                    time.sleep(5)
        if len(self.state['verified'])!=18 or len(self.state['references_verified'])!=4:
            raise ValueError('Incomplete2m evaluation')
        summarize(self.output,self.jobs)
        if any(sha(p)!=value for p,value in {**self.hashes,**self.extra_hashes}.items()):
            raise ValueError('Sources or checkpoints changed')
        write(self.output/'completion_verification.json',dict(completed=True,new_runs=18,references=4,
            envs=64,seeds=[1],gpus=[5,6,7],horizon=900,group_spawn_gap=2.,initial_scenes_exact=True,
            cpu_sr_and_error_verified=True,checkpoint_hashes_unchanged=True,
            fsm_legacy_first_observation_difference_disclosed=True))
        self.manifest['completed']=True
        write(self.output/'manifest.json',self.manifest)
        self.state.update(completed=True,phase='complete',pending=0)
        self.save()


if __name__=='__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('CPU supervisor requires CUDA_VISIBLE_DEVICES=')
    TwoMeter().run()
