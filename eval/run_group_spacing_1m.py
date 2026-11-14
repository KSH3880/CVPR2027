from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import os
import subprocess
import time
import traceback

from run_group_spacing import Suite, ROOT, NAMES, LOCK, write, sha, summarize
from run_group_spacing_comparison import Comparison
from run_group_spacing_ready import wait_reference

OUTPUT = ROOT/'output_etc/check/table1_group_spacing_env64_seed1_20261111/1m'
original_popen=subprocess.Popen


def launch(command,*args,**kwargs):
    if isinstance(command,list):
        command=[str(ROOT/'eval/group_spacing_1m_eval.py') if item==str(ROOT/'eval/group_spacing_eval.py')
                 else item for item in command]
    return original_popen(command,*args,**kwargs)


subprocess.Popen=launch


def memory5():
    output=subprocess.check_output(['nvidia-smi','-i','5',
        '--query-gpu=uuid,memory.free,memory.used','--format=csv,noheader,nounits'],text=True)
    uuid,free,used=[v.strip() for v in output.strip().split(',')]
    processes={}
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
        '--format=csv,noheader,nounits'],text=True)
    for line in output.splitlines():
        device,pid,size=[v.strip() for v in line.split(',')]
        if device==uuid and size.isdigit():
            processes[int(pid)]=int(size)
    return dict(free=int(free),used=int(used),processes=processes)


class OneMeter(Suite):
    def __init__(self):
        super().__init__(Namespace(output=OUTPUT,workers_per_gpu=10,reserve_mib=10000))
        self.jobs=[dict(j,gap=1.) for j in self.jobs if j['seed']==1]
        self.references=[dict(j,gap=1.) for j in self.references if j['seed']==1]
        self.state.update(expected_runs=18,expected_references=4,pending=22,gpus=[5])
        self.extra_hashes={str(p):sha(p) for p in (ROOT/'eval/run_group_spacing_1m.py',
            ROOT/'eval/group_spacing_1m_eval.py',ROOT/'eval/run_group_spacing_comparison.py',
            ROOT.parent/'sct_coopft/tokenhsi/utils/config.py')}
        self.manifest.update(protocol='table1_group_spacing_1m_pilot',group_spawn_gap=1.,
            group_placement='group_center_ring',seeds=[1],gpus=[5],expected_runs=18,references=4,
            jobs=self.jobs,extra_source_hashes=self.extra_hashes,coopft_gpu_guard='GPU5 only; in-memory evaluation adapter')
        write(self.output/'manifest.json',self.manifest)
        self.save()

    def validate(self,job):
        Comparison.spacing_metadata(self,job)
        return super().validate(job)

    def partial(self):
        lines=['# Table 1 · 그룹 간격1m · 중간 결과','',
            f"64환경 × seed1. 검증 완료 {len(self.state['verified'])}/18회. 단위%.",'',
            '| Method | 8H Independent | 8H Mixed | 16H Independent | 16H Mixed |',
            '|---|---:|---:|---:|---:|']
        for model,label in NAMES.items():
            values=[]
            for people in (8,16):
                for mode in ('independent','mixed'):
                    source='stage1' if model=='fsm' and mode=='independent' else model
                    job=dict(model=source,name=f'{mode}_{people}h',seed=1)
                    if not self.proof(job).exists():
                        values.append('—');continue
                    row=json.loads((self.destination(job)/'result.json').read_text())
                    metric='Independent Interaction Success Rate' if mode=='independent' else 'Mixed Interaction Group Success Rate'
                    values.append(f"{100*row['interaction_group_scores']['region'][metric]['rate']:.2f}")
            lines.append('| '+label+' | '+' | '.join(values)+' |')
        lines+=['','SCT+FSM Independent는 SCT 공유. 단일 seed이므로 표준편차 없음. FSM의 기존 초기 관측 차이와 TokenHSI397D 입력 차이를 유지한다.']
        (self.output/'partial_report.md').write_text('\n'.join(lines)+'\n')

    def run(self):
        jobs=self.references+self.jobs
        adoption=self.output/'adoption_gpu5_guard.json'
        adopted=json.loads(adoption.read_text())['running'] if adoption.exists() else {}
        pending=[j for j in jobs if not self.proof(j).exists() and j['key'] not in adopted]
        for job in jobs:
            if self.proof(job).exists():
                self.state['references_verified' if job['model']=='reference' else 'verified'].append(dict(key=job['key'],reused=True))
        self.state['phase']='references_and_ready_evaluations'
        futures={}
        self.partial()
        with ThreadPoolExecutor(max_workers=10) as pool:
            for job in jobs:
                if job['key'] in adopted and not self.proof(job).exists():
                    state=adopted[job['key']]
                    self.state['running'][job['key']]=dict(state,adopted=True)
                    futures[pool.submit(wait_reference,self,job,state['pid'])]=job
            while pending or futures:
                snapshot=memory5()
                while len(futures)<10 and not self.state['failed']:
                    reserved=sum(max(v['reservation_mib']-snapshot['processes'].get(v.get('pid'),0),0)
                                 for v in self.state['running'].values())
                    ready=[j for j in pending if j['model']=='reference' or self.proof(dict(j,model='reference')).exists()]
                    ready.sort(key=lambda j:(j['model']!='reference',j['mode']!='mixed',j['people']))
                    job=next((j for j in ready if snapshot['free']-reserved-10000 >= (6000 if j['people']==8 else 8000)),None)
                    if job is None:
                        break
                    pending.remove(job)
                    with LOCK:
                        self.state['running'][job['key']]=dict(gpu=5,reservation_mib=6000 if job['people']==8 else 8000,
                            phase='starting',started_unix=time.time())
                        self.save()
                    futures[pool.submit(self.run_job,job,5)]=job
                done,_=wait(futures,timeout=5,return_when=FIRST_COMPLETED) if futures else (set(),set())
                for future in done:
                    job=futures.pop(future)
                    with LOCK:
                        details=self.state['running'].pop(job['key'])
                        try:
                            future.result()
                            self.state['references_verified' if job['model']=='reference' else 'verified'].append(dict(key=job['key'],**details))
                        except Exception:
                            self.state['failed'].append(dict(key=job['key'],error=traceback.format_exc()))
                            print(self.state['failed'][-1]['error'],flush=True)
                        self.save()
                    self.partial()
                self.state['pending']=len(pending)
                self.state['gpu_memory']={5:{k:v for k,v in snapshot.items() if k!='processes'}}
                self.save()
                if self.state['failed'] and not futures:
                    raise RuntimeError('Queue stopped after failure; other processes were not interrupted')
                if not futures and pending:
                    time.sleep(5)
        if len(self.state['verified'])!=18 or len(self.state['references_verified'])!=4:
            raise ValueError('Incomplete1m evaluation')
        summarize(self.output,self.jobs)
        if any(sha(p)!=value for p,value in {**self.hashes,**self.extra_hashes}.items()):
            raise ValueError('Sources or checkpoints changed')
        write(self.output/'completion_verification.json',dict(completed=True,new_runs=18,references=4,
            envs=64,seeds=[1],gpus=[5],horizon=900,group_spawn_gap=1.,
            initial_scenes_exact=True,cpu_sr_and_error_verified=True,checkpoint_hashes_unchanged=True,
            fsm_legacy_first_observation_difference_disclosed=True))
        self.manifest['completed']=True
        write(self.output/'manifest.json',self.manifest)
        self.state.update(completed=True,phase='complete',pending=0)
        self.save()


if __name__=='__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('CPU supervisor requires CUDA_VISIBLE_DEVICES=')
    OneMeter().run()
