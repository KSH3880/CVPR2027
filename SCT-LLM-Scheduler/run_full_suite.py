"""Saved one-shot LLM schedules: Mixed scalability + Table2, 64 envs each."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from run_eval16_batch import ROOT, RT, save
from sct_scheduler.client import OneShotScheduler, chat_transport, loads
from sct_scheduler.config import read_settings, chat_endpoint, DEFAULT_BASE_URL, DEFAULT_REASONING_EFFORT
from sct_scheduler.core import build_messages
from sct_scheduler.executable import validate_executable

SETTINGS={'enable_sit':2,'enable_climb':2,'enable_ontop':2,'coupled_2':2,'coupled_3':3,'coupled_4':4}


def jobs():
    rows=[dict(table='table1',name=f'mixed_{n}h',people=n,seed=s)
          for n in (2,4,8,16) for s in (1,6,7)]
    rows += [dict(table='table2',name=k,people=n,seed=s)
             for k,n in SETTINGS.items() for s in (1,6,7)]
    for i,row in enumerate(rows):
        row.update(gpu=6+i%2,key=f"{row['table']}/{row['name']}/general_seed{row['seed']}")
    return rows


def gpu_job(base,gpu_root,job,envs,*,capture=False,preflight=False):
    ref=gpu_root/'reference'/job['table']/job['name']
    output=(ref/f"general_seed{job['seed']}" if capture else gpu_root/'eval'/job['key'])
    output.mkdir(parents=True,exist_ok=False)
    policy=json.loads((ROOT/'stage1_policy.json').read_text())
    cmd=['python','batch_eval_entry.py','--task','HumanoidMACoupledPush','--cfg_train',
         'tokenhsi/data/cfg/train/rlg/stage2_distill_900_v2.yaml','--cfg_env','',
         '--motion_file','tokenhsi/data/dataset_stage1_jointpush.yaml','--checkpoint',
         str(RT/'checkpoints/stage2_v2/nn/Stage2Distill900V2_00002000.pth'),
         '--seed',str(job['seed']),'--num_envs',str(envs),'--num_agents',str(job['people']),
         '--num_objects',str(2*job['people']),'--test','--eval','--headless','--no_video',
         '--eval_protocol','general','--eval_repeats','1','--task_graph','random_scenario',
         '--enable_size_mode','per_action_intersection','--coupled_assignment','ordered',
         '--output_path',str(output)]
    if job['people']>4:cmd.append('--group_spawn_over4')
    env=dict(os.environ,TOKENHSI_GPU=str(job['gpu']),SCT_PHYSICAL_GPU=str(job['gpu']),
        TOKENHSI_CONDA_ENV='tokenhsi_koo',PYTHONDONTWRITEBYTECODE='1',
        TORCH_EXTENSIONS_DIR=str(ROOT/'output_etc/check/torch_extensions'),
        CUDA_CACHE_PATH=str(ROOT/'output_etc/check/cuda_cache'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        STAGE2_SCENE_MODE='mixed',PYTHONHASHSEED=str(job['seed']),REFERENCE_ONLY=str(int(capture)),
        STAGE2_REFERENCE_ROOT=str(ref),STAGE1_POLICY_CHECKPOINT=policy['checkpoint_local'],
        SCT_SCHEDULE_ROOT=str(base/'api'/job['table']/job['name']),SCT_PREFLIGHT=str(int(preflight)),
        SCT_EXECUTION_ONLY='1',SCT_PREFLIGHT_EXCLUDE_LAST=str(int(preflight and job['table']=='table2')))
    env.pop('TABLE2_SETTING',None)
    if job['table']=='table2':env['TABLE2_SETTING']=job['name']
    save(output/'launch.json',dict(command=cmd,gpu=job['gpu'],capture=capture,preflight=preflight))
    with (output/'run.log').open('w') as log:
        proc=subprocess.Popen(['bash','-c','source tokenhsi/scripts/multi_agent/runtime_env.sh || exit\nexec "$@"','_',*cmd],
            cwd=RT,env=env,stdout=log,stderr=subprocess.STDOUT)
        save(output/'process.json',dict(pid=proc.pid));rc=proc.wait()
    save(output/'exit.json',dict(returncode=rc))
    if rc:raise RuntimeError(f"GPU job failed ({rc}): {output/'run.log'}")
    return output


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--envs',type=int,default=64)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();rows=jobs()
    assert args.envs>0 and 1<=args.workers<=16
    if args.prepare_only:
        print(json.dumps(dict(jobs=rows,evaluations=len(rows),scenes=len(rows)*args.envs),indent=2));return
    base=args.output.resolve();assert (ROOT/'output_etc/check').resolve() in base.parents
    base.mkdir(parents=True,exist_ok=False)
    gpu=RT/'output_etc/check'/base.name;gpu.mkdir(parents=True,exist_ok=False)
    settings=read_settings()
    effort=settings.get('SCT_LLM_REASONING_EFFORT','').strip() or DEFAULT_REASONING_EFFORT
    transport=chat_transport(chat_endpoint(settings.get('SCT_LLM_BASE_URL') or DEFAULT_BASE_URL),
        settings['SCT_LLM_MODEL'],settings['SCT_LLM_API_KEY'],timeout_s=180,json_mode=True,reasoning_effort=effort)
    source_files=['run_full_suite.py','verify_full_suite_job.py','export_initial_scene.py',
        'teacher_runtime/batch_eval_entry.py','teacher_runtime/tokenhsi/learning/multi_agent/formal_eval.py',
        'teacher_runtime/eval/table2_eval.py']
    source_files += [str(p.relative_to(ROOT)) for folder in ('sct_scheduler','prompts')
                     for p in (ROOT/folder).glob('*') if p.suffix in ('.py','.txt')]
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_files}
    for name in source_files:
        target=base/'code_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    save(base/'manifest.json',dict(jobs=rows,envs=args.envs,horizon=900,total_scenes=len(rows)*args.envs,
        model=settings['SCT_LLM_MODEL'],reasoning_effort=effort,api_workers=args.workers,
        code_hashes=hashes,independent_evaluations=False,stage1=json.loads((ROOT/'stage1_policy.json').read_text()),
        retry_policy='No automatic API retries; preserve failures; evaluate every executable raw schedule.'))
    lock=threading.RLock()
    status=dict(stage='preflight',completed=False,gpu_output=str(gpu),total_evaluations=len(rows),
        total_scenes=len(rows)*args.envs,api_done=0,api_failed=0,api_nonexecutable=0,
        captured=[],evaluated=[],verified=[],running={},failures=[])
    def update(**kw):
        with lock:status.update(kw);save(base/'status.json',status)
    def unchanged():
        for name,digest in hashes.items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
                raise RuntimeError('Active suite source changed: '+name+'; no further jobs submitted')
    def record_failure(where,exc):
        with lock:
            status['failures'].append(dict(where=where,error=str(exc)));save(base/'status.json',status)
    def verify(job,path,preflight=False):
        ref=path.parents[3]/'reference'/job['table']/job['name']/f"general_seed{job['seed']}"
        # Use the explicit matching reference root (preflight is nested separately).
        ref=(gpu/'preflight' if preflight else gpu)/'reference'/job['table']/job['name']/f"general_seed{job['seed']}"
        cmd=[sys.executable,str(ROOT/'verify_full_suite_job.py'),str(path),str(ref),
             '--schedules',str(base/'api'/job['key']),'--envs',str(2 if preflight else args.envs),
             '--steps',str(30 if preflight else 900)]
        if preflight:cmd.append('--preflight')
        with (path/'verification.log').open('w') as log:
            subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    def gpu_phase(phase):
        def lane(device):
            for job in [j for j in rows if j['gpu']==device]:
                unchanged()
                with lock:status['running'][str(device)]=dict(phase=phase,job=job['key']);save(base/'status.json',status)
                try:
                    if phase=='eval':
                        audit=json.loads((base/'api_checks'/job['key']/'summary.json').read_text())
                        if not audit['executable']:
                            raise RuntimeError('No executable scenes; GPU evaluation omitted')
                    path=gpu_job(base,gpu,job,args.envs,capture=phase=='capture')
                    with lock:
                        status['captured' if phase=='capture' else 'evaluated'].append(job['key']);save(base/'status.json',status)
                    if phase=='eval':
                        verify(job,path)
                        with lock:status['verified'].append(job['key']);save(base/'status.json',status)
                except Exception as exc:record_failure(phase+':'+job['key'],exc)
                finally:
                    with lock:status['running'].pop(str(device),None);save(base/'status.json',status)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(lane,g) for g in (6,7)]
            for f in futures:f.result()
    update()
    try:
        # Exercise largest mixed input, enabling pair and 3-person Table2 hooks.
        for key in ('mixed_16h','enable_sit','coupled_3'):
            job=next(j for j in rows if j['name']==key and j['seed']==1)
            unchanged()
            gpu_job(base,gpu/'preflight',job,2,capture=True)
            path=gpu_job(base,gpu/'preflight',job,2,preflight=True)
            verify(job,path,True)
        update(stage='capture')
        gpu_phase('capture')
        if len(status['captured'])!=len(rows):raise RuntimeError('Incomplete initial captures; API not started')
        unchanged();update(stage='api')
        work=[]
        for job in rows:
            for i in range(args.envs):
                req=loads((gpu/'reference'/job['key']/'requests'/f'env_{i:04d}'/'request.json').read_text())
                messages=build_messages(req)
                assert not any('nominal' in m['content'].lower() for m in messages)
                work.append((job,i,req))
        def api_call(item):
            job,i,req=item;dest=base/'api'/job['key']/f'env_{i:04d}'
            unchanged();err=None
            try:OneShotScheduler(transport).schedule(req,dest)
            except Exception as exc:err=str(exc)
            executable=True
            try:validate_executable(req,loads((dest/'response.raw.txt').read_text()))
            except Exception:executable=False
            return dict(job=job['key'],environment=i,validation_error=err,executable=executable)
        api_rows=[]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for f in as_completed([pool.submit(api_call,item) for item in work]):
                row=f.result();api_rows.append(row)
                with lock:
                    status['api_done']+=1;status['api_failed']+=int(row['validation_error'] is not None)
                    status['api_nonexecutable']+=int(not row['executable']);save(base/'status.json',status)
                if row['validation_error']:save(base/'api_failures.json',[r for r in api_rows if r['validation_error']])
        save(base/'api_audit.json',api_rows)
        for job in rows:
            subset=[r for r in api_rows if r['job']==job['key']]
            save(base/'api_checks'/job['key']/'summary.json',dict(executable=sum(r['executable'] for r in subset),
                 excluded=[r for r in subset if not r['executable']],validation_errors=[r for r in subset if r['validation_error']]))
        update(stage='eval');gpu_phase('eval')
        update(stage='summarizing')
        with (base/'summary.log').open('w') as log:
            subprocess.run([sys.executable,str(ROOT/'verify_full_suite_job.py'),'--summarize',str(base)],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        update(stage='complete' if not status['failures'] else 'complete_with_failures',completed=True)
    except Exception as exc:
        record_failure('supervisor',exc);update(stage='failed');raise


if __name__=='__main__':main()
