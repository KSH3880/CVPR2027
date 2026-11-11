"""Evaluate a completed API replay against its original physical initial records."""
import argparse,json,os,threading,subprocess,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from run_full_suite import jobs,gpu_job
from run_eval16_batch import ROOT,RT

PYTHON='/home/hwanhee/anaconda3/envs/tokenhsi_koo/bin/python'

def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2));tmp.replace(p)

def main():
    p=argparse.ArgumentParser();p.add_argument('--api',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--freeze', choices=['legacy','snapshot_resume'], default='legacy');a=p.parse_args()
    api=a.api.resolve();base=a.output.resolve();assert (ROOT/'output_etc/check').resolve() in base.parents
    s=json.loads((api/'status.json').read_text());assert s['completed'] and s['api_done']==1920
    api_manifest=json.loads((api/'manifest.json').read_text());source=Path(api_manifest['source']);ref=RT/'output_etc/check'/source.name/'reference'
    base.mkdir(parents=True,exist_ok=False);gpu=RT/'output_etc/check'/base.name;gpu.mkdir(parents=True,exist_ok=False)
    (base/'api').symlink_to(api/'api',target_is_directory=True);(gpu/'reference').symlink_to(ref,target_is_directory=True)
    rows=jobs();os.environ['SCT_FREEZE_IMPL']=a.freeze
    paths=[ROOT/'sct_scheduler/isaac_adapter.py',ROOT/'sct_scheduler/snapshot_freeze.py',RT/'batch_eval_entry.py',ROOT/'verify_full_suite_job.py']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    save(base/'manifest.json',dict(jobs=rows,envs=64,horizon=900,total_scenes=1920,api_source=str(api),initial_source=str(ref),freeze_impl=a.freeze,stage1=json.loads((ROOT/'stage1_policy.json').read_text()),code_hashes=hashes,new_api_calls=0))
    state=dict(stage='eval',completed=False,freeze_impl=a.freeze,gpu_output=str(gpu),total_evaluations=30,total_scenes=1920,api_done=1920,api_failed=s['api_failed'],api_nonexecutable=s['api_nonexecutable'],evaluated=[],verified=[],running={},failures=[])
    lock=threading.Lock()
    def update():save(base/'status.json',state)
    update()
    print(json.dumps(dict(output=str(base),gpu_output=str(gpu),evaluations=30,freeze=a.freeze)),flush=True)
    def lane(device):
        for job in [j for j in rows if j['gpu']==device]:
            with lock:state['running'][str(device)]={'job':job['key']};update()
            try:
                for name,h in hashes.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,'Active source changed: '+name
                path=gpu_job(base,gpu,job,64)
                audit=json.loads((path/'release_audit.json').read_text())
                if a.freeze=='snapshot_resume':
                    assert audit['implementation']=='snapshot_resume_v1' and audit['max_restoration_error']==0
                else:
                    assert 'implementation' not in audit and 'restore_initial_velocities' not in audit
                    assert audit['runtime_gate'] is False and audit['box_state_projected'] is False
                save(path/'freeze_selection.json',dict(requested=a.freeze,audit_verified=True))
                with lock:state['evaluated'].append(job['key']);update()
                cmd=[PYTHON,str(ROOT/'verify_full_suite_job.py'),str(path),str(ref/job['key']),'--schedules',str(api/'api'/job['key']),'--envs','64','--steps','900']
                with (path/'verification.log').open('w') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                with lock:state['verified'].append(job['key']);update();print('VERIFIED '+job['key'],flush=True)
            except Exception as e:
                with lock:state['failures'].append({'job':job['key'],'error':str(e)});update()
            finally:
                with lock:state['running'].pop(str(device),None);update()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(lane,g) for g in [6,7]]
        for f in futures:f.result()
    state['stage']='summarizing';update()
    try:
        with (base/'summary.log').open('w') as log:subprocess.run([PYTHON,str(ROOT/'verify_full_suite_job.py'),'--summarize',str(base)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    except Exception as e:state['failures'].append({'job':'summary','error':str(e)})
    state.update(stage='complete' if not state['failures'] else 'complete_with_failures',completed=True);update();print(json.dumps(state),flush=True)

if __name__=='__main__':main()
