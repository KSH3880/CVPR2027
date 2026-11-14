"""Mixed8/16 fixed-radius variant: fresh scenes, one-shot schedules, legacy rollout."""
import os,json,sys,subprocess,threading,hashlib
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor,as_completed
from run_full_suite import gpu_job
from run_eval16_batch import ROOT,RT
from sct_scheduler.config import read_settings,chat_endpoint,DEFAULT_BASE_URL,DEFAULT_REASONING_EFFORT
from sct_scheduler.client import OneShotScheduler,chat_transport,loads
from sct_scheduler.executable import validate_executable
PYTHON='/home/hwanhee/anaconda3/envs/tokenhsi_koo/bin/python'

def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)

def main():
 tag=datetime.now().strftime('%Y%m%d_%H%M%S');base=ROOT/'output_etc/check'/('mixed8_16_radius3_7p5_'+tag);gpu=RT/'output_etc/check'/base.name;base.mkdir();gpu.mkdir()
 rows=[dict(table='table1',name=f'mixed_{n}h',people=n,seed=s,gpu=g,key=f'table1/mixed_{n}h/general_seed{s}') for n,g in [(8,6),(16,6)] for s in [1,6,7]]
 os.environ.update(SCT_GROUP_RADIUS_VARIANT='8h3m_16h7p5m',SCT_FREEZE_IMPL='legacy',SCT_PROTOCOL_VERSION='cvpr2027-evaluation-v1.0.0')
 settings=read_settings();prompt=(ROOT/'prompts/system.txt').read_text();transport0=chat_transport(chat_endpoint(settings.get('SCT_LLM_BASE_URL') or DEFAULT_BASE_URL),settings['SCT_LLM_MODEL'],settings['SCT_LLM_API_KEY'],timeout_s=180,json_mode=True,reasoning_effort=settings.get('SCT_LLM_REASONING_EFFORT') or DEFAULT_REASONING_EFFORT)
 def transport(messages):
  assert messages[0]['content']==prompt,'Prompt changed during run'
  return transport0(messages)
 files=[ROOT/'prompts/system.txt',RT/'tokenhsi/env/tasks/multi_agent/stage2_group_layout.py',RT/'eval/stage2_scalability_eval.py',RT/'tokenhsi/learning/multi_agent/formal_eval.py',RT/'batch_eval_entry.py',ROOT/'sct_scheduler/isaac_adapter.py']
 hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
 save(base/'manifest.json',dict(jobs=rows,envs=64,horizon=900,total_scenes=384,group_radius_m={'8':3,'16':7.5},freeze='legacy',seeds=[1,6,7],new_initial_scenes=True,model=settings['SCT_LLM_MODEL'],code_hashes=hashes,protocol_variant='user requested spacing override; other protocol rules unchanged',retry_policy='none'))
 (base/'system_prompt.txt').write_text(prompt)
 state=dict(stage='capture',completed=False,gpu_output=str(gpu),total_evaluations=6,total_scenes=384,api_done=0,api_total=384,api_progress_percent=0.,api_failed=0,api_nonexecutable=0,captured=[],evaluated=[],verified=[],running={},failures=[]);lock=threading.Lock()
 def update():save(base/'status.json',state)
 def unchanged():
  for name,digest in hashes.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,'Active code changed: '+name
 def phase(capture):
  def lane(device):
   for job in [j for j in rows if j['gpu']==device]:
    with lock:state['running'][str(device)]={'phase':'capture' if capture else 'eval','job':job['key']};update()
    try:
     unchanged();p=gpu_job(base,gpu,job,64,capture=capture)
     if capture:
      with lock:state['captured'].append(job['key']);update()
     else:
      with lock:state['evaluated'].append(job['key']);update()
      assert json.loads((p/'first_policy_input.json').read_text())['passed']
      conditions=json.loads((p/'scalability_protocol.json').read_text());expected=3. if job['people']==8 else 7.5
      assert conditions['group_spawn_radius']==expected and conditions['group_spawn_gap'] is None
      cmd=[PYTHON,str(ROOT/'verify_full_suite_job.py'),str(p),str(gpu/'reference'/job['key']),'--schedules',str(base/'api'/job['key']),'--envs','64','--steps','900']
      with (p/'verification.log').open('w') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
      with lock:state['verified'].append(job['key']);update()
    except Exception as e:
     with lock:state['failures'].append({'job':job['key'],'error':str(e)});update()
     break
    finally:
     with lock:state['running'].pop(str(device),None);update()
  with ThreadPoolExecutor(max_workers=2) as pool:
   fs=[pool.submit(lane,g) for g in [6]]
   for f in fs:f.result()
 update();print(json.dumps(dict(output=str(base))),flush=True)
 phase(True)
 if len(state['captured'])!=6:state['stage']='failed';update();return
 state['stage']='api';update();work=[]
 for job in rows:
  for i in range(64):work.append((job,i,loads((gpu/'reference'/job['key']/'requests'/f'env_{i:04}'/'request.json').read_text())))
 def call(item):
  job,i,req=item;dest=base/'api'/job['key']/f'env_{i:04}';error=None
  try:OneShotScheduler(transport).schedule(req,dest)
  except Exception as e:error=str(e)
  executable=True
  try:validate_executable(req,loads((dest/'response.raw.txt').read_text()))
  except Exception:executable=False
  return dict(job=job['key'],environment=i,error=error,executable=executable)
 audits=[]
 with ThreadPoolExecutor(max_workers=16) as pool:
  for f in as_completed([pool.submit(call,item) for item in work]):
   row=f.result();audits.append(row);state['api_done']+=1;state['api_failed']+=row['error'] is not None;state['api_nonexecutable']+=not row['executable'];state['api_progress_percent']=round(state['api_done']/384*100,2);update()
   if row['error']:save(base/'api_failures.json',[r for r in audits if r['error']])
 save(base/'api_audit.json',audits);state['stage']='eval';update();phase(False)
 state['stage']='summarizing';update()
 try:
  with (base/'summary.log').open('w') as log:subprocess.run([PYTHON,str(ROOT/'verify_full_suite_job.py'),'--summarize',str(base)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
  p=base/'report.md';p.write_text('8H group-center radius 3m;16H radius 7.5m;64env exploration;legacy freeze.\n\n'+p.read_text())
 except Exception as e:state['failures'].append({'job':'summary','error':str(e)})
 state.update(stage='complete' if len(state['verified'])==6 and not state['failures'] else 'complete_with_failures',completed=True);update()
 print(json.dumps(state),flush=True)

if __name__=='__main__':main()
