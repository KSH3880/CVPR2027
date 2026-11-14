"""Automatically continue the active gap0.2 run to the full suite, reusing 2H."""
import json,subprocess,time,os
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parent
prior=ROOT/'output_etc/check/fewshot_2h_gap02_20261112_165256'
base=ROOT/'output_etc/check'/('gap02_full_continuation_'+datetime.now().strftime('%Y%m%d_%H%M%S'));base.mkdir()
target=ROOT/'output_etc/check'/('fewshot_full_gap02_'+base.name.rsplit('_',2)[-2]+'_'+base.name.rsplit('_',1)[-1])
state=dict(stage='waiting_for_current_2h',completed=False,prior=str(prior),full_output=str(target),total_scenes=1920,total_evaluations=30,eta_seconds=None,eta_at=None,eta_scope='current_stage')
def save():
 tmp=base/'status.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(base/'status.json')
save();print(str(base),flush=True)
try:
 while True:
  s=json.loads((prior/'status.json').read_text())
  state['progress']=s
  for k in ['eta_seconds','eta_at','eta_scope']:state[k]=s.get(k)
  save()
  if s.get('completed'):break
  if s['stage']=='failed':raise RuntimeError('2H predecessor failed; inspect logs')
  time.sleep(5)
 cmd=['/home/hwanhee/anaconda3/envs/tokenhsi_koo/bin/python',str(ROOT/'run_fewshot_full.py'),'--output',str(target),'--reuse-api',s['api_output'],'--reuse-eval',s['eval_output']]
 with (base/'full.log').open('w') as f:
  p=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=f,stderr=subprocess.STDOUT)
  state.update(stage='full_suite',child_pid=p.pid);save()
  while p.poll() is None:
   path=target/'status.json'
   if path.exists():
    s=json.loads(path.read_text());state['progress']=s
    for k in ['eta_seconds','eta_at','eta_scope']:state[k]=s.get(k)
    save()
   time.sleep(5)
  if p.returncode:raise RuntimeError(f'Full suite exit {p.returncode}')
 state.update(stage='complete',completed=True,eta_seconds=0);save()
except Exception as exc:
 state.update(stage='failed',error=str(exc));save();raise
