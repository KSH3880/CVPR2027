"""Mixed 2H validation before considering the remaining suite."""
import argparse, json, os, subprocess, time
from pathlib import Path
from sct_scheduler.progress import StageETA
eta=StageETA()
ROOT=Path(__file__).resolve().parent
PYTHON='/home/hwanhee/anaconda3/envs/tokenhsi_koo/bin/python'
p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--reuse-api',type=Path);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
assert 1<=a.workers<=16
base=a.output.resolve();assert ROOT/'output_etc/check' in base.parents
base.mkdir(exist_ok=False);api=base/'api_run';ev=ROOT/'output_etc/check'/('eval_'+base.name)
state=dict(stage='api',completed=False,total_scenes=192,total_evaluations=3,envs=64,seeds=[1,6,7],api_output=str(api),eval_output=str(ev),group_radius_m={'8':3,'16':7.5},independent_only_evaluation=False,api_workers=a.workers,freeze='legacy',protocol='cvpr2027-evaluation-v1.1.0')
def save():
 eta.annotate(state)
 tmp=base/'status.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(base/'status.json')
def run(cmd,log):
 with (base/log).open('w') as f:
  child=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=f,stderr=subprocess.STDOUT)
  state['child_pid']=child.pid;save()
  while child.poll() is None:
   progress=api/'status.json' if state['stage']=='api' else ev/'status.json'
   if progress.exists():
    try:state['progress']=json.loads(progress.read_text());save()
    except json.JSONDecodeError:pass
   time.sleep(5)
  if child.returncode:raise RuntimeError(f'{log}: exit {child.returncode}')
try:
 save();print(json.dumps(state),flush=True)
 run([PYTHON,str(ROOT/'replay_fewshot_api.py'),'--source',str(ROOT/'output_etc/check/eval_protocol_v1_legacy_20261111_165027'),'--additional-source',str(ROOT/'output_etc/check/mixed8_16_radius3_7p5_20261112_082344'),'--settings','mixed_2h','--workers',str(a.workers),'--output',str(api)]+(['--reuse-from',str(a.reuse_api.resolve())] if a.reuse_api else []),'api.log')
 state['stage']='eval';state.pop('progress',None);save()
 run([PYTHON,str(ROOT/'evaluate_fewshot_api.py'),'--api',str(api),'--output',str(ev),'--freeze','legacy'],'eval.log')
 evaluated=json.loads((ev/'status.json').read_text())
 if len(evaluated['verified'])!=3 or evaluated['failures']:raise RuntimeError('Incomplete 2H physical verification; inspect eval status')
 state.update(stage='ready_for_review',completed=True,remaining_suite_started=False,report=str(ev/'report.md'));save()

except Exception as exc:
 state.update(stage='failed',error=str(exc));save();raise
