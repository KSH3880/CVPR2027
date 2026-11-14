"""Single writer for an already-running 2H evaluator; never launches GPU work."""
import argparse,json,time,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('pipeline',type=Path);a=p.parse_args()
base=a.pipeline;path=base/'status.json';state=json.loads(path.read_text())
ev=Path(state['eval_output']);child=state['child_pid']
keys=('eta_seconds','eta_at','eta_scope','eta_basis','eta_updated_at')
for k in keys:state.setdefault(k,None)
state['eta_scope']='current_stage'
while True:
    try:progress=json.loads((ev/'status.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError):progress=None
    if progress:
        state['progress']=progress
        for k in keys:state[k]=progress.get(k)
        state['eta_scope']='current_stage'
        if progress.get('completed'):
            ok=len(progress.get('verified',[]))==3 and not progress.get('failures')
            state.update(stage='ready_for_review' if ok else 'failed',completed=ok,remaining_suite_started=False,report=str(ev/'report.md'))
        elif not Path(f'/proc/{child}').exists():
            state.update(stage='failed',error='Evaluator exited without completion status')
    tmp=path.with_name(f'.status.monitor.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(state,indent=2));tmp.replace(path)
    if state.get('completed') or state['stage']=='failed':break
    time.sleep(2)
