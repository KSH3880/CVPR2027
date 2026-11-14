"""Display-only progress mirror for an already-running pipeline; no process control."""
import json,time,sys
from pathlib import Path

def mirror(path):
    parent=json.loads(path.read_text());api=Path(parent['api_output'])/'status.json';evaluation=Path(parent['eval_output'])/'status.json'
    api_state=json.loads(api.read_text()) if api.exists() else {}
    eval_state=json.loads(evaluation.read_text()) if evaluation.exists() else {}
    total=api_state.get('total',1920);done=api_state.get('api_done',0)
    parent.update(api_done=done,api_total=total,api_progress_percent=round(100*done/total,2),api_failed=api_state.get('api_failed',0),api_nonexecutable=api_state.get('api_nonexecutable',0))
    n=eval_state.get('total_evaluations',30);evaluated=len(eval_state.get('evaluated',[]));verified=len(eval_state.get('verified',[]))
    parent.update(eval_done=evaluated,eval_total=n,eval_progress_percent=round(100*evaluated/n,2),eval_verified=verified,running=eval_state.get('running',{}),progress_updated_at=time.strftime('%Y-%m-%d %H:%M:%S'))
    pending=path.with_name('status.progress.tmp');pending.write_text(json.dumps(parent,indent=2));pending.replace(path)
    return parent.get('completed',False) or parent.get('stage')=='failed'

if __name__=='__main__':
    path=Path(sys.argv[1])
    while True:
        try:
            if mirror(path):break
        except (OSError,ValueError,KeyError) as e:print(type(e).__name__,flush=True)
        time.sleep(5)
