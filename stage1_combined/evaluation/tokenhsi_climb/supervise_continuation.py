"""Run the authorized continuation and leave an exit record for its evaluator."""
from pathlib import Path
import sys,subprocess,os,json,datetime
root=Path(__file__).resolve().parents[2];run=Path(sys.argv[1]).resolve()
env=os.environ.copy();env['OUTPUT_PATH']=str(run/'train')
for name in ['CHECK','MAX_ITERATIONS','RESUME_CHECKPOINT']:env.pop(name,None)
with (run/'train.log').open('w') as log:
    process=subprocess.Popen(['bash','tokenhsi/scripts/multi_agent/tokenhsi_climb_low_lr_train.sh'],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
    (run/'train_process.json').write_text(json.dumps(dict(pid=process.pid,command='tokenhsi_climb_low_lr_train.sh'),indent=2))
    rc=process.wait()
(run/'train_exit.json').write_text(json.dumps(dict(exit_code=rc,finished_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()),indent=2))
sys.exit(rc)
