"""Saved scenes -> new API responses -> legacy-freeze evaluation."""
from pathlib import Path
from datetime import datetime
import subprocess,sys,json
ROOT=Path(__file__).resolve().parent

def main():
    tag=datetime.now().strftime('%Y%m%d_%H%M%S');base=ROOT/'output_etc/check'/('wait_prompt_pipeline_'+tag);base.mkdir()
    api=ROOT/'output_etc/check'/('api_no_unnecessary_wait_'+tag)
    evaluation=ROOT/'output_etc/check'/('eval_no_unnecessary_wait_legacy_'+tag)
    state=dict(stage='api',completed=False,api_output=str(api),eval_output=str(evaluation),freeze='legacy')
    def save():
        temp=base/'status.tmp';temp.write_text(json.dumps(state,indent=2));temp.replace(base/'status.json')
    save();print(json.dumps(dict(pipeline=str(base),**state)),flush=True)
    commands=[('api',[sys.executable,str(ROOT/'replay_saved_api.py'),'--source',str(ROOT/'output_etc/check/full_mixed_table2_env64_20261111_003631'),'--output',str(api),'--workers','16']),
              ('eval',[sys.executable,str(ROOT/'evaluate_saved_api.py'),'--api',str(api),'--output',str(evaluation),'--freeze','legacy'])]
    for stage,cmd in commands:
        state['stage']=stage;save()
        with (base/(stage+'.log')).open('w') as log:
            code=subprocess.call(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if code:
            state.update(stage='failed',error=f'{stage} exited {code}');save();return
    result=json.loads((evaluation/'status.json').read_text())
    state.update(stage=result['stage'],completed=result['completed'],failures=result['failures']);save()

if __name__=='__main__':main()
