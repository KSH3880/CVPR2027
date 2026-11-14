"""Reevaluate saved schedules and apply the frozen common protocol v1 metrics."""
from pathlib import Path
from datetime import datetime
import subprocess,json,os,hashlib
ROOT=Path(__file__).resolve().parent
PYTHON='/home/hwanhee/anaconda3/envs/tokenhsi_koo/bin/python'

def main():
 tag=datetime.now().strftime('%Y%m%d_%H%M%S');base=ROOT/'output_etc/check'/('protocol_v1_pipeline_'+tag);base.mkdir();dest=ROOT/'output_etc/check'/('eval_protocol_v1_legacy_'+tag)
 state=dict(stage='eval',completed=False,eval_output=str(dest),api_output=str(ROOT/'output_etc/check/api_no_unnecessary_wait_20261111_151400'),protocol='cvpr2027-evaluation-v1.0.0',freeze='legacy',envs_per_seed=64)
 def save():
  tmp=base/'status.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(base/'status.json')
 save();print(json.dumps(dict(pipeline=str(base),**state)),flush=True)
 env=dict(os.environ,SCT_PROTOCOL_VERSION=state['protocol'],PYTHONDONTWRITEBYTECODE='1')
 with (base/'eval.log').open('w') as log:
  rc=subprocess.call([PYTHON,str(ROOT/'evaluate_saved_api.py'),'--api',state['api_output'],'--output',str(dest),'--freeze','legacy'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
 if rc or not (dest/'status.json').exists():state.update(stage='failed',error=f'eval exit {rc}');save();return
 d=json.loads((dest/'status.json').read_text())
 if len(d['verified'])!=30:state.update(stage='failed',failures=d['failures']);save();return
 state['stage']='protocol_metrics';save()
 with (base/'metrics.log').open('w') as log:
  rc=subprocess.call([PYTHON,str(ROOT/'protocol_v1_rescore.py'),str(dest)],cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT)
 state.update(stage='complete' if rc==0 else 'failed',completed=rc==0)
 save();print(json.dumps(state),flush=True)

if __name__=='__main__':main()
