"""Collect one-agent AT/PUSH records with native SCT Stage1 checkpoint."""
from pathlib import Path
from datetime import datetime
import os,json,subprocess
ROOT=Path(__file__).resolve().parent;RT=ROOT/'teacher_runtime'
def main():
 tag='single_stage1_eval_geometry_'+datetime.now().strftime('%Y%m%d_%H%M%S');out=RT/'output_etc/check'/tag;out.mkdir();cp=json.loads((ROOT/'stage1_policy.json').read_text())['checkpoint_local'];status=dict(completed=False,output=str(out),gpu=7,done=[],failures=[]);print(json.dumps(status),flush=True)
 def save():(out/'status.json').write_text(json.dumps(status,indent=2))
 save()
 for kind in ['AT','PUSH']:
  dest=out/kind;dest.mkdir();cmd=['python','record_eval_geometry_entry.py','--task','HumanoidMAPush','--cfg_train','tokenhsi/data/cfg/train/rlg/stage1_push.yaml','--cfg_env','','--motion_file','tokenhsi/data/dataset_stage1_jointpush.yaml','--checkpoint',cp,'--seed','11214','--num_envs','32','--test','--eval','--headless','--no_video','--eval_protocol','stage1_single','--eval_single_task',kind,'--eval_repeats','1','--output_path',str(dest)]
  env=dict(os.environ,TOKENHSI_GPU='7',TOKENHSI_CONDA_ENV='tokenhsi_koo',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TORCH_EXTENSIONS_DIR=str(ROOT/'output_etc/check/torch_extensions'),CUDA_CACHE_PATH=str(ROOT/'output_etc/check/cuda_cache'))
  (dest/'launch.json').write_text(json.dumps(dict(command=cmd,gpu=7),indent=2));status['running']=kind;save()
  with (dest/'run.log').open('w') as log:rc=subprocess.call(['bash','-c','source tokenhsi/scripts/multi_agent/runtime_env.sh || exit\nexec "$@"','_',*cmd],cwd=RT,env=env,stdout=log,stderr=subprocess.STDOUT)
  (dest/'exit.json').write_text(json.dumps(dict(returncode=rc)))
  if rc:status['failures'].append(dict(task=kind,returncode=rc));status['running']=None;save();return
  status['done'].append(kind);save()
 status.update(completed=True,running=None);save()
if __name__=='__main__':main()
