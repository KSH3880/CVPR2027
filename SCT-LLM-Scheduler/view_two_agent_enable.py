"""One environment, AT->SIT, one current API call, GPU6 viewer, normal Q exit."""
import os,json,subprocess
from pathlib import Path
from datetime import datetime
from run_full_suite import gpu_job
from run_eval16_batch import ROOT,RT
from sct_scheduler.config import read_settings,chat_endpoint,DEFAULT_BASE_URL,DEFAULT_REASONING_EFFORT
from sct_scheduler.client import OneShotScheduler,chat_transport,loads
from sct_scheduler.executable import validate_executable

def save(p,d):p.write_text(json.dumps(d,indent=2))
def main():
 tag=datetime.now().strftime('%Y%m%d_%H%M%S');base=ROOT/'output_etc/check'/('viewer_enable_2h_'+tag);gpu=RT/'output_etc/check'/base.name;base.mkdir();gpu.mkdir()
 state=dict(stage='capture',completed=False,physical_gpu=6,num_envs=1,num_agents=2,port=6086,gpu_output=str(gpu));save(base/'status.json',state);print(json.dumps(dict(output=str(base))),flush=True)
 os.environ.update(SCT_FREEZE_IMPL='legacy',SCT_PROTOCOL_VERSION='cvpr2027-evaluation-v1.0.0');os.environ.pop('SCT_GROUP_GAP_VARIANT',None)
 job=dict(table='table2',name='enable_sit',people=2,seed=1,gpu=6,key='table2/enable_sit/general_seed1')
 ref=gpu_job(base,gpu,job,1,capture=True);req=loads((ref/'requests/env_0000/request.json').read_text());state['stage']='api';save(base/'status.json',state)
 cfg=read_settings();transport=chat_transport(chat_endpoint(cfg.get('SCT_LLM_BASE_URL') or DEFAULT_BASE_URL),cfg['SCT_LLM_MODEL'],cfg['SCT_LLM_API_KEY'],timeout_s=180,json_mode=True,reasoning_effort=cfg.get('SCT_LLM_REASONING_EFFORT') or DEFAULT_REASONING_EFFORT)
 dest=base/'api'/job['key']/'env_0000'
 try:OneShotScheduler(transport).schedule(req,dest)
 except Exception as e:state['api_validation_error']=str(e)
 answer=loads((dest/'response.raw.txt').read_text());validate_executable(req,answer)
 out=gpu/'viewer';out.mkdir();cmd=json.loads((ref/'launch.json').read_text())['command'];cmd[1]='view_batch_entry.py';cmd.remove('--headless');cmd[cmd.index('--output_path')+1]=str(out)
 cp=json.loads((ROOT/'stage1_policy.json').read_text())['checkpoint_local']
 env=dict(os.environ,TOKENHSI_GPU='6',SCT_PHYSICAL_GPU='6',TOKENHSI_CONDA_ENV='tokenhsi_koo',PYTHONDONTWRITEBYTECODE='1',TORCH_EXTENSIONS_DIR=str(ROOT/'output_etc/check/torch_extensions'),CUDA_CACHE_PATH=str(ROOT/'output_etc/check/cuda_cache'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',STAGE2_SCENE_MODE='mixed',PYTHONHASHSEED='1',REFERENCE_ONLY='0',TABLE2_SETTING='enable_sit',STAGE2_REFERENCE_ROOT=str(ref.parent),STAGE1_POLICY_CHECKPOINT=cp,SCT_SCHEDULE_ROOT=str(base/'api/table2/enable_sit'),SCT_EXECUTION_ONLY='1',SCT_VIEW_ENV='0',SCT_PREFLIGHT='0',VNC_DIR='/home/hwanhee/opt/vnc',NOVNC_DIR='/home/hwanhee/opt/novnc',PORT='6086',GUI_LOG_DIR=str(out/'gui'))
 (out/'gui').mkdir();save(out/'launch.json',dict(command=cmd,physical_gpu=6,freeze='legacy',schedule=answer['schedule']))
 state.update(stage='viewer_starting',schedule=answer['schedule']);save(base/'status.json',state)
 with (out/'run.log').open('w') as log:
  p=subprocess.Popen(['bash','-c','source tokenhsi/scripts/multi_agent/runtime_env.sh || exit\nexec bash tokenhsi/scripts/multi_agent/run-gui.sh "$@"','_',*cmd],cwd=RT,env=env,stdout=log,stderr=subprocess.STDOUT)
  state['supervisor_pid']=p.pid;save(base/'status.json',state)
  import time
  while p.poll() is None:
   if (out/'viewer_ready.json').exists() and state['stage']!='viewer':
    state['stage']='viewer';save(base/'status.json',state);print('[READY] http://localhost:6086/vnc.html?autoconnect=1&resize=remote',flush=True)
   time.sleep(1)
 state.update(stage='closed' if p.returncode==0 else 'failed',completed=True,returncode=p.returncode);save(base/'status.json',state)
 print(json.dumps(state),flush=True)
if __name__=='__main__':main()
