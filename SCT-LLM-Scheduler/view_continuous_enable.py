"""Sequential one-env two-agent views. N skips, Q exits, completion advances."""
import argparse,json,os,subprocess,time,sys
from pathlib import Path
from datetime import datetime
from run_full_suite import gpu_job
from run_eval16_batch import ROOT,RT
from sct_scheduler.config import read_settings,chat_endpoint,DEFAULT_BASE_URL,DEFAULT_REASONING_EFFORT
from sct_scheduler.client import OneShotScheduler,chat_transport,loads
from sct_scheduler.executable import validate_executable
SETTINGS=['enable_sit','enable_climb','enable_ontop','coupled_2']

def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,default=2);parser.add_argument('--episodes',type=int,default=10);parser.add_argument('--port',type=int,default=6086);parser.add_argument('--settings',nargs='+',choices=SETTINGS,default=['enable_sit','enable_climb','enable_ontop']);parser.add_argument('--output',type=Path);parser.add_argument('--inside-gui',action='store_true',help=argparse.SUPPRESS);args=parser.parse_args()
 assert args.episodes>0 and args.seed>0
 base=args.output.resolve() if args.output else ROOT/'output_etc/check'/('viewer_continuous_2h_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
 gpu=RT/'output_etc/check'/base.name
 if not args.inside_gui:
  assert not base.exists();base.mkdir();gpu.mkdir()
  env=dict(os.environ,TOKENHSI_GPU='6',TOKENHSI_CONDA_ENV='tokenhsi_koo',PYTHONDONTWRITEBYTECODE='1',VNC_DIR='/home/hwanhee/opt/vnc',NOVNC_DIR='/home/hwanhee/opt/novnc',PORT=str(args.port),GUI_LOG_DIR=str(gpu/'gui'))
  cmd=['python',str(Path(__file__).resolve()),'--inside-gui','--output',str(base),'--seed',str(args.seed),'--episodes',str(args.episodes),'--port',str(args.port),'--settings',*args.settings]
  print(json.dumps(dict(output=str(base),port=args.port)),flush=True)
  with (base/'supervisor.log').open('w') as log:
   code=subprocess.call(['bash','-c','source tokenhsi/scripts/multi_agent/runtime_env.sh || exit\nexec bash tokenhsi/scripts/multi_agent/run-gui.sh "$@"','_',*cmd],cwd=RT,env=env,stdout=log,stderr=subprocess.STDOUT)
  if code:save(base/'launcher_failure.json',dict(returncode=code))
  return
 assert (ROOT/'output_etc/check').resolve() in base.parents
 os.environ.update(SCT_FREEZE_IMPL='legacy',SCT_PROTOCOL_VERSION='cvpr2027-evaluation-v1.0.0');os.environ.pop('SCT_GROUP_GAP_VARIANT',None)
 state=dict(stage='starting',completed=False,num_envs=1,num_agents=2,physical_gpu=6,port=args.port,gpu_output=str(gpu),max_episodes=args.episodes,history=[],controls={'Space':'play/pause','N':'next episode','Q':'quit all'})
 save(base/'status.json',state)
 cfg=read_settings();transport=chat_transport(chat_endpoint(cfg.get('SCT_LLM_BASE_URL') or DEFAULT_BASE_URL),cfg['SCT_LLM_MODEL'],cfg['SCT_LLM_API_KEY'],timeout_s=180,json_mode=True,reasoning_effort=cfg.get('SCT_LLM_REASONING_EFFORT') or DEFAULT_REASONING_EFFORT)
 cp=json.loads((ROOT/'stage1_policy.json').read_text())['checkpoint_local']
 try:
  for episode in range(args.episodes):
   seed=args.seed+episode;setting=args.settings[episode%len(args.settings)];state.update(episode=episode+1,seed=seed,setting=setting,stage='capture');save(base/'status.json',state)
   job=dict(table='table2',name=setting,people=2,seed=seed,gpu=6,key=f'table2/{setting}/general_seed{seed}')
   ref=gpu_job(base,gpu,job,1,capture=True);req=loads((ref/'requests/env_0000/request.json').read_text());dest=base/'api'/job['key']/'env_0000'
   state['stage']='api';save(base/'status.json',state)
   try:OneShotScheduler(transport).schedule(req,dest)
   except Exception as e:state['last_api_validation_error']=str(e)
   answer=loads((dest/'response.raw.txt').read_text());validate_executable(req,answer)
   out=gpu/'viewer'/f'episode_{episode+1:04}';out.mkdir(parents=True)
   cmd=json.loads((ref/'launch.json').read_text())['command'];cmd[1]='view_batch_entry.py';cmd.remove('--headless');cmd[cmd.index('--output_path')+1]=str(out)
   env=dict(os.environ,SCT_PHYSICAL_GPU='6',SCT_VIEW_CONTINUOUS='1',SCT_VIEW_START_PAUSED='1' if episode==0 else '0',SCT_VIEW_FINISH_HOLD_S='2',SCT_VIEW_ENV='0',TABLE2_SETTING=setting,REFERENCE_ONLY='0',STAGE2_SCENE_MODE='mixed',PYTHONHASHSEED=str(seed),STAGE2_REFERENCE_ROOT=str(ref.parent),STAGE1_POLICY_CHECKPOINT=cp,SCT_SCHEDULE_ROOT=str(base/'api/table2'/setting),SCT_EXECUTION_ONLY='1',SCT_PREFLIGHT='0',TORCH_EXTENSIONS_DIR=str(ROOT/'output_etc/check/torch_extensions'),CUDA_CACHE_PATH=str(ROOT/'output_etc/check/cuda_cache'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
   state.update(stage='viewer_starting',schedule=answer['schedule'],viewer_output=str(out));save(base/'status.json',state)
   save(out/'launch.json',dict(command=cmd,physical_gpu=6,episode=episode+1,seed=seed,setting=setting))
   with (out/'run.log').open('w') as log:
    proc=subprocess.Popen(cmd,cwd=RT,env=env,stdout=log,stderr=subprocess.STDOUT);state['viewer_pid']=proc.pid;save(base/'status.json',state)
    while proc.poll() is None:
     if (out/'viewer_ready.json').exists() and state['stage']!='viewer':state['stage']='viewer';save(base/'status.json',state);print('READY episode',episode+1,setting,'seed',seed,flush=True)
     time.sleep(.5)
   if proc.returncode:raise RuntimeError(f'Viewer exited {proc.returncode}; {out}/run.log')
   result=json.loads((out/'viewer_state.json').read_text());reason=result['exit_reason'];state['history'].append(dict(episode=episode+1,seed=seed,setting=setting,reason=reason,frames=result['frame']));save(base/'status.json',state)
   if reason=='quit':break
  state.update(stage='closed',completed=True);save(base/'status.json',state)
 except Exception as e:
  state.update(stage='failed',completed=True,error=str(e));save(base/'status.json',state);raise

if __name__=='__main__':main()
