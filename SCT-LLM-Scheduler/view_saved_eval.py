"""Replay existing evaluated schedules. Never imports/calls API transport."""
import argparse,json,os,subprocess,time,sys
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parent;RT=ROOT/'teacher_runtime'
DEFAULT=ROOT/'output_etc/check/eval_protocol_v1_legacy_20261111_165027'

def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=DEFAULT);p.add_argument('--setting',choices=['enable_sit','enable_climb','enable_ontop','coupled_2'],default='enable_sit');p.add_argument('--seed',type=int,choices=[1,6,7],default=1);p.add_argument('--start-env',type=int,default=0);p.add_argument('--episodes',type=int,default=10);p.add_argument('--port',type=int,default=6086);p.add_argument('--output',type=Path);p.add_argument('--inside-gui',action='store_true',help=argparse.SUPPRESS);a=p.parse_args();assert 0<=a.start_env<64 and a.episodes>0
 source=a.source.resolve();status=json.loads((source/'status.json').read_text());assert status['completed'];sgpu=Path(status['gpu_output']);ref=sgpu/'reference/table2'/a.setting;apis=source/'api/table2'/a.setting;recorded=sgpu/f'eval/table2/{a.setting}/general_seed{a.seed}'
 assert (recorded/'suite_verification.json').exists()
 base=a.output.resolve() if a.output else ROOT/'output_etc/check'/('viewer_saved_eval_2h_'+datetime.now().strftime('%Y%m%d_%H%M%S'));gpu=RT/'output_etc/check'/base.name
 if not a.inside_gui:
  assert not base.exists();base.mkdir();gpu.mkdir();env=dict(os.environ,TOKENHSI_GPU='6',TOKENHSI_CONDA_ENV='tokenhsi_koo',PYTHONDONTWRITEBYTECODE='1',VNC_DIR='/home/hwanhee/opt/vnc',NOVNC_DIR='/home/hwanhee/opt/novnc',PORT=str(a.port),GUI_LOG_DIR=str(gpu/'gui'))
  cmd=['python',str(Path(__file__).resolve()),'--inside-gui','--output',str(base),'--source',str(source),'--setting',a.setting,'--seed',str(a.seed),'--start-env',str(a.start_env),'--episodes',str(a.episodes),'--port',str(a.port)]
  print(json.dumps(dict(output=str(base),api_calls=0)),flush=True)
  with (base/'supervisor.log').open('w') as log:subprocess.call(['bash','-c','source tokenhsi/scripts/multi_agent/runtime_env.sh || exit\nexec bash tokenhsi/scripts/multi_agent/run-gui.sh "$@"','_',*cmd],cwd=RT,env=env,stdout=log,stderr=subprocess.STDOUT)
  return
 state=dict(stage='starting',completed=False,source=str(source),setting=a.setting,seed=a.seed,physical_gpu=6,internal_envs=64,visible_envs=1,num_agents=2,api_calls=0,port=a.port,gpu_output=str(gpu),history=[],controls={'Space':'play/pause','N':'next saved scene','Q':'quit all'})
 save(base/'status.json',state)
 cp=json.loads((ROOT/'stage1_policy.json').read_text())['checkpoint_local']
 try:
  for episode in range(min(a.episodes,64-a.start_env)):
   index=a.start_env+episode;original=apis/f'general_seed{a.seed}'/f'env_{index:04}';request=json.loads((original/'request.json').read_text());answer=json.loads((original/'response.raw.txt').read_text());out=gpu/f'episode_{episode+1:04}';out.mkdir()
   cmd=json.loads((recorded/'launch.json').read_text())['command'];cmd[1]='view_batch_entry.py';cmd.remove('--headless');cmd[cmd.index('--output_path')+1]=str(out)
   env=dict(os.environ,SCT_FREEZE_IMPL='legacy',SCT_PROTOCOL_VERSION='cvpr2027-evaluation-v1.0.0',SCT_PHYSICAL_GPU='6',SCT_VIEW_CONTINUOUS='1',SCT_VIEW_SAVED_BATCH='1',SCT_VIEW_START_PAUSED='1' if episode==0 else '0',SCT_VIEW_FINISH_HOLD_S='2',SCT_VIEW_ENV=str(index),TABLE2_SETTING=a.setting,REFERENCE_ONLY='0',STAGE2_SCENE_MODE='mixed',PYTHONHASHSEED=str(a.seed),STAGE2_REFERENCE_ROOT=str(ref),STAGE1_POLICY_CHECKPOINT=cp,SCT_SCHEDULE_ROOT=str(apis),SCT_EXECUTION_ONLY='1',SCT_PREFLIGHT='0',TORCH_EXTENSIONS_DIR=str(ROOT/'output_etc/check/torch_extensions'),CUDA_CACHE_PATH=str(ROOT/'output_etc/check/cuda_cache'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
   env.pop('SCT_GROUP_GAP_VARIANT',None)
   state.update(stage='viewer_starting',episode=episode+1,source_environment=index,viewer_output=str(out),schedule=answer['schedule']);save(base/'status.json',state)
   save(out/'replay_source.json',dict(reference=str(ref),request=str(original/'request.json'),response=str(original/'response.raw.txt'),recorded_evaluation=str(recorded),selected_environment=index,api_calls=0))
   with (out/'run.log').open('w') as log:
    proc=subprocess.Popen(cmd,cwd=RT,env=env,stdout=log,stderr=subprocess.STDOUT);state['viewer_pid']=proc.pid;save(base/'status.json',state)
    while proc.poll() is None:
     if (out/'viewer_ready.json').exists() and state['stage']!='viewer':state['stage']='viewer';save(base/'status.json',state);print('READY saved env',index,flush=True)
     time.sleep(.5)
   if proc.returncode:raise RuntimeError(f'viewer exit {proc.returncode}; {out}/run.log')
   done=json.loads((out/'viewer_state.json').read_text());state['history'].append(dict(environment=index,reason=done['exit_reason'],frames=done['frame']));save(base/'status.json',state)
   if done['exit_reason']=='quit':break
  state.update(stage='closed',completed=True);save(base/'status.json',state)
 except Exception as e:state.update(stage='failed',completed=True,error=str(e));save(base/'status.json',state);raise

if __name__=='__main__':main()
