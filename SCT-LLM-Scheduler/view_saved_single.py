"""Launch persistent single-env replay of existing evaluated two-agent scenes."""
from pathlib import Path
from datetime import datetime
import argparse,json,subprocess,os
ROOT=Path(__file__).resolve().parent;RT=ROOT/'teacher_runtime'
def main():
 p=argparse.ArgumentParser();p.add_argument('--setting',choices=['enable_sit','coupled_2'],default='enable_sit');p.add_argument('--seed',type=int,default=1);p.add_argument('--start-env',type=int,default=0);p.add_argument('--episodes',type=int,default=10);p.add_argument('--port',type=int,default=6086);a=p.parse_args()
 source=ROOT/'output_etc/check/eval_protocol_v1_legacy_20261111_165027';sgpu=Path(json.loads((source/'status.json').read_text())['gpu_output']);record=sgpu/f'eval/table2/{a.setting}/general_seed{a.seed}';cmd=json.loads((record/'launch.json').read_text())['command'];cmd[1]='view_saved_single_entry.py';cmd.remove('--headless');cmd[cmd.index('--num_envs')+1]='1'
 tag=datetime.now().strftime('%Y%m%d_%H%M%S');base=ROOT/'output_etc/check'/('viewer_persistent_single_'+tag);out=RT/'output_etc/check'/base.name;base.mkdir();out.mkdir();cmd[cmd.index('--output_path')+1]=str(out)
 env=dict(os.environ,TOKENHSI_GPU='6',TOKENHSI_CONDA_ENV='tokenhsi_koo',PYTHONDONTWRITEBYTECODE='1',VNC_DIR='/home/hwanhee/opt/vnc',NOVNC_DIR='/home/hwanhee/opt/novnc',PORT=str(a.port),GUI_LOG_DIR=str(out/'gui'),SCT_PHYSICAL_GPU='6',SCT_SAVED_SOURCE=str(source),SCT_SAVED_SEED=str(a.seed),SCT_SAVED_SETTING=a.setting,SCT_SAVED_START_ENV=str(a.start_env),SCT_SAVED_EPISODES=str(a.episodes),STAGE1_POLICY_CHECKPOINT=json.loads((ROOT/'stage1_policy.json').read_text())['checkpoint_local'],OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TORCH_EXTENSIONS_DIR=str(ROOT/'output_etc/check/torch_extensions'),CUDA_CACHE_PATH=str(ROOT/'output_etc/check/cuda_cache'))
 print(json.dumps(dict(output=str(base),gpu_output=str(out),api_calls=0,num_envs=1)),flush=True);(base/'launch.json').write_text(json.dumps(dict(command=cmd,gpu_output=str(out),source=str(source),api_calls=0,num_envs=1),indent=2))
 with (out/'run.log').open('w') as log:
  rc=subprocess.call(['bash','-c','source tokenhsi/scripts/multi_agent/runtime_env.sh || exit\nexec bash tokenhsi/scripts/multi_agent/run-gui.sh "$@"','_',*cmd],cwd=RT,env=env,stdout=log,stderr=subprocess.STDOUT)
 (base/'exit.json').write_text(json.dumps(dict(returncode=rc)));print('exit',rc,flush=True)
if __name__=='__main__':main()
