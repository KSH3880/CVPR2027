"""View the saved few-shot test in one physical environment; no API calls."""
import json, os, subprocess
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parent; RT=ROOT/'teacher_runtime'
source=RT/'output_etc/check/eval_protocol_v1_legacy_20261111_165027'
ref=source/'reference/table1/mixed_4h/general_seed1'
record=source/'eval/table1/mixed_4h/general_seed1'
api=ROOT/'output_etc/check/fewshot_single_scene_20261112_test01'
assert json.loads((api/'request.json').read_text())['current_scene']==json.loads((ref/'requests/env_0000/request.json').read_text())['current_scene']
tag='viewer_fewshot_single_'+datetime.now().strftime('%Y%m%d_%H%M%S')
base=ROOT/'output_etc/check'/tag;out=RT/'output_etc/check'/tag;base.mkdir();out.mkdir()
cmd=json.loads((record/'launch.json').read_text())['command'];cmd[1]='view_fewshot_single_entry.py';cmd.remove('--headless');cmd[cmd.index('--num_envs')+1]='1';cmd[cmd.index('--output_path')+1]=str(out)
env=dict(os.environ,TOKENHSI_GPU='6',TOKENHSI_CONDA_ENV='tokenhsi_koo',PYTHONDONTWRITEBYTECODE='1',VNC_DIR='/home/hwanhee/opt/vnc',NOVNC_DIR='/home/hwanhee/opt/novnc',PORT='6086',GUI_LOG_DIR=str(out/'gui'),SCT_PHYSICAL_GPU='6',SCT_VIEW_REFERENCE=str(ref),SCT_VIEW_RECORDED=str(record),SCT_VIEW_API=str(api),STAGE1_POLICY_CHECKPOINT=json.loads((ROOT/'stage1_policy.json').read_text())['checkpoint_local'],OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TORCH_EXTENSIONS_DIR=str(ROOT/'output_etc/check/torch_extensions'),CUDA_CACHE_PATH=str(ROOT/'output_etc/check/cuda_cache'),SCT_FREEZE_IMPL='legacy',STAGE2_SCENE_MODE='mixed',REFERENCE_ONLY='0')
env.pop('TABLE2_SETTING',None);env.pop('SCT_GROUP_GAP_VARIANT',None)
info=dict(command=cmd,gpu_output=str(out),reference=str(ref),api=str(api),api_calls=0,num_envs=1,num_agents=4,physical_gpu=6,port=6086)
(base/'launch.json').write_text(json.dumps(info,indent=2));print(json.dumps(info),flush=True)
with (out/'run.log').open('w') as log:
 rc=subprocess.call(['bash','-c','source tokenhsi/scripts/multi_agent/runtime_env.sh || exit\nexec bash tokenhsi/scripts/multi_agent/run-gui.sh "$@"','_',*cmd],cwd=RT,env=env,stdout=log,stderr=subprocess.STDOUT)
(base/'exit.json').write_text(json.dumps(dict(returncode=rc)));print('exit',rc,flush=True)
