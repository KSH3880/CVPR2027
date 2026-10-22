"""Sequential two-checkpoint,2/4-agent standing audit on explicitly requested GPU5."""
import os,sys,json,subprocess,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];WORK=ROOT/'runs/jointpush/five_task_500_fall_20261021/workspace'
results=[]
for model in ('depth180','depth200_friction04'):
 for agents in (2,4):
  tag=f'{model}_a{agents}_seed2141'
  out=WORK/'output_etc/check'/tag
  assert not out.exists(),f'Refuse overwrite: {out}'
  out.mkdir(parents=True)
  env=dict(os.environ,TOKENHSI_GPU='5',MODEL=model,PROBE_AGENTS=str(agents),NUM_ENVS='64',PROBE_STEPS='600',SEED='2141',OUTPUT_PATH=str(out))
  for key in ('CUDA_MPS_PIPE_DIRECTORY','CUDA_MPS_LOG_DIRECTORY','TOKENHSI_GUI_GPU_MODE','TOKENHSI_GUI_GPU_INDEX'):env.pop(key,None)
  with (out/'evaluation.log').open('w') as log:
   r=subprocess.run(['bash',str(ROOT/'jointpush/scripts/five_task_fall.sh')],cwd=str(ROOT),env=env,stdout=log,stderr=subprocess.STDOUT)
  if r.returncode:raise RuntimeError(f'{tag}: exit {r.returncode}; see evaluation.log')
  record=json.loads((out/'result.json').read_text());results.append(dict(model=model,folder=str(out),**record))
  assert record['finite'] and record['model_rms_unchanged']
  (WORK.parent/'results.json').write_text(json.dumps(results,indent=2))
  print(tag,'falls',record['scene_falls'],'/64',flush=True)
source=Path('/home/hwanhee/koo_cvpr/stage1_fix/ports/ksh_stage1_stage2_base/push')
manifest=json.loads((WORK/'source_manifest.json').read_text())
assert all(hashlib.sha256((source/key).read_bytes()).hexdigest()==value for key,value in manifest.items()),'Source files changed during evaluation'
prep=json.loads((WORK/'preparation.json').read_text())
for model,meta in prep.items():assert hashlib.sha256(Path(meta['checkpoint']).read_bytes()).hexdigest()==meta['checkpoint_sha256']
(WORK.parent/'complete.json').write_text(json.dumps(dict(conditions=len(results),source_files_verified=len(manifest),checkpoints_unchanged=True,all_weights_rms_unchanged=True),indent=2))
