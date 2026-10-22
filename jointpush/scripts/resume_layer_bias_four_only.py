"""Resume the requested layer ablation with FOUR agents only; preserve completed trials."""
import os,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/jointpush/attn_view/native_y2m_layer_bias_ablation_20261021'
assert OUT.exists() and not (OUT/'complete.json').exists()
records=json.loads((OUT/'results.json').read_text());hashes={}
for row in records:
 digest=row['attention_ablation']['initial_state_sha256'];assert hashes.setdefault(row['agents'],digest)==digest
conditions=[('last2',0)]+[('first2',l) for l in (4,6,8,12)]+[(f'layer{i}',4) for i in range(1,5)]+[('last2',4),('all',4)]
def matching_processes(folder):
 found=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:args=(p/'cmdline').read_bytes().split(b'\0')
  except OSError:continue
  if any(v.endswith(b'/scaling_eval.py') for v in args) and str(folder).encode() in args:found.append(int(p.name))
 return found
for mode,strength in conditions:
 agents=4;target='all_foreign' if strength else 'none';tag=f'{mode}_lambda{strength}_a4'
 if any(row['tag']==tag for row in records):continue
 folder=OUT/tag
 if folder.exists():
  started=time.monotonic()
  while matching_processes(folder):
   assert time.monotonic()-started<300,'Existing trial did not finish within 300 seconds'
   time.sleep(1)
  assert (folder/'result.json').exists(),f'Incomplete existing trial: {folder}'
 else:
  folder.mkdir()
  env=dict(os.environ,TOKENHSI_GPU='5',PROBE_AGENTS='4',JOINTPUSH_BOXES='4',NUM_ENVS='64',PROBE_STEPS='600',SEED='2141',OUTPUT_PATH=str(folder),JOINTPUSH_BIAS_CORRECTION='1',JOINTPUSH_BIAS_TARGET=target,JOINTPUSH_BIAS_LAMBDA=str(strength),JOINTPUSH_BIAS_LAYERS=mode,JOINTPUSH_BIAS_SCOPE='all_owner')
  for key in ('JOINTPUSH_ATTENTION_ABLATION','CUDA_MPS_PIPE_DIRECTORY','CUDA_MPS_LOG_DIRECTORY','SCALING_CPU_ARGS','TOKENHSI_GUI_GPU_MODE','TOKENHSI_GUI_GPU_INDEX','JOINTPUSH_CHECKPOINT'):env.pop(key,None)
  (folder/'condition.json').write_text(json.dumps(dict(scope='all_owner',layer_mode=mode,target=target,strength=strength,agents=4,seed=2141,scenes=64,horizon=600),indent=2))
  with (folder/'evaluation.log').open('w') as log:
   r=subprocess.run(['bash',str(ROOT/'jointpush/scripts/scaling_eval.sh'),str(ROOT/'jointpush/checkpoints/PushNativePhaseY2m_00006000.pth')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
  if r.returncode:raise RuntimeError(f'{tag} failed {r.returncode}')
 result=json.loads((folder/'result.json').read_text());meta=result['attention_ablation']
 assert result['model_and_rms_unchanged'] and meta['layer_reconstruction_max_error']<2e-5 and meta['bias_correction_max_error']<2e-6
 assert hashes.setdefault(4,meta['initial_state_sha256'])==meta['initial_state_sha256']
 records.append(dict(tag=tag,layer_mode=mode,folder=str(folder),**result))
 (OUT/'results.json').write_text(json.dumps(records,indent=2))
 print(tag,'falls',result['scene_falls'],'/64','goal',round(result['agent_ever_goal_success_rate']*100,2),flush=True)
(OUT/'complete.json').write_text(json.dumps(dict(completed_records=len(records),four_agent_conditions=sum(r['agents']==4 for r in records),remaining_two_agent_conditions_skipped_by_user=True,initial_hashes=hashes,weights_rms_unchanged=True),indent=2))
