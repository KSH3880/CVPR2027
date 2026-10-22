"""Sequential owner-only exceptions: last2 versus all actor layers, GPU5."""
import os,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/jointpush/attn_view/native_y2m_allowner_bias_20261021'
OUT.mkdir(exist_ok=False)
records=[];hashes={}
conditions=[('last2',0)]+[(mode,l) for mode in ('last2','all') for l in (2,4,6)]
for mode,strength in conditions:
 for agents in (2,4):
  target='all_foreign' if strength else 'none'
  tag=f'{mode}_lambda{strength}_a{agents}';folder=OUT/tag;folder.mkdir()
  env=dict(os.environ,TOKENHSI_GPU='5',PROBE_AGENTS=str(agents),JOINTPUSH_BOXES=str(agents),NUM_ENVS='64',PROBE_STEPS='600',SEED='2141',OUTPUT_PATH=str(folder),JOINTPUSH_BIAS_CORRECTION='1',JOINTPUSH_BIAS_TARGET=target,JOINTPUSH_BIAS_LAMBDA=str(strength),JOINTPUSH_BIAS_LAYERS=mode,JOINTPUSH_BIAS_SCOPE='all_owner')
  for key in ('JOINTPUSH_ATTENTION_ABLATION','CUDA_MPS_PIPE_DIRECTORY','CUDA_MPS_LOG_DIRECTORY','SCALING_CPU_ARGS','TOKENHSI_GUI_GPU_MODE','TOKENHSI_GUI_GPU_INDEX','JOINTPUSH_CHECKPOINT'):env.pop(key,None)
  (folder/'condition.json').write_text(json.dumps(dict(scope='all_owner',layer_mode=mode,target=target,strength=strength,agents=agents,seed=2141,scenes=64,horizon=600),indent=2))
  with (folder/'evaluation.log').open('w') as log:
   r=subprocess.run(['bash',str(ROOT/'jointpush/scripts/scaling_eval.sh'),str(ROOT/'jointpush/checkpoints/PushNativePhaseY2m_00006000.pth')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
  if r.returncode:raise RuntimeError(f'{tag} failed {r.returncode}; see log')
  result=json.loads((folder/'result.json').read_text());meta=result['attention_ablation']
  assert result['model_and_rms_unchanged'] and meta['layer_reconstruction_max_error']<2e-5 and meta['bias_correction_max_error']<2e-6
  assert hashes.setdefault(agents,meta['initial_state_sha256'])==meta['initial_state_sha256']
  records.append(dict(tag=tag,layer_mode=mode,folder=str(folder),**result))
  (OUT/'results.json').write_text(json.dumps(records,indent=2))
  print(tag,'falls',result['scene_falls'],'/64','goal',round(100*result['agent_ever_goal_success_rate'],2),flush=True)
(OUT/'complete.json').write_text(json.dumps(dict(conditions=len(records),initial_hashes=hashes,weights_rms_unchanged=True),indent=2))
