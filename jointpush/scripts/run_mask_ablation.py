"""Sequential, explicitly requested extra-box ablation; never manages MPS/jobs."""
import os,sys,json,itertools,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=False)
conditions=[];initial=None
for k in range(5):
 for boxes in itertools.combinations(range(4,8),k):
  tag='mask_none' if not boxes else 'mask_'+''.join(str(i-4) for i in boxes)
  folder=out/tag;folder.mkdir()
  env=dict(os.environ,TOKENHSI_GPU='5',PROBE_AGENTS='4',JOINTPUSH_BOXES='8',JOINTPUSH_ATTENTION_ABLATION='1',JOINTPUSH_MASK_BOXES=','.join(map(str,boxes)),NUM_ENVS='64',PROBE_STEPS='600',SEED='2141',OUTPUT_PATH=str(folder))
  for key in ('SCALING_CPU_ARGS','CUDA_MPS_PIPE_DIRECTORY','CUDA_MPS_LOG_DIRECTORY','TOKENHSI_GUI_GPU_MODE','TOKENHSI_GUI_GPU_INDEX'):env.pop(key,None)
  with (folder/'evaluation.log').open('w') as log:
   result=subprocess.run(['bash',str(ROOT/'jointpush/scripts/scaling_eval.sh')],cwd=str(ROOT),env=env,stdout=log,stderr=subprocess.STDOUT)
  if result.returncode:raise RuntimeError(f'{tag} exited {result.returncode}; see {folder}/evaluation.log')
  record=json.loads((folder/'result.json').read_text());capture=record['attention_ablation']
  if initial is None:initial=capture['initial_state_sha256']
  assert capture['initial_state_sha256']==initial,'Initial physics/observations differ'
  assert capture['masked_key_probability_max']==0 and record['model_and_rms_unchanged']
  conditions.append(dict(tag=tag,masked_count=k,masked_extras=[i-4 for i in boxes],**record))
  (out/'results.json').write_text(json.dumps(conditions,indent=2))
  print(tag,'falls',record['scene_falls'],'/64','goal',record['agent_ever_goal_success_rate'],flush=True)
(out/'capture_complete.json').write_text(json.dumps(dict(conditions=len(conditions),initial_state_sha256=initial)))
