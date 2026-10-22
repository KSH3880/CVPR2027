"""Compare direct foreign-target visibility with/without extra-box keys."""
import os,sys,json,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=False)
previous=ROOT/'runs/jointpush/attn_view/native_y2m_boxes8_mask_ablation_20261021'
records=[];initial=None
for tag,source,extras,owner_only in [('full_attention','mask_none',[],False),('extra_boxes_hidden','mask_0123',[4,5,6,7],False),('foreign_targets_hidden',None,[],True),('own_box_only',None,[4,5,6,7],True)]:
 folder=out/tag;folder.mkdir()
 if source:
  for name in ('result.json','traces.json','capture.json','states.json','attention_raw.npz','evaluation.log'):
   shutil.copy2(previous/source/name,folder/name)
 else:
  env=dict(os.environ,TOKENHSI_GPU='5',PROBE_AGENTS='4',JOINTPUSH_BOXES='8',JOINTPUSH_ATTENTION_ABLATION='1',JOINTPUSH_OWNER_BOX_ONLY='1',JOINTPUSH_MASK_BOXES=','.join(map(str,extras)),NUM_ENVS='64',PROBE_STEPS='600',SEED='2141',OUTPUT_PATH=str(folder))
  for key in ('SCALING_CPU_ARGS','CUDA_MPS_PIPE_DIRECTORY','CUDA_MPS_LOG_DIRECTORY','TOKENHSI_GUI_GPU_MODE','TOKENHSI_GUI_GPU_INDEX'):env.pop(key,None)
  with (folder/'evaluation.log').open('w') as log:
   r=subprocess.run(['bash',str(ROOT/'jointpush/scripts/scaling_eval.sh')],cwd=str(ROOT),env=env,stdout=log,stderr=subprocess.STDOUT)
  if r.returncode:raise RuntimeError(f'{tag} failed; see {folder}/evaluation.log')
 record=json.loads((folder/'result.json').read_text());capture=record['attention_ablation']
 if initial is None:initial=capture['initial_state_sha256']
 assert capture['initial_state_sha256']==initial and record['model_and_rms_unchanged']
 assert capture['masked_key_probability_max']==0 and capture['layer_reconstruction_max_error']<2e-5
 records.append(dict(tag=tag,reused_from=str(previous/source) if source else None,owner_only=owner_only,extras_hidden=bool(extras),**record))
 (out/'results.json').write_text(json.dumps(records,indent=2))
 print(tag,'falls',record['scene_falls'],'/64','goals',record['agent_ever_goal_success_rate'],flush=True)
(out/'capture_complete.json').write_text(json.dumps(dict(conditions=len(records),initial_state_sha256=initial)))
