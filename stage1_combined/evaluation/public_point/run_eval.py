from pathlib import Path
import argparse,datetime,json,os,subprocess,hashlib
import numpy as np
import torch,yaml
from evaluation.finetuned.run_eval import catalog,ROOT
PUBLIC=Path('/home/injesus1010/.cache/huggingface/hub/models--lianganimation--TokenHSI/blobs/60142bc2a0b44614e78e0ae28537a3ec4d4059cf171278c255793d6083d8db83')
ADAPTER=ROOT/'output/box_chair_table_mixed_full_ft/local_78k_rsi_20261023_144809/BoxChairTableMixedFullFT_23-14-48-11/nn/BoxChairTableMixedFullFT_00000700.pth'
def main():
 p=argparse.ArgumentParser();p.add_argument('--categories',nargs='+',choices=['Chair','Table','Box','Cabinet','Mixed'],default=['Chair','Table']);p.add_argument('--actions',nargs='+',choices=['SIT','CLIMB'],default=['SIT','CLIMB']);p.add_argument('--scene30d',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
 scene_checkpoint=ROOT/'checkpoints/Stage1V2Fixed_00078000.pth' if a.scene30d else ADAPTER
 weights=torch.load(scene_checkpoint,map_location='cpu',weights_only=False);results={}
 if (out/'summary.json').exists():results=json.loads((out/'summary.json').read_text())
 manifest=dict(policy='public_tokenhsi_stage1_40000',checkpoint=str(PUBLIC),sha256=hashlib.sha256(PUBLIC.read_bytes()).hexdigest(),num_envs=512,seeds=[0,1,2],steps=900,protocol='our common point/region evaluation;not original paper benchmark',adapter_checkpoint=str(scene_checkpoint),adapter_note='scene model unused for actions; public354D network strictly loaded',scene30d=a.scene30d,categories=a.categories,actions=a.actions)
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
 for cat in a.categories:
  for action in a.actions:
   records=[]
   for seed in [0,1,2]:
    case=out/cat/action/str(seed);case.mkdir(parents=True,exist_ok=True)
    if not (case/'results.json').exists():
     cfg={k:weights['relation_experiment_config'][k] for k in ['env','sim']};cfg=yaml.safe_load(yaml.safe_dump(cfg,sort_keys=False))
     if a.scene30d:cfg=yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/tokenhsi_climb_30d_full_ft.yaml').read_text())
     e=cfg['env'];e['chairFT']['point_evaluation']=dict(category=cat,action=action,catalog=catalog(cat));e['chairFT']['public_checkpoint']=str(PUBLIC);e.update(numAgents=1,numObjects=1,episodeLength=900,debug=dict(reward=False));(case/'env.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
     cmd=['bash','-c','. tokenhsi/scripts/multi_agent/runtime_env.sh; exec python "$@"','publiceval','evaluation/public_point/launch_case.py','--task','EvaluationTask','--cfg_train','tokenhsi/data/cfg/train/rlg/chair_table_eval_full_ft.yaml','--cfg_env',str(case/'env.yaml'),'--checkpoint',str(scene_checkpoint),'--num_agents','1','--num_objects','1','--num_envs','512','--task_graph',action.lower(),'--eval_skills','loco','--eval_skill_probs','1','--eval_repeats','1','--episode_length','900','--output_path',str(case),'--seed',str(seed),'--headless','--no_video','--test','--eval','--motion_file','tokenhsi/data/dataset_loco_sit_carry_climb.yaml']
     if a.scene30d:cmd[cmd.index('tokenhsi/data/cfg/train/rlg/chair_table_eval_full_ft.yaml')]='tokenhsi/data/cfg/train/rlg/tokenhsi_climb_30d_full_ft.yaml'
     env=os.environ.copy();env['PUBLIC_SCENE30']='1' if a.scene30d else '0';env.update(TOKENHSI_GPU='0',TOKENHSI_CONDA_ENV='tokenhsi118',PYTORCH_JIT='0',PYTHONPATH=str(ROOT)+':'+str(ROOT/'tokenhsi'),PYTHONUNBUFFERED='1')
     print('START',cat,action,seed,flush=True)
     with (case/'run.log').open('w') as log:rc=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
     if rc:raise RuntimeError(f'case failed rc={rc}: {case}')
    d=json.loads((case/'results.json').read_text());assert d['policy']['epoch']==40000;records.append(d);print('DONE',cat,action,seed,d['point_ever'],flush=True)
   results[cat+'_'+action]=dict(records=records,point_mean=100*np.mean([x['point_ever'] for x in records]),point_std=100*np.std([x['point_ever'] for x in records],ddof=1),region_mean=100*np.mean([x['region_ever'] for x in records]),region_std=100*np.std([x['region_ever'] for x in records],ddof=1));(out/'summary.json').write_text(json.dumps(results,indent=2))
 (out/'completed.json').write_text(json.dumps(dict(rc=0)));print('PUBLIC_EVAL_DONE',out,flush=True)
if __name__=='__main__':main()
