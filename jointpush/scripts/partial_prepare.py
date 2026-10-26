"""Prepare the local RawCuboid Partial Stage2 workspace; never write to jhh."""
import argparse,copy,hashlib,json,os,shutil
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'jointpush/ports/jhh_raw_cuboid'
EXTERNAL=Path('/home/hwanhee/jhh/TokenHSI_stage1_stage2_base')
RUN=EXTERNAL/'output/stage1_v2_fixed_push_360_raw_cuboid/Stage1V2FixedPush360RawCuboid_21-16-00-23'

def verify_source():
 m=json.loads((ROOT/'jointpush/jhh_source_manifest.json').read_text())
 for name,value in m['files'].items():
  if hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()!=value:raise ValueError('Local source changed: '+name)
 for name,target in m.get('data_links',{}).items():
  p=SOURCE/name
  if not p.exists():
   if p.is_symlink():raise ValueError('Broken read-only dataset link: '+name)
   p.parent.mkdir(parents=True,exist_ok=True);p.symlink_to(target,target_is_directory=True)
 return m

def prepare(checkpoint,work):
 verify_source()
 if checkpoint is None:checkpoint=max((RUN/'nn').glob('Stage1V2FixedPush360RawCuboid_0*.pth'),key=lambda p:int(p.stem.rsplit('_',1)[1]))
 checkpoint=Path(checkpoint).resolve(strict=True);work=Path(work).resolve()
 import torch
 weights=torch.load(checkpoint,map_location='cpu',weights_only=False)
 saved=weights['relation_experiment_config'];baseline=copy.deepcopy(saved['env'])
 if 'stage1Push' not in baseline or baseline['stage1Push']['rsi'].get('mode')!='raw' or 'size_range' not in baseline['stage1Push']['box']:raise ValueError('Expected RawCuboid baseline checkpoint')
 import sys
 sys.path[:0]=[str(SOURCE/'tokenhsi'),str(SOURCE)]
 previous=Path.cwd()
 try:
  os.chdir(SOURCE)
  from utils.stage1_push import check_contract
  check_contract(weights,baseline)
 finally:os.chdir(previous)
 dest=ROOT/'jointpush/checkpoints/jhh_raw_cuboid'/checkpoint.name;dest.parent.mkdir(parents=True,exist_ok=True)
 signature=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
 if checkpoint!=dest:
  if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest()!=signature:raise ValueError('Different checkpoint already occupies local destination')
  if not dest.exists():shutil.copy2(checkpoint,dest)
 env=copy.deepcopy(baseline);push=baseline['stage1Push'];box=push['box']
 spec=dict(version=4,physical_boxes=8,size_min=[r[0] for r in box['size_range']],size_max=[r[1] for r in box['size_range']],depth_at_least_height=box.get('depth_at_least_height',False),mass_per_agent=[15.0,25.0],spawn_distance=push['navigation']['spawn_radius_range'],spawn_mode='goal_opposite_180',assignment_mode='random_groups_and_unique_regions',region_assignment='exclusive_reset_random_v1',spawn_min_agent_distance=.9,spawn_radius_semantics='center_with_clearance',spawn_clearance=push['navigation']['clearance'],goal_distance=push['goal_distance_range'],rsi='loco_only',hand_height_fraction=.5,coupled_probability=.8,group_preset='random_scenario',max_group_size=4)
 env.update(numAgents=4,numObjects=8,numEnvs=1024,envSpacing=80,episodeLength=push['navigation']['episode_length'],cameraFollow=False,partitionPush=spec,stage2={'adapter':'raw_cuboid_partial_v1'},pushBox={'friction':box['friction']})
 env['partialReward']=dict(version=1,baseline='stage1_v2_fixed_push_360_raw_cuboid',contact_midpoint_only=push['contact_midpoint_only'],contact_height=push['contact_height'],face_tolerance=.18,gate='all_group_agents_both_hands_on_physical_rear_face',gate_terms=['at_state','at_progress','at_success'],progress_reference='physical_shared_box_center',approach_stand_off=push['navigation']['approach_stand_off'],clearance=push['navigation']['clearance'],teammate_sharing='same_physical_group_90_10')
 env['box']['reset'].pop('platformPhysics',None);env['box']['reset']['randomHeight']=False
 env['video']['enable']=False;env['debug']['reward']=False
 env['relationGraph']['partial_push_adapter']=True;env['relationGraph']['edge_capacity']=8
 env['templateRsi']={name:[1.]+[0.]*(len(env['skill'])-1) for name in env['templateRsi']}
 env['skillInitProb']=[1.]+[0.]*(len(env['skill'])-1)
 env['eval']['skill']=['loco'];env['eval']['skillInitProb']=[1.]
 train=copy.deepcopy(saved['train']);train['params']['algo']['name']='ma';train['params']['network']['name']='amp_multi_agent_shared_support';train['params']['config']['name']='PartialRawCuboid';train['params']['load_checkpoint']=False;train['params'].pop('load_path',None)
 work.mkdir(parents=True,exist_ok=True);(work/'configs').mkdir(exist_ok=True)
 for name in ('tokenhsi','body_models','lpanlib'):
  p=work/name
  if not p.exists():p.symlink_to(SOURCE/name,target_is_directory=True)
  elif p.resolve()!=(SOURCE/name).resolve():raise ValueError('Workspace source mismatch')
 config=dict(env=env,sim=saved['sim'],experiment=dict(name='PartialRawCuboid',transfer=dict(checkpoint=str(dest),checkpoint_sha256=signature,source_epoch=weights['epoch'])))
 (work/'configs/env.yaml').write_text(yaml.safe_dump(config,sort_keys=False));(work/'configs/train.yaml').write_text(yaml.safe_dump(train,sort_keys=False));(work/'configs/baseline.yaml').write_text(yaml.safe_dump(baseline,sort_keys=False))
 report=dict(checkpoint=str(dest),source_checkpoint=str(checkpoint),checkpoint_sha256=signature,source_epoch=weights['epoch'],workspace=str(work),mode='partial_joint_push_stage2',agents=4,environments=1024,objects=8,external_read_only=True)
 (work/'preparation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));return report
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path);p.add_argument('--workspace',type=Path);p.add_argument('--verify',action='store_true');a=p.parse_args()
 if a.verify:verify_source();print('Local source manifest verified; no GPU used')
 elif a.workspace:prepare(a.checkpoint,a.workspace)
 else:p.error('--workspace is required unless --verify is used')
