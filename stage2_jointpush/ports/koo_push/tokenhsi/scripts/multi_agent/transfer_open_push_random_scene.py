"""Prepare SE(2)-reused RSI bank and transfer only PUSH scene/binding metadata."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import torch
import yaml
from transfer_open_push_push180 import expected_contract as base_contract,file_hash,TRAIN_CONFIG,TARGET_MANIFEST
from utils.open_push_spec import validate_env
from utils.open_push_rsi import bank_contract
from utils.relation_task_spec import checkpoint_metadata

SOURCE_CONFIG=Path('tokenhsi/data/cfg/multi_agent/open_push_push180_random_lanes.yaml')
TARGET_CONFIG=Path('tokenhsi/data/cfg/multi_agent/open_push_push180_random_scene.yaml')


def expected_contract(env,bank_hash):
    result=base_contract(env,TARGET_MANIFEST,bank_hash)
    result['lane_randomization']={'version':1,'swap_probability':.5}
    if 'pushSceneRandomization' in env:result['push_scene']=env['pushSceneRandomization']
    return result


def allowed_configs(source,target):
    validate_env(source);validate_env(target)
    allowed=copy.deepcopy(source)
    allowed['pushSceneRandomization']=target['pushSceneRandomization']
    allowed['openPushRSI']['bank_file']=target['openPushRSI']['bank_file']
    if source.get('pushSceneRandomization') is not None or target!=allowed:
        raise ValueError('Only PUSH scene sampling and bound RSI bank path may change')


def prepare(weights,source,target,source_hash,target_hash,output):
    allowed_configs(source,target)
    required=('model','optimizer','epoch','frame','running_mean_std','reward_mean_std','amp_input_mean_std','relation_experiment_config')
    if weights.get('amp_enabled') is not True or any(k not in weights for k in required):
        raise ValueError('Complete AMP training state required')
    if weights.get('open_push_contract')!=expected_contract(source,source_hash):
        raise ValueError('Source must match PUSH180 random-lane contract')
    if weights.get('relation_metadata')!=checkpoint_metadata(source['relationReward']):
        raise ValueError('Reward metadata differs')
    snapshot=weights['relation_experiment_config']
    if (snapshot['num_envs'],snapshot['num_agents'],snapshot['num_objects'],snapshot['observation_size'])!=(2048,2,4,684):
        raise ValueError('Training dimensions differ')
    if any(snapshot['env'].get(k)!=v for k,v in source.items()):raise ValueError('Saved source config differs')
    if snapshot['train']['params']['network']!=yaml.safe_load(TRAIN_CONFIG.read_text())['params']['network']:
        raise ValueError('Source network differs')
    result=dict(weights);result['open_push_contract']=expected_contract(target,target_hash)
    snapshot=copy.deepcopy(snapshot);snapshot['env'].update(copy.deepcopy(target));snapshot['env']['checkpoint']=str(output)
    snapshot['train']['params']['load_path']=str(output);snapshot['train']['params']['config']['name']='OpenPushPush180RandomScene'
    result['relation_experiment_config']=snapshot
    return result


def prepare_bank(source,target):
    allowed_configs(source,target)
    path=Path(source['openPushRSI']['bank_file']);raw=path.read_bytes();bank=json.loads(raw)
    if bank.get('screen')!='PD_12_steps' or bank['contract']!=bank_contract(dict(source,motion_file=str(TARGET_MANIFEST))):
        raise ValueError('Source PD bank differs')
    new=copy.deepcopy(bank);new['contract']=bank_contract(dict(target,motion_file=str(TARGET_MANIFEST)))
    new['se2_reuse']={'source_sha256':hashlib.sha256(raw).hexdigest(),'method':'Rigid XY translation and yaw preserve single-owner RSI geometry; full random scenes separately audited'}
    dest=Path(target['openPushRSI']['bank_file'])
    if dest.exists():
        if json.loads(dest.read_text())!=new:raise ValueError('Existing target bank differs; preserve it')
    else:dest.write_text(json.dumps(new,indent=2)+'\n')
    return hashlib.sha256(raw).hexdigest(),file_hash(dest)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint',type=Path,nargs='?');p.add_argument('output',type=Path,nargs='?');p.add_argument('--bank-only',action='store_true')
    args=p.parse_args()
    if not args.bank_only and (args.checkpoint is None or args.output is None):p.error('checkpoint and output required')
    if not args.bank_only and (args.output.exists() or args.output.resolve()==args.checkpoint.resolve()):raise ValueError('New output required')
    source=yaml.safe_load(SOURCE_CONFIG.read_text())['env'];target=yaml.safe_load(TARGET_CONFIG.read_text())['env']
    a,b=prepare_bank(source,target)
    if args.bank_only:
        print(json.dumps({'bank':target['openPushRSI']['bank_file'],'source_hash':a,'target_hash':b}));return
    weights=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    result=prepare(weights,source,target,a,b,args.output)
    report=dict(source=str(args.checkpoint.resolve()),source_sha256=file_hash(args.checkpoint),epoch=int(weights['epoch']),frame=int(weights['frame']),source_bank_sha256=a,target_bank_sha256=b,change='PUSH XY/yaw/direction/start/binding randomized; same cube/reward/AMP',output=str(args.output.resolve()))
    result['open_push_scene_transfer']=report;args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as f:torch.save(result,f)
    args.output.with_suffix('.transfer.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
