"""Transfer PUSH180 full training state, enabling only 50/50 lane assignment."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import torch
import yaml
from transfer_open_push_push180 import expected_contract, file_hash, TRAIN_CONFIG, TARGET_MANIFEST
from utils.open_push_spec import validate_env
from utils.open_push_rsi import bank_contract
from utils.relation_task_spec import checkpoint_metadata

SOURCE_CONFIG=Path('tokenhsi/data/cfg/multi_agent/open_push_late_rsi_hinge_friction_push2m_push180.yaml')
TARGET_CONFIG=Path('tokenhsi/data/cfg/multi_agent/open_push_push180_random_lanes.yaml')


def prepare(weights, source, target, bank_hash, output):
    validate_env(source);validate_env(target)
    allowed=copy.deepcopy(source);allowed['openPushRandomizeLanes']=True
    if source.get('openPushRandomizeLanes',False) or target!=allowed:
        raise ValueError('Only openPushRandomizeLanes=true may change')
    required=('model','optimizer','running_mean_std','reward_mean_std','amp_input_mean_std','epoch','frame','relation_experiment_config')
    if weights.get('amp_enabled') is not True or any(k not in weights for k in required):
        raise ValueError('Complete AMP training state required')
    if weights.get('open_push_contract')!=expected_contract(source,TARGET_MANIFEST,bank_hash):
        raise ValueError('Source must match the fixed-lane PUSH180 experiment')
    if weights.get('relation_metadata')!=checkpoint_metadata(source['relationReward']):
        raise ValueError('Reward metadata differs')
    snapshot=weights['relation_experiment_config']
    if (snapshot['num_envs'],snapshot['num_agents'],snapshot['num_objects'],snapshot['observation_size'])!=(2048,2,4,684):
        raise ValueError('Training dimensions differ')
    if any(snapshot['env'].get(k)!=v for k,v in source.items()):
        raise ValueError('Saved source config differs')
    if snapshot['train']['params']['network']!=yaml.safe_load(TRAIN_CONFIG.read_text())['params']['network']:
        raise ValueError('Source network differs')
    result=dict(weights)
    result['open_push_contract']=copy.deepcopy(weights['open_push_contract'])
    result['open_push_contract']['lane_randomization']={'version':1,'swap_probability':.5}
    snapshot=copy.deepcopy(snapshot);snapshot['env'].update(copy.deepcopy(target))
    snapshot['env']['checkpoint']=str(output)
    snapshot['train']['params']['load_path']=str(output)
    snapshot['train']['params']['config']['name']='OpenPushPush180RandomLanes'
    result['relation_experiment_config']=snapshot
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('checkpoint',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args()
    if args.output.exists() or args.output.resolve()==args.checkpoint.resolve():
        raise ValueError('Output must be a new file; preserve source')
    source=yaml.safe_load(SOURCE_CONFIG.read_text())['env'];target=yaml.safe_load(TARGET_CONFIG.read_text())['env']
    raw=Path(source['openPushRSI']['bank_file']).read_bytes();bank=json.loads(raw)
    for env in (source,target):
        if bank.get('screen')!='PD_12_steps' or bank['contract']!=bank_contract(dict(env,motion_file=str(TARGET_MANIFEST))):
            raise ValueError('Unchanged screened RSI bank required')
    weights=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    result=prepare(weights,source,target,hashlib.sha256(raw).hexdigest(),args.output)
    report=dict(source_checkpoint=str(args.checkpoint.resolve()),source_sha256=file_hash(args.checkpoint),epoch=int(weights['epoch']),frame=int(weights['frame']),change='Only owner-to-physical-lane assignment randomized 50/50',target_config=str(TARGET_CONFIG),output=str(args.output.resolve()))
    result['open_push_lane_transfer']=report
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as f:torch.save(result,f)
    args.output.with_suffix('.transfer.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
