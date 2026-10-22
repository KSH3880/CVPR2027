"""Explicit CPU resume preparation: only native PUSH box Y and clearance change."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import torch
import yaml
from utils.native_push_rsi import native_contract
from utils.open_push_spec import validate_env, checkpoint_contracts_match
from utils.relation_task_spec import checkpoint_metadata, check_checkpoint_metadata

BASE='push_native_phase_xy'
TARGET='push_native_phase_y2m'


def transfer(source,output,screen_path='output/push_native_phase_y2m_check/screen.json'):
    source,output=Path(source),Path(output)
    if output.exists() or output.with_suffix('.json').exists():
        raise ValueError('Preserve the existing transfer output')
    old=yaml.safe_load(Path(f'tokenhsi/data/cfg/multi_agent/{BASE}.yaml').read_text())['env']
    new=yaml.safe_load(Path(f'tokenhsi/data/cfg/multi_agent/{TARGET}.yaml').read_text())['env']
    for env in (old,new):
        validate_env(env)
        env['motion_file']='tokenhsi/data/dataset_push_native_phase_xy.yaml'
    comparable=deepcopy(new)
    comparable['pushBox']=deepcopy(old['pushBox'])
    comparable['box']['build']['baseSize']=old['box']['build']['baseSize']
    comparable['pushScene']['path_separation']=old['pushScene']['path_separation']
    comparable['openPushRSI']['bank_file']=old['openPushRSI']['bank_file']
    if comparable!=old or new['pushBox']!=dict(size=[1.5,2.,1.3],
            size_profiles=[[round(1+i*.1,1),2.,1.3] for i in range(6)],mass=20.,friction=.6):
        raise ValueError('Only box Y2m, profile list, bank path and clearance may change')
    old_train=yaml.safe_load(Path(f'tokenhsi/data/cfg/train/rlg/{BASE}.yaml').read_text())
    new_train=yaml.safe_load(Path(f'tokenhsi/data/cfg/train/rlg/{TARGET}.yaml').read_text())
    comparable=deepcopy(new_train)
    comparable['params']['config']['name']=old_train['params']['config']['name']
    if comparable!=old_train:raise ValueError('Train settings must be identical except name')
    old_bank_path=Path(old['openPushRSI']['bank_file'])
    bank_path=Path(new['openPushRSI']['bank_file'])
    old_bank=json.loads(old_bank_path.read_text());bank=json.loads(bank_path.read_text())
    if bank['contract']!=native_contract(new) or bank['candidates']!=old_bank['candidates']:
        raise ValueError('Native bank contract/candidate pool differs')
    audit=json.loads(Path(screen_path).read_text());short=audit['short_screen']
    if not (audit['device']=='cpu' and audit['profiles']==6 and audit['partial_reset'] and
            audit['original_dof_preserved'] and short['proposed']==short['accepted']==949 and
            not short['rejected'] and short['both_agents'] and short['all_profiles'] and
            bank['screen']=='PD_0.1s_all_profiles'):
        raise ValueError('Complete Y2m CPU physics validation is required')
    target=audit['contract']
    if (target['push_box']!=new['pushBox'] or target['rsi']!=new['openPushRSI'] or
            target['rsi_bank_sha256']!=hashlib.sha256(bank_path.read_bytes()).hexdigest() or
            target['push_scene']['parameters']!=new['pushScene'] or
            target['push_at_reward']['parameters']!=new['relationReward']):
        raise ValueError('Target audit is stale')
    weights=torch.load(source,map_location='cpu',weights_only=False)
    check_checkpoint_metadata(weights,checkpoint_metadata(old['relationReward']))
    expected=deepcopy(target)
    expected['push_box']=old['pushBox'];expected['rsi']=old['openPushRSI']
    expected['rsi_bank_sha256']=hashlib.sha256(old_bank_path.read_bytes()).hexdigest()
    expected['push_scene']['parameters']=old['pushScene']
    if weights['open_push_contract']!=expected:
        raise ValueError('Source checkpoint does not match the original experiment')
    if weights['relation_task_instance']!=dict(graph=old['relationGraph'],num_agents=2,num_objects=2):
        raise ValueError('Source graph differs')
    resolved=weights['relation_experiment_config']
    if any(resolved['env'].get(k)!=v for k,v in old.items()):
        raise ValueError('Source saved environment differs')
    saved_train=deepcopy(resolved['train'])
    for key in ('seed','load_checkpoint','load_path'):
        if key in old_train['params']:saved_train['params'][key]=old_train['params'][key]
        else:saved_train['params'].pop(key,None)
    if saved_train!=old_train:raise ValueError('Source training config differs')
    if checkpoint_contracts_match(weights['open_push_contract'],target):
        raise ValueError('Unconverted source unexpectedly matches Y2m')
    # Optimizer, model, every RMS, epoch/frame and best score remain intact.
    # Live simulator state/replay is not portable and is not restored.
    weights['open_push_contract']=target
    weights['env_state']=None
    resolved['env']=new;resolved['train']=new_train
    report=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_epoch=int(weights['epoch']),source_frame=int(weights['frame']),
        target_env=TARGET,screen=str(screen_path),
        change='Only local Y2m, 6 X profiles, matching RSI placement and path clearance2.8m',
        optimizer_preserved=True,epoch_frame_preserved=True,model_and_rms_preserved=True,
        live_simulator_and_rollout_replay_restored=False)
    weights['native_y2m_resume']=report
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(weights,output)
    output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--screen',default='output/push_native_phase_y2m_check/screen.json')
    args=parser.parse_args()
    transfer(args.source,args.output,args.screen)
