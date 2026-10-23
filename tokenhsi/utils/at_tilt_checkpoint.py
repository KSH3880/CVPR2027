"""Explicit reward-only migration; all learned state remains unchanged."""
from copy import deepcopy
import hashlib
from pathlib import Path
import yaml
from utils.relation_task_spec import checkpoint_metadata

BASE='open_push_contact_at_friction06_random_size_goal1to25_at01'
CFG=Path('tokenhsi/data/cfg/multi_agent')


def migrate_checkpoint(weights):
    source=yaml.safe_load((CFG/(BASE+'.yaml')).read_text())['env']
    target=yaml.safe_load((CFG/(BASE+'_tilt035.yaml')).read_text())['env']
    expected=deepcopy(target)
    if expected['relationReward']['contact_at'].pop('max_tilt')!=.35:
        raise ValueError('This migration requires an AT limit of 0.35 rad')
    if expected!=source:raise ValueError('Only AT max_tilt may change')
    snapshot=weights['relation_experiment_config']
    old=deepcopy(snapshot['env'])
    # Fields added by runtime, not by the experiment YAML.
    for key in ('render_device','task_plan','motion_file','seed','numObservations','numActions'):
        old.pop(key,None)
    if old!=source:raise ValueError('Source checkpoint is not the original AT01 experiment')
    train=yaml.safe_load(Path('tokenhsi/data/cfg/train/rlg/open_push_unified_amp.yaml').read_text())
    if snapshot['train']['params']['network']!=train['params']['network']:
        raise ValueError('Network configuration changed')
    if weights.get('relation_metadata')!=checkpoint_metadata(source['relationReward']) or weights.get('amp_enabled') is not True:
        raise ValueError('Source reward metadata/AMP differs')
    contract=weights['open_push_contract']
    if contract.get('contact_at')!=source['relationReward']['contact_at'] or contract.get('object_obs_size')!=40 or 'policy_observation' in contract:
        raise ValueError('Requires original door-inclusive policy contract')
    for path,key in ((source['openPushRSI']['bank_file'],'rsi_bank_sha256'),(snapshot['env']['motion_file'],'manifest_sha256')):
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=contract[key]:
            raise ValueError('Source bank/manifest changed')
    result=dict(weights)
    result['open_push_contract']=deepcopy(contract)
    result['open_push_contract']['contact_at']=deepcopy(target['relationReward']['contact_at'])
    result['relation_metadata']=checkpoint_metadata(target['relationReward'])
    result['relation_experiment_config']=deepcopy(snapshot)
    result['relation_experiment_config']['env']['relationReward']=deepcopy(target['relationReward'])
    result['at_tilt_migration']={'source_epoch':weights.get('epoch'),'max_tilt_rad':.35,'change':'AT current success and its saturation only'}
    return result


PUSH_BASE=BASE+'_no_doors'


def migrate_push_only_checkpoint(weights):
    """570D physical PUSH profile: add only the AT gate, keeping all state."""
    from utils.push_only import training_contract
    source=yaml.safe_load((CFG/(PUSH_BASE+'.yaml')).read_text())['env']
    target=yaml.safe_load((CFG/(PUSH_BASE+'_tilt035.yaml')).read_text())['env']
    expected=deepcopy(target)
    if expected['relationReward']['contact_at'].pop('max_tilt')!=.35 or expected!=source:
        raise ValueError('Only AT max_tilt=0.35 may change')
    snapshot=weights['relation_experiment_config']
    old=deepcopy(snapshot['env'])
    for key in ('render_device','task_plan','motion_file','seed','numObservations','numActions','checkpoint'):
        old.pop(key,None)
    if old!=source:raise ValueError('Requires the physical PUSH-only 570D experiment')
    train_path='tokenhsi/data/cfg/train/rlg/open_push_unified_amp.yaml'
    train=yaml.safe_load(Path(train_path).read_text())
    if snapshot['train']['params']['network']!=train['params']['network']:
        raise ValueError('Network configuration changed')
    source['motion_file']=snapshot['env']['motion_file']
    target['motion_file']=source['motion_file']
    if source['motion_file']!='tokenhsi/data/dataset_push_only_push180.yaml':
        raise ValueError('PUSH-only motion manifest changed')
    if weights.get('open_push_contract')!=training_contract(source,train_path):
        raise ValueError('PUSH-only checkpoint physics/reference/input contract differs')
    if weights.get('relation_metadata')!=checkpoint_metadata(source['relationReward']) or weights.get('amp_enabled') is not True:
        raise ValueError('Source reward metadata/AMP differs')
    if snapshot.get('observation_size')!=570:
        raise ValueError('Requires observation width 570')
    result=dict(weights)
    result['open_push_contract']=training_contract(target,train_path)
    result['relation_metadata']=checkpoint_metadata(target['relationReward'])
    result['relation_experiment_config']=deepcopy(snapshot)
    result['relation_experiment_config']['env']['relationReward']=deepcopy(target['relationReward'])
    result['at_tilt_migration']={'source_epoch':weights.get('epoch'),'max_tilt_rad':.35,'profile':'push_only_physical_v1','change':'AT current success and its saturation only'}
    return result
