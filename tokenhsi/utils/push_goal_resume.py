"""Explicit full-state migration for the fixed-face PUSH 3--5m curriculum."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import yaml
from utils.open_push_rsi import bank_contract
from utils.open_push_spec import validate_env
from utils.push_only import training_contract
from utils.relation_task_spec import checkpoint_metadata

CONFIG = Path('tokenhsi/data/cfg/multi_agent')
SOURCE = CONFIG / 'push_fixed130_fixed_face_midpoint.yaml'
TARGET = CONFIG / 'push_fixed130_fixed_face_midpoint_goal3to5.yaml'
TRAIN = Path('tokenhsi/data/cfg/train/rlg/open_push_unified_amp.yaml')
MANIFEST = 'tokenhsi/data/dataset_push_only_push180.yaml'


def environments():
    source, target = [yaml.safe_load(p.read_text())['env'] for p in (SOURCE, TARGET)]
    for env in (source, target):
        env['motion_file'] = MANIFEST
        validate_env(env)
    allowed = deepcopy(source)
    allowed['pushGoalRandomization']['distance_range'] = [3., 5.]
    allowed['pushSceneRandomization'].update(arena_radius=8., min_station_distance=12.)
    allowed['envSpacing'] = 20
    allowed['openPushRSI']['bank_file'] = target['openPushRSI']['bank_file']
    if source['pushGoalRandomization']['distance_range'] != [1., 2.5] or allowed != target:
        raise ValueError('Only the goal range, workspace and bank path may change')
    return source, target


def project_bank(source, target):
    path = Path(source['openPushRSI']['bank_file'])
    raw = path.read_bytes()
    bank = json.loads(raw)
    if bank['contract'] != bank_contract(source) or bank['screen'] != 'PD_12_steps':
        raise ValueError('Original PD-screened bank contract differs')
    old, new = deepcopy(bank['contract']), bank_contract(target)
    for key in ('push_scene', 'push_goal'):
        old[key] = new[key]
    if old != new:
        raise ValueError('Physical/reference screen settings changed')
    result = deepcopy(bank)
    result['contract'] = new
    result['goal_range_provenance'] = dict(source_bank=str(path),
        source_sha256=hashlib.sha256(raw).hexdigest(),
        method='Reuse identical PD-screened poses under rigid XY/yaw transforms. '
        'Goal bearing law and local reset geometry unchanged; goal radius and '
        'station separation increase. Actual target resets separately audited; no new all-candidate PD screen.')
    return result


def migrate(weights, source, target):
    required = ('model', 'optimizer', 'epoch', 'frame', 'running_mean_std',
                'reward_mean_std', 'amp_input_mean_std', 'relation_experiment_config')
    if weights.get('amp_enabled') is not True or any(k not in weights for k in required):
        raise ValueError('Complete AMP training state required')
    if weights.get('open_push_contract') != training_contract(source, TRAIN):
        raise ValueError('Source must match the original fixed-face experiment')
    if weights.get('relation_metadata') != checkpoint_metadata(source['relationReward']):
        raise ValueError('Reward metadata differs')
    snapshot = weights['relation_experiment_config']
    if tuple(snapshot[k] for k in ('num_envs', 'num_agents', 'num_objects', 'observation_size')) != (2048, 2, 4, 570):
        raise ValueError('Training dimensions differ')
    if any(snapshot['env'].get(k) != v for k, v in source.items()):
        raise ValueError('Source config snapshot differs')
    if snapshot['train']['params']['network'] != yaml.safe_load(TRAIN.read_text())['params']['network']:
        raise ValueError('Network differs')
    if json.loads(Path(target['openPushRSI']['bank_file']).read_text()) != project_bank(source, target):
        raise ValueError('Target bank candidates/provenance differ')
    # All learned state, including optimizer, AMP/RMS and counters, stays intact.
    result = dict(weights)
    result['open_push_contract'] = training_contract(target, TRAIN)
    snapshot = deepcopy(snapshot)
    snapshot['env'].update(deepcopy(target))
    result['relation_experiment_config'] = snapshot
    return result
