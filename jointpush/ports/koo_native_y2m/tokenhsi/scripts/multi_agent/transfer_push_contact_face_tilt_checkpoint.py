"""Explicit transfer from common-face CONTACT+AT to AT soft tilt shaping."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import torch
import yaml
from utils.relation_task_spec import checkpoint_metadata, check_checkpoint_metadata
from utils.open_push_spec import validate_env
from utils.open_push_rsi import bank_contract


def transfer(source, source_env_path, target_env_path, target_train_path, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError('Transfer output already exists; choose a new path')
    old = yaml.safe_load(Path(source_env_path).read_text())['env']
    new = yaml.safe_load(Path(target_env_path).read_text())['env']
    validate_env(old); validate_env(new)
    angles=new['relationReward']['contact_at']
    import math
    for key,degrees in [('max_tilt',10),('tilt_full_angle',5),('tilt_zero_angle',15)]:
        if angles.get(key) != math.radians(degrees):
            raise ValueError('Target must enable 5/15-degree AT shaping and 10-degree success')
    comparable = deepcopy(new)
    for key in ('max_tilt','tilt_full_angle','tilt_zero_angle'):
        comparable['relationReward']['contact_at'].pop(key)
    comparable['openPushRSI']['bank_file'] = old['openPushRSI']['bank_file']
    if comparable != old or any(k in old['relationReward']['contact_at'] for k in ('max_tilt','tilt_full_angle','tilt_zero_angle')):
        raise ValueError('Only AT tilt shaping/success and bank path may differ')
    for env in (old, new):
        env['motion_file'] = 'tokenhsi/data/dataset_open_push.yaml'
        bank = json.loads(Path(env['openPushRSI']['bank_file']).read_text())
        if bank['contract'] != bank_contract(env) or bank['screen'] != 'PD_12_steps':
            raise ValueError('RSI bank contract differs')
    old_bank = Path(old['openPushRSI']['bank_file'])
    new_bank = Path(new['openPushRSI']['bank_file'])
    old_data, new_data = json.loads(old_bank.read_text()), json.loads(new_bank.read_text())
    new_data.pop('contact_at_tilt_soft_reuse', None)
    new_data['contract'] = old_data['contract']
    if new_data != old_data:
        raise ValueError('RSI candidates or screen data differ')
    weights = torch.load(source, map_location='cpu')
    check_checkpoint_metadata(weights, checkpoint_metadata(old['relationReward']))
    contract = weights['open_push_contract']
    if (contract['push_at_reward']['parameters'] != old['relationReward'] or
            contract['rsi'] != old['openPushRSI'] or
            contract['rsi_bank_sha256'] != hashlib.sha256(old_bank.read_bytes()).hexdigest()):
        raise ValueError('Source checkpoint reward/RSI contract differs')
    if weights['relation_task_instance'] != dict(graph=old['relationGraph'], num_agents=2, num_objects=2):
        raise ValueError('Source must be the two-agent door-free PUSH experiment')
    new_contract = deepcopy(contract)
    new_contract['push_at_reward']['parameters'] = new['relationReward']
    new_contract['rsi'] = new['openPushRSI']
    new_contract['rsi_bank_sha256'] = hashlib.sha256(new_bank.read_bytes()).hexdigest()
    weights['open_push_contract'] = new_contract
    weights['relation_metadata'] = checkpoint_metadata(new['relationReward'])
    # Actor, critic, discriminator and all normalization tensors remain intact.
    weights['optimizer']['state'] = {}
    weights['env_state'] = None
    weights['last_mean_rewards'] = -100500
    resolved = weights.get('relation_experiment_config')
    if resolved:
        resolved['env'] = new
        resolved['train'] = yaml.safe_load(Path(target_train_path).read_text())
    report = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        target_env=str(target_env_path), target_train=str(target_train_path),
        source_epoch=int(weights['epoch']), source_frame=int(weights['frame']),
        transferred='actor/critic/discriminator weights and input/value/AMP normalization',
        optimizer_reset=True, best_reward_reset=True, rollout_replay_restored=False,
        change='AT state/progress soft tilt 5..15deg; AT success <=10deg; CONTACT unchanged',
        original_contract=contract, new_contract=new_contract)
    weights['reward_transfer'] = report
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(weights, output)
    output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('source','source_epoch','optimizer_reset','change')}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--source-env', default='tokenhsi/data/cfg/multi_agent/push_contact_face_at.yaml')
    p.add_argument('--target-env', default='tokenhsi/data/cfg/multi_agent/push_contact_face_at_tilt_soft.yaml')
    p.add_argument('--target-train', default='tokenhsi/data/cfg/train/rlg/push_contact_face_at_tilt_soft.yaml')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    transfer(a.source, a.source_env, a.target_env, a.target_train, a.output)
