"""CPU-only, explicit owner-square -> box-radial 3..5m checkpoint transfer."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import torch
import yaml
from utils.open_push_spec import validate_env, checkpoint_contracts_match
from utils.push_scene import validate_scene
from utils.open_push_rsi import bank_contract
from utils.relation_task_spec import checkpoint_metadata, check_checkpoint_metadata

BASE = 'push_contact_face_at_tilt_soft'


def transfer(source, output, audit_path,
             source_env_path=f'tokenhsi/data/cfg/multi_agent/{BASE}.yaml',
             target_env_path=f'tokenhsi/data/cfg/multi_agent/{BASE}_3to5m.yaml',
             source_train_path=f'tokenhsi/data/cfg/train/rlg/{BASE}.yaml',
             target_train_path=f'tokenhsi/data/cfg/train/rlg/{BASE}_3to5m.yaml'):
    source, output = Path(source), Path(output)
    if output.exists() or output.with_suffix('.json').exists():
        raise ValueError('Transfer output already exists')
    old = yaml.safe_load(Path(source_env_path).read_text())['env']
    new = yaml.safe_load(Path(target_env_path).read_text())['env']
    for env in (old, new):
        validate_env(env); validate_scene(env)
        env['motion_file'] = 'tokenhsi/data/dataset_open_push.yaml'
    if old['pushScene']['random_at_goal'] != dict(reach=2., min_distance=1., agent_clearance=.5):
        raise ValueError('Unexpected source goal distribution')
    if new['pushScene']['random_at_goal'] != dict(sampling='box_radial', min_distance=3., max_distance=5., agent_clearance=.5):
        raise ValueError('Target must use box-radial 3..5m goals')
    comparable = deepcopy(new)
    comparable['pushScene']['random_at_goal'] = old['pushScene']['random_at_goal']
    if comparable != old:
        raise ValueError('Only goal distribution may change')
    old_train = yaml.safe_load(Path(source_train_path).read_text())
    new_train = yaml.safe_load(Path(target_train_path).read_text())
    comparable = deepcopy(new_train)
    comparable['params']['config']['name'] = old_train['params']['config']['name']
    if comparable != old_train:
        raise ValueError('Only train experiment name may change')
    bank = Path(new['openPushRSI']['bank_file'])
    if json.loads(bank.read_text())['contract'] != bank_contract(new):
        raise ValueError('RSI bank contract differs')
    audit = json.loads(Path(audit_path).read_text())
    target = audit['contract']
    if (audit['device'] != 'cpu' or audit['reset_agent_samples'] < 1000 or
            not audit['both_assignments_rsi'] or not audit['bank_preserved'] or
            target['push_scene'] != dict(revision=1, parameters=new['pushScene'], box_reset=new['box']['reset']) or
            target['push_at_reward']['parameters'] != new['relationReward'] or
            target['rsi_bank_sha256'] != hashlib.sha256(bank.read_bytes()).hexdigest()):
        raise ValueError('Target CPU connection audit differs or is incomplete')
    weights = torch.load(source, map_location='cpu')
    check_checkpoint_metadata(weights, checkpoint_metadata(old['relationReward']))
    expected_source = deepcopy(target)
    expected_source['push_scene']['parameters'] = old['pushScene']
    if weights['open_push_contract'] != expected_source:
        raise ValueError('Source physics/network/RSI/scene contract differs')
    if weights['relation_task_instance'] != dict(graph=old['relationGraph'], num_agents=2, num_objects=2):
        raise ValueError('Source graph differs')
    resolved = weights['relation_experiment_config']
    saved_train = deepcopy(resolved['train'])
    # These three entries are resolved from CLI at launch, not model settings.
    for key in ('seed', 'load_checkpoint', 'load_path'):
        if key in old_train['params']:
            saved_train['params'][key] = old_train['params'][key]
        else:
            saved_train['params'].pop(key, None)
    if any(resolved['env'].get(k) != v for k, v in old.items()) or saved_train != old_train:
        raise ValueError('Source saved env/train differs')
    if checkpoint_contracts_match(weights['open_push_contract'], target):
        raise ValueError('Unconverted source unexpectedly matches target')
    weights['open_push_contract'] = target
    weights['optimizer']['state'] = {}
    weights['env_state'] = None
    weights['last_mean_rewards'] = -100500
    resolved['env'] = new
    resolved['train'] = new_train
    report = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_epoch=int(weights['epoch']), source_frame=int(weights['frame']),
        target_env=str(target_env_path), target_train=str(target_train_path), audit=str(audit_path),
        change='Only goal sampling: box-centred distance uniform 3..5m, random angle, joint path rejection',
        preserved='model/actor/critic/discriminator and observation/value/AMP RMS; epoch/frame',
        optimizer_reset=True, best_reward_reset=True, rollout_replay_restored=False)
    weights['goal_distance_transfer'] = report
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(weights, output)
    output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--audit', default=f'output/{BASE}_3to5m_check/audit.json')
    a = p.parse_args()
    transfer(a.source, a.output, a.audit)
