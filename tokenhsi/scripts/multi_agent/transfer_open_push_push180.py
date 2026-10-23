"""Explicit full-state transfer: friction0.5 legacy pair -> PUSH2m wide/front21.

Keep learned weights, optimizer, statistics and counters; validate and replace
only the authorized experiment metadata. The source file is never overwritten.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import torch
import yaml
from utils.open_push_rsi import bank_contract
from utils.open_push_spec import validate_env
from utils.relation_task_spec import checkpoint_metadata

SOURCE_CONFIG = Path('tokenhsi/data/cfg/multi_agent/open_push_late_rsi_hinge_friction.yaml')
TARGET_CONFIG = Path('tokenhsi/data/cfg/multi_agent/open_push_late_rsi_hinge_friction_push2m_push180.yaml')
SOURCE_MANIFEST = Path('tokenhsi/data/dataset_open_push.yaml')
TARGET_MANIFEST = Path('tokenhsi/data/dataset_open_push_push180.yaml')
TRAIN_CONFIG = Path('tokenhsi/data/cfg/train/rlg/open_push_unified_amp.yaml')
TARGET_EXPERIMENT = 'OpenPushLateRSIHingeFrictionPush2mWidePush180'


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_contract(env, manifest, bank_hash):
    network = yaml.safe_load(TRAIN_CONFIG.read_text())['params']['network']
    result = dict(revision=5,
        open_reward_contract='rsi_baseline_plus_hand_angle_v1',
        push_reward_contract='rsi_baseline_plus_hand_distance_midpoint_xy_v2',
        shuffle_token_order=env['relationGraph'].get('shuffle_token_order', False),
        task_probabilities=env['relationGraph']['task_probabilities'],
        manifest_sha256=file_hash(manifest), skill=env['skill'],
        expert_mapping=env['ampExpertMapping'], init=env['skillInitProb'],
        door=env['door'], hold=env['doorHold'],
        spawn_distance_range=env['doorSpawnDistanceRange'], push_box=env['pushBox'],
        task_labels=['OPEN', 'PUSH'], object_obs_size=40,
        share_edge_encoder=network['transformer'].get('share_edge_encoder', False),
        rsi=env['openPushRSI'], rsi_bank_sha256=bank_hash)
    if 'openPushLayout' in env:
        result['layout'] = dict(env['openPushLayout'])
    return copy.deepcopy(result)


def prepare(weights, source, target, old_bank, new_bank, old_hash, new_hash, output):
    validate_env(source); validate_env(target)
    allowed = copy.deepcopy(source)
    allowed['relationReward']['push']['target_distance'] = 2.0
    allowed['envSpacing'] = 14
    allowed['openPushLayout'] = dict(door_x=4.0, push_x=-4.0, lane_spacing=10.0)
    allowed['openPushRSI']['bank_file'] = target['openPushRSI']['bank_file']
    if source['relationReward']['push']['target_distance'] != 1.5 or allowed != target:
        raise ValueError('Only PUSH1.5->2m, wide layout and RSI bank path may change')
    for env, manifest, bank in ((source, SOURCE_MANIFEST, old_bank),
                                (target, TARGET_MANIFEST, new_bank)):
        if bank.get('screen') != 'PD_12_steps' or bank['contract'] != bank_contract(dict(env, motion_file=str(manifest))):
            raise ValueError('RSI bank does not match config/reference files')
        motions = yaml.safe_load(manifest.read_text())['motions']
        for skill in ('doorOpen', 'push'):
            if {r['motion_id'] for r in bank['candidates'][skill]} != set(range(len(motions[skill]))):
                raise ValueError('Every reference clip requires screened RSI candidates')
    old = yaml.safe_load(SOURCE_MANIFEST.read_text())['motions']
    new = yaml.safe_load(TARGET_MANIFEST.read_text())['motions']
    selected = json.loads(Path('tokenhsi/data/bones_push180_selection.json').read_text())['clips']
    if (len(old['push']) != 2 or len(new['push']) != 21 or new['loco'] != old['loco']
            or new['doorOpen'] != old['doorOpen']
            or any(row not in new['push'] for row in old['push'])
            or {Path(row['file']).parts[-3] for row in new['push']} != set(selected)
            or any(row['weight'] != 1.0 for row in new['push'])):
        raise ValueError('Manifest must preserve loco/OPEN/legacy pair and select exactly front21')
    required = ('model', 'optimizer', 'epoch', 'frame', 'running_mean_std',
                'reward_mean_std', 'amp_input_mean_std', 'relation_experiment_config')
    if weights.get('amp_enabled') is not True or any(k not in weights for k in required):
        raise ValueError('Complete AMP training state required')
    if weights.get('open_push_contract') != expected_contract(source, SOURCE_MANIFEST, old_hash):
        raise ValueError('Source checkpoint contract differs from the legacy friction0.5 experiment')
    if weights.get('relation_metadata') != checkpoint_metadata(source['relationReward']):
        raise ValueError('Source reward metadata differs')
    snapshot = weights['relation_experiment_config']
    if (snapshot['num_envs'], snapshot['num_agents'], snapshot['num_objects'], snapshot['observation_size']) != (2048, 2, 4, 684):
        raise ValueError('Source training dimensions differ')
    if any(snapshot['env'].get(k) != v for k, v in source.items()):
        raise ValueError('Saved source config differs')
    if snapshot['train']['params']['network'] != yaml.safe_load(TRAIN_CONFIG.read_text())['params']['network']:
        raise ValueError('Source network differs')
    result = dict(weights)
    result['open_push_contract'] = expected_contract(target, TARGET_MANIFEST, new_hash)
    result['relation_metadata'] = checkpoint_metadata(copy.deepcopy(target['relationReward']))
    snapshot = copy.deepcopy(snapshot)
    snapshot['env'].update(copy.deepcopy(target))
    snapshot['env']['motion_file'] = str(TARGET_MANIFEST)
    snapshot['env']['checkpoint'] = str(output)
    snapshot['train']['params']['load_path'] = str(output)
    snapshot['train']['params']['config']['name'] = TARGET_EXPERIMENT
    result['relation_experiment_config'] = snapshot
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve() == args.checkpoint.resolve():
        raise ValueError('Output must be a new file; source is preserved')
    source = yaml.safe_load(SOURCE_CONFIG.read_text())['env']
    target = yaml.safe_load(TARGET_CONFIG.read_text())['env']
    raw = [Path(e['openPushRSI']['bank_file']).read_bytes() for e in (source, target)]
    banks = [json.loads(b) for b in raw]
    hashes = [hashlib.sha256(b).hexdigest() for b in raw]
    weights = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    transferred = prepare(weights, source, target, *banks, *hashes, args.output)
    report = dict(source_checkpoint=str(args.checkpoint.resolve()),
        source_sha256=file_hash(args.checkpoint), source_epoch=int(weights['epoch']),
        frame=int(weights['frame']), source_config=str(SOURCE_CONFIG),
        target_config=str(TARGET_CONFIG), target_manifest=str(TARGET_MANIFEST),
        target_bank_sha256=hashes[1],
        change='PUSH1.5->2m; layout doorX4/pushX-4/lane10/envSpacing14; front PUSH2->21 with rebuilt RSI',
        preserved=['model', 'optimizer', 'running_mean_std', 'reward_mean_std',
                   'amp_input_mean_std', 'epoch', 'frame', 'last_mean_rewards', 'env_state'])
    transferred['open_push_transfer'] = report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as f:
        torch.save(transferred, f)
    report['output'] = str(args.output.resolve())
    args.output.with_suffix('.transfer.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
