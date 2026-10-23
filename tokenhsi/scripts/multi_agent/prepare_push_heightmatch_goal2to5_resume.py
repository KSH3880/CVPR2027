"""Preserve the height-matched PUSH training state while changing goals 3--5 to 2--5m."""
from isaacgym import gymapi  # Isaac Gym must precede torch.
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import torch
import yaml
from utils.open_push_spec import validate_env
from utils.push_goal_resume import project_bank
from utils.push_only import training_contract
from utils.relation_task_spec import checkpoint_metadata

PREFIX = 'push_fixed130_point_zband_heightmatch_goal'
TRAIN = Path('tokenhsi/data/cfg/train/rlg/open_push_unified_amp.yaml')


def contract(env):
    result = training_contract(env, TRAIN)
    result.update(push_objective='fixed_material_point_height_band_v2',
                  push_contact_point=deepcopy(env['pushContactPoint']),
                  push_point_reward=deepcopy(env['pushPointReward']))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite checkpoint')
    source, target = [yaml.safe_load(Path(f'tokenhsi/data/cfg/multi_agent/{PREFIX}{r}.yaml').read_text())['env'] for r in ('3to5', '2to5')]
    for env in (source, target):
        env['motion_file'] = 'tokenhsi/data/dataset_push_only_push180.yaml'
        validate_env(env)
    allowed = deepcopy(source)
    allowed['pushGoalRandomization']['distance_range'] = [2., 5.]
    allowed['openPushRSI']['bank_file'] = target['openPushRSI']['bank_file']
    if source['pushGoalRandomization']['distance_range'] != [3., 5.] or allowed != target:
        raise ValueError('Only goal radius and bank path may change')
    weights = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if weights.get('open_push_contract') != contract(source):
        raise ValueError('Source checkpoint contract differs')
    if weights.get('relation_metadata') != checkpoint_metadata(source['relationReward']):
        raise ValueError('Reward metadata differs')
    required = ('model', 'optimizer', 'epoch', 'frame', 'running_mean_std', 'reward_mean_std', 'amp_input_mean_std')
    if weights.get('amp_enabled') is not True or any(k not in weights for k in required):
        raise ValueError('Complete AMP training state required')
    snapshot = weights['relation_experiment_config']
    if tuple(snapshot[k] for k in ('num_envs', 'num_agents', 'num_objects', 'observation_size')) != (2048, 2, 4, 570):
        raise ValueError('Training dimensions differ')
    if any(snapshot['env'].get(k) != v for k, v in source.items()):
        raise ValueError('Source config snapshot differs')
    if snapshot['train']['params']['network'] != yaml.safe_load(TRAIN.read_text())['params']['network']:
        raise ValueError('Network differs')
    bank = project_bank(source, target)
    bank['goal_range_provenance']['method'] = 'Reuse identical height-screened poses; only goal radius changes from 3--5 to 2--5m. No new all-candidate PD screen.'
    path = Path(target['openPushRSI']['bank_file'])
    if path.exists():
        if json.loads(path.read_text()) != bank:
            raise ValueError('Existing target bank differs')
    else:
        with path.open('x') as f:
            f.write(json.dumps(bank, indent=2) + '\n')
    result = dict(weights)
    result['open_push_contract'] = contract(target)
    result['relation_experiment_config'] = deepcopy(snapshot)
    result['relation_experiment_config']['env'].update(deepcopy(target))
    report = dict(source=str(args.checkpoint.resolve()), source_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(), epoch=int(weights['epoch']), frame=int(weights['frame']), goal_distance_range_m=[2., 5.], preserved='All learned state including model/optimizer/AMP/RMS/counters', changed='Goal range and RSI bank metadata only')
    result['push_goal_range_migration'] = report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as f:
        torch.save(result, f)
    args.output.with_suffix('.migration.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
