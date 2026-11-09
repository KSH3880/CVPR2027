from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch

PUSH = 11


def contract(env):
    spec = deepcopy(env['stage1Push'])
    spec.pop('warm_start_checkpoint', None)
    spec['rsi_bank_sha256'] = hashlib.sha256(Path(spec['rsi']['bank']).read_bytes()).hexdigest()
    return dict(version=1, task_ids=['sit', 'climb', 'carry_at', 'carry_ontop', 'push'],
                amp_families=['carry', 'sit', 'climb', 'push'], settings=spec)


def push_rows(count, options, device, preset='random_scenario'):
    if preset == 'push':
        return torch.ones(count, device=device, dtype=torch.bool)
    if preset != 'random_scenario' or not options['enabled']:
        return torch.zeros(count, device=device, dtype=torch.bool)
    old_count = round(count * options['old_task_fraction'])
    if not 0 < old_count < count:
        raise ValueError('Mixed PUSH training needs both old and PUSH environments')
    # Interleave both pools so the first four viewer environments show both.
    return (torch.arange(count, device=device) * old_count // count ==
            (torch.arange(count, device=device) + 1) * old_count // count)


def check_contract(weights, env):
    saved, expected = deepcopy(weights.get('stage1_push_contract')), contract(env)
    if saved is not None:
        for key in ('old_task_fraction', 'teacher_kl_coef'):
            saved['settings'].pop(key, None)
            expected['settings'].pop(key, None)
    if saved != expected:
        raise ValueError('PUSH checkpoint/config contract mismatch; warm-start needs the transfer loader')


def check_navigation_transfer(weights, env):
    saved, expected = deepcopy(weights.get('stage1_push_contract')), contract(env)
    if saved is None or 'navigation' not in env['stage1Push']:
        raise ValueError('Navigation warm-start requires a PUSH checkpoint and navigation config')
    raw_transfer = expected['settings']['rsi'].get('mode') == 'raw' and any(
        key in expected['settings']['box'] for key in ('size_range', 'size_ranges'))
    for spec in (saved['settings'], expected['settings']):
        spec['source_checkpoint'] = str(Path(spec['source_checkpoint']).resolve())
        for key in ('navigation', 'goal_distance_range', 'contact_midpoint_only', 'contact_mode', 'old_task_fraction', 'teacher_kl_coef', 'success_radius'):
            spec.pop(key, None)
        if raw_transfer:
            spec.pop('contact_height', None)
            spec.pop('rsi_bank_sha256', None)
            spec['box'] = {'friction': spec['box']['friction']}
            for key in ('mode', 'bank', 'clearance', 'uniform_goal_distance', 'goal_distance_max'):
                spec['rsi'].pop(key, None)
    if saved != expected:
        raise ValueError('PUSH navigation transfer cannot change physics, AMP, RSI, or source teacher')


def load_bank(options, manifest, motion_ids=None):
    import yaml
    path = Path(manifest)
    rows = yaml.safe_load(path.read_text())['motions']['push']
    bank = json.loads(Path(options['rsi']['bank']).read_text())
    expected_hashes = bank['contract']['motion_hashes']['push']
    selected = range(len(rows)) if motion_ids is None else motion_ids
    if len(rows) != len(expected_hashes) or any(
        hashlib.sha256((path.parent / rows[index]['file']).read_bytes()).hexdigest() != expected_hashes[index]
        for index in selected
    ):
        raise ValueError('PUSH RSI motion hashes/order differ from the manifest')
    expected = bank['contract']['push_box']
    if expected != options['box']:
        raise ValueError('PUSH box physics differs from the RSI bank')
    return bank


def main():
    import argparse
    import tempfile
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--checkpoint')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--source')
    parser.add_argument('--old-fraction', type=float)
    parser.add_argument('--teacher-kl', type=float)
    parser.add_argument('--warm-start')
    parser.add_argument('--require-navigation', action='store_true')
    parser.add_argument('--require-raw', action='store_true')
    args = parser.parse_args()
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)['relation_experiment_config']
        data = dict(env=saved['env'], sim=saved['sim'])
    else:
        data = yaml.safe_load(Path(args.config).read_text())
    spec = data['env']['stage1Push']
    if args.require_navigation and 'navigation' not in spec:
        parser.error('PUSH 360 requires its navigation config/checkpoint; use PUSH_CHECKPOINT for an old PUSH warm-start')
    if args.require_raw and (spec['rsi'].get('mode') != 'raw' or not any(
            key in spec['box'] for key in ('size_range', 'size_ranges'))):
        parser.error('PUSH raw requires its variable-size config/checkpoint; use PUSH_CHECKPOINT for an old PUSH warm-start')
    for key, value in [('source_checkpoint', args.source), ('old_task_fraction', args.old_fraction), ('teacher_kl_coef', args.teacher_kl)]:
        if value is not None:
            spec[key] = value
    if args.warm_start:
        if args.checkpoint:
            parser.error('warm-start and resume are mutually exclusive')
        weights = torch.load(args.warm_start, map_location='cpu', weights_only=False)
        check_navigation_transfer(weights, data['env'])
        spec['warm_start_checkpoint'] = args.warm_start
    if not 0 < spec['old_task_fraction'] < 1 or spec['teacher_kl_coef'] < 0:
        parser.error('old fraction must be in (0,1); teacher KL must be nonnegative')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', prefix='push_env_', suffix='.yaml', dir=args.output_dir, delete=False) as stream:
        yaml.safe_dump(data, stream, sort_keys=False)
        print(stream.name)


if __name__ == '__main__':
    main()
