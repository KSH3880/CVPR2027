#!/usr/bin/env python3
"""Inspect the imported reward run or launch its local viewer."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'output/stage2_900_v4_reward'


def checkpoints():
    return sorted(RESULTS.glob('*/nn/*_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].pth'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['list', 'inspect', 'view', 'vnc'])
    parser.add_argument('--checkpoint', help='Local checkpoint path; default: highest numbered epoch')
    parser.add_argument('--gpu', default='6')
    parser.add_argument('--agents', type=int, default=4)
    parser.add_argument('--envs', type=int, default=1)
    parser.add_argument('--objects', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    paths = checkpoints()
    if args.mode == 'list':
        for path in paths:
            print(path.relative_to(ROOT))
        return
    if not paths:
        parser.error('No imported checkpoints')
    path = Path(args.checkpoint) if args.checkpoint else max(paths, key=lambda p: int(p.stem.rsplit('_', 1)[1]))
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if ROOT not in path.parents or not path.is_file():
        parser.error('Choose an existing checkpoint inside this checkout')
    sidecar = path.parent.parent / 'relation_config.yaml'
    if not sidecar.is_file():
        parser.error('Missing saved relation_config.yaml')
    if args.mode == 'inspect':
        import torch
        sys.path.insert(0, str(ROOT / 'tokenhsi'))
        from utils.experiments import load_environment
        config, source = load_environment('stage2_900_v4_reward', '', str(path), saved=True)
        model = torch.load(str(path), map_location='cpu', weights_only=False)
        tensors = {k: v for k, v in model['model'].items() if torch.is_tensor(v)}
        report = {
            'checkpoint': str(path.relative_to(ROOT)),
            'saved_config': str(Path(source).relative_to(ROOT)),
            'epoch': model.get('epoch'),
            'frame': model.get('frame'),
            'model_tensor_count': len(tensors),
            'all_model_tensors_finite': all(bool(torch.isfinite(v).all()) for v in tensors.values()),
            'reward_source': config['env']['stage2']['reward_source'],
            'episode_length': config['env']['episodeLength'],
            'before': config['env']['before'],
            'premature_motion_penalty': config['env']['stage2JointPush']['premature_motion_penalty'],
            'seed_file': (path.parent.parent / 'random_seed.txt').read_text().strip(),
            'note': 'CPU inspection only; training reward is not formal evaluation success rate',
        }
        print(json.dumps(report, indent=2, default=str))
        out = ROOT / 'output_etc/check/inspection'
        out.mkdir(parents=True, exist_ok=True)
        (out / (path.parent.parent.name + '_' + path.stem + '.json')).write_text(json.dumps(report, indent=2, default=str) + '\n')
        if not report['all_model_tensors_finite']:
            raise SystemExit('Nonfinite model tensor')
        return
    env = os.environ.copy()
    env['TOKENHSI_GPU'] = args.gpu
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['TORCH_EXTENSIONS_DIR'] = str(ROOT / 'output_etc/check/torch_extensions')
    env['OUTPUT_PATH'] = str(ROOT / 'output_etc/check/viewer' / path.stem)
    script = 'stage2_900_v4_reward_' + ('vnc' if args.mode == 'vnc' else 'test') + '.sh'
    command = ['bash', str(ROOT / 'tokenhsi/scripts/multi_agent' / script), str(path),
               str(args.agents), str(args.envs), str(args.objects), str(args.repeats)]
    if args.dry_run:
        print(json.dumps({'cwd': str(ROOT), 'command': command,
                          'env': {k: env[k] for k in ('TOKENHSI_GPU', 'OUTPUT_PATH', 'TORCH_EXTENSIONS_DIR')}}, indent=2))
        return
    subprocess.run(command, cwd=str(ROOT), env=env, check=True)


if __name__ == '__main__':
    main()
