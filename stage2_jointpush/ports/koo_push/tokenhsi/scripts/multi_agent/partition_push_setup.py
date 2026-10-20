"""Resolve Stage1 paths or restore a saved Partition Push configuration on CPU."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from utils.partition_push_paths import checkpoint_path, resolve_stage1_checkpoint
import yaml



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    resolve = sub.add_parser('checkpoint'); resolve.add_argument('path',nargs='?')
    config = sub.add_parser('config')
    config.add_argument('checkpoint'); config.add_argument('output')
    config.add_argument('--size', nargs=3, type=float)
    config.add_argument('--mass', type=float, help='Mass per participating human in kg; total is n times this value')
    args = parser.parse_args()
    if args.command == 'checkpoint':
        print(resolve_stage1_checkpoint(args.path)); return
    import torch
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    env = saved['relation_experiment_config']['env']
    if saved.get('open_push_contract', {}).get('policy') != 'partition_push_ca4':
        raise ValueError('Expected a Partition Push Stage2 checkpoint')
    if env.get('partitionPush',{}).get('version') != 2:
        raise ValueError('This profile requires a version2 variable-group checkpoint; start from Stage1 for older runs')
    if args.size is not None or args.mass is not None:
        import math
        values = (args.size or [])+([] if args.mass is None else [args.mass])
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError('Ablation size/mass must be finite and positive')
        if args.size:
            env['partitionPush']['size_min'] = env['partitionPush']['size_max'] = args.size
        if args.mass is not None:
            env['partitionPush']['mass_per_agent'] = [args.mass]*2
        env['partitionPushAblation'] = True
    native = yaml.safe_load((Path(__file__).resolve().parents[2]/'data/cfg/multi_agent/partition_push_ca4.yaml').read_text())
    dest = Path(args.output); dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.safe_dump(dict(env=env, sim=native['sim']), sort_keys=False))
    print(dest.resolve())


if __name__ == '__main__': main()
