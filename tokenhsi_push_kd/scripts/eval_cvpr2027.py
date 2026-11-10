"""Run the versioned single-skill protocol with three user-selected seeds."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tokenhsi'), str(ROOT)]
from learning.cvpr2027_scoring import seed_statistics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--tasks', nargs='+', default=['sit', 'climb', 'at', 'on_top', 'push'], choices=['sit', 'climb', 'at', 'on_top', 'push'])
    p.add_argument('--output')
    p.add_argument('--gpu', default='0')
    p.add_argument('--cuda-launch-blocking', choices=['0', '1'], default='0',
                   help='CUDA synchronization debugging (1); normal execution defaults to 0')
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--protocol', choices=['1109', '1108'], default='1109')
    p.add_argument('--seeds', nargs=3, type=int, help='Three distinct user-selected seeds; required for full evaluation')
    p.add_argument('--smoke', action='store_true', help='2 scenes and seed 0; not a paper result')
    p.add_argument('--snapshot-root', help='Replay <root>/<task>/seed_<seed>/initial.npz from another method')
    args = p.parse_args()
    if args.seeds is not None and (len(set(args.seeds)) != 3 or any(s < 0 or s >= 2**32 for s in args.seeds)):
        p.error('--seeds requires three distinct integers in [0, 2**32)')
    if not args.smoke and args.seeds is None:
        p.error('Specify --seeds S1 S2 S3; the protocol leaves seed selection to the user')
    if args.smoke and args.seeds is not None:
        p.error('--smoke uses seed 0 only; omit --seeds')
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        p.error('checkpoint does not exist')
    out = Path(args.output).resolve() if args.output else ROOT/'output_etc/eval'/('cvpr2027_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True, exist_ok=True)
    runtime = Path(os.environ.get('TOKENHSI_RUNTIME', '/home/injesus1010/anaconda3/envs/tokenhsi118'))
    env = os.environ.copy()
    env.update(PATH=str(runtime/'bin')+':'+env['PATH'], LD_LIBRARY_PATH=str(runtime/'lib')+':'+env.get('LD_LIBRARY_PATH', ''),
               PYTHONPATH=str(ROOT)+':'+str(ROOT/'tokenhsi'), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               CXX='/usr/bin/g++-11', MAX_JOBS='2', TOKENHSI_LOCAL_CUDA_COMPAT='1',
               TORCH_EXTENSIONS_DIR=str(ROOT/'output_etc/check/torch_extensions'), PYTHONUNBUFFERED='1',
               CUDA_VISIBLE_DEVICES='' if args.cpu else args.gpu,
               CUDA_LAUNCH_BLOCKING=args.cuda_launch_blocking,
               TOKENHSI_EVAL_PROTOCOL='cvpr2027_single', CVPR_PROTOCOL_REVISION=args.protocol,
               NATIVE_EVAL_WARMUP='0', TOKENHSI_PUSH_INIT='loco')
    seeds = [0] if args.smoke else args.seeds
    summary = dict(protocol='cvpr2027_single_'+args.protocol, xy_fraction=1. if args.protocol == '1109' else .8,
                   checkpoint=str(checkpoint), seeds=seeds,
                   scenes_per_seed=2 if args.smoke else 512, horizon=900, non_paper_smoke=args.smoke,
                   scope='Single-person skills only; no Independent N>1, Enable, Coupled, Mixed tables', results={})
    (out/'run.json').write_text(json.dumps(summary, indent=2))
    for task in args.tasks:
        reports = []
        for seed in seeds:
            dest = out/task/f'seed_{seed}'
            dest.mkdir(parents=True, exist_ok=True)
            if (dest/'result.json').exists():
                raise RuntimeError('Refusing to overwrite existing evaluation: '+str(dest))
            env['CVPR_EVAL_OUT'] = str(dest)
            env.pop('CVPR_REPLAY_INITIAL', None)
            if args.snapshot_root:
                snapshot = Path(args.snapshot_root).resolve()/task/f'seed_{seed}'/'initial.npz'
                if not snapshot.is_file():
                    raise FileNotFoundError(snapshot)
                env['CVPR_REPLAY_INITIAL'] = str(snapshot)
            device = 'cpu' if args.cpu else 'cuda:0'
            cmd = [str(runtime/'bin/python'), 'scripts/train_local.py', '--task', 'HumanoidTokenHSIPushKD',
                   '--cfg_env', 'tokenhsi/data/cfg/multi_task/push_kd.yaml', '--cfg_train', 'tokenhsi/data/cfg/train/rlg/push_kd.yaml',
                   '--motion_file', 'tokenhsi/data/dataset_push_kd.yaml', '--test', '--eval', '--eval_task', task,
                   '--checkpoint', str(checkpoint), '--num_envs', str(summary['scenes_per_seed']), '--headless',
                   '--sim_device', device, '--rl_device', device, '--pipeline', 'cpu' if args.cpu else 'gpu', '--seed', str(seed)]
            print(f'Evaluating {task}, seed {seed}: {dest}', flush=True)
            with (dest/'eval.log').open('w') as log:
                subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            report = json.loads((dest/'result.json').read_text())
            assert json.loads((dest/'cpu_rescore.json').read_text())['verified']
            reports.append(report)
        summary['results'][task] = dict(seeds=reports, scoring_mode=reports[0]['scoring_mode'],
                                        region_definition=reports[0]['region_definition'],
                                        scene_kind=reports[0]['scene_kind'])
        for metric in ('region', 'point'):
            rates = [r[metric]['rate'] for r in reports]
            summary['results'][task][metric] = seed_statistics(rates) if not args.smoke else dict(smoke_rate=rates[0])
        summary['results'][task]['errors'] = {}
        for metric in ('point', 'region'):
            means = [r['errors'][metric]['mean_cm'] for r in reports]
            summary['results'][task]['errors'][metric] = dict(
                unit='cm', samples_per_seed=[r['errors'][metric]['samples'] for r in reports],
                **(seed_statistics(means) if not args.smoke else dict(smoke_mean_cm=means[0])))
        (out/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary['results'], indent=2), flush=True)
    print('Saved '+str(out/'summary.json'), flush=True)


if __name__ == '__main__':
    main()
