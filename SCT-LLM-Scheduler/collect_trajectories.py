"""Run bounded Stage2 recordings on a local snapshot, never on the ksh checkout."""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT/'teacher_runtime'
SETTINGS = dict(enable_sit=2, enable_climb=2, enable_ontop=2, coupled_2=2, coupled_3=3, coupled_4=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, choices=(6, 7), required=True)
    parser.add_argument('--settings', nargs='+', choices=SETTINGS, required=True)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--envs', type=int, default=16)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    if args.envs <= 0 or args.seed < 0 or not args.tag.replace('_', '').isalnum():
        parser.error('Invalid envs, seed or tag')
    folder = RUNTIME/'output_etc/check'/args.tag
    folder.mkdir(parents=True, exist_ok=False)
    snapshot = json.loads((RUNTIME/'snapshot.json').read_text())
    base_env = dict(os.environ, TOKENHSI_GPU=str(args.gpu), PYTHONDONTWRITEBYTECODE='1',
                    TORCH_EXTENSIONS_DIR=str(ROOT/'output_etc/check/torch_extensions'),
                    CUDA_CACHE_PATH=str(ROOT/'output_etc/check/cuda_cache'),
                    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for setting in args.settings:
        output = folder/setting
        output.mkdir()
        command = ['python', 'record_entry.py', '--task', 'HumanoidMACoupledPush',
            '--cfg_train', 'tokenhsi/data/cfg/train/rlg/stage2_distill_900_v3.yaml', '--cfg_env', '',
            '--motion_file', 'tokenhsi/data/dataset_stage1_jointpush.yaml',
            '--checkpoint', snapshot['checkpoint_local'], '--seed', str(args.seed),
            '--num_envs', str(args.envs), '--num_agents', str(SETTINGS[setting]),
            '--num_objects', str(2*SETTINGS[setting]), '--test', '--eval', '--headless', '--no_video',
            '--eval_protocol', 'general', '--eval_repeats', '1', '--task_graph', 'random_scenario',
            '--enable_size_mode', 'per_action_intersection', '--coupled_assignment', 'ordered',
            '--output_path', str(output)]
        env = dict(base_env, SCT_RECORD_SETTING=setting)
        (output/'launch.json').write_text(json.dumps(dict(command=command, gpu=args.gpu,
            checkpoint_sha256=snapshot['checkpoint_sha256'], envs=args.envs, seed=args.seed), indent=2))
        print('START', setting, 'GPU', args.gpu, 'log', output/'run.log', flush=True)
        with (output/'run.log').open('w') as stream:
            result = subprocess.run(['bash', '-c', 'source tokenhsi/scripts/multi_agent/runtime_env.sh || exit\nexec "$@"',
                                     '_', *command], cwd=RUNTIME, env=env, stdout=stream, stderr=subprocess.STDOUT)
        (output/'exit.json').write_text(json.dumps({'returncode': result.returncode}))
        print('END', setting, 'returncode', result.returncode, flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
