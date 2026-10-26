"""Run two paired, single-person PUSH start conditions concurrently."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import numpy as np
from utils.push_start_evaluation import MODES, make_start_plan
from utils.finetune_evaluation import atomic_json, summarize_rows, distance_bins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--reference-checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--num-envs', type=int, default=512)
    parser.add_argument('--horizon', type=int, default=1800)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root/'run_manifest.json').exists():
        raise FileExistsError('Use a new output directory')
    manifest = dict(status='preparing', checkpoint=str(Path(args.checkpoint).resolve()),
                    modes=list(MODES), num_envs=args.num_envs, horizon=args.horizon,
                    seeds=[0, 1, 2], started=datetime.now().isoformat(), planned_policy_runs=6)
    atomic_json(root/'run_manifest.json', manifest)

    def common(seed, plan):
        return [sys.executable, 'evaluate_single_match.py', '--benchmark-task', 'push_at',
                '--reference-checkpoint', args.reference_checkpoint, '--initial-plan', str(plan),
                '--task', 'HumanoidMACarry', '--num_agents', '1', '--num_objects', '1',
                '--num_envs', str(args.num_envs), '--episode_length', str(args.horizon),
                '--headless', '--test', '--eval', '--no_video', '--seed', str(seed),
                '--motion_file', 'tokenhsi/data/dataset_loco_sit_carry_climb.yaml']

    # Generate paired scenes on CPU before launching either native CUDA process.
    for seed in (0, 1, 2):
        directory = root/'initial_states'
        directory.mkdir(exist_ok=True)
        plan = directory/f'base_seed{seed}.npz'
        command = common(seed, plan) + ['--prepare-only', '--sim_device', 'cpu', '--pipeline', 'cpu',
                  '--rl_device', 'cpu', '--num_threads', '1', '--output_path', str(directory)]
        cpu_env = os.environ.copy()
        cpu_env.update(CUDA_VISIBLE_DEVICES='-1', TOKENHSI_GPU='-1')
        print('[prepare]', seed, flush=True)
        with plan.with_suffix('.log').open('w') as log:
            code = subprocess.run(command, env=cpu_env, stdout=log, stderr=subprocess.STDOUT).returncode
        if code:
            manifest.update(status='failed', failed_stage='prepare', seed=seed, exit_code=code)
            atomic_json(root/'run_manifest.json', manifest)
            raise RuntimeError(f'CPU initial state preparation failed: {plan.with_suffix(".log")}')
        with np.load(plan) as archive:
            base = {key: archive[key].copy() for key in archive.files}
        for mode in MODES:
            np.savez_compressed(directory/f'{mode}_seed{seed}.npz', **make_start_plan(base, mode, seed+1729))
    manifest['status'] = 'running'
    atomic_json(root/'run_manifest.json', manifest)
    failed = threading.Event()

    def worker(mode):
        runs = []
        folder = root/mode
        folder.mkdir(exist_ok=True)
        for seed in (0, 1, 2):
            if failed.is_set():
                return False
            out = folder/f'push_at_seed{seed}'
            out.mkdir()
            command = common(seed, root/'initial_states'/f'{mode}_seed{seed}.npz') + [
                '--push-start-mode', mode, '--checkpoint', args.checkpoint, '--output_path', str(out),
                '--sim_device', 'cuda:0', '--pipeline', 'gpu', '--rl_device', 'cuda:0']
            print('[start]', mode, seed, flush=True)
            with (out/'eval.log').open('w') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                atomic_json(folder/'status.json', dict(status='running', seed=seed, pid=process.pid, command=command))
                code = process.wait()
            if code:
                failed.set()
                atomic_json(folder/'status.json', dict(status='failed', seed=seed, exit_code=code))
                return False
            runs.append(json.loads((out/'results.json').read_text()))
            rows = [row for run in runs for row in run['agents']]
            rates = [run['summary']['point_ever_rate'] for run in runs]
            atomic_json(folder/'summary.json', dict(mode=mode, completed_seeds=[run['seed'] for run in runs],
                pooled=summarize_rows(rows), point_ever_mean=float(np.mean(rates)),
                point_ever_std_ddof1=float(np.std(rates, ddof=1)) if len(rates)>1 else None,
                by_initial_box_goal_distance=distance_bins(rows)))
            print('[complete]', mode, seed, flush=True)
        atomic_json(folder/'status.json', dict(status='complete'))
        return True

    with ThreadPoolExecutor(max_workers=2) as pool:
        success = list(pool.map(worker, MODES))
    manifest.update(status='complete' if all(success) else 'failed', finished=datetime.now().isoformat())
    atomic_json(root/'run_manifest.json', manifest)
    if not all(success):
        raise SystemExit('Evaluation failed; no automatic GPU retry')


if __name__ == '__main__':
    main()
