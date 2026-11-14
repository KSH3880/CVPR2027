import argparse
import concurrent.futures
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tokenhsi'))
import torch
import stage2_xy100
from run_stage2_scalability import verify

PREVIOUS = ROOT/'output_etc/check/v3_2000_tables12_seeds167_20261110'
STAGE1 = ROOT/'pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth'
V2 = ROOT/'output/stage2_distill_900_v2/Stage2Distill900V2_08-19-13-51/nn/Stage2Distill900V2_00002000.pth'
V3 = ROOT/'output/stage2_distill_900_v3/Stage2Distill900V3_09-20-41-11/nn/Stage2Distill900V3_00002000.pth'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(value, indent=2))
    pending.replace(path)


def check_run(path, reference, people, mode):
    check = verify(path, reference, people, 512)
    result = json.loads((path/'result.json').read_text())
    proof = result['bias_ablation']
    for key in ('first_policy_input_exact', 'no_ablation_action_exact', 'observation_unchanged_by_probe',
                'rng_restored_after_probe', 'model_weights_unchanged', 'normalization_unchanged'):
        if not proof[key]:
            raise ValueError('Ablation verification failed: '+key)
    native = mode in ('task', 'none')
    expected = dict(actor=904 if native else 903, critic=903)
    if result['encoder_attention_checks'] != expected or proof['mode'] != mode:
        raise ValueError('Unexpected ablation or encoder call counts')
    if result['distractor_attention'] != 'all_observed_tokens':
        raise ValueError('Observed distractor attention changed')
    before = torch.load(path/'first_policy_input.pt', map_location='cpu', weights_only=False)
    canonical = torch.load(reference/'observed_scene.pt', map_location='cpu', weights_only=False)
    if not torch.equal(before, canonical):
        raise ValueError('Actual first policy input differs from canonical observation')
    if native:
        policy = result['stage1_policy']
        if policy['coordination_calls'] or not all(policy[key] for key in ('native_actor_exact', 'weights_unchanged', 'normalization_exact')):
            raise ValueError('Stage1 baseline policy verification failed')
    check.update(bias_ablation=mode, first_policy_input_exact=True,
                 no_ablation_action_exact=True, weights_and_normalization_unchanged=True)
    return check


def run(args):
    stage2_xy100.install()
    torch.set_num_threads(1)
    out = args.output.resolve()
    if ROOT/'output_etc/check' not in out.parents:
        raise ValueError('Evaluation output must be under output_etc/check')
    out.mkdir(parents=True, exist_ok=True)
    (out/'logs').mkdir(exist_ok=True)
    state_path = out/'status.json'
    state = dict(started_utc=datetime.now(timezone.utc).isoformat(), seeds=args.seeds, envs=512,
                 horizon=900, workers_per_gpu=args.workers_per_gpu, expected_runs=30,
                 verified=[], running={}, failed=[], completed=False)
    lock, cpu_lock = threading.Lock(), threading.Lock()
    hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in (STAGE1, V2, V3)}
    previous_hashes = {}
    references = {}
    jobs = [('task', n, seed) for n in (16, 8, 4, 2, 1) for seed in args.seeds]
    jobs += [('dependency', n, seed) for n in (16, 8, 4, 2) for seed in args.seeds]
    jobs += [('none', 1, seed) for seed in args.seeds]
    # Interleave the two stages so both ablations begin immediately.
    jobs.sort(key=lambda j: (-j[1], j[2], j[0]))
    for mode, people, seed in jobs:
        scene = 'mixed' if mode == 'dependency' else 'independent'
        name = scene+'_'+str(people)+'h'
        reference = out/'reference'/name/f'general_seed{seed}'
        reference.parent.mkdir(parents=True, exist_ok=True)
        origin = PREVIOUS/'reference/table1'/name/reference.name
        if (origin/'reference_info.json').exists() and not reference.exists():
            reference.symlink_to(origin.resolve(), target_is_directory=True)
        references[(mode, people, seed)] = reference
        if people > 1:
            model = 'ours' if mode == 'dependency' else 'stage1'
            baseline = PREVIOUS/'table1'/model/name/f'general_seed{seed}'
            for filename in ('result.json', 'initial.pt', 'physics_only_reset.pt'):
                path = (baseline/filename).resolve()
                previous_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    write(out/'source_hashes.json', dict(checkpoints=hashes, baseline_files=previous_hashes))
    write(out/'manifest.json', dict(seeds=args.seeds, envs=512, horizon=900, xy_extent_percent=100,
        objects='2N', stage1_checkpoint=str(STAGE1.resolve()), stage2_checkpoint=str(V3),
        carrier_checkpoint=str(V2), inference_only=True, expected_runs=30, completed=False))

    def save():
        write(state_path, state)

    def execute(job, gpu):
        mode, people, seed = job
        scene = 'mixed' if mode == 'dependency' else 'independent'
        name = scene+'_'+str(people)+'h'
        folder = out/mode/name
        path = folder/f'general_seed{seed}'
        reference = references[job]
        proof_path = folder/f'verification_seed{seed}.json'
        key = f'{mode}/{people}/{seed}'
        with lock:
            state['running'][key] = dict(mode=mode, agents=people, seed=seed, gpu=gpu,
                started_utc=datetime.now(timezone.utc).isoformat())
            save()
        if not proof_path.exists():
            checkpoint = V3 if mode == 'dependency' else V2
            config = 'stage2_distill_900_v3' if mode == 'dependency' else 'stage2_distill_900_v2'
            command = [sys.executable, str(ROOT/'eval/bias_ablation_eval.py'),
                '--task', 'HumanoidMACoupledPush', '--cfg_train', f'tokenhsi/data/cfg/train/rlg/{config}.yaml',
                '--cfg_env', '', '--motion_file', 'tokenhsi/data/dataset_stage1_jointpush.yaml',
                '--checkpoint', str(checkpoint), '--seed', str(seed), '--num_envs', '512',
                '--num_agents', str(people), '--num_objects', str(2*people), '--test', '--eval',
                '--headless', '--no_video', '--eval_protocol', 'general', '--eval_repeats', '1',
                '--task_graph', 'random_scenario', '--enable_size_mode', 'per_action_intersection',
                '--coupled_assignment', 'ordered']
            if people > 4:
                command.append('--group_spawn_over4')
            env = dict(os.environ, TOKENHSI_GPU=str(gpu), STAGE2_SCENE_MODE=scene, BIAS_ABLATION=mode,
                STAGE2_REFERENCE_ROOT=str(reference.parent), PYTHONHASHSEED=str(seed),
                OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
            for key_env in ('REFERENCE_ONLY', 'STAGE1_POLICY_CHECKPOINT', 'BIAS_ABLATION_PROBE_ONLY',
                            'TABLE2_SETTING', 'FSM_STAGE1_CHECKPOINT', 'CHECK', 'MAX_ITERATIONS'):
                env.pop(key_env, None)
            with (reference.parent/f'.seed{seed}.lock').open('a') as ref_lock:
                fcntl.flock(ref_lock, fcntl.LOCK_EX)
                if not (reference/'reference_info.json').exists():
                    reference.mkdir(parents=True, exist_ok=True)
                    with (out/'logs'/f'{name}_seed{seed}_reference.log').open('w') as log:
                        subprocess.run(['bash', '-c', 'source tokenhsi/scripts/multi_agent/runtime_env.sh\nexec "$@"',
                            '_', *command, '--output_path', str(reference)], cwd=ROOT,
                            env=dict(env, REFERENCE_ONLY='1'), stdout=log, stderr=subprocess.STDOUT, check=True)
            if not (path/'bias_ablation.json').exists():
                if path.exists() and any(path.iterdir()):
                    raise ValueError('Incomplete output exists: '+str(path))
                path.mkdir(parents=True, exist_ok=True)
                if mode in ('task', 'none'):
                    env['STAGE1_POLICY_CHECKPOINT'] = str(STAGE1.resolve())
                with (out/'logs'/f'{mode}_{people}h_seed{seed}.log').open('w') as log:
                    subprocess.run(['bash', '-c', 'source tokenhsi/scripts/multi_agent/runtime_env.sh\nexec "$@"',
                        '_', *command, '--output_path', str(path)], cwd=ROOT, env=env,
                        stdout=log, stderr=subprocess.STDOUT, check=True)
            with cpu_lock:
                proof = check_run(path, reference, people, mode)
                write(proof_path, proof)
        with lock:
            state['verified'].append(dict(mode=mode, agents=people, seed=seed, gpu=gpu))
            state['running'].pop(key, None)
            save()
        print(f'GPU{gpu} {mode} {people}H seed{seed} verified', flush=True)

    def worker(gpu, subset):
        for job in subset:
            try:
                execute(job, gpu)
            except Exception:
                error = traceback.format_exc()
                print(error, flush=True)
                with lock:
                    state['failed'].append(dict(job=job, gpu=gpu, error=error))
                    state['running'].pop(f'{job[0]}/{job[1]}/{job[2]}', None)
                    save()

    save()
    count = 2*args.workers_per_gpu
    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
        futures = [pool.submit(worker, 6+index%2, jobs[index::count]) for index in range(count)]
        for future in futures:
            future.result()
    for path, digest in {**hashes, **previous_hashes}.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError('Checkpoint or baseline artifact changed: '+path)
    state.update(completed=not state['failed'] and len(state['verified']) == 30,
        source_hashes_unchanged=True, finished_utc=datetime.now(timezone.utc).isoformat())
    save()
    subprocess.run([sys.executable, str(ROOT/'eval/summarize_bias_ablation.py'), str(out)], cwd=ROOT, check=True)
    if not state['completed']:
        raise SystemExit('Some bias ablations failed; see status.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs=3, default=[1, 6, 7])
    parser.add_argument('--workers-per-gpu', type=int, default=5)
    args = parser.parse_args()
    if len(set(args.seeds)) != 3 or min(args.seeds) < 0 or not 1 <= args.workers_per_gpu <= 5:
        parser.error('Use three distinct nonnegative seeds and 1..5 workers per GPU')
    run(args)
