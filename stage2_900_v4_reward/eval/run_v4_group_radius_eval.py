import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from run_stage2_scalability import ROOT, verify
import stage2_xy100


OUTPUT = ROOT/'output_etc/check/v4_1500_2000_group_radius_env64_seed1_20261112'
BASELINE = ROOT/'output_etc/check/table1_group_radius_8r3_16r7p5_env64_seed1_20261112'
RUN = ROOT/'output/stage2_distill_900_v4/Stage2Distill900V4_11-21-37-16'
GPUS = (5, 6, 7)
RADII = {8: 3., 16: 7.5}
EPOCHS = (1500, 2000)


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def destination(job):
    return OUTPUT/str(job['epoch'])/job['name']/'general_seed1'


def validate(job):
    path = destination(job)
    reference = OUTPUT/'reference'/job['name']/'general_seed1'
    proof = verify(path, reference, job['people'], 64)
    result = read(path/'result.json')
    policy = read(path/'policy_preservation.json')
    if not read(path/'first_policy_input.json')['exact']:
        raise ValueError('Actual first policy input differs')
    if policy['epoch'] != job['epoch'] or not all(policy[k] for k in
            ('weights_exact_before_after', 'normalization_exact_before_after')):
        raise ValueError('Checkpoint preservation failed')
    steps = result['simulated_steps']
    if result['encoder_attention_checks'] != dict(actor=steps, critic=steps):
        raise ValueError('Distractor attention check failed')
    if result['seed'] != 1 or result['enabling_success'] != 'simultaneous':
        raise ValueError('Seed or success definition differs')
    spacing = read(path/'group_spacing.json')
    if (spacing['group_placement'], spacing['group_spawn_radius'], spacing['group_spawn_gap']) != (
            'fixed_radius_group_center_ring', job['radius'], None):
        raise ValueError('Wrong group placement')
    initial = torch.load(path/'initial.pt', map_location='cpu', weights_only=False)
    groups = initial['groups'].clone()
    rows, followers = (initial['predecessor'] >= 0).nonzero(as_tuple=True)
    groups[rows, followers] = groups[rows, initial['predecessor'][rows, followers]]
    same = groups[:, :, None] == groups[:, None, :]
    sizes = same.sum(-1, keepdim=True)
    centers = torch.bmm(same.float(), initial['roots'][..., :2])/sizes
    counts = (1/sizes.squeeze(-1).float()).sum(-1)
    origin = (centers/sizes).sum(1)/counts[:, None]
    radius_error = float(((centers-origin[:, None]).norm(dim=-1)-job['radius']).abs().max())
    gap = 2*job['radius']*math.sin(math.pi/int(counts.round().max()))
    if radius_error > .002 or abs(spacing['minimum_initial_group_gap']-gap) > .002:
        raise ValueError('Actual group ring geometry differs')
    result['comparison_protocol'] = 'cvpr2027_table1_fixed_group_radius_env64'
    write(path/'result.json', result)
    proof.update(job=job, actual_first_input_exact=True, policy_preserved=True,
        group_radius_verified=True, maximum_radius_error_m=radius_error,
        group_ring_analytic_gap_verified=True)
    write(path/'verification.json', proof)
    return proof


def launch(job, gpu):
    path = destination(job)
    path.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TOKENHSI_GPU=str(gpu), SPACING_MODEL='ours', SPACING_METERS='native',
        GROUP_SPAWN_RADIUS=str(job['radius']), STAGE2_SCENE_MODE=job['mode'],
        STAGE2_REFERENCE_ROOT=str(OUTPUT/'reference'/job['name']), PYTHONHASHSEED='1',
        PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    for key in ('REFERENCE_ONLY', 'TABLE2_SETTING', 'STAGE1_POLICY_CHECKPOINT',
                'FSM_STAGE1_CHECKPOINT', 'CHECK', 'MAX_ITERATIONS', 'RESUME_CHECKPOINT',
                'STAGE2_SAMPLING_AUDIT', 'BIAS_ABLATION'):
        env.pop(key, None)
    command = [sys.executable, str(ROOT/'eval/group_spacing_eval.py'), '--task', 'HumanoidMACoupledPush',
        '--cfg_train', 'tokenhsi/data/cfg/train/rlg/stage2_distill_900_v4.yaml', '--cfg_env', '',
        '--motion_file', 'tokenhsi/data/dataset_stage1_jointpush.yaml', '--checkpoint', job['checkpoint'],
        '--seed', '1', '--num_envs', '64', '--num_agents', str(job['people']),
        '--num_objects', str(2*job['people']), '--test', '--eval', '--headless', '--no_video',
        '--eval_protocol', 'general', '--eval_repeats', '1', '--task_graph', 'random_scenario',
        '--enable_size_mode', 'per_action_intersection', '--coupled_assignment', 'ordered',
        '--group_spawn_over4', '--output_path', str(path)]
    log_path = OUTPUT/'logs'/f"{job['epoch']}_{job['name']}.log"
    with log_path.open('w') as log:
        process = subprocess.Popen(['bash', '-c',
            'source tokenhsi/scripts/multi_agent/runtime_env.sh\nexec "$@"', '_', *command],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    return process


def report(jobs, completed):
    rows = []
    lines = ['# V4 1.5k / 2k · 그룹 중심 고정 반경 비교', '',
        f'64환경·seed1·900step·XY100%·M=2N. 8H R=3m /16H R=7.5m. 검증 {len(completed)}/8회.', '',
        '| Checkpoint | 8H Independent | 8H Mixed | 16H Independent | 16H Mixed |',
        '|---|---:|---:|---:|---:|']
    scores = {}
    for epoch in EPOCHS:
        cells = []
        scores[str(epoch)] = {}
        for job in (j for j in jobs if j['epoch'] == epoch):
            if job['key'] not in completed:
                cells.append('—')
                continue
            result = read(destination(job)/'result.json')
            values = result['interaction_group_scores']['region']
            metric = ('Independent Interaction Success Rate' if job['mode'] == 'independent'
                      else 'Mixed Interaction Group Success Rate')
            value = 100*values[metric]['rate']
            cells.append(f'{value:.2f}')
            scores[str(epoch)][job['name']] = value
            rows.append(dict(job, scores=values, errors=result['errors'], source=str(destination(job))))
        lines.append(f'| V4 {epoch/1000:g}k | '+' | '.join(cells)+' |')
    if len(completed) == 8:
        names = [j['name'] for j in jobs if j['epoch'] == 1500]
        lines.append('| 2k − 1.5k (pp) | '+' | '.join(
            f"{scores['2000'][name]-scores['1500'][name]:+.2f}" for name in names)+' |')
    lines += ['', '단위%. 동일 초기 장면을 사용한 단일 seed 탐색 평가이며 표준편차를 계산하지 않는다.',
              '기존 5m 최소 간격 결과와 구분한다. Mixed는 I/E/C 유형별 SR의 동일 가중 평균이다.']
    text = '\n'.join(lines)+'\n'
    (OUTPUT/'partial_report.md').write_text(text)
    if len(completed) == 8:
        (OUTPUT/'report.md').write_text(text)
    write(OUTPUT/'summary.json', dict(completed=len(completed) == 8, envs=64, seeds=[1],
        horizon=900, group_spawn_radii_m=RADII, results=scores, runs=rows))


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU supervisor requires CUDA_VISIBLE_DEVICES=')
    torch.set_num_threads(1)
    stage2_xy100.install()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT/'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (OUTPUT/'supervisor.pid').write_text(str(os.getpid())+'\n')
    (OUTPUT/'logs').mkdir(exist_ok=True)
    jobs, hashes, checkpoints = [], {}, {}
    for epoch in EPOCHS:
        checkpoint = RUN/'nn'/f'Stage2Distill900V4_{epoch:08d}.pth'
        weights = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if weights['epoch'] != epoch or not all(torch.isfinite(t).all() for t in weights['model'].values()):
            raise ValueError('Invalid checkpoint')
        checkpoints[str(epoch)] = dict(path=str(checkpoint), sha256=sha(checkpoint))
        hashes[str(checkpoint)] = checkpoints[str(epoch)]['sha256']
        del weights
        for people in (8, 16):
            for mode in ('independent', 'mixed'):
                name = f'{mode}_{people}h'
                jobs.append(dict(epoch=epoch, people=people, mode=mode, name=name, radius=RADII[people],
                    seed=1, checkpoint=str(checkpoint), key=f'{epoch}/{name}'))
    for job in jobs[:4]:
        source = BASELINE/'reference'/job['name']/'general_seed1'
        target = OUTPUT/'reference'/job['name']/'general_seed1'
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.symlink_to(source.resolve(), target_is_directory=True)
        for name in ('reference_info.json', 'initial.pt', 'observed_scene.pt', 'group_spacing.json'):
            hashes[str(source/name)] = sha(source/name)
    for name in ('run_v4_group_radius_eval.py', 'group_spacing_eval.py', 'run_stage2_scalability.py',
                 'stage2_protocol_eval.py', 'stage2_scalability_eval.py', 'stage2_scalability_layout.py',
                 'stage2_protocol_sampling.py', 'stage2_distractor_physics.py', 'stage2_xy100.py'):
        hashes[str(ROOT/'eval'/name)] = sha(ROOT/'eval'/name)
    previous = OUTPUT/'source_hashes.json'
    if previous.exists() and read(previous) != hashes:
        raise ValueError('Sources changed; use a new output folder')
    write(previous, hashes)
    manifest = dict(protocol='table1_fixed_group_radius_pilot', envs=64, seeds=[1], horizon=900,
        xy_extent_percent=100, objects='2N', group_spawn_radii_m=RADII, gpus=GPUS,
        checkpoints=checkpoints, reference_suite=str(BASELINE), jobs=jobs, completed=False)
    write(OUTPUT/'manifest.json', manifest)
    state = dict(completed=False, expected_runs=8, verified=[], running={}, failed=[],
        pending=8, gpus=GPUS, workers_per_gpu=3, reserve_mib=10000, started_unix=time.time())
    pending, running = jobs.copy(), {}
    for job in jobs:
        if (destination(job)/'verification.json').exists():
            validate(job)
            state['verified'].append(job['key'])
            pending.remove(job)
    while pending or running:
        free = subprocess.check_output(['nvidia-smi', '-i', '5,6,7', '--query-gpu=index,memory.free',
            '--format=csv,noheader,nounits'], text=True)
        available = {int(line.split(',')[0]): int(line.split(',')[1]) for line in free.splitlines()}
        for gpu in sorted(GPUS, key=lambda g: sum(x[2] == g for x in running.values())):
            active = sum(x[2] == gpu for x in running.values())
            if pending and not state['failed'] and active < 3 and available[gpu]-active*8000 >= 18000:
                job = pending.pop(0)
                process = launch(job, gpu)
                running[job['key']] = process, job, gpu
                state['running'][job['key']] = dict(pid=process.pid, gpu=gpu, started_unix=time.time())
                print('START', job['key'], 'GPU'+str(gpu), 'PID', process.pid, flush=True)
        for key, (process, job, gpu) in list(running.items()):
            if process.poll() is None:
                continue
            try:
                if process.returncode:
                    raise RuntimeError(f'Evaluation exited {process.returncode}; see logs/{job["epoch"]}_{job["name"]}.log')
                validate(job)
                state['verified'].append(key)
                print('VERIFIED', key, 'GPU'+str(gpu), flush=True)
            except Exception as error:
                state['failed'].append(dict(key=key, error=str(error)))
                print('FAILED', key, str(error), flush=True)
            del running[key]
            del state['running'][key]
        state.update(pending=len(pending), updated_unix=time.time())
        write(OUTPUT/'status.json', state)
        report(jobs, state['verified'])
        if state['failed'] and not running:
            raise RuntimeError('Queue stopped after failure')
        if pending or running:
            time.sleep(5)
    if any(sha(Path(path)) != digest for path, digest in hashes.items()):
        raise ValueError('Checkpoint, source, or reference changed')
    write(OUTPUT/'completion_verification.json', dict(completed=True, runs=8, reused_references=4,
        envs=64, seeds=[1], gpus=GPUS, same_initial_scenes_and_actual_first_input=True,
        actual_group_radii_verified=True, cpu_sr_counts_errors_verified=True,
        policy_weights_and_normalization_preserved=True, source_and_checkpoint_hashes_unchanged=True))
    manifest['completed'] = state['completed'] = True
    write(OUTPUT/'manifest.json', manifest)
    write(OUTPUT/'status.json', state)


if __name__ == '__main__':
    main()
