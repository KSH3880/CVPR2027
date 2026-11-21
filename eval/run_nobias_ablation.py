import argparse
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from run_v4_formal import ROOT, OUTPUT as BASELINE, jobs as baseline_jobs
from run_stage2_scalability import verify
from run_v4_group_radius_eval import read, write, sha
import stage2_xy100
import torch

EXPERIMENT = 'stage2_distill_900_v4_nobias'
RUN = ROOT/'output/stage2_distill_900_v4_nobias/Stage2Distill900V4NoBias_15-22-05-35'
CHECKPOINT = RUN/'nn/Stage2Distill900V4NoBias_00002000.pth'
OUTPUT = ROOT/'output_etc/check/depbias_retrained_nobias_2000_mixed_env512_seeds167_20261115'
GPU = 6
LIMIT = 85000
RESERVATIONS = {2: 6500, 4: 8000, 8: 11000, 16: 15000}
METRIC = 'Mixed Interaction Group Success Rate'


def jobs():
    return [dict(job, key=f"mixed_{job['people']}h_seed{job['seed']}")
            for job in baseline_jobs() if job['table'] == 'table1' and job['mode'] == 'mixed']


def destination(job):
    return OUTPUT/job['name']/f"general_seed{job['seed']}"


def command(job):
    result = [sys.executable, str(ROOT/'eval/nobias_ablation_eval.py'), '--task', 'HumanoidMACoupledPush',
        '--cfg_train', f'tokenhsi/data/cfg/train/rlg/{EXPERIMENT}.yaml', '--cfg_env', '',
        '--motion_file', 'tokenhsi/data/dataset_stage1_jointpush.yaml', '--checkpoint', str(CHECKPOINT),
        '--seed', str(job['seed']), '--num_envs', '512', '--num_agents', str(job['people']),
        '--num_objects', str(2*job['people']), '--test', '--eval', '--headless', '--no_video',
        '--eval_protocol', 'general', '--eval_repeats', '1', '--task_graph', 'random_scenario',
        '--enable_size_mode', 'per_action_intersection', '--coupled_assignment', 'ordered',
        '--output_path', str(destination(job))]
    if job['people'] > 4:
        result.append('--group_spawn_over4')
    return result


def memory():
    row = subprocess.check_output(['nvidia-smi', '-i', str(GPU), '--query-gpu=uuid,memory.used',
        '--format=csv,noheader,nounits'], text=True).strip().split(',')
    uuid, used = row[0].strip(), int(row[1])
    processes = {}
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
        '--format=csv,noheader,nounits'], text=True)
    for line in output.splitlines():
        gpu, pid, amount = [value.strip() for value in line.split(',')]
        if gpu == uuid and amount.isdigit():
            processes[int(pid)] = int(amount)
    return used, processes


def reservation(job, actual, ready):
    return 512 if ready else max(RESERVATIONS[job['people']]-actual, 0)


def launch(job):
    destination(job).mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    for key in ('CUDA_MPS_PIPE_DIRECTORY', 'CUDA_MPS_LOG_DIRECTORY', 'REFERENCE_ONLY',
                'TABLE2_SETTING', 'STAGE1_POLICY_CHECKPOINT', 'FSM_STAGE1_CHECKPOINT',
                'CHECK', 'MAX_ITERATIONS', 'RESUME_CHECKPOINT', 'ENV_CONFIG', 'STAGE1_CHECKPOINT',
                'STAGE2_SAMPLING_AUDIT', 'BIAS_ABLATION', 'DEPBIAS_MODE', 'SPACING_MODEL',
                'SPACING_METERS', 'GROUP_SPAWN_RADIUS'):
        env.pop(key, None)
    env.update(TOKENHSI_GPU=str(GPU), STAGE2_SCENE_MODE='mixed',
        STAGE2_REFERENCE_ROOT=str(Path(job['reference']).parent), PYTHONHASHSEED=str(job['seed']),
        PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    with (OUTPUT/'logs'/f"{job['key']}.log").open('w') as log:
        return subprocess.Popen(['bash', '-c',
            'source tokenhsi/scripts/multi_agent/runtime_env.sh\nexec "$@"', '_', *command(job)],
            cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)


def validate(job):
    path = destination(job)
    proof = verify(path, Path(job['reference']), job['people'], 512)
    result = read(path/'result.json')
    if (result['experiment'], result['seed'], result['scene_mode']) != (EXPERIMENT, job['seed'], 'mixed'):
        raise ValueError('Wrong student, seed or scene mode')
    if result['encoder_attention_checks'] != dict(actor=result['simulated_steps'], critic=result['simulated_steps']):
        raise ValueError('Observed distractor attention differs')
    if result['distractor_attention'] != 'all_observed_tokens' or result['enabling_success'] != 'simultaneous':
        raise ValueError('Evaluation protocol differs')
    policy = read(path/'policy_preservation.json')
    if policy['epoch'] != 2000 or not all(policy[key] for key in
            ('weights_exact_before_after', 'normalization_exact_before_after')):
        raise ValueError('Policy or normalization changed')
    if not read(path/'first_policy_input.json')['exact']:
        raise ValueError('Actual first input differs')
    bias = read(path/'nobias_verification.json')
    if bias['dependency_projection_parameters'] != 0 or not bias['trained_without_dependency_bias']:
        raise ValueError('Dependency bias was not removed during training')
    if not bias['dependency_fusion_preserved'] or not bias['sct_task_bias_preserved']:
        raise ValueError('Unexpected fusion or SCT change')
    if job['radius'] is not None:
        if result['group_spawn_radius'] != job['radius'] or result['group_spawn_gap'] is not None:
            raise ValueError('Group radius configuration differs')
        initial = torch.load(path/'initial.pt', map_location='cpu', weights_only=False)
        groups = initial['groups'].clone()
        rows, followers = (initial['predecessor'] >= 0).nonzero(as_tuple=True)
        groups[rows, followers] = groups[rows, initial['predecessor'][rows, followers]]
        same = groups[:, :, None] == groups[:, None, :]
        sizes = same.sum(-1, keepdim=True)
        centers = torch.bmm(same.float(), initial['roots'][..., :2])/sizes
        counts = (1/sizes.squeeze(-1).float()).sum(-1)
        origin = (centers/sizes).sum(1)/counts[:, None]
        error = float(((centers-origin[:, None]).norm(dim=-1)-job['radius']).abs().max())
        if error > .002:
            raise ValueError('Actual group radius differs')
        proof.update(group_radius_verified=True, maximum_radius_error_m=error)
    proof.update(first_policy_input_exact=True, policy_preserved=True,
        no_dependency_bias=True, reference=job['reference'])
    write(OUTPUT/'checks'/f"{job['key']}.json", proof)


def report(state):
    rows, runs = {}, []
    lines = ['# Component ablation · retrained w/o Dep. Bias', '',
        f"Mixed·512환경·seed1/6/7·900step. 검증 {len(state['verified'])}/12회.", '',
        '| Method | 2-agent SR | 4-agent SR | 8-agent SR | 16-agent SR |', '|---|---:|---:|---:|---:|']
    for label in ('Ours', 'w/o Dep. Bias'):
        cells, rows[label] = [], {}
        for people in (2, 4, 8, 16):
            values = []
            for job in (job for job in jobs() if job['people'] == people):
                if label != 'Ours' and job['key'] not in state['verified']:
                    continue
                path = BASELINE/'table1'/job['name']/f"general_seed{job['seed']}" if label == 'Ours' else destination(job)
                score = read(path/'result.json')['interaction_group_scores']['region'][METRIC]['rate']*100
                values.append(score)
                if label != 'Ours':
                    runs.append(dict(job, mixed_sr_percent=score, source=str(path)))
            if len(values) == 3:
                rows[label][str(people)] = dict(mean=statistics.mean(values), std=statistics.stdev(values), seeds=values)
                cells.append(f'{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}')
            else:
                cells.append(f'대기 {len(values)}/3')
        lines.append('| '+label+' | '+' | '.join(cells)+' |')
    lines += ['', 'Mixed는 장면 내 I/E/C SR의 동일 가중 평균이며 seed별 계산 후 평균±표본std(ddof=1)다.',
        'w/o Dep. Bias는 두 nobias 10k teacher로 재학습한 student 2k다. 기존 Ours V4 2k는 Enable34k/Coupled30k teacher를 사용한다.']
    (OUTPUT/'report.md').write_text('\n'.join(lines)+'\n')
    write(OUTPUT/'summary.json', dict(completed=state['completed'], tables=rows, runs=runs))


def main():
    global OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    OUTPUT = args.output.resolve()
    if ROOT/'output_etc/check' not in OUTPUT.parents:
        raise ValueError('Evaluation output must be below output_etc/check')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU supervisor requires CUDA_VISIBLE_DEVICES=')
    torch.set_num_threads(1)
    stage2_xy100.install()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT/'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUTPUT/'manifest.json').exists():
        raise ValueError('Use a new output directory; existing evaluation is preserved')
    for name in ('logs', 'checks'):
        (OUTPUT/name).mkdir(exist_ok=True)
    all_jobs = jobs()
    references = {}
    for job in all_jobs:
        path = Path(job['reference'])
        info = read(path/'reference_info.json')
        if (info['envs'], info['agents'], info['objects'], info['seed']) != (512, job['people'], 2*job['people'], job['seed']):
            raise ValueError('Reference dimensions differ')
        for name in ('reference_info.json', 'initial.pt', 'observed_scene.pt'):
            references[str(path/name)] = sha(path/name)
    sources = [Path(__file__), ROOT/'eval/nobias_ablation_eval.py', ROOT/'eval/v4_formal_eval.py',
        ROOT/'eval/stage2_scalability_eval.py', ROOT/'eval/stage2_protocol_sampling.py',
        ROOT/'eval/stage2_scalability_layout.py', ROOT/'eval/stage2_distractor_physics.py',
        ROOT/f'tokenhsi/data/cfg/train/rlg/{EXPERIMENT}.yaml',
        ROOT/f'tokenhsi/data/cfg/multi_agent/{EXPERIMENT}.yaml']
    sources += [ROOT/'tokenhsi/learning/multi_agent'/name for name in
        ('human_coordination.py', 'stage2_push_network.py', 'stage2_relation_fusion.py', 'coupled_relation_fusion.py', 'formal_eval.py')]
    hashes = {str(path): sha(path) for path in sources}
    manifest = dict(protocol_id='cvpr2027-evaluation-v1.1.0', experiment=EXPERIMENT,
        checkpoint=str(CHECKPOINT), checkpoint_epoch=2000, gpu=GPU, memory_limit_mib=LIMIT,
        startup_reservations_mib=RESERVATIONS, ready_worker_margin_mib=512,
        expansion_gate='first_16h_actual_policy_input_and_no_bias_verified',
        envs=512, seeds=[1, 6, 7], horizon=900, xy_extent_percent=100, mode='mixed',
        jobs=all_jobs, sources=hashes, references=references, completed=False)
    if args.prepare_only:
        write(OUTPUT/'preparation.json', dict(manifest, commands=[command(job) for job in all_jobs]))
        return
    write(OUTPUT/'manifest.json', manifest)
    (OUTPUT/'supervisor.pid').write_text(str(os.getpid())+'\n')
    state = dict(phase='waiting_checkpoint', completed=False, expected_runs=12, verified=[],
        running={}, failed=[], pending=12, started_unix=time.time())

    def save():
        state['updated_unix'] = time.time()
        write(OUTPUT/'status.json', state)
        report(state)

    save()
    previous_size = None
    while True:
        size = CHECKPOINT.stat().st_size if CHECKPOINT.exists() else None
        if size and size == previous_size:
            break
        previous_size = size
        time.sleep(5)
    weights = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    graph = weights['relation_experiment_config']['env']['relationGraph']
    if weights['epoch'] != 2000 or graph.get('human_relation_bias') is not False:
        raise ValueError('Expected a trained no-bias epoch2000 checkpoint')
    if any('coordination.blocks' in key and 'bias_projection' in key for key in weights['model']):
        raise ValueError('Student contains dependency bias parameters')
    if not all(torch.isfinite(value).all() for value in weights['model'].values()):
        raise ValueError('Non-finite model parameters')
    manifest['checkpoint_sha256'] = sha(CHECKPOINT)
    manifest['teacher_contract'] = weights['distillation_contract']
    del weights
    write(OUTPUT/'manifest.json', manifest)
    state['phase'] = 'evaluation'
    pending = sorted(all_jobs, key=lambda job: -job['people'])
    running = {}
    while pending or running:
        used, processes = memory()
        state['gpu_used_mib'] = used
        reserved = sum(reservation(job, processes.get(process.pid, 0),
            (destination(job)/'evaluation_ready.json').exists()) for process, job in running.values())
        while pending and not state['failed']:
            if running and not state['verified'] and not any(
                    (destination(job)/'evaluation_ready.json').exists() for _, job in running.values()):
                break
            job = next((job for job in pending if used+reserved+RESERVATIONS[job['people']]+1024 <= LIMIT), None)
            if job is None:
                break
            process = launch(job)
            running[job['key']] = process, job
            pending.remove(job)
            reserved += RESERVATIONS[job['people']]
            state['running'][job['key']] = dict(pid=process.pid, gpu=GPU, started_unix=time.time())
            print('START', job['key'], 'PID', process.pid, flush=True)
        for key, (process, job) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            try:
                if code:
                    raise RuntimeError(f'Worker exited {code}: logs/{key}.log')
                validate(job)
                state['verified'].append(key)
                print('VERIFIED', key, flush=True)
            except Exception as error:
                state['failed'].append(dict(key=key, error=str(error)))
                print('FAILED', key, str(error), flush=True)
            del running[key]
            del state['running'][key]
        state['pending'] = len(pending)
        state['reserved_mib'] = reserved
        save()
        if state['failed'] and not running:
            state['phase'] = 'failed'
            save()
            raise RuntimeError('Queue stopped after failure; artifacts preserved')
        if pending or running:
            time.sleep(5)
    if any(sha(Path(path)) != digest for path, digest in {**hashes, **references, str(CHECKPOINT):manifest['checkpoint_sha256']}.items()):
        raise ValueError('Source, reference or checkpoint changed')
    state.update(phase='complete', completed=True)
    manifest['completed'] = True
    write(OUTPUT/'manifest.json', manifest)
    write(OUTPUT/'completion_verification.json', dict(completed=True, runs=12, gpu=GPU,
        checkpoint_epoch=2000, protocol_id=manifest['protocol_id'],
        first_policy_inputs_exact=True, policy_normalization_preserved=True,
        cpu_sr_counts_errors_verified=True, no_dependency_bias=True,
        source_reference_checkpoint_hashes_unchanged=True))
    save()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        pid_file = OUTPUT/'supervisor.pid'
        if pid_file.is_file() and pid_file.read_text().strip() == str(os.getpid()):
            state = read(OUTPUT/'status.json')
            state['phase'] = 'failed'
            state['failed'].append(dict(key='supervisor', error=str(error)))
            write(OUTPUT/'status.json', state)
        raise
