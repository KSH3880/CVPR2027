import fcntl
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch

from run_stage2_scalability import ROOT, verify
from run_v4_formal import CHECKPOINT, OUTPUT as BASELINE, jobs as formal_jobs
from run_v4_group_radius_eval import read, write, sha
import stage2_xy100

OUTPUT = ROOT/'output_etc/check/depbias_v4_2000_4h16h_seeds167_20261114'
GPUS = (6, 7)
SEEDS = (1, 6, 7)
METRICS = {'I': 'Independent Interaction Success Rate', 'E': 'Enabling Interaction Group Success Rate',
           'C': 'Coupled Interaction Group Success Rate', 'Mixed': 'Mixed Interaction Group Success Rate'}


def destination(job):
    return OUTPUT/job['condition']/job['name']/f"general_seed{job['seed']}"


def validate(job, baseline=False):
    path = (BASELINE/'table1'/job['name']/f"general_seed{job['seed']}" if baseline else destination(job))
    reference = Path(job['reference'])
    proof = verify(path, reference, job['people'], 512)
    result = read(path/'result.json')
    if result['experiment'] != 'stage2_distill_900_v4' or result['seed'] != job['seed']:
        raise ValueError('Wrong checkpoint configuration or seed')
    if result['distractor_attention'] != 'all_observed_tokens' or result['enabling_success'] != 'simultaneous':
        raise ValueError('Evaluation protocol changed')
    if baseline:
        preservation = read(path/'policy_preservation.json')
        if not read(path/'first_policy_input.json')['exact'] or preservation['epoch'] != 2000:
            raise ValueError('Baseline first input or epoch differs')
    else:
        preservation = result['depbias_ablation']
        for key in ('first_policy_input_exact', 'disabled_action_exact', 'observation_unchanged', 'rng_restored'):
            if not preservation[key]:
                raise ValueError(key)
        if preservation['mode'] != job['condition'] or any(
                count != result['simulated_steps'] for count in preservation['calls'].values()):
            raise ValueError('Intervention mode or frame count differs')
        first = torch.load(path/'first_policy_input.pt', map_location='cpu', weights_only=False)
        if not torch.equal(first, torch.load(reference/'observed_scene.pt', map_location='cpu', weights_only=False)):
            raise ValueError('Actual first input differs')
    if not all(preservation[k] for k in ('weights_exact_before_after', 'normalization_exact_before_after')):
        raise ValueError('Policy changed')
    expected_calls = result['simulated_steps'] + (0 if baseline else 3)
    if result['encoder_attention_checks'] != dict(actor=expected_calls, critic=expected_calls):
        raise ValueError('Attention count differs')
    if job['radius'] is not None:
        spacing = read(path/'group_spacing.json')
        if spacing['group_spawn_radius'] != 7.5 or spacing['group_spawn_gap'] is not None:
            raise ValueError('Group radius differs')
        initial = torch.load(path/'initial.pt', map_location='cpu', weights_only=False)
        groups = initial['groups'].clone()
        rows, followers = (initial['predecessor'] >= 0).nonzero(as_tuple=True)
        groups[rows, followers] = groups[rows, initial['predecessor'][rows, followers]]
        same = groups[:, :, None] == groups[:, None, :]
        sizes = same.sum(-1, keepdim=True)
        centers = torch.bmm(same.float(), initial['roots'][..., :2])/sizes
        counts = (1/sizes.squeeze(-1).float()).sum(-1)
        origin = (centers/sizes).sum(1)/counts[:, None]
        error = float(((centers-origin[:, None]).norm(dim=-1)-7.5).abs().max())
        gap = 15*math.sin(math.pi/int(counts.round().max()))
        if error > .002 or abs(spacing['minimum_initial_group_gap']-gap) > .002:
            raise ValueError('Actual group ring geometry differs')
        proof.update(group_radius_exact=True, maximum_radius_error_m=error)
    proof.update(source=str(path), policy_preserved=True, first_policy_input_exact=True)
    return proof


def launch(job, gpu):
    path = destination(job)
    if path.exists():
        raise ValueError('Refusing to overwrite '+str(path))
    path.mkdir(parents=True)
    env = dict(os.environ)
    for key in ('CUDA_MPS_PIPE_DIRECTORY', 'CUDA_MPS_LOG_DIRECTORY', 'REFERENCE_ONLY', 'TABLE2_SETTING',
                'STAGE1_POLICY_CHECKPOINT', 'FSM_STAGE1_CHECKPOINT', 'CHECK', 'MAX_ITERATIONS',
                'RESUME_CHECKPOINT', 'GROUP_SPAWN_RADIUS', 'SPACING_METERS', 'STAGE2_SAMPLING_AUDIT',
                'BIAS_ABLATION', 'SPACING_MODEL'):
        env.pop(key, None)
    env.update(TOKENHSI_GPU=str(gpu), STAGE2_SCENE_MODE='mixed', DEPBIAS_MODE=job['condition'],
        STAGE2_REFERENCE_ROOT=str(Path(job['reference']).parent), PYTHONHASHSEED=str(job['seed']),
        PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1')
    if job['radius'] is not None:
        env.update(SPACING_MODEL='ours', SPACING_METERS='native', GROUP_SPAWN_RADIUS='7.5')
    command = [sys.executable, str(ROOT/'eval/depbias_ablation_eval.py'), '--task', 'HumanoidMACoupledPush',
        '--cfg_train', 'tokenhsi/data/cfg/train/rlg/stage2_distill_900_v4.yaml', '--cfg_env', '',
        '--motion_file', 'tokenhsi/data/dataset_stage1_jointpush.yaml', '--checkpoint', str(CHECKPOINT),
        '--seed', str(job['seed']), '--num_envs', '512', '--num_agents', str(job['people']),
        '--num_objects', str(2*job['people']), '--test', '--eval', '--headless', '--no_video',
        '--eval_protocol', 'general', '--eval_repeats', '1', '--task_graph', 'random_scenario',
        '--enable_size_mode', 'per_action_intersection', '--coupled_assignment', 'ordered',
        '--output_path', str(path)]
    if job['people'] > 4:
        command.append('--group_spawn_over4')
    with (OUTPUT/'logs'/f"{job['key']}.log").open('w') as log:
        return subprocess.Popen(['bash', '-c',
            'source tokenhsi/scripts/multi_agent/runtime_env.sh\nexec "$@"', '_', *command],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)


def report(jobs, state):
    rows, tables = [], {}
    lines = ['# Dependency bias · V4 2k', '',
        f"512환경·seed1/6/7·900step·XY100%·16H R7.5m. 신규 검증 {len(state['verified'])}/12회.", '',
        '| Condition | 4-agent Mixed SR (%) | 16-agent Mixed SR (%) |', '|---|---:|---:|']
    for condition in ('original', 'off', 'shuffled'):
        tables[condition] = {}
        cells = []
        for people in (4, 16):
            runs = []
            for seed in SEEDS:
                name = f'mixed_{people}h'
                key = f'{condition}_{people}h_seed{seed}'
                if condition != 'original' and key not in state['verified']:
                    continue
                path = (BASELINE/'table1' if condition == 'original' else OUTPUT/condition)/name/f'general_seed{seed}'
                score = read(path/'result.json')['interaction_group_scores']['region']
                row = dict(condition=condition, people=people, seed=seed, source=str(path), scores=score)
                runs.append(row)
                rows.append(row)
            if len(runs) != 3:
                cells.append(f'대기 ({len(runs)}/3)')
                continue
            values = {metric: [100*r['scores'][label]['rate'] for r in runs] for metric, label in METRICS.items()}
            stats = {metric: dict(mean=statistics.mean(v), std=statistics.stdev(v), seeds=v)
                     for metric, v in values.items()}
            tables[condition][str(people)] = stats
            cells.append(f"{stats['Mixed']['mean']:.2f} ± {stats['Mixed']['std']:.2f}")
        lines.append('| '+condition+' | '+' | '.join(cells)+' |')
    lines += ['', 'OFF는 attention dependency bias만0. Shuffled는 SELF/padding을 유지하고 NONE 포함 타인 bias를 행별 셔플한다.',
        '장면별 순열은 별도 CPU RNG(seed=100000+평가seed)로 생성해 episode 동안 사람 ID 기준으로 고정한다. Actor/critic·모든 head/layer가 같은 순열을 사용한다.',
        '관측·실제 관계·공유 embedding·feature fusion·SCT·checkpoint·정규화는 유지한다. 추론 개입이며 재학습 ablation이 아니다.',
        'Mixed는 seed별 I/E/C SR 동일 가중 평균, 표는3seed 평균±표본std. I/E/C 및 seed별 값은 summary.json에 저장한다.',
        '셔플은 우연히 원래 값을 유지할 수 있고, 행별 값이 모두 같으면 변화가 없다. 별도 셔플 seed 반복은 하지 않는다.']
    write(OUTPUT/'summary.json', dict(completed=state['completed'], tables=tables, runs=rows))
    (OUTPUT/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Supervisor must run on CPU')
    torch.set_num_threads(1)
    stage2_xy100.install()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT/'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (OUTPUT/'supervisor.pid').write_text(str(os.getpid())+'\n')
    (OUTPUT/'logs').mkdir(exist_ok=True)
    (OUTPUT/'checks').mkdir(exist_ok=True)
    originals = [j for j in formal_jobs() if j['table'] == 'table1' and j['mode'] == 'mixed' and j['people'] in (4, 16)]
    jobs = [dict(j, condition=condition, key=f"{condition}_{j['people']}h_seed{j['seed']}")
            for j in originals for condition in ('off', 'shuffled')]
    hashes = {str(CHECKPOINT): sha(CHECKPOINT)}
    if read(BASELINE/'manifest.json')['checkpoint_sha256'] != hashes[str(CHECKPOINT)]:
        raise ValueError('Original checkpoint SHA differs')
    for source in ('depbias_ablation.py', 'depbias_ablation_eval.py', 'run_depbias_ablation.py'):
        hashes[str(ROOT/'eval'/source)] = sha(ROOT/'eval'/source)
    for job in originals:
        source = BASELINE/'table1'/job['name']/f"general_seed{job['seed']}"
        for path in [source/'result.json', source/'policy_preservation.json',
                     Path(job['reference'])/'initial.pt', Path(job['reference'])/'observed_scene.pt',
                     Path(job['reference'])/'reference_info.json']:
            hashes[str(path)] = sha(path)
        print('VERIFY ORIGINAL', job['key'], flush=True)
        write(OUTPUT/'checks'/f"original_{job['people']}h_seed{job['seed']}.json", validate(job, baseline=True))
    write(OUTPUT/'source_hashes.json', hashes)
    manifest = dict(protocol_id='cvpr2027-evaluation-v1.1.0', checkpoint=str(CHECKPOINT),
        checkpoint_sha256=hashes[str(CHECKPOINT)], envs=512, seeds=SEEDS, horizon=900, xy_extent_percent=100,
        objects='2N', gpus=GPUS, expected_runs=12, reused_original_runs=6, jobs=jobs,
        shuffle='row-wise non-self valid keys including NONE; fixed canonical person IDs',
        permutation_seed='100000 + evaluation seed', workers_per_gpu=1, completed=False)
    write(OUTPUT/'manifest.json', manifest)
    state = dict(completed=False, expected_runs=12, verified=[], running={}, failed=[], pending=12)
    pending, running = jobs.copy(), {}
    report(jobs, state)
    while pending or running:
        usage = subprocess.check_output(['nvidia-smi', '-i', '6,7', '--query-gpu=index,memory.free',
                                         '--format=csv,noheader,nounits'], text=True)
        free = {int(row.split(',')[0]): int(row.split(',')[1]) for row in usage.splitlines()}
        for gpu in GPUS:
            if pending and not state['failed'] and gpu not in running:
                job = pending[0]
                reserve = 14000 if job['people'] == 16 else 8000
                if free[gpu] < reserve+12000:
                    continue
                pending.pop(0)
                process = launch(job, gpu)
                running[gpu] = process, job
                state['running'][job['key']] = dict(pid=process.pid, gpu=gpu, started_unix=time.time())
                print('START', job['key'], 'GPU', gpu, 'PID', process.pid, flush=True)
        for gpu, (process, job) in list(running.items()):
            if process.poll() is None:
                continue
            try:
                if process.returncode:
                    raise RuntimeError(f'Process exit {process.returncode}; see logs/{job["key"]}.log')
                write(OUTPUT/'checks'/f"{job['key']}.json", validate(job))
                state['verified'].append(job['key'])
                print('VERIFIED', job['key'], flush=True)
            except Exception as error:
                state['failed'].append(dict(key=job['key'], error=str(error)))
                print('FAILED', job['key'], str(error), flush=True)
            del running[gpu]
            del state['running'][job['key']]
        state.update(pending=len(pending), updated_unix=time.time())
        write(OUTPUT/'status.json', state)
        report(jobs, state)
        if state['failed'] and not running:
            raise RuntimeError('Queue stopped after failure')
        if pending or running:
            time.sleep(10)
    if any(sha(Path(path)) != digest for path, digest in hashes.items()):
        raise ValueError('Source/checkpoint/reference/original result changed')
    manifest['completed'] = state['completed'] = True
    write(OUTPUT/'manifest.json', manifest)
    write(OUTPUT/'status.json', state)
    write(OUTPUT/'completion_verification.json', dict(completed=True, new_runs=12, reused_original_runs=6,
        cpu_recomputed_all_18=True, source_hashes_unchanged=True, first_policy_inputs_exact=True,
        weights_and_normalization_preserved=True, actual_group_radius_verified=True))
    report(jobs, state)


if __name__ == '__main__':
    main()
