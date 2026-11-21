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
from run_v4_group_radius_eval import read, write, sha
import stage2_xy100

OUTPUT = ROOT/'output_etc/check/taskbias_stage1_26k_2h4h8h16h_seeds167_20261115'
STAGE1 = ROOT/'pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth'
CHECKPOINT = ROOT/'output/stage2_distill_900_v2/Stage2Distill900V2_08-19-13-51/nn/Stage2Distill900V2_00002000.pth'
OLD = ROOT/'output_etc/check/v3_2000_tables12_seeds167_20261110'
RAD = ROOT/'output_etc/check/table1_group_radius_8r3_16r7p5_env512_seeds167_baselines_20261112'
OFF = ROOT/'output_etc/check/bias_ablation_20261110/task'
SEEDS, PEOPLE, GPUS = (1, 6, 7), (2, 4, 8, 16), (6, 7)


def jobs():
    result = []
    for people in PEOPLE:
        for seed in SEEDS:
            name, leaf = f'independent_{people}h', f'general_seed{seed}'
            reference = (OLD/'reference/table1' if people <= 4 else RAD/'reference')/name/leaf
            for condition in ('original', 'off', 'shuffled'):
                source = None
                if condition == 'original':
                    source = (OLD/'table1/stage1' if people <= 4 else RAD/'stage1')/name/leaf
                elif condition == 'off' and people <= 4:
                    source = OFF/name/leaf
                result.append(dict(condition=condition, people=people, seed=seed, name=name,
                    key=f'{condition}_{people}h_seed{seed}', reference=str(reference),
                    source=str(source) if source else None, radius={8:3., 16:7.5}.get(people)))
    return result


def destination(job):
    return OUTPUT/job['condition']/job['name']/f"general_seed{job['seed']}"


def validate(job):
    path, reference = destination(job), Path(job['reference'])
    proof = verify(path, reference, job['people'], 512)
    result, policy = read(path/'result.json'), read(path/'stage1_policy.json')
    if result['seed'] != job['seed'] or policy['epoch'] != 26000 or policy['sha256'] != sha(STAGE1):
        raise ValueError('Wrong Stage1 checkpoint or seed')
    for key in ('native_actor_exact', 'normalization_exact', 'weights_unchanged', 'normalization_unchanged'):
        if not policy[key]:
            raise ValueError(key)
    if policy['coordination_calls'] or policy['coordination_used'] or policy['merge_shared']:
        raise ValueError('Stage1 architecture differs')
    steps = result['simulated_steps']
    diagnostic = 0 if job['condition'] == 'original' else 3
    if result['encoder_attention_checks'] != dict(actor=steps+diagnostic+1, critic=steps+diagnostic):
        raise ValueError('Attention count differs')
    if job['condition'] != 'original':
        bias = read(path/'bias_ablation.json')
        for key in ('first_policy_input_exact', 'no_ablation_action_exact', 'observation_unchanged_by_probe',
                    'rng_restored_after_probe', 'model_weights_unchanged', 'normalization_unchanged'):
            if not bias[key]:
                raise ValueError(key)
        calls = bias['removed_bias_calls'] if job['source'] else bias['calls']
        if calls != dict(actor=steps, critic=steps):
            raise ValueError('Intervention missing on a frame')
        expected = 'task' if job['source'] else job['condition']
        if bias['mode'] != expected:
            raise ValueError('Wrong bias intervention')
        first = torch.load(path/'first_policy_input.pt', map_location='cpu', weights_only=False)
        if not torch.equal(first, torch.load(reference/'observed_scene.pt', map_location='cpu', weights_only=False)):
            raise ValueError('First input differs')
    if job['radius'] is not None:
        spacing = read(path/'group_spacing.json')
        if spacing['group_spawn_radius'] != job['radius'] or spacing['group_spawn_gap'] is not None:
            raise ValueError('Wrong ring placement')
        initial = torch.load(path/'initial.pt', map_location='cpu', weights_only=False)
        roots = initial['roots'][..., :2]
        error = float(((roots-roots.mean(1, keepdim=True)).norm(dim=-1)-job['radius']).abs().max())
        gap = 2*job['radius']*math.sin(math.pi/job['people'])
        if error > .002 or abs(spacing['minimum_initial_group_gap']-gap) > .002:
            raise ValueError('Actual ring geometry differs')
        proof.update(actual_radius_verified=True, maximum_radius_error_m=error)
    proof.update(stage1_policy_preserved=True, source=job['source'], intervention=job['condition'])
    return proof


def report(all_jobs, state):
    tables, rows = {}, []
    lines = ['# Stage1 26k SCT task bias', '',
        'Independent · 512 envs · 900 steps · seeds 1/6/7 · XY100% · M=2N · 8H R3m · 16H R7.5m', '',
        '| Condition | 2 agents | 4 agents | 8 agents | 16 agents |', '|---|---:|---:|---:|---:|']
    for condition in ('original', 'off', 'shuffled'):
        tables[condition], cells = {}, []
        for people in PEOPLE:
            selected = [j for j in all_jobs if j['condition'] == condition and j['people'] == people and j['key'] in state['verified']]
            values = []
            for job in selected:
                result = read(destination(job)/'result.json')
                score = result['interaction_group_scores']['region']['Independent Interaction Success Rate']
                values.append(100*score['rate'])
                rows.append(dict(condition=condition, people=people, seed=job['seed'], score=score, source=str(destination(job))))
            if len(values) == 3:
                stats = dict(mean=statistics.mean(values), std=statistics.stdev(values), seeds=values)
                tables[condition][str(people)] = stats
                cells.append(f"{stats['mean']:.2f} ± {stats['std']:.2f}")
            else:
                cells.append(f'대기 ({len(values)}/3)')
        lines.append('| '+condition+' | '+' | '.join(cells)+' |')
    lines += ['', '평균±표본 표준편차(%). 원본12회 및 동일 배치의 OFF 2/4H 6회를 CPU 재검산하여 재사용한다.',
        'OFF: SCT에서 계산한 유한 attention bias를0으로 설정. 기존 task bias table 0과 동등함을 CPU 검증했다.',
        'Shuffled: 사람·물체·목표 전체 토큰의 각 query 행에서 SELF와 padding을 유지하며 NONE 포함 key bias를 섞는다.',
        '별도 CPU RNG(100000+평가seed), scene/query별 순열을 episode 동안 canonical token ID 기준으로 고정하고 모든 head/layer 및 actor/critic이 공유한다.',
        '개입은 task fusion의 dense attention bias 계산 후, padding 및 내부 token permutation 전에 적용한다. 실제 task packet·task embedding·feature fusion·GTA·action head·checkpoint·정규화는 유지한다.',
        'Stage1 native actor/정규화 일치와 DCT 미사용을 검증한다. 재학습 없는 추론 개입이며, 2명도 물체·목표 토큰이 있어 셔플이 항등변환이 아니다.',
        '세부 seed와 분모/분자는 summary.json, 실행별 검증은 checks/에 기록한다.']
    write(OUTPUT/'summary.json', dict(completed=state['completed'], tables=tables, runs=rows))
    (OUTPUT/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Supervisor must run on CPU')
    torch.set_num_threads(1)
    stage2_xy100.install()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT/'manifest.json').exists():
        raise ValueError('Refusing to overwrite evaluation')
    lock = (OUTPUT/'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (OUTPUT/'supervisor.pid').write_text(str(os.getpid())+'\n')
    for folder in ('logs', 'checks'):
        (OUTPUT/folder).mkdir(exist_ok=True)
    sources = [STAGE1, CHECKPOINT, CHECKPOINT.parent.parent/'relation_config.yaml',
        ROOT/'tokenhsi/data/cfg/train/rlg/stage2_distill_900_v2.yaml',
        ROOT/'tokenhsi/data/cfg/multi_agent/stage2_distill_900_v2.yaml']
    sources += [ROOT/'eval'/name for name in ('taskbias_ablation.py', 'taskbias_ablation_eval.py',
        'run_taskbias_ablation.py', 'test_taskbias_ablation.py', 'depbias_ablation.py', 'bias_ablation.py',
        'bias_ablation_eval.py', 'stage1_protocol_policy.py', 'group_spacing_eval.py', 'stage2_scalability_eval.py',
        'run_stage2_scalability.py', 'stage2_xy100.py', 'stage2_distractor_physics.py')]
    sources += [ROOT/'tokenhsi/learning/multi_agent'/name for name in ('task_role_encoder.py', 'amp_network_builder_ma.py')]
    hashes = {str(p):sha(p) for p in sources}
    all_jobs = jobs()
    state = dict(completed=False, verified=[], running={}, failed=[], pending=18)
    for job in all_jobs:
        ref = Path(job['reference'])
        for name in ('initial.pt', 'observed_scene.pt', 'reference_info.json'):
            hashes[str(ref/name)] = sha(ref/name)
        if not job['source']:
            continue
        target = destination(job)
        target.mkdir(parents=True, exist_ok=False)
        for item in Path(job['source']).iterdir():
            if item.is_file():
                hashes[str(item.resolve())] = sha(item)
                if item.name != 'rescore.json':
                    (target/item.name).symlink_to(item.resolve())
        write(OUTPUT/'checks'/f"{job['key']}.json", validate(job))
        state['verified'].append(job['key'])
        print('REUSED VERIFIED', job['key'], flush=True)
    write(OUTPUT/'source_hashes.json', hashes)
    manifest = dict(completed=False, stage1_checkpoint=str(STAGE1), stage1_sha256=sha(STAGE1),
        carrier_checkpoint=str(CHECKPOINT), gpus=GPUS, seeds=SEEDS, envs=512, horizon=900,
        new_runs=18, reused_runs=18, total_runs=36, jobs=all_jobs)
    write(OUTPUT/'manifest.json', manifest)
    pending, running = [j for j in all_jobs if not j['source']], {}
    while pending or running:
        usage = subprocess.check_output(['nvidia-smi', '-i', '6,7', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True)
        free = {int(row.split(',')[0]):int(row.split(',')[1]) for row in usage.splitlines()}
        for gpu in GPUS:
            if pending and not state['failed'] and gpu not in running:
                job = pending[0]
                if free[gpu] < {2:17000, 4:20000, 8:22000, 16:26000}[job['people']]:
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
                    raise RuntimeError(f'Exit {process.returncode}; see logs/{job["key"]}.log')
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
        report(all_jobs, state)
        if state['failed'] and not running:
            raise RuntimeError('Queue stopped after failure')
        if pending or running:
            time.sleep(10)
    if any(sha(Path(path)) != digest for path, digest in hashes.items()):
        raise ValueError('Checkpoint/source/reference/original changed')
    manifest['completed'] = state['completed'] = True
    write(OUTPUT/'manifest.json', manifest)
    write(OUTPUT/'status.json', state)
    write(OUTPUT/'completion_verification.json', dict(completed=True, new_runs=18, reused_runs=18,
        cpu_recomputed_all_36=True, source_hashes_unchanged=True, stage1_policy_preserved=True,
        matched_initial_states=True, actual_radius_verified=True))
    report(all_jobs, state)


def launch(job, gpu):
    path = destination(job)
    if path.exists():
        raise ValueError('Refusing to overwrite '+str(path))
    path.mkdir(parents=True)
    env = dict(os.environ)
    for key in ('CUDA_MPS_PIPE_DIRECTORY', 'CUDA_MPS_LOG_DIRECTORY', 'REFERENCE_ONLY', 'TABLE2_SETTING',
                'STAGE1_POLICY_CHECKPOINT', 'FSM_STAGE1_CHECKPOINT', 'CHECK', 'MAX_ITERATIONS',
                'RESUME_CHECKPOINT', 'GROUP_SPAWN_RADIUS', 'SPACING_METERS', 'STAGE2_SAMPLING_AUDIT',
                'BIAS_ABLATION', 'SPACING_MODEL', 'BIAS_ABLATION_PROBE_ONLY', 'DEPBIAS_MODE'):
        env.pop(key, None)
    env.update(TOKENHSI_GPU=str(gpu), STAGE2_SCENE_MODE='independent', BIAS_ABLATION=job['condition'], STAGE1_POLICY_CHECKPOINT=str(STAGE1),
        STAGE2_REFERENCE_ROOT=str(Path(job['reference']).parent), PYTHONHASHSEED=str(job['seed']),
        PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1')
    if job['radius'] is not None:
        env.update(SPACING_MODEL='stage1_ablation', SPACING_METERS='native', GROUP_SPAWN_RADIUS=str(job['radius']))
    entry = 'taskbias_ablation_eval.py'
    command = [sys.executable, str(ROOT/'eval'/entry), '--task', 'HumanoidMACoupledPush',
        '--cfg_train', 'tokenhsi/data/cfg/train/rlg/stage2_distill_900_v2.yaml', '--cfg_env', '',
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



if __name__ == '__main__':
    main()
