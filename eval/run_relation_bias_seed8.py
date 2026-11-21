import fcntl
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch

import run_depbias_ablation as previous
from run_depbias_ablation import ROOT, CHECKPOINT, METRICS, read, write, sha
import stage2_xy100

OUTPUT = ROOT/'output_etc/check/relation_bias_v4_2000_4h_seed8_20261114'
PRIOR = ROOT/'output_etc/check/relation_bias_v4_2000_4h_seeds167_20261114'
REFERENCE = OUTPUT/'reference/mixed_4h/general_seed8'
CONDITIONS = ('original', 'enabling_off', 'coupled_off', 'off')


def destination(condition):
    if condition == 'reference':
        return REFERENCE
    base = OUTPUT/'original/table1' if condition == 'original' else OUTPUT/condition
    return base/'mixed_4h/general_seed8'


def launch(condition, gpu):
    path = destination(condition)
    path.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    for key in ('CUDA_MPS_PIPE_DIRECTORY', 'CUDA_MPS_LOG_DIRECTORY', 'REFERENCE_ONLY', 'TABLE2_SETTING',
                'STAGE1_POLICY_CHECKPOINT', 'FSM_STAGE1_CHECKPOINT', 'CHECK', 'MAX_ITERATIONS',
                'RESUME_CHECKPOINT', 'STAGE2_SAMPLING_AUDIT', 'BIAS_ABLATION', 'DEPBIAS_MODE',
                'SPACING_MODEL', 'SPACING_METERS', 'GROUP_SPAWN_RADIUS'):
        env.pop(key, None)
    env.update(TOKENHSI_GPU=str(gpu), STAGE2_SCENE_MODE='mixed', STAGE2_REFERENCE_ROOT=str(REFERENCE.parent),
        PYTHONHASHSEED='8', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    if condition == 'reference':
        entry = 'stage2_scalability_eval.py'
        env['REFERENCE_ONLY'] = '1'
    elif condition == 'original':
        entry = 'v4_formal_eval.py'
    else:
        entry = 'depbias_ablation_eval.py' if condition == 'off' else 'relation_bias_ablation_eval.py'
        env['DEPBIAS_MODE'] = condition
    command = [sys.executable, str(ROOT/'eval'/entry), '--task', 'HumanoidMACoupledPush',
        '--cfg_train', 'tokenhsi/data/cfg/train/rlg/stage2_distill_900_v4.yaml', '--cfg_env', '',
        '--motion_file', 'tokenhsi/data/dataset_stage1_jointpush.yaml', '--checkpoint', str(CHECKPOINT),
        '--seed', '8', '--num_envs', '512', '--num_agents', '4', '--num_objects', '8',
        '--test', '--eval', '--headless', '--no_video', '--eval_protocol', 'general', '--eval_repeats', '1',
        '--task_graph', 'random_scenario', '--enable_size_mode', 'per_action_intersection',
        '--coupled_assignment', 'ordered', '--output_path', str(path)]
    with (OUTPUT/'logs'/f'{condition}.log').open('w') as log:
        return subprocess.Popen(['bash', '-c',
            'source tokenhsi/scripts/multi_agent/runtime_env.sh\nexec "$@"', '_', *command],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)


def free_memory():
    result = subprocess.check_output(['nvidia-smi', '-i', '6,7', '--query-gpu=index,memory.free',
                                      '--format=csv,noheader,nounits'], text=True)
    return {int(row.split(',')[0]): int(row.split(',')[1]) for row in result.splitlines()}


def report(state):
    prior = read(PRIOR/'summary.json')
    rows = []
    lines = ['# Relation-specific bias · 4H · 추가 seed 8', '',
        'V4 2k·512환경·900step·XY100%·기본4명배치/8물체. 동일 초기 snapshot·checkpoint 사용.', '',
        '| 조건 | Independent SR | Enabling SR | Coupled SR | Mixed SR |', '|---|---:|---:|---:|---:|']
    for condition in CONDITIONS:
        if condition not in state['verified']:
            continue
        path = destination(condition)
        scores = read(path/'result.json')['interaction_group_scores']['region']
        rows.append(dict(condition=condition, seed=8, source=str(path), scores=scores))
        lines.append('| '+condition+' | '+' | '.join(f'{100*scores[m]["rate"]:.2f}' for m in METRICS.values())+' |')
    tables, deltas = {}, {}
    if state['completed']:
        base = next(r for r in rows if r['condition'] == 'original')['scores']
        lines += ['', '| seed 8: Original 대비(%p) | Independent | Enabling | Coupled | Mixed |',
                  '|---|---:|---:|---:|---:|']
        for row in rows[1:]:
            delta = {m:100*(row['scores'][label]['rate']-base[label]['rate']) for m,label in METRICS.items()}
            deltas[row['condition']] = delta
            lines.append('| '+row['condition']+' | '+' | '.join(f'{v:+.2f}' for v in delta.values())+' |')
        lines += ['', '4개 평가 seed(1/6/7/8) 평균±표본표준편차, SR(%):', '',
            '| 조건 | Independent | Enabling | Coupled | Mixed |', '|---|---:|---:|---:|---:|']
        for condition in CONDITIONS:
            values = [r for r in prior['runs']+rows if r['condition'] == condition]
            if sorted(r['seed'] for r in values) != [1,6,7,8]:
                raise ValueError('Unexpected aggregate seeds')
            tables[condition] = {}
            for m,label in METRICS.items():
                rates = [100*r['scores'][label]['rate'] for r in values]
                tables[condition][m] = dict(mean=statistics.mean(rates), std=statistics.stdev(rates), seeds=rates)
            lines.append('| '+condition+' | '+' | '.join(f'{v["mean"]:.2f} ± {v["std"]:.2f}'
                         for v in tables[condition].values())+' |')
    lines += ['', 'Mixed는 I/E/C SR 동일 가중 평균. 기존3seed 결과는 보존하고4seed 집계를 별도로 기록한다.',
        '새 학습 없이 계산된 attention bias만 개입한다. 개별 제거는 해당 관계만, All OFF는 모든 dependency bias를0으로 만든다.',
        '실제 관계·feature fusion·SCT·action head·checkpoint·정규화는 유지한다. 추론 개입의 결과이며 학습 필수성·통계적 유의성을 단정하지 않는다.']
    write(OUTPUT/'summary.json', dict(completed=state['completed'], runs=rows, seed8_delta_pp=deltas,
        aggregate_seeds=[1,6,7,8], aggregate_tables=tables, prior_summary=str(PRIOR/'summary.json')))
    (OUTPUT/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Supervisor requires CPU mode')
    torch.set_num_threads(1)
    stage2_xy100.install()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT/'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUTPUT/'manifest.json').exists():
        raise ValueError('Refusing to overwrite existing evaluation')
    for name in ('logs', 'checks'):
        (OUTPUT/name).mkdir(exist_ok=True)
    (OUTPUT/'supervisor.pid').write_text(str(os.getpid())+'\n')
    sources = [CHECKPOINT, CHECKPOINT.parent.parent/'relation_config.yaml', PRIOR/'summary.json',
        PRIOR/'completion_verification.json', ROOT/'tokenhsi/data/cfg/train/rlg/stage2_distill_900_v4.yaml',
        ROOT/'tokenhsi/data/cfg/multi_agent/stage2_distill_900_v4.yaml']
    sources += [ROOT/'eval'/name for name in ('run_relation_bias_seed8.py', 'relation_bias_ablation.py',
        'relation_bias_ablation_eval.py', 'depbias_ablation.py', 'depbias_ablation_eval.py',
        'v4_formal_eval.py', 'stage2_scalability_eval.py', 'stage2_protocol_eval.py',
        'stage2_protocol_sampling.py', 'stage2_xy100.py', 'stage2_distractor_physics.py',
        'run_depbias_ablation.py', 'run_stage2_scalability.py')]
    sources += [ROOT/'tokenhsi/learning/multi_agent'/name for name in
                ('human_coordination.py', 'stage2_relation_fusion.py', 'coupled_relation_fusion.py')]
    hashes = {str(path):sha(path) for path in sources}
    if not read(PRIOR/'completion_verification.json')['completed'] or (
            read(PRIOR/'manifest.json')['checkpoint_sha256'] != hashes[str(CHECKPOINT)]):
        raise ValueError('Prior experiment is incomplete or checkpoint differs')
    manifest = dict(protocol_id='cvpr2027-evaluation-v1.1.0', checkpoint=str(CHECKPOINT),
        checkpoint_sha256=hashes[str(CHECKPOINT)], envs=512, agents=4, objects=8, seed=8, horizon=900,
        xy_extent_percent=100, group_radius_m=None, gpus=[6,7], workers_per_gpu=1, new_runs=4,
        reference=str(REFERENCE), conditions=CONDITIONS, inference_only=True, completed=False)
    write(OUTPUT/'manifest.json', manifest)
    available = free_memory()
    while max(available.values()) < 20000:
        time.sleep(10)
        available = free_memory()
    gpu = max(available, key=available.get)
    print('REFERENCE GPU', gpu, flush=True)
    if launch('reference', gpu).wait():
        raise RuntimeError('Reference generation failed')
    info = read(REFERENCE/'reference_info.json')
    if tuple(info[k] for k in ('seed','envs','agents','objects','simulated_steps')) != (8,512,4,8,0):
        raise ValueError('Reference configuration differs')
    for name in ('initial.pt','observed_scene.pt','reference_info.json'):
        hashes[str(REFERENCE/name)] = sha(REFERENCE/name)
    write(OUTPUT/'source_hashes.json', hashes)
    previous.OUTPUT, previous.BASELINE = OUTPUT, OUTPUT/'original'
    state = dict(completed=False, expected_runs=4, verified=[], running={}, failed=[], pending=4)
    pending, running = list(CONDITIONS), {}
    while pending or running:
        available = free_memory()
        for gpu in (6,7):
            if pending and not state['failed'] and gpu not in running and available[gpu] >= 20000:
                condition = pending.pop(0)
                process = launch(condition, gpu)
                running[gpu] = process, condition
                state['running'][condition] = dict(pid=process.pid, gpu=gpu)
                print('START', condition, 'GPU', gpu, flush=True)
        for gpu,(process,condition) in list(running.items()):
            if process.poll() is None:
                continue
            try:
                if process.returncode:
                    raise RuntimeError(f'Process exited {process.returncode}')
                job = dict(condition=condition, name='mixed_4h', seed=8, people=4, radius=None, reference=str(REFERENCE))
                proof = previous.validate(job, baseline=condition == 'original')
                write(OUTPUT/'checks'/f'{condition}.json', proof)
                state['verified'].append(condition)
                print('VERIFIED', condition, flush=True)
            except Exception as error:
                state['failed'].append(dict(condition=condition,error=str(error)))
                print('FAILED', condition, str(error), flush=True)
            del running[gpu]
            del state['running'][condition]
        state.update(pending=len(pending), updated_unix=time.time())
        write(OUTPUT/'status.json', state)
        report(state)
        if state['failed'] and not running:
            raise RuntimeError('Evaluation failed; see status.json')
        if pending or running:
            time.sleep(5)
    if len(state['verified']) != 4 or any(sha(Path(p)) != digest for p,digest in hashes.items()):
        raise ValueError('Incomplete evaluation or source hash changed')
    manifest['completed'] = state['completed'] = True
    write(OUTPUT/'manifest.json', manifest)
    write(OUTPUT/'status.json', state)
    report(state)
    write(OUTPUT/'completion_verification.json', dict(completed=True, new_runs=4,
        reference_seed=8, cpu_recomputed_all_four=True, first_inputs_exact=True,
        policy_and_normalization_preserved=True, source_and_reference_hashes_unchanged=True))


if __name__ == '__main__':
    main()
