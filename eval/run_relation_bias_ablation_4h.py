import concurrent.futures
import fcntl
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

import torch

import run_depbias_ablation as previous
from run_depbias_ablation import ROOT, CHECKPOINT, SEEDS, METRICS, read, write, sha, formal_jobs
import stage2_xy100

OUTPUT = ROOT/'output_etc/check/relation_bias_v4_2000_4h_seeds167_20261114'
ORIGINAL = previous.BASELINE
ALL_OFF = previous.OUTPUT
CONDITIONS = ('enabling_off', 'coupled_off')


def destination(job):
    return OUTPUT/job['condition']/job['name']/f"general_seed{job['seed']}"


def launch(job, gpu):
    path = destination(job)
    path.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    for key in ('CUDA_MPS_PIPE_DIRECTORY', 'CUDA_MPS_LOG_DIRECTORY', 'REFERENCE_ONLY', 'TABLE2_SETTING',
                'STAGE1_POLICY_CHECKPOINT', 'FSM_STAGE1_CHECKPOINT', 'CHECK', 'MAX_ITERATIONS',
                'RESUME_CHECKPOINT', 'STAGE2_SAMPLING_AUDIT', 'BIAS_ABLATION',
                'SPACING_MODEL', 'SPACING_METERS', 'GROUP_SPAWN_RADIUS'):
        env.pop(key, None)
    env.update(TOKENHSI_GPU=str(gpu), STAGE2_SCENE_MODE='mixed', DEPBIAS_MODE=job['condition'],
        STAGE2_REFERENCE_ROOT=str(Path(job['reference']).parent), PYTHONHASHSEED=str(job['seed']),
        PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1')
    command = [sys.executable, str(ROOT/'eval/relation_bias_ablation_eval.py'), '--task', 'HumanoidMACoupledPush',
        '--cfg_train', 'tokenhsi/data/cfg/train/rlg/stage2_distill_900_v4.yaml', '--cfg_env', '',
        '--motion_file', 'tokenhsi/data/dataset_stage1_jointpush.yaml', '--checkpoint', str(CHECKPOINT),
        '--seed', str(job['seed']), '--num_envs', '512', '--num_agents', '4', '--num_objects', '8',
        '--test', '--eval', '--headless', '--no_video', '--eval_protocol', 'general', '--eval_repeats', '1',
        '--task_graph', 'random_scenario', '--enable_size_mode', 'per_action_intersection',
        '--coupled_assignment', 'ordered', '--output_path', str(path)]
    with (OUTPUT/'logs'/f"{job['key']}.log").open('w') as log:
        return subprocess.Popen(['bash', '-c',
            'source tokenhsi/scripts/multi_agent/runtime_env.sh\nexec "$@"', '_', *command],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)


def baseline_view(source, target, hashes):
    target.mkdir(parents=True, exist_ok=False)
    for item in source.iterdir():
        if item.is_file():
            hashes[str(item.resolve())] = sha(item)
            if item.name != 'rescore.json':
                (target/item.name).symlink_to(item.resolve())


def report(state):
    rows, tables = [], {}
    lines = ['# Relation-specific dependency bias · V4 2k', '',
        f"4명 Mixed·512환경·seed1/6/7·900step·XY100%·기본4명배치. 신규 검증 {len(state['verified'])}/6회.", '',
        '| Condition | Independent SR | Enabling SR | Coupled SR | Mixed SR |', '|---|---:|---:|---:|---:|']
    for condition in ('original', 'enabling_off', 'coupled_off', 'off'):
        runs = []
        for seed in SEEDS:
            name = f'general_seed{seed}'
            if condition in CONDITIONS and f'{condition}_seed{seed}' not in state['verified']:
                continue
            path = (ORIGINAL/'table1/mixed_4h'/name if condition == 'original' else
                    ALL_OFF/'off/mixed_4h'/name if condition == 'off' else
                    OUTPUT/condition/'mixed_4h'/name)
            row = dict(condition=condition, seed=seed, source=str(path),
                       scores=read(path/'result.json')['interaction_group_scores']['region'])
            runs.append(row)
            rows.append(row)
        if len(runs) != 3:
            lines.append('| '+condition+' | '+' | '.join([f'대기 ({len(runs)}/3)']*4)+' |')
            continue
        tables[condition] = {}
        for name, metric in METRICS.items():
            values = [100*run['scores'][metric]['rate'] for run in runs]
            tables[condition][name] = dict(mean=statistics.mean(values), std=statistics.stdev(values), seeds=values)
        lines.append('| '+condition+' | '+' | '.join(
            f"{value['mean']:.2f} ± {value['std']:.2f}" for value in tables[condition].values())+' |')
    lines += ['', 'SR은%·3평가seed 평균±표본표준편차. Independent는 Mixed 장면 내 독립 과제이며, Mixed는I/E/C 동일 가중 평균이다.']
    lines += ['Enabling OFF는 relation ID2(Follower→Predecessor), Coupled OFF는ID3(공동 참여자 양방향)의 계산된 attention bias만0으로 만든다.',
        'Actor/critic 전층에 적용하며 SELF/NONE/다른 관계 bias·padding·실제 관계·feature fusion·SCT·action head·checkpoint·정규화는 유지한다.',
        '기존 Original/All OFF6회를 재사용하고 새 조건6회만 시뮬레이션한다. 공통 snapshot/첫 입력·CPU SR/Error·기본4명배치·가중치를 검증한다.',
        '같은 장면 내 상호 영향 때문에 비대상 관계 SR도 변할 수 있다. 추론 개입이며 학습 시 구조의 필수성을 증명하지 않는다.']
    if state['completed']:
        lines += ['', '| Original 대비 변화(%p) | Independent | Enabling | Coupled | Mixed |', '|---|---:|---:|---:|---:|']
        for condition in (*CONDITIONS, 'off'):
            delta = {key: tables[condition][key]['mean']-tables['original'][key]['mean'] for key in METRICS}
            lines.append('| '+condition+' | '+' | '.join(f'{v:+.2f}' for v in delta.values())+' |')
    write(OUTPUT/'summary.json', dict(completed=state['completed'], tables=tables, runs=rows))
    (OUTPUT/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Supervisor requires CPU mode')
    torch.set_num_threads(1)
    stage2_xy100.install()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock_file = (OUTPUT/'supervisor.lock').open('a')
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUTPUT/'manifest.json').exists():
        raise ValueError('Refusing to overwrite an existing evaluation')
    (OUTPUT/'logs').mkdir(exist_ok=True)
    (OUTPUT/'checks').mkdir(exist_ok=True)
    (OUTPUT/'supervisor.pid').write_text(str(os.getpid())+'\n')
    originals = [j for j in formal_jobs() if j['table'] == 'table1' and j['name'] == 'mixed_4h']
    jobs = [dict(j, condition=c, key=f"{c}_seed{j['seed']}") for j in originals for c in CONDITIONS]
    sources = [CHECKPOINT, CHECKPOINT.parent.parent/'relation_config.yaml',
        ROOT/'tokenhsi/data/cfg/train/rlg/stage2_distill_900_v4.yaml',
        ROOT/'tokenhsi/data/cfg/multi_agent/stage2_distill_900_v4.yaml']
    sources += [ROOT/'eval'/name for name in ('relation_bias_ablation.py', 'relation_bias_ablation_eval.py',
        'run_relation_bias_ablation_4h.py', 'depbias_ablation_eval.py', 'run_depbias_ablation.py',
        'test_relation_bias_ablation.py', 'stage2_scalability_eval.py', 'group_spacing_eval.py')]
    sources += [ROOT/'tokenhsi/learning/multi_agent'/name for name in
                ('human_coordination.py', 'stage2_relation_fusion.py', 'coupled_relation_fusion.py')]
    hashes = {str(path): sha(path) for path in sources}
    if read(ALL_OFF/'manifest.json')['checkpoint_sha256'] != hashes[str(CHECKPOINT)] or not read(ALL_OFF/'completion_verification.json')['completed']:
        raise ValueError('Prior ablation is incompatible or incomplete')
    if read(ORIGINAL/'manifest.json')['checkpoint_sha256'] != hashes[str(CHECKPOINT)]:
        raise ValueError('Original checkpoint differs')
    previous.OUTPUT = OUTPUT/'reused'
    previous.BASELINE = OUTPUT/'reused/original'
    for job in originals:
        reference = Path(job['reference'])
        for name in ('initial.pt', 'observed_scene.pt', 'reference_info.json'):
            hashes[str(reference/name)] = sha(reference/name)
        for condition in ('original', 'off'):
            source = (ORIGINAL/'table1' if condition == 'original' else ALL_OFF/'off')/job['name']/f"general_seed{job['seed']}"
            target = (previous.BASELINE/'table1' if condition == 'original' else previous.OUTPUT/'off')/job['name']/source.name
            baseline_view(source, target, hashes)
            print('VERIFY REUSED', condition, job['seed'], flush=True)
            proof = previous.validate(dict(job, condition=condition), baseline=condition == 'original')
            proof['source'] = str(source)
            write(OUTPUT/'checks'/f"{condition}_seed{job['seed']}.json", proof)
    previous.OUTPUT = OUTPUT
    previous.BASELINE = ORIGINAL
    write(OUTPUT/'source_hashes.json', hashes)
    manifest = dict(protocol_id='cvpr2027-evaluation-v1.1.0', checkpoint=str(CHECKPOINT),
        checkpoint_sha256=hashes[str(CHECKPOINT)], config='stage2_distill_900_v4', envs=512, agents=4,
        objects=8, seeds=SEEDS, horizon=900, xy_extent_percent=100, group_radius_m=None,
        gpus=[6, 7], workers_per_gpu=1, relation_ids=dict(enabling=2, coupled=3),
        inference_only=True, new_runs=6, reused_runs=6, jobs=jobs, completed=False)
    write(OUTPUT/'manifest.json', manifest)
    state = dict(completed=False, expected_runs=6, verified=[], running={}, failed=[], pending=6)
    state_lock, validation_lock = threading.Lock(), threading.Lock()

    def save():
        state['updated_unix'] = time.time()
        write(OUTPUT/'status.json', state)
        report(state)

    save()

    def worker(gpu, condition):
        for job in (j for j in jobs if j['condition'] == condition):
            while True:
                if state['failed']:
                    return
                free = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free',
                                                     '--format=csv,noheader,nounits'], text=True).strip())
                if free >= 20000:
                    break
                time.sleep(10)
            try:
                process = launch(job, gpu)
                with state_lock:
                    state['pending'] -= 1
                    state['running'][job['key']] = dict(pid=process.pid, gpu=gpu, started_unix=time.time())
                    save()
                print('START', job['key'], 'GPU', gpu, 'PID', process.pid, flush=True)
                if process.wait():
                    raise RuntimeError(f"Evaluation failed: logs/{job['key']}.log")
                with validation_lock:
                    proof = previous.validate(job)
                    intervention = read(destination(job)/'depbias_ablation.json')
                    expected_id = manifest['relation_ids']['enabling' if condition == 'enabling_off' else 'coupled']
                    if intervention['relation_id'] != expected_id or not all(
                            c['target_zero'] and c['other_relations_exact'] for c in intervention['checks'].values()):
                        raise ValueError('Selective intervention validation failed')
                    write(OUTPUT/'checks'/f"{job['key']}.json", proof)
                with state_lock:
                    state['verified'].append(job['key'])
                    state['running'].pop(job['key'])
                    save()
                print('VERIFIED', job['key'], flush=True)
            except Exception as error:
                with state_lock:
                    state['failed'].append(dict(key=job['key'], error=str(error)))
                    state['running'].pop(job['key'], None)
                    save()
                print('FAILED', job['key'], str(error), flush=True)
                return

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, gpu, condition) for gpu, condition in zip((6, 7), CONDITIONS)]
        for future in futures:
            future.result()
    if state['failed'] or len(state['verified']) != 6:
        raise RuntimeError('Evaluation incomplete; see status.json')
    if any(sha(Path(path)) != digest for path, digest in hashes.items()):
        raise ValueError('Checkpoint/config/code/reference or reused artifact changed')
    manifest['completed'] = state['completed'] = True
    write(OUTPUT/'manifest.json', manifest)
    save()
    write(OUTPUT/'completion_verification.json', dict(completed=True, new_runs=6, reused_runs=6,
        cpu_recomputed_all_12=True, first_inputs_exact=True, policy_and_normalization_preserved=True,
        selective_bias_verified_all_frames=True, native_four_person_layout_verified=True,
        checkpoint_config_code_reference_and_reused_artifact_hashes_unchanged=True))


if __name__ == '__main__':
    main()
