import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import run_v4_group_radius_eval as existing
from run_group_spacing_formal import memory

ROOT = existing.ROOT
OUTPUT = ROOT/'output_etc/check/v4_2000_32h64h_radius_env64_seed1_20261112'
CHECKPOINT = existing.RUN/'nn/Stage2Distill900V4_00002000.pth'
existing.OUTPUT = OUTPUT
STATES = {}


def write(path, data):
    existing.write(path, data)


def command(job, path, reference):
    entry = 'group_radius_probe.py' if reference else 'group_spacing_eval.py'
    return ['bash', '-c', 'source tokenhsi/scripts/multi_agent/runtime_env.sh\nexec "$@"', '_',
        sys.executable, str(ROOT/'eval'/entry), '--task', 'HumanoidMACoupledPush',
        '--cfg_train', 'tokenhsi/data/cfg/train/rlg/stage2_distill_900_v4.yaml', '--cfg_env', '',
        '--motion_file', 'tokenhsi/data/dataset_stage1_jointpush.yaml', '--checkpoint', str(CHECKPOINT),
        '--seed', '1', '--num_envs', '64', '--num_agents', str(job['people']),
        '--num_objects', str(2*job['people']), '--test', '--eval', '--headless', '--no_video',
        '--eval_protocol', 'general', '--eval_repeats', '1', '--task_graph', 'random_scenario',
        '--enable_size_mode', 'per_action_intersection', '--coupled_assignment', 'ordered',
        '--group_spawn_over4', '--output_path', str(path)]


def execute(job, path, reference):
    path.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TOKENHSI_GPU=str(job['gpu']),
        SPACING_MODEL='reference' if reference else 'ours', SPACING_METERS='native',
        GROUP_SPAWN_RADIUS=str(job['radius']), STAGE2_SCENE_MODE=job['mode'],
        STAGE2_REFERENCE_ROOT=str(OUTPUT/'reference'/job['name']),
        SPACING_PROBE_ATTEMPTS='1024', PYTHONHASHSEED='1', PYTHONDONTWRITEBYTECODE='1',
        PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    for key in ('REFERENCE_ONLY', 'TABLE2_SETTING', 'STAGE1_POLICY_CHECKPOINT',
                'FSM_STAGE1_CHECKPOINT', 'CHECK', 'MAX_ITERATIONS', 'RESUME_CHECKPOINT',
                'STAGE2_SAMPLING_AUDIT', 'BIAS_ABLATION'):
        env.pop(key, None)
    if reference:
        env['REFERENCE_ONLY'] = '1'
    start = time.monotonic()
    with (path/'run.log').open('w') as log:
        process = subprocess.Popen(command(job, path, reference), cwd=ROOT, env=env,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write(path/'process.json', dict(pid=process.pid, gpu=job['gpu'], job=job, reference=reference))
        STATES[job['key']] = dict(job, phase='layout' if reference else 'evaluation',
            pid=process.pid, source=str(path), start_unix=time.time())
        while process.poll() is None:
            if reference and time.monotonic()-start > 180:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                return dict(status='layout_timeout', elapsed_s=time.monotonic()-start)
            time.sleep(2)
    return dict(status='complete' if process.returncode == 0 else 'error',
        exit_code=process.returncode, elapsed_s=time.monotonic()-start)


def lane(people, gpu):
    for mode in ('independent', 'mixed'):
        key = f'{mode}_{people}h'
        job = dict(key=key, name=key, people=people, gpu=gpu, mode=mode, epoch=2000,
            checkpoint=str(CHECKPOINT), seed=1, envs=64)
        attempts = []
        try:
            while memory()[gpu]['free'] < 30000:
                STATES[key] = dict(job, phase='waiting_for_30GB_free')
                time.sleep(10)
            for radius in range(10, 101, 5):
                job['radius'] = radius
                folder = OUTPUT/'layout_attempts'/key/f'radius{radius}'/'general_seed1'
                result = execute(job, folder, True)
                probe = folder/'sampling_probe.json'
                if probe.exists():
                    rows = existing.read(probe)['rows']
                    result['attempts'] = len(rows)
                    result['remaining_scenes'] = rows[-1]['remaining'] if rows else None
                result.update(radius=radius, source=str(folder))
                attempts.append(result)
                write(OUTPUT/f'{key}_attempts.json', attempts)
                print(key, result, flush=True)
                if result['status'] == 'complete':
                    proof = existing.read(folder/'radius_verification.json')
                    if not proof['radius_verified']:
                        raise ValueError('Reference radius verification failed')
                    ref = OUTPUT/'reference'/key
                    ref.mkdir(parents=True, exist_ok=True)
                    (ref/'general_seed1').symlink_to(folder, target_is_directory=True)
                    job['layout_elapsed_s'] = result['elapsed_s']
                    break
                log = (folder/'run.log').read_text(errors='replace')
                if result['status'] == 'error' and not ('layout' in log.lower() and 'RuntimeError' in log):
                    raise RuntimeError('Non-layout error; inspect '+str(folder/'run.log'))
            else:
                raise RuntimeError('No layout accepted through R=100m')
            dest = existing.destination(job)
            result = execute(job, dest, False)
            if result['status'] != 'complete':
                raise RuntimeError('Evaluation failed; inspect '+str(dest/'run.log'))
            STATES[key] = dict(job, phase='cpu_verification', source=str(dest))
            existing.validate(job)
            STATES[key] = dict(job, phase='complete', source=str(dest),
                evaluation_elapsed_s=result['elapsed_s'], attempts=attempts)
        except Exception as error:
            STATES[key] = dict(job, phase='failed', error=str(error), attempts=attempts)
            print(key, 'FAILED', str(error), flush=True)


def report():
    states = dict(STATES)
    lines = ['# Ours V4 2k · 32H/64H 탐색 평가', '',
        '64환경·seed1·900step·XY100%·M=2N. GPU6/7. 조건별 R=10m부터5m 증가.',
        '배치 제한180초는 프로세스 준비·모델 로딩까지 포함. 실패/시간 초과 기록은 보존한다.',
        '성능이 아닌 배치 생성으로 반경 선택. 단일 seed이며 표준편차 없음.', '',
        '| 인원 | 모드 | 반경(m) | 배치 준비(s) | Independent SR | Enabling Group SR | Coupled Group SR | Mixed Group SR | 상태 |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---|']
    results = []
    for key, state in sorted(states.items()):
        cells = ['—']*4
        if state['phase'] == 'complete':
            result = existing.read(Path(state['source'])/'result.json')
            values = result['interaction_group_scores']['region']
            metrics = ('Independent Interaction Success Rate', 'Enabling Interaction Group Success Rate',
                'Coupled Interaction Group Success Rate', 'Mixed Interaction Group Success Rate')
            cells = ['N/A' if values[k]['rate'] is None else f"{100*values[k]['rate']:.2f}" for k in metrics]
            results.append(dict(job=state, scores=values, errors=result['errors']))
        lines.append(f"| {state['people']} | {state['mode']} | {state.get('radius', '—')} | "
            + (f"{state['layout_elapsed_s']:.1f}" if 'layout_elapsed_s' in state else '—')
            + ' | '+' | '.join(cells)+f" | {state['phase']} |")
    write(OUTPUT/'status.json', states)
    write(OUTPUT/'summary.json', dict(envs=64, seeds=[1], horizon=900, runs=results,
        completed=len(results), expected=4))
    (OUTPUT/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Run supervisor with CUDA_VISIBLE_DEVICES=')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT/'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUTPUT/'manifest.json').exists():
        raise ValueError('Existing run; inspect status instead of overwriting')
    sources = [Path(__file__), ROOT/'eval/group_spacing_eval.py', ROOT/'eval/group_radius_probe.py',
        ROOT/'eval/group_spacing_sampling_probe.py', ROOT/'eval/run_v4_group_radius_eval.py',
        ROOT/'eval/stage2_protocol_sampling.py', ROOT/'tokenhsi/learning/multi_agent/formal_eval.py']
    hashes = {str(p): existing.sha(p) for p in sources}
    checkpoint_hash = existing.sha(CHECKPOINT)
    write(OUTPUT/'manifest.json', dict(protocol='cvpr2027-evaluation-v1.1.0',
        extension='32H/64H adaptive group radius screen; not standard Table1 cohort',
        checkpoint=str(CHECKPOINT), checkpoint_sha256=checkpoint_hash, source_sha256=hashes,
        envs=64, seeds=[1], horizon=900, gpus=[6, 7], radius_start_m=10,
        radius_increment_m=5, radius_search_ceiling_m=100, layout_timeout_including_startup_s=180,
        jobs=[dict(people=n, mode=m) for n in (32,64) for m in ('independent','mixed')]))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(lane, 32, 6), pool.submit(lane, 64, 7)]
        while not all(f.done() for f in futures):
            report()
            time.sleep(10)
        for f in futures:
            f.result()
    report()
    write(OUTPUT/'completion_verification.json', dict(completed=sum(s['phase']=='complete' for s in STATES.values()),
        expected=4, checkpoint_unchanged=existing.sha(CHECKPOINT)==checkpoint_hash,
        source_unchanged={p:existing.sha(Path(p))==v for p,v in hashes.items()}))


if __name__ == '__main__':
    main()
