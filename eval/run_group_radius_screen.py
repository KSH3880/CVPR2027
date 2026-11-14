import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_group_spacing_formal import memory
from run_group_spacing import CARRIER, ROOT


def run_job(job, output, timeout, attempts):
    path = output/f"{job['people']}h"/f"radius{job['radius']:g}"/job['mode']/f"seed{job['seed']}"
    path.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TOKENHSI_GPU=str(job['gpu']), SPACING_MODEL='reference',
        SPACING_METERS='native', GROUP_SPAWN_RADIUS=str(job['radius']), STAGE2_SCENE_MODE=job['mode'],
        REFERENCE_ONLY='1', SPACING_PROBE_ATTEMPTS=str(attempts), PYTHONHASHSEED=str(job['seed']),
        PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    for key in ('CHECK', 'MAX_ITERATIONS', 'RESUME_CHECKPOINT', 'STAGE1_POLICY_CHECKPOINT',
                'FSM_STAGE1_CHECKPOINT', 'STAGE2_SAMPLING_AUDIT', 'BIAS_ABLATION'):
        env.pop(key, None)
    command = [sys.executable, str(ROOT/'eval/group_radius_probe.py'), '--task', 'HumanoidMACoupledPush',
        '--cfg_train', 'tokenhsi/data/cfg/train/rlg/stage2_distill_900_v2.yaml', '--cfg_env', '',
        '--motion_file', 'tokenhsi/data/dataset_stage1_jointpush.yaml', '--checkpoint', str(CARRIER),
        '--seed', str(job['seed']), '--num_envs', str(job['envs']), '--num_agents', str(job['people']),
        '--num_objects', str(2*job['people']), '--test', '--eval', '--headless', '--no_video',
        '--eval_protocol', 'general', '--eval_repeats', '1', '--task_graph', 'random_scenario',
        '--enable_size_mode', 'per_action_intersection', '--coupled_assignment', 'ordered',
        '--group_spawn_over4', '--output_path', str(path)]
    start = time.monotonic()
    with (path/'run.log').open('w') as log:
        process = subprocess.Popen(['bash', '-c', 'source tokenhsi/scripts/multi_agent/runtime_env.sh\nexec "$@"',
            '_', *command], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        (path/'pid').write_text(str(process.pid)+'\n')
        try:
            code = process.wait(timeout=timeout)
            status = 'complete' if code == 0 else 'attempt_limit_or_error'
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                code = process.wait()
            status = 'time_limit'
    probe = json.loads((path/'sampling_probe.json').read_text()) if (path/'sampling_probe.json').exists() else {}
    proof = json.loads((path/'radius_verification.json').read_text()) if (path/'radius_verification.json').exists() else {}
    if status == 'complete' and not proof.get('radius_verified'):
        status = 'verification_failed'
    result = dict(job, status=status, elapsed_s=time.monotonic()-start, exit_code=code, source=str(path),
        attempts=len(probe.get('rows', [])), remaining_scenes=probe.get('rows', [{}])[-1].get('remaining'),
        radius_verification=proof)
    (path/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def report(output, jobs, results, args):
    state = dict(completed=len(results) == len(jobs), expected_runs=len(jobs), finished=len(results),
        envs=args.envs, seeds=args.seeds, time_limit_s=args.timeout, attempt_limit=args.attempts,
        purpose='Initial layout feasibility only; no policy rollout/SR', jobs=jobs, results=results)
    temporary = output/'status.pending'
    temporary.write_text(json.dumps(state, indent=2)+'\n');temporary.replace(output/'status.json')
    lines = ['# 그룹 중심 고정 반경 · 배치 생성 검사', '',
        f'{args.envs}환경 × seeds{args.seeds}. 실행당 제한{args.timeout:g}초·최대{args.attempts}회 재시도.',
        '시간에는 초기 실행 준비가 포함된다. 제한 초과는 기하학적 불가능의 증명이 아니다. 정책 SR은 측정하지 않는다.', '',
        '| 인원 | 반경(m) | 모드 | seed | 결과 | 시간(s) | 재시도 횟수 | 미완성 장면 |',
        '|---:|---:|---|---:|---|---:|---:|---:|']
    for row in sorted(results, key=lambda x: (x['people'], x['radius'], x['mode'], x['seed'])):
        lines.append(f"| {row['people']} | {row['radius']:g} | {row['mode']} | {row['seed']} | {row['status']} | {row['elapsed_s']:.1f} | {row['attempts']} | {row['remaining_scenes']} |")
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--envs', type=int, default=64)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1])
    parser.add_argument('--radii8', type=float, nargs='*', default=[1.5, 2.5, 3.5])
    parser.add_argument('--radii16', type=float, nargs='*', default=[4., 6., 8.])
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--attempts', type=int, default=64)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('Use CPU supervisor: CUDA_VISIBLE_DEVICES=')
    output = args.output.resolve();output.relative_to(ROOT/'output_etc/check');output.mkdir(parents=True, exist_ok=True)
    jobs = [dict(people=n, radius=r, mode=mode, seed=seed, envs=args.envs)
        for n, radii in [(8, args.radii8), (16, args.radii16)] for r in radii
        for mode in ('independent', 'mixed') for seed in args.seeds]
    if not jobs or len(jobs) > 12:
        parser.error('Run one batch of 1..12 jobs')
    for i, job in enumerate(jobs):
        job['gpu'] = (5, 6, 7)[i % 3]
    snapshot = memory()
    for gpu in (5, 6, 7):
        required = (16000 if gpu == 5 else 12000)+sum(8000 if j['envs'] <= 64 else 14000 for j in jobs if j['gpu'] == gpu)
        if snapshot[gpu]['free'] < required:
            raise RuntimeError(f'GPU{gpu} free memory below startup budget; no jobs launched')
    sources = [ROOT/'eval'/name for name in ('run_group_radius_screen.py', 'group_radius_probe.py',
        'group_spacing_eval.py', 'group_spacing_sampling_probe.py', 'stage2_protocol_sampling.py')]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (output/'manifest.json').write_text(json.dumps(dict(jobs=jobs, envs=args.envs, seeds=args.seeds,
        source_sha256=hashes, policy_rollout=False, group_placement='fixed_radius_group_center_ring',
        original_people_spawn_radius_m=2.5, within_group_rule_retained=True,
        followup_enable_layout_retained=True, wall_timeout_s=args.timeout), indent=2)+'\n')
    results = [];report(output, jobs, results, args)
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        for future in as_completed([pool.submit(run_job, j, output, args.timeout, args.attempts) for j in jobs]):
            row = future.result();results.append(row);report(output, jobs, results, args)
            print(row['people'], row['radius'], row['mode'], row['seed'], row['status'],
                  round(row['elapsed_s'], 1), row['remaining_scenes'], flush=True)
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != value for p, value in hashes.items()):
        raise RuntimeError('Probe sources changed during execution')


if __name__ == '__main__':
    main()
