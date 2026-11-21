import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from single_size_results import ROOT, MODELS, SEEDS, geometry, load, read, write, verify, summarize
import torch
from single_carry_sizes import CARRY_MODE, NINE_SIZES

OUTPUT = ROOT/'output_etc/check/single_size_native_common_env512_seeds167_20261120'
if CARRY_MODE == 'original9':
    OUTPUT = ROOT/'output_etc/check/single_size_original9_native_common_env512_seeds167_20261120'
OURS = ROOT/'output/stage2_distill_900_v4/Stage2Distill900V4_11-21-37-16/nn/Stage2Distill900V4_00002000.pth'
CHECKPOINTS = dict(ours=OURS, sct=ROOT/'pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth',
    coopft=ROOT.parent/'sct_coopft/output/sct_coopft/SCTCoopFT_10-22-24-41/nn/SCTCoopFT_00007000.pth',
    tokenhsi=ROOT.parent/'tokenhsi_push/checkpoints/TokenHSIPushKD_00040500.pth')
GPUS = (5, 6, 7) if CARRY_MODE == 'original9' else (5, 6)
LIMIT = 85000
RESERVE = 5500


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jobs():
    return [dict(variant=v, model=m, seed=s, key=f'{v}/{m}/seed{s}', phase='formal', steps=900)
            for v in ('native', 'common') for m in MODELS for s in SEEDS]


def destination(job):
    if job['phase'] == 'reference':
        return OUTPUT/'reference'/job['variant']/f'general_seed{job["seed"]}'
    return OUTPUT/('development' if job['phase']=='probe' else 'rollouts')/job['variant']/job['model']/f'general_seed{job["seed"]}'


def command(job, gpu):
    env = dict(os.environ)
    for key in ('CUDA_MPS_PIPE_DIRECTORY', 'CUDA_MPS_LOG_DIRECTORY', 'REFERENCE_ONLY', 'TABLE2_SETTING',
                'STAGE1_POLICY_CHECKPOINT', 'FSM_STAGE1_CHECKPOINT', 'CHECK', 'MAX_ITERATIONS',
                'RESUME_CHECKPOINT', 'ENV_CONFIG', 'STAGE1_CHECKPOINT', 'STAGE2_SAMPLING_AUDIT',
                'BIAS_ABLATION', 'GROUP_SPAWN_RADIUS', 'SPACING_METERS'):
        env.pop(key, None)
    env.update(TOKENHSI_GPU=str(gpu), STAGE2_SCENE_MODE='independent', SINGLE_SIZE=job['variant'],
        SINGLE_MODEL=job['model'], SINGLE_STEPS=str(job['steps']), SINGLE_CARRY=CARRY_MODE,
        STAGE2_REFERENCE_ROOT=str(OUTPUT/'reference'/job['variant']),
        REFERENCE_ONLY=str(int(job['phase']=='reference')), PYTHONHASHSEED=str(job['seed']),
        PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    args = [sys.executable, str(ROOT/'eval/single_size_eval.py'), '--task', 'HumanoidMACoupledPush',
        '--cfg_train', 'tokenhsi/data/cfg/train/rlg/stage2_distill_900_v4.yaml', '--cfg_env', '',
        '--motion_file', 'tokenhsi/data/dataset_stage1_jointpush.yaml', '--checkpoint', str(OURS),
        '--seed', str(job['seed']), '--num_envs', '512', '--num_agents', '1', '--num_objects', '2',
        '--test', '--eval', '--headless', '--no_video', '--eval_protocol', 'general', '--eval_repeats', '1',
        '--task_graph', 'random_scenario', '--enable_size_mode', 'per_action_intersection',
        '--coupled_assignment', 'ordered', '--output_path', str(destination(job))]
    return ['bash', '-c', 'source tokenhsi/scripts/multi_agent/runtime_env.sh\nexec "$@"', '_', *args], env


def memory():
    lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used',
        '--format=csv,noheader,nounits'], text=True).splitlines()
    rows = [line.split(',') for line in lines]
    used = {int(i):int(amount) for i, uid, amount in rows if int(i) in GPUS}
    uuids = {uid.strip():int(i) for i, uid, amount in rows if int(i) in GPUS}
    processes = {}
    for line in subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                                       '--format=csv,noheader,nounits'], text=True).splitlines():
        uid, pid, amount = [x.strip() for x in line.split(',')]
        if uid in uuids and amount.isdigit():
            processes[int(pid)] = int(amount)
    return used, processes


def run_phase(phase_jobs, state):
    pending = phase_jobs.copy()
    running = {}
    while pending or running:
        used, allocations = memory()
        state['gpu_memory_mib'] = used
        with (OUTPUT/'memory_observations.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(unix=time.time(), memory_mib=used))+'\n')
        for pid, (process, job, gpu) in list(running.items()):
            if process.poll() is None:
                continue
            path = destination(job)
            if process.returncode:
                state['failed'].append(dict(job=job, exit_code=process.returncode))
            else:
                try:
                    if job['phase'] == 'reference':
                        proof = dict(geometry=geometry(path, job['variant']))
                    else:
                        reference = OUTPUT/'reference'/job['variant']/f'general_seed{job["seed"]}'
                        proof = verify(path, reference, job['variant'], job['model'], job['steps'])
                    write(path/'verification.json', proof)
                    state['done'].append(job['phase']+'/'+job['key'])
                    if job['phase'] == 'formal':
                        state['verified'].append(job['key'])
                        summarize(OUTPUT, state['verified'])
                except Exception as error:
                    state['failed'].append(dict(job=job, error=repr(error)))
            del running[pid]
        if state['failed']:
            pending.clear()
        for gpu in sorted(GPUS, key=lambda g: used[g]):
            while pending:
                reserve = 0
                for pid, (_, job, assigned) in running.items():
                    if assigned == gpu:
                        ready = (destination(job)/'first_policy_input.json').exists()
                        reserve += 512 if ready else max(RESERVE-allocations.get(pid, 0), 0)
                if used[gpu]+reserve+RESERVE+1024 > LIMIT:
                    break
                job = pending.pop(0)
                path = destination(job)
                path.mkdir(parents=True, exist_ok=False)
                cmd, env = command(job, gpu)
                logfile = OUTPUT/'logs'/(job['phase']+'_'+job['key'].replace('/', '_')+'.log')
                with logfile.open('w') as log:
                    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT)
                running[proc.pid] = (proc, job, gpu)
                write(path/'execution.json', dict(pid=proc.pid, gpu=gpu, command=cmd,
                    started_unix=time.time(), log=str(logfile)))
                print('START', job['phase'], job['key'], 'GPU', gpu, 'PID', proc.pid, flush=True)
        state.update(running={str(pid):dict(job=j, gpu=g) for pid, (_,j,g) in running.items()}, pending=len(pending))
        write(OUTPUT/'status.json', state)
        if pending or running:
            time.sleep(5)
    if state['failed']:
        raise RuntimeError('Evaluation/verification failed; see status and logs')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT/'logs').mkdir(exist_ok=True)
    lock = (OUTPUT/'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    all_jobs = jobs()
    if (OUTPUT/'status.json').exists():
        raise ValueError('Output already contains a run; preserve existing results')
    sources = [ROOT/'eval'/name for name in ('single_size_eval.py', 'single_size_results.py', 'run_single_size.py',
        'single_carry_sizes.py',
        'stage2_protocol_sampling.py', 'stage2_scalability_layout.py', 'stage2_distractor_physics.py',
        'stage1_protocol_policy.py', 'stage2_full_observation.py', 'stage2_xy100.py')]
    sources += [ROOT/'FSM/dct'/name for name in ('policy.py', 'common.py')]
    sources += [ROOT/'tokenhsi'/name for name in ('learning/multi_agent/formal_eval.py',
        'utils/evaluation_protocol.py', 'utils/evaluation_errors.py', 'utils/coupled_push_spec.py')]
    sources += [ROOT.parent/'tokenhsi_push/scripts/tokenhsi_table_policy.py', *CHECKPOINTS.values()]
    original_config = ROOT.parent/'tokenhsi_push/tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml'
    sources.append(original_config)
    if CARRY_MODE == 'original9':
        import yaml
        config = yaml.safe_load(original_config.read_text())
        def find_sizes(value):
            if isinstance(value, dict):
                if 'testSizes' in value:
                    return value['testSizes']
                for child in value.values():
                    found = find_sizes(child)
                    if found is not None:
                        return found
        if find_sizes(config) != [list(size) for size in NINE_SIZES]:
            raise ValueError('Original carry testSizes differs')
    hashes = {str(p):sha(p) for p in sources}
    manifest = dict(envs=512, agents=1, objects=2, seeds=SEEDS, steps=900, gpus=GPUS,
        gpu_total_limit_mib=LIMIT, reservation_mib=RESERVE, jobs=all_jobs, expected_runs=24,
        checkpoints={m:str(p) for m,p in CHECKPOINTS.items()}, source_sha256=hashes,
        carry_sizes=CARRY_MODE, original9_sizes=NINE_SIZES if CARRY_MODE == 'original9' else None,
        variants=dict(native='Table1 sizes with selected carry protocol', common='Enable common sizes with selected carry protocol; PUSH unchanged'),
        started_unix=time.time())
    write(OUTPUT/'manifest.json', manifest)
    if args.prepare_only:
        print('Prepared 24 jobs, checkpoints/sources present', flush=True)
        return
    (OUTPUT/'supervisor.pid').write_text(str(os.getpid())+'\n')
    state = dict(completed=False, done=[], verified=[], failed=[], phase='reference', started_unix=time.time())
    summarize(OUTPUT, [])
    references = [dict(variant=v, seed=s, model='ours', steps=900, phase='reference', key=f'{v}/seed{s}')
                  for v in ('native', 'common') for s in SEEDS]
    run_phase(references, state)
    for seed in SEEDS:
        a = load(OUTPUT/f'reference/native/general_seed{seed}/initial.pt')
        b = load(OUTPUT/f'reference/common/general_seed{seed}/initial.pt')
        if not torch.equal(a['packet'], b['packet']):
            raise ValueError('Size conditions changed task assignment')
    probes = [dict(j, phase='probe', steps=60) for j in all_jobs if j['seed']==1 and
              (j['variant']=='common' or j['model']=='ours')]
    state['phase'] = 'development'
    run_phase(probes, state)
    state['phase'] = 'formal'
    run_phase(all_jobs, state)
    if any(sha(p) != value for p,value in hashes.items()):
        raise ValueError('A tracked source/checkpoint changed during evaluation')
    for v in ('native', 'common'):
        for seed in SEEDS:
            counts = [[read(OUTPUT/f'rollouts/{v}/{m}/general_seed{seed}/result.json')['scores']['region'][t]['trials']
                       for t in ('SIT', 'CLIMB', 'AT', 'ON_TOP', 'PUSH')] for m in MODELS]
            if any(x != counts[0] for x in counts):
                raise ValueError('Model task counts differ')
    state.update(completed=True, phase='complete', finished_unix=time.time())
    write(OUTPUT/'status.json', state)
    write(OUTPUT/'completion_verification.json', dict(formal_runs=len(state['verified']),
        expected_runs=24, episodes=24*512, all_sources_unchanged=True, model_task_counts_exact=True,
        shared_references=6, development_runs=len(probes), finished_unix=time.time()))
    summarize(OUTPUT, state['verified'])
    print('COMPLETE 24/24', flush=True)


if __name__ == '__main__':
    main()
