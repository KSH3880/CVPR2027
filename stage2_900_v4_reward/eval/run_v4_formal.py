import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from run_stage2_scalability import ROOT, verify
from run_v4_group_radius_eval import sha, read, write
from table2_metrics import replay
from coupled_cooperation_metrics import calculate
import stage2_xy100


OUTPUT = ROOT/'output_etc/check/v4_2000_tables12_radius_seeds167_20261112'
OLD = ROOT/'output_etc/check/v3_2000_tables12_seeds167_20261110'
RADIUS = ROOT/'output_etc/check/table1_group_radius_8r3_16r7p5_env512_seeds167_baselines_20261112'
CHECKPOINT = ROOT/'output/stage2_distill_900_v4/Stage2Distill900V4_11-21-37-16/nn/Stage2Distill900V4_00002000.pth'
TEAMHOI = ROOT/'output_etc/check/coupled_cooperation_four_models_20261111/reference'
GPUS = (5, 6, 7)
SEEDS = (1, 6, 7)
SETTINGS = dict(enable_sit=2, enable_climb=2, enable_ontop=2, coupled_2=2, coupled_3=3, coupled_4=4)
RESERVATIONS = {2: 5000, 3: 6000, 4: 7000, 8: 10000, 16: 14000}


def jobs():
    result = [dict(table='table1', name=f'{mode}_{n}h', mode=mode, people=n, seed=seed,
                   radius={8: 3., 16: 7.5}.get(n))
              for n in (2, 4, 8, 16) for mode in ('independent', 'mixed') for seed in SEEDS]
    result += [dict(table='table2', name=name, mode='mixed', people=n, seed=seed, radius=None)
               for name, n in SETTINGS.items() for seed in SEEDS]
    for job in result:
        job['key'] = f"{job['table']}/{job['name']}/seed{job['seed']}"
        base = RADIUS if job['radius'] is not None else OLD
        job['reference'] = str(base/'reference'/job['name']/f"general_seed{job['seed']}"
            if base == RADIUS else base/'reference'/job['table']/job['name']/f"general_seed{job['seed']}")
    return result


def destination(job):
    return OUTPUT/job['table']/job['name']/f"general_seed{job['seed']}"


def proof_path(job):
    return OUTPUT/'checks'/job['table']/job['name']/f"seed{job['seed']}.json"


def ready(job):
    path = Path(job['reference'])
    if not all((path/name).is_file() for name in ('reference_info.json', 'initial.pt', 'observed_scene.pt')):
        return False
    return job['radius'] is None or (RADIUS/'checks/reference'/job['name']/f"seed{job['seed']}.json").is_file()


def prepare_reference(job, hashes):
    source = Path(job['reference'])
    info = read(source/'reference_info.json')
    if (info['envs'], info['agents'], info['objects'], info['seed'], info['simulated_steps']) != (
            512, job['people'], 2*job['people'], job['seed'], 0):
        raise ValueError('Reference dimensions differ')
    if job['radius'] is not None:
        spacing = read(source/'group_spacing.json')
        if spacing['group_spawn_radius'] != job['radius']:
            raise ValueError('Reference radius differs')
    target = OUTPUT/'reference'/job['table']/job['name']/source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.symlink_to(source.resolve(), target_is_directory=True)
    for name in ('reference_info.json', 'initial.pt', 'observed_scene.pt'):
        hashes[str(source/name)] = sha(source/name)
    write(OUTPUT/'reference_hashes.json', hashes)


def launch(job, gpu):
    path = destination(job)
    path.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TOKENHSI_GPU=str(gpu), STAGE2_SCENE_MODE=job['mode'],
        STAGE2_REFERENCE_ROOT=str(Path(job['reference']).parent), PYTHONHASHSEED=str(job['seed']),
        PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    for name in ('REFERENCE_ONLY', 'TABLE2_SETTING', 'STAGE1_POLICY_CHECKPOINT',
                 'FSM_STAGE1_CHECKPOINT', 'CHECK', 'MAX_ITERATIONS', 'RESUME_CHECKPOINT',
                 'GROUP_SPAWN_RADIUS', 'SPACING_METERS', 'STAGE2_SAMPLING_AUDIT', 'BIAS_ABLATION'):
        env.pop(name, None)
    entry = ROOT/'eval/v4_formal_eval.py'
    if job['radius'] is not None:
        env.update(SPACING_MODEL='ours', SPACING_METERS='native', GROUP_SPAWN_RADIUS=str(job['radius']))
        entry = ROOT/'eval/group_spacing_eval.py'
    if job['table'] == 'table2':
        env['TABLE2_SETTING'] = job['name']
    command = [sys.executable, str(entry), '--task', 'HumanoidMACoupledPush', '--cfg_train',
        'tokenhsi/data/cfg/train/rlg/stage2_distill_900_v4.yaml', '--cfg_env', '', '--motion_file',
        'tokenhsi/data/dataset_stage1_jointpush.yaml', '--checkpoint', str(CHECKPOINT), '--seed', str(job['seed']),
        '--num_envs', '512', '--num_agents', str(job['people']), '--num_objects', str(2*job['people']),
        '--test', '--eval', '--headless', '--no_video', '--eval_protocol', 'general', '--eval_repeats', '1',
        '--task_graph', 'random_scenario', '--enable_size_mode', 'per_action_intersection',
        '--coupled_assignment', 'ordered', '--output_path', str(path)]
    if job['people'] > 4:
        command.append('--group_spawn_over4')
    log = OUTPUT/'logs'/f"{job['table']}_{job['name']}_seed{job['seed']}.log"
    with log.open('w') as stream:
        process = subprocess.Popen(['bash', '-c',
            'source tokenhsi/scripts/multi_agent/runtime_env.sh\nexec "$@"', '_', *command],
            cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    return process


def validate(job):
    path = destination(job)
    proof = verify(path, Path(job['reference']), job['people'], 512)
    record = read(path/'result.json')
    policy = read(path/'policy_preservation.json')
    if record['experiment'] != 'stage2_distill_900_v4' or record['seed'] != job['seed']:
        raise ValueError('Experiment/seed differs')
    if record['encoder_attention_checks'] != dict(actor=900, critic=900):
        raise ValueError('Unexpected attention call counts')
    if record['distractor_attention'] != 'all_observed_tokens' or record['enabling_success'] != 'simultaneous':
        raise ValueError('Attention or success definition differs')
    if not read(path/'first_policy_input.json')['exact'] or policy['epoch'] != 2000:
        raise ValueError('Actual first input or checkpoint differs')
    if not all(policy[k] for k in ('weights_exact_before_after', 'normalization_exact_before_after')):
        raise ValueError('Policy preservation failed')
    if job['radius'] is not None:
        spacing = read(path/'group_spacing.json')
        if spacing['group_spawn_radius'] != job['radius'] or spacing['group_spawn_gap'] is not None:
            raise ValueError('Wrong group radius')
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
        proof.update(group_radius_verified=True, maximum_radius_error_m=radius_error,
                     group_ring_analytic_gap_verified=True)
        record['comparison_protocol'] = 'cvpr2027_table1_fixed_group_radius_env512'
    if job['table'] == 'table2':
        write(path/'table2_recomputed.json', replay(path))
        proof['cpu_gpu_table2_exact'] = True
        table = read(path/'table2.json')
        table['policy'] = 'stage2_v4_2k'
        write(path/'table2.json', table)
        record['table2'] = table
        if job['name'].startswith('coupled'):
            output = OUTPUT/'latest_cooperation'
            output.mkdir(exist_ok=True)
            latest = dict(model='ours_v4_2k', people=job['people'], seed=job['seed'], source=str(path),
                          name=f"v4_{job['people']}h_seed{job['seed']}")
            row = calculate(latest, str(output), str(TEAMHOI))
            proof['latest_cooperation_first_success_exact'] = row['first_success_exact']
    record['protocol_id'] = 'cvpr2027-evaluation-v1.1.0'
    write(path/'result.json', record)
    proof.update(first_policy_input_exact=True, policy_preserved=True, reference=job['reference'])
    proof_path(job).parent.mkdir(parents=True, exist_ok=True)
    write(proof_path(job), proof)


def partial(all_jobs, state):
    lines = ['# V4 2k · Table 1/2 정식 평가', '',
        f"512환경·seed1/6/7·900step·XY100%. 검증 {len(state['verified'])}/42회.", '',
        '| 표 | 설정 | seed1 | seed6 | seed7 |', '|---|---|---:|---:|---:|']
    for table, name in dict.fromkeys((j['table'], j['name']) for j in all_jobs):
        values = []
        for seed in SEEDS:
            job = next(j for j in all_jobs if (j['table'], j['name'], j['seed']) == (table, name, seed))
            if job['key'] not in state['verified']:
                values.append('—')
                continue
            index = (0 if job['mode'] == 'independent' else 3) if table == 'table1' else (1 if name.startswith('enable') else 2)
            metrics = ['Independent Interaction Success Rate', 'Enabling Interaction Group Success Rate',
                       'Coupled Interaction Group Success Rate', 'Mixed Interaction Group Success Rate']
            rate = read(destination(job)/'result.json')['interaction_group_scores']['region'][metrics[index]]['rate']
            values.append(f'{100*rate:.2f}')
        lines.append('| '+table+' | '+name+' | '+' | '.join(values)+' |')
    (OUTPUT/'partial_report.md').write_text('\n'.join(lines)+'\n')


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU supervisor requires CUDA_VISIBLE_DEVICES=')
    stage2_xy100.install()
    torch.set_num_threads(1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT/'logs').mkdir(exist_ok=True)
    lock = (OUTPUT/'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (OUTPUT/'supervisor.pid').write_text(str(os.getpid())+'\n')
    all_jobs = jobs()
    weights = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    if weights['epoch'] != 2000 or not all(torch.isfinite(t).all() for t in weights['model'].values()):
        raise ValueError('Invalid V4 checkpoint')
    del weights
    sources = [CHECKPOINT, ROOT/'eval/run_v4_formal.py', ROOT/'eval/v4_formal_eval.py',
        ROOT/'eval/run_v4_group_radius_eval.py', ROOT/'eval/group_spacing_eval.py',
        ROOT/'eval/summarize_v4_formal.py', ROOT/'eval/table2_eval.py', ROOT/'eval/table2_metrics.py',
        ROOT/'eval/coupled_cooperation_metrics.py', ROOT/'eval/run_stage2_scalability.py',
        ROOT/'eval/stage2_scalability_eval.py', ROOT/'eval/stage2_protocol_sampling.py',
        ROOT/'eval/stage2_scalability_layout.py', ROOT/'eval/stage2_distractor_physics.py',
        ROOT/'eval/stage2_xy100.py', ROOT.parent/'evaluation_protocol/README.md',
        TEAMHOI/'task_util.py', TEAMHOI/'torch_utils.py']
    hashes = {str(p): sha(p) for p in sources}
    old = OUTPUT/'source_hashes.json'
    if old.exists() and read(old) != hashes:
        raise ValueError('Sources changed; use a new output folder')
    write(old, hashes)
    references = read(OUTPUT/'reference_hashes.json') if (OUTPUT/'reference_hashes.json').exists() else {}
    manifest = dict(protocol_id='cvpr2027-evaluation-v1.1.0', checkpoint=str(CHECKPOINT),
        checkpoint_sha256=hashes[str(CHECKPOINT)], envs=512, seeds=SEEDS, horizon=900, xy_extent_percent=100,
        objects='2N', gpus=GPUS, group_spawn_radii_m={8: 3., 16: 7.5}, jobs=all_jobs,
        expected_runs=42, table1_runs=24, table2_runs=18, completed=False)
    write(OUTPUT/'manifest.json', manifest)
    state = dict(completed=False, verified=[], failed=[], running={}, pending=42, expected_runs=42,
        gpus=GPUS, phase='probes', started_unix=time.time())
    pending, running = all_jobs.copy(), {}
    for job in all_jobs:
        if proof_path(job).exists():
            validate(job)
            state['verified'].append(job['key'])
            pending.remove(job)
    probes = {'table1/mixed_16h/seed1', 'table2/coupled_3/seed1'}
    while pending or running:
        free = subprocess.check_output(['nvidia-smi', '-i', '5,6,7', '--query-gpu=index,memory.free',
            '--format=csv,noheader,nounits'], text=True)
        available = {int(line.split(',')[0]): int(line.split(',')[1]) for line in free.splitlines()}
        probing = not probes.issubset(state['verified'])
        state['phase'] = 'probes' if probing else 'full_evaluation'
        for gpu in sorted(GPUS, key=lambda g: (sum(x[2] == g for x in running.values()), -available[g])):
            active = [j for _, j, g in running.values() if g == gpu]
            reserved = sum(RESERVATIONS[j['people']] for j in active)
            reserve = 16000 if gpu == 5 else 12000
            candidates = [j for j in pending if ready(j) and (not probing or j['key'] in probes)]
            job = next((j for j in candidates if available[gpu]-reserved-reserve >= RESERVATIONS[j['people']]), None)
            if job and not state['failed'] and len(active) < (1 if probing else 3):
                prepare_reference(job, references)
                process = launch(job, gpu)
                pending.remove(job)
                running[job['key']] = process, job, gpu
                state['running'][job['key']] = dict(pid=process.pid, gpu=gpu, started_unix=time.time())
                print('START', job['key'], 'GPU'+str(gpu), 'PID', process.pid, flush=True)
        for key, (process, job, gpu) in list(running.items()):
            if process.poll() is None:
                continue
            try:
                if process.returncode:
                    raise RuntimeError(f'Evaluation exited {process.returncode}; see logs/{job["table"]}_{job["name"]}_seed{job["seed"]}.log')
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
        partial(all_jobs, state)
        if state['failed'] and not running:
            raise RuntimeError('Queue stopped after failure')
        if pending or running:
            time.sleep(5)
    if len(state['verified']) != 42 or any(sha(Path(p)) != digest for p, digest in {**hashes, **references}.items()):
        raise ValueError('Incomplete suite or source/reference changed')
    from summarize_v4_formal import summarize
    summarize(OUTPUT)
    write(OUTPUT/'completion_verification.json', dict(completed=True, runs=42, table1=24, table2=18,
        envs=512, seeds=SEEDS, gpus=GPUS, first_actual_policy_input_exact=True, policy_preserved=True,
        cpu_sr_counts_errors_verified=True, table2_cpu_replay_verified=True,
        group_radii_verified=True, latest20cm_cooperation_verified=True, source_reference_hashes_unchanged=True))
    state['completed'] = manifest['completed'] = True
    state['phase'] = 'complete'
    write(OUTPUT/'manifest.json', manifest)
    write(OUTPUT/'status.json', state)


if __name__ == '__main__':
    main()
