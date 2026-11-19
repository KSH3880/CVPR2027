from pathlib import Path
import sys

import yaml

import numpy as np

from suite import ROOT, SETTINGS, MODELS, LABELS, SEEDS, command as base_command
from suite import source_hashes, read, write, sha

OUT = ROOT/'runs/table89_delay/cohort'
CONDITION = 'random_1_5'
GPU_CAPS = {6: 88000, 7: 88000}
PROBES = (('enable_sit', 'sct'), ('enable_sit', 'fsm'), ('enable_sit', 'tokenhsi'),
          ('coupled_3', 'ours'), ('coupled_4', 'coopft'), ('coupled_4', 'fsm'))


def env_count(setting, seed):
    return 170 if list(SETTINGS).index(setting) % 3 == SEEDS.index(seed) else 171


def key(job):
    return f"{job['setting']}/{job['model']}_d{job['delay']}_seed{job['seed']}"


def jobs():
    return [dict(setting=s, seed=seed, model=m, delay=d)
        for seed in SEEDS for s in SETTINGS for d in (0, CONDITION) for m in MODELS]


def schedule(setting, seed, condition):
    index = list(SETTINGS).index(setting)
    rng_seed = [20261115, 512, seed, index]
    rng = np.random.default_rng(np.random.SeedSequence(rng_seed))
    n = env_count(setting, seed)
    requested = rng.uniform(1., 5., n) if condition == CONDITION else np.zeros(n)
    steps = np.rint(requested*30).astype(int)
    return dict(setting=setting, seed=seed, condition=condition, env_ids=list(range(n)),
        rng_seed=rng_seed, distribution='Uniform[1,5) rounded to 1/30 second' if condition else 'zero',
        requested_seconds=requested.tolist(), steps=steps.tolist(), actual_seconds=(steps/30).tolist())


def schedule_path(job):
    return OUT/'schedules'/f"{job['setting']}_seed{job['seed']}_d{job['delay']}.json"


def command(job, path, reference=False, steps=900, gpu=6):
    base = dict(job, delay=0)
    cmd, env = base_command(base, path, reference=reference, steps=steps, gpu=gpu)
    cmd[cmd.index('--num_envs')+1] = str(env_count(job['setting'], job['seed']))
    env.update(STAGE2_REFERENCE_ROOT=str(OUT/'reference'/job['setting']), DCT_OUTPUT=str(path))
    if not reference:
        for name in ('evaluate.py', 'tokenhsi_eval.py'):
            entry = str(ROOT/'FSM/dct'/name)
            if entry in cmd:
                cmd[cmd.index(entry)] = str(ROOT/'FSM/dct/cohort_eval.py')
        env['DCT_DELAY_SCHEDULE'] = str(schedule_path(job))
    return cmd, env


def cpu_command(job, path, destination, steps=900):
    return [sys.executable, str(ROOT/'FSM/dct/analyze_cohort.py'), '--source', str(path),
        '--destination', str(destination), '--setting', job['setting'], '--model', job['model'],
        '--seed', str(job['seed']), '--steps', str(steps), '--delay', str(job['delay'])]


def prepare():
    paper = read(ROOT/'FSM/dct/paper_manifest.json')
    checkpoint_hashes = {Path(path).name: digest for path, digest in paper['source_sha256'].items()
                         if path.endswith('.pth')}
    for checkpoint in set(MODELS.values()):
        if not checkpoint.is_file():
            raise FileNotFoundError(f'Extract the Table 8/9 inputs archive in the repository root: {checkpoint}')
        if sha(checkpoint) != checkpoint_hashes[checkpoint.name]:
            raise ValueError(f'Checkpoint differs from the paper evaluation: {checkpoint}')
    data = ROOT/'tokenhsi/data'
    motions = yaml.safe_load((data/'dataset_stage1_jointpush.yaml').read_text())['motions']
    for entries in motions.values():
        for entry in entries:
            for field in ('file', 'obj_file'):
                if field in entry and not (data/entry[field]).is_file():
                    raise FileNotFoundError(f'Missing motion input: {data/entry[field]}')
    for relative in (
        'sct_coopft/tokenhsi/learning/multi_agent/mean_cooperation.py',
        'sct_coopft/tokenhsi/learning/multi_agent/sct_coopft_network.py',
        'tokenhsi_push/scripts/tokenhsi_table_policy.py',
        'tokenhsi_push/tokenhsi/data/cfg/train/rlg/push_kd.yaml',
    ):
        if not (ROOT/'table89_assets'/relative).is_file():
            raise FileNotFoundError(f'Extract the Table 8/9 inputs archive in the repository root: {relative}')
    sources = source_hashes()
    for name in ('cohort.py', 'cohort_eval.py', 'analyze_cohort.py', 'run_cohort.py',
                 '../controller.py', '../state.py'):
        p = (ROOT/'FSM/dct'/name).resolve()
        sources[str(p)] = sha(p)
    schedules = {}
    for seed in SEEDS:
        for setting in SETTINGS:
            for delay in (0, CONDITION):
                j = dict(setting=setting, seed=seed, delay=delay)
                p, value = schedule_path(j), schedule(setting, seed, delay)
                if p.exists() and read(p) != value:
                    raise ValueError('Schedule changed')
                if not p.exists():
                    write(p, value)
                schedules[str(p)] = sha(p)
    for j in jobs():
        a, e = command(j, OUT/'unused', gpu=6)
        b, f = command(j, OUT/'unused', gpu=7)
        if a != b or f != dict(e, TOKENHSI_GPU='7'):
            raise ValueError('GPU changed evaluation contract')
        for flag, expected in (('--num_envs', env_count(j['setting'], j['seed'])),
                               ('--num_agents', SETTINGS[j['setting']]), ('--seed', j['seed'])):
            if a[a.index(flag)+1] != str(expected):
                raise ValueError('Wrong cohort command')
    manifest = dict(family_envs=512, seeds=list(SEEDS), steps=900, logical_evaluations=60,
        shard_runs=180, total_trials=30720, settings=SETTINGS,
        allocation={str(seed): {s: env_count(s, seed) for s in SETTINGS} for seed in SEEDS},
        models={m: str(p) for m,p in MODELS.items()}, conditions=[0, CONDITION],
        source_sha256=sources, schedule_sha256=schedules, memory_caps_mib={str(g): v for g,v in GPU_CAPS.items()},
        normalized_initial_human_velocities=True, reference_root=str(OUT/'reference'),
        aggregation='512 pooled trials per family/seed; also equal-weight setting means; seed mean and sample std')
    p = OUT/'manifest.json'
    if p.exists() and read(p) != manifest:
        started = ('reference', 'development', 'rollouts', 'analysis', 'supervisor.lock')
        if any((OUT/name).exists() for name in started):
            raise ValueError('Manifest changed after evaluation started; use a fresh run directory or the original commit')
        previous = p.with_name(f'manifest_before_{sha(p)[:12]}.json')
        if not previous.exists():
            write(previous, read(p))
        write(p, manifest)
    elif not p.exists():
        write(p, manifest)
    write(OUT/'prepare_verification.json', dict(passed=True, commands=360, schedules=36,
        family_trials_per_seed=512, subtype_trials_across_seeds=512, total_trials=30720))
