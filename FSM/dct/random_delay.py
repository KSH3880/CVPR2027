from pathlib import Path
import sys

import numpy as np

from suite import ROOT, OUT as BASE, SETTINGS, MODELS, LABELS, jobs as base_jobs
from suite import command as base_command, source_hashes, read, write, sha

OUT = ROOT/'output_etc/check/dct_random_delay_env64_seed1_20261115'
CONDITION = 'random_1_5'


def jobs():
    return [j for j in base_jobs() if j['seed'] == 1 and j['delay'] == 0]


def key(job):
    return f"{job['setting']}/{job['model']}"


def schedule(setting):
    index = list(SETTINGS).index(setting)
    rng = np.random.default_rng(np.random.SeedSequence([20261115, 1, index]))
    requested = rng.uniform(1., 5., 64)
    steps = np.rint(requested*30).astype(int)
    return dict(setting=setting, seed=1, env_ids=list(range(64)),
        distribution='Uniform[1,5) seconds, rounded to nearest 1/30 second',
        rng_seed=[20261115, 1, index], requested_seconds=requested.tolist(),
        steps=steps.tolist(), actual_seconds=(steps/30).tolist())


def command(job, path, steps=900):
    cmd, env = base_command(job, path, steps=steps, gpu=6)
    for entry in ('evaluate.py', 'tokenhsi_eval.py'):
        original = str(ROOT/'FSM/dct'/entry)
        if original in cmd:
            cmd[cmd.index(original)] = str(ROOT/'FSM/dct/random_eval.py')
    env['DCT_DELAY_SCHEDULE'] = str(OUT/'schedules'/f"{job['setting']}.json")
    return cmd, env


def cpu_command(job, path, destination, steps=900):
    return [sys.executable, str(ROOT/'FSM/dct/analyze_random.py'), '--source', str(path),
        '--destination', str(destination), '--setting', job['setting'], '--model', job['model'],
        '--seed', '1', '--steps', str(steps)]


def prepare():
    previous = read(BASE/'manifest.json')['source_sha256']
    if source_hashes() != previous:
        raise ValueError('Previous suite source/checkpoint changed')
    baseline = []
    for job in jobs():
        row = BASE/'analysis'/f"{job['setting']}/{job['model']}_d0_seed1.json"
        value = read(row)
        if not value['verified'] or value['analyzed_steps'] != 900 or value['trials'] != 64:
            raise ValueError('Invalid baseline')
        baseline.append(dict(job=key(job), analysis=str(row), sha256=sha(row)))
        cmd, env = command(job, OUT/'rollouts'/key(job))
        if env['TOKENHSI_GPU'] != '6' or env['DCT_STEPS'] != '900':
            raise ValueError('Wrong runtime')
        for flag, expected in (('--seed', '1'), ('--num_envs', '64'),
                               ('--num_agents', str(SETTINGS[job['setting']]))):
            if cmd[cmd.index(flag)+1] != expected:
                raise ValueError('Wrong command')
    schedule_hashes = {}
    for setting in SETTINGS:
        path = OUT/'schedules'/f'{setting}.json'
        expected = schedule(setting)
        if path.exists() and read(path) != expected:
            raise ValueError('Schedule changed')
        if not path.exists():
            write(path, expected)
        schedule_hashes[setting] = sha(path)
    sources = dict(previous)
    for name in ('random_delay.py', 'random_eval.py', 'analyze_random.py', 'run_random.py'):
        path = ROOT/'FSM/dct'/name
        sources[str(path)] = sha(path)
    manifest = dict(gpu=6, memory_cap_mib=85000, envs=64, seed=1, steps=900,
        new_runs=30, baseline_runs=30, condition=CONDITION, schedule_sha256=schedule_hashes,
        source_sha256=sources, baselines=baseline, models={k: str(v) for k,v in MODELS.items()},
        settings=SETTINGS, paired='same scene, env id, delay schedule and selected agent across models',
        reference_root=str(BASE/'reference'))
    path = OUT/'manifest.json'
    if path.exists() and read(path) != manifest:
        raise ValueError('Manifest changed')
    if not path.exists():
        write(path, manifest)
    write(OUT/'prepare_verification.json', dict(passed=True, commands=30, baselines=30,
        schedules=6, previous_sources_unchanged=True))
