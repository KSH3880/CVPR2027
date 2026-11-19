import os
from pathlib import Path
import sys

from common import ROOT, OUT as OLD, MODELS as BASE_MODELS, read, write, sha

OUT = ROOT/'output_etc/check/dct_delay_env64_seeds167_30s_20261115'
TOKEN_OLD = ROOT/'output_etc/check/dct_delay_tokenhsi_40500_env64_seed1_20261115'
MODELS = dict(BASE_MODELS, tokenhsi=ROOT/'table89_assets/tokenhsi_push/checkpoints/TokenHSIPushKD_00040500.pth')
SETTINGS = dict(enable_sit=2, enable_climb=2, enable_ontop=2, coupled_2=2, coupled_3=3, coupled_4=4)
SEEDS = (1, 6, 7)
DELAYS = (0, 2, 4)
LABELS = dict(sct='SCT', fsm='SCT+FSM', coopft='CoopFT7k', ours='DCT', tokenhsi='TokenHSI')


def key(job):
    return f"{job['setting']}/{job['model']}_d{job['delay']}_seed{job['seed']}"


def jobs():
    result = []
    for seed in SEEDS:
        for setting in SETTINGS:
            for delay in DELAYS:
                for model in MODELS:
                    reused = seed == 1 and setting in ('enable_sit', 'coupled_3')
                    source = (TOKEN_OLD if model == 'tokenhsi' else OLD)/f'rollouts/{setting}/{model}_d{delay}'
                    job = dict(setting=setting, model=model, delay=delay, seed=seed, reused=reused)
                    job['source'] = str(source if reused else OUT/'rollouts'/key(job))
                    result.append(job)
    return result


def command(job, path, reference=False, steps=900, gpu=6):
    import run as pilot
    pilot.SETTINGS.update(SETTINGS)
    cmd, env = pilot.command(path, job['setting'], job['model'], job['delay'], reference, steps)
    cmd[cmd.index('--seed')+1] = str(job['seed'])
    env.update(TOKENHSI_GPU=str(gpu), PYTHONHASHSEED=str(job['seed']),
        STAGE2_REFERENCE_ROOT=str(OUT/'reference'/job['setting']), DCT_OUTPUT=str(path))
    if job['model'] == 'tokenhsi' and not reference:
        cmd[cmd.index(str(ROOT/'FSM/dct/evaluate.py'))] = str(ROOT/'FSM/dct/tokenhsi_eval.py')
    return cmd, env


def source_hashes():
    paths = [ROOT/'FSM/dct'/name for name in ('common.py', 'freeze.py', 'policy.py', 'evaluate.py',
        'tokenhsi_eval.py', 'run.py', 'suite.py', 'run_suite.py', 'analyze_suite.py')]
    paths += list(MODELS.values())
    paths += [ROOT/'eval'/name for name in ('table2_sampling.py', 'stage2_distractor_physics.py',
        'analyze_coordination_behavior.py', 'coupled_cooperation_metrics.py')]
    paths += list((ROOT/'FSM/dct/reference').glob('*.py'))
    return {str(p): sha(p) for p in paths}


def cpu_command(job, path, destination, steps=900):
    return [sys.executable, str(ROOT/'FSM/dct/analyze_suite.py'), '--source', str(path),
        '--destination', str(destination), '--setting', job['setting'], '--model', job['model'],
        '--seed', str(job['seed']), '--delay', str(job['delay']), '--steps', str(steps)]
