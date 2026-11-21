import argparse
import fcntl
import os

from common import ROOT, OUT as BASE, SETTINGS, DELAYS, read, write, sha
import run

TOKEN = ROOT.parent/'tokenhsi_push'
CHECKPOINT = TOKEN/'checkpoints/TokenHSIPushKD_00040500.pth'
OUT = BASE.parent/'dct_delay_tokenhsi_40500_env64_seed1_20261115'
base_command = run.command


def command(path, setting, model='tokenhsi', delay=0, reference=False, steps=1050):
    cmd, env = base_command(path, setting, model, delay, reference, steps)
    cmd[cmd.index(str(ROOT/'FSM/dct/evaluate.py'))] = str(ROOT/'FSM/dct/tokenhsi_eval.py')
    env.update(STAGE2_REFERENCE_ROOT=str(BASE/'reference'/setting), DCT_OUTPUT=str(path))
    return cmd, env


def sources():
    paths = [ROOT/'FSM/dct'/name for name in ('common.py','freeze.py','policy.py','evaluate.py','run.py',
        'analyze.py','tokenhsi_eval.py','run_tokenhsi.py','analyze_tokenhsi.py')]
    paths += [TOKEN/'scripts/tokenhsi_table_policy.py', TOKEN/'tokenhsi/data/cfg/train/rlg/push_kd.yaml', CHECKPOINT]
    paths += list((TOKEN/'tokenhsi/learning/transformer').glob('*.py'))
    paths += [TOKEN/f'tokenhsi/env/tasks/basic_interaction_skills/humanoid_{name}.py' for name in ('carry','sit','climb')]
    paths += [BASE/'summary.json',BASE/'paired_changes.json',BASE/'completion_verification.json']
    paths += list((BASE/'reference').glob('*/general_seed1/*.pt'))
    for path,digest in read(BASE/'formal_sources.json').items():
        if sha(ROOT/path) != digest:
            raise ValueError('Original runtime source changed '+path)
    return {str(path):sha(path) for path in paths}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=('development','formal'),required=True)
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'supervisor.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    hashes=sources()
    manifest=OUT/'manifest.json'
    if manifest.exists():
        if read(manifest)['source_sha256'] != hashes:
            raise ValueError('Sources changed since setup')
    else:
        write(manifest,dict(model='tokenhsi',checkpoint=str(CHECKPOINT),epoch=40500,gpu=6,
            envs=64,seed=1,steps=1050,delays_seconds=DELAYS,settings=SETTINGS,
            baseline=str(BASE),expected_rollouts=6,source_sha256=hashes,
            input_limitation='Original397D self/assigned-task input; no other-human or distractor tokens'))
    development=args.phase=='development'
    if not development and not (OUT/'development_verification.json').exists():
        raise ValueError('Verify development first')
    folder='development' if development else 'rollouts'
    jobs=[dict(path=f'{folder}/{s}/tokenhsi_d{d}',setting=s,model='tokenhsi',delay=d,
               steps=135 if development else 1050) for s in SETTINGS for d in ((4,) if development else DELAYS)]
    state=dict(completed=False,phase=args.phase,gpu=6,finished=[],failed=[],running={},pid=os.getpid())
    pending=[]
    for job in jobs:
        path=OUT/job['path']
        if (path/'policy_verification.json').exists() and (path/'result.json').exists():
            state['finished'].append(job['path'])
        elif path.exists():
            raise ValueError('Inspect incomplete output before retry: '+str(path))
        else:
            pending.append(job)
    run.OUT, run.command = OUT, command
    run.queue(pending,6,state,args.phase)
    state.update(completed=True)
    write(OUT/'status.json',state)


if __name__=='__main__':
    main()
