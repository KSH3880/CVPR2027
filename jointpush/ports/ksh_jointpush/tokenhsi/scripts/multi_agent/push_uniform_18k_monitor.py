import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def run(*args):
    return subprocess.check_output(list(args),text=True).strip()


def unit_state(unit):
    return dict(line.split('=',1) for line in run('systemctl','--user','show',unit,
        '-p','MainPID','-p','ActiveState','-p','SubState').splitlines())


def record(plan,status,**fields):
    path=Path(plan['state_file'])
    data={'time':datetime.now().astimezone().isoformat(),'status':status,**fields}
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n')
    temporary.replace(path)
    print(json.dumps(data),flush=True)


def verify_files(plan):
    for path,digest in plan['file_hashes'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Monitored input changed: '+path)


def verify_old_training(plan):
    state=unit_state(plan['old_unit'])
    if state['ActiveState']!='active' or int(state['MainPID'])!=plan['old_main_pid']:
        raise RuntimeError('Original training service changed or stopped')
    proc=Path('/proc')/str(plan['old_learner_pid'])
    command=proc.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode().strip()
    if str(proc.joinpath('cwd').resolve())!=str(ROOT) or command!=plan['old_command']:
        raise RuntimeError('Original training process identity changed')
    return state


def scores(directory,plan):
    directory=Path(directory)
    raw=json.loads((directory/'probe_result.json').read_text())
    a=np.load(directory/'probe_trials.npz')
    reference=np.load(Path(plan['baseline_directory'])/'probe_trials.npz')
    keys=('box','goal','sizes','mass','roots','dof_pos','dof_vel','motion_ids','motion_times','initial_obs','rsi')
    if not all(np.array_equal(a[k],reference[k]) for k in keys):
        raise RuntimeError('Evaluation initial states differ from the baseline')
    if raw['step']!=5999 or raw['horizon_steps']!=6000 or raw['seed']!=42 or raw['mass_factor']!=1:
        raise RuntimeError('Evaluation protocol differs from the baseline')
    if raw['official_reconciliation_error']>1e-7 or raw['physical_mass_max_error']>1e-3:
        raise RuntimeError('Evaluation success or physics reconciliation failed')
    if not all(np.isfinite(a[k]).all() for k in a.files):
        raise RuntimeError('Evaluation contains nonfinite values')
    return {'trials':int(a['rsi'].size),'successes':int((a['first_success']>=0).sum()),
        'stalled10':int((a['joint_still_max']>=300).sum()),'deterministic':raw['deterministic']}


def keep_current(current,baseline):
    return current['successes']>=baseline['successes'] and current['stalled10']<=baseline['stalled10']


def evaluate(plan):
    import torch
    checkpoint=Path(plan['checkpoint'])
    weights=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if weights['epoch']!=18000 or weights['frame']!=2359296000:
        raise RuntimeError('Expected the exact18000 checkpoint and frame budget')
    if weights['stage1_push_contract']!=plan['old_contract']:
        raise RuntimeError('The18000 checkpoint contract changed')
    del weights
    jobs=[]
    for mode in ('mean','stochastic'):
        destination=Path(plan['evaluation_root'])/mode
        destination.mkdir(parents=True,exist_ok=True)
        command=['bash',plan['evaluation_script'],str(checkpoint),str(destination)]
        environment=dict(os.environ,TOKENHSI_GPU='0',PROBE_MASS_FACTOR='1',PROBE_STOCHASTIC=str(int(mode=='stochastic')))
        stream=destination.with_suffix('.log').open('w')
        jobs.append((mode,subprocess.Popen(command,env=environment,stdout=stream,stderr=subprocess.STDOUT),stream))
    failures=[]
    try:
        for mode,process,stream in jobs:
            code=process.wait(timeout=1800)
            if code:failures.append((mode,code))
    finally:
        for _,process,stream in jobs:
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()
            stream.close()
    if failures:raise RuntimeError('Evaluation failed: '+str(failures))
    result={mode:scores(Path(plan['evaluation_root'])/mode,plan) for mode,_,_ in jobs}
    if not result['mean']['deterministic'] or result['stochastic']['deterministic']:
        raise RuntimeError('Evaluation action modes differ')
    return result


def launch_next(plan):
    verify_files(plan)
    verify_old_training(plan)
    if run('systemctl','--user','show',plan['next_unit'],'-p','LoadState','--value')!='not-found':
        raise RuntimeError('The next service name is already in use')
    record(plan,'stopping_old',old_unit=plan['old_unit'])
    subprocess.run(['systemctl','--user','stop',plan['old_unit']],check=True)
    if unit_state(plan['old_unit'])['ActiveState']=='active' or Path('/proc',str(plan['old_learner_pid'])).exists():
        raise RuntimeError('Original learner did not stop')
    output=ROOT/'output'/f"{plan['next_experiment']}_kl0_fixed78000_{datetime.now():%Y%m%d_%H%M%S}"
    output.mkdir(parents=True)
    command=['systemd-run','--user','--unit='+plan['next_unit'],
        '--property=WorkingDirectory='+str(ROOT),'--property=StandardOutput=append:'+str(output/'train.log'),
        '--property=StandardError=inherit','--property=KillMode=control-group','--property=TimeoutStopSec=60',
        '--setenv=TOKENHSI_GPU=0','--setenv=HEADLESS=1','--setenv=TEACHER_KL_COEF=0',
        '--setenv=PYTHONUNBUFFERED=1','--setenv=PYTHONDONTWRITEBYTECODE=1',
        '--setenv=OUTPUT_PATH='+str(output),'--setenv=STAGE1_CHECKPOINT='+plan['source_checkpoint'],'/usr/bin/env']
    for variable in ['CHECK','MAX_ITERATIONS','PUSH_CHECKPOINT','RESUME_CHECKPOINT','STAGE1_PUSH_ENV_CONFIG',
        'STAGE1_PUSH_EXPERIMENT','STAGE1_PUSH_RUN_NAME','MINIBATCH_SIZE','OLD_TASK_FRACTION','CUDA_MPS_PIPE_DIRECTORY',
        'CUDA_MPS_LOG_DIRECTORY','CUDA_MPS_ACTIVE_THREAD_PERCENTAGE','CUDA_MPS_PINNED_DEVICE_MEM_LIMIT',
        'SEED','TASK_CAMERA','EPISODE_LENGTH','TASK_GRAPH','EVAL_SKILLS','EVAL_SKILL_PROBS','PUSH_SPAWN_MODE','PUSH_RSI_MODE']:
        command.extend(['-u',variable])
    command.extend(['/usr/bin/bash',str(ROOT/plan['next_train_script']),'2','2048','4'])
    record(plan,'starting_next',next_unit=plan['next_unit'],output=str(output),command=command)
    subprocess.run(command,check=True)
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    deadline=time.monotonic()+240
    while time.monotonic()<deadline:
        state=unit_state(plan['next_unit'])
        if state['ActiveState'] in ('failed','inactive'):raise RuntimeError('Next training stopped during startup')
        reports=list(output.glob('*/push_transfer_report.json'))
        summaries=list(output.glob('*/summaries'))
        if reports and summaries:
            report=json.loads(reports[0].read_text())
            if report['source_epoch']!=78000:raise RuntimeError('Next training did not transfer from fixed78000')
            acc=EventAccumulator(str(summaries[0]),size_guidance={'scalars':0});acc.Reload()
            tags=acc.Tags()['scalars']
            if 'info/epochs' in tags and acc.Scalars('info/epochs')[-1].value>=5:
                if acc.Scalars('info/epochs')[0].value>2:raise RuntimeError('Next training did not start fresh')
                if any(not np.isfinite([r.value for r in acc.Scalars(tag)]).all() for tag in tags):
                    raise RuntimeError('Next training has nonfinite scalars')
                if any(r.value!=0 for r in acc.Scalars('losses/teacher_old_task_kl')):
                    raise RuntimeError('Next training has nonzero teacher KL')
                gpu=run('nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv')
                group=run('systemctl','--user','show',plan['next_unit'],'-p','ControlGroup','--value')
                pids=Path('/sys/fs/cgroup',group.lstrip('/'),'cgroup.procs').read_text().split()
                learners=[]
                for pid in pids:
                    proc=Path('/proc')/pid
                    command=proc.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode()
                    if '--cfg_train tokenhsi/data/cfg/train/rlg/'+plan['next_experiment']+'.yaml' in command:
                        if str(proc.joinpath('cwd').resolve())!=str(ROOT) or '--num_envs 2048' not in command:
                            raise RuntimeError('Next learner cwd or environment count differs')
                        learners.append(int(pid))
                if len(learners)!=1 or learners[0] not in [int(line.split(',')[0]) for line in gpu.splitlines()[1:]]:
                    raise RuntimeError('Next learner was not verified on the GPU')
                record(plan,'switched',next_unit=plan['next_unit'],output=str(output),source_epoch=78000,
                    observed_epoch=acc.Scalars('info/epochs')[-1].value,finite_scalar_tags=len(tags),learner_pid=learners[0],gpu=gpu)
                return
        time.sleep(10)
    raise RuntimeError('Next training startup verification timed out')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('plan')
    parser.add_argument('--check-plan',action='store_true')
    parser.add_argument('--assess-existing')
    args=parser.parse_args()
    os.chdir(ROOT)
    plan=json.loads(Path(args.plan).read_text())
    baseline=scores(plan['baseline_directory'],plan)
    if args.assess_existing:
        current=scores(args.assess_existing,plan)
        print(json.dumps({'baseline':baseline,'current':current,'keep_current':keep_current(current,baseline)}))
        return
    verify_files(plan)
    verify_old_training(plan)
    if run('loginctl','show-user',str(os.getuid()),'-p','Linger','--value')!='yes':
        raise RuntimeError('User linger is required')
    if args.check_plan:
        print(json.dumps({'plan_valid':True,'baseline':baseline,'poll_seconds':plan['poll_seconds'],
            'checkpoint':plan['checkpoint'],'next_unit':plan['next_unit']}))
        return
    checkpoint=Path(plan['checkpoint'])
    while True:
        verify_old_training(plan)
        if checkpoint.exists():
            time.sleep(max(0,60-(time.time()-checkpoint.stat().st_mtime)))
            break
        saved=sorted(checkpoint.parent.glob('*.pth'))
        latest=saved[-1].name if saved else None
        record(plan,'waiting_for_18000',latest_saved_checkpoint=latest,poll_seconds=plan['poll_seconds'])
        time.sleep(plan['poll_seconds'])
    verify_files(plan)
    record(plan,'evaluating_18000',checkpoint=str(checkpoint))
    result=evaluate(plan)
    decision={'baseline':baseline,'evaluation':result,'keep_current':keep_current(result['mean'],baseline)}
    Path(plan['decision_file']).write_text(json.dumps(decision,indent=2)+'\n')
    if decision['keep_current']:
        record(plan,'keep_current',**decision)
    else:
        launch_next(plan)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        if len(os.sys.argv)>1:
            plan=json.loads(Path(os.sys.argv[1]).read_text())
            record(plan,'error',error=str(error))
        raise
