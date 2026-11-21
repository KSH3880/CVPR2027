import argparse
import fcntl
import os
import subprocess
import sys
import time

from common import ROOT, OUT, OURS, MODELS, SETTINGS, DELAYS, COOP_ROOT, read, write, sha


def command(path, setting, model='ours', delay=0, reference=False, steps=1050):
    env = dict(os.environ)
    for key in ('CUDA_MPS_PIPE_DIRECTORY','CUDA_MPS_LOG_DIRECTORY','REFERENCE_ONLY','STAGE1_POLICY_CHECKPOINT',
                'FSM_STAGE1_CHECKPOINT','CHECK','MAX_ITERATIONS','RESUME_CHECKPOINT','GROUP_SPAWN_RADIUS',
                'SPACING_METERS','STAGE2_SAMPLING_AUDIT','BIAS_ABLATION','SPACING_MODEL'):
        env.pop(key, None)
    env.update(TOKENHSI_GPU='6', TABLE2_SETTING=setting, DCT_MODEL=model, DCT_DELAY=str(delay), DCT_STEPS=str(steps),
        STAGE2_REFERENCE_ROOT=str(OUT/'reference'/setting), STAGE2_SCENE_MODE='mixed',
        PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', PYTHONHASHSEED='1',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    if reference:
        env['REFERENCE_ONLY'] = '1'
    people = SETTINGS[setting]
    args = [sys.executable, str(ROOT/'FSM/dct/evaluate.py'), '--task','HumanoidMACoupledPush',
        '--cfg_train','tokenhsi/data/cfg/train/rlg/stage2_distill_900_v4.yaml','--cfg_env','',
        '--motion_file','tokenhsi/data/dataset_stage1_jointpush.yaml','--checkpoint',str(OURS),
        '--seed','1','--num_envs','64','--num_agents',str(people),'--num_objects',str(2*people),
        '--test','--eval','--headless','--no_video','--eval_protocol','general','--eval_repeats','1',
        '--task_graph','random_scenario','--enable_size_mode','per_action_intersection',
        '--coupled_assignment','ordered','--output_path',str(path)]
    return ['bash','-c','source tokenhsi/scripts/multi_agent/runtime_env.sh\nexec "$@"','_',*args], env


def launch(job):
    path = OUT/job['path']
    path.mkdir(parents=True, exist_ok=False)
    cmd, env = command(path, job['setting'], job.get('model','ours'), job.get('delay',0),
                       job.get('reference',False), job.get('steps',1050))
    log = (path/'run.log').open('w')
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    log.close()
    return proc


def free_memory(running, reserve):
    free = int(subprocess.check_output(['nvidia-smi','-i','6','--query-gpu=memory.free',
        '--format=csv,noheader,nounits'], text=True).strip())
    output = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory',
        '--format=csv,noheader,nounits'], text=True)
    usage = {}
    for row in output.splitlines():
        try:
            pid, used = map(int, row.split(','))
            usage[pid] = used
        except ValueError:
            pass
    pending_reserve = sum(max(0, reserve-usage.get(proc.pid,0)) for proc,job in running.values())
    return free-pending_reserve


def queue(jobs, workers, state, phase):
    pending, running = jobs.copy(), {}
    state.update(phase=phase, pending=[j['path'] for j in pending])
    while pending or running:
        for key,(proc,job) in list(running.items()):
            if proc.poll() is None:
                continue
            del running[key]
            if proc.returncode:
                state['failed'].append(dict(path=key, returncode=proc.returncode))
                write(OUT/'status.json', state)
                raise RuntimeError('Evaluation failed: '+key)
            state['finished'].append(key)
            print('DONE',key,flush=True)
        while pending and len(running)<workers and free_memory(running,6000)>18000:
            job=pending.pop(0)
            proc=launch(job)
            running[job['path']]=(proc,job)
            print('START',job['path'],'PID',proc.pid,flush=True)
        state.update(running={key:dict(pid=p.pid) for key,(p,j) in running.items()},
                     pending=[j['path'] for j in pending])
        write(OUT/'status.json',state)
        time.sleep(3)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--phase',choices=('reference','development','formal'),required=True)
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'supervisor.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (OUT/'supervisor.pid').write_text(str(os.getpid()))
    state=dict(completed=False,phase=args.phase,gpu=6,finished=[],failed=[],running={})
    if args.phase=='reference':
        jobs=[dict(path=f'reference/{s}/general_seed1',setting=s,reference=True) for s in SETTINGS]
        write(OUT/'manifest.json',dict(envs=64,seed=1,steps=1050,gpu=6,models={m:str(p) for m,p in MODELS.items()},
            checkpoints_sha256={m:sha(p) for m,p in MODELS.items()},delays_seconds=DELAYS,
            settings=SETTINGS,expected_rollouts=24,normalized_initial_human_velocities=True))
    elif args.phase=='development':
        jobs=[dict(path=f'development/{s}/{m}_d4',setting=s,model=m,delay=4,steps=135)
              for s in SETTINGS for m in MODELS]
    else:
        if not (OUT/'development_verification.json').exists():
            raise ValueError('Development must be verified first')
        sources = list((ROOT/'FSM/dct').glob('*.py'))
        sources += [ROOT/p for p in ('FSM/controller.py','FSM/state.py',
            'eval/table2_metrics.py','eval/table2_sampling.py','eval/stage1_protocol_policy.py',
            'eval/stage2_distractor_physics.py','eval/coupled_cooperation_metrics.py',
            'eval/analyze_coordination_behavior.py','tokenhsi/learning/multi_agent/formal_eval.py',
            'tokenhsi/learning/multi_agent/stage2_push_network.py',
            'tokenhsi/data/cfg/train/rlg/stage2_distill_900_v4.yaml')]
        sources += [COOP_ROOT/f'tokenhsi/learning/multi_agent/{name}.py'
                    for name in ('mean_cooperation','sct_coopft_network','stage2_push_network')]
        sources += list((OUT/'reference').glob('*/general_seed1/*.pt'))
        provenance = OUT/'formal_sources.json'
        hashes = {str(p):sha(p) for p in sources}
        if provenance.exists() and read(provenance) != hashes:
            raise ValueError('Formal evaluation sources changed')
        write(provenance, hashes)
        jobs=[dict(path=f'rollouts/{s}/{m}_d{d}',setting=s,model=m,delay=d)
              for s in SETTINGS for d in DELAYS for m in MODELS]
    pending=[]
    for job in jobs:
        path=OUT/job['path']
        complete = (path/'reference_info.json').exists() if job.get('reference') else (path/'policy_verification.json').exists()
        if complete:
            state['finished'].append(job['path'])
        elif path.exists():
            archive=OUT/'failed_setup'/f"{job['path'].replace('/','_')}_{int(time.time())}"
            path.rename(archive)
            pending.append(job)
        else:
            pending.append(job)
    queue(pending,args.workers,state,args.phase)
    state.update(completed=True)
    write(OUT/'status.json',state)


if __name__=='__main__':
    main()
