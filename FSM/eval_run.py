import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tokenhsi'),str(ROOT/'eval')]
import torch
from table2_sampling import SETTINGS
from eval_verify import verify

FULL = ROOT/'output/stage2_distill_900_v2/Stage2Distill900V2_08-19-13-51/nn/Stage2Distill900V2_00002000.pth'
STAGE1 = ROOT/'pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth'
JOBS = [f'{mode}:{n}' for n in (2,4,8,16) for mode in ('independent','mixed')]+list(SETTINGS)


def write_json(path,value):
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(value,indent=2))
    pending.replace(path)


def run(args):
    torch.set_num_threads(1)
    output = args.output.resolve()
    if ROOT/'output_etc/check' not in output.parents:
        raise ValueError('FSM output must be under output_etc/check')
    output.mkdir(parents=True,exist_ok=True)
    hashes = {str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in (FULL,STAGE1)}
    status = dict(gpu=args.gpu,worker=args.worker,jobs=args.jobs,seeds=args.seeds,envs=args.envs,steps=args.steps,
        checkpoint=str(STAGE1.resolve()),checkpoint_hashes=hashes,completed=[],failed=[],running=None,
        started_utc=datetime.now(timezone.utc).isoformat(),finished=False)
    suffix = '' if args.worker=='main' else '_'+args.worker
    status_file = output/f'worker_gpu{args.gpu}{suffix}.json'

    def save():
        write_json(status_file,status)

    save()
    for job in args.jobs:
        setting = job if job in SETTINGS else None
        mode,people = ('mixed',SETTINGS[job]) if setting else (job.split(':')[0],int(job.split(':')[1]))
        name = setting or f'{mode}_{people}h'
        folder = output/('table2' if setting else 'table1')/name
        folder.mkdir(parents=True,exist_ok=True)
        lock_name = f'.seed{args.seeds[0]}.lock' if args.seed_worker else '.eval.lock'
        job_lock = (folder/lock_name).open('a')
        fcntl.flock(job_lock,fcntl.LOCK_EX)
        source = ROOT/'output_etc/check'
        reference = (source/'table2_20261109/reference'/setting if setting else
                     Path(json.loads((source/'stage1_26k_protocol_20261108'/name/'manifest.json').read_text())['reference']))
        if args.envs!=512:
            reference = output/'reference'/name
        reference.mkdir(parents=True,exist_ok=True)
        manifest = dict(policy='SCT_26k_plus_FSM',job=job,agents=people,objects=2*people,envs=args.envs,
            horizon=args.steps,seeds=args.seeds,gpu=args.gpu,reference=str(reference),checkpoint_hashes=hashes,
            xy_extent_percent=100,coupled_release='all original participants arrived at least once',completed=False)
        if args.seed_worker:
            manifest = json.loads((folder/'manifest.json').read_text())
            if args.seeds[0] not in manifest['seeds']:
                raise ValueError('Seed absent from canonical evaluation manifest')
        else:
            write_json(folder/'manifest.json',manifest)
        verified = []
        for seed in args.seeds:
            destination = folder/f'general_seed{seed}'
            status['running'] = dict(job=job,seed=seed,path=str(destination),started=time.time())
            save()
            print(f'GPU{args.gpu} {job} seed{seed} start',flush=True)
            environment = dict(os.environ,TOKENHSI_GPU=str(args.gpu),STAGE2_SCENE_MODE=mode,
                FSM_STAGE1_CHECKPOINT=str(STAGE1.resolve()),FSM_STEPS=str(args.steps),
                STAGE2_REFERENCE_ROOT=str(reference),PYTHONHASHSEED=str(seed),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
            for key in ('TABLE2_SETTING','STAGE1_POLICY_CHECKPOINT','REFERENCE_ONLY'):
                environment.pop(key,None)
            if setting:
                environment['TABLE2_SETTING'] = setting
            common = ['--task','HumanoidMACoupledPush','--cfg_train','tokenhsi/data/cfg/train/rlg/stage2_distill_900_v2.yaml',
                '--cfg_env','','--motion_file','tokenhsi/data/dataset_stage1_jointpush.yaml','--checkpoint',str(FULL),
                '--seed',str(seed),'--num_envs',str(args.envs),'--num_agents',str(people),'--num_objects',str(2*people),
                '--test','--eval','--headless','--no_video','--eval_protocol','general','--eval_repeats','1',
                '--task_graph','random_scenario','--enable_size_mode','per_action_intersection','--coupled_assignment','ordered']
            if people>4:
                common.append('--group_spawn_over4')
            try:
                ref = reference/f'general_seed{seed}'
                if not (ref/'reference_info.json').exists():
                    ref.mkdir(parents=True,exist_ok=True)
                    entry = ROOT/'eval'/('table2_eval.py' if setting else 'stage2_scalability_eval.py')
                    with (folder/f'general_seed{seed}_reference.log').open('w') as log:
                        subprocess.run(['bash','-c','source tokenhsi/scripts/multi_agent/runtime_env.sh\nexec "$@"',
                            '_',sys.executable,str(entry),*common,'--output_path',str(ref)],cwd=ROOT,
                            env=dict(environment,REFERENCE_ONLY='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
                if not (destination/'verification.json').exists():
                    if not (destination/'result.json').exists():
                        if destination.exists() and any(destination.iterdir()):
                            raise ValueError('Incomplete output exists; preserve it and choose a fresh output: '+str(destination))
                        destination.mkdir(parents=True,exist_ok=True)
                        with (folder/f'general_seed{seed}.log').open('w') as log:
                            subprocess.run(['bash','-c','source tokenhsi/scripts/multi_agent/runtime_env.sh\nexec "$@"',
                                '_',sys.executable,str(ROOT/'FSM/eval_entry.py'),*common,'--output_path',str(destination)],
                                cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
                    checks = verify(destination,ref,people,args.envs,args.steps)
                else:
                    checks = json.loads((destination/'verification.json').read_text())
                verified.append(checks)
                status['completed'].append(dict(job=job,seed=seed,path=str(destination)))
                print(f'GPU{args.gpu} {job} seed{seed} verified',flush=True)
            except Exception as error:
                status['failed'].append(dict(job=job,seed=seed,error=str(error)))
                save()
                raise
            save()
        if args.seed_worker:
            with (folder/'.manifest.lock').open('a') as manifest_lock:
                fcntl.flock(manifest_lock,fcntl.LOCK_EX)
                manifest = json.loads((folder/'manifest.json').read_text())
                files = [folder/f'general_seed{s}/verification.json' for s in manifest['seeds']]
                if all(f.exists() for f in files):
                    write_json(folder/'verification.json',[json.loads(f.read_text()) for f in files])
                    manifest['completed'] = True
                    write_json(folder/'manifest.json',manifest)
        else:
            write_json(folder/'verification.json',verified)
            manifest['completed'] = True
            write_json(folder/'manifest.json',manifest)
        job_lock.close()
    for path,digest in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
            raise ValueError('Checkpoint changed during evaluation')
    status.update(finished=True,running=None,finished_utc=datetime.now(timezone.utc).isoformat())
    save()
    subprocess.run([sys.executable,str(ROOT/'FSM/eval_summary.py'),str(output)],cwd=ROOT,check=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',type=int,choices=(6,7),required=True)
    parser.add_argument('--worker',default='main')
    parser.add_argument('--seed-worker',action='store_true')
    parser.add_argument('--jobs',nargs='+',choices=JOBS,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[1,3,6])
    parser.add_argument('--envs',type=int,default=512)
    parser.add_argument('--steps',type=int,default=900)
    args = parser.parse_args()
    if not args.worker.replace('_','').isalnum():
        parser.error('Worker name must contain only letters, digits and underscores')
    if args.seed_worker and (len(args.jobs)!=1 or len(args.seeds)!=1):
        parser.error('Seed worker requires exactly one setting and one seed')
    if args.envs<1 or not 2<=args.steps<=900 or len(set(args.seeds))!=len(args.seeds) or min(args.seeds)<0:
        parser.error('Use positive envs, 2..900 steps, and distinct nonnegative seeds')
    run(args)
