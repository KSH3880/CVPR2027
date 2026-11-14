import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent/'sct_coopft'
BASELINE = SOURCE/'output_etc/check/sct_coopft_4000_tables12_seeds167_20261111'
CANONICAL = ROOT/'output_etc/check/v3_2000_tables12_seeds167_20261110'
CHECKPOINTS = SOURCE/'output/sct_coopft/SCTCoopFT_10-22-24-41/nn'
sys.path.insert(0, str(ROOT/'tokenhsi'))
from run_stage2_scalability import verify
from table2_metrics import replay
from coupled_cooperation_metrics import calculate, sha
from stage2_xy100 import install

install()
torch.set_num_threads(1)
LOCK = threading.RLock()
VERIFY_LOCK = threading.Lock()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def memory():
    text = subprocess.check_output(['nvidia-smi', '-i', '6,7', '--query-gpu=index,uuid,memory.free,memory.used',
                                    '--format=csv,noheader,nounits'], text=True)
    result = {}
    for line in text.strip().splitlines():
        index, uuid, free, used = [v.strip() for v in line.split(',')]
        result[int(index)] = dict(uuid=uuid, free=int(free), used=int(used), processes={})
    text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                                    '--format=csv,noheader,nounits'], text=True)
    for line in text.strip().splitlines():
        uuid, pid, used = [v.strip() for v in line.split(',')]
        for row in result.values():
            if row['uuid'] == uuid and used.isdigit():
                row['processes'][int(pid)] = int(used)
    return result


class Suite:
    def __init__(self, args):
        self.args = args
        self.output = args.output.resolve()
        self.output.relative_to(ROOT/'output_etc/check')
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = (self.output/'supervisor.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (self.output/'supervisor.pid').write_text(str(os.getpid())+'\n')
        base = json.loads((BASELINE/'manifest.json').read_text())
        self.jobs = [dict(job, epoch=epoch, key=f"{epoch}/{job['table']}/{job['name']}/seed{job['seed']}")
                     for epoch in (3000, 6000) for job in base['jobs']]
        self.state = dict(completed=False, expected_runs=84, verified=[], running={}, failed=[], pending=84,
                          gpus=[6, 7], reserve_mib=args.reserve_mib, max_workers_per_gpu=args.workers_per_gpu,
                          phase='preparing', started_unix=time.time(), peak_gpu_used_mib={6:0, 7:0})
        self.manifest = dict(protocol_id='cvpr2027-evaluation-v1.0.0', epochs=[3000,6000],
            seeds=[1,6,7], envs=512, horizon=900, xy_extent_percent=100, objects='2N',
            canonical_suite=str(CANONICAL), baseline_4000=str(BASELINE), expected_runs=84,
            primary_start_displacement_m=.2, checkpoints={}, jobs=self.jobs, completed=False)
        for epoch in (3000,6000):
            source = CHECKPOINTS/f'SCTCoopFT_{epoch:08d}.pth'
            checkpoint = torch.load(source, map_location='cpu', weights_only=False)
            if checkpoint['epoch'] != epoch or not all(torch.isfinite(v).all() for v in checkpoint['model'].values() if v.is_floating_point()):
                raise ValueError('Invalid checkpoint')
            staged = self.output/f'input/{epoch}/nn'/source.name
            staged.parent.mkdir(parents=True, exist_ok=True)
            if not staged.exists():
                shutil.copyfile(source, staged)
            digest = sha(source)
            if sha(staged) != digest:
                raise ValueError('Staged checkpoint differs')
            self.manifest['checkpoints'][str(epoch)] = dict(source=str(source), staged=str(staged),
                sha256=digest, frame=checkpoint.get('frame'), model_tensors=len(checkpoint['model']))
        files = set(json.loads((BASELINE/'source_hashes.json').read_text()))
        files.update(str(ROOT/'eval'/name) for name in ('run_coopft_tables.py','coopft_checkpoint_eval.py',
                     'summarize_coopft_tables.py','coupled_cooperation_metrics.py'))
        files.update(str(BASELINE/name) for name in ('entry.py','report.md','summary.json','manifest.json','completion_verification.json'))
        files.update(str(ROOT/'output_etc/check/coupled_cooperation_four_models_20261111/reference'/name)
                     for name in ('task_util.py','torch_utils.py'))
        self.hashes = {p:sha(p) for p in files}
        old = self.output/'source_hashes.json'
        if old.exists() and json.loads(old.read_text()) != self.hashes:
            raise ValueError('Sources differ from the original suite; use a new output folder')
        write(old, self.hashes)
        write(self.output/'manifest.json', self.manifest)
        self.save()

    def save(self):
        with LOCK:
            self.state['updated_unix'] = time.time()
            write(self.output/'status.json', self.state)

    def destination(self, job):
        return self.output/str(job['epoch'])/job['table']/job['name']/f"general_seed{job['seed']}"

    def proof(self, job):
        return self.output/'checks'/str(job['epoch'])/job['table']/job['name']/f"seed{job['seed']}.json"

    def validate(self, job):
        path, reference = self.destination(job), Path(job['reference'])
        result = json.loads((path/'result.json').read_text())
        if result['seed'] != job['seed'] or result['experiment'] != 'sct_coopft':
            raise ValueError('Experiment/seed mismatch')
        if result['encoder_attention_checks'] != dict(actor=900, critic=900):
            raise ValueError('Encoder attention checks incomplete')
        actual = json.loads((path/'first_policy_input.json').read_text())
        policy = json.loads((path/'coopft_policy_verification.json').read_text())
        if actual['epoch'] != job['epoch'] or not actual['exact'] or policy['epoch'] != job['epoch']:
            raise ValueError('Wrong checkpoint or first actual input')
        if not all(policy[k] for k in ('first_policy_input_exact','weights_exact_before_after','normalization_exact_before_after')):
            raise ValueError('Policy preservation verification failed')
        proof = verify(path, reference, job['people'], 512)
        canonical = CANONICAL/job['table']/'ours'/job['name']/f"general_seed{job['seed']}"/'result.json'
        other = json.loads(canonical.read_text())
        for criterion in ('point','region'):
            for name,row in result['interaction_group_scores'][criterion].items():
                if 'groups' in row and row['groups'] != other['interaction_group_scores'][criterion][name]['groups']:
                    raise ValueError('Canonical group counts differ')
        if job['table']=='table2':
            write(path/'table2_recomputed.json', replay(path))
            proof['cpu_gpu_table2_exact'] = True
            if job['name'].startswith('coupled_'):
                latest = dict(model=f"coopft_{job['epoch']}", people=job['people'], seed=job['seed'],
                              source=str(path), name=f"coopft_{job['epoch']}_{job['people']}h_seed{job['seed']}")
                directory = self.output/'latest_cooperation'
                directory.mkdir(exist_ok=True)
                row = calculate(latest, str(directory),
                    str(ROOT/'output_etc/check/coupled_cooperation_four_models_20261111/reference'))
                proof['latest_cooperation_result'] = str(directory/latest['name']/'result.json')
                proof['latest_cooperation_first_success_exact'] = row['first_success_exact']
        proof.update(epoch=job['epoch'], first_policy_input_exact=True, policy_preserved=True,
                     checkpoint_sha256=self.manifest['checkpoints'][str(job['epoch'])]['sha256'],
                     canonical_group_counts_exact=True)
        write(self.proof(job), proof)

    def run_job(self, job, gpu):
        path = self.destination(job)
        path.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ, TOKENHSI_GPU=str(gpu), STAGE2_SCENE_MODE=job['mode'],
            STAGE2_REFERENCE_ROOT=str(Path(job['reference']).parent), PYTHONHASHSEED=str(job['seed']),
            PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        for name in ('REFERENCE_ONLY','TABLE2_SETTING','STAGE1_POLICY_CHECKPOINT','FSM_STAGE1_CHECKPOINT',
                     'CHECK','MAX_ITERATIONS','RESUME_CHECKPOINT','STAGE2_SAMPLING_AUDIT'):
            environment.pop(name,None)
        if job['table']=='table2':
            environment['TABLE2_SETTING']=job['name']
        command = [sys.executable,str(ROOT/'eval/coopft_checkpoint_eval.py'),'--task','HumanoidMACoupledPush',
            '--cfg_train','tokenhsi/data/cfg/train/rlg/sct_coopft.yaml','--cfg_env','',
            '--motion_file','tokenhsi/data/dataset_stage1_jointpush.yaml',
            '--checkpoint',self.manifest['checkpoints'][str(job['epoch'])]['staged'],
            '--seed',str(job['seed']),'--num_envs','512','--num_agents',str(job['people']),
            '--num_objects',str(2*job['people']),'--test','--eval','--headless','--no_video',
            '--eval_protocol','general','--eval_repeats','1','--task_graph','random_scenario',
            '--enable_size_mode','per_action_intersection','--coupled_assignment','ordered','--output_path',str(path)]
        if job['people']>4:
            command.append('--group_spawn_over4')
        if not (path/'result.json').exists():
            log = self.output/'logs'/f"{job['epoch']}_{job['table']}_{job['name']}_seed{job['seed']}.log"
            with log.open('w') as stream:
                process = subprocess.Popen(['bash','-c','source tokenhsi/scripts/multi_agent/runtime_env.sh\nexec "$@"',
                    '_',*command], cwd=SOURCE, env=environment, stdout=stream, stderr=subprocess.STDOUT)
                with LOCK:
                    self.state['running'][job['key']]['pid'] = process.pid
                    self.save()
                if process.wait():
                    raise RuntimeError('Evaluation failed; '+str(log))
        with LOCK:
            self.state['running'][job['key']]['phase'] = 'cpu_verification'
            self.save()
        with VERIFY_LOCK:
            self.validate(job)
        print('VERIFIED',job['key'],'GPU'+str(gpu),flush=True)

    def schedule(self, jobs, probe=False):
        pending = [j for j in jobs if not self.proof(j).exists()]
        reservations = {2:6000,3:7000,4:10000,8:16000,16:26000}
        futures = {}
        with ThreadPoolExecutor(max_workers=2*self.args.workers_per_gpu) as pool:
            while pending or futures:
                snapshot = memory()
                with LOCK:
                    for gpu,row in snapshot.items():
                        self.state['peak_gpu_used_mib'][gpu] = max(self.state['peak_gpu_used_mib'][gpu],row['used'])
                    for state in self.state['running'].values():
                        used = snapshot[state['gpu']]['processes'].get(state.get('pid'),0)
                        state['peak_process_mib'] = max(state.get('peak_process_mib',0),used)
                    self.state['gpu_memory'] = {gpu:{k:v for k,v in row.items() if k!='processes'} for gpu,row in snapshot.items()}
                for gpu in (6,7):
                    active = [(f,j) for f,(g,j) in futures.items() if g==gpu]
                    limit = 1 if probe else self.args.workers_per_gpu
                    while len(active)<limit and pending and not self.state['failed']:
                        reserved = sum(max(self.state['running'][j['key']]['reservation_mib']-
                            snapshot[gpu]['processes'].get(self.state['running'][j['key']].get('pid'),0),0) for _,j in active)
                        selected = next((i for i,j in enumerate(pending) if
                            (not probe or (6 if j['epoch']==3000 else 7)==gpu) and
                            snapshot[gpu]['free']-reserved-self.args.reserve_mib >= reservations[j['people']]),None)
                        if selected is None:
                            break
                        job = pending.pop(selected)
                        with LOCK:
                            self.state['running'][job['key']] = dict(gpu=gpu,pid=None,phase='starting',
                                reservation_mib=reservations[job['people']],started_unix=time.time(),peak_process_mib=0)
                            self.save()
                        future = pool.submit(self.run_job,job,gpu)
                        futures[future]=gpu,job
                        active.append((future,job))
                done,_ = wait(futures,timeout=5,return_when=FIRST_COMPLETED) if futures else (set(),set())
                for future in done:
                    gpu,job = futures.pop(future)
                    with LOCK:
                        details = self.state['running'].pop(job['key'])
                        try:
                            future.result()
                            self.state['verified'].append(dict(key=job['key'],gpu=gpu,**{k:v for k,v in details.items() if k!='gpu'}))
                        except Exception:
                            self.state['failed'].append(dict(key=job['key'],error=traceback.format_exc()))
                            print(self.state['failed'][-1]['error'],flush=True)
                        self.state['pending']=len(pending)
                        self.save()
                if self.state['failed'] and not futures:
                    raise RuntimeError('Queue stopped after failure; existing processes were not interrupted')
                if not futures and pending:
                    time.sleep(5)
                self.save()

    def run(self):
        (self.output/'logs').mkdir(exist_ok=True)
        self.state['verified'] = [dict(key=j['key'],reused=True) for j in self.jobs if self.proof(j).exists()]
        probes = [j for j in self.jobs if j['seed']==1 and (j['table'],j['name']) in
                  (('table1','mixed_16h'),('table2','coupled_3'))]
        self.state['phase']='probe'
        self.schedule(probes,probe=True)
        self.state['phase']='full_evaluation'
        pending = sorted([j for j in self.jobs if not self.proof(j).exists()],key=lambda j:(-j['people'],j['name'],j['seed'],j['epoch']))
        self.schedule(pending)
        if len(self.state['verified']) != 84:
            raise ValueError('Unexpected completed count')
        self.state['phase']='summarizing'
        self.save()
        subprocess.run([sys.executable,str(ROOT/'eval/summarize_coopft_tables.py'),str(self.output)],check=True,cwd=ROOT)
        if any(sha(p)!=digest for p,digest in self.hashes.items()):
            raise ValueError('Source/checkpoint changed')
        for row in self.manifest['checkpoints'].values():
            if sha(row['source'])!=row['sha256'] or sha(row['staged'])!=row['sha256']:
                raise ValueError('Checkpoint changed')
        self.manifest['completed']=True
        write(self.output/'manifest.json',self.manifest)
        self.state.update(completed=True,phase='complete',pending=0)
        self.save()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'output_etc/check/coopft_3000_6000_tables12_seeds167_20261111')
    parser.add_argument('--workers-per-gpu',type=int,default=8)
    parser.add_argument('--reserve-mib',type=int,default=10000)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or not 1<=args.workers_per_gpu<=8 or args.reserve_mib<8000:
        parser.error('CPU supervisor requires CUDA_VISIBLE_DEVICES=, 1..8 workers/GPU and >=8000MiB reserve')
    Suite(args).run()
