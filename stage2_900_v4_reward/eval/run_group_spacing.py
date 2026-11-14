import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'eval'), str(ROOT/'tokenhsi'), str(ROOT/'FSM')]
import torch
from run_coopft_tables import memory, write
from run_stage2_scalability import verify
from eval_verify import verify as verify_fsm
import stage2_xy100

stage2_xy100.install()
torch.set_num_threads(1)
LOCK = threading.RLock()
VERIFY_LOCK = threading.Lock()
NAMES = dict(tokenhsi='TokenHSI (+ PUSH)', stage1='SCT (Stage 1 26k)',
             fsm='SCT + FSM', coopft='SCT + Coop. FT 6k', ours='Ours (V3 2k)')
CHECKPOINTS = dict(
    tokenhsi=ROOT.parent/'tokenhsi_push/checkpoints/TokenHSIPushKD_00040500.pth',
    stage1=ROOT/'pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth',
    fsm=ROOT/'pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth',
    coopft=ROOT.parent/'sct_coopft/output/sct_coopft/SCTCoopFT_10-22-24-41/nn/SCTCoopFT_00006000.pth',
    ours=ROOT/'output/stage2_distill_900_v3/Stage2Distill900V3_09-20-41-11/nn/Stage2Distill900V3_00002000.pth')
CARRIER = ROOT/'output/stage2_distill_900_v2/Stage2Distill900V2_08-19-13-51/nn/Stage2Distill900V2_00002000.pth'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(output, jobs, envs=64):
    scores, rows = {}, []
    people_counts = sorted({job['people'] for job in jobs})
    for job in jobs:
        path = output/job['model']/job['name']/f"general_seed{job['seed']}"
        result = json.loads((path/'result.json').read_text())
        row = dict(job, source=str(path), scores=result['interaction_group_scores']['region'], errors=result['errors'])
        rows.append(row)
    names = {model: label for model, label in NAMES.items() if any(j['model'] == model for j in jobs)}
    for model in names:
        scores[model] = {}
        for people in people_counts:
            for mode in ('independent','mixed'):
                source = 'stage1' if model == 'fsm' and mode == 'independent' else model
                name = f'{mode}_{people}h'
                records = [r for r in rows if r['model'] == source and r['name'] == name]
                key = ('Independent Interaction Success Rate' if mode == 'independent'
                       else 'Mixed Interaction Group Success Rate')
                values = [100*r['scores'][key]['rate'] for r in records]
                scores[model][name] = dict(mean=statistics.mean(values), std=statistics.stdev(values) if len(values)>1 else None,
                    seeds=[r['seed'] for r in records], values=values, source_model=source)
    seeds=sorted({j['seed'] for j in jobs})
    summary = dict(envs=envs, seeds=seeds, horizon=900,
                   results=scores, seed_runs=rows, fsm_independent_reused_from='stage1')
    write(output/'summary.json',summary)
    lines = ['# Table 1 · 그룹 간격 확인', '',
        f'{envs}환경 × seed{seeds} × 900step·XY100%·M=2N. 신규{len(jobs)}회. SCT+FSM Independent는 SCT 결과 공유.', '',
        '관계·거리·크기·ordered·distractor 규칙은 유지하며 그룹 간격 조건은 각 실행의 group_spacing.json에 기록한다.', '',
        '| Method | '+' | '.join(f'{people}H {mode}' for people in people_counts for mode in ('Independent','Mixed'))+' |',
        '|---|'+'---:|'*(2*len(people_counts))]
    for model,label in names.items():
        values = [scores[model][f'{mode}_{people}h'] for people in people_counts for mode in ('independent','mixed')]
        lines.append('| '+label+' | '+' | '.join(f"{v['mean']:.2f}"+(f"±{v['std']:.2f}" if v['std'] is not None else '') for v in values)+' |')
    lines += ['', '단위%. 단일 seed는 표준편차를 계산하지 않는다. 복수 seed는 평균±표본표준편차(ddof=1). Mixed는 Independent/Enabling/Coupled 그룹 성공률의 동일 가중 평균.', '',
        'TokenHSI는 기존 고정397D 자기/담당 과제 입력 차이가 있다. SCT+FSM은 기존 reset 시 관측 재계산 동작을 보존한 참고값이며 실제 첫 입력 차이는 각 first_policy_input.json에 기록한다.',
        '간격 조건 간 초기 장면은 재샘플링으로 달라질 수 있으므로 차이 전체를 이격 효과로 해석하지 않는다.', '',
        '## Seed별 네 가지 Success Rate (%)', '',
        '| Method | Setting | Seed | Independent Interaction Success Rate | Enabling Interaction Group Success Rate | Coupled Interaction Group Success Rate | Mixed Interaction Group Success Rate |',
        '|---|---|---:|---:|---:|---:|---:|']
    metrics = ['Independent Interaction Success Rate','Enabling Interaction Group Success Rate',
               'Coupled Interaction Group Success Rate','Mixed Interaction Group Success Rate']
    for row in rows:
        values = [row['scores'][key]['rate'] for key in metrics]
        text = ['N/A' if v is None else f'{100*v:.4f}' for v in values]
        lines.append('| '+NAMES[row['model']]+' | '+row['name']+' | '+str(row['seed'])+' | '+' | '.join(text)+' |')
    (output/'report.md').write_text('\n'.join(lines)+'\n')


class Suite:
    def __init__(self,args):
        self.args, self.output = args,args.output.resolve()
        self.envs = getattr(args, 'envs', 64)
        self.models = getattr(args, 'models', tuple(NAMES))
        self.output.relative_to(ROOT/'output_etc/check')
        self.output.mkdir(parents=True,exist_ok=True)
        self.lock = (self.output/'supervisor.lock').open('a')
        fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (self.output/'supervisor.pid').write_text(str(os.getpid())+'\n')
        self.jobs = [dict(model=model,mode=mode,people=people,seed=seed,name=f'{mode}_{people}h',
            key=f'{model}/{mode}_{people}h/seed{seed}') for model in self.models for people in (8,16)
            for mode in ('independent','mixed') if not (model=='fsm' and mode=='independent') for seed in (1,6,7)]
        self.references = [dict(model='reference',mode=mode,people=people,seed=seed,name=f'{mode}_{people}h',
            key=f'reference/{mode}_{people}h/seed{seed}') for people in (8,16)
            for mode in ('independent','mixed') for seed in (1,6,7)]
        self.state = dict(completed=False,expected_runs=len(self.jobs),expected_references=12,verified=[],
            references_verified=[],running={},failed=[],pending=len(self.jobs)+12,phase='preparing',started_unix=time.time(),
            gpus=[6,7],workers_per_gpu=args.workers_per_gpu,reserve_mib=args.reserve_mib)
        self.checkpoints = dict(CHECKPOINTS)
        self.hashes = {str(p):sha(p) for p in {CHECKPOINTS[k] for k in self.models}|{CARRIER}}
        if 'coopft' in self.models:
            staged = self.output/'input/coopft/nn'/CHECKPOINTS['coopft'].name
            staged.parent.mkdir(parents=True,exist_ok=True)
            if not staged.exists():
                shutil.copyfile(CHECKPOINTS['coopft'],staged)
            self.checkpoints['coopft'] = staged
            if sha(staged)!=self.hashes[str(CHECKPOINTS['coopft'])]:
                raise ValueError('Staged Coop. FT checkpoint differs')
        sources = [ROOT/'eval'/name for name in ('group_spacing_eval.py','run_group_spacing.py',
            'stage2_scalability_eval.py','stage2_scalability_layout.py','stage2_protocol_sampling.py',
            'stage2_distractor_physics.py','stage1_protocol_eval.py','stage1_protocol_policy.py','stage2_xy100.py')]
        sources += [ROOT/'FSM'/name for name in ('eval_entry.py','controller.py','state.py','eval_verify.py')]
        sources += [ROOT.parent/'tokenhsi_push/scripts'/name for name in ('tokenhsi_table_eval.py','tokenhsi_table_policy.py')]
        sources += [ROOT.parent/'sct_coopft/eval'/name for name in ('stage2_scalability_eval.py',
            'stage2_scalability_layout.py','stage2_protocol_sampling.py','stage2_distractor_physics.py')]
        self.hashes.update({str(p):sha(p) for p in sources})
        previous = self.output/'source_hashes.json'
        if previous.exists() and json.loads(previous.read_text())!=self.hashes:
            raise ValueError('Sources changed; use a new output folder')
        write(previous,self.hashes)
        self.manifest = dict(completed=False,protocol='table1_group_spacing_removed_pilot',envs=self.envs,
            seeds=[1,6,7],horizon=900,xy_extent_percent=100,objects='2N',gpus=[6,7],
            group_spawn_gap=None,group_placement='native_agent_offsets_without_group_translation',
            ring_rng_draws_retained=True,layout_attempt_limit=1024,
            checkpoints={k:dict(source=str(CHECKPOINTS[k]),sha256=sha(CHECKPOINTS[k])) for k in self.models},
            jobs=self.jobs,expected_runs=len(self.jobs),references=12,fsm_independent_reused_from='stage1',
            known_limitations=['TokenHSI fixed397D input','FSM legacy reset observation recomputation preserved'])
        write(self.output/'manifest.json',self.manifest)
        (self.output/'logs').mkdir(exist_ok=True)
        self.save()

    def save(self):
        with LOCK:
            self.state['updated_unix']=time.time()
            write(self.output/'status.json',self.state)

    def destination(self,job):
        return self.output/job['model']/job['name']/f"general_seed{job['seed']}"

    def proof(self,job):
        return self.output/'checks'/job['model']/job['name']/f"seed{job['seed']}.json"

    def validate(self,job):
        path = self.destination(job)
        spacing = json.loads((path/'group_spacing.json').read_text())
        if spacing['group_spawn_gap'] != job.get('gap') or not spacing['scenes_with_group_gap_below_5m']:
            raise ValueError('Natural group center verification failed')
        if job.get('gap') is not None and spacing['minimum_initial_group_gap'] < job['gap']-.01:
            raise ValueError('Group spacing below configured gap')
        if job['model']=='reference':
            record=json.loads((path/'reference_info.json').read_text())
            if (record['envs'],record['agents'],record['simulated_steps'])!=(self.envs,job['people'],0):
                raise ValueError('Reference dimensions mismatch')
            proof = dict(reference_only=True,natural_spacing=True)
        else:
            reference=self.output/'reference'/job['name']/f"general_seed{job['seed']}"
            result=json.loads((path/'result.json').read_text())
            if (result['seed'],result['xy_extent_percent'],result['enabling_success'])!=(job['seed'],100,'simultaneous'):
                raise ValueError('Seed or success criterion differs')
            proof=(verify_fsm(path,reference,job['people'],self.envs,900) if job['model']=='fsm'
                   else verify(path,reference,job['people'],self.envs))
            actual=json.loads((path/'first_policy_input.json').read_text())
            if not actual['exact'] and job['model']!='fsm':
                raise ValueError('Actual first input differs')
            proof.update(actual_first_input_exact=actual['exact'],known_fsm_input_difference=job['model']=='fsm')
            if job['model'] in ('ours','coopft'):
                policy=json.loads((path/'policy_preservation.json').read_text())
                if policy['epoch']!=(2000 if job['model']=='ours' else 6000):
                    raise ValueError('Wrong checkpoint epoch')
                if result['encoder_attention_checks']!=dict(actor=900,critic=900):
                    raise ValueError('Unexpected attention checks')
            if job['model']=='stage1':
                policy=result['stage1_policy']
                if not all(policy[k] for k in ('native_actor_exact','weights_unchanged','normalization_exact','normalization_unchanged')) or policy['coordination_calls']:
                    raise ValueError('Stage1 native policy changed')
            if job['model']=='tokenhsi':
                policy=json.loads((path/'tokenhsi_policy.json').read_text())
                if policy['checkpoint_sha256']!=self.hashes[str(CHECKPOINTS['tokenhsi'])]:
                    raise ValueError('Wrong TokenHSI checkpoint')
            reference_result = torch.load(reference/'initial.pt',map_location='cpu',weights_only=False)
            live=torch.load(path/'initial.pt',map_location='cpu',weights_only=False)
            if not torch.equal(reference_result['groups'],live['groups']):
                raise ValueError('Group assignments differ')
            proof['shared_group_assignments_exact']=True
        proof.update(model=job['model'],name=job['name'],seed=job['seed'],group_spacing=spacing)
        write(self.proof(job),proof)

    def run_job(self,job,gpu):
        path=self.destination(job)
        path.mkdir(parents=True,exist_ok=True)
        model=job['model']
        source=ROOT.parent/'sct_coopft' if model=='coopft' else ROOT
        checkpoint=self.checkpoints[model] if model in ('ours','coopft','tokenhsi') else CARRIER
        cfg='sct_coopft' if model=='coopft' else 'stage2_distill_900_v3' if model=='ours' else 'stage2_distill_900_v2'
        env=dict(os.environ,TOKENHSI_GPU=str(gpu),SPACING_MODEL=model,STAGE2_SCENE_MODE=job['mode'],
            SPACING_METERS='native' if job.get('gap') is None else str(job['gap']),
            STAGE2_REFERENCE_ROOT=str(self.output/'reference'/job['name']),
            TOKENHSI_COMMON_ROOT=str(ROOT),PYTHONHASHSEED=str(job['seed']),PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        env.pop('GROUP_SPAWN_RADIUS',None)
        if job.get('radius') is not None:
            env['GROUP_SPAWN_RADIUS']=str(job['radius'])
        for key in ('REFERENCE_ONLY','TABLE2_SETTING','STAGE1_POLICY_CHECKPOINT','FSM_STAGE1_CHECKPOINT',
                    'CHECK','MAX_ITERATIONS','RESUME_CHECKPOINT','STAGE2_SAMPLING_AUDIT','BIAS_ABLATION'):
            env.pop(key,None)
        if model=='reference':
            env['REFERENCE_ONLY']='1'
        if model=='stage1':
            env['STAGE1_POLICY_CHECKPOINT']=str(CHECKPOINTS['stage1'])
        if model=='fsm':
            env['FSM_STAGE1_CHECKPOINT']=str(CHECKPOINTS['fsm'])
        command=[sys.executable,str(getattr(self.args,'entry',ROOT/'eval/group_spacing_eval.py')),'--task','HumanoidMACoupledPush',
            '--cfg_train',f'tokenhsi/data/cfg/train/rlg/{cfg}.yaml','--cfg_env','',
            '--motion_file','tokenhsi/data/dataset_stage1_jointpush.yaml','--checkpoint',str(checkpoint),
            '--seed',str(job['seed']),'--num_envs',str(self.envs),'--num_agents',str(job['people']),
            '--num_objects',str(2*job['people']),'--test','--eval','--headless','--no_video',
            '--eval_protocol','general','--eval_repeats','1','--task_graph','random_scenario',
            '--enable_size_mode','per_action_intersection','--coupled_assignment','ordered',
            '--group_spawn_over4','--output_path',str(path)]
        marker='reference_info.json' if model=='reference' else 'result.json'
        if not (path/marker).exists():
            with (self.output/'logs'/f"{model}_{job['name']}_seed{job['seed']}.log").open('w') as log:
                process=subprocess.Popen(['bash','-c','source tokenhsi/scripts/multi_agent/runtime_env.sh\nexec "$@"','_',*command],
                    cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
                with LOCK:
                    self.state['running'][job['key']]['pid']=process.pid
                    self.save()
                if process.wait():
                    raise RuntimeError('Evaluation failed: '+str(path))
        with LOCK:
            self.state['running'][job['key']]['phase']='cpu_verification'
            self.save()
        with VERIFY_LOCK:
            self.validate(job)
        print('VERIFIED',job['key'],'GPU'+str(gpu),flush=True)

    def schedule(self,jobs):
        pending=[job for job in jobs if not self.proof(job).exists()]
        futures={}
        with ThreadPoolExecutor(max_workers=2*self.args.workers_per_gpu) as pool:
            while pending or futures:
                snapshot=memory()
                for gpu in sorted((6,7),key=lambda g:-snapshot[g]['free']):
                    active=[(f,j) for f,(g,j) in futures.items() if g==gpu]
                    while pending and len(active)<self.args.workers_per_gpu and not self.state['failed']:
                        reserved=sum(max(self.state['running'][j['key']]['reservation_mib']-
                            snapshot[gpu]['processes'].get(self.state['running'][j['key']].get('pid'),0),0) for _,j in active)
                        required=6000 if pending[0]['people']==8 else 8000
                        if snapshot[gpu]['free']-reserved-self.args.reserve_mib<required:
                            break
                        job=pending.pop(0)
                        with LOCK:
                            self.state['running'][job['key']]=dict(gpu=gpu,reservation_mib=required,phase='starting',started_unix=time.time())
                            self.save()
                        future=pool.submit(self.run_job,job,gpu)
                        futures[future]=gpu,job
                        active.append((future,job))
                done,_=wait(futures,timeout=5,return_when=FIRST_COMPLETED) if futures else (set(),set())
                for future in done:
                    gpu,job=futures.pop(future)
                    with LOCK:
                        details=self.state['running'].pop(job['key'])
                        try:
                            future.result()
                            target='references_verified' if job['model']=='reference' else 'verified'
                            self.state[target].append(dict(key=job['key'],gpu=gpu,**{k:v for k,v in details.items() if k!='gpu'}))
                        except Exception:
                            self.state['failed'].append(dict(key=job['key'],error=traceback.format_exc()))
                            print(self.state['failed'][-1]['error'],flush=True)
                        self.save()
                self.state['pending']=len(pending)
                self.state['gpu_memory']={gpu:{k:v for k,v in row.items() if k!='processes'} for gpu,row in snapshot.items()}
                self.save()
                if self.state['failed'] and not futures:
                    raise RuntimeError('Queue stopped after failure; existing processes were not interrupted')
                if not futures and pending:
                    time.sleep(5)

    def run(self):
        for job in self.references+self.jobs:
            if self.proof(job).exists():
                self.state['references_verified' if job['model']=='reference' else 'verified'].append(dict(key=job['key'],reused=True))
        self.state['phase']='reference_scenes'
        self.schedule(self.references)
        self.state['phase']='model_probes'
        self.schedule([j for j in self.jobs if j['people']==16 and j['mode']=='mixed' and j['seed']==1])
        self.state['phase']='full_evaluation'
        self.schedule(self.jobs)
        if len(self.state['verified'])!=54 or len(self.state['references_verified'])!=12:
            raise ValueError('Incomplete evaluation')
        self.state['phase']='summarizing'
        self.save()
        summarize(self.output,self.jobs)
        if any(sha(p)!=value for p,value in self.hashes.items()):
            raise ValueError('Source or checkpoint changed')
        proof=dict(completed=True,new_runs=54,references=12,envs=64,seeds=[1,6,7],horizon=900,
            initial_scenes_exact=True,cpu_sr_and_error_verified=True,group_spacing_removed=True,
            checkpoint_hashes_unchanged=True,fsm_legacy_first_observation_difference_disclosed=True)
        write(self.output/'completion_verification.json',proof)
        self.manifest['completed']=True
        write(self.output/'manifest.json',self.manifest)
        self.state.update(completed=True,phase='complete',pending=0)
        self.save()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'output_etc/check/table1_no_group_spacing_env64_seeds167_20261111_retry1024')
    parser.add_argument('--workers-per-gpu',type=int,default=8)
    parser.add_argument('--reserve-mib',type=int,default=10000)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or not 1<=args.workers_per_gpu<=10 or args.reserve_mib<8000:
        parser.error('Use CPU supervisor, 1..10 workers/GPU and >=8000MiB reserve')
    Suite(args).run()
