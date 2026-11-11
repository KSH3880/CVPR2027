"""Capture all initial scenes, persist one API result per scene, then evaluate offline."""
import argparse,json,os,subprocess,sys,time,hashlib
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parent
RT=ROOT/'teacher_runtime'

def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_suffix('.pending');pending.write_text(json.dumps(data,indent=2));pending.replace(path)

def gpu_job(base,seed,envs,reference,output,*,capture=False,preflight=False):
    stage1=json.loads((ROOT/'stage1_policy.json').read_text())
    output.mkdir(parents=True,exist_ok=False)
    cmd=['python','batch_eval_entry.py','--task','HumanoidMACoupledPush','--cfg_train','tokenhsi/data/cfg/train/rlg/stage2_distill_900_v2.yaml','--cfg_env','','--motion_file','tokenhsi/data/dataset_stage1_jointpush.yaml','--checkpoint',str(RT/'checkpoints/stage2_v2/nn/Stage2Distill900V2_00002000.pth'),'--seed',str(seed),'--num_envs',str(envs),'--num_agents','16','--num_objects','32','--test','--eval','--headless','--no_video','--eval_protocol','general','--eval_repeats','1','--task_graph','random_scenario','--enable_size_mode','per_action_intersection','--coupled_assignment','ordered','--group_spawn_over4','--output_path',str(output)]
    env=dict(os.environ,TOKENHSI_GPU='7',SCT_PHYSICAL_GPU='7',PYTHONDONTWRITEBYTECODE='1',TORCH_EXTENSIONS_DIR=str(ROOT/'output_etc/check/torch_extensions'),CUDA_CACHE_PATH=str(ROOT/'output_etc/check/cuda_cache'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',STAGE2_SCENE_MODE='mixed',PYTHONHASHSEED=str(seed),REFERENCE_ONLY=str(int(capture)),STAGE2_REFERENCE_ROOT=str(reference),STAGE1_POLICY_CHECKPOINT=stage1['checkpoint_local'],SCT_SCHEDULE_ROOT=str(base/'api'),SCT_PREFLIGHT=str(int(preflight)))
    save(output/'launch.json',dict(command=cmd,physical_gpu=7,capture=capture,preflight=preflight))
    with (output/'run.log').open('w') as log:
        p=subprocess.Popen(['bash','-c','source tokenhsi/scripts/multi_agent/runtime_env.sh || exit\nexec "$@"','_',*cmd],cwd=RT,env=env,stdout=log,stderr=subprocess.STDOUT)
        save(output/'process.json',dict(pid=p.pid));rc=p.wait()
    save(output/'exit.json',dict(returncode=rc))
    if rc:raise RuntimeError('GPU job failed; see '+str(output/'run.log'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--preflight-only',action='store_true');p.add_argument('--workers',type=int,default=16);args=p.parse_args()
    base=args.output.resolve();assert (ROOT/'output_etc/check').resolve() in base.parents
    base.mkdir(parents=True,exist_ok=False)
    gpu_root=RT/'output_etc/check'/base.name
    gpu_root.mkdir(parents=True,exist_ok=False)
    status=dict(gpu_output=str(gpu_root),stage='preflight',seeds=[1,6,7],envs_per_seed=512,agents=16,objects=32,horizon=900,api_workers=args.workers,completed=False)
    save(base/'status.json',status)
    try:
        ref=gpu_root/'preflight/reference'
        gpu_job(base,1,2,ref,ref/'general_seed1',capture=True)
        gpu_job(base,1,2,ref,gpu_root/'preflight/eval/general_seed1',preflight=True)
        result=json.loads((gpu_root/'preflight/eval/general_seed1/result.json').read_text());proof=result['stage1_policy']
        assert proof['native_actor_exact'] and proof['normalization_exact'] and proof['coordination_calls']==0
        assert result['llm_scheduler']['first_step']['max_displacement_m']<.5
        save(base/'preflight_passed.json',dict(stage1_policy=proof,initial_checks=result['llm_scheduler']))
        if args.preflight_only:
            status.update(stage='preflight_passed',completed=True);save(base/'status.json',status);return
        ref=gpu_root/'reference'
        for seed in (1,6,7):
            status.update(stage='capture',seed=seed);save(base/'status.json',status)
            gpu_job(base,seed,512,ref,ref/('general_seed'+str(seed)),capture=True)
        from sct_scheduler.config import read_settings,chat_endpoint,DEFAULT_BASE_URL
        from sct_scheduler.client import chat_transport,OneShotScheduler
        from sct_scheduler.core import build_messages
        settings=read_settings();transport=chat_transport(chat_endpoint(settings.get('SCT_LLM_BASE_URL') or DEFAULT_BASE_URL),settings['SCT_LLM_MODEL'],settings['SCT_LLM_API_KEY'],timeout_s=180,json_mode=True,reasoning_effort='medium')
        jobs=[]
        for seed in (1,6,7):
            for i in range(512):
                inp=ref/('general_seed'+str(seed))/'requests'/('env_%04d'%i)/'request.json'
                req=json.loads(inp.read_text());build_messages(req)
                jobs.append((seed,i,req))
        status.update(stage='api',total=len(jobs),api_done=0,api_failed=0);save(base/'status.json',status)
        def call(job):
            seed,i,req=job;dest=base/'api'/('general_seed'+str(seed))/('env_%04d'%i)
            try:
                ans=OneShotScheduler(transport).schedule(req,dest)
                return dict(seed=seed,environment=i,ok=True)
            except Exception as exc:return dict(seed=seed,environment=i,ok=False,error=str(exc))
        failures=[]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(call,j) for j in jobs]):
                row=future.result();status['api_done']+=1
                if not row['ok']:failures.append(row);status['api_failed']+=1
                save(base/'status.json',status);save(base/'api_failures.json',failures)
        if failures:raise RuntimeError('API failures preserved; no incomplete batch evaluated: '+str(len(failures)))
        for seed in (1,6,7):
            status.update(stage='eval',seed=seed);save(base/'status.json',status)
            gpu_job(base,seed,512,ref,gpu_root/'eval'/('general_seed'+str(seed)))
        status.update(stage='cpu_verification');save(base/'status.json',status)
        with (base/'verification.log').open('w') as log:
            subprocess.run([sys.executable,str(ROOT/'verify_eval16_batch.py'),str(base)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        status.update(stage='complete',completed=True);save(base/'status.json',status)
    except Exception as exc:
        status.update(stage='failed',error=str(exc));save(base/'status.json',status);raise
if __name__=='__main__':main()
