"""Paired 16-agent diagnostic using existing scenes/schedules; no API calls."""
import json,os,subprocess,threading,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np

ROOT=Path(__file__).resolve().parent
RT=ROOT/'teacher_runtime'
SOURCE='full_mixed_table2_env64_20261111_003631'
BASE=ROOT/'output_etc/check/fsm_llm_paired_16h_env64_20261111'
GPU=RT/'output_etc/check'/BASE.name
LOCK=threading.Lock()
STATE={'completed':False,'running':{},'finished':[],'failures':[]}

def save(p,d):
    p.write_text(json.dumps(d,indent=2))

def status():
    save(BASE/'status.json',STATE)

def run(seed,gpu,mode):
    out=GPU/f'seed{seed}'/mode;out.mkdir(parents=True,exist_ok=False)
    src=RT/'output_etc/check'/SOURCE
    cmd=json.loads((src/f'eval/table1/mixed_16h/general_seed{seed}/launch.json').read_text())['command']
    cmd[1]='fsm_diagnostic/guarded_entry.py' if mode=='fsm' else 'batch_eval_entry.py'
    cmd[cmd.index('--output_path')+1]=str(out)
    cp=json.loads((ROOT/'stage1_policy.json').read_text())['checkpoint_local']
    env=dict(os.environ,TOKENHSI_GPU=str(gpu),SCT_PHYSICAL_GPU=str(gpu),TOKENHSI_CONDA_ENV='tokenhsi_koo',
        PYTHONDONTWRITEBYTECODE='1',TORCH_EXTENSIONS_DIR=str(ROOT/'output_etc/check/torch_extensions'),
        CUDA_CACHE_PATH=str(ROOT/'output_etc/check/cuda_cache'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        STAGE2_SCENE_MODE='mixed',PYTHONHASHSEED=str(seed),REFERENCE_ONLY='0',
        STAGE2_REFERENCE_ROOT=str(src/'reference/table1/mixed_16h'),FSM_STAGE1_CHECKPOINT=cp,FSM_STEPS='900',
        SCT_FREEZE_IMPL='snapshot_resume',SCT_EXECUTION_ONLY='1',SCT_PREFLIGHT='0',
        SCT_SCHEDULE_ROOT=str(ROOT/'output_etc/check'/SOURCE/'api/table1/mixed_16h'))
    env.pop('TABLE2_SETTING',None)
    env.pop('STAGE1_POLICY_CHECKPOINT',None)
    if mode=='llm_snapshot':env['STAGE1_POLICY_CHECKPOINT']=cp
    save(out/'launch.json',dict(command=cmd,physical_gpu=gpu,mode=mode,seed=seed))
    with (out/'run.log').open('w') as log:
        proc=subprocess.Popen(['bash','-c','source tokenhsi/scripts/multi_agent/runtime_env.sh || exit\nexec "$@"','_',*cmd],cwd=RT,env=env,stdout=log,stderr=subprocess.STDOUT)
        with LOCK:
            STATE['running'][str(gpu)]={'seed':seed,'mode':mode,'pid':proc.pid};status()
        rc=proc.wait()
    save(out/'exit.json',dict(returncode=rc))
    if rc:raise RuntimeError(f'{seed}/{mode}: exit {rc}; {out}/run.log')
    proof=json.loads((out/'stage1_policy.json').read_text())
    assert proof['native_actor_exact'] and proof['normalization_exact'] and proof['coordination_calls']==0
    with LOCK:
        STATE['finished'].append({'seed':seed,'mode':mode});STATE['running'].pop(str(gpu),None);status()

def stats(z,indices):
    parent=z['predecessor'][indices];packet=z['packet'][indices];ever=z['ever'][indices,:,1];fall=z['fall_step'][indices]>=0
    dependent=parent>=0;row,follow=np.where(dependent);lead=parent[row,follow]
    leaders=np.zeros_like(dependent);leaders[row,lead]=True
    v=dict(scenes=len(indices),predecessors=int(leaders.sum()),predecessor_successes=int(ever[leaders].sum()),
           predecessor_falls=int(fall[leaders].sum()),followers=int(dependent.sum()),
           follower_successes=int(ever[dependent].sum()),follower_falls=int(fall[dependent].sum()),
           simultaneous_pair_successes=int(z['simultaneous'][indices,:,1][dependent].sum()))
    for kind,name in [(0,'SIT'),(1,'CLIMB'),(3,'ON_TOP')]:
        m=packet[row,follow,1]==kind
        v[name]=dict(pairs=int(m.sum()),predecessor_successes=int(ever[row[m],lead[m]].sum()),predecessor_falls=int(fall[row[m],lead[m]].sum()))
    return v

def summarize():
    report={'per_seed':{},'aggregate':{},'source':SOURCE,'focus':'enabling predecessors','api_calls':0}
    for seed in (1,6,7):
        p=GPU/f'seed{seed}';audit=json.loads((p/'llm_snapshot/schedule_execution_audit.json').read_text());excluded_ids=[item['environment'] for item in audit['excluded']];keep=[i for i in range(64) if i not in excluded_ids]
        f=np.load(p/'fsm/trace.npz');l=np.load(p/'llm_snapshot/trace.npz')
        oldpath=RT/'output_etc/check'/SOURCE/f'eval/table1/mixed_16h/general_seed{seed}'
        oldaudit=json.loads((oldpath/'schedule_execution_audit.json').read_text());assert oldaudit['excluded']==audit['excluded']
        o=np.load(oldpath/'trace.npz')
        for key in ['packet','predecessor','groups','goals','sizes']:
            assert np.array_equal(f[key][keep],l[key]),(seed,key)
            assert np.array_equal(o[key],l[key]),(seed,'legacy',key)
        report['per_seed'][str(seed)]={'excluded_scene_indices':excluded_ids,
            'fsm':stats(f,keep),'llm_snapshot':stats(l,list(range(len(keep)))),'llm_legacy':stats(o,list(range(len(keep))))}
    for mode in ['fsm','llm_snapshot','llm_legacy']:
        rows=[report['per_seed'][str(s)][mode] for s in (1,6,7)];agg={}
        for k,v in rows[0].items():
            agg[k]={j:sum(r[k][j] for r in rows) for j in v} if isinstance(v,dict) else sum(r[k] for r in rows)
        for metric,den in [('predecessor_successes','predecessors'),('predecessor_falls','predecessors'),('follower_successes','followers'),('simultaneous_pair_successes','followers')]:
            agg[metric+'_percent']=100*agg[metric]/agg[den]
        report['aggregate'][mode]=agg
    save(BASE/'comparison.json',report)

def worker(gpu,seeds):
    for seed in seeds:
        for mode in ['fsm','llm_snapshot']:run(seed,gpu,mode)

if __name__=='__main__':
    BASE.mkdir(parents=True,exist_ok=False);GPU.mkdir(parents=True,exist_ok=False)
    save(BASE/'manifest.json',dict(seeds=[1,6,7],envs_per_seed=64,people=16,steps=900,
        source=SOURCE,api_calls=0,paired_gpu={'1':6,'6':7,'7':6},
        freeze='snapshot_resume for new LLM; original FSM; saved legacy included separately',
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [ROOT/'sct_scheduler/snapshot_freeze.py',RT/'fsm_diagnostic/controller.py',RT/'fsm_diagnostic/state.py',RT/'fsm_diagnostic/eval_entry.py',RT/'batch_eval_entry.py']}))
    status()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,6,[1,7]),pool.submit(worker,7,[6])]
        for f in futures:
            try:f.result()
            except Exception as e:
                with LOCK:STATE['failures'].append(str(e));status()
    if not STATE['failures']:
        try:summarize();STATE['completed']=True
        except Exception as e:STATE['failures'].append('summary: '+str(e))
    status();print(json.dumps(STATE),flush=True)
