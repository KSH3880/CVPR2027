import fcntl
import json
import subprocess
import sys
import time

import torch

from ind_mechanisms_common import ROOT, OUT, command, write, digest
from run_ind_mechanisms import memory, process_memory, fits, validate


def run():
    lock=(OUT/'development/.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pending=[(c,m) for c in ('cross','separated') for m in ('sct','ours')]
    active={}
    complete=[]
    while pending or active:
        state=json.loads((OUT/'status.json').read_text())
        if state.get('halted'):
            raise RuntimeError('Main evaluation halted; development not released')
        used,processes=memory(),process_memory()
        for gpu,job in list(active.items()):
            job['peak']=max(job['peak'],processes.get(job['process'].pid,0))
            if job['process'].poll() is None:
                continue
            job['log'].close()
            if job['process'].returncode:
                raise RuntimeError('Development failed: '+str(job['path']))
            validate(job['path'],job['scene'],900)
            write(job['path']/'execution.json',dict(gpu=gpu,peak_process_mib=job['peak'],elapsed_seconds=time.monotonic()-job['start']))
            complete.append(str(job['path']))
            del active[gpu]
        if not any(key.startswith('solo_') for key in state['waiting']):
            for gpu in (6,7):
                if gpu in active or not pending:
                    continue
                reserved=sum(max(0,j['reserve_mib']-processes.get(j['pid'],0))
                    for j in state['running'].values() if j['gpu']==gpu)
                if not fits(used[gpu],reserved,6144,70000):
                    continue
                condition,model=pending.pop(0)
                path=OUT/f'development/{condition}_{model}_seed101'
                path.mkdir()
                scene=OUT/f'development/{condition}_seed101.pt'
                cmd,env=command('ind_mechanisms_eval.py',path,101,2,gpu,dict(IND_SCENE=str(scene),IND_MODEL=model))
                log=(path/'run.log').open('w')
                proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[gpu]=dict(process=proc,log=log,path=path,scene=scene,peak=0,start=time.monotonic())
        write(OUT/'development/status.json',dict(waiting=pending,completed=complete,
            running={str(gpu):dict(pid=j['process'].pid,path=str(j['path']),peak_mib=j['peak']) for gpu,j in active.items()},
            waits_for_a_queue=any(key.startswith('solo_') for key in state['waiting'])))
        time.sleep(3)
    for condition in ('cross','separated'):
        a,b=[OUT/f'development/{condition}_{model}_seed101' for model in ('sct','ours')]
        for name in ('initial.pt','first_policy_input.pt'):
            left,right=[torch.load(p/name,map_location='cpu',weights_only=False) for p in (a,b)]
            equal=torch.equal(left,right) if torch.is_tensor(left) else all(torch.equal(left[k],right[k]) for k in left)
            if not equal:
                raise ValueError('Development inputs differ: '+condition+' '+name)
    scene_hashes={str(p):digest(p) for p in (OUT/'scenes').glob('*.pt')}
    write(OUT/'b_ready.json',dict(development_runs=complete,model_initial_states_exact=True,
        geometry_seed=101,source_pool_seed=1,formal_scenes_sha256=scene_hashes,
        acceptance='geometry, state, policy, finite trajectories and CPU scoring; no performance-based filtering'))


if __name__=='__main__':
    run()
