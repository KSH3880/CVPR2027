"""Common scene/metrics on the baseline's exact continuous three-repeat loop."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from evaluation.climb_paired.metrics import score_trace
from evaluation.climb_paired.native_loop import build_loop
from evaluation.climb_paired.repeat_metrics import repeat_statistics
from evaluation.finetuned.backend import classify

BASELINE=Path(__file__).resolve().parents[3]/'TokenHSI/tokenhsi/learning/transformer/trans_players.py'


def cpu(value):return value.detach().cpu().numpy().copy()


def rng_hash():
    return dict(cpu=hashlib.sha256(cpu(torch.get_rng_state()).tobytes()).hexdigest(),
                cuda=hashlib.sha256(cpu(torch.cuda.get_rng_state()).tobytes()).hexdigest())


class RepeatTracker:
    def __init__(self,player):
        self.player=player;self.task=player.env.task;self.N=self.task.num_envs
        self.out=Path(self.task.cfg['args'].output_path);self.records=[]
        self.ids=cpu(self.task._evaluation_ids)
        entries=self.task._evaluation_catalog
        self.action=self.task._action
        self.surface=np.array([e['seat'] if self.action=='SIT' else [0,0,e['metadata_height']] for e in entries])[self.ids]
        self.extent=np.array([e['extent'] for e in entries])[self.ids]
        assert (self.action,self.task._category) in [('CLIMB','Table'),('SIT','Chair')]

    def progress(self,status):
        path=self.out/'progress.json';temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(status=status,seed=int(self.task.cfg['args'].seed),repeat=self.repeat,
            num_repeats=3,frame=self.frame,max_steps=int(self.task.max_episode_length),remaining_first_episodes=int(self.alive.sum())),indent=2))
        temporary.replace(path)

    def begin(self,repeat):
        self.repeat=repeat;self.frame=0;self.path=self.out/f'repeat_{repeat}';self.path.mkdir()
        self.rng_start=rng_hash()
        self.alive=np.ones(self.N,bool);self.everp=np.zeros(self.N,bool);self.everr=np.zeros(self.N,bool)
        self.precision=np.full(self.N,np.inf);self.end=np.full(self.N,-1)
        self.roots=[];self.objects=[];self.active=[];self.done=[];self.reset_masks=[]
        np.savez_compressed(self.path/'initial.npz',root=cpu(self.task._humanoid_root_states),
            dof=cpu(self.task._dof_state),objects=cpu(self.task._box_states),model_ids=self.ids)
        self.collect(np.zeros(self.N,bool),np.zeros(self.N,bool))
        self.progress('running')
        print('REPEAT_START',repeat,'seed',self.task.cfg['args'].seed,flush=True)

    def after_reset(self,ids):
        # At the start of each block all environments were reset by native env_reset().
        if self.frame==0:ids=torch.arange(self.N,device=self.task.device)
        else:ids=torch.as_tensor(ids,dtype=torch.long,device=self.task.device)
        self.player._public_reset_env_ids=ids
        self.next_reset_mask=np.zeros(self.N,bool);self.next_reset_mask[cpu(ids)]=True

    def collect(self,done,reset_mask):
        root=cpu(self.task._humanoid_root_states[:,0,:3]);objects=cpu(self.task._box_states[:,0])
        assert np.isfinite(root).all() and np.isfinite(objects).all()
        point,region,target=classify(root,objects,self.surface,self.extent,self.action)
        self.roots.append(root);self.objects.append(objects);self.active.append(self.alive.copy());self.done.append(done.copy());self.reset_masks.append(reset_mask.copy())
        self.everp|=point&self.alive;self.everr|=region&self.alive
        distance=np.linalg.norm(root.astype(np.float64)-target,axis=-1)
        self.precision=np.where(point&self.alive,np.minimum(self.precision,distance),self.precision)
        ended=done&self.alive;self.end[ended]=self.frame;self.alive&=~done

    def step(self,done,info):
        self.frame+=1
        assert self.frame<=self.task.max_episode_length,'Incomplete episode past evaluation time limit'
        self.collect(cpu(done).reshape(self.N,-1).any(-1),self.next_reset_mask)
        if self.frame%150==0:
            self.progress('running')
            print('REPEAT_FRAME',self.repeat,self.frame,'remaining',int(self.alive.sum()),flush=True)
        patched=dict(info)
        patched['success']=torch.as_tensor(self.everp,device=self.task.device)
        patched['precision']=torch.as_tensor(np.where(np.isfinite(self.precision),self.precision,0),device=self.task.device)
        return patched

    def finish(self,repeat):
        assert repeat==self.repeat and not self.alive.any()
        trace=self.path/'trace.npz'
        np.savez_compressed(trace,root=np.stack(self.roots),objects=np.stack(self.objects),alive=np.stack(self.active),
            done=np.stack(self.done),surface=self.surface,extent=self.extent,model_ids=self.ids,end_frame=self.end,reset_before_action=np.stack(self.reset_masks))
        validation=score_trace(trace,self.action)
        assert validation['point_count']==int(self.everp.sum()) and validation['region_count']==int(self.everr.sum())
        row=dict(repeat=repeat,seed=int(self.task.cfg['args'].seed),num_envs=self.N,point_count=int(self.everp.sum()),region_count=int(self.everr.sum()),
            point_ever=float(self.everp.mean()),region_ever=float(self.everr.mean()),max_end_frame=int(self.end.max()),
            max_steps=int(self.task.max_episode_length),control_dt=float(self.task.dt),trace_rescore_exact=True,
            trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),rng_start=self.rng_start,rng_end=rng_hash())
        (self.path/'results.json').write_text(json.dumps(row,indent=2))
        (self.path/'independent_validation.json').write_text(json.dumps(dict(ok=True,scipy_rescore=validation),indent=2))
        self.records.append(row)
        self.progress('repeat_complete')
        print('REPEAT_DONE',repeat,'point',row['point_ever']*100,'region',row['region_ever']*100,flush=True)

    def complete(self,native):
        assert len(self.records)==3
        for repeat,row in enumerate(self.records):
            assert native[f'ObjectSet_test_{repeat}']['success_trials']==row['point_count']
        result=dict(seed=int(self.task.cfg['args'].seed),num_envs=self.N,repeats=self.records,
            repeat_statistics=repeat_statistics(self.records),num_repeats=3,reseed_between_repeats=False,
            max_steps=int(self.task.max_episode_length),time_limit_seconds=float(self.task.dt)*self.task.max_episode_length,
            original_loop_sha256=hashlib.sha256(BASELINE.read_bytes()).hexdigest(),native_point_metrics=native,
            protocol='chair_sit_common40s_native_continuous3_v1' if self.action=='SIT' else 'table_climb_common40s_native_continuous3_v1',primary_included_repeats=[1,2],warmup_repeat_preserved=0)
        (self.out/'results.json').write_text(json.dumps(result,indent=2))


def run_continuous(player):
    assert player.is_determenistic, 'Continuous comparison requires deterministic actions'
    return build_loop(BASELINE,RepeatTracker)(player)
