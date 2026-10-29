"""Chair AT40s on the original continuous-repeat control flow."""
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from evaluation.climb_paired.native_loop import build_loop
from evaluation.climb_paired.repeat_metrics import repeat_statistics
from evaluation.hold_at.metrics import classify,first_episode,sha256

PROTOCOL='chair_hold_at_common40s_native_continuous3_v1'
TABLE_PROTOCOL='table_hold_at_common40s_native_continuous3_v1'
BASELINE=Path(__file__).resolve().parents[3]/'TokenHSI/tokenhsi/learning/transformer/trans_players.py'


def cpu(value):return value.detach().cpu().numpy().copy()


def rng_hash():
    return dict(cpu=hashlib.sha256(cpu(torch.get_rng_state()).tobytes()).hexdigest(),
                cuda=hashlib.sha256(cpu(torch.cuda.get_rng_state()).tobytes()).hexdigest())


def score_trace(path):
    """Recompute first-episode ever from recorded positions, including terminal actions."""
    with np.load(path) as data:
        objects,goals=data['objects'],data['goals']
        assert np.isfinite(objects).all() and np.isfinite(goals).all()
        success=np.linalg.norm(objects[...,:3].astype(np.float64)-goals.astype(np.float64),axis=-1)<=.2
        active,ever,lengths=first_episode(success,data['done'])
        np.testing.assert_array_equal(active,data['active'])
        np.testing.assert_array_equal(success,data['point'])
        np.testing.assert_array_equal(success,data['region'])
        _,ever32,_=first_episode(classify(objects,goals,np.float32),data['done'])
        np.testing.assert_array_equal(ever,ever32)
        np.testing.assert_array_equal(lengths,data['episode_steps'])
        assert len(data['objects'])==max(lengths)
        terminal=data['terminate'][lengths-1,np.arange(len(lengths))]
        assert (lengths[~terminal]==int(data['max_steps'])-1).all()
        assert (lengths<=int(data['max_steps'])-1).all()
        return dict(num_envs=len(ever),point_count=int(ever.sum()),region_count=int(ever.sum()),
                    max_end_frame=int(max(lengths)),num_frames=len(objects),float32_float64_ever_equal=True,
                    initial_frame_scored=False,terminal_frame_included=True)


class CarryTracker:
    def __init__(self,player):
        self.player=player;self.task=player.env.task;self.N=self.task.num_envs
        self.out=Path(self.task.cfg['args'].output_path);self.records=[]
        assert self.task.num_agents==self.task.num_objects==1 and self.task._is_eval
        assert self.task.max_episode_length==1200 and abs(float(self.task.dt)-1/30)<1e-7
        category=self.task.cfg['env']['holdAtFT']['category']
        assert category in ['ArmChair_Normal','Table_Circle']
        self.protocol=TABLE_PROTOCOL if category=='Table_Circle' else PROTOCOL
        assert self.task.cfg['env']['holdAtFT']['asset_split']=='test'

    def progress(self,status):
        path=self.out/'progress.json';temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(status=status,seed=int(self.task.cfg['args'].seed),repeat=self.repeat,
            num_repeats=3,frame=self.frame,max_steps=1200,remaining_first_episodes=int(self.alive.sum())),indent=2))
        temporary.replace(path)

    def begin(self,repeat):
        self.repeat=repeat;self.frame=0;self.path=self.out/f'repeat_{repeat}';self.path.mkdir()
        self.rng_start=rng_hash();self.alive=np.ones(self.N,bool);self.ever=np.zeros(self.N,bool)
        self.lengths=np.zeros(self.N,int);self.precision=np.full(self.N,np.inf)
        self.trace={k:[] for k in ['root','objects','goals','active','done','terminate','point','region','reset_before_action']}
        t=self.task
        initial=dict(root=cpu(t._humanoid_root_states[:,0]),dof_pos=cpu(t._dof_pos),dof_vel=cpu(t._dof_vel),
            objects=cpu(t._box_states[:,0]),goals=cpu(t._tar_pos[:,0]),sizes=cpu(t._box_size[:,0]),
            model_ids=cpu(t._object_model_ids),origins=cpu(t._env_origins),ground_z=cpu(t._object_ground_z),front=cpu(t._object_front))
        distance=np.linalg.norm(initial['objects'][:,:2]-initial['root'][:,:2],axis=-1)
        assert ((distance>=1-1e-5)&(distance<=5+1e-5)).all()
        assert (np.abs(initial['goals'][:,:2]-initial['origins'][:,:2])<=4.5+1e-5).all()
        for target in [initial['root'][:,:2],initial['objects'][:,:2]]:
            assert (((initial['goals'][:,:2]-target)**2).sum(-1)>=2-1e-5).all()
        np.testing.assert_allclose(initial['goals'][:,2],initial['ground_z'],atol=1e-6,rtol=0)
        np.savez_compressed(self.path/'initial.npz',**initial)
        self.progress('running');print('REPEAT_START',repeat,'seed',t.cfg['args'].seed,flush=True)

    def after_reset(self,ids):
        ids=torch.arange(self.N,device=self.task.device) if self.frame==0 else torch.as_tensor(ids,dtype=torch.long,device=self.task.device)
        self.player._public_reset_env_ids=ids
        self.reset_mask=np.zeros(self.N,bool);self.reset_mask[cpu(ids)]=True

    def step(self,done,info):
        self.frame+=1;assert self.frame<=1200
        objects=cpu(self.task._box_states[:,0]);goals=cpu(self.task._tar_pos[:,0])
        assert np.isfinite(objects).all() and np.isfinite(goals).all()
        success=classify(objects,goals)
        ending=cpu(done).reshape(self.N,-1).any(-1)
        values=dict(root=cpu(self.task._humanoid_root_states[:,0]),objects=objects,goals=goals,
            active=self.alive.copy(),done=ending,terminate=cpu(info['terminate']).reshape(self.N).astype(bool),
            point=success,region=success.copy(),reset_before_action=self.reset_mask.copy())
        for key,value in values.items():self.trace[key].append(value)
        self.ever|=success&self.alive;self.lengths+=self.alive
        distance=np.linalg.norm(objects[:,:3].astype(np.float64)-goals.astype(np.float64),axis=-1)
        self.precision=np.where(success&self.alive,np.minimum(self.precision,distance),self.precision)
        self.alive&=~ending
        if self.frame%150==0:
            self.progress('running');print('REPEAT_FRAME',self.repeat,self.frame,'remaining',int(self.alive.sum()),flush=True)
        patched=dict(info)
        patched['success']=torch.as_tensor(self.ever,device=self.task.device)
        patched['precision']=torch.as_tensor(np.where(np.isfinite(self.precision),self.precision,0),device=self.task.device)
        return patched

    def finish(self,repeat):
        assert repeat==self.repeat and not self.alive.any()
        assert float(self.task._sampling_failures)==0
        trace=self.path/'trace.npz'
        np.savez_compressed(trace,**{key:np.stack(value) for key,value in self.trace.items()},
            episode_steps=self.lengths,max_steps=1200)
        validation=score_trace(trace);assert validation['point_count']==int(self.ever.sum())
        row=dict(repeat=repeat,seed=int(self.task.cfg['args'].seed),num_envs=self.N,
            point_count=int(self.ever.sum()),region_count=int(self.ever.sum()),point_ever=float(self.ever.mean()),region_ever=float(self.ever.mean()),
            max_end_frame=int(max(self.lengths)),trace_sha256=sha256(trace),rng_start=self.rng_start,rng_end=rng_hash())
        (self.path/'results.json').write_text(json.dumps(row,indent=2))
        (self.path/'independent_validation.json').write_text(json.dumps(validation,indent=2))
        self.records.append(row);self.progress('repeat_complete')
        print('REPEAT_DONE',repeat,'region',100*row['region_ever'],flush=True)

    def complete(self,native):
        assert len(self.records)==3
        for i,row in enumerate(self.records):assert native[f'ObjectSet_test_{i}']['success_trials']==row['point_count']
        result=dict(protocol=self.protocol,seed=int(self.task.cfg['args'].seed),num_envs=self.N,max_steps=1200,
            time_limit_seconds=float(self.task.dt)*1200,num_repeats=3,reseed_between_repeats=False,
            initial_frame_scored=False,terminal_frame_included=True,primary_included_repeats=[1,2],
            repeats=self.records,repeat_statistics=repeat_statistics(self.records),original_loop_sha256=sha256(BASELINE),
            native_control_common_at_metrics=native)
        (self.out/'results.json').write_text(json.dumps(result,indent=2))


def run_continuous(player):
    assert player.is_determenistic
    return build_loop(BASELINE,CarryTracker)(player)
