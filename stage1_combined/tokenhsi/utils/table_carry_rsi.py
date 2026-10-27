"""Model-specific mesh-adjusted carry RSI with matching corrected AMP history."""
import json, os
from pathlib import Path
import numpy as np
import torch
from isaacgym.torch_utils import quat_rotate, quat_mul
from utils.hold_at_ft import object_catalog
from env.tasks.multi_agent.humanoid_ma_carry import build_amp_observations
from utils.unified_training import append_family


class TableCarryRsi:
    def __init__(self, task):
        self.t=task;cfg=task.cfg['env']['tableCarryRsi']
        self.path=Path(os.environ.get('TABLE_RSI_BANK',cfg['bank']))
        self.audit=bool(cfg.get('audit',False));self.probability=float(os.environ.get('TABLE_RSI_PROB', cfg.get('probability',.5)))
        self.data={};self.active=torch.zeros(task.num_envs,dtype=torch.bool,device=task.device)
        self.sample=torch.full((task.num_envs,),-1,dtype=torch.long,device=task.device)
        self.history=torch.zeros(task.num_envs,9,task._num_amp_obs_per_step,device=task.device)
        self.keys=[Path(e['path']).name for e in object_catalog(os.environ.get('HOLD_AT_OBJECT_ROOT',task.cfg['env']['holdAtFT']['object_root']),task.cfg['env']['holdAtFT']['asset_split'],'Table_Circle')]
        self.manifest=json.loads((self.path/'manifest.json').read_text())
        if not self.audit and not self.manifest.get('physics_validated'):
            raise ValueError('Table RSI bank must pass physics validation first')
        self.missing=0; self.applied=0; self.requested=0

    def load(self, key):
        if key not in self.data:
            path=self.path/key/'carryWith.npz'
            if not path.is_file():self.data[key]=None;return None
            raw=dict(np.load(path));valid=np.ones(len(raw['root_state']),bool) if self.audit else raw['physics_pass'].astype(bool)
            if not valid.any():self.data[key]=None;return None
            d={k:torch.tensor(v,device=self.t.device,dtype=torch.float32) for k,v in raw.items() if k!='physics_pass'}
            d['valid']=torch.tensor(np.flatnonzero(valid),device=self.t.device,dtype=torch.long);self.data[key]=d
        return self.data[key]

    def apply(self, ids):
        t=self.t;self.active[ids]=False;self.sample[ids]=-1
        chosen=ids[torch.rand(len(ids),device=t.device)<self.probability]
        self.requested += len(chosen)
        for model in t._object_model_ids[chosen].unique().tolist():
            rows=chosen[t._object_model_ids[chosen]==model];d=self.load(self.keys[model])
            if d is None:self.missing+=len(rows);continue
            ix=d['valid'][torch.arange(len(rows),device=t.device)%len(d['valid'])] if self.audit else d['valid'][torch.randint(len(d['valid']),(len(rows),),device=t.device)]
            q=t._box_states[rows,0,3:7].clone();origin=t._env_origins[rows]
            def transform(states):
                states=states.clone();shape=states.shape
                qq=q.reshape(len(rows),*([1]*(states.ndim-2)),4).expand(*shape[:-1],4).reshape(-1,4)
                states[...,:3]=quat_rotate(qq,states[...,:3].reshape(-1,3)).reshape(*shape[:-1],3)+origin.reshape(len(rows),*([1]*(states.ndim-2)),3)
                states[...,3:7]=quat_mul(qq,states[...,3:7].reshape(-1,4)).reshape(*shape[:-1],4)
                for start in (7,10):states[...,start:start+3]=quat_rotate(qq,states[...,start:start+3].reshape(-1,3)).reshape(*shape[:-1],3)
                return states
            root=transform(d['root_state'][ix]);body=transform(d['body_state'][ix]);obj=transform(d['object_state'][ix])
            zero=torch.zeros_like(rows)
            t._set_env_state(rows,zero,root[:,:3],root[:,3:7],d['dof'][ix],root[:,7:10],root[:,10:13],d['dof_vel'][ix])
            t._kinematic_humanoid_rigid_body_states[rows,0]=body;t._every_env_init_dof_pos[rows,0]=d['dof'][ix];t._box_states[rows,0]=obj
            hr=transform(d['history_root'][ix]).reshape(-1,13);hb=transform(d['history_body'][ix]).reshape(-1,t.num_bodies,13)
            hist=build_amp_observations(hr[:,:3],hr[:,3:7],hr[:,7:10],hr[:,10:13],d['history_dof'][ix].reshape(-1,32),d['history_dof_vel'][ix].reshape(-1,32),hb[:,t._key_body_ids,:3],t._local_root_obs,t._root_height_obs,t._dof_obs_size,t._dof_offsets).reshape(len(rows),9,-1)
            labels=torch.zeros(len(rows),9,dtype=torch.long,device=t.device)
            self.history[rows]=append_family(hist,labels,3);self.active[rows]=True;self.sample[rows]=ix;self.applied+=len(rows)

    def init_history(self, ids):
        rows=ids[self.active[ids]]
        if len(rows):self.t._hist_amp_obs_buf[rows,0]=self.history[rows]
        if self.missing and not hasattr(self, '_reported_missing'):
            print(f'[Table RSI] {self.missing} requested resets lack valid physical candidates; explicit locomotion fallback',flush=True)
            self._reported_missing=True
