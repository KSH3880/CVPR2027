"""Audited native RSI placement and matching AMP history, observation-independent."""
from pathlib import Path
import json
import torch
import numpy as np
from isaacgym.torch_utils import quat_rotate,quat_mul
from utils.edge_scenario_spec import classify_templates
from utils.unified_training import append_family
from utils.native_rsi_sampling import pack_pools,sample_pools


class NativeRsiPool:
    def __init__(self, task, options):
        self.t=task;self.path=Path(options['bank']);self.probability=float(options.get('probability',.5));self.data={};self.keys=task._native_asset_keys;self.late_probability=options.get('late_probability')
        m=json.loads((self.path/'manifest.json').read_text())
        assert abs(m['history_dt']-task.dt)<1e-8 and m['history_steps']==task._num_amp_obs_steps-1
        assert task._local_root_obs and not task._root_height_obs
        self.active=torch.zeros(task.num_envs,task.num_agents,device=task.device,dtype=torch.bool)
        self.phase=torch.zeros_like(self.active);self.lowering=torch.zeros_like(self.active)
        self.history=torch.zeros(task.num_envs,task.num_agents,9,task._num_amp_obs_per_step,device=task.device)
        self.counts={'sit':0,'climb':0,'missing':0,'late_draws':0,'full_draws':0,'nonphase_samples':0};self.last={}

    def load(self,key,skill):
        pair=(key,skill)
        if pair not in self.data:
            p=self.path/key/(skill+'.npz')
            if not p.exists():self.data[pair]=None
            else:
                raw=dict(np.load(p));d={k:torch.tensor(v,device=self.t.device,dtype=torch.float32) for k,v in raw.items() if k in ['root_state','body_state','dof','dof_vel','amp_history','phase']}
                if self.late_probability is not None:
                    arrays=pack_pools(raw['clip'],raw['frame'],raw['phase'],raw['source_clip_weights'])
                    d['pools']=tuple(torch.tensor(v,device=self.t.device) for v in arrays)
                self.data[pair]=d
        return self.data[pair]

    def apply(self,env_ids):
        t=self.t;self.active[env_ids]=False;self.phase[env_ids]=False;self.lowering[env_ids]=False
        rows=env_ids[t._chair_env_mask[env_ids]].repeat_interleave(t.num_agents);owners=torch.arange(t.num_agents,device=t.device).repeat(len(rows)//t.num_agents)
        template=classify_templates(t.relation_runtime.graph,rows,owners,True);physical=t._scenario_physical_object(rows,owners)
        selected=torch.rand(len(rows),device=t.device)<self.probability
        groups={}
        for i,(r,o,p,sk,choose) in enumerate(zip(rows.tolist(),owners.tolist(),physical.tolist(),template.tolist(),selected.tolist())):
            if choose and sk in (1,2):groups.setdefault((self.keys[r][p],'sit' if sk==1 else 'climb'),[]).append(i)
        for (key,skill),indices in groups.items():
            d=self.load(key,skill)
            if d is None:self.counts['missing']+=len(indices);continue
            ii=torch.tensor(indices,device=t.device);r=rows[ii];o=owners[ii];p=physical[ii];n=len(r)
            if self.late_probability is None:sample=torch.randint(len(d['root_state']),(n,),device=t.device)
            else:
                sample,late=sample_pools(*d['pools'],n,self.late_probability)
                self.counts['late_draws']+=int(late.sum());self.counts['full_draws']+=int((~late).sum());self.counts['nonphase_samples']+=int((d['phase'][sample]<.5).sum())
            root=d['root_state'][sample].clone();body=d['body_state'][sample].clone();obj=t._box_states[r,p];q=obj[:,3:7];translation=torch.cat([obj[:,:2],torch.zeros(n,1,device=t.device)],-1)
            root[:,:3]=quat_rotate(q,root[:,:3])+translation;root[:,3:7]=quat_mul(q,root[:,3:7]);root[:,7:10]=quat_rotate(q,root[:,7:10]);root[:,10:13]=quat_rotate(q,root[:,10:13])
            qq=q[:,None].expand(-1,t.num_bodies,-1).reshape(-1,4)
            body[:,:,:3]=quat_rotate(qq,body[:,:,:3].reshape(-1,3)).reshape(n,t.num_bodies,3)+translation[:,None];body[:,:,3:7]=quat_mul(qq,body[:,:,3:7].reshape(-1,4)).reshape(n,t.num_bodies,4)
            for start in [7,10]:body[:,:,start:start+3]=quat_rotate(qq,body[:,:,start:start+3].reshape(-1,3)).reshape(n,t.num_bodies,3)
            t._set_env_state(r,o,root[:,:3],root[:,3:7],d['dof'][sample],root[:,7:10],root[:,10:13],d['dof_vel'][sample]);t._kinematic_humanoid_rigid_body_states[r,o]=body;t._every_env_init_dof_pos[r,o]=d['dof'][sample]
            labels=t._amp_family(r,o)[:,None].expand(-1,9);hist=append_family(d['amp_history'][sample],labels,getattr(t,'_amp_num_families',3));assert hist.shape[1:]==self.history.shape[2:]
            self.history[r,o]=hist;self.active[r,o]=True
            self.phase[r,o]=d['phase'][sample]>.5
            surface=t._box_size[r,p,2]/2+t._chair_seats[r,p,2]
            height=d['root_state'][sample,2]-(surface+(.1 if skill=='sit' else .94))
            self.lowering[r,o]=(~self.phase[r,o])&(height>.05)&(height<.35)&(skill=='sit');self.counts[skill]+=n
        self.last=dict(counts=self.counts.copy(),active=int(self.active[env_ids].sum()))

    def init_history(self,env_ids):
        rows,owners=torch.nonzero(self.active[env_ids],as_tuple=True);rows=env_ids[rows]
        if len(rows):self.t._hist_amp_obs_buf[rows,owners]=self.history[rows,owners]
        assert torch.isfinite(self.t._amp_obs_buf[env_ids]).all()
