"""Single-person/native-object evaluation adapter for the 35D full-FT policy."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from isaacgym import gymapi
from env.tasks.multi_agent.humanoid_chair_ft import HumanoidChairFT,NoBoxRsi
from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
from learning.multi_agent.chair_ft_agent import ChairFTPlayer


def classify(root, objects, surface, extent, action):
    """Independent float64 root point / oriented inner-80%-XY region classifier."""
    root,objects,surface,extent=[np.asarray(x,dtype=np.float64) for x in (root,objects,surface,extent)]
    q=objects[:,3:7];v=surface
    uv=np.cross(q[:,:3],v);rot=v+2*(q[:,3:4]*uv+np.cross(q[:,:3],uv))
    target=objects[:,:3]+rot;target[:,2]+=.1 if action=='SIT' else .94
    delta=root-target
    inv=q.copy();inv[:,:3]*=-1
    uv=np.cross(inv[:,:3],delta);local=delta+2*(inv[:,3:4]*uv+np.cross(inv[:,:3],uv))
    return np.linalg.norm(delta,axis=-1)<=.2, (np.abs(local[:,:2])<=extent*.4).all(-1)&(np.abs(delta[:,2])<=.2),target


class EvaluationTask(HumanoidChairFT):
    def __init__(self,*args,**kwargs):
        self._solo_cache_class=NoBoxRsi
        HumanoidMACarry.__init__(self,*args,**kwargs)
        assert self._is_eval and self.num_agents==1 and self.num_objects==1
        self._size_template_probabilities.zero_();self._size_template_probabilities[...,1:3]=.5

    def _load_box_asset(self):
        opts=self.cfg['env']['chairFT']['point_evaluation'];entries=opts['catalog']
        self._category=opts['category'];self._action=opts['action'];N=self.num_envs
        ids=torch.multinomial(torch.full((len(entries),),1/len(entries),device=self.device),N,replacement=True)
        self._evaluation_ids=ids;self._evaluation_catalog=entries
        def values(key):return torch.tensor([e[key] for e in entries],device=self.device,dtype=torch.float32)[ids,None]
        self._box_size=values('bbox');self._box_scale=self._box_size/torch.tensor(self._build_base_size,device=self.device)
        self._box_density=torch.full((N,),100.,device=self.device)
        self._chair_features=values('feature');self._chair_seats=values('seat');self._chair_seat_extents=values('extent')
        self._climb_metadata_heights=values('metadata_height');self._chair_kinds=torch.full((N,1),2 if self._category=='Chair' else 1,device=self.device,dtype=torch.long)
        self._rsi_asset_sizes=torch.full((N,1,3),.45,device=self.device)
        self._box_assets=[];cache={}
        for index in ids.cpu().tolist():
            e=entries[index]
            if index not in cache:
                options=gymapi.AssetOptions();options.fix_base_link=True;options.default_dof_drive_mode=gymapi.DOF_MODE_NONE
                options.use_mesh_materials=True;options.mesh_normal_mode=gymapi.COMPUTE_PER_VERTEX;options.vhacd_enabled=True
                options.vhacd_params.resolution=100000;options.vhacd_params.max_convex_hulls=128;options.vhacd_params.max_num_vertices_per_ch=64
                asset=self.gym.load_asset(self.sim,e['path'],'asset.urdf',options)
                if asset is None or self.gym.get_asset_rigid_body_count(asset)!=1:raise ValueError('Invalid native object')
                cache[index]=asset
            self._box_assets.append(cache[index])


class EvaluationPlayer(ChairFTPlayer):
    @torch.no_grad()
    def run_eval(self):
        task=self.env.task;out=Path(task.cfg['args'].output_path);out.mkdir(parents=True,exist_ok=True)
        action=task._action;N=task.num_envs;steps=900
        from utils.edge_ontop_spec import batched
        obs=self.env_reset();self.get_batch_size(obs['obs'],1)
        graph=task.relation_runtime.graph
        rel=batched(graph.edge_relation,N);valid=batched(graph.edge_valid,N)
        edge=((rel==9)|(rel==10))&valid
        assert (edge.sum(-1)==1).all()
        target_slot=edge.long().argmax(-1)
        ids=task._evaluation_ids.cpu().numpy();catalog=task._evaluation_catalog
        surface=np.array([e['seat'] if action=='SIT' or task._category=='Chair' else [0,0,e['metadata_height']] for e in catalog])[ids]
        extent=np.array([e['extent'] for e in catalog])[ids]
        alive=np.ones(N,dtype=bool);everp=np.zeros(N,dtype=bool);everr=everp.copy();end=np.full(N,-1)
        roots=[];objects=[];active=[];done_trace=[]
        def cpu(x):return x.detach().cpu().numpy().copy()
        np.savez_compressed(out/'initial.npz',root=cpu(task._humanoid_root_states),dof=cpu(task._dof_state),objects=cpu(task._box_states),model_ids=ids)
        def collect(frame,done):
            nonlocal alive,everp,everr
            root=cpu(task._humanoid_root_states[:,0,:3]);obj=cpu(task._box_states[:,0]);p,r,target=classify(root,obj,surface,extent,action)
            assert np.isfinite(root).all() and np.isfinite(obj).all(), 'Nonfinite simulation state'
            _,diag=task._evaluate_ontop_context()
            np.testing.assert_allclose(cpu(diag['target'])[np.arange(N),target_slot.cpu().numpy()],target,rtol=0,atol=2e-6)
            roots.append(root);objects.append(obj);active.append(alive.copy());done_trace.append(done.copy())
            everp|=p&alive;everr|=r&alive
            ended=done&alive;end[ended]=frame;alive&=~done
        collect(0,np.zeros(N,dtype=bool))
        for frame in range(1,steps+1):
            if not isinstance(obs,dict):obs={"obs":obs}
            act=self.get_action(obs,True);obs,_,done,_=self.env_step(self.env,act)
            collect(frame,cpu(done).reshape(N,-1).any(-1))
            if frame%150==0:print("EVAL_FRAME",frame,"alive",int(alive.sum()),flush=True)
            if not alive.any():break
        end[alive]=frame
        trace=dict(root=np.stack(roots),objects=np.stack(objects),alive=np.stack(active),done=np.stack(done_trace),surface=surface,extent=extent,model_ids=ids,end_frame=end)
        np.savez_compressed(out/'trace.npz',**trace)
        # Rescore the saved numerical observations independently of task success/feet conditions.
        saved=np.load(out/'trace.npz');pp=[];rr=[]
        for root,obj in zip(saved['root'],saved['objects']):
            p,r,_=classify(root,obj,saved['surface'],saved['extent'],action);pp.append(p);rr.append(r)
        np.testing.assert_array_equal(everp,(np.stack(pp)&saved['alive']).any(0));np.testing.assert_array_equal(everr,(np.stack(rr)&saved['alive']).any(0))
        for index in range(N):
            ending=np.flatnonzero(saved['done'][:,index])
            expected_end=int(ending[0]) if len(ending) else len(saved['done'])-1
            assert int(end[index])==expected_end
            np.testing.assert_array_equal(saved['alive'][:,index],np.arange(len(saved['alive']))<=expected_end)
        result=dict(terminal_frame_validated=True,trace_sha256=hashlib.sha256((out/'trace.npz').read_bytes()).hexdigest(),category=task._category,action=action,num_envs=N,seed=task.cfg['args'].seed,point_ever=float(everp.mean()),region_ever=float(everr.mean()),ended=int((~alive).sum()),max_end_frame=int(end.max()),trace_rescore_exact=True,model_counts=np.bincount(ids,minlength=len(catalog)).tolist())
        (out/'results.json').write_text(json.dumps(result,indent=2));print('FINETUNED_EVAL',json.dumps(result),flush=True)
