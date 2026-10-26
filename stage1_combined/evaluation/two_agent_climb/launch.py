"""78k native 2-human/4-object evaluation, replacing only the two CLIMB targets."""
from pathlib import Path
import sys,os,json,importlib.util
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tokenhsi'),str(ROOT)]
import isaacgym
from isaacgym import gymapi
import torch
import numpy as np
from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
from learning.multi_agent.ma_players import MAPlayerContinuous

class TwoAgentClimb(HumanoidMACarry):
    def _load_box_asset(self):
        super()._load_box_asset()
        self.category=os.environ['CLIMB_CATEGORY']
        self.fixed_targets=os.environ.get('CLIMB_FIXED_TARGETS','0')=='1'
        self.native_heights=self._box_size[...,2]/2
        self.native_ids=torch.full((self.num_envs,2),-1,device=self.device,dtype=torch.long)
        self.catalog=[]
        if self.category=='Box':
            if self.fixed_targets:
                for e in range(self.num_envs):
                    for a in range(2):
                        opt=gymapi.AssetOptions();opt.fix_base_link=True;opt.angular_damping=.01;opt.linear_damping=.01;opt.max_angular_velocity=100.;opt.density=100.;opt.default_dof_drive_mode=gymapi.DOF_MODE_NONE
                        self._box_assets[e*4+a]=self.gym.create_box(self.sim,*self._box_size[e,a].cpu().tolist(),opt)
            return
        folder=ROOT/'tokenhsi/data/dataset_amass_climb/objects/test'/self.category
        for path in sorted(folder.iterdir()):
            if not path.is_dir():continue
            c=json.loads((path/'config.json').read_text());assert c['center']==[0,0,0]
            self.catalog.append(dict(path=str(path),bbox=c['bbox'],height=c['tarSitPos'][2],id=path.name))
        ids=torch.multinomial(torch.ones(len(self.catalog),device=self.device),self.num_envs*2,replacement=True).reshape(self.num_envs,2)
        self.native_ids=ids;cache={}
        for e,row in enumerate(ids.cpu().tolist()):
            for a,mid in enumerate(row):
                item=self.catalog[mid]
                if mid not in cache:
                    opt=gymapi.AssetOptions();opt.angular_damping=.01;opt.linear_damping=.01;opt.max_angular_velocity=100.;opt.density=100.;opt.default_dof_drive_mode=gymapi.DOF_MODE_NONE
                    # Preserve original Stage1 dynamic object behavior, replacing the asset only.
                    opt.fix_base_link=self.fixed_targets;opt.use_mesh_materials=True;opt.mesh_normal_mode=gymapi.COMPUTE_PER_VERTEX
                    opt.vhacd_enabled=True;opt.vhacd_params.resolution=100000;opt.vhacd_params.max_convex_hulls=128;opt.vhacd_params.max_num_vertices_per_ch=64
                    opt.replace_cylinder_with_capsule=False
                    cache[mid]=self.gym.load_asset(self.sim,item['path'],'asset.urdf',opt)
                    assert cache[mid] is not None and self.gym.get_asset_rigid_body_count(cache[mid])==1
                self._box_assets[e*4+a]=cache[mid]
                self._box_size[e,a]=torch.tensor(item['bbox'],device=self.device)
                self.native_heights[e,a]=item['height']
        self._box_scale=self._box_size/torch.tensor(self._build_base_size,device=self.device)
        self._rsi_asset_sizes=self._box_size.clone();self._rsi_asset_sizes[:,:2]=.45
    def targets(self,ids=None):
        objects=self._box_states if ids is None else self._box_states[ids]
        heights=self.native_heights if ids is None else self.native_heights[ids]
        result=objects[...,:3].clone();result[...,2]+=heights+self._char_h
        return self._logical_box_values(result) if ids is None else self._gather_entity_slots(result,self._logical_box_order[ids])
    def _evaluate_ontop_context(self,env_ids=None):
        from env.tasks.multi_agent.edge_interaction_reward import evaluate_interaction_edges
        from utils.edge_ontop_spec import select_graph
        bodies=self._rigid_body_pos if env_ids is None else self._kinematic_humanoid_rigid_body_states[env_ids,...,:3]
        graph=self.relation_runtime.graph if env_ids is None else select_graph(self.relation_runtime.graph,env_ids)
        phi,diag=evaluate_interaction_edges(bodies[...,self._key_body_ids[[0,1]],:],bodies[...,self._key_body_ids[[2,3]],:],bodies[...,0,:],self._logical_box_values(self._box_states,env_ids),self._logical_box_values(self._box_size,env_ids),self._tar_pos if env_ids is None else self._tar_pos[env_ids],graph,self._relation_cfg,self._char_h,climb_targets=self.targets(env_ids))
        if env_ids is None:self._ontop_last_diag=diag
        return phi,diag

class TwoAgentPlayer(MAPlayerContinuous):
    @torch.no_grad()
    def run_eval(self):
        task=self.env.task;out=Path(task.cfg['args'].output_path);out.mkdir(parents=True,exist_ok=True)
        N=task.num_envs;results={}
        def cpu(x):return x.detach().cpu().numpy().copy()
        for repeat in range(3):
            obs=self.env_reset();self.get_batch_size(obs['obs'],1)
            assert task.num_agents==2 and task.num_objects==4
            np.testing.assert_array_equal(cpu(task._logical_box_order),np.broadcast_to(np.arange(4),(N,4)))
            np.savez_compressed(out/f'initial_{repeat}.npz',root=cpu(task._humanoid_root_states),dof=cpu(task._dof_state),objects=cpu(task._box_states),size=cpu(task._box_size),models=cpu(task.native_ids))
            fixed_pose=task._box_states[:,:2,:7].clone();max_fixed_drift=0.
            alive=torch.ones(N,device=self.device,dtype=torch.bool);ever=torch.zeros(N,2,device=self.device,dtype=torch.bool)
            best=torch.full((N,2),float('inf'),device=self.device);trace=[];resets=[]
            for step in range(task.max_episode_length+1):
                obs=self.env_reset(resets);action=self.get_action(obs,True)
                obs,_,done,info=self.env_step(self.env,action)
                root=task._humanoid_root_states[...,:3];target=task.targets()[:,:2]
                if task.fixed_targets and alive.any():
                    drift=float((task._box_states[alive,:2,:7]-fixed_pose[alive]).abs().max())
                    max_fixed_drift=max(max_fixed_drift,drift)
                    assert drift<2e-4,('Fixed target moved beyond float coordinate tolerance',drift)
                error=(root-target).norm(dim=-1);ever|=(error<=.2)&alive[:,None];best=torch.minimum(best,torch.where(alive[:,None],error,float('inf')))
                ended=done.reshape(N,2).bool().any(-1)
                trace.append((cpu(root),cpu(target),cpu(alive),cpu(ended)))
                alive&=~ended;resets=ended.nonzero().flatten()*2
                if not alive.any():break
            assert not alive.any()
            arrays=dict(root=np.stack([v[0] for v in trace]),target=np.stack([v[1] for v in trace]),alive=np.stack([v[2] for v in trace]),done=np.stack([v[3] for v in trace]))
            np.savez_compressed(out/f'trace_{repeat}.npz',**arrays)
            distance=np.linalg.norm(arrays['root'].astype('float64')-arrays['target'].astype('float64'),axis=-1)
            success=((distance<=.2)&arrays['alive'][...,None]).any(0);minimum=np.where(arrays['alive'][...,None],distance,np.inf).min(0)
            np.testing.assert_array_equal(success,cpu(ever));assert np.isfinite(distance).all()
            assert ((arrays['done']&arrays['alive']).sum(0)==1).all()
            record=dict(fixed_targets=task.fixed_targets,max_fixed_pose_drift=max_fixed_drift,agent_success_count=int(success.sum()),agent_trials=N*2,agent_success_rate=float(success.mean()),both_agents_success_rate=float(success.all(-1).mean()),success_precision_m=float(minimum[success].mean()) if success.any() else None,episode_length=task.max_episode_length,agent_success_20s=float(((distance[:599]<=.2)&arrays['alive'][:599,:,None]).any(0).mean()),agent_success_10s=float(((distance[:299]<=.2)&arrays['alive'][:299,:,None]).any(0).mean()),trace_rescore_exact=True,steps=len(trace))
            results[str(repeat)]=record;print('TWO_AGENT_CLIMB',repeat,json.dumps(record),flush=True)
        (out/'results.json').write_text(json.dumps(dict(category=task.category,fixed_targets=task.fixed_targets,protocol='78k_2humans_4objects_target_only_swap',raw=results,mean_success_repeats_1_2=float(np.mean([results[str(i)]['agent_success_rate'] for i in (1,2)]))),indent=2))

import utils.parse_task as parser
parser.TwoAgentClimb=TwoAgentClimb
spec=importlib.util.spec_from_file_location('two_agent_run',ROOT/'tokenhsi/run.py');run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
original=run.build_alg_runner
def build(observer):
    runner=original(observer);runner.player_factory.register_builder('ma',lambda **kwargs:TwoAgentPlayer(**kwargs));return runner
run.build_alg_runner=build
run.main()
