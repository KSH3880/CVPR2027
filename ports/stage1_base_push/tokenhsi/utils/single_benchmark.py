"""True one-person benchmark environment; training classes/configs stay untouched."""
from copy import deepcopy
import math
import numpy as np
import torch
from isaacgym import gymapi
from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
from utils.size_rsi import sample_sizes
from utils.stage1_push_spec import stable_push_profiles
from utils.finetune_evaluation import AT_SIZES


class SingleBenchmarkEnv(HumanoidMACarry):
    def __init__(self, cfg, *args, benchmark_task, initial_plan=None, **kwargs):
        self.benchmark_task = benchmark_task
        self.initial_plan = initial_plan
        assert cfg['env']['numAgents'] == 1
        assert cfg['env']['numObjects'] == (2 if benchmark_task=='holding_ontop' else 1)
        super().__init__(cfg, *args, **kwargs)

    def _load_motion(self, motion_file):
        super()._load_motion(motion_file)
        if self.benchmark_task=='push_at':
            from utils.motion_lib import MotionLib
            self._motion_lib['loco'] = MotionLib(
                motion_file='tokenhsi/data/dataset_stage1_push_finetune.yaml',skill='push_loco',
                dof_body_ids=self._dof_body_ids,dof_offsets=self._dof_offsets,
                key_body_ids=self._key_body_ids.cpu().numpy(),device=self.device)

    def _load_box_asset(self):
        if self.initial_plan is not None:
            self._box_size=torch.as_tensor(self.initial_plan['physical_sizes'],device=self.device).clone()
        elif self.benchmark_task=='push_at':
            profiles=torch.tensor(stable_push_profiles(),device=self.device)
            self._box_size=profiles[torch.randint(len(profiles),(self.num_envs,1),device=self.device)]
        else:
            self._box_size=sample_sizes(self.num_envs,self.device,self._task_graph_preset,
                self.cfg['env']['box']['build']['taskSizeRanges'],num_agents=1,num_objects=self.num_objects)
            if self.benchmark_task=='holding_at':
                sizes=self._box_size.new_tensor(AT_SIZES)
                indices=torch.randint(9,(self.num_envs,),device=self.device)
                if self.num_envs>=9:
                    indices[:9]=torch.arange(9,device=self.device)
                    indices=indices[torch.randperm(self.num_envs,device=self.device)]
                self._box_size[:,0]=sizes[indices]
        if self.benchmark_task=='push_at':
            # The environment's generic reset machinery uses carry layout, while
            # its live graph packet below identifies PUSH (task ID4).
            self._push_size_probabilities=torch.zeros(self.num_envs,1,5,device=self.device)
            self._push_size_probabilities[...,3]=1
        self._box_scale=self._box_size/self._box_size.new_tensor(self._build_base_size)
        flat=self._box_size.flatten(0,1)
        self._box_density=(20./flat.prod(-1) if self.benchmark_task=='push_at'
                           else torch.full((len(flat),),100.,device=self.device))
        self._box_assets=[];cache={}
        for i,dims in enumerate(flat.cpu().tolist()):
            key=tuple(dims)+(float(self._box_density[i]),)
            if key not in cache:
                options=gymapi.AssetOptions()
                options.density=key[-1]
                options.angular_damping=.01;options.linear_damping=.01
                options.max_angular_velocity=100.
                options.default_dof_drive_mode=gymapi.DOF_MODE_NONE
                options.fix_base_link=(self.initial_plan is not None and self.benchmark_task in ('sit','climb'))
                cache[key]=self.gym.create_box(self.sim,*dims,options)
            self._box_assets.append(cache[key])

    def _plan_pose(self, env, state):
        origin=self.gym.get_env_origin(env)
        t=gymapi.Transform()
        t.p=gymapi.Vec3(float(state[0]-origin.x),float(state[1]-origin.y),float(state[2]-origin.z))
        t.r=gymapi.Quat(*map(float,state[3:7]))
        return t

    def _build_box(self, env_id, object_id, env_ptr):
        if self.initial_plan is None:
            return super()._build_box(env_id,object_id,env_ptr)
        pose=self._plan_pose(env_ptr,self.initial_plan['physical_boxes'][env_id,object_id])
        handle=self.gym.create_actor(env_ptr,self._box_assets[env_id*self.num_objects+object_id],
            pose,'box_'+str(object_id),env_id,0,0)
        self._box_handles.append(handle)
        if self.benchmark_task=='push_at':
            props=self.gym.get_actor_rigid_shape_properties(env_ptr,handle)
            for p in props:p.friction=.4;p.restitution=0.
            self.gym.set_actor_rigid_shape_properties(env_ptr,handle,props)

    def _build_platforms(self, env_id, agent_id, env_ptr):
        if self.initial_plan is None:
            return super()._build_platforms(env_id,agent_id,env_ptr)
        # Static GPU actors cannot be teleported after prepare_sim. Create them
        # at their final sampled transforms rather than just changing root tensors.
        for key,handles in [('platform_pos',self._platform_handles),('tar_platform_pos',self._tar_platform_handles)]:
            t=gymapi.Transform();origin=self.gym.get_env_origin(env_ptr)
            pos=self.initial_plan[key][env_id,agent_id]
            t.p=gymapi.Vec3(float(pos[0]-origin.x),float(pos[1]-origin.y),float(pos[2]-origin.z))
            handles.append(self.gym.create_actor(env_ptr,self._platform_asset,t,key,env_id,self._PLATFORM_COLLISION_FILTER,0))

    def _sample_episode_graph(self, ids):
        super()._sample_episode_graph(ids)
        if self.benchmark_task=='push_at':
            g=self.relation_runtime.graph
            rel=g.edge_relation[ids]
            g.edge_relation[ids]=torch.where(rel==6,12,rel)

    def _reset_boxes(self, ids):
        if self.benchmark_task!='push_at':
            return super()._reset_boxes(ids)
        human=self._humanoid_root_states[ids,0,:2]
        center=self._env_origins[ids,:2]
        positions=torch.empty_like(human);goals=torch.empty_like(human)
        yaw=torch.empty(len(ids),device=self.device)
        pending=torch.arange(len(ids),device=self.device)
        for _ in range(2048):
            if not len(pending):break
            direction=torch.randn(len(pending),2,device=self.device)
            direction/=direction.norm(dim=-1,keepdim=True)
            xy=human[pending]+(1+9*torch.rand(len(pending),1,device=self.device))*direction
            angle=2*math.pi*torch.rand(len(pending),device=self.device)
            target=center[pending]+(2*torch.rand_like(xy)-1)*4.5
            delta=target-xy;distance=delta.norm(dim=-1)
            forward=torch.stack((-torch.sin(angle),torch.cos(angle)),-1)
            valid=(distance>=1)&((target-human[pending]).norm(dim=-1)>=1)
            valid &= ((delta/distance.clamp_min(1e-6)[:,None])*forward).sum(-1)>=math.cos(math.pi/4)
            rows=pending[valid]
            positions[rows]=xy[valid];goals[rows]=target[valid];yaw[rows]=angle[valid]
            pending=pending[~valid]
        if len(pending):raise RuntimeError('Single PUSH scene rejection exhausted')
        self._box_states[ids,0,:2]=positions
        self._box_states[ids,0,2]=self._box_size[ids,0,2]/2
        self._box_states[ids,0,3:7]=torch.stack((torch.zeros_like(yaw),torch.zeros_like(yaw),torch.sin(yaw/2),torch.cos(yaw/2)),-1)
        self._box_states[ids,0,7:]=0
        self._single_push_goals=self._tar_pos.clone()
        self._single_push_goals[ids,0,:2]=goals
        self._single_push_goals[ids,0,2]=self._box_size[ids,0,2]/2
        self._platform_pos[ids]=self._platform_default_pos[ids]

    def _reset_task(self, ids):
        if self.benchmark_task=='push_at':
            self._tar_pos[ids]=self._single_push_goals[ids]
            self._tar_platform_pos[ids]=self._tar_platform_default_pos[ids]
            return
        return super()._reset_task(ids)

    def _ontop_scene_infeasible(self, ids):
        # The MD specifies the source/goal samplers, not an additional pairwise
        # bounding-disc rejection. All tests start from loco, not interaction RSI.
        return torch.zeros(len(ids),dtype=torch.bool,device=self.device)

    def _reset_envs(self, ids):
        if self.initial_plan is None:
            return super()._reset_envs(ids)
        if not len(ids):return
        if self._fixed_platforms and not self._platform_warmup_done:
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim,True)
            self._refresh_sim_tensors()
            self._platform_warmup_done=True
        plan=self.initial_plan
        for attr,key in [('_humanoid_root_states','humanoid_root_state'),('_dof_pos','dof_pos'),
            ('_dof_vel','dof_vel'),('_box_states','physical_boxes'),('_tar_pos','goals'),
            ('_platform_pos','platform_pos'),('_tar_platform_pos','tar_platform_pos'),
            ('_logical_box_order','logical_box_order'),
            ('_kinematic_humanoid_rigid_body_states','kinematic_humanoid')]:
            getattr(self,attr)[ids]=torch.as_tensor(plan[key],device=self.device)[ids]
        graph=self.relation_runtime.graph
        for key in ('edge_valid','edge_owner','edge_relation','edge_src','edge_dst','required_goal',
                    'prereq_mask','term_index'):
            getattr(graph,key)[ids]=torch.as_tensor(plan[key],device=self.device)[ids]
        self._edge_goal_owners[ids]=True
        self._every_env_init_dof_pos[ids]=self._dof_pos[ids]
        owner=torch.zeros_like(ids)
        self._reset_default_slots=None
        self._reset_ref_slots={'loco':(ids,owner)}
        self._reset_ref_motion_ids={'loco':torch.as_tensor(plan['loco_motion_ids'],device=self.device)[ids]}
        self._reset_ref_motion_times={'loco':torch.as_tensor(plan['loco_motion_times'],device=self.device)[ids]}
        self._reset_env_tensors(ids)
        self._refresh_sim_tensors()
        self._reset_relation_history(ids)
        self._compute_observations(ids)
        self._init_amp_obs(ids)


def save_initial_plan(task, path):
    graph=task.relation_runtime.graph
    arrays={key:getattr(graph,key).cpu().numpy().copy() for key in
        ('edge_valid','edge_owner','edge_relation','edge_src','edge_dst','required_goal','prereq_mask','term_index')}
    for key,attr in [('humanoid_root_state','_humanoid_root_states'),('dof_pos','_dof_pos'),
        ('dof_vel','_dof_vel'),('physical_boxes','_box_states'),('physical_sizes','_box_size'),
        ('goals','_tar_pos'),('platform_pos','_platform_pos'),('tar_platform_pos','_tar_platform_pos'),
        ('logical_box_order','_logical_box_order'),('kinematic_humanoid','_kinematic_humanoid_rigid_body_states')]:
        arrays[key]=getattr(task,attr).cpu().numpy().copy()
    arrays['loco_motion_ids']=task._reset_ref_motion_ids['loco'].cpu().numpy().copy()
    arrays['loco_motion_times']=task._reset_ref_motion_times['loco'].cpu().numpy().copy()
    np.savez_compressed(path,**arrays)
