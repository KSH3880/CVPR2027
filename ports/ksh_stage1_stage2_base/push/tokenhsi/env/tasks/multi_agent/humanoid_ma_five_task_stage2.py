"""Stage2 shared-full-success architecture with PUSH_AT-only coupling."""
import math
from isaacgym import gymapi
import torch
from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
from env.tasks.multi_agent.humanoid_ma_five_task import HumanoidMAFiveTask
from env.tasks.multi_agent.humanoid_ma_shared_support import LocoRsi
from utils.edge_ontop_spec import select_graph, copy_graph_rows
from utils.edge_context_spec import PUSH_CONTACT
from utils.door_reference import rotate, multiply, yaw_from_quat
from utils.push_point import common_face_state, common_face_contact


class HumanoidMAFiveTaskStage2(HumanoidMAFiveTask):
    _solo_cache_class = LocoRsi
    __init__ = HumanoidMACarry.__init__
    _reset_ref_state_init = HumanoidMACarry._reset_ref_state_init

    def get_relation_suffix_size(self):
        return 6*self.num_agents

    def _load_box_asset(self):
        from utils.size_rsi import sample_sizes
        from utils.stage1_push import sample_box_sizes
        env = self.cfg['env']
        n,m,o = self.num_envs,self.num_agents,self.num_objects
        native = sample_sizes(n,self.device,'random_scenario',env['box']['build']['taskSizeRanges'],
                              num_agents=m,num_objects=o)
        native[:,m:2*m] = sample_box_sizes(n,self.device,env)[:,:m]
        self._box_size = native
        self._box_scale = native/torch.tensor(self._build_base_size,device=self.device)
        density = torch.full((n,o),100.,device=self.device)
        density[:,m:2*m] = env['pushBox']['mass']/native[:,m:2*m].prod(-1)
        self._box_density = density.flatten()
        assets={}
        self._box_assets=[]
        for index,dimensions in enumerate(native.reshape(-1,3).cpu().tolist()):
            push = m <= index%o < 2*m
            key=tuple(dimensions),push
            if key not in assets:
                options=gymapi.AssetOptions()
                options.density=env['pushBox']['mass']/math.prod(dimensions) if push else 100.
                options.angular_damping=options.linear_damping=.01
                options.max_angular_velocity=100.
                options.default_dof_drive_mode=gymapi.DOF_MODE_NONE
                asset=self.gym.create_box(self.sim,*dimensions,options)
                if push:
                    props=self.gym.get_asset_rigid_shape_properties(asset)
                    for prop in props:
                        prop.friction=env['pushBox']['friction']
                        prop.restitution=env['pushBox']['restitution']
                    self.gym.set_asset_rigid_shape_properties(asset,props)
                assets[key]=asset
            self._box_assets.append(assets[key])

    def _build_assignment_tensors(self):
        HumanoidMACarry._build_assignment_tensors(self)
        n,m=self.num_envs,self.num_agents
        self._groups=torch.arange(m,device=self.device)[None].expand(n,-1).clone()
        self._shared=torch.zeros(n,m,device=self.device,dtype=torch.bool)
        self._templates=torch.full((n,m),3,device=self.device,dtype=torch.long)

    def _sample_episode_graph(self,ids):
        from utils.five_task_stage2 import sample_graph
        graph,groups,shared,templates,sources=sample_graph(self._size_template_probabilities[ids],
            self._relation_graph_spec,self.cfg['env']['sharedSupport'],self._task_graph_preset)
        copy_graph_rows(self.relation_runtime.graph,ids,graph)
        self._groups[ids],self._shared[ids],self._templates[ids]=groups,shared,templates
        self._agent_box_assignment[ids]=sources
        self._edge_goal_owners[ids]=True
        self._ontop_maintain_steps[ids]=0
        self._sampling_counts[0]+=len(ids)
        self._sampling_counts[1:7]+=torch.bincount(templates.flatten(),minlength=6)
        self._sampling_counts[7]+=shared.any(-1).sum()
        for offset,rel in enumerate((6,7,8,9,10),8):
            self._sampling_counts[offset]+=((graph.edge_relation==rel)&graph.edge_valid).sum()

    def _sample_box_assignments(self,ids):
        return

    def _compute_clean_scene_obs(self,env_ids=None):
        ids=torch.arange(self.num_envs,device=self.device) if env_ids is None else env_ids
        raw=HumanoidMACarry._compute_clean_scene_obs(self,env_ids)
        physical=torch.where(self._shared[ids],self._agent_box_assignment[ids],-1)
        return torch.cat((raw,physical.float()),-1)

    def _assigned_push_state(self,ids):
        physical=self._agent_box_assignment[ids]
        boxes=self._box_states[ids[:,None],physical]
        sizes=self._box_size[ids[:,None],physical]
        goals=self._tar_pos[ids[:,None],torch.where(self._shared[ids],self._groups[ids],
                          torch.arange(self.num_agents,device=self.device)[None])]
        return boxes,sizes,goals,physical

    def _evaluate_ontop_context(self,env_ids=None):
        phi,diag=HumanoidMACarry._evaluate_ontop_context(self,env_ids)
        ids=torch.arange(self.num_envs,device=self.device) if env_ids is None else env_ids
        bodies=self._rigid_body_pos if env_ids is None else self._kinematic_humanoid_rigid_body_states[ids,...,:3]
        boxes,sizes,goals,_=self._assigned_push_state(ids)
        score=common_face_state(bodies[:,:,self._key_body_ids[:2]],boxes,sizes,goals)
        if env_ids is not None:
            self._push_hits[ids]=False
        graph=select_graph(self.relation_runtime.graph,ids)
        mask=graph.edge_valid&(graph.edge_relation==PUSH_CONTACT)
        phi=torch.where(mask,score.gather(1,graph.edge_owner),phi)
        distance=(bodies[:,:,0,:2]-boxes[...,:2]).norm(dim=-1)
        approach=1/(1+(distance-.5).clamp_min(0))
        diag['progress']=torch.where(mask,approach.gather(1,graph.edge_owner),diag['progress'])
        return phi,diag

    def _reset_push_slots(self,ids):
        m=self.num_agents
        chosen=self._templates[ids]==5
        local,owner=chosen.nonzero(as_tuple=True)
        rows=ids[local]
        self._push_hits[ids]=False
        if not len(rows):
            return
        leader=torch.where(self._shared[rows,owner],self._groups[rows,owner],owner)
        # Exactly one sampled box/goal per group, irrespective of packet order.
        pairs=torch.stack((rows,leader),-1).unique(dim=0)
        es,group=pairs.unbind(-1)
        physical=m+group
        boxes=self._box_states[es,physical].clone()
        centers=boxes.new_tensor([[-7.,-7.],[7.,7.],[-7.,7.],[7.,-7.]])
        boxes[:,:2]=self._env_origins[es,:2]+centers[group]+(torch.rand(len(es),2,device=self.device)-.5)
        boxes[:,2]=self._box_size[es,physical,2]/2
        yaw=torch.rand(len(es),device=self.device)*2*math.pi
        boxes[:,3:7]=0
        boxes[:,5],boxes[:,6]=(yaw/2).sin(),(yaw/2).cos()
        boxes[:,7:]=0
        self._box_states[es,physical]=boxes
        forward=rotate(boxes[:,3:7],boxes.new_tensor([0.,1.,0.]).expand(len(es),3))
        lo,hi=self.cfg['env']['pushGoalRandomization']['distance_range']
        distance=lo+(hi-lo)*torch.rand(len(es),device=self.device)
        self._tar_pos[es,group]=boxes[:,:3]+forward*distance[:,None]
        physical=m+leader
        box=self._box_states[rows,physical]
        forward=rotate(box[:,3:7],box.new_tensor([0.,1.,0.]).expand(len(rows),3))
        tangent=torch.stack((forward[:,1],-forward[:,0],torch.zeros(len(rows),device=self.device)),-1)
        same=(self._groups[rows]==leader[:,None])&self._shared[rows]
        same|=torch.arange(m,device=self.device)[None]==owner[:,None]
        rank=(same&(torch.arange(m,device=self.device)[None]<owner[:,None])).sum(-1)
        lateral=(rank-(same.sum(-1)-1)/2)*.9
        # Preserve each sampled loco pose, rotating and translating it rigidly.
        pose=self._kinematic_humanoid_rigid_body_states[rows,owner]
        angle=yaw_from_quat(box[:,3:7])+math.pi/2-yaw_from_quat(pose[:,0,3:7])
        q=pose.new_zeros(len(rows),4)
        q[:,2],q[:,3]=(angle/2).sin(),(angle/2).cos()
        qs=q[:,None].expand(-1,self.num_bodies,-1)
        body=rotate(qs,pose[...,:3]-pose[:,:1,:3])
        rotations=multiply(qs,pose[...,3:7])
        velocities,angular=rotate(qs,pose[...,7:10]),rotate(qs,pose[...,10:13])
        lo,hi=self.cfg['env']['pushTask']['loco_distance_range_m']
        distance=lo+(hi-lo)*torch.rand(len(rows),device=self.device)
        root=box[:,:3]-forward*distance[:,None]+tangent*lateral[:,None]
        root[:,2]=pose[:,0,2]
        body+=root[:,None]
        geometry=self._push_reset_geometry()
        body[...,2]+=geometry.floor_shift(body,rotations)[:,None]
        shift=geometry.rear_shift(body,rotations,box,self._box_size[rows,physical])
        body-=forward[:,None]*shift[:,None,None]
        self._set_pose(rows,owner,body,rotations,velocities,angular,
                       self._dof_pos[rows,owner],self._dof_vel[rows,owner])

    def _physics_step(self):
        ids=torch.arange(self.num_envs,device=self.device)
        for _ in range(self.control_freq_inv):
            self.render()
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim,True)
            self._refresh_sim_tensors()
            boxes,sizes,goals,physical=self._assigned_push_state(ids)
            self._push_hits|=common_face_contact(self._rigid_body_pos[:,:,self._key_body_ids[:2]],
                self._contact_forces[:,:,self._key_body_ids[:2]],boxes,
                self._ontop_box_contact[ids[:,None],physical],sizes,goals,self.cfg['env']['pushTask'])

    def _ontop_scene_infeasible(self, ids):
        invalid = HumanoidMACarry._ontop_scene_infeasible(self, ids)
        roots = self._humanoid_root_states[ids, :, :2]
        distance = (roots[:, :, None]-roots[:, None]).norm(dim=-1)
        off_diagonal = ~torch.eye(self.num_agents, device=self.device, dtype=torch.bool)
        return invalid | ((distance < .7) & off_diagonal).flatten(1).any(-1)
