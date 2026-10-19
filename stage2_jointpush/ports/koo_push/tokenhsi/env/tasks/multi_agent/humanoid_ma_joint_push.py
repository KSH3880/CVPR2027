"""Shared two-person PUSH+AT, loco-only RSI, original conditioned Push AMP."""
import hashlib
from pathlib import Path
from isaacgym import gymapi
import torch
from env.tasks.multi_agent.humanoid_ma_open_push import HumanoidMAOpenPush
from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
from env.tasks.multi_agent.joint_push_reward import joint_push_reward
from utils.door_reference import rotate, multiply, yaw_from_quat
from utils.door_hold import collect_foot_support


def cpa_penalty(pos, vel, distance, discount, dt):
    p=pos[:,:,None,:2]-pos[:,None,:,:2];v=vel[:,:,None,:2]-vel[:,None,:,:2]
    dot=(p*v).sum(-1)
    closing=(-dot/(p.norm(dim=-1)*v.norm(dim=-1)).clamp_min(1e-6)).clamp(0,1)
    time=(-dot/v.square().sum(-1).clamp_min(1e-6)).clamp_min(0)
    risk=((distance-(p+time[...,None]*v).norm(dim=-1))/distance).clamp(0,1)
    penalty=closing*risk*discount**(time/dt)
    return penalty.masked_fill(torch.eye(pos.shape[1],device=p.device,dtype=torch.bool),0).amax(-1)


class HumanoidMAJointPush(HumanoidMAOpenPush):
    RSI_MODE = "loco_only"
    def __init__(self,cfg,*args,**kwargs):
        self._joint_push=True
        spec=cfg['env']['jointPush']
        self._three_agent_preview=cfg['env'].get('jointPushThreeAgentPreview',False)
        preview_agents=cfg['env'].get('jointPushPreviewAgents',3)
        random_preview=self._three_agent_preview and bool(spec.get('shape_randomization'))
        if self._three_agent_preview and (not cfg['args'].test or cfg['env']['numAgents']!=preview_agents or preview_agents not in (3,4) or (not random_preview and (cfg['env']['pushBox']['mass']!=60 or spec.get('random_spawn')))):
            raise ValueError('Three-agent preview requires test mode, 3 humans, 60kg and fixed lanes')
        if cfg['env']['numAgents']!=(preview_agents if self._three_agent_preview else 2) or cfg['env']['numObjects']!=4:
            raise ValueError('Joint Push requires two agents and four observation object slots')
        if cfg['env'].get('openPushRSI') or cfg['env']['skillInitProb']!=[1.,0.,0.]:
            raise ValueError('Joint Push requires loco-only RSI')
        if spec['rsi']!=self.RSI_MODE or spec['amp_history']!='repeat_reset_state':
            raise ValueError('Unsupported joint Push initialization')
        if spec.get('goal_mode','goal360') not in ('goal360','box_forward'):
            raise ValueError('Unsupported joint Push goal mode')
        if spec.get("face_mode", "goal_dynamic") not in ("goal_dynamic", "reset_fixed"):
            raise ValueError("Unsupported Push face mode")
        spawn=spec.get('random_spawn')
        if spawn is not None and (self.RSI_MODE!='loco_only' or spawn.get('mode') not in ('rear_independent','surround_independent')):
            raise ValueError('Random spawn currently requires loco-only RSI')
        self._joint_push_spec=spec
        from env.tasks.multi_agent.joint_push_reward import validate_config
        validate_config(spec['reward'])
        if cfg['env']['relationReward'].get('joint_push_reward')!=spec['reward']:
            raise ValueError('Effective reward metadata differs')
        if spec['collision'] != {'mode':'cpa','ttc_discount':.99}:
            raise ValueError('Joint Push requires Carry CPA collision contract')
        if not kwargs.get("headless",False):
            cfg["env"]["cameraFollow"]=False
        super().__init__(cfg,*args,**kwargs)
        if self._three_agent_preview:
            import json
            mass=sum(p.mass for p in self.gym.get_actor_rigid_body_properties(self.envs[0],self._box_handles[2]))
            assert (random_preview or abs(mass-60)<1e-3) and self._humanoid_root_states.shape[1]==preview_agents
            assert self.relation_runtime.graph.edge_src.shape[-1]==2*preview_agents
            report=dict(agents=self.num_agents,actual_mass_kg=mass,size=self._box_size[0,2].tolist(),edges=2*preview_agents)
            directory=Path(self.cfg['args'].output_path);directory.mkdir(parents=True,exist_ok=True)
            (directory/'three_agent_preview_runtime.json').write_text(json.dumps(report,indent=2))
            print('[Joint Push preview runtime]',report,flush=True)
        import os
        if self.RSI_MODE=='loco_only' and os.environ.get('JOINT_PUSH_AUDIT_PATH'):
            from utils.joint_push_audit import audit_runtime
            audit_runtime(self,os.environ['JOINT_PUSH_AUDIT_PATH'])

    def validate_inference_checkpoint(self,checkpoint):
        if not self._three_agent_preview:
            from utils.relation_task_spec import check_checkpoint_metadata, checkpoint_metadata
            check_checkpoint_metadata(checkpoint,checkpoint_metadata(self._relation_cfg))
            if checkpoint.get('open_push_contract')!=self.get_open_push_contract():
                raise ValueError('OPEN/PUSH checkpoint contract differs')
            return
        import copy
        saved=copy.deepcopy(checkpoint.get('relation_experiment_config',{}).get('env',{}))
        if self._joint_push_spec.get('shape_randomization'):
            if saved.get('numAgents')!=2 or saved.get('pushBox')!=self.cfg['env']['pushBox'] or saved.get('jointPush')!=self._joint_push_spec or checkpoint.get('open_push_contract')!=self.get_open_push_contract():
                raise ValueError('Random preview must preserve the source physics/reward contract')
            print(f'[Joint Push preview] shared 2-person policy -> {self.num_agents}-person; inference only',flush=True)
            return
        if saved.get('pushBox',{}).get('mass')!=40 or checkpoint.get('open_push_contract',{}).get('policy')!='push_at_ca4':
            raise ValueError('Preview requires the original 40kg Stage2 policy')
        saved['pushBox']['mass']=60
        if saved['pushBox']!=self.cfg['env']['pushBox'] or saved['jointPush']!=self._joint_push_spec:
            raise ValueError('Preview must preserve source size, friction and task settings')
        print('[Joint Push preview] 2-person/40kg policy -> 3-person/60kg; inference only',flush=True)

    def _load_box_asset(self):
        super()._load_box_asset()
        import os,math
        override=os.environ.get('JOINT_PUSH_VIEW_MASS')
        if override is not None:
            mass=float(override)
            if not self.cfg['args'].test or not self.cfg['args'].eval or self._three_agent_preview or not math.isfinite(mass) or mass<=0:
                raise ValueError('Mass diagnostic requires an ordinary finite-mass evaluation')
            size=self.cfg['env']['pushBox']['size']
            options=gymapi.AssetOptions();options.density=mass/math.prod(size)
            options.angular_damping=.01;options.linear_damping=.01
            self._mass_preview_original_push_asset=self._push_asset
            self._push_asset=self.gym.create_box(self.sim,*size,options)
            self._mass_preview_constructed=mass
        if self.headless:
            return
        # Viewer-only visual removal. Keep the parked actors/collisions and obs slots.
        import tempfile
        import xml.etree.ElementTree as ET
        self._hidden_asset_dir=tempfile.TemporaryDirectory(prefix='jointpush_hidden_')
        directory=Path(self._hidden_asset_dir.name)
        source=Path(__file__).resolve().parents[3]/'assets/door/door_hold.urdf'
        tree=ET.parse(str(source))
        for link in tree.getroot().findall('link'):
            for visual in list(link.findall('visual')):link.remove(visual)
            visual=ET.SubElement(link,'visual')
            geometry=ET.SubElement(visual,'geometry')
            ET.SubElement(geometry,'sphere',radius='0.000001')
        tree.write(str(directory/'door.urdf'))
        options=gymapi.AssetOptions();options.fix_base_link=True;options.collapse_fixed_joints=False
        options.default_dof_drive_mode=gymapi.DOF_MODE_NONE
        hidden=self.gym.load_asset(self.sim,str(directory),'door.urdf',options)
        assert self.gym.get_asset_dof_count(hidden)==1
        assert self.gym.get_asset_rigid_body_count(hidden)==self._door_body_count
        assert self.gym.find_asset_rigid_body_index(hidden,'panel')==self._door_leaf_index
        assert self.gym.find_asset_rigid_body_index(hidden,'handle')==self._door_handle_index
        self._door_asset=hidden
        x,y,z=self.cfg['env']['pushBox']['size'];mass=self.cfg['env']['pushBox']['mass']
        xml=f'<robot name="hidden_box"><link name="box"><visual><geometry><sphere radius="0.000001"/></geometry></visual><inertial><mass value="{mass}"/><inertia ixx="{mass*(y*y+z*z)/12}" iyy="{mass*(x*x+z*z)/12}" izz="{mass*(x*x+y*y)/12}" ixy="0" ixz="0" iyz="0"/></inertial><collision><geometry><box size="{x} {y} {z}"/></geometry></collision></link></robot>'
        (directory/'box.urdf').write_text(xml)
        options=gymapi.AssetOptions();options.angular_damping=.01;options.linear_damping=.01
        self._invisible_push_asset=self.gym.load_asset(self.sim,str(directory),'box.urdf',options)
        assert self.gym.get_asset_rigid_body_count(self._invisible_push_asset)==1

    def _build_box(self,env_id,object_id,env_ptr):
        inactive=getattr(self,'_invisible_push_asset',getattr(self,'_mass_preview_original_push_asset',None))
        if object_id!=3 or inactive is None:
            return super()._build_box(env_id,object_id,env_ptr)
        visible=self._push_asset
        try:
            self._push_asset=inactive
            return super()._build_box(env_id,object_id,env_ptr)
        finally:self._push_asset=visible

    def get_open_push_contract(self):
        contract=super().get_open_push_contract()
        contract.update(joint_push=self._joint_push_spec,
                        push_reward_contract='joint_push_symmetric_midpoints_v1',
                        policy='push_at_ca4',shared_payload=2,shared_goal=0)
        return contract

    def _init_relation_runtime(self):
        super()._init_relation_runtime()
        self._push_binding.zero_()

    def _sample_tasks(self,env_ids):
        return torch.ones(len(env_ids),self.num_agents,device=self.device,dtype=torch.long)

    def _sample_box_assignments(self,env_ids):
        self._logical_box_order[env_ids]=torch.arange(4,device=self.device)
        self._agent_box_assignment[env_ids]=2
        graph=self.relation_runtime.graph
        if self._three_agent_preview:
            return  # Runtime already holds three shared PUSH+AT bundles.
        graph.edge_src[env_ids]=torch.tensor([0,4,1,4],device=self.device)
        graph.edge_dst[env_ids]=torch.tensor([4,6,4,6],device=self.device)
        graph.edge_relation[env_ids]=torch.tensor([12,7,12,7],device=self.device)
        graph.edge_valid[env_ids]=True
        graph.required_goal[env_ids]=torch.tensor([False,True,False,True],device=self.device)

    def _reset_boxes(self,env_ids):
        super()._reset_boxes(env_ids)
        yaw=torch.rand(len(env_ids),device=self.device)*2*torch.pi
        box=self._box_states[env_ids,2].clone();box[:,:2]=0;box[:,2]=self._box_size[env_ids,2,2]/2
        box[:,3:7]=0;box[:,5]=torch.sin(yaw/2);box[:,6]=torch.cos(yaw/2);box[:,7:]=0
        self._box_states[env_ids,2]=box
        self._box_states[env_ids,3,:2]=box.new_tensor([5.,5.])
        self._box_states[env_ids,:2,:2]=box.new_tensor([[5.,-5.],[-5.,5.]])
        self._push_reset_states[env_ids]=self._box_states[env_ids,2:]
        # All directions, matching Goal360; fixed goal survives contact changes.
        aligned=self._joint_push_spec.get('goal_mode','goal360')=='box_forward'
        angle=None if aligned else torch.rand(len(env_ids),device=self.device)*2*torch.pi
        lo,hi=self._joint_push_spec['goal_distance']
        distance=lo+(hi-lo)*torch.rand(len(env_ids),device=self.device)
        if aligned:
            from env.tasks.multi_agent.joint_push_goal_geometry import aligned_goal_offset
            offset=aligned_goal_offset(yaw,distance)
        else:
            offset=torch.stack((distance*angle.cos(),distance*angle.sin(),torch.zeros_like(distance)),-1)
        self._push_goal_offsets[env_ids]=offset[:,None]
        # Store one box-local face for both agents; update only the reset subset.
        if not hasattr(self, '_push_face_local_normal'):
            self._push_face_local_normal=torch.zeros(self.num_envs,2,device=self.device)
        c,s=yaw.cos(),yaw.sin()
        local=torch.stack((c*offset[:,0]+s*offset[:,1],-s*offset[:,0]+c*offset[:,1]),-1)
        axis=(local[:,1].abs()>=local[:,0].abs()).long()
        sign=torch.where(local.gather(1,axis[:,None])>=0,1.,-1.)
        self._push_face_local_normal[env_ids]=torch.zeros_like(local).scatter(1,axis[:,None],sign)
        if self._joint_push_spec.get('random_spawn'):
            from env.tasks.multi_agent.joint_push_spawn import sample_rear_spawn
            spawn=self._joint_push_spec['random_spawn']
            if not hasattr(self,'_push_spawn_depth'):
                self._push_spawn_depth=torch.zeros(self.num_envs,self.num_agents,device=self.device)
                self._push_spawn_lateral=torch.zeros_like(self._push_spawn_depth)
                self._push_spawn_heading=torch.zeros_like(self._push_spawn_depth)
            if spawn['mode']=='surround_independent':
                from env.tasks.multi_agent.joint_push_spawn import sample_surround_spawn
                if self._joint_push_spec.get('goal_mode')!='box_forward' and self._joint_push_spec['reward'].get('contact_mode') not in ('directional_surface','assigned_faces'):
                    raise ValueError('Surround profile currently requires local +Y initial forward')
                depth,lateral,heading=sample_surround_spawn(len(env_ids),self.device,
                    self._box_size[env_ids,2],self._joint_push_spec['spawn_distance'],spawn['minimum_root_gap'],num_agents=self.num_agents)
            else:
                depth,lateral,heading=sample_rear_spawn(len(env_ids),self.device,
                    self._joint_push_spec['spawn_distance'],spawn['lateral_limit'],spawn['minimum_root_gap'])
            self._push_spawn_depth[env_ids]=depth
            self._push_spawn_lateral[env_ids]=lateral
            self._push_spawn_heading[env_ids]=heading

    def _reset_task(self,env_ids):
        target=self._box_states[env_ids,2,:3]+self._push_goal_offsets[env_ids,0]
        self._tar_pos[env_ids]=target[:,None]

    def _reset_ref_state_init(self,slot_env,slot_agent):
        HumanoidMACarry._reset_ref_state_init(self,slot_env,slot_agent)
        root=self._humanoid_root_states[slot_env,slot_agent].clone()
        body=self._kinematic_humanoid_rigid_body_states[slot_env,slot_agent].clone()
        from env.tasks.multi_agent.joint_push_reward import push_face_geometry
        boxes=self._box_states[slot_env,2]
        goals=boxes[:,:3]+self._push_goal_offsets[slot_env,0]
        center,normal,tangent,half_width,_=push_face_geometry(boxes,goals,self._box_size[slot_env,2],self._joint_push_spec['reward'])
        facing=torch.atan2(normal[:,1],normal[:,0])
        if self._joint_push_spec.get('random_spawn'):
            facing=facing+self._push_spawn_heading[slot_env,slot_agent]
        turn=facing-yaw_from_quat(root[:,3:7])
        q=torch.zeros_like(root[:,3:7]);q[:,2]=torch.sin(turn/2);q[:,3]=torch.cos(turn/2)
        lo,hi=self._joint_push_spec['spawn_distance']
        distance=lo+(hi-lo)*torch.rand(len(slot_env),device=self.device)
        side=(slot_agent.to(root.dtype)*2/(self.num_agents-1)-1)*half_width*(.65 if self._three_agent_preview else .5)
        if self._joint_push_spec.get('random_spawn'):
            distance=self._push_spawn_depth[slot_env,slot_agent]
            side=self._push_spawn_lateral[slot_env,slot_agent]
        target=root[:,:3].clone();target[:,:2]=center[:,:2]-normal*distance[:,None]+tangent*side[:,None]
        expanded=q[:,None].expand_as(body[...,3:7])
        body[...,:3]=rotate(expanded,body[...,:3]-root[:,None,:3])+target[:,None]
        body[...,3:7]=multiply(expanded,body[...,3:7]);body[...,7:10]=rotate(expanded,body[...,7:10]);body[...,10:13]=rotate(expanded,body[...,10:13])
        self._set_env_state(slot_env,slot_agent,target,multiply(q,root[:,3:7]),self._dof_pos[slot_env,slot_agent].clone(),
            rotate(q,root[:,7:10]),rotate(q,root[:,10:13]),self._dof_vel[slot_env,slot_agent].clone())
        self._kinematic_humanoid_rigid_body_states[slot_env,slot_agent]=body

    def _reset_envs(self,env_ids):
        if not len(env_ids):return
        self._reset_default_slots=None;self._reset_ref_slots={};self._reset_ref_motion_ids={};self._reset_ref_motion_times={}
        self._reset_boxes(env_ids);self._agent_lanes[env_ids]=torch.arange(self.num_agents,device=self.device)%2;self._push_binding[env_ids]=0
        self.relation_runtime.reset(env_ids,self._sample_tasks(env_ids),self._owner_push_boxes(env_ids))
        self._sample_box_assignments(env_ids);self._reset_actors(env_ids);self._reset_task(env_ids)
        if self._joint_push_spec['reward'].get('contact_mode')=='assigned_faces':
            from env.tasks.multi_agent.joint_push_assigned_contact import assign_contact_targets
            spec=self._joint_push_spec['assigned_contact']
            assignment=assign_contact_targets(self._box_states[env_ids,2],self._tar_pos[env_ids,0],
                self._box_size[env_ids,2],self._humanoid_root_states[env_ids],spec,
                self._joint_push_spec['reward']['hand_target_height'])
            if not hasattr(self,'_assigned_contact_buffers'):
                self._assigned_contact_buffers={k:torch.zeros((self.num_envs,)+v.shape[1:],device=v.device,dtype=v.dtype) for k,v in assignment.items()}
            for key,value in assignment.items():self._assigned_contact_buffers[key][env_ids]=value
        if self.viewer is not None:
            if not hasattr(self,'_push_visual_initial_roots'):
                self._push_visual_initial_roots=self._humanoid_root_states.clone()
                self._push_visual_initial_yaw=torch.zeros(self.num_envs,device=self.device)
            self._push_visual_initial_roots[env_ids]=self._humanoid_root_states[env_ids]
            self._push_visual_initial_yaw[env_ids]=yaw_from_quat(self._box_states[env_ids,2,3:7])
            if 0 in env_ids:
                print('[Joint Push viewer] initial headings (deg):',
                    (yaw_from_quat(self._humanoid_root_states[0,:,3:7])*180/torch.pi).tolist(),
                    'spawn mode:',self._joint_push_spec.get('random_spawn',{}).get('mode','fixed_lanes'),flush=True)
        self._reset_env_tensors(env_ids);self._sync_door_reset_states(env_ids)
        self._compute_observations(env_ids);self._init_amp_obs(env_ids)
        self._compute_amp_observations(env_ids)
        self._hist_amp_obs_buf[env_ids]=self._curr_amp_obs_buf[env_ids,:,None]
        self._update_joint_colors(env_ids)
        if self.viewer is not None and (env_ids==0).any() and not getattr(self,"_joint_push_camera_initialized",False):
            self._place_joint_push_camera()
            self._joint_push_camera_initialized=True

    def _init_camera(self):
        # The base constructor must not replace the task's one-time side view.
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._cam_prev_char_pos=self._humanoid_root_states[0,0,:3].detach().cpu().numpy().copy()
        self._camera_follow_flag=False

    def _place_joint_push_camera(self):
        # Set a side view once. Subsequent episode resets preserve user camera edits.
        roots=self._humanoid_root_states[0,:,:3]
        center=roots.mean(0)
        box=self._box_states[0,2,:3]
        goal=self._tar_pos[0,0]
        yaw=yaw_from_quat(self._box_states[0,2,3:7])
        forward=torch.stack((-yaw.sin(),yaw.cos()))
        distance=max(6.,float((goal[:2]-center[:2]).norm())*.9)
        target=(center+box+goal)/3;target[2]=1.
        lateral=torch.stack((-forward[1],forward[0]))
        camera=target.clone();camera[:2]+=distance*lateral;camera[2]=3.
        self.gym.viewer_camera_look_at(self.viewer,None,
            gymapi.Vec3(*camera.detach().cpu().tolist()),
            gymapi.Vec3(*target.detach().cpu().tolist()))
        self._camera_follow_flag=False
        print('[Joint Push viewer] initial side camera; free-camera position preserved on resets',flush=True)
        self._cam_prev_char_pos=self._humanoid_root_states[0,0,:3].detach().cpu().numpy().copy()

    def _update_joint_colors(self,ids):
        if self.viewer is None:return
        for eid in ids.cpu().tolist():
            for o,handle in enumerate(self._box_handles[eid*4:eid*4+4]):
                color=gymapi.Vec3(1.,.8,.1) if o==2 else gymapi.Vec3(.45,.45,.45)
                for b in range(self._door_body_count if o<2 else 1):
                    self.gym.set_rigid_body_color(self.envs[eid],handle,b,gymapi.MESH_VISUAL,color)

    def render(self, sync_frame_time=False):
        super().render(sync_frame_time)
        if self.viewer is None or not hasattr(self,'_push_visual_initial_roots'):
            return
        from utils.joint_push_visual import overlay_lines
        vertices,colors=overlay_lines(self._tar_pos[0,0].detach().cpu().numpy(),
            self.cfg['env']['pushBox']['size'],float(self._push_visual_initial_yaw[0]),
            self._push_visual_initial_roots[0].detach().cpu().numpy(),
            self._humanoid_root_states[0].detach().cpu().numpy())
        self.gym.clear_lines(self.viewer)
        self.gym.add_lines(self.viewer,self.envs[0],len(vertices),vertices,colors)

    def _physics_step(self):
        for _ in range(self.control_freq_inv):
            self.render();self.gym.simulate(self.sim);self.gym.fetch_results(self.sim,True);self._refresh_sim_tensors()
            collect_foot_support(self)

    def _joint_push_face_geometry(self):
        if self._joint_push_spec.get('face_mode','goal_dynamic')!='reset_fixed':
            return None
        from env.tasks.multi_agent.joint_push_reward import push_face_geometry
        return push_face_geometry(self._box_states[:,2],self._tar_pos[:,0],
            self._box_size[:,2],self._joint_push_spec['reward'],
            local_normal=self._push_face_local_normal)

    def _compute_reward(self,actions):
        assigned_targets=None
        if self._joint_push_spec['reward'].get('contact_mode')=='assigned_faces':
            from env.tasks.multi_agent.joint_push_assigned_contact import world_targets
            assigned_targets=tuple(world_targets(self._box_states[:,2],self._assigned_contact_buffers[k])
                for k in ('local_hands','local_approach'))
        r=joint_push_reward(self._humanoid_root_states,self._rigid_body_pos[:,:,self._key_body_ids[:2]],
            self._box_states[:,2],self._tar_pos[:,0],self._box_size[:,2],self._joint_push_spec['reward'],face_geometry=self._joint_push_face_geometry(),assigned_targets=assigned_targets)
        runtime=self.relation_runtime
        runtime.phi.copy_(r['phi_raw'].reshape(self.num_envs,2*self.num_agents))
        runtime.own_success.copy_(r['own_success'].reshape(self.num_envs,2*self.num_agents));runtime.achieved.copy_(runtime.own_success)
        runtime.done.copy_(r['done']);runtime.error.copy_((self._box_states[:,2,:2]-self._tar_pos[:,0,:2]).norm(dim=-1)[:,None])
        power=-self._power_coefficient*(self.dof_force_tensor*self._dof_vel).abs().sum(-1) if self._power_reward else torch.zeros_like(r['agent_task_reward'])
        collision=-self._agent_collision_coeff*cpa_penalty(self._humanoid_root_states[...,:3],self._humanoid_root_states[...,7:10],self._agent_collision_dist,.99,self.dt) if self._agent_collision_penalty else torch.zeros_like(power)
        total=r['agent_task_reward']+power+collision;self.rew_buf[:]=total.flatten()
        terms=torch.stack((r['state_component'].sum(-1),r['progress_component'].sum(-1),r['success_component'].sum(-1),power,collision,total),-1).reshape(-1,6)
        self.extras['reward_terms']=terms;self._reward_term_sums+=terms.sum(0);self._reward_term_count+=len(terms)
        self.extras['joint_push_dense_at_gate']=r['dense_at_gate'].clone()
        self.extras['joint_push_gate']=r['coupled_contact'].clone()
        self.extras['joint_push_symmetry']=r['symmetry_phi'].clone()
        self.extras['joint_push_separation']=r['separation'].clone()
        self.extras['joint_push_box_pos']=self._box_states[:,2,:3].clone()
        if assigned_targets is not None:
            self.extras['joint_push_hand_targets']=assigned_targets[0].clone()
            self.extras['joint_push_approach_targets']=assigned_targets[1].clone()
            self.extras['joint_push_two_faces']=self._assigned_contact_buffers['two_faces'].clone()

    def consume_relation_diagnostics(self):
        return {}
