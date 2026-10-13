"""Inference-only Stage1 baseline: one real box, two virtual half-box tasks."""
import json
from pathlib import Path
import torch
import yaml
from env.tasks.multi_agent.humanoid_ma_joint_push import HumanoidMAJointPush
from env.tasks.multi_agent.virtual_push_observation import split_box_observation
from utils.door_reference import yaw_from_quat


class HumanoidMAJointPushVirtual(HumanoidMAJointPush):
    def __init__(self,cfg,*args,**kwargs):
        if not cfg.get('args').test:
            raise ValueError('Virtual half-box baseline is inference-only')
        self._curve_spec=cfg['env'].get('curvePush')
        if self._curve_spec and cfg['env']['pushBox']['mass']!=60.:
            raise ValueError('Curve Push probe requires a 60kg physical box')
        self._virtual_split_axis=None
        self._virtual_forward_sign=None
        super().__init__(cfg,*args,**kwargs)

    def validate_inference_checkpoint(self,checkpoint):
        metadata=checkpoint.get('relation_metadata',{})
        old=checkpoint.get('relation_experiment_config',{})
        if metadata.get('schema_version')!=12 or metadata.get('relation_taxonomy',{}).get('push')!=12 or metadata.get('context_fusion')!='semantic_only':
            raise ValueError('Virtual baseline requires koo OPEN/PUSH Stage1')
        if 'joint_push' in checkpoint.get('open_push_contract',{}):
            raise ValueError('Virtual baseline requires original Stage1, not trained Stage2')
        if (old.get('num_agents'),old.get('num_objects'),old.get('observation_size'))!=(2,4,self.get_obs_size()):
            raise ValueError('Virtual baseline Stage1 observation dimensions differ')
        target=yaml.safe_load(Path(self.cfg['args'].cfg_train).read_text())['params']['network']
        if old['train']['params']['network']!=target:
            raise ValueError('Virtual baseline source network differs')
        print('[virtual Push] Strict Stage1 model restore; physical/observation geometry deliberately differs',flush=True)

    def _reset_boxes(self,env_ids):
        super()._reset_boxes(env_ids)
        if self._virtual_split_axis is None:
            self._virtual_split_axis=torch.zeros(self.num_envs,device=self.device,dtype=torch.long)
            self._virtual_forward_sign=torch.ones(self.num_envs,device=self.device)
        yaw=yaw_from_quat(self._box_states[env_ids,2,3:7])
        if self._curve_spec:
            from env.tasks.multi_agent.curve_push_geometry import sample_curve_goal_offset
            distance=self._push_goal_offsets[env_ids,0,:2].norm(dim=-1)
            offset,angle=sample_curve_goal_offset(yaw,distance,self._curve_spec['angle_abs_deg'])
            self._push_goal_offsets[env_ids]=offset[:,None]
            initial=dict(num_envs=len(env_ids),mass=float(self.gym.get_actor_rigid_body_properties(self.envs[0],self._box_handles[2])[0].mass),
                box_yaw=yaw.tolist(),goal_offset=offset.tolist(),misalignment_deg=(angle*180/torch.pi).tolist(),
                push_face='fixed_local_minus_y',virtual_size=[1.3,1.3,1.3])
            directory=Path(self.cfg['args'].output_path);directory.mkdir(parents=True,exist_ok=True)
            with (directory/'curve_reset.jsonl').open('a') as f:f.write(json.dumps(initial)+'\n')
        offset=self._push_goal_offsets[env_ids,0]
        x=yaw.cos()*offset[:,0]+yaw.sin()*offset[:,1]
        y=-yaw.sin()*offset[:,0]+yaw.cos()*offset[:,1]
        # Freeze the initial lateral axis for the episode: no mid-push slot swaps.
        self._virtual_split_axis[env_ids]=torch.where(x.abs()>=y.abs(),1,0)
        self._virtual_forward_sign[env_ids]=torch.where(torch.where(x.abs()>=y.abs(),x,y)>=0,1.,-1.)

    def _reset_ref_state_init(self,slot_env,slot_agent):
        super()._reset_ref_state_init(slot_env,slot_agent)
        # Match each owner's spawn lane to its virtual cube, independent of face sign.
        from env.tasks.multi_agent.joint_push_reward import push_face_geometry
        boxes=self._box_states[slot_env,2]
        goals=boxes[:,:3]+self._push_goal_offsets[slot_env,0]
        center,normal,tangent,_,_=push_face_geometry(boxes,goals,self.cfg['env']['pushBox']['size'],self._joint_push_spec['reward'])
        virtual=self._virtual_geometry(slot_env)[0]
        assigned=virtual[torch.arange(len(slot_env),device=self.device),slot_agent,:2]
        root=self._humanoid_root_states[slot_env,slot_agent].clone()
        lateral=((assigned-root[:,:2])*tangent).sum(-1)
        shift=torch.cat((tangent*lateral[:,None],torch.zeros_like(lateral[:,None])),-1)
        body=self._kinematic_humanoid_rigid_body_states[slot_env,slot_agent].clone()
        body[...,:3]+=shift[:,None]
        self._set_env_state(slot_env,slot_agent,root[:,:3]+shift,root[:,3:7],self._dof_pos[slot_env,slot_agent].clone(),
            root[:,7:10],root[:,10:13],self._dof_vel[slot_env,slot_agent].clone())
        self._kinematic_humanoid_rigid_body_states[slot_env,slot_agent]=body

    def _sample_box_assignments(self,env_ids):
        super()._sample_box_assignments(env_ids)
        graph=self.relation_runtime.graph
        graph.edge_src[env_ids]=torch.tensor([0,4,1,5],device=self.device)
        graph.edge_dst[env_ids]=torch.tensor([4,6,5,7],device=self.device)

    def _virtual_geometry(self,env_ids=None):
        ids=slice(None) if env_ids is None else env_ids
        return split_box_observation(self._box_states[ids,2],self.cfg['env']['pushBox']['size'],self._tar_pos[ids,0],self._virtual_split_axis[ids],self._virtual_forward_sign[ids])

    def _logical_box_values(self,values,env_ids=None):
        result=super()._logical_box_values(values,env_ids)
        if self._virtual_split_axis is None:return result
        if values is self._box_states:
            result=result.clone();result[:,2:]=self._virtual_geometry(env_ids)[0]
        elif values is self._box_bps:
            result=result.clone();result[:,2:]=self._virtual_geometry(env_ids)[1]
        return result

    def _compute_clean_scene_obs(self,env_ids=None):
        obs=super()._compute_clean_scene_obs(env_ids)
        if self._virtual_split_axis is None:return obs
        # Goal positions are pose records; node features remain the original ones.
        states,corners,targets,sizes=self._virtual_geometry(env_ids)
        base=2*self.get_clean_humanoid_obs_size()+4*self.get_clean_object_obs_size()+2*self.get_clean_goal_obs_size()
        poses=obs[:,base:base+8*7].view(-1,8,7)
        ids=slice(None) if env_ids is None else env_ids
        poses[:,6:,:3]=targets-self._env_origins[ids,None]
        return obs

    def _reset_envs(self,env_ids):
        super()._reset_envs(env_ids)
        if not len(env_ids) or not self._curve_spec:return
        from env.tasks.multi_agent.curve_push_geometry import fixed_rear_geometry
        center,normal,tangent,_,_=fixed_rear_geometry(self._box_states[env_ids,2],self.cfg['env']['pushBox']['size'],self._joint_push_spec['reward'])
        virtual=self._virtual_geometry(env_ids)[0]
        root=self._humanoid_root_states[env_ids,:,:2]
        lateral=((root-virtual[:,:,:2])*tangent[:,None]).sum(-1)
        depth=((root-center[:,None,:2])*normal[:,None]).sum(-1)
        assert lateral.abs().max()<1e-4, 'Spawn owner and virtual cube lane differ'
        lo,hi=self._joint_push_spec['spawn_distance']
        assert (depth>=-hi-1e-4).all() and (depth<=-lo+1e-4).all(), 'Spawn is not behind fixed rear face'
        directory=Path(self.cfg['args'].output_path)
        report=dict(num_envs=len(env_ids),spawn_lane_error_max=float(lateral.abs().max()),
            spawn_rear_depth_range=[float(depth.min()),float(depth.max())],
            virtual_size=self._virtual_geometry(env_ids)[3][0].tolist(),physical_mass=self.cfg['env']['pushBox']['mass'])
        with (directory/'curve_spawn_audit.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')

    def _joint_push_face_geometry(self):
        if not self._curve_spec:return None
        from env.tasks.multi_agent.curve_push_geometry import fixed_rear_geometry
        return fixed_rear_geometry(self._box_states[:,2],self.cfg['env']['pushBox']['size'],self._joint_push_spec['reward'])

    def _compute_reward(self,actions):
        super()._compute_reward(actions)
        # These are physical shared-box diagnostics, never virtual success.
        if int(self.progress_buf.max())%30:return
        r=self.extras
        hands=self._rigid_body_pos[:,:,self._key_body_ids[:2]].mean(2)
        virtual=self._virtual_geometry()[0]
        report=dict(step=int(self.progress_buf.max()),num_envs=self.num_envs,
            physical_goal_distance_mean=float((self._box_states[:,2,:2]-self._tar_pos[:,0,:2]).norm(dim=-1).mean()),
            physical_box_pos=self._box_states[0,2,:3].tolist(),physical_goal=self._tar_pos[0,0].tolist(),
            joint_gate_fraction=float(r['joint_push_gate'].float().mean()),symmetry_mean=float(r['joint_push_symmetry'].mean()),
            hand_midpoint_separation_mean=float(r['joint_push_separation'].mean()),
            hand_midpoints=hands[0].tolist(),virtual_centers=virtual[0,:,:3].tolist(),
            virtual_sizes=self._virtual_geometry()[3][0].tolist(),root_positions=self._humanoid_root_states[0,:,:3].tolist(),physical_size=self.cfg['env']['pushBox']['size'],
            policy='original_stage1_no_training',split='rear_bottom_aligned_1.3m_cubes',
            curve_probe=bool(self._curve_spec),box_yaw=yaw_from_quat(self._box_states[:,2,3:7]).tolist())
        directory=Path(self.cfg['args'].output_path);directory.mkdir(parents=True,exist_ok=True)
        with (directory/'virtual_push_trace.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
