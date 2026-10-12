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
        offset=self._push_goal_offsets[env_ids,0]
        x=yaw.cos()*offset[:,0]+yaw.sin()*offset[:,1]
        y=-yaw.sin()*offset[:,0]+yaw.cos()*offset[:,1]
        # Freeze the initial lateral axis for the episode: no mid-push slot swaps.
        self._virtual_split_axis[env_ids]=torch.where(x.abs()>=y.abs(),1,0)
        self._virtual_forward_sign[env_ids]=torch.where(torch.where(x.abs()>=y.abs(),x,y)>=0,1.,-1.)

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
            virtual_sizes=self._virtual_geometry()[3][0].tolist(),physical_size=self.cfg['env']['pushBox']['size'],
            policy='original_stage1_no_training',split='rear_bottom_aligned_1.3m_cubes')
        directory=Path(self.cfg['args'].output_path);directory.mkdir(parents=True,exist_ok=True)
        with (directory/'virtual_push_trace.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
