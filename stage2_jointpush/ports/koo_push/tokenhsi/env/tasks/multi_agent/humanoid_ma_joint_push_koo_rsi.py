"""Shared Push RSI using koo's screened frames and exact sampling settings."""
import hashlib
import json
from pathlib import Path
import torch
import yaml
from env.tasks.multi_agent.humanoid_ma_joint_push import HumanoidMAJointPush
from env.tasks.multi_agent.joint_push_reward import push_face_geometry
from utils.open_push_rsi import OpenPushRSIBank,bank_contract,align_push_pose,lower_push_arms
from utils.door_reference import rotate,multiply


class HumanoidMAJointPushKooRSI(HumanoidMAJointPush):
    RSI_MODE='koo_push_mixed'

    def __init__(self,cfg,*args,**kwargs):
        super().__init__(cfg,*args,**kwargs)
        self._koo_spec=self._joint_push_spec['koo_rsi']
        source=yaml.safe_load(Path(self._koo_spec['source_config']).read_text())['env']
        source['motion_file']=self.cfg['env']['motion_file']
        spec=source['openPushRSI']
        if self._koo_spec['sampling']!=spec:
            raise ValueError('Joint RSI sampling differs from koo source config')
        data=json.loads(Path(spec['bank_file']).read_text())
        if data.get('screen')!='PD_12_steps' or data.get('contract')!=bank_contract(source):
            raise ValueError('Koo source bank physics/reference contract differs')
        self._koo_bank=OpenPushRSIBank(spec,self._motion_lib,data['contract'],proposals=data['candidates'])
        self._koo_bank_sha=hashlib.sha256(Path(spec['bank_file']).read_bytes()).hexdigest()
        self._koo_mask=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        self._koo_samples={}
        self._koo_goal=torch.zeros(self.num_envs,3,device=self.device)

    def get_open_push_contract(self):
        contract=super().get_open_push_contract()
        if hasattr(self,'_koo_bank_sha'):contract['koo_source_bank_sha256']=self._koo_bank_sha
        return contract

    def validate_inference_checkpoint(self,checkpoint):
        from utils.relation_task_spec import check_checkpoint_metadata,checkpoint_metadata
        check_checkpoint_metadata(checkpoint,checkpoint_metadata(self._relation_cfg))
        # This is an explicit RSI-only diagnostic of an existing Stage2 model.
        old=dict(checkpoint.get('open_push_contract',{}));new=dict(self.get_open_push_contract())
        for contract in (old,new):
            contract.pop('koo_source_bank_sha256',None)
            joint=dict(contract.get('joint_push',{}));joint.pop('rsi',None);joint.pop('koo_rsi',None)
            contract['joint_push']=joint
        if old!=new:raise ValueError('RSI diagnostic allows only initialization change; physics/reward differ')
        print('[Joint Push RSI] Strict Stage2 restore; explicit koo RSI-only diagnostic',flush=True)

    def _reset_boxes(self,env_ids):
        super()._reset_boxes(env_ids)
        selected=torch.rand(len(env_ids),device=self.device)<self._koo_spec['sampling']['interaction_probability']
        self._koo_mask[env_ids]=selected
        self._koo_goal[env_ids]=self._box_states[env_ids,2,:3]+self._push_goal_offsets[env_ids,0]
        chosen=env_ids[selected]
        if len(chosen):
            sample=self._koo_bank.sample('push',len(chosen))
            for key,value in sample.items():
                if key not in self._koo_samples:self._koo_samples[key]=torch.zeros(self.num_envs,device=self.device,dtype=value.dtype)
                self._koo_samples[key][chosen]=value
            # One payload progression per environment, shared by both people.
            self._box_states[chosen,2,:3]+=sample['box_progress'][:,None]*self._push_goal_offsets[chosen,0]

    def _reset_task(self,env_ids):
        self._tar_pos[env_ids]=self._koo_goal[env_ids,None]

    def _reset_ref_state_init(self,slot_env,slot_agent):
        selected=self._koo_mask[slot_env]
        if (~selected).any():super()._reset_ref_state_init(slot_env[~selected],slot_agent[~selected])
        if not selected.any():return
        env=slot_env[selected];agent=slot_agent[selected]
        mids=self._koo_samples['motion_id'][env].long();times=self._koo_samples['time'][env]
        lib=self._motion_lib['push']
        root,rot,dof,vel,ang,dof_vel,_=lib.get_motion_state(mids,times)
        body,body_rot,body_vel,body_ang=lib.get_motion_state_max(mids,times)
        body,body_rot,body_vel,body_ang,dof,dof_vel=lower_push_arms(body,body_rot,body_vel,body_ang,dof,dof_vel,
            lib._motions[0].skeleton_tree.parent_indices.tolist(),self._key_body_ids[:2].cpu().tolist(),
            self._dof_body_ids,self._dof_offsets,self._relation_cfg['push']['hand_target_height'])
        center,normal,tangent,half,_=push_face_geometry(self._box_states[env,2],self._koo_goal[env],
            self.cfg['env']['pushBox']['size'],self._joint_push_spec['reward'])
        center=center.clone();center[:,:2]+=tangent*((agent.to(center.dtype)*2-1)*half*.5)[:,None]
        side=self._relation_cfg['push']['hand_lateral_offset']
        targets=center[:,None].expand(-1,2,-1).clone()
        targets[:,0,:2]-=tangent*side;targets[:,1,:2]+=tangent*side
        body,body_rot,body_vel,body_ang,q=align_push_pose(body,body_rot,body_vel,body_ang,targets,
            self._key_body_ids[:2].cpu().tolist(),self._koo_spec['sampling']['stand_off'])
        self._set_env_state(env,agent,body[:,0],multiply(q,rot),dof,rotate(q,vel),rotate(q,ang),dof_vel)
        self._kinematic_humanoid_rigid_body_states[env,agent]=torch.cat((body,body_rot,body_vel,body_ang),-1)
        self._every_env_init_dof_pos[env,agent]=dof
        self._reset_ref_slots['push']=(env,agent);self._reset_ref_motion_ids['push']=mids;self._reset_ref_motion_times['push']=times

    def _reset_envs(self,env_ids):
        super()._reset_envs(env_ids)
        if not len(env_ids):return
        selected=self._koo_mask[env_ids]
        self._rsi_interaction[env_ids]=selected[:,None]
        self._rsi_late[env_ids]=False;self._rsi_phase[env_ids]=0
        chosen=env_ids[selected]
        if len(chosen):
            self._rsi_late[chosen]=self._koo_samples['late_requested'][chosen,None]
            self._rsi_phase[chosen]=self._koo_samples['phase'][chosen,None]
            assert torch.isfinite(self._humanoid_root_states[chosen]).all()
            assert torch.isfinite(self._dof_pos[chosen]).all()
            hands=self._kinematic_humanoid_rigid_body_states[chosen][:,:,self._key_body_ids[:2],:3]
            center,normal,_,_,_=push_face_geometry(self._box_states[chosen,2],self._koo_goal[chosen],
                self.cfg['env']['pushBox']['size'],self._joint_push_spec['reward'])
            depth=((hands.mean(2)-center[:,None])[...,:2]*normal[:,None]).sum(-1)
            assert torch.allclose(depth,torch.full_like(depth,-.06),atol=1e-4)
        else:depth=self._koo_goal.new_empty(0)
        report=dict(reset_envs=len(env_ids),interaction_envs=int(selected.sum()),
            late_requested=int(self._koo_samples['late_requested'][chosen].sum()) if len(chosen) else 0,
            sampling=self._koo_spec['sampling'],actual_mass=self.cfg['env']['pushBox']['mass'],
            hand_midpoint_depth_range=[float(depth.min()),float(depth.max())] if len(chosen) else None,
            source_screen='PD_12_steps',shared_geometry_rescreened=False,
            reference_phase=self._koo_samples['phase'][chosen].tolist() if len(chosen) else [])
        directory=Path(self.cfg['args'].output_path);directory.mkdir(parents=True,exist_ok=True)
        with (directory/'koo_rsi_reset_audit.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
