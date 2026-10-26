import torch

from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
from env.tasks.multi_agent.humanoid_ma_stage2_jointpush import HumanoidMAStage2JointPush
from env.tasks.multi_agent.edge_ontop_task import SampledOnTopTaskMixin
from env.tasks.multi_agent.jointpush_solo_rsi_cache import LazyJointPushSoloCache
from utils import stage2_jointpush_scene as scene
from utils.stage2_jointpush_rsi import sample_push_initialization, hand_targets, coupled_body_overlap
from utils.jointpush_geometry import rotate, multiply
from utils.jointpush_rsi_geometry import align_push_pose, safe_place


class HumanoidMAStage2JointPushRsi(HumanoidMAStage2JointPush):
    _solo_cache_class = LazyJointPushSoloCache

    def _init_relation_runtime(self):
        super()._init_relation_runtime()
        from utils.jointpush_rsi import PushBank
        self._bank = None if self._is_eval else PushBank(self.cfg['env'],self._motion_lib)
        self._interaction = torch.zeros_like(self._push_hits)
        self._layout_failed = torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)

    def _sample_episode_graph(self, ids):
        super()._sample_episode_graph(ids)
        self._interaction[ids] = False if self._is_eval else sample_push_initialization(
            self._groups[ids],self._templates_by_owner[ids],self._joint['rsi']['interaction_probability'])

    def _sample_box_assignments(self, ids):
        return

    def _reset_ref_state_init(self, env, owner):
        if self._is_eval:
            return super()._reset_ref_state_init(env,owner)
        push = self._templates_by_owner[env,owner] == 5
        if (~push).any():
            HumanoidMACarry._reset_ref_state_init(self,env[~push],owner[~push])
        es, humans = env[push],owner[push]
        if not len(es):
            return
        super()._reset_ref_state_init(es,humans)
        interaction = self._interaction[es,humans]
        if interaction.any():
            selected_env,selected_owner = es[interaction],humans[interaction]
            loco_env,loco_owner = self._reset_ref_slots['loco']
            keep = ~self._interaction[loco_env,loco_owner]
            self._reset_ref_slots['loco'] = (loco_env[keep],loco_owner[keep])
            self._reset_ref_motion_ids['loco'] = self._reset_ref_motion_ids['loco'][keep]
            self._reset_ref_motion_times['loco'] = self._reset_ref_motion_times['loco'][keep]
            mids,times = self._bank.sample(len(selected_env))
            self._record_reference('push',selected_env,selected_owner,mids,times)

    def _reset_ontop_context_envs(self, ids):
        if self._is_eval:
            return super()._reset_ontop_context_envs(ids)
        return SampledOnTopTaskMixin._reset_ontop_context_envs(self,ids)

    def _configure_ontop_targets(self, ids):
        super()._configure_ontop_targets(ids)
        if self._is_eval:
            return
        self._layout_failed[ids] = scene.sample_push(self,ids)
        self._apply_push_rsi()
        scene.place_distractors(self,ids)

    def _apply_push_rsi(self):
        slots = self._reset_ref_slots.get('push')
        if slots is None:
            return
        es,owners = slots
        mids,times = self._reset_ref_motion_ids['push'],self._reset_ref_motion_times['push']
        lib = self._motion_lib['push']
        _,q,dof,vel,ang,dof_vel,_ = lib.get_motion_state(mids,times)
        body,bq,bv,ba = lib.get_motion_state_max(mids,times)
        physical = self._agent_box_assignment[es,owners]
        boxes,sizes = self._box_states[es,physical],self._box_size[es,physical]
        targets = hand_targets(boxes,sizes,self._tar_pos[es,owners],self._members[es,owners],
            self._regions[es,owners],self._joint['contact'])
        body,bq,bv,ba,turn = align_push_pose(body,bq,bv,ba,targets,
            self._key_body_ids[:2].cpu().tolist(),self._joint['rsi']['stand_off'])
        body = safe_place(body,bq,boxes,sizes,targets,self._bank.geometry)
        self._set_env_state(es,owners,body[:,0],multiply(turn,q),dof,rotate(turn,vel),rotate(turn,ang),dof_vel)
        self._kinematic_humanoid_rigid_body_states[es,owners] = torch.cat((body,bq,bv,ba),-1)
        self._every_env_init_dof_pos[es,owners] = dof

    def _ontop_scene_infeasible(self, ids):
        if self._is_eval:
            return super()._ontop_scene_infeasible(ids)
        bad = self._layout_failed[ids] | scene.scene_infeasible(self,ids,rsi=True)
        bad |= coupled_body_overlap(self._kinematic_humanoid_rigid_body_states[ids],
            self._bank.geometry,self._groups[ids],self._interaction[ids])
        for skill in ('sit','climb'):
            slots = self._reset_ref_slots.get(skill)
            if slots is None:
                continue
            es,owner = slots
            rows = torch.searchsorted(ids,es)
            physical = self._agent_box_assignment[es,owner]
            body = self._kinematic_humanoid_rigid_body_states[es,owner,...,:3]
            from utils.stage1_v2 import pose_geometry_valid
            feet = body[:,self._key_body_ids[-2:]]
            invalid = ~pose_geometry_valid(body,feet,self._box_states[es,physical,None],self._box_size[es,physical,None])
            invalid |= feet[...,2].amin(-1) < -.02
            bad[rows[invalid]] = True
        return bad

    def _init_amp_obs_ref(self, slots, motion_ids, motion_times, skill_name):
        if skill_name != 'loco':
            return HumanoidMACarry._init_amp_obs_ref(self,slots,motion_ids,motion_times,skill_name)
        return super()._init_amp_obs_ref(slots,motion_ids,motion_times,skill_name)
