import torch

from env.tasks.multi_agent import bridge_carry_boundary_rsi as carry
from env.tasks.multi_agent.bridge_carry_rsi_cache import CarryRsiCache
from env.tasks.multi_agent.humanoid_ma_sensor_bridge_before import HumanoidMASensorBridgeBefore
from env.tasks.multi_agent.reference_motion import ReferenceMotionMixin
from env.tasks.multi_agent.stage2_layout import invalid_before_layout
from utils.bridge_before_carry_rsi import KINDS, validate


class HumanoidMASensorBridgeBeforeCarryRsi(HumanoidMASensorBridgeBefore):
    """Training episodes may start the AT crosser on a present bridge with its box already in hand."""

    def __init__(self, cfg, *args, **kwargs):
        self._carry_spec = validate(cfg['env'])
        super().__init__(cfg, *args, **kwargs)
        if not cfg['args'].eval:
            self._carry_cache = CarryRsiCache(self)

    def _init_bridge_curriculum(self):
        super()._init_bridge_curriculum()
        self._carry_cache = None
        self._carry_request = torch.zeros_like(self._compact_kind)
        self._carry_clip = torch.zeros_like(self._compact_kind)
        self._carry_time = torch.zeros_like(self._compact_kind, dtype=torch.float)
        self._carry_done = torch.zeros_like(self._compact_kind, dtype=torch.bool)
        self._carry_dof = torch.zeros_like(self._dof_pos)
        self._carry_dof_vel = torch.zeros_like(self._dof_vel)
        self._carry_counts = torch.zeros(len(KINDS), 3, device=self.device)
        self._carry_episodes = torch.zeros(2, device=self.device)
        self._carry_failures = torch.zeros(3, device=self.device)

    def _draw_episode_graph(self, ids):
        result = super()._draw_episode_graph(ids)
        self._carry_request[ids] = 0
        if self.cfg['args'].eval:
            return result
        rsi = ids[torch.rand(len(ids), device=self.device) < self._carry_spec['episode_probability']]
        if len(rsi):
            # The box can only be carried across a bridge that is already in place.
            self._bridge_start_mode[rsi] = 0
            self._bridge_near[rsi] = False
            self._bridge_aligned[rsi] = True
            probabilities = torch.tensor(self._carry_spec['boundary_probabilities'], device=self.device)
            kinds = torch.multinomial(probabilities, len(rsi), replacement=True)+1
            crosser = self._roles[rsi, 1]
            self._carry_request[rsi, crosser] = kinds
            self._compact_requested[rsi, crosser] = kinds
        return result

    def _place_scene(self, ids):
        request = self._carry_request[ids]
        self._compact_kind[ids] = torch.where(request > 0, 0, self._compact_kind[ids])
        self._carry_done[ids] = False
        super()._place_scene(ids)
        rows, people = ((request > 0) & ~self._place_failed[ids, None]).nonzero(as_tuple=True)
        if len(rows):
            self._place_carry(ids[rows], people)

    def _place_carry(self, env, people):
        if self._carry_cache is None:
            self._carry_cache = CarryRsiCache(self)
        kinds = self._carry_request[env, people]
        pending = torch.ones(len(env), device=self.device, dtype=torch.bool)
        for _ in range(self._scene_cfg['bridge']['boundary_rsi']['attempts']):
            picked = pending.nonzero().flatten()
            es, owners = env[picked], people[picked]
            slots = self._agent_box_assignment[es, owners]
            mids, frames, available = self._carry_cache.sample(self._box_size[es, slots])
            self._carry_failures[0] += (~available).sum()
            pending[picked[~available]] = False
            picked, es, owners, slots = picked[available], es[available], owners[available], slots[available]
            if len(picked):
                mids = mids[available]
                times = frames[available]*self._motion_lib['carryWith']._motion_dt[mids]
                human, body, box, dof, dof_vel, progress = carry.candidate(self, es, owners, mids, times, kinds[picked])
                good = carry.valid(self, es, owners, human, body, box).nonzero().flatten()
                if not len(good):
                    continue
                e, p, s = es[good], owners[good], slots[good]
                old_root, old_box = self._humanoid_root_states[e, p].clone(), self._box_states[e, s].clone()
                self._humanoid_root_states[e, p] = human[good]
                self._box_states[e, s] = box[good]
                # Moving the leader closer must not break the BEFORE pair layout, as in the loco boundary RSI.
                late = invalid_before_layout(self, e)
                self._humanoid_root_states[e[late], p[late]] = old_root[late]
                self._box_states[e[late], s[late]] = old_box[late]
                self._carry_failures[2] += late.sum()
                good = good[~late]
                e, p = es[good], owners[good]
                self._kinematic_humanoid_rigid_body_states[e, p] = body[good]
                self._carry_dof[e, p], self._carry_dof_vel[e, p] = dof[good], dof_vel[good]
                self._carry_clip[e, p], self._carry_time[e, p] = mids[good], times[good]
                self._carry_done[e, p] = True
                kind = kinds[picked[good]]
                self._compact_kind[e, p] = kind
                self._cross_phase[e, p] = kind-1
                self._cross_progress[e, p] = progress[good]
                pending[picked[good]] = False
            if not pending.any():
                break
        self._carry_failures[1] += pending.sum()

    def _reset_env_tensors(self, ids):
        rows, people = self._carry_done[ids].nonzero(as_tuple=True)
        env = ids[rows]
        self._dof_pos[env, people] = self._carry_dof[env, people]
        self._dof_vel[env, people] = self._carry_dof_vel[env, people]
        self._every_env_init_dof_pos[env, people] = self._carry_dof[env, people]
        request, done = self._carry_request[ids], self._carry_done[ids]
        for i in range(len(KINDS)):
            asked = request == i+1
            self._carry_counts[i] += torch.stack((asked.sum(), (asked & done).sum(), (asked & ~done).sum()))
        episode = (request > 0).any(-1)
        self._carry_episodes += torch.stack((episode.sum(), (~episode).sum()))
        super()._reset_env_tensors(ids)

    def _init_amp_obs_ref(self, slots, motion_ids, motion_times, skill_name):
        # CoupledPush maps every reference to loco/push_loco; carryWith history must use its own clips.
        if skill_name == 'carryWith':
            return ReferenceMotionMixin._init_amp_obs_ref(self, slots, motion_ids, motion_times, skill_name)
        return super()._init_amp_obs_ref(slots, motion_ids, motion_times, skill_name)

    def _init_amp_obs(self, ids):
        slots = self._reset_ref_slots.get('loco')
        if slots is not None:
            keep = ~self._carry_done[slots]
            mids, times = self._reset_ref_motion_ids['loco'], self._reset_ref_motion_times['loco']
            self._reset_ref_slots['loco'] = tuple(x[keep] for x in slots)
            self._reset_ref_motion_ids['loco'], self._reset_ref_motion_times['loco'] = mids[keep], times[keep]
        super()._init_amp_obs(ids)
        if slots is not None:
            self._reset_ref_slots['loco'] = slots
            self._reset_ref_motion_ids['loco'], self._reset_ref_motion_times['loco'] = mids, times
        rows, people = self._carry_done[ids].nonzero(as_tuple=True)
        env = ids[rows]
        if len(env):
            self._init_amp_obs_ref((env, people), self._carry_clip[env, people], self._carry_time[env, people], 'carryWith')

    def _consume_sampling_diagnostics(self):
        result = super()._consume_sampling_diagnostics()
        for i, name in enumerate(KINDS):
            for j, label in enumerate(('requested', 'executed', 'fallback')):
                result['carry_rsi/'+name+'/'+label] = self._carry_counts[i, j].clone()
        for i, name in enumerate(('rsi', 'ordinary')):
            result['carry_rsi/episodes/'+name] = self._carry_episodes[i].clone()
        for i, name in enumerate(('fallback_cache', 'fallback_placement', 'before_layout_rejects')):
            result['carry_rsi/'+name] = self._carry_failures[i].clone()
        self._carry_counts.zero_()
        self._carry_episodes.zero_()
        self._carry_failures.zero_()
        return result
