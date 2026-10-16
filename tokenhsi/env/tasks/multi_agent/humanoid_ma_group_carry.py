from isaacgym import gymapi
import torch

from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
from utils.group_carry_spec import GROUP_SPEC, physical_mapping, entity_validity
from utils.task_role_spec import task_graph_packet


class HumanoidMAGroupCarry(HumanoidMACarry):
    def _validate_joint_carry_env(self, env):
        super()._validate_joint_carry_env(env)
        if env['relationGraph'] != GROUP_SPEC or env.get('numPhysicalObjects') != 12:
            raise ValueError('Four-agent carry requires 12 physical objects and eight logical slots')

    def _load_box_asset(self):
        from utils.size_rsi import sample_sizes
        n = self.num_envs
        solo = sample_sizes(n*2, self.device, 'holding_at').reshape(n, 2, 4, 3)
        self._box_size = torch.empty(n, 12, 3, device=self.device)
        self._box_size[:, :4] = solo[:, :, :2].reshape(n, 4, 3)
        self._box_size[:, 4:8] = solo[:, :, 2:].reshape(n, 4, 3)
        self._box_size[:, 8:10] = torch.tensor(self._build_base_size, device=self.device)
        self._box_size[:, 10:] = torch.tensor(self.cfg['env']['box']['build']['jointSupportSize'], device=self.device)
        self._box_scale = self._box_size/torch.tensor(self._build_base_size, device=self.device)
        self._box_density = torch.full((n*12,), 100., device=self.device)
        self._joint_env_mask = torch.zeros(n, device=self.device, dtype=torch.bool)
        self._box_assets = []
        for size in self._box_size.reshape(-1, 3).cpu().tolist():
            options = gymapi.AssetOptions()
            options.angular_damping = options.linear_damping = .01
            options.max_angular_velocity = 100.
            options.density = 100.
            options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            self._box_assets.append(self.gym.create_box(self.sim, *size, options))

    def _build_assignment_tensors(self):
        super()._build_assignment_tensors()
        self._group_leader = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.long)
        self._group_joint = torch.zeros_like(self._group_leader, dtype=torch.bool)
        self._group_skill = torch.zeros_like(self._group_leader)
        self._group_frame = torch.full_like(self._group_leader, -1)
        self._group_source_role = torch.full_like(self._group_leader, -1)
        self._group_reset_counts = torch.zeros(3, device=self.device)
        self._group_initial_holding = torch.zeros_like(self._group_joint)
        self._group_metrics = torch.zeros(2, 2, 8, device=self.device)
        self._group_rsi_counts = torch.zeros(2, 2, len(self._skill), device=self.device)

    def _sample_episode_graph(self, ids):
        from utils.group_carry_spec import sample_graph
        from utils.edge_ontop_spec import copy_graph_rows
        from env.tasks.multi_agent.edge_context_reward import owner_sum
        new = sample_graph(len(ids), self._relation_graph_spec, self.device, self._task_graph_preset)
        copy_graph_rows(self.relation_runtime.graph, ids, new)
        mapping, leaders, joint = physical_mapping(task_graph_packet(new, len(ids)))
        self._logical_box_order[ids] = mapping
        self._agent_box_assignment[ids] = mapping.gather(1, leaders)
        self._group_leader[ids], self._group_joint[ids] = leaders, joint
        self._edge_goal_owners[ids] = owner_sum((new.required_goal & new.edge_valid).float(), new)>0
        self._ontop_maintain_steps[ids] = 0
        self._sampling_counts[0] += len(ids)
        self._group_reset_counts += torch.bincount(joint.sum(-1)//2, minlength=3)
        self._update_box_assignment_colors(ids)

    def _reset_envs(self, ids):
        from env.tasks.multi_agent.group_carry_reset import reset_groups
        if len(ids):
            reset_groups(self, ids)

    def _compute_clean_scene_obs(self, env_ids=None, grasp_view=True):
        raw = super()._compute_clean_scene_obs(env_ids, grasp_view=False)
        valid = entity_validity(raw[:, -20:], 4, 8)
        m, o = self.num_agents, self.num_objects
        start = m*self.get_clean_humanoid_obs_size()
        end = start+o*self.get_clean_object_obs_size()
        pose_start = end+m
        pose_end = pose_start+(2*m+o)*7
        objects = raw[:, start:end].reshape(-1, o, 30)
        objects.masked_fill_(~valid[:, 4:12, None], 0.)
        raw[:, end:pose_start].masked_fill_(~valid[:, 12:], 0.)
        poses = raw[:, pose_start:pose_end].reshape(-1, 2*m+o, 7)
        poses.masked_fill_(~valid[..., None], 0.)
        poses[..., 6] = torch.where(valid, poses[..., 6], 1.)
        if self._grasp_view and grasp_view:
            from learning.multi_agent.grasp_view import make_grasp_views
            sides = self._grasp_sides if env_ids is None else self._grasp_sides[env_ids]
            return make_grasp_views(raw, sides, self._relation_cfg['joint_carry']['anchor_inset'])
        return raw

    def _reset_relation_history(self, ids):
        super()._reset_relation_history(ids)
        from env.tasks.multi_agent.edge_context_reward import owner_sum
        from utils.edge_ontop_spec import select_graph
        graph = select_graph(self.relation_runtime.graph, ids)
        holding = owner_sum((self.relation_runtime.own_success[ids] &
            (graph.edge_relation == 6)).float(), graph).bool()
        self._group_initial_holding[ids] = torch.where(self._group_joint[ids],
            self.relation_runtime.coupled_holding[ids], holding)
        leaders = self._group_leader[ids] == torch.arange(4, device=self.device)[None]
        tasks = task_graph_packet(graph, len(ids)).reshape(-1, 4, 5)
        for joint in (0, 1):
            for task in (0, 1):
                chosen = leaders & (self._group_joint[ids] == joint) & (tasks[..., 1] == task+2)
                self._group_rsi_counts[joint, task] += torch.bincount(
                    self._group_skill[ids][chosen], minlength=self._group_rsi_counts.shape[-1])

    def _record_ontop_diagnostics(self, result, diag):
        super()._record_ontop_diagnostics(result, diag)
        from env.tasks.multi_agent.edge_context_reward import owner_sum
        graph = self.relation_runtime.graph
        holding = owner_sum((result['own_success'] & (graph.edge_relation == 6)).float(), graph).bool()
        gate = torch.where(self._group_joint, self.relation_runtime.coupled_holding, holding)
        leaders = self._group_leader == torch.arange(4, device=self.device)[None]
        tasks = task_graph_packet(graph, self.num_envs).reshape(-1, 4, 5)
        loco = self._group_skill == self._skill.index('loco') if 'loco' in self._skill else torch.zeros_like(gate)
        for joint in (0, 1):
            for task in (0, 1):
                humans = (self._group_joint == joint) & (tasks[..., 1] == task+2)
                chosen = humans & leaders
                self._group_metrics[joint, task] += torch.stack((chosen.sum(),
                    (gate & chosen).sum(), (self.relation_runtime.done & chosen).sum(),
                    (self._group_initial_holding & chosen).sum(), (loco & chosen).sum(),
                    (loco & gate & chosen).sum(), (holding & humans).sum(), humans.sum()))

    def _consume_sampling_diagnostics(self):
        resets = self._sampling_counts[0].clone()
        out = {'sampling/resets': resets,
            'sampling/physical_retries': self._sampling_retries.clone(),
            'sampling/physical_failures': self._sampling_failures.clone()}
        for count in range(3):
            out['sampling/pairs_'+str(count)] = self._group_reset_counts[count]/resets.clamp_min(1)
        leaders = self._group_leader == torch.arange(4, device=self.device)[None]
        pair = self._group_joint & leaders
        out['joint/coupled_holding'] = (self.relation_runtime.coupled_holding & pair).sum()/pair.sum().clamp_min(1)
        out['sampling/joint_human_fraction'] = self._group_joint.float().mean()
        for joint, label in ((1, 'joint'), (0, 'solo')):
            for task, name in ((0, 'at'), (1, 'ontop')):
                counts = self._group_metrics[joint, task]
                prefix = 'groups/'+label+'_'+name+'/'
                for index, metric in ((1, 'holding_gate'), (2, 'success'), (3, 'initial_rsi_holding')):
                    out[prefix+metric] = counts[index]/counts[0].clamp_min(1)
                out[prefix+'loco_pickup'] = counts[5]/counts[4].clamp_min(1)
                out[prefix+'human_holding'] = counts[6]/counts[7].clamp_min(1)
                out[prefix+'group_transitions'] = counts[0].clone()
                out[prefix+'human_transitions'] = counts[7].clone()
                rsi = self._group_rsi_counts[joint, task]
                for index, skill in enumerate(self._skill):
                    out[prefix+'reset_'+skill] = rsi[index]/rsi.sum().clamp_min(1)
        self._group_metrics.zero_()
        self._group_rsi_counts.zero_()
        for index, name in enumerate(self._skill):
            out['sampling/rsi_'+name] = self._stage2_rsi_counts[index]/(resets*4).clamp_min(1)
        out['sampling/platform_fraction'] = self._joint_loco_platform.active.float().mean()
        self._sampling_counts.zero_()
        self._group_reset_counts.zero_()
        self._sampling_retries.zero_()
        self._sampling_failures.zero_()
        self._stage2_rsi_counts.zero_()
        return out

    def _video_focus_points(self):
        packet = task_graph_packet(self.relation_runtime.graph, self.num_envs)[:1]
        valid = entity_validity(packet, 4, 8)[0, 4:12]
        return torch.cat((self._humanoid_root_states[0, :, :3], self._logical_box_values(self._box_states)[0, valid, :3]))

    def _update_camera(self):
        if self.viewer is None:
            return
        points = self._video_focus_points()
        low, high = points.amin(0), points.amax(0)
        center = (low+high)/2
        center[2] = max(float(center[2]), .8)
        radius = max(float((high-low)[:2].norm())*.8, 3.)
        if self._camera_follow_flag:
            self.gym.viewer_camera_look_at(self.viewer, None,
                gymapi.Vec3(float(center[0])+radius*.65, float(center[1])-radius, 2.8+radius*.2),
                gymapi.Vec3(*center.tolist()))
        self._cam_prev_char_pos[:] = self._humanoid_root_states[0, 0, :3].cpu().numpy()
