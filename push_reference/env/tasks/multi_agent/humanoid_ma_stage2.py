import math

from isaacgym import gymapi
import torch

from push_reference.env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
from push_reference.env.tasks.multi_agent.stage2_solo_rsi_cache import Stage2SoloRsiCache
from push_reference.learning.multi_agent.grasp_view import make_grasp_views, initial_grasp_sides
from push_reference.utils.stage2_spec import sample_graph, physical_mapping
from push_reference.utils.task_role_spec import task_graph_packet
from push_reference.utils.size_rsi import sample_sizes


class HumanoidMAStage2(HumanoidMACarry):
    _solo_cache_class = Stage2SoloRsiCache

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from push_reference.utils.stage2_rsi import JointRsiLibrary
        self._joint_rsi = JointRsiLibrary(self)
        from push_reference.utils.motion_lib import MotionLib
        path = self.cfg['env']['stage2']['amp_motion_file']
        self._stage2_amp_libs = {
            name: MotionLib(path, name, self._dof_body_ids, self._dof_offsets,
                            self._key_body_ids.cpu().numpy(), self.device)
            for name in (() if self._is_eval else ('backward', 'sideways'))}

    def _load_box_asset(self):
        # Wider simulator spacing must not enlarge the Stage1 distractor arena.
        self._box_spawn_radius = 4.5
        n, m = self.num_envs, self.num_agents
        build = self.cfg['env']['box']['build']
        preset = self._task_graph_preset if self._task_graph_preset in ('sit', 'climb', 'holding_at', 'holding_ontop') else 'random_scenario'
        solo = sample_sizes(n, self.device, preset, ranges=build['taskSizeRanges'],
                            num_agents=m, num_objects=8)
        self._box_size = torch.empty(n, 12, 3, device=self.device)
        self._box_size[:, :8] = solo
        for axis, (lo, hi) in enumerate(self.cfg['env']['stage2']['joint_size_ranges']):
            self._box_size[:, 8:10, axis] = torch.randint(round(lo*20), round(hi*20)+1,
                                                          (n, 2), device=self.device)/20
        for axis, (lo, hi) in enumerate(build['taskSizeRanges']['SUPPORT']):
            self._box_size[:, 10:, axis] = torch.randint(round(lo*20), round(hi*20)+1,
                                                        (n, 2), device=self.device)/20
        self._box_scale = self._box_size / torch.tensor(self._build_base_size, device=self.device)
        self._box_density = torch.full((n*12,), 100., device=self.device)
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
        n, m = self.num_envs, self.num_agents
        self._groups = torch.arange(m, device=self.device)[None].expand(n, -1).clone()
        self._joint = torch.zeros(n, m, device=self.device, dtype=torch.bool)
        self._grasp_sides = torch.zeros(n, m, device=self.device)
        self._joint_skills = torch.zeros(n, m, device=self.device, dtype=torch.long)
        self._joint_boxes = torch.zeros(n, m, 13, device=self.device)
        self._joint_terminal = torch.zeros(n, m, 2, device=self.device)
        self._joint_frames = torch.full((n, m), -1, device=self.device, dtype=torch.long)

    def get_obs_size(self):
        return super().get_obs_size()*self.num_agents

    def _compute_clean_scene_obs(self, env_ids=None):
        raw = super()._compute_clean_scene_obs(env_ids)
        sides = self._grasp_sides if env_ids is None else self._grasp_sides[env_ids]
        return make_grasp_views(raw, sides, self.cfg['env']['stage2']['anchor_inset'])

    def _video_focus_points(self):
        return torch.cat((self._humanoid_root_states[0, :, :3],
                          self._logical_box_values(self._box_states)[0, :, :3]), 0)

    def _sample_episode_graph(self, ids):
        from push_reference.utils.edge_ontop_spec import copy_graph_rows
        from push_reference.env.tasks.multi_agent.edge_context_reward import owner_sum
        graph, groups, joint = sample_graph(self._size_template_probabilities[ids],
                                            self._relation_graph_spec, self._task_graph_preset)
        copy_graph_rows(self.relation_runtime.graph, ids, graph)
        self._groups[ids], self._joint[ids] = groups, joint
        self._logical_box_order[ids] = physical_mapping(groups, joint)
        self._agent_box_assignment[ids] = self._logical_box_order[ids].gather(1, groups)
        self._edge_goal_owners[ids] = owner_sum((graph.required_goal & graph.edge_valid).float(), graph)>0
        self._ontop_maintain_steps[ids] = 0
        self._sampling_counts[0] += len(ids)
        packet = task_graph_packet(graph, len(ids)).reshape(-1, 4, 5).long()
        self._sampling_counts[1:6] += torch.bincount((packet[..., 1]+1).flatten(), minlength=5)
        self._sampling_counts[6] += joint.any(-1).sum()
        for offset, relation in enumerate((6, 7, 8, 9, 10), 7):
            self._sampling_counts[offset] += ((graph.edge_relation == relation) & graph.edge_valid).sum()
        self._update_box_assignment_colors(ids)

    def _sample_box_assignments(self, ids):
        return

    def _reset_actors(self, ids):
        self._stage2_reset_ids = ids
        return super()._reset_actors(ids)

    def _reset_ref_state_init(self, envs, humans):
        packet = task_graph_packet(self.relation_runtime.graph, self.num_envs).reshape(-1, 4, 5)
        if (self._is_eval and 'putDown' in self._skill
                and self._skill_init_prob[self._skill.index('putDown')] > 0
                and (packet[envs, humans, 1] == 3).any()):
            raise ValueError('ON_TOP evaluation cannot request putDown RSI')
        solo = ~self._joint[envs, humans]
        if solo.any():
            super()._reset_ref_state_init(envs[solo], humans[solo])
        selected = envs[self._joint[envs, humans] & (self._groups[envs, humans] == humans)]
        leaders = humans[self._joint[envs, humans] & (self._groups[envs, humans] == humans)]
        if not len(selected):
            return
        if self._is_eval:
            weights = self._skill_init_prob.expand(len(selected), -1)
        else:
            templates = torch.where(packet[selected, leaders, 1] == 3, 4, 3)
            weights = self._template_rsi_weights[templates]
        skills = torch.multinomial(weights, 1).flatten()
        for index, name in enumerate(self._skill):
            mask = skills == index
            es, first = selected[mask], leaders[mask]
            if not len(es):
                continue
            second = ((self._groups[es] == first[:, None]) &
                      (torch.arange(4, device=self.device)[None] != first[:, None])).long().argmax(-1)
            people = torch.stack((first, second), -1)
            payload = self._agent_box_assignment[es, first]
            if name == 'loco':
                lib = self._motion_lib[name]
                clips = lib.sample_motions(2*len(es))
                times = lib.sample_time_rsi(clips)
                p, q, dof, v, w, dv, _ = lib.get_motion_state(clips, times)
                root = torch.cat((p, q, v, w), -1).reshape(-1, 2, 13)
                body = torch.cat(lib.get_motion_state_max(clips, times), -1).reshape(len(es), 2, self.num_bodies, 13)
                dof, dv = dof.reshape(len(es), 2, -1), dv.reshape(len(es), 2, -1)
                box = torch.zeros(len(es), 13, device=self.device)
                box[:, 6] = 1.
                box[:, 2] = self._box_size[es, payload, 2]/2
                terminal = torch.zeros(len(es), 2, device=self.device)
                frame = torch.full((len(es),), -1, device=self.device, dtype=torch.long)
            else:
                if name not in ('pickUp', 'carryWith', 'putDown'):
                    raise ValueError('Joint RSI supports loco/pickUp/carryWith/putDown')
                data = self._joint_rsi.sample(name, self._box_size[es, payload])
                root, dof, dv, body, box, terminal, frame = [data[key] for key in
                    ('root_state', 'dof_position', 'dof_velocity', 'body_state',
                     'box_state', 'terminal_box_xy', 'source_frame')]
            order = torch.rand(len(es), 2, device=self.device).argsort(-1)
            row = torch.arange(len(es), device=self.device)[:, None]
            root, dof, dv, body = [value[row, order] for value in (root, dof, dv, body)]
            center = self._env_origins[es] + self._agent_spawn_offsets[first]
            shift = center[:, None]-box[:, None, :3]*box.new_tensor([1., 1., 0.])
            root[..., :3] += shift
            body[..., :3] += shift[:, :, None]
            terminal += shift[:, 0, :2]
            box[:, :2] = center[:, :2]
            self._humanoid_root_states[es[:, None], people] = root
            self._dof_pos[es[:, None], people] = dof
            self._dof_vel[es[:, None], people] = dv
            self._kinematic_humanoid_rigid_body_states[es[:, None], people] = body
            self._every_env_init_dof_pos[es[:, None], people] = dof
            self._joint_skills[es[:, None], people] = index
            self._joint_boxes[es, first] = box
            self._joint_terminal[es, first] = terminal
            self._joint_frames[es[:, None], people] = frame[:, None]

    def _collect_random_slots(self, skills):
        slots = super()._collect_random_slots(skills)
        requested = torch.tensor([self._skill.index(name) for name in skills if name in self._skill], device=self.device)
        selected = (self._joint & (self._groups == torch.arange(4, device=self.device)[None]) &
                    (self._joint_skills[..., None] == requested).any(-1))
        included = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        included[self._stage2_reset_ids] = True
        selected &= included[:, None]
        es, people = selected.nonzero(as_tuple=True)
        if not len(es):
            return slots
        if slots is None:
            return es, people
        return torch.cat((slots[0], es)), torch.cat((slots[1], people))

    def _reset_boxes(self, ids):
        # Reference-slot metadata contains only solo actors; shared objects are restored once.
        super()._reset_boxes(ids)
        es, people = (self._joint[ids] & (self._groups[ids] == torch.arange(4, device=self.device)[None])).nonzero(as_tuple=True)
        es = ids[es]
        payload = self._agent_box_assignment[es, people]
        advanced = self._joint_skills[es, people] != (self._skill.index('loco') if 'loco' in self._skill else -1)
        self._box_states[es[advanced], payload[advanced]] = self._joint_boxes[es[advanced], people[advanced]]
        for env, leader in zip(es[~advanced].tolist(), people[~advanced].tolist()):
            members = (self._groups[env] == leader).nonzero().flatten()
            box = self._box_states[env, self._agent_box_assignment[env, leader], :2]
            angle = torch.rand((), device=self.device)*2*math.pi
            angles = angle + torch.arange(2, device=self.device)*math.pi
            radius = 1.+9.*torch.rand(2, device=self.device)
            desired = box + radius[:, None]*torch.stack((angles.cos(), angles.sin()), -1)
            delta = desired-self._humanoid_root_states[env, members, :2]
            self._humanoid_root_states[env, members, :2] += delta
            self._kinematic_humanoid_rigid_body_states[env, members, :, :2] += delta[:, None]
        active = torch.zeros(len(ids), 12, device=self.device, dtype=torch.bool)
        active.scatter_(1, self._logical_box_order[ids], True)
        row, physical = (~active).nonzero(as_tuple=True)
        env = ids[row]
        self._box_states[env, physical, :2] = self._env_origins[env, :2]+1000.
        self._box_states[env, physical, 0] += physical*2.

    def _reset_task(self, ids):
        super()._reset_task(ids)
        if "putDown" not in self._skill:
            return
        es, leaders = (self._joint[ids] & (self._groups[ids] == torch.arange(4, device=self.device)[None]) &
                       (self._joint_skills[ids] == self._skill.index('putDown'))).nonzero(as_tuple=True)
        es = ids[es]
        self._tar_pos[es, leaders, :2] = self._joint_terminal[es, leaders]
        size = self._box_size[es, self._agent_box_assignment[es, leaders]]
        self._tar_pos[es, leaders, 2] = size[:, 2]/2

    def _ontop_scene_infeasible(self, ids):
        from push_reference.env.tasks.multi_agent.stage2_scene import scene_infeasible
        return scene_infeasible(self, ids)

    def _reset_relation_history(self, ids):
        self._stage2_reset_ids = ids
        raw = super()._compute_clean_scene_obs(ids)
        self._grasp_sides[ids] = initial_grasp_sides(raw, 4, 8)
        return super()._reset_relation_history(ids)

    def _evaluate_ontop_context(self, env_ids=None):
        phi, diag = super()._evaluate_ontop_context(env_ids)
        from push_reference.env.tasks.multi_agent.stage2_scene import joint_holding_geometry
        return joint_holding_geometry(self, phi, diag, env_ids)

    def _init_amp_obs(self, ids):
        super()._init_amp_obs(ids)
        row, human = self._joint[ids].nonzero(as_tuple=True)
        self._hist_amp_obs_buf[ids[row], human] = self._curr_amp_obs_buf[ids[row], human, None]

    def _compute_agent_collision_violation(self, roots, min_dist):
        from push_reference.env.tasks.multi_agent.stage2_scene import cpa_collision_penalty
        same = (self._groups[:, :, None] == self._groups[:, None, :]) & self._joint[:, :, None]
        payload = self._agent_box_assignment
        row = torch.arange(self.num_envs, device=self.device)[:, None]
        box = self._box_states[row, payload]
        radius = self._box_size[row, payload, :2].norm(dim=-1)/2 + 1.
        near = (roots[..., :2]-box[..., :2]).norm(dim=-1) <= radius
        released = same & near[:, :, None] & near[:, None, :]
        return cpa_collision_penalty(roots, self._humanoid_root_states[..., 7:10], min_dist,
                                     self._agent_collision_config["ttc_discount"], self.dt, released)

    def _fetch_conditioned_amp_demo(self, num_samples):
        from push_reference.utils.stage2_amp import sample_demo
        return sample_demo(self, num_samples)
