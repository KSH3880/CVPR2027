"""SceneResetMixin: shared fixed Stage1/BEFORE environment behavior."""
import numpy as np
import torch
from isaacgym import gymapi
from isaacgym import gymtorch
from env.tasks.multi_agent.scene_features import scenario_neutral_targets
from isaacgym.torch_utils import to_torch, quat_from_angle_axis
from utils.edge_context_spec import at_goal_marker_slots
from utils.edge_scenario_spec import classify_templates, agent_object_indices, agent_goal_indices, scenario_templates

class SceneResetMixin:
    def _reset_envs(self, env_ids):
        if self._fixed_platforms and (not self._platform_warmup_done) and len(env_ids):
            if len(env_ids) != self.num_envs:
                raise ValueError("Fixed platforms require a full first reset")
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self._refresh_sim_tensors()
            self._platform_warmup_done = True
            print(
                "[stage1] Initial platform physics step complete", flush=True
            )
        return self._reset_ontop_context_envs(env_ids)

    def _reset_actors(self, env_ids):
        slot_env, slot_agent = self._expand_slots(env_ids)
        if self._state_init == self.StateInit.Default:
            self._reset_default(slot_env, slot_agent)
            self._reset_default_slots = (slot_env, slot_agent)
        elif (
            self._state_init == self.StateInit.Start
            or self._state_init == self.StateInit.Random
        ):
            self._reset_ref_state_init(slot_env, slot_agent)
        elif self._state_init == self.StateInit.Hybrid:
            probs = to_torch(
                np.array([self._hybrid_init_prob] * slot_env.shape[0]),
                device=self.device,
            )
            ref_mask = torch.bernoulli(probs) == 1.0
            if ref_mask.sum() > 0:
                self._reset_ref_state_init(slot_env[ref_mask], slot_agent[ref_mask])
            rest = torch.logical_not(ref_mask)
            if rest.sum() > 0:
                self._reset_default(slot_env[rest], slot_agent[rest])
                self._reset_default_slots = (slot_env[rest], slot_agent[rest])
        else:
            assert False, "Unsupported state initialization strategy: {:s}".format(
                str(self._state_init)
            )
        return

    def _reset_default(self, slot_env, slot_agent):
        super()._reset_default(slot_env, slot_agent)
        self._kinematic_humanoid_rigid_body_states[slot_env, slot_agent] = (
            self._initial_humanoid_rigid_body_states[slot_env, slot_agent]
        )
        self._every_env_init_dof_pos[slot_env, slot_agent] = self._initial_dof_pos[
            slot_env, slot_agent
        ]
        return

    def _scenario_physical_object(self, slot_env, slot_agent):
        logical = agent_object_indices(
            self.relation_runtime.graph, slot_env, slot_agent
        )
        return self._logical_box_order[slot_env, logical]

    def _scenario_goal(self, slot_env, slot_agent):
        return agent_goal_indices(self.relation_runtime.graph, slot_env, slot_agent)

    def _reset_ref_state_init(self, slot_env, slot_agent):
        num_slots = slot_env.shape[0]
        if self._template_rsi is not None and (not self._is_eval):
            graph = self.relation_runtime.graph
            template = classify_templates(graph, slot_env, slot_agent, True)
            weights = self._template_rsi_weights[template]
            sk_ids = torch.multinomial(weights, 1).flatten()
            logical = agent_object_indices(graph, slot_env, slot_agent)
            advanced = torch.tensor(
                [name != "loco" for name in self._skill], device=self.device
            )[sk_ids]
            for _ in range(16):
                conflict = (
                    (slot_env[:, None] == slot_env[None, :])
                    & (logical[:, None] == logical[None, :])
                    & advanced[:, None]
                    & advanced[None, :]
                    & ~torch.eye(num_slots, dtype=torch.bool, device=self.device)
                )
                retry = conflict.any(-1)
                if not retry.any():
                    break
                sk_ids[retry] = torch.multinomial(weights[retry], 1).flatten()
                advanced = torch.tensor(
                    [name != "loco" for name in self._skill], device=self.device
                )[sk_ids]
            if retry.any():
                raise RuntimeError(
                    "No valid template RSI combination after 16 attempts"
                )
            from utils.size_rsi import resolve_skills

            requested = sk_ids.clone()
            sk_ids, fallback = resolve_skills(
                sk_ids,
                weights,
                self._size_rsi_cache.availability(slot_env, slot_agent, template),
                self._skill.index("loco"),
            )
            self._size_rsi_requested.index_put_(
                (template, requested),
                torch.ones_like(template, dtype=torch.float),
                accumulate=True,
            )
            self._size_rsi_fallbacks.index_add_(0, template, fallback.float())
        else:
            sk_ids = torch.multinomial(
                self._skill_init_prob, num_samples=num_slots, replacement=True
            )
        for uid, sk_name in enumerate(self._skill):
            curr_motion_lib = self._motion_lib[sk_name]
            sel = (sk_ids == uid).nonzero().squeeze(-1)
            if len(sel) == 0:
                continue
            curr_env = slot_env[sel]
            curr_agent = slot_agent[sel]
            offset = self._agent_spawn_offsets[curr_agent] + self._env_origins[curr_env]
            if not self._is_eval:
                motion_ids, motion_times = self._size_rsi_cache.sample(
                    sk_name, curr_env, curr_agent, template[sel]
                )
            else:
                motion_ids = curr_motion_lib.sample_motions(len(sel))
                if (
                    self._state_init == self.StateInit.Random
                    or self._state_init == self.StateInit.Hybrid
                ):
                    motion_times = curr_motion_lib.sample_time_rsi(motion_ids)
                elif self._state_init == self.StateInit.Start:
                    motion_times = torch.zeros(len(sel), device=self.device)
                else:
                    assert (
                        False
                    ), "Unsupported state initialization strategy: {:s}".format(
                        str(self._state_init)
                    )
            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = (
                curr_motion_lib.get_motion_state(motion_ids, motion_times)
            )
            self._set_env_state(
                curr_env,
                curr_agent,
                root_pos=root_pos + offset,
                root_rot=root_rot,
                dof_pos=dof_pos,
                root_vel=root_vel,
                root_ang_vel=root_ang_vel,
                dof_vel=dof_vel,
            )
            self._reset_ref_slots[sk_name] = (curr_env, curr_agent)
            self._reset_ref_motion_ids[sk_name] = motion_ids
            self._reset_ref_motion_times[sk_name] = motion_times
            body_pos, body_rot, body_vel, body_ang_vel = (
                curr_motion_lib.get_motion_state_max(motion_ids, motion_times)
            )
            body_pos = body_pos + offset.unsqueeze(-2)
            self._kinematic_humanoid_rigid_body_states[curr_env, curr_agent] = (
                torch.cat((body_pos, body_rot, body_vel, body_ang_vel), dim=-1)
            )
            self._every_env_init_dof_pos[curr_env, curr_agent] = dof_pos
        return

    def _set_env_state(
        self,
        slot_env,
        slot_agent,
        root_pos,
        root_rot,
        dof_pos,
        root_vel,
        root_ang_vel,
        dof_vel,
    ):
        self._humanoid_root_states[slot_env, slot_agent, 0:3] = root_pos
        self._humanoid_root_states[slot_env, slot_agent, 3:7] = root_rot
        self._humanoid_root_states[slot_env, slot_agent, 7:10] = root_vel
        self._humanoid_root_states[slot_env, slot_agent, 10:13] = root_ang_vel
        self._dof_pos[slot_env, slot_agent] = dof_pos
        self._dof_vel[slot_env, slot_agent] = dof_vel
        return

    def _collect_random_slots(self, skills):
        """Slots whose box / target must be sampled instead of taken from a motion clip."""
        envs, agents = ([], [])
        if self._reset_default_slots is not None:
            envs.append(self._reset_default_slots[0])
            agents.append(self._reset_default_slots[1])
        for sk_name in skills:
            if self._reset_ref_slots.get(sk_name) is not None:
                envs.append(self._reset_ref_slots[sk_name][0])
                agents.append(self._reset_ref_slots[sk_name][1])
        if len(envs) == 0:
            return None
        return (torch.cat(envs, dim=0), torch.cat(agents, dim=0))

    def _reset_boxes(self, env_ids):
        self._reset_all_boxes_random_arena(env_ids)
        self._platform_pos[env_ids] = self._platform_default_pos[env_ids]
        reference_object_skills = ["pickUp", "carryWith", "putDown"]
        reference_object_skills += ["sit"]
        reference_object_skills += ["climb"]
        for sk_name in reference_object_skills:
            if self._reset_ref_slots.get(sk_name) is None:
                continue
            curr_env, curr_agent = self._reset_ref_slots[sk_name]
            offset = self._agent_spawn_offsets[curr_agent] + self._env_origins[curr_env]
            if sk_name in ("sit", "climb"):
                root_pos, root_rot = self._motion_lib[
                    sk_name
                ].get_obj_motion_state_single_frame(self._reset_ref_motion_ids[sk_name])
            else:
                root_pos, root_rot = self._motion_lib[sk_name].get_obj_motion_state(
                    motion_ids=self._reset_ref_motion_ids[sk_name],
                    motion_times=self._reset_ref_motion_times[sk_name],
                )
            root_pos = root_pos + offset
            curr_box = self._scenario_physical_object(curr_env, curr_agent)
            box_size = self._box_size[curr_env, curr_box]
            if sk_name in ("sit", "climb"):
                root_pos[:, 2] = box_size[:, 2] / 2
            else:
                on_ground = box_size[:, 2] / 2 > root_pos[:, 2]
                root_pos[on_ground, 2] = box_size[on_ground, 2] / 2
            self._box_states[curr_env, curr_box, 0:3] = root_pos
            self._box_states[curr_env, curr_box, 3:7] = root_rot
            self._box_states[curr_env, curr_box, 7:13] = 0.0
            self._platform_pos[curr_env, curr_agent] = self._platform_default_pos[
                curr_env, curr_agent
            ]
        owner_range = self.cfg["env"]["box"]["reset"].get("ownerLocoDistanceRange")
        random_slots = (
            None if owner_range is None else self._collect_random_slots(["loco"])
        )
        if random_slots is not None:
            curr_env, curr_agent = random_slots
            K = curr_env.shape[0]
            curr_box = self._scenario_physical_object(curr_env, curr_agent)
            from utils.stage1_sampling import sample_source_xy

            templates = classify_templates(
                self.relation_runtime.graph, curr_env, curr_agent, True
            )
            root_pos_xy = sample_source_xy(
                self._humanoid_root_states[curr_env, curr_agent, :2], templates,
                max_distance=self.cfg['env'].get('before', {}).get('source_max_distance',
                    self.cfg['env'].get('stage2JointPush', {}).get('solo_source_max_distance', 10.))
            )
            root_pos_z = self._box_size[curr_env, curr_box, 2] / 2
            axis = (
                torch.tensor([[0.0, 0.0, 1.0]], device=self.device)
                .reshape(1, 3)
                .expand([K, -1])
            )
            coeff = 1.0 if self._reset_random_rot else 0.0
            ang = torch.rand((K,), device=self.device) * 2 * np.pi * coeff
            root_rot = quat_from_angle_axis(ang, axis)
            root_pos = torch.cat([root_pos_xy, root_pos_z.unsqueeze(-1)], dim=-1)
            self._box_states[curr_env, curr_box, 0:3] = root_pos
            self._box_states[curr_env, curr_box, 3:7] = root_rot
            self._box_states[curr_env, curr_box, 7:13] = 0.0
        return

    def _reset_task(self, env_ids):
        self._tar_pos[env_ids] = scenario_neutral_targets(
            self._env_origins, env_ids, self.num_agents
        )
        self._tar_platform_pos[env_ids] = self._tar_platform_default_pos[env_ids]
        for sk_name in ["putDown"]:
            if self._reset_ref_slots.get(sk_name) is None:
                continue
            curr_env, curr_agent = self._reset_ref_slots[sk_name]
            templates = classify_templates(
                self.relation_runtime.graph, curr_env, curr_agent, True
            )
            is_at = templates == scenario_templates(self._relation_graph_spec).index(
                "HOLDING_AT"
            )
            curr_env, curr_agent = (curr_env[is_at], curr_agent[is_at])
            if not len(curr_env):
                continue
            motion_ids = self._reset_ref_motion_ids[sk_name][is_at]
            curr_box = self._scenario_physical_object(curr_env, curr_agent)
            goal_index = self._scenario_goal(curr_env, curr_agent)
            offset = self._agent_spawn_offsets[curr_agent] + self._env_origins[curr_env]
            root_pos, root_rot = self._motion_lib[sk_name].get_obj_motion_state(
                motion_ids=motion_ids,
                motion_times=self._motion_lib[sk_name].get_motion_length(motion_ids),
            )
            root_pos = root_pos + offset
            root_pos[:, 2] = self._box_size[curr_env, curr_box, 2] / 2
            self._tar_pos[curr_env, goal_index] = root_pos
            self._tar_platform_pos[curr_env, curr_agent] = (
                self._tar_platform_default_pos[curr_env, curr_agent]
            )
        random_slots = self._collect_random_slots(["loco", "pickUp", "carryWith"])
        if random_slots is not None:
            curr_env, curr_agent = random_slots
            template = classify_templates(
                self.relation_runtime.graph, curr_env, curr_agent, True
            )
            keep = template == 3
            curr_env, curr_agent = (curr_env[keep], curr_agent[keep])
            if not len(curr_env):
                return
            K = curr_env.shape[0]
            curr_box = self._scenario_physical_object(curr_env, curr_agent)
            goal_index = self._scenario_goal(curr_env, curr_agent)
            from utils.stage1_sampling import sample_goal_xy
            new_xy = sample_goal_xy(
                self._humanoid_root_states[curr_env, curr_agent, :2],
                self._box_states[curr_env, curr_box, :2], self._env_origins[curr_env, :2])
            new_z = self._box_size[curr_env, curr_box, 2] / 2
            new_target_pos = torch.cat([new_xy, new_z.unsqueeze(-1)], dim=-1)
            self._tar_pos[curr_env, goal_index] = new_target_pos
        return

    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)
        if self._is_eval:
            flat = self._flat_slot_ids(env_ids)
            self._success_buf[flat] = 0
            self._precision_buf[flat] = float("Inf")
        ids = [self._box_actor_ids[env_ids].contiguous().view(-1)]
        if self._enable_markers:
            self._marker_pos[env_ids] = self._tar_pos[env_ids]
            from utils.edge_ontop_spec import select_graph

            graph = select_graph(self.relation_runtime.graph, env_ids)
            at_edges, goal_slots, active = at_goal_marker_slots(graph)
            self._marker_pos[env_ids, :, 2] = torch.where(
                active, self._tar_pos[env_ids, :, 2], 20.0
            )
            shown = (
                torch.ones_like(env_ids, dtype=torch.bool)
                if self.viewer is not None
                else env_ids == 0
            )
            rows, edges = (at_edges & shown[:, None]).nonzero(as_tuple=True)
            assignments = (
                torch.stack(
                    (
                        env_ids[rows],
                        goal_slots[rows, edges],
                        graph.edge_owner[rows, edges],
                    ),
                    dim=-1,
                )
                .cpu()
                .tolist()
            )
            for env_id, goal_id, owner in assignments:
                marker = self._marker_handles[env_id * self.num_agents + goal_id]
                self.gym.set_rigid_body_color(
                    self.envs[env_id],
                    marker,
                    0,
                    gymapi.MESH_VISUAL,
                    self._agent_color(owner),
                )
            ids.append(self._marker_actor_ids[env_ids].contiguous().view(-1))
        self._platform_states[env_ids, :, 3:6] = 0.0
        self._platform_states[env_ids, :, 6] = 1.0
        self._platform_states[env_ids, :, 7:13] = 0.0
        self._tar_platform_states[env_ids, :, 3:6] = 0.0
        self._tar_platform_states[env_ids, :, 6] = 1.0
        self._tar_platform_states[env_ids, :, 7:13] = 0.0
        ids.extend(
            [
                self._platform_actor_ids[env_ids].contiguous().view(-1),
                self._tar_platform_actor_ids[env_ids].contiguous().view(-1),
            ]
        )
        ids = torch.cat(ids)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(ids),
            len(ids),
        )
        return

    def _sample_arena_xy(self, centers):
        """Uniform samples by area inside the configured arena disc."""
        count = centers.shape[0]
        angle = torch.rand(count, device=self.device) * (2.0 * np.pi)
        radius = (
            torch.sqrt(torch.rand(count, device=self.device)) * self._box_spawn_radius
        )
        offset = torch.stack(
            [torch.cos(angle), torch.sin(angle)], dim=-1
        ) * radius.unsqueeze(-1)
        return centers + offset

    def _reset_all_boxes_random_arena(self, env_ids):
        """Inference reset: every physical box is independent of every agent owner."""
        K, O = (env_ids.shape[0], self.num_physical_objects)
        centers = self._env_origins[env_ids, 0:2]
        agent_xy = self._humanoid_root_states[env_ids, :, 0:2]
        positions = torch.zeros(K, O, 2, device=self.device)
        for object_id in range(O):
            candidate = self._sample_arena_xy(centers)
            for _ in range(64):
                near_agent = (
                    torch.sum((candidate.unsqueeze(1) - agent_xy) ** 2, dim=-1)
                    .min(dim=-1)
                    .values
                    < self._box_min_agent_dist**2
                )
                if object_id > 0:
                    near_box = (
                        torch.sum(
                            (candidate.unsqueeze(1) - positions[:, :object_id]) ** 2,
                            dim=-1,
                        )
                        .min(dim=-1)
                        .values
                        < self._box_min_box_dist**2
                    )
                    invalid = torch.logical_or(near_agent, near_box)
                else:
                    invalid = near_agent
                if not torch.any(invalid):
                    break
                candidate[invalid] = self._sample_arena_xy(centers[invalid])
            positions[:, object_id] = candidate
        self._box_states[env_ids, :, 0:2] = positions
        self._box_states[env_ids, :, 2] = self._box_size[env_ids, :, 2] / 2.0
        axis = torch.zeros(K * O, 3, device=self.device)
        axis[:, 2] = 1.0
        coeff = 1.0 if self._reset_random_rot else 0.0
        angle = torch.rand(K * O, device=self.device) * (2.0 * np.pi) * coeff
        self._box_states[env_ids, :, 3:7] = quat_from_angle_axis(angle, axis).view(
            K, O, 4
        )
        self._box_states[env_ids, :, 7:13] = 0.0
        self._ground_stage1_standalone_targets(env_ids)
        return

