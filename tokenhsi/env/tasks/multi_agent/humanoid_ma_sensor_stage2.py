import math
import os

import torch

from env.tasks.multi_agent import square_sensor as sensor
from env.tasks.multi_agent.humanoid_ma_stage2 import HumanoidMAStage2
from utils import stage2_spec, coupled_push_spec
from utils import coupled_push_scene
from utils.sensor_stage2_spec import bridge_roles
from utils.edge_ontop_spec import copy_graph_rows
from utils.sensor_stage2_spec import BRIDGE, CLIFF, PUSH, ROOM, WAREHOUSE, SURFACE_A, paired_people, sample_surfaces
from env.tasks.multi_agent.sensor_stage2_observation import WALL_CONTACT_BODIES
from env.tasks.multi_agent.sensor_stage2_geometry import SensorStage2Geometry
from env.tasks.multi_agent.sensor_stage2_placement import SensorStage2Placement
from env.tasks.multi_agent.sensor_stage2_observation import SensorStage2Observation
from env.tasks.multi_agent.sensor_stage2_curriculum import SensorStage2Curriculum


class HumanoidMASensorStage2(SensorStage2Curriculum, SensorStage2Geometry, SensorStage2Placement, SensorStage2Observation, HumanoidMAStage2):

    def __init__(self, cfg, *args, **kwargs):
        env = cfg['env']
        self._scene_cfg = env['sensorScene']
        self._wall_collision_scale = 1.
        if 'wall_collision_coefficient' in self._scene_cfg:
            self._wall_collision_scale = self._scene_cfg['wall_collision_coefficient']/env['agentCollisionCoeff']
        self._push_box_collision_scale = self._scene_cfg.get('push_box_collision_coefficient', 0.)/env['agentCollisionCoeff']
        self._bridge_training_init = self._scene_cfg['bridge'].get('start_distribution')
        self._bridge_curriculum = self._scene_cfg['bridge'].get('curriculum')
        self._curriculum_eval_mode = os.environ.get('BRIDGE_CURRICULUM_EVAL', 'cooperative')
        if self._curriculum_eval_mode not in ('cooperative', 'training', 'walk_entry') or (os.environ.get('BRIDGE_CURRICULUM_EVAL') and not cfg['args'].eval):
            raise ValueError('BRIDGE_CURRICULUM_EVAL is evaluation-only: cooperative, training or walk_entry')
        self._scene_names = tuple(self._scene_cfg['mixture'])
        self._warehouse = self._scene_cfg.get('warehouse')
        self._warehouse_coop = self._warehouse.get('cooperative') if self._warehouse else None
        self._warehouse_before_probability = self._warehouse_coop.get('before_pairs_probability', 0.) if self._warehouse_coop else 0.
        self._warehouse_variant_choice = os.environ.get('WAREHOUSE_VARIANT', 'mixed')
        if self._warehouse_variant_choice not in ('mixed', 'push_before', 'before_pairs'):
            raise ValueError('WAREHOUSE_VARIANT must be mixed, push_before or before_pairs')
        if os.environ.get('WAREHOUSE_VARIANT') and (not cfg['args'].eval or not self._warehouse_before_probability):
            raise ValueError('WAREHOUSE_VARIANT requires evaluation of a BEFORE-pairs warehouse config')
        self._warehouse_before_cache = None
        self._warehouse_door_progress = self._warehouse.get('leader_door_progress', False) if self._warehouse else False
        # Like the door start distribution, leader-done starts are training-only.
        self._warehouse_done_probability = 0. if cfg['args'].eval or not self._warehouse_coop else \
            self._warehouse_coop.get('leader_done_probability', 0.)
        self._warehouse_start = self._warehouse.get('start_distribution') if self._warehouse else None
        self._warehouse_start_choice = os.environ.get('WAREHOUSE_START')
        if self._warehouse_start_choice and (not cfg['args'].eval or not self._warehouse_start):
            raise ValueError('WAREHOUSE_START is for evaluating the warehouse RSI experiment only')
        self._warehouse_start_choice = self._warehouse_start_choice or ('closed' if cfg['args'].eval else 'mixed')
        if self._warehouse_start_choice not in ('closed','partial','open','mixed'):
            raise ValueError('WAREHOUSE_START must be closed, partial, open or mixed')
        spec = self._scene_cfg['sensor']
        self._sensor_points = spec['resolution']**2
        self._sensor_obs_width = env['numAgents']*self._sensor_points
        self._scene_spacing = env['sceneSpacing']
        probabilities = [self._scene_cfg['mixture'][name] for name in self._scene_names]
        chosen = os.environ.get('SCENARIO')
        if chosen:
            if not cfg['args'].eval or chosen not in self._scene_names:
                raise ValueError('SCENARIO must select a configured scene for evaluation only')
            probabilities = [float(name == chosen) for name in self._scene_names]
        if os.environ.get('SENSOR_MASK') == '1' and not cfg['args'].eval:
            raise ValueError('SENSOR_MASK is an evaluation-only option')
        if os.environ.get('SENSOR_VIS') == '1':
            env['enableDebugVis'] = True
        self._scene_probabilities_list = probabilities
        self._terrain_ids = None
        super().__init__(cfg, *args, **kwargs)
        self._arena_scale = float(env['observationArenaScale'])
        self._sensor_grid = sensor.sensor_grid(spec['resolution'], spec['range'], self.device)
        self._foot_vertices = sensor.load_foot_vertices(env['asset'], self.device)
        self._foot_body_ids = [self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.humanoid_handles_all[0][0], name) for name in ('right_foot', 'left_foot')]

    def get_relation_suffix_size(self):
        return super().get_relation_suffix_size()+self._sensor_obs_width

    def _build_assignment_tensors(self):
        super()._build_assignment_tensors()
        n, m = self.num_envs, self.num_agents
        device = self.device
        self._scene_probabilities = torch.tensor(self._scene_probabilities_list, device=device)
        self._scene_ids = torch.zeros(n, dtype=torch.long, device=device)
        self._scene_origins = self._env_origins[:, None]+self._scene_offsets[None]
        self._base_scene_origins = self._scene_origins.clone()
        self._terrain_actor_records = torch.tensor(self._terrain_actor_records,device=device)
        self._static_centers = self._static_local.clone()
        self._static_centers[..., :2] += self._scene_origins[:, :, None, :2]
        self._static_centers[..., 2] += self._scene_origins[:, :, None, 2]
        self._wall_starts = self._wall_local[0]+self._scene_origins[:, ROOM, None, :2]
        self._wall_ends = self._wall_local[1]+self._scene_origins[:, ROOM, None, :2]
        self._roles = torch.arange(m, device=device)[None].repeat(n, 1)
        self._surface = torch.full((n, m), SURFACE_A, dtype=torch.long, device=device)
        self._start_surface = self._surface.clone()
        self._unreachable = torch.zeros(n, m, dtype=torch.bool, device=device)
        self._place_failed = torch.zeros(n, dtype=torch.bool, device=device)
        self._bridge_near = torch.zeros(n, dtype=torch.bool, device=device)
        self._bridge_aligned = torch.zeros(n, dtype=torch.bool, device=device)
        self._bridge_carry_from_start = torch.zeros(n, dtype=torch.bool, device=device)
        self._bridge_side = torch.zeros(n, dtype=torch.bool, device=device)
        self._bridge_landing = torch.zeros(n, 2, device=device)
        self._bridge_landing_world = torch.zeros(n, 3, device=device)
        self._bridge_arrived = torch.zeros(n, dtype=torch.bool, device=device)
        self._bridge_crossed = torch.zeros(n, dtype=torch.bool, device=device)
        self._bridge_start_mode = torch.zeros(n, dtype=torch.long, device=device)
        self._bridge_entered = torch.zeros(n, dtype=torch.bool, device=device)
        self._bridge_start_outcomes = torch.zeros(3, 3, 5, device=device)
        self._bridge_start_samples = torch.zeros(3, 3, device=device)
        # episodes, BEFORE pairs, independent non-PUSH people, independent PUSH people, unreachable people
        self._scene_sampling = torch.zeros(len(self._scene_names), 5, device=device)
        # ended episodes, env steps, agent falls, timeouts
        self._scene_outcomes = torch.zeros(len(self._scene_names), 4, device=device)
        # wall-contact person steps, person steps, wall terminations
        self._wall_outcomes = torch.zeros(len(self._scene_names), 3, device=device)
        self._wall_contact_steps = torch.zeros(n, m, dtype=torch.long, device=device)
        self._wall_contact_body_ids = [self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.humanoid_handles_all[0][0], name) for name in WALL_CONTACT_BODIES]
        self._wall_contact_shapes = sensor.load_body_shapes(self.cfg['env']['asset'], WALL_CONTACT_BODIES, device)
        duration = self._scene_cfg.get('wall_contact_termination')
        self._wall_contact_limit = math.ceil(duration/self.dt-1e-6) if duration else None
        # non-pusher PUSH-box contact steps, non-pusher steps, terminations
        self._push_box_outcomes = torch.zeros(3, device=device)
        self._push_box_contact_steps = torch.zeros(n, m, dtype=torch.long, device=device)
        duration = self._scene_cfg.get('push_box_contact_termination')
        self._push_box_contact_limit = math.ceil(duration/self.dt-1e-6) if duration else None
        # start modes original/near/aligned: ended episodes, arrived, crossed, crosser falls
        self._bridge_outcomes = torch.zeros(3, 4, device=device)
        # person steps within 1m of a cliff edge, stopped there, cross requests, blocked straight paths, room AT people
        self._scene_counters = torch.zeros(5, device=device)
        self._bridge_count = torch.ones(n,dtype=torch.long,device=device)
        self._bridge_slot = torch.full((n,),self._bridge_bank,dtype=torch.long,device=device)
        self._bridge_pushers = torch.zeros(n,m,dtype=torch.bool,device=device)
        self._push_goals = self._center_tar_pos
        self._bridge_count_outcomes = torch.zeros(3,4,device=device)
        self._bridge_contact_counts = torch.zeros(3,2,device=device)
        if self._bridge_curriculum:
            self._init_bridge_curriculum()
        if self._warehouse:
            from env.tasks.multi_agent.warehouse_sensor import allocate
            allocate(self)
            if self._warehouse_door_progress:
                self._warehouse_door_passed = torch.zeros(n, m, dtype=torch.bool, device=device)
            if self._warehouse_before_probability:
                from env.tasks.multi_agent import warehouse_before
                warehouse_before.allocate(self)
                if not self.cfg['args'].eval and os.environ.get('WAREHOUSE_BEFORE_CACHE', '1') != '0':
                    from env.tasks.multi_agent.warehouse_before_cache import BeforeLayoutCache
                    self._warehouse_before_cache = BeforeLayoutCache(n, self.cfg['args'].seed,
                        shuffle_edges=self._relation_graph_spec['shuffle_edge_order'],
                        warmup_resets=int(os.environ.get('WAREHOUSE_BEFORE_CACHE_WARMUP',
                            '32' if self._state_init == self.StateInit.Default else '0')))
            if self._warehouse_done_probability:
                from env.tasks.multi_agent import warehouse_done
                warehouse_done.allocate(self)

    def _reset_envs(self, env_ids):
        if len(env_ids):
            if self._warehouse_door_progress:
                self._warehouse_door_passed[env_ids] = False
            self._wall_contact_steps[env_ids] = 0
            self._push_box_contact_steps[env_ids] = 0
            chosen = torch.multinomial(self._scene_probabilities,len(env_ids),replacement=True)
            self._scene_ids[env_ids] = chosen
            counts = torch.randint(1,4,(len(env_ids),),device=self.device)
            counts[chosen == WAREHOUSE] = 1
            if self._warehouse_coop:
                selected = chosen == WAREHOUSE
                if selected.any():
                    counts[selected] = torch.multinomial(torch.tensor(self._warehouse_coop['pusher_probabilities'],
                        device=self.device), int(selected.sum()), True)+1
                if self._warehouse_before_probability:
                    from env.tasks.multi_agent import warehouse_before
                    warehouse_before.sample(self, env_ids, selected)
                if self._warehouse_done_probability:
                    from env.tasks.multi_agent import warehouse_done
                    warehouse_done.sample(self, env_ids, selected)
            spans = counts
            if self._bridge_curriculum:
                counts, spans = self._sample_bridge_curriculum(env_ids, counts)
            self._bridge_count[env_ids] = counts
            self._bridge_slot[env_ids] = self._bridge_bank+spans-1
            if self._warehouse_coop:
                from env.tasks.multi_agent.warehouse_coop import select_geometry
                select_geometry(self, env_ids[chosen == WAREHOUSE])
            for key, values in self._bridge_geometry.items():
                target = getattr(self,key)
                selected = values[env_ids,spans-1]
                if key.startswith('_bridge_'):
                    target[env_ids] = selected
                else:
                    target[env_ids,BRIDGE] = selected[:,BRIDGE]
            self._scene_origins[env_ids,BRIDGE] = self._base_scene_origins[env_ids,BRIDGE]
            self._scene_origins[env_ids,BRIDGE,0] += (spans-1)*2*self._scene_spacing
            self._static_centers[env_ids,BRIDGE] = self._static_local[env_ids,BRIDGE]+self._scene_origins[env_ids,BRIDGE,None]
            self._env_origins[env_ids] = self._scene_origins[env_ids,chosen]
            self._bridge_arrived[env_ids] = False
            self._bridge_crossed[env_ids] = False
            self._bridge_entered[env_ids] = False
            if self._warehouse:
                self._warehouse_opened[env_ids] = False
                self._warehouse_arrived[env_ids] = False
                self._warehouse_entered[env_ids] = False
                if self._warehouse_start:
                    from env.tasks.multi_agent.warehouse_sensor import sample_starts
                    starts = env_ids[chosen == WAREHOUSE]
                    if self._warehouse_before_probability:
                        starts = starts[~self._warehouse_before_pairs[starts]]
                    if self._warehouse_done_probability:
                        done = self._warehouse_leader_done[starts]
                        self._warehouse_start_mode[starts[done]] = 2
                        starts = starts[~done]
                    sample_starts(self,starts)
        return super()._reset_envs(env_ids)

    def _draw_episode_graph(self, ids):
        env = self.cfg['env']
        sizes = self._box_size[ids,self._leader_bank:self._leader_bank+self.num_agents]
        probabilities = self._relation_graph_spec['template_probabilities']
        domain,groups,pre,tasks = stage2_spec.sample_episode(sizes,env['box']['build']['taskSizeRanges'],
            probabilities,self._task_graph_preset,env['stage2']['scene_weights'],num_objects=self.num_objects)
        if self._scene_cfg['cliff'].get('sampling'):
            from utils.sensor_safety_spec import cliff_roles
            rows = (self._scene_ids[ids] == CLIFF).nonzero().flatten()
            if len(rows):
                groups[rows], pre[rows], tasks[rows] = cliff_roles(sizes[rows], probabilities,
                    self._scene_cfg['cliff']['sampling']['all_edge_probability'])
                domain[rows] = 1
        if self._scene_cfg['room'].get('sampling'):
            from utils.sensor_safety_spec import room_roles
            rows = (self._scene_ids[ids] == ROOM).nonzero().flatten()
            if len(rows):
                domain[rows], groups[rows], pre[rows], tasks[rows] = room_roles(sizes[rows],
                    env['box']['build']['taskSizeRanges'], probabilities,
                    self._scene_cfg['room']['sampling'], self._task_graph_preset)
        rows = (self._scene_ids[ids]==BRIDGE).nonzero().flatten()
        self._bridge_pushers[ids] = False
        if len(rows):
            g,p,t,roles,push = bridge_roles(sizes[rows],env['box']['build']['taskSizeRanges'],probabilities,self._bridge_count[ids[rows]])
            groups[rows],pre[rows],tasks[rows] = g,p,t
            self._roles[ids[rows]] = roles
            self._bridge_pushers[ids[rows]] = push
        rows = (self._scene_ids[ids] == WAREHOUSE).nonzero().flatten()
        if len(rows):
            pairing = self._warehouse.get('worker_pairing_probability', env['before']['pairing_probability'])
            if self._warehouse_coop:
                from utils.warehouse_coop import sample_roles_graph
                g,p,t,roles = sample_roles_graph(sizes[rows],env['box']['build']['taskSizeRanges'],
                    probabilities,self._bridge_count[ids[rows]],pairing)
            else:
                from utils.warehouse_sensor import sample_roles_graph
                g,p,t,roles = sample_roles_graph(sizes[rows],env['box']['build']['taskSizeRanges'],
                    probabilities,'random_scenario',pairing)
            groups[rows],pre[rows],tasks[rows] = g,p,t
            domain[rows] = 0
            self._roles[ids[rows]] = roles
            if self._warehouse_before_probability:
                from env.tasks.multi_agent import warehouse_before
                flagged = rows[self._warehouse_before_pairs[ids[rows]]]
                if len(flagged):
                    g,p,t,roles = warehouse_before.sample_roles(sizes[flagged],
                        env['box']['build']['taskSizeRanges'],probabilities)
                    groups[flagged],pre[flagged],tasks[flagged] = g,p,t
                    self._roles[ids[flagged]] = roles
        if self._bridge_curriculum:
            groups, pre, tasks = self._curriculum_roles(ids, groups, pre, tasks)
        self._surface[ids],self._start_surface[ids],self._unreachable[ids] = sample_surfaces(
            tasks,pre,self._roles[ids],self._scene_ids[ids],self._bridge_pushers[ids])
        mixture = self._scene_cfg['bridge']['start_mixture']
        self._bridge_near[ids] = torch.rand(len(ids),device=self.device)<mixture['near_probability']
        self._bridge_aligned[ids] = self._bridge_near[ids] & (torch.rand(len(ids),device=self.device)<mixture['near_aligned_probability'])
        if self._bridge_training_init:
            from utils.sensor_stage2_spec import sample_bridge_starts
            mode = sample_bridge_starts(self._bridge_training_init, len(ids), self.device)
            self._bridge_start_mode[ids] = mode
            self._bridge_near[ids] = mode == 1
            self._bridge_aligned[ids] = (mode == 0) | ((mode == 1) &
                (torch.rand(len(ids), device=self.device) < mixture['near_aligned_probability']))
        self._bridge_side[ids] = torch.rand(len(ids),device=self.device)<.5
        if self._bridge_training_init:
            self._bridge_side[ids] = True
        carry = tasks.gather(1,self._roles[ids,1:2]).squeeze(1)
        self._bridge_carry_from_start[ids] = ((carry==3)|(carry==4)) & (torch.rand(len(ids),device=self.device)<self._scene_cfg['bridge']['carry_from_start_probability'])
        if self._bridge_curriculum:
            self._curriculum_surfaces(ids)
        return domain,groups,pre,tasks

    def _count_sampling(self, current, predecessor, templates, roles, unreachable):
        independent = ~paired_people(predecessor) & (templates != PUSH)
        for index in range(len(self._scene_names)):
            chosen = current == index
            self._scene_sampling[index] += torch.stack((chosen.sum(),(predecessor[chosen]>=0).sum(),
                independent[chosen].sum(),(templates[chosen]==PUSH).sum(),unreachable[chosen].sum())).float()

    def _sample_episode_graph(self, ids):
        domain, groups, predecessor, templates = self._draw_episode_graph(ids)
        graph = stage2_spec.compose_graph(templates, groups, predecessor, self.num_objects,
                                   self._relation_graph_spec['shuffle_edge_order'])
        mapping, primary, slots = stage2_spec.physical_mapping(templates, groups, predecessor, graph)
        # Replace only the physical leader payload; logical aliases follow the same mapping.
        for leader in range(self.num_agents):
            selected = (predecessor == leader).any(-1)
            old = primary[:,leader].clone()
            new = self._leader_bank+leader
            mapping = torch.where(selected[:,None] & (mapping==old[:,None]),new,mapping)
            primary = torch.where(selected[:,None] & (primary==old[:,None]),new,primary)
        push = self._bridge_pushers[ids] | ((self._scene_ids[ids] == WAREHOUSE)[:,None] & (templates == PUSH))
        old = primary.gather(1,self._roles[ids,:1])
        mapping = torch.where(push.any(-1)[:,None] & (mapping==old),self._bridge_slot[ids,None],mapping)
        primary = torch.where(push,self._bridge_slot[ids,None],primary)
        self._groups[ids], self._templates[ids] = groups, templates
        self._before_predecessor[ids] = predecessor
        self._before_near_pairs[ids] = (predecessor >= 0) & (torch.rand_like(predecessor, dtype=torch.float) < .5)
        self._members[ids] = torch.where(templates == 5, coupled_push_spec.group_sizes(groups), 1)
        self._regions[ids] = torch.where(self._members[ids] > 1, coupled_push_spec.assign_regions(groups), 0)
        self._agent_box_assignment[ids], self._logical_box_order[ids] = primary, mapping
        self._stage2_task_slots[ids], self._scene_domains[ids] = slots, domain
        copy_graph_rows(self.relation_runtime.graph, ids, graph)
        self.relation_runtime.before[ids] = stage2_spec.before_members(predecessor)
        if self.relation_runtime.relaxed_at_owners is not None:
            self.relation_runtime.relaxed_at_owners[ids] = self._members[ids] == 1
        self._edge_goal_owners[ids] = True
        self._ontop_maintain_steps[ids] = 0
        self._sampling_counts[0] += len(ids)
        self._sampling_counts[1:7] += torch.bincount(templates.flatten(), minlength=6)
        self._stage2_domain_counts += torch.bincount(domain, minlength=3)
        self._push_hits[ids] = False
        self._update_box_assignment_colors(ids)
        self._count_sampling(self._scene_ids[ids],predecessor,templates,self._roles[ids],self._unreachable[ids])
        if self._bridge_training_init:
            bridge = ids[self._scene_ids[ids] == BRIDGE]
            key = self._bridge_start_mode[bridge]*3+self._bridge_count[bridge]-1
            self._bridge_start_samples += torch.bincount(key, minlength=9).reshape(3, 3)

    def _reset_ontop_context_envs(self, ids):
        if not len(ids):
            return
        cache = self._warehouse_before_cache
        if cache is not None:
            cache.warmup(self, ids)
        self._sample_episode_graph(ids)
        hits = cache.restore(self, ids) if cache is not None else ids[:0]
        if len(hits):
            self._update_box_assignment_colors(hits)
        fresh = ids[~torch.isin(ids, hits)] if len(hits) else ids
        self._reset_default_slots = None
        self._reset_ref_slots,self._reset_ref_motion_ids,self._reset_ref_motion_times = {},{},{}
        self._reset_actors(fresh)
        saved = (self._reset_ref_slots.copy(),self._reset_ref_motion_ids.copy(),self._reset_ref_motion_times.copy())
        compact = hasattr(self, '_waypoint_phase') and not self._waypoint_baseline
        snapshot = fresh if compact else torch.arange(self.num_envs, device=self.device)
        flat_roots = self._humanoid_root_states[snapshot].clone()
        flat_bodies = self._kinematic_humanoid_rigid_body_states[snapshot].clone()
        snapshot_rows = torch.empty(self.num_envs, dtype=torch.long, device=self.device)
        snapshot_rows[snapshot] = torch.arange(len(snapshot), device=self.device)
        pending = fresh
        attempts = self.cfg['env']['stage2JointPush']['layout_attempts']
        for attempt in range(4*attempts):
            if not len(pending):
                break
            self._humanoid_root_states[pending] = flat_roots[snapshot_rows[pending]]
            self._kinematic_humanoid_rigid_body_states[pending] = flat_bodies[snapshot_rows[pending]]
            if attempt and attempt % attempts == 0:
                saved = self._resample_before_references(pending, saved, include_coupled=True)
                flat_roots[snapshot_rows[pending]] = self._humanoid_root_states[pending]
                flat_bodies[snapshot_rows[pending]] = self._kinematic_humanoid_rigid_body_states[pending]
            elif attempt and self.cfg['env'].get('stage2') is not None:
                saved = self._resample_before_references(pending, saved)
            self._reset_ref_slots,self._reset_ref_motion_ids,self._reset_ref_motion_times = (v.copy() for v in saved)
            self._keep_reset_slots_for_envs(pending)
            self._reset_boxes(pending)
            self._reset_task(pending)
            self._configure_ontop_targets(pending)
            if self._individual_push_goals:
                self._center_tar_pos[pending] = self._tar_pos[pending]
            rejected = coupled_push_scene.sample_push(self,pending)
            coupled_push_scene.place_distractors(self,pending)
            from utils.stage1_platforms import reset_stage1_platforms
            reset_stage1_platforms(self,pending)
            rejected |= HumanoidMAStage2._ontop_scene_infeasible(self,pending)
            warehouse = self._scene_ids[pending] == WAREHOUSE
            # Apron teams keep the flat BEFORE layout, so its own rejection still applies.
            base = rejected & self._warehouse_before_pairs[pending] if self._warehouse_before_probability else None
            rejected[warehouse] = False
            self._place_scene(pending, room_retry=attempt >= attempts)
            if warehouse.any():
                from env.tasks.multi_agent.scene_validation import scene_infeasible
                from env.tasks.multi_agent.stage2_layout import invalid_before_layout
                selected = pending[warehouse]
                rejected[warehouse] = scene_infeasible(self,selected) | invalid_before_layout(self,selected)
            if base is not None:
                rejected |= base
            self._center_tar_pos[pending] = self._tar_pos[pending]
            rejected |= self._place_failed[pending]
            self._sampling_retries += rejected.sum()
            pending = pending[rejected]
            if not len(pending):
                break
        if len(pending):
            details = dict(envs=pending[:16].tolist(), scenes=self._scene_ids[pending[:16]].tolist(),
                domains=self._scene_domains[pending[:16]].tolist(), tasks=self._templates[pending[:16]].tolist(),
                groups=self._groups[pending[:16]].tolist())
            raise RuntimeError('Stage2 jointpush layout exhausted after reference refresh: '+str(details))
        if self._coupled_assignment != 'random':
            from utils.coupled_push_assignment import assign_regions
            self._regions[ids] = assign_regions(
                self._groups[ids], self._humanoid_root_states[ids],
                self._assigned_box_values(self._box_states, ids),
                self._assigned_box_values(self._box_size, ids), self._regions[ids],
                self._coupled_assignment, self.cfg['env']['stage2JointPush']['human_clearance'])
            self._regions[ids] = torch.where(self._members[ids] > 1, self._regions[ids], 0)
        elif self._warehouse and self._warehouse.get('waypoint', {}).get('view_mode') == 'leader_ontop_follower':
            from env.tasks.multi_agent.warehouse_sensor import match_push_regions
            match_push_regions(self, ids)
        if self._individual_push_goals:
            boxes = self._assigned_box_values(self._box_states, ids)
            sizes = self._assigned_box_values(self._box_size, ids)
            tokens, _ = coupled_push_spec.region_states(boxes, sizes, self._members[ids], self._regions[ids])
            offset = tokens[..., :3]-boxes[..., :3]
            self._tar_pos[ids] = torch.where((self._members[ids] > 1)[..., None],
                self._center_tar_pos[ids]+offset, self._tar_pos[ids])
        self._reset_ref_slots,self._reset_ref_motion_ids,self._reset_ref_motion_times = saved
        if self._state_init == self.StateInit.Default:
            self._reset_default_slots = self._expand_slots(fresh)
        if not self._is_eval:
            templates = self._templates_by_owner[ids].flatten()
            self._size_rsi_executed[:,0] += torch.bincount(templates,minlength=6)
        self._reset_env_tensors(ids)
        self._refresh_sim_tensors()
        self._reset_relation_history(ids)
        if self._coupled_peer_mode != 'all':
            self._single_peer_selector.reset(ids)
        self._compute_observations(ids)
        self._init_amp_obs(ids)
        if cache is not None:
            cache.finish(self, fresh)
