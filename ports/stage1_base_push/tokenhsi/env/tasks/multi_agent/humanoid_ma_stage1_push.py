"""Additive Stage1 PUSH_AT task; original tasks and policy architecture are retained."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json
import hashlib
import os
import math
import yaml
from isaacgym import gymapi, gymtorch
import torch
from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry, build_amp_observations
from env.tasks.multi_agent.stage1_push_reward import Stage1PushRuntime
from env.tasks.multi_agent.push_contact import (push_rear_xy, push_hand_contact, push_rsi_hand_targets)
from utils.stage1_push_spec import (PUSH, PUSH_TEMPLATE, TEMPLATES, compose_graph,
                                   sample_asset_sizes, validate_env)
from utils.edge_scenario_spec import classify_templates
from utils.edge_ontop_spec import copy_graph_rows, select_graph, batched
from utils.door_reference import rotate, multiply, yaw_from_quat


class LegacyRsiCache:
    """Keep legacy screening available; PUSH uses its already screened native bank."""
    def __init__(self, task):
        self.task, self.cache = task, None

    def _load(self):
        if self.cache is None:
            from env.tasks.multi_agent.size_rsi_cache import SizeRsiCache
            self.cache = SizeRsiCache(self.task)
        return self.cache

    def availability(self, *args):
        return self._load().availability(*args)

    def sample(self, *args):
        return self._load().sample(*args)


class HumanoidMAStage1Push(HumanoidMACarry):
    def __init__(self, cfg, *args, **kwargs):
        validate_env(cfg['env'])
        self._viewer_cubes = os.environ.get('TOKENHSI_VIEWER_PUSH_CUBES') == '1'
        self._viewer_xy_random = os.environ.get('TOKENHSI_VIEWER_PUSH_XY_RANDOM') == '1'
        self._viewer_cube_side = os.environ.get('TOKENHSI_VIEWER_PUSH_CUBE_SIDE')
        self._viewer_cube_friction = os.environ.get('TOKENHSI_VIEWER_PUSH_CUBE_FRICTION')
        cube_sides = os.environ.get('TOKENHSI_VIEWER_PUSH_CUBE_SIDES')
        box_ranges = os.environ.get('TOKENHSI_VIEWER_PUSH_BOX_RANGES')
        xy_range = os.environ.get('TOKENHSI_VIEWER_PUSH_XY_RANGE')
        self._viewer_xy_range = None
        if xy_range is not None:
            if (not self._viewer_cubes or self._viewer_xy_random or box_ranges is not None
                    or self._viewer_cube_side is not None or cube_sides is not None):
                raise ValueError('XY_RANGE requires CUBES=1 without other size overrides')
            values = [float(value) for value in xy_range.split(',')]
            if (len(values) != 2 or not all(math.isfinite(v) for v in values)
                    or not 1. <= values[0] <= values[1] <= 2.2
                    or any(abs(v*10-round(v*10)) > 1e-6 for v in values)):
                raise ValueError('XY_RANGE needs min,max in [1,2.2] on the .1m grid; Z stays sampled')
            self._viewer_xy_range = [round(v*10) for v in values]
        self._viewer_box_ranges = None
        if box_ranges is not None:
            if (not self._viewer_cubes or self._viewer_xy_random
                    or self._viewer_cube_side is not None or cube_sides is not None):
                raise ValueError('BOX_RANGES requires CUBES=1 without other size overrides')
            values = [float(value) for value in box_ranges.split(',')]
            if (len(values) != 6 or not all(math.isfinite(value) for value in values)
                    or any(not 1. <= values[i] <= values[i+1] <= 1.7 for i in (0, 2, 4))
                    or values[2:4] != values[4:6]):
                raise ValueError('BOX_RANGES needs xmin,xmax,ymin,ymax,zmin,zmax in [1,1.7], equal Y/Z ranges')
            self._viewer_box_ranges = values
        self._viewer_cube_sides = [1., 1.1, 1.2, 1.3, 1.4, 1.5]
        if cube_sides is not None:
            if not self._viewer_cubes or self._viewer_xy_random or self._viewer_cube_side is not None:
                raise ValueError('CUBE_SIDES requires CUBES=1 without XY_RANDOM or fixed CUBE_SIDE')
            self._viewer_cube_sides = [float(value) for value in cube_sides.split(',')]
            if not all(math.isfinite(value) and 1. <= value <= 1.5 for value in self._viewer_cube_sides):
                raise ValueError('Viewer cube sides must be in [1.0, 1.5] metres')
        if self._viewer_xy_random and (not self._viewer_cubes or self._viewer_cube_side is None):
            raise ValueError('Viewer XY random sizes require CUBES=1 and CUBE_SIDE for fixed Z')
        if not self._viewer_cubes and (self._viewer_cube_side is not None or self._viewer_cube_friction is not None):
            raise ValueError('Cube side/friction overrides require TOKENHSI_VIEWER_PUSH_CUBES=1')
        if self._viewer_cube_side is not None:
            self._viewer_cube_side = float(self._viewer_cube_side)
            if not math.isfinite(self._viewer_cube_side) or not 1. <= self._viewer_cube_side <= 1.5:
                raise ValueError('Viewer cube side must be in [1.0, 1.5] metres')
        if self._viewer_cube_friction is not None:
            self._viewer_cube_friction = float(self._viewer_cube_friction)
            if not math.isfinite(self._viewer_cube_friction) or self._viewer_cube_friction < 0.:
                raise ValueError('Viewer cube friction must be finite and nonnegative')
        if self._viewer_cubes:
            args_config = cfg.get('args')
            if (not getattr(args_config, 'test', False)
                    or getattr(args_config, 'headless', True)
                    or os.environ.get('TOKENHSI_GUI_GPU_MODE') != 'native'
                    or getattr(args_config, 'task_graph', None) != 'push_at'
                    or getattr(args_config, 'eval_skills', None) != 'loco'
                    or cfg['env']['numEnvs'] > 16):
                raise ValueError('Viewer cubes require native GUI test, PUSH_AT/loco and <=16 environments')
        self._amp_family_count = 4
        self._solo_cache_class = LegacyRsiCache
        self._push_spec = cfg['env']['pushExtension']
        self._source_push_env = yaml.safe_load(Path(self._push_spec['source_env']).read_text())['env']
        self._source_push_env['motion_file'] = 'tokenhsi/data/dataset_push_native_phase_xy.yaml'
        if cfg['env'].get('_native_geometry_check'):
            if getattr(cfg.get('args'), 'device', None) != 'cpu':
                raise ValueError('Unscreened RSI inspection is CPU-only')
            self._source_push_env['_native_geometry_check'] = True
        self._contact_cfg = dict(push=self._push_spec['physical_push'],
            contact_at=self._push_spec['contact'], push_at=self._push_spec['push_at'])
        super().__init__(cfg, *args, **kwargs)
        self._env_origins.zero_()

    def _build_agent_spawn_offsets(self, device_type, device_id):
        super()._build_agent_spawn_offsets(device_type, device_id)
        self._agent_spawn_offsets[:, :2] = self._agent_spawn_offsets.new_tensor(
            self.cfg['env']['pushScene']['owner_anchor_xy'])

    def _load_box_asset(self):
        size_env = self.cfg['env']
        if self._task_graph_preset != 'random_scenario':
            name = {'push_at':'PUSH_AT','holding_at':'HOLDING_AT',
                    'holding_ontop':'HOLDING_ON_TOP','sit':'SIT','climb':'CLIMB'}.get(self._task_graph_preset)
            if name is not None:
                size_env = deepcopy(size_env)
                size_env['relationGraph']['template_probabilities'] = {
                    key:float(key == name) for key in TEMPLATES}
        self._box_size, self._push_size_probabilities = sample_asset_sizes(
            self.num_envs, size_env, self.device)
        push_owners = self._push_size_probabilities[...,5] > 0
        self._push_asset_mask = torch.cat((push_owners,push_owners),-1)
        if self._viewer_cubes:
            if self._viewer_xy_range is not None:
                low, high = self._viewer_xy_range
                self._box_size[..., :2] = torch.randint(low, high+1,
                    (*self._box_size.shape[:2], 2), device=self.device) / 10.
            elif self._viewer_box_ranges is not None:
                bounds = self._box_size.new_tensor(self._viewer_box_ranges).reshape(3, 2)
                self._box_size = bounds[:, 0] + torch.rand_like(self._box_size) * (bounds[:, 1]-bounds[:, 0])
                # Equal Y/Z ranges: sorting gives uniform samples over the allowed Y>=Z triangle.
                yz = self._box_size[..., 1:].sort(dim=-1, descending=True).values
                self._box_size[..., 1:] = yz
            elif self._viewer_cube_side is not None:
                self._box_size.fill_(self._viewer_cube_side)
            else:
                sides = torch.tensor(self._viewer_cube_sides, device=self.device)
                indices = torch.randint(len(sides), self._box_size.shape[:2], device=self.device)
                self._box_size = sides[indices, None].expand(-1, -1, 3).clone()
            if self._viewer_xy_random:
                self._box_size[..., :2] = torch.randint(10, 16,
                    (*self._box_size.shape[:2], 2), device=self.device) / 10.
            print('[viewer boxes] actual box XYZ (m):', self._box_size.cpu().tolist(), flush=True)
        self._box_scale = self._box_size / self._box_size.new_tensor(self._build_base_size)
        flat = self._box_size.flatten(0, 1)
        push = torch.ones_like(flat[:, 1], dtype=torch.bool) if self._viewer_cubes else self._push_asset_mask.flatten()
        self._box_density = torch.where(push, self.cfg['env']['pushBox']['mass']/flat.prod(-1), 100.)
        self._box_assets = []
        cache = {}
        for index, dims in enumerate(flat.cpu().tolist()):
            density = float(self._box_density[index])
            key = tuple(dims) + (round(density, 6),)
            if key not in cache:
                options = gymapi.AssetOptions()
                options.density = density
                options.angular_damping = .01
                options.linear_damping = .01
                options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
                cache[key] = self.gym.create_box(self.sim, *dims, options)
            self._box_assets.append(cache[key])

    def _build_box(self, env_id, object_id, env_ptr):
        super()._build_box(env_id, object_id, env_ptr)
        handle = self._box_handles[-1]
        if self._viewer_cubes or bool(self._push_asset_mask[env_id, object_id]):
            props = self.gym.get_actor_rigid_shape_properties(env_ptr, handle)
            for prop in props:
                prop.friction = (self._viewer_cube_friction if self._viewer_cubes
                    and self._viewer_cube_friction is not None else self.cfg['env']['pushBox']['friction'])
                prop.restitution = 0.
            self.gym.set_actor_rigid_shape_properties(env_ptr, handle, props)
        if self._viewer_cubes:
            mass = sum(p.mass for p in self.gym.get_actor_rigid_body_properties(env_ptr, handle))
            friction = [p.friction for p in self.gym.get_actor_rigid_shape_properties(env_ptr, handle)]
            print(f'[viewer cubes] env={env_id} box={object_id} mass={mass} friction={friction}', flush=True)
        # Non-overlapping initial actors before the inherited fixed-shelf warmup.
        state = self.gym.get_actor_rigid_body_states(env_ptr, handle, gymapi.STATE_ALL)
        state['pose']['p']['x'][:] = 5. * object_id
        state['pose']['p']['y'][:] = 8.
        self.gym.set_actor_rigid_body_states(env_ptr, handle, state, gymapi.STATE_ALL)

    def _load_motion(self, motion_file):
        from utils.motion_lib import MotionLib
        motion = yaml.safe_load(Path(motion_file).read_text())['motions']
        probabilities = self.cfg['env']['relationGraph']['template_probabilities']
        names = ('loco', 'push') if probabilities['PUSH_AT'] == 1. else tuple(motion)
        if self._push_spec.get('sampling_profile') == 'stage1_square_cone_v1' and 'push_loco' not in names:
            names = tuple(names) + ('push_loco',)
        self._skill_categories = list(motion)
        self._motion_lib = {name: MotionLib(motion_file=motion_file, skill=name,
            dof_body_ids=self._dof_body_ids, dof_offsets=self._dof_offsets,
            key_body_ids=self._key_body_ids.cpu().numpy(), device=self.device) for name in names}

    def _init_relation_runtime(self):
        super()._init_relation_runtime()
        self.relation_runtime = Stage1PushRuntime(self.num_envs, self.relation_runtime.graph,
            self._relation_cfg, self.device, contact_weights=self._push_spec['contact_weights'])
        from utils.native_push_rsi import NativePushBank
        self._push_bank = NativePushBank(self._source_push_env['openPushRSI'],
            self._motion_lib, self._source_push_env)
        self._push_plan = self._box_states.clone()
        self._push_initial_plan = self._box_states.clone()
        self._push_goals = self._tar_pos.clone()
        self._push_generator = torch.Generator(device=self.device).manual_seed(torch.initial_seed())
        self._rsi_interaction = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self._rsi_late = torch.zeros_like(self._rsi_interaction)
        self._push_hits = torch.zeros_like(self._rsi_interaction)
        from utils.stage1_episode_metrics import Stage1EpisodeMetrics
        self._task_episode_metrics = Stage1EpisodeMetrics(self.num_envs, self.num_agents, self.device)

    def _push_goal_distance(self):
        return (self._box_states[:, :2, :2] - self._tar_pos[..., :2]).norm(dim=-1)

    def _reset_relation_history(self, ids):
        super()._reset_relation_history(ids)
        env, agent = self._expand_slots(ids)
        templates = classify_templates(self.relation_runtime.graph, env, agent).reshape(len(ids), self.num_agents)
        self._task_episode_metrics.reset(ids, templates, self._rsi_interaction[ids],
            self._push_goal_distance()[ids], self.relation_runtime.done[ids])

    def _finish_relation_diagnostics(self):
        super()._finish_relation_diagnostics()
        self._task_episode_metrics.finish(self.reset_buf.bool(), self._push_goal_distance(),
                                         self.relation_runtime.done)

    def consume_relation_diagnostics(self):
        result = super().consume_relation_diagnostics()
        result.update(self._task_episode_metrics.consume())
        return result

    def _sample_episode_graph(self, ids):
        weights = self._size_template_probabilities[ids]
        preset = self._task_graph_preset
        if preset == 'random_scenario':
            template = torch.multinomial(weights.flatten(0, 1), 1).reshape(len(ids), 2)
            if not (weights.gather(-1, template[...,None]).squeeze(-1)>0).all():
                raise RuntimeError('Task sampler selected a zero-probability template')
        else:
            name = {'push_at':'PUSH_AT', 'holding_at':'HOLDING_AT',
                    'holding_ontop':'HOLDING_ON_TOP', 'sit':'SIT', 'climb':'CLIMB'}.get(preset)
            if name is None:
                return super()._sample_episode_graph(ids)
            template = torch.full((len(ids),2), TEMPLATES.index(name), device=self.device, dtype=torch.long)
            if not (weights.gather(-1, template[..., None]).squeeze(-1)>0).all():
                raise ValueError('Requested task is incompatible with the sampled physical asset sizes')
        graph = compose_graph(template, shuffle=self._relation_graph_spec['shuffle_edge_order'])
        copy_graph_rows(self.relation_runtime.graph, ids, graph)
        self._edge_goal_owners[ids] = True
        self._ontop_maintain_steps[ids] = 0
        self._sampling_counts[0] += len(ids)
        for index in range(6):
            self._sampling_counts[1+index] += (template == index).sum()
        for index, relation in enumerate((6, 7, 8, 9, 10, PUSH), 8):
            self._sampling_counts[index] += ((graph.edge_relation == relation) & graph.edge_valid).sum()

    def _push_slots(self, ids):
        env, agent = self._expand_slots(ids)
        mask = classify_templates(self.relation_runtime.graph, env, agent) == PUSH_TEMPLATE
        return env[mask], agent[mask]

    def _reset_actors(self, ids):
        from utils.stage1_push_scene import sample_four_box_scene
        scene_env = self._source_push_env
        if self._viewer_box_ranges is not None:
            scene_env = deepcopy(scene_env)
            maximum = self._viewer_box_ranges[1::2]
            scene_env['pushBox']['size'] = maximum
            scene_env['pushScene']['path_separation'] = max(
                scene_env['pushScene']['path_separation'], math.hypot(*maximum[:2]) + .3)
        xy, q, goals = sample_four_box_scene(len(ids), scene_env,
                                             self.device, self._push_generator)
        self._push_plan[ids] = self._initial_box_states[ids]
        self._push_plan[ids, :, :2] = xy
        self._push_plan[ids, :, 3:7] = q
        self._push_plan[ids, :, 2] = self._box_size[ids, :, 2]/2
        self._push_plan[ids, :, 7:] = 0.
        self._push_initial_plan[ids] = self._push_plan[ids]
        self._push_goals[ids, :, :2] = goals
        self._push_goals[ids, :, 2] = self._box_size[ids, :2, 2]/2
        self._rsi_interaction[ids] = False
        self._rsi_late[ids] = False
        return super()._reset_actors(ids)

    def _reset_ref_state_init(self, env, agent):
        push = classify_templates(self.relation_runtime.graph, env, agent) == PUSH_TEMPLATE
        if (~push).any():
            super()._reset_ref_state_init(env[~push], agent[~push])
        e, a = env[push], agent[push]
        if not len(e):
            return
        interaction = torch.rand(len(e), device=self.device) < self.cfg['env']['openPushRSI']['interaction_probability']
        if self._is_eval:
            skills = self.cfg['env']['eval']['skill']
            if skills != ['loco']:
                raise ValueError('PUSH evaluation starts from loco; train RSI is not an eval skill override')
            interaction.zero_()
        for skill, selected in (('loco', ~interaction), ('push', interaction)):
            ee, aa = e[selected], a[selected]
            if not len(ee):
                continue
            lib = self._motion_lib['push_loco' if skill == 'loco' and
                self._push_spec.get('sampling_profile') == 'stage1_square_cone_v1' else skill]
            if not self._is_eval:
                kind=torch.full_like(ee,PUSH_TEMPLATE)
                selected_skill=torch.full_like(ee,self._skill.index(skill))
                self._size_rsi_requested.index_put_((kind,selected_skill),
                    torch.ones_like(ee,dtype=torch.float),accumulate=True)
            if skill == 'push':
                sample = self._push_bank.sample('push', len(ee))
                mids, times = sample['motion_id'].long(), sample['time']
                from utils.stage1_push_scene import advance_rsi_box
                self._push_plan[ee, aa, :3] = advance_rsi_box(
                    self._push_initial_plan[ee, aa, :3], self._push_goals[ee, aa], sample['box_progress'])
                self._rsi_interaction[ee, aa] = True
                self._rsi_late[ee, aa] = sample['late_requested']
            else:
                mids = lib.sample_motions(len(ee))
                times = lib.sample_time_rsi(mids)
            root, root_q, dof, vel, ang, dof_vel, _ = lib.get_motion_state(mids, times)
            body, body_q, body_vel, body_ang = lib.get_motion_state_max(mids, times)
            if skill == 'push':
                from utils.open_push_rsi import align_push_pose
                from utils.native_push_rsi import safe_place
                boxes = self._push_plan[ee, aa]
                targets = push_rsi_hand_targets(boxes[:, None], self._push_goals[ee, aa, None],
                    self._box_size[ee, aa, None], self._contact_cfg)[:, 0]
                body, body_q, body_vel, body_ang, rotation = align_push_pose(
                    body, body_q, body_vel, body_ang, targets,
                    self._key_body_ids[:2].cpu().tolist(), self.cfg['env']['openPushRSI']['stand_off'])
                body = safe_place(body, body_q, boxes, self._box_size[ee, aa], targets, self._push_bank.geometry)
            else:
                angle = torch.rand(len(ee), device=self.device)*2*torch.pi - yaw_from_quat(root_q)
                rotation = torch.zeros_like(root_q)
                rotation[:, 2], rotation[:, 3] = torch.sin(angle/2), torch.cos(angle/2)
                expanded = rotation[:, None].expand_as(body_q)
                target = root.clone()
                target[:, :2] = self._agent_spawn_offsets[aa, :2]
                body = rotate(expanded, body-root[:, None])+target[:, None]
                body_q = multiply(expanded, body_q)
                body_vel, body_ang = rotate(expanded, body_vel), rotate(expanded, body_ang)
            self._set_env_state(ee, aa, body[:, 0], multiply(rotation, root_q), dof,
                rotate(rotation, vel), rotate(rotation, ang), dof_vel)
            self._kinematic_humanoid_rigid_body_states[ee, aa] = torch.cat((body, body_q, body_vel, body_ang), -1)
            self._every_env_init_dof_pos[ee, aa] = dof
            previous = self._reset_ref_slots.get(skill)
            self._reset_ref_slots[skill] = (ee, aa) if previous is None else tuple(torch.cat((p,v)) for p,v in zip(previous,(ee,aa)))
            for mapping, value in ((self._reset_ref_motion_ids,mids),(self._reset_ref_motion_times,times)):
                mapping[skill] = value if skill not in mapping else torch.cat((mapping[skill],value))

    def _reset_boxes(self, ids):
        env, agent = self._push_slots(ids)
        if len(env) != 2*len(ids):
            super()._reset_boxes(ids)
        else:
            self._platform_pos[ids] = self._platform_default_pos[ids]
            self._box_states[ids] = self._push_plan[ids]
        self._box_states[env, agent] = self._push_plan[env, agent]
        # A PUSH owner's additional physical box must stay outside its corridor
        # even when the other owner is assigned a legacy task.
        self._box_states[env, agent+2] = self._push_plan[env, agent+2]

    def _reset_task(self, ids):
        super()._reset_task(ids)
        env, agent = self._push_slots(ids)
        self._tar_pos[env, agent] = self._push_goals[env, agent]

    def _evaluate_ontop_context(self, env_ids=None):
        phi, diagnostic = super()._evaluate_ontop_context(env_ids)
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        graph = select_graph(self.relation_runtime.graph, ids)
        push = graph.edge_relation == PUSH
        owner = graph.edge_owner
        kin = self._rigid_body_pos[ids] if env_ids is None else self._kinematic_humanoid_rigid_body_states[ids, ..., :3]
        hands = kin[:, :, self._key_body_ids[:2]]
        boxes = self._box_states[ids, :2]
        size = self._box_size[ids, :2]
        distance = push_rear_xy(hands, boxes, self._tar_pos[ids], size).amax(-1)
        contact_phi = torch.exp(-self._push_spec['contact']['near_distance_scale']*distance.square())
        root_distance = (kin[:, :, 0, :2]-boxes[..., :2]).norm(dim=-1)
        contact = self._push_spec['contact']
        contact_progress = 1/(1+(root_distance-contact['progress_delta']).clamp_min(0)/contact['progress_sigma'])
        phi = torch.where(push, contact_phi.gather(1, owner), phi)
        diagnostic['progress'] = torch.where(push, contact_progress.gather(1,owner), diagnostic['progress'])
        diagnostic['z_error'] = diagnostic['z_error'].masked_fill(push, 0.)
        diagnostic['distance'] = torch.where(push,distance.gather(1,owner),diagnostic['distance'])
        diagnostic['distance_xy'] = torch.where(push,root_distance.gather(1,owner),diagnostic['distance_xy'])
        return phi, diagnostic

    def _collect_push_contact(self):
        return push_hand_contact(
            self._rigid_body_pos[:, :, self._key_body_ids[:2]],
            self._contact_forces[:, :, self._key_body_ids[:2]], self._box_states[:, :2],
            self._ontop_box_contact[:, :2], self._box_size[:, :2],
            self._push_spec['physical_push'], self._tar_pos, self._contact_cfg)

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._push_hits.zero_()

    def _physics_step(self):
        for _ in range(self.control_freq_inv):
            self.render()
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self._refresh_sim_tensors()
            self._push_hits |= self._collect_push_contact()

    def _compute_reward(self, actions):
        self.relation_runtime.contact.copy_(self._push_hits)
        result = super()._compute_reward(actions)
        self._task_episode_metrics.observe(self._push_goal_distance(), self.relation_runtime.done)
        return result

    def _amp_family(self, env_ids=None, slot_agent=None):
        family = super()._amp_family(env_ids, slot_agent)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs,device=self.device)
        if slot_agent is None:
            slot_agent = torch.arange(2,device=self.device).repeat(len(env_ids))
            env_ids = env_ids.repeat_interleave(2)
        template = classify_templates(self.relation_runtime.graph,env_ids,slot_agent)
        return torch.where(template == PUSH_TEMPLATE, 3, family)

    def _init_amp_obs_ref(self, slots, motion_ids, motion_times, skill_name):
        if skill_name != 'loco' or self._push_spec.get('sampling_profile') != 'stage1_square_cone_v1':
            return super()._init_amp_obs_ref(slots,motion_ids,motion_times,skill_name)
        env,agent = slots
        push = classify_templates(self.relation_runtime.graph,env,agent) == PUSH_TEMPLATE
        for mask,name in ((~push,'loco'),(push,'push_loco')):
            if mask.any():
                super()._init_amp_obs_ref((env[mask],agent[mask]),motion_ids[mask],motion_times[mask],name)

    def _fetch_conditioned_amp_demo(self, num_samples):
        from utils.unified_training import append_family
        p = self.cfg['env']['relationGraph']['template_probabilities']
        family_weights = torch.tensor([p['HOLDING_AT']+p['HOLDING_ON_TOP'], p['SIT'],p['CLIMB'],p['PUSH_AT']],device=self.device)
        family = torch.multinomial(family_weights,num_samples,replacement=True)
        result = torch.empty(num_samples,self.get_num_amp_obs(),device=self.device)
        for value in range(4):
            rows=(family==value).nonzero().flatten()
            if not len(rows):continue
            weights = self.cfg['env']['ampExpertMapping'][('carry','sit','climb','push')[value]]
            selected=torch.multinomial(torch.as_tensor(weights,device=self.device),len(rows),replacement=True)
            for index,name in enumerate(self.cfg['env']['skill']):
                dest=rows[selected==index]
                if not len(dest):continue
                library_name = 'push_loco' if value == 3 and name == 'loco' and self._push_spec.get('sampling_profile') == 'stage1_square_cone_v1' else name
                lib=self._motion_lib[library_name]; mids=lib.sample_motions(len(dest))
                history=self.dt*(self._num_amp_obs_steps-1)
                times=lib.sample_time(mids,truncate_time=history)+history
                obs=self.build_amp_obs_demo(mids,times,lib).reshape(len(dest),self._num_amp_obs_steps,-1)
                result[dest]=append_family(obs,family[dest,None].expand(-1,self._num_amp_obs_steps),4).flatten(1)
        return result
