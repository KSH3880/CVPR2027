from pathlib import Path

from isaacgym import gymapi
from isaacgym.torch_utils import quat_from_angle_axis, quat_mul, quat_rotate
import torch

from env.tasks.multi_agent.humanoid_ma_bridge import HumanoidMABridge
from env.tasks.multi_agent.push_reward import BridgePushRuntime, contact_geometry
from env.tasks.multi_agent.push_rsi import collision_shapes, face_retreat, reference_state
from utils.bridge_spec import SCENES, push_episode_graph, sample_scenes, sample_templates, token_mapping
from utils.push_box_asset import create_plate_box
from utils.push_navigation import approach_distance, sample_box_properties
from utils.stage1_push import PUSH, load_bank
from utils.unified_training import EXPERT_PROBS, append_family

DEPENDENT, PRESENT, INDEPENDENT = range(len(SCENES))
SCENE_METRICS = ('episodes', 'crossed', 'crosser_falls', 'push_success')
Z = (0., 0., 1.)


class HumanoidMABridgePush(HumanoidMABridge):
    """The floor human pushes a cube into the gap (dependent), or the bridge is already in place
    while the floor human does SIT/CLIMB (present) or pushes a decoy cube elsewhere (independent)."""
    _extra_slots = 2

    def __init__(self, cfg, *args, **kwargs):
        self._push_options = cfg['env']['bridge']['push']
        super().__init__(cfg, *args, **kwargs)
        bank = load_bank(self._push_options, cfg['env']['motion_file'])
        self._push_candidates = torch.tensor([[row['motion_id'], row['time']] for row in bank['candidates']['push']],
                                             device=self.device)
        names = self._motion_lib['push']._motions[0].skeleton_tree.node_names
        asset = Path(cfg['env']['asset']['assetRoot']) / cfg['env']['asset']['assetFileName']
        self._push_collision_shapes = collision_shapes(asset, names, self.device)

    def _sample_bridge_geometry(self, n):
        sizes, masses = sample_box_properties(2 * n, self._push_options['box'], self.device)
        self._push_cube_size, self._push_cube_mass = sizes.view(n, 2, 3), masses.view(n, 2)
        side = self._push_cube_size[:, 0]
        b = self._bridge_cfg
        return side[:, 0] + 2 * b['clearance'], side, side[:, 2] + b['top_drop']

    def _create_bridge_assets(self):
        n, p, slot = self.num_envs, self.num_physical_objects, self._bridge_slot
        self._box_size[:, slot + 1] = self._push_cube_size[:, 1]
        self._box_density.view(n, p)[:, slot:] = self._push_cube_mass / self._push_cube_size.prod(-1)
        plate = self._push_options['box']['bottom_plate']
        directory = Path(self.cfg['args'].output_path) / 'push_assets'
        for env in range(n):
            for cube in range(2):
                index = env * p + slot + cube
                options = gymapi.AssetOptions()
                options.angular_damping = options.linear_damping = .01
                options.max_angular_velocity = 100.
                options.density = float(self._box_density[index])
                options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
                self._box_assets[index] = create_plate_box(
                    self.gym, self.sim, self._push_cube_size[env, cube].tolist(),
                    float(self._push_cube_mass[env, cube]), options, plate, directory, index)

    def _build_box(self, env_id, object_id, env_ptr):
        super(HumanoidMABridge, self)._build_box(env_id, object_id, env_ptr)
        if object_id < self._bridge_slot:
            return
        handle = self._box_handles[-1]
        props = self.gym.get_actor_rigid_shape_properties(env_ptr, handle)
        for prop in props:
            prop.friction = self._push_options['box']['friction']
            prop.restitution = 0.
        props[1].friction = self._push_options['box']['bottom_plate']['friction']
        self.gym.set_actor_rigid_shape_properties(env_ptr, handle, props)

    def _build_platform_state_tensors(self):
        super()._build_platform_state_tensors()
        n, m = self.num_envs, self.num_agents
        self._bridge_scene = torch.zeros(n, dtype=torch.long, device=self.device)
        self._bridge_connected = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._pushing = torch.zeros(n, m, dtype=torch.bool, device=self.device)
        self._push_rsi = torch.zeros_like(self._pushing)
        self._push_hits = torch.zeros_like(self._pushing)
        self._push_payload = torch.full((n, m), self._bridge_slot, dtype=torch.long, device=self.device)
        self._push_goal = torch.zeros(n, 3, device=self.device)
        self._push_motion_ids = torch.zeros(n, m, dtype=torch.long, device=self.device)
        self._push_motion_times = torch.zeros(n, m, device=self.device)
        self._scene_metrics = torch.zeros(len(SCENES), len(SCENE_METRICS), device=self.device)

    def _init_ontop_context_runtime(self):
        super()._init_ontop_context_runtime()
        self.relation_runtime = BridgePushRuntime(self, self.relation_runtime)

    def _sample_episode_graph(self, ids):
        b, k, m, slot = self._bridge_cfg, len(ids), self.num_agents, self._bridge_slot
        rows = torch.arange(k, device=self.device)
        scene = sample_scenes(k, self._task_graph_preset, b['scene_probabilities'], self.device)
        crosser = torch.randint(m, (k,), device=self.device)
        pusher = 1 - crosser
        pushing = scene != PRESENT
        templates = sample_templates(k, 'random_scenario', b['task_probabilities'], self.device)
        templates[rows, pusher] = torch.where(pushing, 3, templates[rows, pusher])
        people = torch.arange(m, device=self.device)[None]
        primary = people + templates.clamp_max(2) * m
        payload = torch.where(scene == DEPENDENT, slot, slot + 1)
        primary[rows, pusher] = torch.where(pushing, payload, primary[rows, pusher])
        other_cube = torch.where(scene == DEPENDENT, slot + 1, slot)
        self._logical_box_order[ids] = token_mapping(primary, other_cube, 4 * m)
        self._templates[ids], self._agent_box_assignment[ids] = templates, primary
        self._task_token_count[ids] = m
        self._bridge_scene[ids] = scene
        self._pushing[ids] = (people == pusher[:, None]) & pushing[:, None]
        self._crosser[ids] = people == crosser[:, None]
        self._floor_start[ids] = people == pusher[:, None]
        self._push_payload[ids] = torch.where(self._pushing[ids], payload[:, None], slot)
        self._before_distance_modes[ids], self._before_near_pairs[ids] = 0, False
        graph, self._before_predecessor[ids] = push_episode_graph(templates, crosser, scene, self.num_objects)
        self._finish_episode_graph(ids, graph, templates)

    def _cube_state(self, xy, z, toward):
        # Cube resting on the floor whose local +Y (pushing direction) points along world +/-y.
        state = torch.zeros(len(xy), 13, device=self.device)
        state[:, :2], state[:, 2] = xy, z
        yaw = torch.where(toward > 0, 0., torch.pi)
        state[:, 3:7] = quat_from_angle_axis(yaw, torch.tensor(Z, device=self.device).expand(len(xy), -1))
        return state

    def _reset_boxes(self, ids):
        self._reset_all_boxes_random_arena(ids)
        self._platform_pos[ids] = self._platform_default_pos[ids]
        b, k, m, slot = self._bridge_cfg, len(ids), self.num_agents, self._bridge_slot
        rows = torch.arange(k, device=self.device)
        origin = self._env_origins[ids]
        scene = self._bridge_scene[ids]
        crosser = self._crosser[ids].long().argmax(-1)
        pusher = 1 - crosser
        mapping = self._logical_box_order[ids]
        tail = mapping[:, m:]
        distractor = torch.where(tail[:, 0] < slot, tail[:, 0], tail[:, 1])
        half = self._bridge_size[ids] / 2
        layout, state = self._far_stage_layout(ids, torch.stack((mapping[rows, crosser], distractor), -1), half)

        # The bridge enters the channel from the end nearer to its landing site.
        landing = layout['landing']
        offset = layout['far_stage'][:, 1]
        half_a, half_b = self._stage_half_size[ids, 1], self._far_stage_half_size[ids, 1]
        ends = torch.stack((torch.maximum(half_a, offset + half_b), torch.maximum(half_a, half_b - offset)), -1)
        approach = self._uniform(b['approach_distance'], k)
        distance = ends + half[:, 1:2] + approach[:, None] - torch.stack((landing[:, 1], -landing[:, 1]), -1)
        side = distance.argmin(-1)
        sign = torch.where(side == 0, 1., -1.)
        end = ends[rows, side]
        start = torch.stack((landing[:, 0], sign * (end + half[:, 1] + approach)), -1)

        goal = torch.cat((landing + origin[:, :2], (origin[:, 2] + half[:, 2])[:, None]), -1)
        dependent, present, independent = scene == DEPENDENT, scene == PRESENT, scene == INDEPENDENT
        bridge_xy = torch.where(dependent[:, None], start, landing) + origin[:, :2]
        self._bridge_states[ids] = self._cube_state(bridge_xy, goal[:, 2], torch.where(dependent, -sign, sign))
        self._bridge_goal[ids], self._bridge_start[ids] = goal, torch.cat((bridge_xy, goal[:, 2:]), -1)
        self._bridge_present[ids] = ~dependent
        self._bridge_connected[ids] = ~dependent
        self._bridge_wait[ids], self._bridge_duration[ids] = 0., 1.

        # Floor items (decoy cube or the floor task box) beside the channel, beyond the stage ends.
        lateral = torch.where(torch.rand(k, device=self.device) < .5, 1., -1.) * self._uniform(b['floor_offset'], k)
        floor_xy = torch.stack((landing[:, 0] + lateral, sign * (end + self._uniform(b['decoy_distance'], k))), -1)
        floor_xy = floor_xy + origin[:, :2]
        decoy = self._cube_state(floor_xy, origin[:, 2] + self._box_size[ids, slot + 1, 2] / 2, sign)
        keep = dependent | independent
        self._box_states[ids[keep], slot + 1] = decoy[keep]
        self._push_goal[ids] = torch.where(dependent[:, None], goal, torch.cat(
            (floor_xy + sign[:, None] * torch.stack((torch.zeros(k, device=self.device),
             self._uniform(b['decoy_goal_distance'], k)), -1), decoy[:, 2:3]), -1))
        floor_box = mapping[rows, pusher]
        size = self._box_size[ids[present], floor_box[present]]
        self._box_states[ids[present], floor_box[present]] = self._cube_state(
            floor_xy[present], origin[present, 2] + size[:, 2] / 2, sign[present])

        # Humans: the crosser on stage A, the floor human beside its object.
        xy = torch.zeros(k, m, 2, device=self.device)
        target = torch.zeros_like(xy)
        lift = torch.zeros(k, m, device=self.device)
        xy[rows, crosser] = layout['humans'][:, 0] + origin[:, :2]
        target[rows, crosser] = state[:, 0, :2]
        lift[rows, crosser] = self._stage_height[ids]
        away = torch.stack((torch.zeros(k, device=self.device), sign * self._uniform(b['floor_task_distance'], k)), -1)
        xy[rows, pusher] = torch.where(dependent[:, None], bridge_xy, floor_xy) + away
        target[rows, pusher] = torch.where(dependent[:, None], bridge_xy, floor_xy)
        self._place_humans(ids, xy, self._face(xy, target), lift)
        self._push_rsi[ids] = self._pushing[ids] & (torch.rand(k, 1, device=self.device) < self._push_options['rsi_probability'])
        self._push_hits[ids] = False
        self._start_push_rsi(ids)

    def _uniform(self, bounds, n):
        return bounds[0] + (bounds[1] - bounds[0]) * torch.rand(n, device=self.device)

    def _start_push_rsi(self, ids):
        rows, human = self._push_rsi[ids].nonzero(as_tuple=True)
        env = ids[rows]
        slots = self._reset_ref_slots.get('loco')
        if slots is not None:
            keep = ~self._push_rsi[slots]
            self._reset_ref_slots['loco'] = tuple(value[keep] for value in slots)
            for store in (self._reset_ref_motion_ids, self._reset_ref_motion_times):
                store['loco'] = store['loco'][keep]
        if not len(env):
            return
        options = self._push_options
        clips, times = self._sample_push_rsi(len(env))
        p, rot, vel, ang, dof, dof_vel = reference_state(self._motion_lib['push'], clips, times)
        box = self._box_states[env, self._push_payload[env, human]]
        size = self._box_size[env, self._push_payload[env, human]]
        direction = quat_rotate(box[:, 3:7], torch.tensor([0., 1., 0.], device=self.device).expand(len(env), -1))
        hands = p[:, self._key_body_ids[:2]]
        forward = hands.mean(1) - p[:, 0]
        angle = torch.atan2(direction[:, 1], direction[:, 0]) - torch.atan2(forward[:, 1], forward[:, 0])
        turn = quat_from_angle_axis(angle, torch.tensor(Z, device=self.device).expand(len(env), -1))
        turns = turn[:, None].expand(-1, self.num_bodies, -1).reshape(-1, 4)
        rotated = quat_rotate(turns, p.reshape(-1, 3)).reshape_as(p)
        point = box[:, :3] - (size[:, 1] / 2)[:, None] * direction
        shift = point - direction * options['rsi']['stand_off'] - rotated[:, self._key_body_ids[:2]].mean(1)
        shift[:, 2] = 0
        p = rotated + shift[:, None]
        rot = quat_mul(turns, rot.reshape(-1, 4)).reshape_as(rot)
        retreat = face_retreat(p, rot, point, direction, self._push_collision_shapes, options['rsi']['clearance'])
        p -= retreat[:, None, None] * direction[:, None]
        vel = quat_rotate(turns, vel.reshape(-1, 3)).reshape_as(vel)
        ang = quat_rotate(turns, ang.reshape(-1, 3)).reshape_as(ang)
        self._set_env_state(env, human, p[:, 0], rot[:, 0], dof, vel[:, 0], ang[:, 0], dof_vel)
        self._kinematic_humanoid_rigid_body_states[env, human] = torch.cat((p, rot, vel, ang), -1)
        self._every_env_init_dof_pos[env, human] = dof
        self._push_motion_ids[env, human], self._push_motion_times[env, human] = clips, times

    def _sample_push_rsi(self, count):
        records = self._push_candidates
        clips = records[:, 0].long().unique()
        weights = self._motion_lib['push']._motion_weights[clips]
        chosen = clips[torch.multinomial(weights, count, replacement=True)]
        late = torch.rand(count, device=self.device) < self._push_options['rsi']['late_probability']
        times = torch.empty(count, device=self.device)
        for index, (clip, use_late) in enumerate(zip(chosen.tolist(), late.tolist())):
            pool = records[records[:, 0] == clip, 1].sort().values
            if use_late:
                pool = pool[-max(1, (len(pool) * 3 + 9) // 10):]
            times[index] = pool[torch.randint(len(pool), (), device=self.device)]
        return chosen, times

    def _init_amp_obs(self, env_ids):
        super()._init_amp_obs(env_ids)
        rows, human = self._push_rsi[env_ids].nonzero(as_tuple=True)
        if len(rows):
            env = env_ids[rows]
            self._init_amp_obs_ref((env, human), self._push_motion_ids[env, human],
                                   self._push_motion_times[env, human], 'push')

    def _reset_task(self, ids):
        super()._reset_task(ids)
        rows, human = self._pushing[ids].nonzero(as_tuple=True)
        self._tar_pos[ids[rows], human] = self._push_goal[ids[rows]]

    def _amp_family(self, env_ids=None, slot_agent=None):
        family = super()._amp_family(env_ids, slot_agent)
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        if slot_agent is None:
            ids, slot_agent = ids.repeat_interleave(self.num_agents), torch.arange(self.num_agents, device=self.device).repeat(len(ids))
        return torch.where(self._pushing[ids, slot_agent], 3, family)

    def _fetch_conditioned_amp_demo(self, count):
        probabilities = torch.tensor(self._push_options['demo_family_probabilities'], device=self.device)
        family = torch.multinomial(probabilities, count, replacement=True)
        skills = list(self.cfg['env']['skill']) + ['push', 'push_loco']
        weights = torch.zeros(4, len(skills), device=self.device)
        weights[:3, :len(skills) - 2] = torch.tensor(EXPERT_PROBS, device=self.device)
        weights[3, -2], weights[3, -1] = .67, .33
        chosen = torch.multinomial(weights[family], 1).flatten()
        result = torch.empty(count, self.get_num_amp_obs(), device=self.device)
        steps = self._num_amp_obs_steps
        for index, name in enumerate(skills):
            selected = (chosen == index).nonzero().flatten()
            if not len(selected):
                continue
            lib = self._motion_lib[name]
            clips = lib.sample_motions(len(selected))
            truncate = self.dt * (steps - 1)
            times = lib.sample_time(clips, truncate_time=truncate) + truncate
            obs = self.build_amp_obs_demo(clips, times, lib).reshape(len(selected), steps, -1)
            result[selected] = append_family(obs, family[selected, None].expand(-1, steps), 4).flatten(1)
        return result

    def _push_state(self, ids=None):
        ids = torch.arange(self.num_envs, device=self.device) if ids is None else ids
        payload = self._push_payload[ids]
        box = self._box_states[ids[:, None], payload]
        size = self._box_size[ids[:, None], payload]
        hands = self._rigid_body_pos[ids][:, :, self._key_body_ids[:2]]
        forces = self._contact_forces[ids][:, :, self._key_body_ids[:2]]
        phi, hit, point, normal = contact_geometry(hands, box, size, forces, self._ontop_box_contact[ids[:, None], payload])
        diff = box[..., :3] - self._tar_pos[ids]
        roots = self._humanoid_root_states[ids, :, :3]
        distance = approach_distance(roots, box[..., :3], normal, size / 2, self._push_options['approach_stand_off'],
                                     self._push_options['approach_clearance'])
        return dict(phi_contact=phi, hit=hit, phi_at=torch.exp(-10 * diff.square().sum(-1)), z_error=diff[..., 2],
                    distance=diff.norm(dim=-1), progress_contact=1 / (1 + distance),
                    progress_at=1 / (1 + diff[..., :2].norm(dim=-1).sub(.1).clamp_min(0)))

    def _evaluate_ontop_context(self, env_ids=None):
        phi, diag = super()._evaluate_ontop_context(env_ids)
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        graph = self.relation_runtime.graph
        owner = graph.edge_owner[ids]
        primary = graph.edge_relation[ids] == PUSH
        mask = self._pushing[ids].gather(1, owner) & graph.edge_valid[ids]
        data = self._push_state(ids)

        def pick(contact, at):
            return torch.where(primary, data[contact].gather(1, owner), data[at].gather(1, owner))

        phi = torch.where(mask, pick('phi_contact', 'phi_at'), phi)
        diag['progress'] = torch.where(mask, pick('progress_contact', 'progress_at'), diag['progress'])
        diag['z_error'] = torch.where(mask, data['z_error'].gather(1, owner), diag['z_error'])
        diag['distance'] = torch.where(mask, data['distance'].gather(1, owner), diag['distance'])
        return phi, diag

    def _compute_reward(self, actions):
        super()._compute_reward(actions)
        # PUSH moves its own box on purpose: no box speed penalty for the pusher (as in PUSH Stage1).
        terms = self.extras['reward_terms'].view(self.num_envs, self.num_agents, -1)
        correction = torch.zeros_like(terms)
        correction[..., -1] = correction[..., 5] = -terms[..., 5] * self._pushing
        terms += correction
        self.rew_buf.copy_(terms[..., -1].flatten())
        self._reward_term_sums += correction.sum((0, 1))

    def pre_physics_step(self, actions):
        super(HumanoidMABridge, self).pre_physics_step(actions)
        self._push_hits.zero_()

    def _physics_step(self):
        for _ in range(self.control_freq_inv):
            self.render()
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self._refresh_sim_tensors()
            self._push_hits |= self._push_state()['hit'] & self._pushing

    def _bridge_arrived(self):
        near = (self._bridge_states[:, :2] - self._bridge_goal[:, :2]).norm(dim=-1) < .2
        self._bridge_connected |= near
        return self._bridge_connected

    def _record_bridge_metrics(self, fallen):
        super()._record_bridge_metrics(fallen)
        ending = self.reset_buf.bool()
        rows = torch.arange(self.num_envs, device=self.device)
        pusher = (~self._crosser).long().argmax(-1)
        payload = self._box_states[rows, self._push_payload[rows, pusher], :2]
        pushed = ((payload - self._push_goal[:, :2]).norm(dim=-1) < .2) & (self._bridge_scene != PRESENT)
        crossed = (self._bridge_crossed & self._crosser).any(-1)
        falls = (fallen & self._crosser).any(-1)
        for scene in range(len(SCENES)):
            selected = self._bridge_scene == scene
            self._scene_metrics[scene] += torch.stack(((ending & selected).sum(), (ending & selected & crossed).sum(),
                (falls & selected).sum(), (ending & selected & pushed).sum())).float()

    def _consume_sampling_diagnostics(self):
        result = super()._consume_sampling_diagnostics()
        for scene, name in enumerate(SCENES):
            episodes = self._scene_metrics[scene, 0]
            for index, key in enumerate(SCENE_METRICS[1:], 1):
                result['bridge/{}_{}'.format(name, key)] = self._scene_metrics[scene, index] / episodes.clamp_min(1)
            result['bridge/{}_episodes'.format(name)] = episodes.clone()
        self._scene_metrics.zero_()
        return result
