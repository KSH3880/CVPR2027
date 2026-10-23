"""Evaluation-only subclass and frozen actor. Import after isaacgym."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from isaacgym import gymapi
from isaacgym.torch_utils import quat_rotate, quat_mul, quat_from_angle_axis
from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry

AT_SIZES = [.22, .27, .32, .37, .42, .47, .52, .57]


def make_sizes(task, count, seed):
    rng = np.random.default_rng(seed)
    if task == 'at':
        choices = np.array([[s, s, s] for s in AT_SIZES]+[[.30, .30, .40]])
        if count < len(choices):
            ids = rng.choice(len(choices), count, replace=False)
        else:
            ids = np.r_[np.arange(len(choices)), rng.integers(0, len(choices), count-len(choices))]
            rng.shuffle(ids)
        return choices[ids, None].astype(np.float32)
    ranges = {'sit': [(.25, .80), (.25, .80), (.35, .50)],
              'climb': [(.40, .80), (.40, .80), (.20, .70)],
              'ontop': [(.20, .60)]*3}
    if task not in ranges:
        raise ValueError(task)
    sources = np.stack([rng.integers(round(lo*20), round(hi*20)+1, count)/20 for lo, hi in ranges[task]], -1)
    if task != 'ontop':
        return sources[:, None].astype(np.float32)
    supports = np.stack([rng.integers(round(lo*20), round(hi*20)+1, count)/20
                         for lo, hi in ((.40, .80), (.40, .80), (.25, .50))], -1)
    return np.stack((sources, supports), 1).astype(np.float32)


def task_class(options, families, push_spec):
    class SingleMatchTask(HumanoidMACarry):
        def __init__(self, *args, **kwargs):
            self._amp_num_families = families
            self._single_push = options.task == 'push'
            super().__init__(*args, **kwargs)
            if self._single_push and options.start == 'rsi':
                from env.tasks.multi_agent.push_rsi import EvalPushMotionLib, collision_shapes
                from utils.stage1_push import load_bank
                bank = load_bank(push_spec, 'tokenhsi/data/dataset_stage1_push.yaml')
                source_ids = sorted({v['motion_id'] for v in bank['candidates']['push']})
                self._single_push_lib = EvalPushMotionLib(source_ids, 'tokenhsi/data/dataset_stage1_push.yaml',
                    'push', self._dof_body_ids, self._dof_offsets, self._key_body_ids.cpu().numpy(), self.device)
                self._single_candidates = torch.tensor([[source_ids.index(v['motion_id']), v['time'], v['box_progress']]
                    for v in bank['candidates']['push']], device=self.device)
                names = self._single_push_lib._motions[0].skeleton_tree.node_names
                self._single_shapes = collision_shapes('tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml', names, self.device)

        def _load_box_asset(self):
            n, o = self.num_envs, self.num_objects
            if self._single_push:
                from utils.push_navigation import sample_box_properties
                sizes, masses = sample_box_properties(n, push_spec['box'], self.device)
                self._box_size = sizes[:, None]
                self._box_density = masses/sizes.prod(-1)
                # The simulator uses PUSH geometry; old-task eligibility is only an
                # internal graph scaffold. Policy receives its actual PUSH token.
                self._rsi_asset_sizes = torch.full_like(self._box_size, .4)
            else:
                self._box_size = torch.tensor(make_sizes(options.task, n, options.seed), device=self.device)
                self._box_density = torch.full((n*o,), 100., device=self.device)
            self._box_scale = self._box_size/torch.tensor(self._build_base_size, device=self.device)
            self._box_assets = []
            for size, density in zip(self._box_size.reshape(-1, 3).cpu().tolist(), self._box_density.cpu().tolist()):
                asset = gymapi.AssetOptions()
                asset.angular_damping = asset.linear_damping = .01
                asset.max_angular_velocity = 100.
                asset.density = density
                asset.default_dof_drive_mode = gymapi.DOF_MODE_NONE
                asset.fix_base_link = options.task in ('sit', 'climb')
                self._box_assets.append(self.gym.create_box(self.sim, *size, asset))

        def _build_box(self, env_id, object_id, env_ptr):
            super()._build_box(env_id, object_id, env_ptr)
            if self._single_push:
                handle = self._box_handles[-1]
                props = self.gym.get_actor_rigid_shape_properties(env_ptr, handle)
                for prop in props:
                    prop.friction = push_spec['box']['friction']
                self.gym.set_actor_rigid_shape_properties(env_ptr, handle, props)

        def _ontop_scene_infeasible(self, ids):
            # The protocol specifies center-distance rejection only. Do not add
            # the training scene's extra bbox/goal/RSI rejection constraints.
            return torch.zeros(len(ids), dtype=torch.bool, device=self.device)

        def _configure_ontop_targets(self, ids):
            super()._configure_ontop_targets(ids)
            if options.task == 'ontop':
                angle = torch.rand(len(ids), device=self.device)*2*torch.pi
                self._box_states[ids, 1, 3:7] = quat_from_angle_axis(angle,
                    torch.tensor([0., 0., 1.], device=self.device).expand(len(ids), -1))

        def _compute_observations(self, env_ids=None):
            result = super()._compute_observations(env_ids)
            if self._single_push:
                ids = slice(None) if env_ids is None else env_ids
                self.obs_buf[ids, -4] = 4  # valid, PUSH, actor, payload, target
            return result

        def _reset_ontop_context_envs(self, ids):
            if not self._single_push:
                return super()._reset_ontop_context_envs(ids)
            if not len(ids):
                return
            self._sample_episode_graph(ids)
            self._reset_ref_slots, self._reset_ref_motion_ids, self._reset_ref_motion_times = {}, {}, {}
            self._reset_default_slots = None
            self._reset_actors(ids)
            self._sample_box_assignments(ids)
            self._platform_pos[ids] = self._platform_default_pos[ids]
            self._tar_platform_pos[ids] = self._tar_platform_default_pos[ids]
            k = len(ids)
            axis = torch.tensor([0., 0., 1.], device=self.device).expand(k, -1)
            angle = torch.rand(k, device=self.device)*2*torch.pi
            q = quat_from_angle_axis(angle, axis)
            normal = quat_rotate(q, torch.tensor([0., 1., 0.], device=self.device).expand(k, -1))
            box = self._box_states[ids, 0].clone()
            box.zero_(); box[:, :2] = self._env_origins[ids, :2]
            box[:, 2] = self._box_size[ids, 0, 2]/2
            box[:, 3:7] = q
            self._box_states[ids, 0] = box
            distance = options.distance_min+(options.distance_max-options.distance_min)*torch.rand(k, device=self.device)
            self._tar_pos[ids, 0] = box[:, :3]+distance[:, None]*normal
            self._single_zone_yaw = getattr(self, '_single_zone_yaw', torch.zeros(self.num_envs, device=self.device))
            self._single_zone_yaw[ids] = angle
            self._place_push_human(ids, box, normal)
            self._reset_env_tensors(ids)
            self._refresh_sim_tensors()
            self._reset_relation_history(ids)
            self._compute_observations(ids)
            self._compute_amp_observations(ids)
            self._hist_amp_obs_buf[ids] = self._curr_amp_obs_buf[ids, :, None]

        def _place_push_human(self, ids, box, normal):
            from env.tasks.multi_agent.push_rsi import reference_state, face_retreat
            from utils.torch_utils import calc_heading
            k = len(ids)
            owner = torch.zeros(k, dtype=torch.long, device=self.device)
            if options.start == 'rsi':
                records = self._single_candidates
                clips = records[:, 0].long().unique()
                chosen = clips[torch.multinomial(self._single_push_lib._motion_weights[clips], k, replacement=True)]
                late = torch.rand(k, device=self.device) < push_spec['rsi']['late_probability']
                sampled = []
                for clip, use_late in zip(chosen.tolist(), late.tolist()):
                    pool = records[records[:, 0] == clip]
                    pool = pool[pool[:, 1].argsort()]
                    if use_late:
                        pool = pool[-max(1, (len(pool)*3+9)//10):]
                    sampled.append(pool[torch.randint(len(pool), (), device=self.device)])
                sampled = torch.stack(sampled)
                p, rot, vel, ang, dof, dv = reference_state(self._single_push_lib, sampled[:, 0].long(), sampled[:, 1],
                    self._dof_body_ids, self._dof_offsets, correct_hand_height=False)
                forward = p[:, self._key_body_ids[:2]].mean(1)-p[:, 0]
                turn_angle = torch.atan2(normal[:, 1], normal[:, 0])-torch.atan2(forward[:, 1], forward[:, 0])
                target = box[:, :3]-(self._box_size[ids, 0, 1]/2+push_spec['rsi']['stand_off'])[:, None]*normal
            else:
                # Reuse the native loco reference pose/velocities selected above.
                state = self._kinematic_humanoid_rigid_body_states[ids, 0].clone()
                p, rot, vel, ang = state[..., :3], state[..., 3:7], state[..., 7:10], state[..., 10:13]
                dofs = self._dof_state.view(self.num_envs, 1, self.num_dof, 2)[ids, 0]
                dof, dv = dofs[..., 0].clone(), dofs[..., 1].clone()
                pending = torch.ones(k, dtype=torch.bool, device=self.device)
                target = box[:, :3].clone()
                for _ in range(256):
                    radius = 1.65+(6.-1.65)*torch.rand(k, device=self.device)
                    azimuth = torch.rand(k, device=self.device)*2*torch.pi
                    delta = radius[:, None]*torch.stack((azimuth.cos(), azimuth.sin()), -1)
                    candidate = box[:, :2]+delta
                    local = torch.stack(((delta*normal[:, :2]).sum(-1),
                        delta[:, 0]*normal[:, 1]-delta[:, 1]*normal[:, 0]), -1)
                    half = self._box_size[ids, 0][:, [1, 0]]/2
                    gap = (local.abs()-half).clamp_min(0).norm(dim=-1)
                    accept = pending & (gap >= .45)
                    target[accept, :2] = candidate[accept]
                    pending &= ~accept
                    if not pending.any():
                        break
                if pending.any():
                    raise RuntimeError('PUSH loco spawn rejection exhausted')
                toward = box[:, :2]-target[:, :2]
                turn_angle = torch.atan2(toward[:, 1], toward[:, 0])-calc_heading(rot[:, 0])
            turn = quat_from_angle_axis(turn_angle, torch.tensor([0., 0., 1.], device=self.device).expand(k, -1))
            turns = turn[:, None].expand(-1, self.num_bodies, -1).reshape(-1, 4)
            p = quat_rotate(turns, p.reshape(-1, 3)).reshape_as(p)
            anchor = p[:, self._key_body_ids[:2]].mean(1) if options.start == 'rsi' else p[:, 0]
            shift = target-anchor; shift[:, 2] = 0
            p += shift[:, None]
            rot = quat_mul(turns, rot.reshape(-1, 4)).reshape_as(rot)
            if options.start == 'rsi':
                face = box[:, :3]-(self._box_size[ids, 0, 1]/2)[:, None]*normal
                retreat = face_retreat(p, rot, face, normal, self._single_shapes, push_spec['rsi']['clearance'])
                p -= retreat[:, None, None]*normal[:, None]
            vel = quat_rotate(turns, vel.reshape(-1, 3)).reshape_as(vel)
            ang = quat_rotate(turns, ang.reshape(-1, 3)).reshape_as(ang)
            self._set_env_state(ids, owner, p[:, 0], rot[:, 0], dof, vel[:, 0], ang[:, 0], dv)
            self._kinematic_humanoid_rigid_body_states[ids, 0] = torch.cat((p, rot, vel, ang), -1)
            self._every_env_init_dof_pos[ids, 0] = dof

    return SingleMatchTask


def build_frozen_actor(task, weights, config, push):
    from learning.multi_agent.amp_network_builder_ma import AMPMultiAgentBuilder
    from learning.multi_agent.stage1_push_network import Stage1PushBuilder
    from learning.multi_agent.scene_normalizer import SceneRunningMeanStd
    builder = Stage1PushBuilder() if push else AMPMultiAgentBuilder()
    builder.load(config['params']['network'])
    model = torch.nn.Module()
    model.a2c_network = builder.build('amp', actions_num=32, input_shape=(task.get_obs_size(),),
        amp_input_shape=(task.get_num_amp_obs(),), value_size=1,
        num_agents=1, num_objects=task.num_objects, observation_mode=task.get_policy_obs_mode(),
        scene_entity_sizes=task.get_scene_entity_sizes(), scene_kinematic_size=task.get_scene_kinematic_size(),
        scene_arena_scale=task.get_scene_arena_scale(), relation_reward_mode=task._relation_cfg['mode'],
        relation_graph_spec=task._relation_graph_spec, device=task.device)
    model.to(task.device).load_state_dict(weights['model'], strict=True)
    normalizer = SceneRunningMeanStd(task.get_scene_entity_sizes(), [1, task.num_objects, 1],
        task.get_scene_normalized_entity_sizes(), task.get_scene_kinematic_size(), task.get_relation_suffix_size()).to(task.device)
    normalizer.load_state_dict(weights['running_mean_std'], strict=True)
    model.eval(); normalizer.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    def act(obs):
        mu, _ = model.a2c_network.eval_actor(normalizer(obs))
        if not torch.isfinite(mu).all():
            raise RuntimeError('Non-finite policy action')
        return mu.clamp(-1, 1)
    return act
