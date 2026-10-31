"""TokenHSI five-task interface, with PUSH semantics from Stage1 26k.

The baseline remains one humanoid per environment. This is per-owner PUSH
parity, not parity with the two-human/four-object benchmark scene.
"""
import os
from pathlib import Path
import torch
from isaacgym import gymapi
from isaacgym import gymtorch
from env.tasks.multi_task.push_integration import HumanoidTrajSitCarryClimbPush
from env.tasks.multi_task.humanoid_traj_sit_carry_climb import HumanoidTrajSitCarryClimb
from utils.jointpush_rsi import PushBank
from utils.jointpush_rsi_geometry import geometry, align_push_pose, safe_place
from utils.jointpush_geometry import rotate, multiply, push_rsi_hand_targets
from utils.push_reward import contact, reward


class HumanoidTokenHSIPushKD(HumanoidTrajSitCarryClimbPush):
    def post_process_disc_dataset_collection(self, cfg):
        # Native code removes skills absent from eval_task. Our discriminator
        # expert is always PUSH-only and is not needed for old-task evaluation.
        return

    def register_task_push_pre_init(self, cfg):
        self._push_cfg = cfg['env']['push']
        self._joint = cfg['env']['jointPush']
        self._push_episode_length = self._push_cfg['episodeLength']
        self._num_tasks += 1
        self._each_subtask_obs_size.append(42)
        self._multiple_task_names.append('push')

    def register_task_push_post_init(self, cfg):
        self._push_skill = ['loco_push', 'push']
        self._push_skill_init_prob = torch.tensor([.5, .5], device=self.device)
        self._push_park_pos = torch.tensor([-40., 0., 0.], device=self.device)
        roots = self._root_states.view(self.num_envs, self.get_num_actors_per_env(), -1)
        self._push_box_states = roots[:, self._push_box_handles[0]]
        self._push_marker_states = roots[:, self._push_marker_handles[0]]
        self._push_box_actor_ids = self._humanoid_actor_ids+self._push_box_handles[0]
        self._push_marker_actor_ids = self._humanoid_actor_ids+self._push_marker_handles[0]
        forces = gymtorch.wrap_tensor(self.gym.acquire_net_contact_force_tensor(self.sim))
        index = self.gym.find_actor_rigid_body_index(self.envs[0], self._push_box_handles[0], 'box', gymapi.DOMAIN_ENV)
        self._push_box_contact = forces.view(self.num_envs, -1, 3)[:, index]
        self._push_hand_ids = self._key_body_ids[:2]
        self._push_tar_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._push_contact_hit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._push_success = torch.zeros_like(self._push_contact_hit)
        self._push_ever = torch.zeros_like(self._push_contact_hit)
        self._push_cand_idx = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._bank = PushBank(cfg['env'], self._motion_lib)
        asset = Path(cfg['env']['asset']['assetRoot'])/cfg['env']['asset']['assetFileName']
        self._push_geometry = geometry(asset, self._motion_lib['push']._motions[0].skeleton_tree.node_names, self.device)

    def _create_envs(self, num_envs, spacing, num_per_row):
        profiles = torch.tensor(self._joint['box']['profiles'], device=self.device)
        self._push_box_size = profiles[torch.randint(len(profiles), (num_envs,), device=self.device)]
        self._push_box_mass = self._push_box_size.prod(-1)*self._joint['box']['density']
        self._push_box_bps = self._box_corners(self._push_box_size)
        self._push_box_handles, self._push_marker_handles = [], []
        self._push_box_assets = []
        for size, mass in zip(self._push_box_size.cpu().tolist(), self._push_box_mass.cpu().tolist()):
            opt = gymapi.AssetOptions()
            opt.density = mass/(size[0]*size[1]*size[2])
            opt.linear_damping = .01
            opt.angular_damping = .01
            self._push_box_assets.append(self.gym.create_box(self.sim, *size, opt))
        opt = gymapi.AssetOptions()
        opt.fix_base_link = True
        self._push_marker_asset = self.gym.load_asset(self.sim, 'tokenhsi/data/assets/mjcf', 'location_marker.urdf', opt)
        HumanoidTrajSitCarryClimb._create_envs(self, num_envs, spacing, num_per_row)

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        HumanoidTrajSitCarryClimb._build_env(self, env_id, env_ptr, humanoid_asset)
        pose = gymapi.Transform()
        pose.p.x = -40.
        pose.p.z = float(self._push_box_size[env_id, 2])/2
        actor = self.gym.create_actor(env_ptr, self._push_box_assets[env_id], pose, 'push_box', env_id, 0, 0)
        shapes = self.gym.get_actor_rigid_shape_properties(env_ptr, actor)
        for s in shapes:
            s.friction, s.restitution = self._joint['box']['friction'], 0.
        self.gym.set_actor_rigid_shape_properties(env_ptr, actor, shapes)
        self._push_box_handles.append(actor)
        pose.p.z = -5.
        self._push_marker_handles.append(self.gym.create_actor(env_ptr, self._push_marker_asset, pose, 'push_marker', self.num_envs+10, 1, 0))

    def _reset_task_indicator(self, env_ids):
        if self._is_eval:
            if self._eval_task == 'traj':
                raise ValueError('TRAJ is excluded from this baseline')
            self._task_indicator[env_ids] = self.TaskUID[self._eval_task].value
        else:
            if self.num_envs < 2 or self.num_envs % 2:
                raise ValueError('Training needs an even environment count, default 2048')
            # Fixed PUSH pool: 50% of rollout steps, independent of episode lengths.
            old = torch.multinomial(torch.tensor([0., 1/3, 1/3, 1/3], device=self.device), len(env_ids), replacement=True)
            self._task_indicator[env_ids] = torch.where(env_ids < self.num_envs//2, 4, old)
        self._push_cand_idx[env_ids] = -1
        for name in self._multiple_task_names:
            ids = env_ids[self._task_indicator[env_ids] == self.TaskUID[name].value]
            if not len(ids):
                continue
            self._task_mask[ids] = False
            self._task_mask[ids, self.TaskUID[name].value] = True
            self._reset_ref_env_ids[name], self._reset_ref_motion_ids[name], self._reset_ref_motion_times[name] = {}, {}, {}
            skills, probs = getattr(self, '_'+name+'_skill'), getattr(self, '_'+name+'_skill_init_prob')
            if name == 'push' and self._is_eval:
                mode = os.environ.get('TOKENHSI_PUSH_INIT', 'loco')
                if mode not in ('loco', 'rsi'):
                    raise ValueError('TOKENHSI_PUSH_INIT must be loco or rsi')
                probs = torch.tensor([float(mode == 'loco'), float(mode == 'rsi')], device=self.device)
            selected = torch.multinomial(probs, len(ids), replacement=True)
            for k, skill in enumerate(skills):
                rows = ids[selected == k]
                if not len(rows):
                    continue
                lib = self._motion_lib[skill]
                if name == 'push' and skill == 'push':
                    mids, times = self._bank.sample(len(rows))
                    self._push_cand_idx[rows] = 0
                else:
                    mids = lib.sample_motions(len(rows))
                    times = lib.sample_time_rsi(mids)
                self._reset_ref_env_ids[name][skill] = rows
                self._reset_ref_motion_ids[name][skill] = mids
                self._reset_ref_motion_times[name][skill] = times

    def _reset_task_push(self, env_ids):
        self._push_ever[env_ids] = False
        other = env_ids[self._task_indicator[env_ids] != 4]
        self._push_box_states[other] = 0
        self._push_box_states[other, :3] = self._push_park_pos
        self._push_box_states[other, 6] = 1
        self._push_marker_states[other, 2] = -5
        ids = env_ids[self._task_indicator[env_ids] == 4]
        if not len(ids):
            return
        n, d = len(ids), self.device
        box = self._push_box_states[ids].clone()
        box.zero_()
        yaw = torch.rand(n, device=d)*2*torch.pi
        box[:, :2] = box.new_tensor([20., 20.])
        box[:, 2] = self._push_box_size[ids, 2]/2
        box[:, 5], box[:, 6] = (yaw/2).sin(), (yaw/2).cos()
        angle = yaw+torch.pi/2+(torch.rand(n, device=d)-.5)*torch.pi/2
        distance = 2+8*torch.rand(n, device=d)
        goal = box[:, :3].clone()
        goal[:, :2] += distance[:, None]*torch.stack((angle.cos(), angle.sin()), -1)
        self._push_box_states[ids], self._push_tar_pos[ids] = box, goal
        # No box_progress: RSI and loco retain the sampled 2..10m goal distance.
        rsi = self._push_cand_idx[ids] >= 0
        loco = ids[~rsi]
        if len(loco):
            from utils.jointpush_geometry import yaw_from_quat
            states = self._humanoid_root_states[loco].clone()
            heading = yaw_from_quat(states[:, 3:7])
            # Reject root positions inside the rotated box + 0.45m clearance.
            indices = (~rsi).nonzero().flatten()
            pending = torch.arange(len(loco), device=d)
            pos = states[:, :2].clone()
            for _ in range(2048):
                if not len(pending):
                    break
                rad = 2+8*torch.rand(len(pending), device=d)
                a = 2*torch.pi*torch.rand(len(pending), device=d)
                offset = rad[:, None]*torch.stack((a.cos(), a.sin()), -1)
                local_yaw = yaw[indices[pending]]
                local = torch.stack((local_yaw.cos()*offset[:, 0]+local_yaw.sin()*offset[:, 1], -local_yaw.sin()*offset[:, 0]+local_yaw.cos()*offset[:, 1]), -1)
                gap = (local.abs()-self._push_box_size[loco[pending], :2]/2).clamp_min(0).norm(dim=-1)
                ok = gap >= .45
                pos[pending[ok]] = box[indices[pending[ok]], :2]+offset[ok]
                pending = pending[~ok]
            if len(pending):
                raise RuntimeError('PUSH root placement failed')
            to_box = box[indices, :2]-pos
            turn = torch.atan2(to_box[:, 1], to_box[:, 0])-heading
            q = states.new_zeros(len(loco), 4)
            q[:, 2], q[:, 3] = (turn/2).sin(), (turn/2).cos()
            kin = self._kinematic_humanoid_rigid_body_states[loco].clone()
            kin[..., :3] = rotate(q[:, None].expand(-1, kin.shape[1], -1), kin[..., :3]-states[:, None, :3])+torch.cat((pos, states[:, 2:3]), -1)[:, None]
            kin[..., 3:7] = multiply(q[:, None].expand(-1, kin.shape[1], -1), kin[..., 3:7])
            for a0, a1 in ((7, 10), (10, 13)):
                kin[..., a0:a1] = rotate(q[:, None].expand(-1, kin.shape[1], -1), kin[..., a0:a1])
            self._kinematic_humanoid_rigid_body_states[loco] = kin
            states[:, :2], states[:, 3:7] = pos, multiply(q, states[:, 3:7])
            states[:, 7:10], states[:, 10:13] = rotate(q, states[:, 7:10]), rotate(q, states[:, 10:13])
            self._humanoid_root_states[loco] = states
        rows = ids[rsi]
        if len(rows):
            kin = self._kinematic_humanoid_rigid_body_states[rows].clone()
            targets = push_rsi_hand_targets(self._push_box_states[rows, None], self._push_tar_pos[rows, None], self._push_box_size[rows, None], dict(push=self._joint['contact'], contact_at=True))[:, 0]
            body, bq, bv, ba, turn = align_push_pose(kin[..., :3], kin[..., 3:7], kin[..., 7:10], kin[..., 10:13], targets, self._push_hand_ids.cpu().tolist(), .06)
            body = safe_place(body, bq, self._push_box_states[rows], self._push_box_size[rows], targets, self._push_geometry)
            self._kinematic_humanoid_rigid_body_states[rows] = torch.cat((body, bq, bv, ba), -1)
            states = self._humanoid_root_states[rows].clone()
            states[:, :3], states[:, 3:7] = body[:, 0], multiply(turn, states[:, 3:7])
            states[:, 7:10], states[:, 10:13] = rotate(turn, states[:, 7:10]), rotate(turn, states[:, 10:13])
            self._humanoid_root_states[rows] = states
        self._push_marker_states[ids] = 0
        self._push_marker_states[ids, :3] = goal
        self._push_marker_states[ids, 2] = .02
        self._push_marker_states[ids, 6] = 1

    def _physics_step(self):
        self._push_contact_hit.zero_()
        for _ in range(self.control_freq_inv):
            self.render()
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self._refresh_sim_tensors()
            _, hit = contact(self._rigid_body_pos[:, self._push_hand_ids], self._push_box_states, self._push_tar_pos, self._push_box_size, self._contact_forces[:, self._push_hand_ids], self._push_box_contact, self._joint['contact'])
            self._push_contact_hit |= hit

    def _compute_reward(self, actions):
        HumanoidTrajSitCarryClimb._compute_reward(self, actions)
        r, success, _ = reward(self._humanoid_root_states[:, :3], self._rigid_body_pos[:, self._push_hand_ids], self._push_box_states, self._push_tar_pos, self._push_box_size, self._push_contact_hit, self._joint['contact'])
        if self._power_reward:
            r -= self._power_coefficient*(self.dof_force_tensor*self._dof_vel).abs().sum(-1)
        mask = self._task_indicator == 4
        self.rew_buf[mask] = r[mask]
        self._push_success[:] = success & mask
        self._push_ever |= self._push_success

    def _compute_reset(self):
        HumanoidTrajSitCarryClimb._compute_reset(self)
        mask = self._task_indicator == 4
        self.reset_buf[mask] = ((self.progress_buf >= 1199) | self._terminate_buf.bool())[mask].long()

    def post_physics_step(self):
        # Reuse only native TokenHSI bookkeeping, not delivered PUSH diagnostics.
        from env.tasks.humanoid import Humanoid
        Humanoid.post_physics_step(self)
        self._update_hist_amp_obs()
        self._compute_amp_observations()
        self.extras['amp_obs'] = self._amp_obs_buf.view(-1, self.get_num_amp_obs())
        self.extras['policy_obs'] = self.obs_buf.clone()
        for name in self._multiple_task_names:
            self.extras[name] = self._task_indicator == self.TaskUID[name].value
        self.extras['push_success_ever'] = self._push_ever.clone()
        if self._is_eval:
            if self._eval_task == 'push':
                self._success_buf[:] = self._push_success.long()
                self._precision_buf[:] = (self._push_box_states[:, :2]-self._push_tar_pos[:, :2]).norm(dim=-1)
            elif self._eval_task in ('sit', 'climb'):
                self._compute_metrics_evaluation_v1()
            elif self._eval_task == 'traj':
                self._compute_metrics_evaluation_v2()
            else:
                self._compute_metrics_evaluation_v3()
            self.extras['success'], self.extras['precision'] = self._success_buf, self._precision_buf

    def fetch_amp_obs_demo(self, num_samples):
        # Only PUSH expert samples enter the trainable discriminator.
        selected = torch.randint(2, (num_samples,), device=self.device)
        result = torch.empty(num_samples, self.get_num_amp_obs(), device=self.device)
        for i, skill in enumerate(('loco_push', 'push')):
            rows = (selected == i).nonzero().flatten()
            if not len(rows):
                continue
            lib = self._motion_lib[skill]
            mids = lib.sample_motions(len(rows))
            history = self.dt*(self._num_amp_obs_steps-1)
            times = lib.sample_time(mids, truncate_time=history)+history
            obs = self.build_amp_obs_demo(mids, times, lib)
            labels = obs.new_zeros(len(rows)*self._num_amp_obs_steps, 5)
            labels[:, 4] = 1
            result[rows] = torch.cat((obs, labels), -1).view(len(rows), -1)
        return result
