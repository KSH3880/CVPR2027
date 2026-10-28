# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
from enum import Enum
import numpy as np
import torch

from isaacgym import gymapi
from isaacgym import gymtorch

from env.tasks.humanoid import Humanoid, dof_to_obs
from utils import gym_util
from utils.motion_lib import MotionLib
from isaacgym.torch_utils import *

from utils import torch_utils
from utils import traj_generator

class HumanoidTraj(Humanoid):
    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):

        # configs for task
        self._enable_task_obs = cfg["env"]["enableTaskObs"]
        
        # traj following task
        self._num_traj_samples = cfg["env"]["numTrajSamples"]
        self._traj_sample_timestep = cfg["env"]["trajSampleTimestep"]
        self._speed_min = cfg["env"]["speedMin"]
        self._speed_max = cfg["env"]["speedMax"]
        self._accel_max = cfg["env"]["accelMax"]
        self._sharp_turn_prob = cfg["env"]["sharpTurnProb"]
        self._sharp_turn_angle = cfg["env"]["sharpTurnAngle"]
        self._fail_dist = 4.0

        # configs for amp
        state_init = cfg["env"]["stateInit"]
        self._state_init = HumanoidTraj.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        assert(self._num_amp_obs_steps >= 2)

        self._power_reward = cfg["env"]["power_reward"]
        self._power_coefficient = cfg["env"]["power_coefficient"]

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        motion_file = cfg['env']['motion_file']
        self._load_motion(motion_file)

        self._amp_obs_buf = torch.zeros((self.num_envs, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float)
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]
        
        self._amp_obs_demo_buf = None

        # tensors for task
        self._prev_root_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)

        # traj following task
        self._build_traj_generator()

        if (not self.headless):
            self._build_marker_state_tensors()

        # tensors for enableTrackInitState
        self._every_env_init_dof_pos = torch.zeros((self.num_envs, self.num_dof), device=self.device, dtype=torch.float)

        # tensors for fixing obs bug
        self._kinematic_humanoid_rigid_body_states = torch.zeros((self.num_envs, self.num_bodies, 13), device=self.device, dtype=torch.float)

        ###### evaluation!!!
        self._is_eval = cfg["args"].eval
        if self._is_eval:

            self._success_buf = torch.zeros((self.num_envs), device=self.device, dtype=torch.long)
            self._precision_buf = torch.zeros((self.num_envs), device=self.device, dtype=torch.float)

            self._success_threshold = cfg["env"]["eval"]["successThreshold"]

        return
    
    def get_obs_size(self):
        obs_size = super().get_obs_size()
        if (self._enable_task_obs):
            task_obs_size = self.get_task_obs_size()
            obs_size += task_obs_size
        return obs_size
    
    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            obs_size += 2 * self._num_traj_samples
        return obs_size

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._prev_root_pos[:] = self._humanoid_root_states[..., 0:3]
        return

    def _update_marker(self):
        traj_samples = self._fetch_traj_samples()
        self._traj_marker_pos[:] = traj_samples
        self._traj_marker_pos[..., 2] = self._char_h

        actor_ids = torch.cat([self._traj_marker_actor_ids], dim=-1)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(actor_ids), len(actor_ids))
        return
    
    def _create_envs(self, num_envs, spacing, num_per_row):
        if (not self.headless):
            self._traj_marker_handles = [[] for _ in range(self.num_envs)]
            self._load_marker_asset()

        super()._create_envs(num_envs, spacing, num_per_row)
        return
    
    def _load_marker_asset(self):
        asset_root = "tokenhsi/data/assets/mjcf/"
        asset_file = "location_marker.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 1.0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._marker_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        
        if (not self.headless):
            self._build_marker(env_id, env_ptr)

        return
    
    def _build_traj_generator(self):
        num_envs = self.num_envs
        episode_dur = self.max_episode_length * self.dt # 300 * 0.033333 = 10s
        num_verts = 101 # kinematic trajectory is described by 101 verts (100 segms, duration of each segm is 0.1s)
        dtheta_max = 2.0
        self._traj_gen = traj_generator.TrajGenerator(num_envs, episode_dur, num_verts,
                                                      self.device, dtheta_max,
                                                      self._speed_min, self._speed_max,
                                                      self._accel_max, self._sharp_turn_prob, self._sharp_turn_angle)

        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        root_pos = self._humanoid_root_states[:, 0:3]
        self._traj_gen.reset(env_ids, root_pos)

        return
    
    def _build_marker(self, env_id, env_ptr):
        col_group = self.num_envs + 10
        col_filter = 1
        segmentation_id = 0
        default_pose = gymapi.Transform()

        for i in range(self._num_traj_samples):

            marker_handle = self.gym.create_actor(env_ptr, self._marker_asset, default_pose, "marker", col_group, col_filter, segmentation_id)
            self.gym.set_actor_scale(env_ptr, marker_handle, 0.5)
            self.gym.set_rigid_body_color(env_ptr, marker_handle, 0,
                                          gymapi.MESH_VISUAL,
                                          gymapi.Vec3(1.0, 0.0, 0.0))
            self._traj_marker_handles[env_id].append(marker_handle)

        return

    def _build_marker_state_tensors(self):
        num_actors = self._root_states.shape[0] // self.num_envs

        self._traj_marker_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 1:(1 + self._num_traj_samples), :]
        self._traj_marker_pos = self._traj_marker_states[..., :3]

        self._traj_marker_actor_ids = self._humanoid_actor_ids.unsqueeze(-1) + to_torch(self._traj_marker_handles, dtype=torch.int32, device=self.device)
        self._traj_marker_actor_ids = self._traj_marker_actor_ids.flatten()

        return
    
    def _reset_task(self, env_ids):

        ##### traj following task
        self._reset_traj_follow_task(env_ids)

        return

    def _reset_traj_follow_task(self, env_ids):
        root_pos = self._humanoid_root_states[env_ids, 0:3]
        self._traj_gen.reset(env_ids, root_pos)

        return
    
    def _compute_observations(self, env_ids=None):
        humanoid_obs = self._compute_humanoid_obs(env_ids)
        
        if (self._enable_task_obs):
            task_obs = self._compute_task_obs(env_ids)
            obs = torch.cat([humanoid_obs, task_obs], dim=-1)
        else:
            obs = humanoid_obs

        if (env_ids is None):
            self.obs_buf[:] = obs
        else:
            self.obs_buf[env_ids] = obs
        return
    
    def _fetch_traj_samples(self, env_ids=None):
        # 5 seconds with 0.5 second intervals, 10 samples.
        if (env_ids is None):
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        timestep_beg = self.progress_buf[env_ids] * self.dt
        timesteps = torch.arange(self._num_traj_samples, device=self.device, dtype=torch.float)
        timesteps = timesteps * self._traj_sample_timestep
        traj_timesteps = timestep_beg.unsqueeze(-1) + timesteps

        env_ids_tiled = torch.broadcast_to(env_ids.unsqueeze(-1), traj_timesteps.shape)

        traj_samples_flat = self._traj_gen.calc_pos(env_ids_tiled.flatten(), traj_timesteps.flatten())
        traj_samples = torch.reshape(traj_samples_flat, shape=(env_ids.shape[0], self._num_traj_samples, traj_samples_flat.shape[-1]))

        return traj_samples

    def _compute_task_obs(self, env_ids=None):
        if (env_ids is None):
            root_states = self._humanoid_root_states
        else:
            root_states = self._humanoid_root_states[env_ids]

        traj_samples = self._fetch_traj_samples(env_ids)
        obs = compute_location_observations(root_states, traj_samples)

        return obs

    def _compute_reward(self, actions):
        root_pos = self._humanoid_root_states[..., 0:3]
        root_rot = self._humanoid_root_states[..., 3:7]

        time = self.progress_buf * self.dt
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        traj_tar_pos = self._traj_gen.calc_pos(env_ids, time)

        reward = compute_traj_reward(root_pos, traj_tar_pos)

        power = torch.abs(torch.multiply(self.dof_force_tensor, self._dof_vel)).sum(dim = -1)
        power_reward = -self._power_coefficient * power

        if self._power_reward:
            self.rew_buf[:] = reward + power_reward
        else:
            self.rew_buf[:] = reward

        return
    
    def render(self, sync_frame_time=False):
        super().render(sync_frame_time)

        if self.viewer:
            self._draw_task()
        return

    def _draw_task(self):
        self._update_marker()

        traj_cols = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)

        self.gym.clear_lines(self.viewer)

        for i, env_ptr in enumerate(self.envs):
            # traj
            verts = self._traj_gen.get_traj_verts(i)
            verts[..., 2] = self._char_h
            lines = torch.cat([verts[:-1], verts[1:]], dim=-1).cpu().numpy()
            curr_cols = np.broadcast_to(traj_cols, [lines.shape[0], traj_cols.shape[-1]])
            self.gym.add_lines(self.viewer, env_ptr, lines.shape[0], lines, curr_cols)

        return

    def post_physics_step(self):
        super().post_physics_step()
        
        self._update_hist_amp_obs()
        self._compute_amp_observations()

        amp_obs_flat = self._amp_obs_buf.view(-1, self.get_num_amp_obs())
        self.extras["amp_obs"] = amp_obs_flat
        self.extras["policy_obs"] = self.obs_buf.clone()

        if self._is_eval:
            self._compute_metrics_evaluation()
            self.extras["success"] = self._success_buf
            self.extras["precision"] = self._precision_buf

        return
    
    def _compute_metrics_evaluation(self):
        root_pos = self._humanoid_root_states[..., 0:3]
        root_rot = self._humanoid_root_states[..., 3:7]

        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        timesteps = torch.ones_like(env_ids, device=self.device, dtype=torch.float)

        coeff = 0.98

        timesteps[:] = self.max_episode_length * coeff * self.dt # float('Inf')
        traj_final_tar_pos = self._traj_gen.calc_pos(env_ids, timesteps)

        timesteps[:] = self.progress_buf * self.dt
        traj_curr_tar_pos = self._traj_gen.calc_pos(env_ids, timesteps)

        pos_diff = traj_final_tar_pos[..., 0:2] - root_pos[..., 0:2]
        pos_err = torch.norm(pos_diff, p=2, dim=-1)
        dist_mask = pos_err <= self._success_threshold
        time_mask = self.progress_buf >= self.max_episode_length * coeff

        success_mask = torch.logical_and(dist_mask, time_mask)
        self._success_buf[success_mask] += 1

        pos_diff = traj_curr_tar_pos[..., 0:2] - root_pos[..., 0:2]
        pos_err = torch.norm(pos_diff, p=2, dim=-1)
        self._precision_buf += pos_err

        return

    def get_num_amp_obs(self):
        return self._num_amp_obs_steps * self._num_amp_obs_per_step

    def fetch_amp_obs_demo(self, num_samples):

        if (self._amp_obs_demo_buf is None):
            self._build_amp_obs_demo_buf(num_samples)
        else:
            assert(self._amp_obs_demo_buf.shape[0] == num_samples)
        
        motion_ids = self._motion_lib.sample_motions(num_samples)
        
        # since negative times are added to these values in build_amp_obs_demo,
        # we shift them into the range [0 + truncate_time, end of clip]
        truncate_time = self.dt * (self._num_amp_obs_steps - 1)
        motion_times0 = self._motion_lib.sample_time(motion_ids, truncate_time=truncate_time)
        motion_times0 += truncate_time

        amp_obs_demo = self.build_amp_obs_demo(motion_ids, motion_times0)
        self._amp_obs_demo_buf[:] = amp_obs_demo.view(self._amp_obs_demo_buf.shape)
        amp_obs_demo_flat = self._amp_obs_demo_buf.view(-1, self.get_num_amp_obs())

        return amp_obs_demo_flat

    def build_amp_obs_demo(self, motion_ids, motion_times0):
        dt = self.dt

        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps])
        motion_times = motion_times0.unsqueeze(-1)
        time_steps = -dt * torch.arange(0, self._num_amp_obs_steps, device=self.device)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
        amp_obs_demo = build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel,
                                              dof_pos, dof_vel, key_pos,
                                              self._local_root_obs, self._root_height_obs,
                                              self._dof_obs_size, self._dof_offsets)
        return amp_obs_demo

    def _build_amp_obs_demo_buf(self, num_samples):
        self._amp_obs_demo_buf = torch.zeros((num_samples, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float32)
        return
        
    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)

        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        num_key_bodies = len(key_bodies)

        if (asset_file == "mjcf/amp_humanoid.xml"):
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + 28 + 3 * num_key_bodies # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
        elif (asset_file == "mjcf/phys_humanoid.xml") or (asset_file == "mjcf/phys_humanoid_v2.xml") or (asset_file == "mjcf/phys_humanoid_v3.xml"):
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + 28 + 2 * 2 + 3 * num_key_bodies # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
        else:
            print("Unsupported character config file: {s}".format(asset_file))
            assert(False)

        return

    def _load_motion(self, motion_file):
        assert(self._dof_offsets[-1] == self.num_dof)

        ext = os.path.splitext(motion_file)[1]
        if (ext == ".yaml"):
            self._motion_lib = MotionLib(motion_file=motion_file,
                                         skill=self.cfg["env"]["skill"],
                                         dof_body_ids=self._dof_body_ids,
                                         dof_offsets=self._dof_offsets,
                                         key_body_ids=self._key_body_ids.cpu().numpy(), 
                                         device=self.device)
        else:
            raise NotImplementedError

        return
    
    def _reset_envs(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        if (len(env_ids) > 0):
            self._reset_actors(env_ids)
            self._reset_task(env_ids)
            self._reset_env_tensors(env_ids)
            self._refresh_sim_tensors()
            self._compute_observations(env_ids)
        
        self._init_amp_obs(env_ids)

        return
    
    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)

        if self._is_eval:
            self._success_buf[env_ids] = 0
            self._precision_buf[env_ids] = 0 # not Inf

        return

    def _reset_actors(self, env_ids):
        if (self._state_init == HumanoidTraj.StateInit.Default):
            self._reset_default(env_ids)
        elif (self._state_init == HumanoidTraj.StateInit.Start
              or self._state_init == HumanoidTraj.StateInit.Random):
            self._reset_ref_state_init(env_ids)
        elif (self._state_init == HumanoidTraj.StateInit.Hybrid):
            self._reset_hybrid_state_init(env_ids)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))
        return
    
    def _reset_default(self, env_ids):
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        self._reset_default_env_ids = env_ids

        if (len(self._reset_default_env_ids) > 0):
            self._kinematic_humanoid_rigid_body_states[self._reset_default_env_ids] = self._initial_humanoid_rigid_body_states[self._reset_default_env_ids]

        self._every_env_init_dof_pos[self._reset_default_env_ids] = self._initial_dof_pos[env_ids] # for "enableTrackInitState"

        return

    def _reset_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        motion_ids = self._motion_lib.sample_motions(num_envs)
        
        if (self._state_init == HumanoidTraj.StateInit.Random
            or self._state_init == HumanoidTraj.StateInit.Hybrid):
            motion_times = self._motion_lib.sample_time(motion_ids)
        elif (self._state_init == HumanoidTraj.StateInit.Start):
            motion_times = torch.zeros(num_envs, device=self.device)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)

        self._set_env_state(env_ids=env_ids, 
                            root_pos=root_pos, 
                            root_rot=root_rot, 
                            dof_pos=dof_pos, 
                            root_vel=root_vel, 
                            root_ang_vel=root_ang_vel, 
                            dof_vel=dof_vel)

        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids = motion_ids
        self._reset_ref_motion_times = motion_times

        if (len(self._reset_ref_env_ids) > 0):
            body_pos, body_rot, body_vel, body_ang_vel \
                = self._motion_lib.get_motion_state_max(self._reset_ref_motion_ids, self._reset_ref_motion_times)
            self._kinematic_humanoid_rigid_body_states[self._reset_ref_env_ids] = torch.cat((body_pos, body_rot, body_vel, body_ang_vel), dim=-1)
        
        self._every_env_init_dof_pos[self._reset_ref_env_ids] = dof_pos # for "enableTrackInitState"

        return

    def _reset_hybrid_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        ref_probs = to_torch(np.array([self._hybrid_init_prob] * num_envs), device=self.device)
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        ref_reset_ids = env_ids[ref_init_mask]
        if (len(ref_reset_ids) > 0):
            self._reset_ref_state_init(ref_reset_ids)

        default_reset_ids = env_ids[torch.logical_not(ref_init_mask)]
        if (len(default_reset_ids) > 0):
            self._reset_default(default_reset_ids)

        return

    def _init_amp_obs(self, env_ids):
        self._compute_amp_observations(env_ids)

        if (len(self._reset_default_env_ids) > 0):
            self._init_amp_obs_default(self._reset_default_env_ids)

        if (len(self._reset_ref_env_ids) > 0):
            self._init_amp_obs_ref(self._reset_ref_env_ids, self._reset_ref_motion_ids,
                                   self._reset_ref_motion_times)
        
        return

    def _init_amp_obs_default(self, env_ids):
        curr_amp_obs = self._curr_amp_obs_buf[env_ids].unsqueeze(-2)
        self._hist_amp_obs_buf[env_ids] = curr_amp_obs
        return

    def _init_amp_obs_ref(self, env_ids, motion_ids, motion_times):
        dt = self.dt
        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps - 1])
        motion_times = motion_times.unsqueeze(-1)
        time_steps = -dt * (torch.arange(0, self._num_amp_obs_steps - 1, device=self.device) + 1)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
        amp_obs_demo = build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel, 
                                              dof_pos, dof_vel, key_pos, 
                                              self._local_root_obs, self._root_height_obs, 
                                              self._dof_obs_size, self._dof_offsets)
        self._hist_amp_obs_buf[env_ids] = amp_obs_demo.view(self._hist_amp_obs_buf[env_ids].shape)
        return
    
    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self._humanoid_root_states[env_ids, 0:3] = root_pos
        self._humanoid_root_states[env_ids, 3:7] = root_rot
        self._humanoid_root_states[env_ids, 7:10] = root_vel
        self._humanoid_root_states[env_ids, 10:13] = root_ang_vel
        
        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel
        return

    def _update_hist_amp_obs(self, env_ids=None):
        if (env_ids is None):
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[:, i + 1] = self._amp_obs_buf[:, i]
        else:
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[env_ids, i + 1] = self._amp_obs_buf[env_ids, i]
        return
    
    def _compute_amp_observations(self, env_ids=None):
        if (env_ids is None):
            key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
            self._curr_amp_obs_buf[:] = build_amp_observations(self._rigid_body_pos[:, 0, :],
                                                               self._rigid_body_rot[:, 0, :],
                                                               self._rigid_body_vel[:, 0, :],
                                                               self._rigid_body_ang_vel[:, 0, :],
                                                               self._dof_pos, self._dof_vel, key_body_pos,
                                                               self._local_root_obs, self._root_height_obs, 
                                                               self._dof_obs_size, self._dof_offsets)
        else:
            kinematic_rigid_body_pos = self._kinematic_humanoid_rigid_body_states[:, :, 0:3]
            key_body_pos = kinematic_rigid_body_pos[:, self._key_body_ids, :]
            self._curr_amp_obs_buf[env_ids] = build_amp_observations(self._kinematic_humanoid_rigid_body_states[env_ids, 0, 0:3],
                                                                   self._kinematic_humanoid_rigid_body_states[env_ids, 0, 3:7],
                                                                   self._kinematic_humanoid_rigid_body_states[env_ids, 0, 7:10],
                                                                   self._kinematic_humanoid_rigid_body_states[env_ids, 0, 10:13],
                                                                   self._dof_pos[env_ids], self._dof_vel[env_ids], key_body_pos[env_ids],
                                                                   self._local_root_obs, self._root_height_obs, 
                                                                   self._dof_obs_size, self._dof_offsets)
        return
    
    def _compute_reset(self):
        time = self.progress_buf * self.dt
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        tar_pos = self._traj_gen.calc_pos(env_ids, time)

        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                           self._contact_forces, self._contact_body_ids,
                                                           self._rigid_body_pos, tar_pos,
                                                           self.max_episode_length, self._fail_dist,
                                                           self._enable_early_termination, self._termination_heights)
        return


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, 
                           local_root_obs, root_height_obs, dof_obs_size, dof_offsets):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, int, List[int]) -> Tensor
    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = torch_utils.quat_to_tan_norm(root_rot_obs)
    
    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h
    
    local_root_vel = quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    local_end_pos = quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])
    
    dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets)
    obs = torch.cat((root_h_obs, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos), dim=-1)
    return obs

@torch.jit.script
def compute_location_observations(root_states, traj_samples):
    # type: (Tensor, Tensor) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    heading_rot_exp = torch.broadcast_to(heading_rot.unsqueeze(1), (traj_samples.shape[0], traj_samples.shape[1], 4))
    root_pos_exp = torch.broadcast_to(root_pos.unsqueeze(1), (traj_samples.shape[0], traj_samples.shape[1], 3))
    
    local_traj_samples = quat_rotate(heading_rot_exp.reshape(-1, 4), traj_samples.reshape(-1, 3) - root_pos_exp.reshape(-1, 3))

    obs = local_traj_samples[..., 0:2].reshape(root_pos.shape[0], -1)

    return obs

@torch.jit.script
def compute_traj_reward(root_pos, tar_pos):
    # type: (Tensor, Tensor) -> Tensor
    pos_err_scale = 2.0

    pos_diff = tar_pos[..., 0:2] - root_pos[..., 0:2]
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)

    pos_reward = torch.exp(-pos_err_scale * pos_err)

    reward = pos_reward

    return reward

@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                           tar_pos, max_episode_length, fail_dist,
                           enable_early_termination, termination_heights):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, bool, Tensor) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_fallen = torch.logical_and(fall_contact, fall_height)
        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= (progress_buf > 1)

        root_pos = rigid_body_pos[..., 0, :]
        tar_delta = tar_pos[..., 0:2] - root_pos[..., 0:2]
        tar_dist_sq = torch.sum(tar_delta * tar_delta, dim=-1)
        tar_fail = tar_dist_sq > fail_dist * fail_dist

        has_failed = torch.logical_or(has_fallen, tar_fail)

        terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated
