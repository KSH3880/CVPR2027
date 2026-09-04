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
import sys
from enum import Enum
import numpy as np
import torch
import yaml
import json
import pickle
import trimesh

from isaacgym import gymapi
from isaacgym import gymtorch

from env.tasks.humanoid import Humanoid, dof_to_obs
from utils import gym_util
from utils.motion_lib import MotionLib
from isaacgym.torch_utils import *

from utils import torch_utils
from utils import traj_generator

class HumanoidCompTrajCarry(Humanoid):
    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):

        # configs for task
        self._enable_task_obs = cfg["env"]["enableTaskObs"]
        self._only_height_handheld_reward = cfg["env"]["onlyHeightHandHeldReward"]

        self._enable_fall_termination = cfg["env"]["enableFallTermination"]

        self._mode = cfg["env"]["mode"] # determine which set of objects to use (train or test)
        assert self._mode in ["train", "test"]

        if cfg["args"].eval:
            self._mode = "test"

        # configs for box
        box_cfg = cfg["env"]["box"]
        self._build_base_size = box_cfg["build"]["baseSize"]
        self._build_random_size = box_cfg["build"]["randomSize"]
        self._build_random_mode_equal_proportion = box_cfg["build"]["randomModeEqualProportion"]
        self._build_x_scale_range = box_cfg["build"]["scaleRangeX"]
        self._build_y_scale_range = box_cfg["build"]["scaleRangeY"]
        self._build_z_scale_range = box_cfg["build"]["scaleRangeZ"]
        self._build_scale_sample_interval = box_cfg["build"]["scaleSampleInterval"]
        self._build_test_sizes = box_cfg["build"]["testSizes"]

        self._reset_random_height = box_cfg["reset"]["randomHeight"]
        self._reset_random_height_prob = box_cfg["reset"]["randomHeightProb"]
        self._reset_maxTopSurfaceHeight = box_cfg["reset"]["maxTopSurfaceHeight"]

        self._enable_bbox_obs = box_cfg["obs"]["enableBboxObs"]
        
        # traj following task
        self._num_traj_samples = cfg["env"]["numTrajSamples"]
        self._traj_sample_timestep = cfg["env"]["trajSampleTimestep"]
        self._speed_min = cfg["env"]["speedMin"]
        self._speed_max = cfg["env"]["speedMax"]
        self._accel_max = cfg["env"]["accelMax"]
        self._sharp_turn_prob = cfg["env"]["sharpTurnProb"]
        self._sharp_turn_angle = cfg["env"]["sharpTurnAngle"]
        self._fail_dist = 4.0

        # manage multi task obs
        self._num_tasks = 3
        task_obs_size_traj = self._num_traj_samples * 2
        task_obs_size_carry = 3 + 3 + 6 + 3 + 3 + 3 * 8 # tar_pos, box_pos, box_6drot, box_linvel, box_angvel, box_bbox
        self._each_subtask_obs_size = [
            task_obs_size_traj + (task_obs_size_carry - 3), # new comp task obs
            task_obs_size_traj, # traj
            task_obs_size_carry, # carry
        ]
        self._multiple_task_names = ["new", "traj", "carry"]
        self._enable_task_mask_obs = False

        # set major task for AdaptNet
        self._major_task = cfg["env"]["majorTask"]
        assert self._major_task in self._multiple_task_names
        self._major_task_ind = self._multiple_task_names.index(self._major_task)

        # configs for amp
        state_init = cfg["env"]["stateInit"]
        self._state_init = HumanoidCompTrajCarry.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        assert(self._num_amp_obs_steps >= 2)

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = {} # to enable multi-skill reference init, use dict instead of list
        self._reset_ref_motion_ids = {}
        self._reset_ref_motion_times = {}

        self._power_reward = cfg["env"]["power_reward"]
        self._power_coefficient = cfg["env"]["power_coefficient"]

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        self._skill = cfg["env"]["skill"]
        self._skill_init_prob = torch.tensor(cfg["env"]["skillInitProb"], device=self.device, dtype=torch.float) # probs for state init
        self._skill_disc_prob = torch.tensor(cfg["env"]["skillDiscProb"], device=self.device, dtype=torch.float) # probs for amp obs demo fetch

        motion_file = cfg['env']['motion_file']
        self._load_motion(motion_file)

        self._amp_obs_buf = torch.zeros((self.num_envs, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float)
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]
        
        self._amp_obs_demo_buf = None

        # tensors for task
        self._prev_root_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self._prev_box_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self._tar_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float) # target location of the box, 3d xyz

        spacing = cfg["env"]["envSpacing"]
        if spacing <= 0.5:
            self._tar_pos_dist = torch.distributions.uniform.Uniform(
                torch.tensor([-4.5, -4.5, 0.5], device=self.device),
                torch.tensor([4.5, 4.5, 1.0], device=self.device))
        else:
            self._tar_pos_dist = torch.distributions.uniform.Uniform(
                torch.tensor([-(spacing - 0.5), -(spacing - 0.5), 0.5], device=self.device),
                torch.tensor([(spacing - 0.5), (spacing - 0.5), 1.0], device=self.device))

        # traj following task
        self._build_traj_generator()

        if (not self.headless):
            self._build_marker_state_tensors()
        
        # tensors for box
        self._build_box_tensors()

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

            self._skill = cfg["env"]["eval"]["skill"]
            self._skill_init_prob = torch.tensor(cfg["env"]["eval"]["skillInitProb"], device=self.device, dtype=torch.float) # probs for state init

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
            obs_size = sum(self._each_subtask_obs_size)
        return obs_size

    def get_multi_task_info(self):

        num_subtasks = self._num_tasks
        each_subtask_obs_size = self._each_subtask_obs_size

        each_subtask_obs_mask = torch.zeros(num_subtasks, sum(each_subtask_obs_size), dtype=torch.bool, device=self.device)

        index = torch.cumsum(torch.tensor([0] + each_subtask_obs_size), dim=0).to(self.device)
        for i in range(num_subtasks):
            each_subtask_obs_mask[i, index[i]:index[i + 1]] = True

        info = {
            "onehot_size": num_subtasks,
            "tota_subtask_obs_size": sum(each_subtask_obs_size),
            "each_subtask_obs_size": each_subtask_obs_size,
            "each_subtask_obs_mask": each_subtask_obs_mask,
            "each_subtask_obs_indx": index,
            "enable_task_mask_obs": self._enable_task_mask_obs,

            "each_subtask_name": self._multiple_task_names,
            "major_task_indx": self._major_task_ind
        }
        return info
    
    def _build_box_tensors(self):
        num_actors = self.get_num_actors_per_env()
        idx = self._box_handles[0]
        self._box_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., idx, :]
        
        self._box_actor_ids = self._humanoid_actor_ids + idx

        self._initial_box_states = self._box_states.clone()
        self._initial_box_states[:, 7:13] = 0

        self._build_box_bps()

        return

    def _build_box_bps(self):
        bps_0 = torch.vstack([     self._box_size[:, 0] / 2,      self._box_size[:, 1] / 2, -1 * self._box_size[:, 2] / 2]).t().unsqueeze(-2)
        bps_1 = torch.vstack([-1 * self._box_size[:, 0] / 2,      self._box_size[:, 1] / 2, -1 * self._box_size[:, 2] / 2]).t().unsqueeze(-2)
        bps_2 = torch.vstack([-1 * self._box_size[:, 0] / 2, -1 * self._box_size[:, 1] / 2, -1 * self._box_size[:, 2] / 2]).t().unsqueeze(-2)
        bps_3 = torch.vstack([     self._box_size[:, 0] / 2, -1 * self._box_size[:, 1] / 2, -1 * self._box_size[:, 2] / 2]).t().unsqueeze(-2)
        bps_4 = torch.vstack([     self._box_size[:, 0] / 2,      self._box_size[:, 1] / 2,      self._box_size[:, 2] / 2]).t().unsqueeze(-2)
        bps_5 = torch.vstack([-1 * self._box_size[:, 0] / 2,      self._box_size[:, 1] / 2,      self._box_size[:, 2] / 2]).t().unsqueeze(-2)
        bps_6 = torch.vstack([-1 * self._box_size[:, 0] / 2, -1 * self._box_size[:, 1] / 2,      self._box_size[:, 2] / 2]).t().unsqueeze(-2)
        bps_7 = torch.vstack([     self._box_size[:, 0] / 2, -1 * self._box_size[:, 1] / 2,      self._box_size[:, 2] / 2]).t().unsqueeze(-2)
        self._box_bps = torch.cat([bps_0, bps_1, bps_2, bps_3, bps_4, bps_5, bps_6, bps_7], dim=1).to(self.device) # (num_envs, 8, 3)

        return

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._prev_root_pos[:] = self._humanoid_root_states[..., 0:3]
        self._prev_box_pos[:] = self._box_states[..., 0:3]
        return

    def _update_marker(self):
        traj_samples = self._fetch_traj_samples()
        self._traj_marker_pos[:] = traj_samples
        self._traj_marker_pos[..., 2] = self._char_h

        actor_ids = torch.cat([self._traj_marker_actor_ids, self._box_actor_ids,], dim=-1)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(actor_ids), len(actor_ids))
        return
    
    def _create_envs(self, num_envs, spacing, num_per_row):
        if (not self.headless):
            self._traj_marker_handles = [[] for _ in range(self.num_envs)]
            self._load_marker_asset()
        
        self._box_handles = []
        self._load_box_asset()

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
    
    def _load_box_asset(self):
        
        # rescale
        self._box_scale = torch.ones((self.num_envs, 3), dtype=torch.float32, device=self.device)
        if self._build_random_size:

            assert int((self._build_x_scale_range[1] - self._build_x_scale_range[0]) % self._build_scale_sample_interval) == 0
            assert int((self._build_y_scale_range[1] - self._build_y_scale_range[0]) % self._build_scale_sample_interval) == 0
            assert int((self._build_z_scale_range[1] - self._build_z_scale_range[0]) % self._build_scale_sample_interval) == 0

            x_scale_linespace = torch.arange(self._build_x_scale_range[0], self._build_x_scale_range[1] + self._build_scale_sample_interval, self._build_scale_sample_interval)
            y_scale_linespace = torch.arange(self._build_y_scale_range[0], self._build_y_scale_range[1] + self._build_scale_sample_interval, self._build_scale_sample_interval)
            z_scale_linespace = torch.arange(self._build_z_scale_range[0], self._build_z_scale_range[1] + self._build_scale_sample_interval, self._build_scale_sample_interval)

            if self._build_random_mode_equal_proportion == False:

                num_scales = len(x_scale_linespace) * len(y_scale_linespace) * len(z_scale_linespace)
                scale_pool = torch.zeros((num_scales, 3), device=self.device)
                idx = 0
                for curr_x in x_scale_linespace:
                    for curr_y in y_scale_linespace:
                        for curr_z in z_scale_linespace:
                            scale_pool[idx] = torch.tensor([curr_x, curr_y, curr_z])
                            idx += 1
            
            else:
                num_scales = len(x_scale_linespace)
                scale_pool = torch.zeros((num_scales, 3), device=self.device)
                idx = 0
                for curr_x in x_scale_linespace:
                    scale_pool[idx] = torch.tensor([curr_x, curr_x, curr_x])
                    idx += 1

                if self._mode == "test":
                    test_sizes = torch.tensor(self._build_test_sizes, device=self.device)
                    scale_pool = torch.zeros((test_sizes.shape[0], 3), device=self.device)
                    num_scales = test_sizes.shape[0]

                    for axis in range(3):
                        scale_pool[:, axis] = test_sizes[:, axis] / self._build_base_size[axis]

            if self.num_envs >= num_scales:
                self._box_scale[:num_scales] = scale_pool[:num_scales] # copy

                sampled_scale_id = torch.multinomial(torch.ones(num_scales) * (1.0 / num_scales), num_samples=(self.num_envs - num_scales), replacement=True)
                self._box_scale[num_scales:] = scale_pool[sampled_scale_id]

                shuffled_id = torch.randperm(self.num_envs)
                self._box_scale = self._box_scale[shuffled_id]

            else:
                sampled_scale_id = torch.multinomial(torch.ones(num_scales) * (1.0 / num_scales), num_samples=self.num_envs, replacement=True)
                self._box_scale = scale_pool[sampled_scale_id]

        self._box_size = torch.tensor(self._build_base_size, device=self.device).reshape(1, 3) * self._box_scale # (num_envs, 3)

        # create asset
        self._box_assets = []
        for i in range(self.num_envs):
            asset_options = gymapi.AssetOptions()
            asset_options.angular_damping = 0.01
            asset_options.linear_damping = 0.01
            asset_options.max_angular_velocity = 100.0
            asset_options.density = 100.0
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            self._box_assets.append(self.gym.create_box(self.sim, self._box_size[i, 0], self._box_size[i, 1], self._box_size[i, 2], asset_options))
        
        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)

        self._build_box(env_id, env_ptr)
        
        if (not self.headless):
            self._build_marker(env_id, env_ptr)

        return
    
    def _build_box(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        default_pose = gymapi.Transform()
        default_pose.p.x = self._box_size[env_id, 0] / 2 + 5
        default_pose.p.y = 0
        default_pose.p.z = self._box_size[env_id, 2] / 2 # ensure no penetration between box and ground plane
    
        box_handle = self.gym.create_actor(env_ptr, self._box_assets[env_id], default_pose, "box", col_group, col_filter, segmentation_id)
        self._box_handles.append(box_handle)

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

        idx = self._traj_marker_handles[0][0]

        self._traj_marker_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., idx:(idx + self._num_traj_samples), :]
        self._traj_marker_pos = self._traj_marker_states[..., :3]

        self._traj_marker_actor_ids = self._humanoid_actor_ids.unsqueeze(-1) + to_torch(self._traj_marker_handles, dtype=torch.int32, device=self.device)
        self._traj_marker_actor_ids = self._traj_marker_actor_ids.flatten()

        return
    
    def _reset_task(self, env_ids):

        ##### traj following task
        self._reset_traj_follow_task(env_ids)

        ##### carry task

        #  for all skills and reset default
        random_env_ids = []
        if len(self._reset_default_env_ids) > 0:
            random_env_ids.append(self._reset_default_env_ids)
        for sk_name in self._skill:
            if self._reset_ref_env_ids.get(sk_name) is not None:
                random_env_ids.append(self._reset_ref_env_ids[sk_name])

        if len(random_env_ids) > 0:
            ids = torch.cat(random_env_ids, dim=0)

            ##### Previous version (before Oct 9, 2024): randomly generate target locations

            # new_target_pos = self._tar_pos_dist.sample((len(ids),))
            # new_target_pos[:, 2] = self._box_size[ids, 2] / 2 # place the box on the ground

            # min_dist = 1.0

            # # check if the new pos is too close to character or box
            # target_overlap = torch.logical_or(
            #     torch.sum((new_target_pos[..., :2] - self._humanoid_root_states[ids, :2]) ** 2, dim=-1) < min_dist,
            #     torch.sum((new_target_pos[..., :2] - self._box_states[ids, :2]) ** 2, dim=-1) < min_dist
            # )
            # while(torch.sum(target_overlap) > 0):
            #     new_target_pos[target_overlap] = self._tar_pos_dist.sample((torch.sum(target_overlap),))
            #     new_target_pos[:, 2] = self._box_size[ids, 2] / 2 # place the box on the ground
            #     target_overlap = torch.logical_or(
            #         torch.sum((new_target_pos[..., :2] - self._humanoid_root_states[ids, :2]) ** 2, dim=-1) < min_dist,
            #         torch.sum((new_target_pos[..., :2] - self._box_states[ids, :2]) ** 2, dim=-1) < min_dist
            #     )

            # if self._reset_random_height:
            #     num_envs = ids.shape[0]
            #     probs = to_torch(np.array([self._reset_random_height_prob] * num_envs), device=self.device)
            #     mask = torch.bernoulli(probs) == 1.0
                
            #     if mask.sum() > 0:
            #         new_target_pos[mask, 2] += torch.rand(mask.sum(), device=self.device) * (self.cfg["env"]["box"]["reset"]["maxCenterHeight"] - self.cfg["env"]["box"]["reset"]["minCenterHeight"]) + self.cfg["env"]["box"]["reset"]["minCenterHeight"]
            #         new_target_pos[mask, 2] = self._regulate_height(new_target_pos[mask, 2], self._box_size[ids[mask]])

            ##### Current version (after Oct 9, 2024): place target locations at the end points of trajs

            timesteps = torch.ones_like(self.progress_buf[ids]) * self.max_episode_length * self.dt * 10.0
            traj_ends = self._traj_gen.calc_pos(ids, timesteps)
            traj_ends[..., 2] = self._box_size[ids, 2] / 2

            self._tar_pos[ids] = traj_ends

        return

    def _regulate_height(self, h, box_size):
        top_surface_z = h + box_size[:, 2] / 2
        top_surface_z = torch.clamp_max(top_surface_z, self._reset_maxTopSurfaceHeight)
        return top_surface_z - box_size[:, 2] / 2

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
            box_states = self._box_states
            box_bps = self._box_bps
            tar_pos = self._tar_pos
        else:
            root_states = self._humanoid_root_states[env_ids]
            box_states = self._box_states[env_ids]
            box_bps = self._box_bps[env_ids]
            tar_pos = self._tar_pos[env_ids]

        traj_samples = self._fetch_traj_samples(env_ids)
        traj_obs = compute_location_observations(root_states, traj_samples)

        carry_obs = compute_carry_observations(root_states, box_states, box_bps, tar_pos,
                                            self._enable_bbox_obs)
        
        obs = torch.cat([carry_obs[..., :-3], traj_obs, traj_obs, carry_obs], dim=-1)

        return obs

    def _compute_reward(self, actions):
        root_pos = self._humanoid_root_states[..., 0:3]
        root_rot = self._humanoid_root_states[..., 3:7]
        rigid_body_pos = self._rigid_body_pos
        box_pos = self._box_states[..., 0:3]
        box_rot = self._box_states[..., 3:7]
        hands_ids = self._key_body_ids[[0, 1]]

        time = self.progress_buf * self.dt
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        traj_tar_pos = self._traj_gen.calc_pos(env_ids, time)

        traj_r = compute_traj_reward(root_pos, traj_tar_pos, box_pos, rigid_body_pos, hands_ids)
        handheld_r = compute_handheld_reward(rigid_body_pos, box_pos, hands_ids, self._only_height_handheld_reward)

        reward = 0.5 * traj_r + 0.5 * handheld_r

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
        
        if self._show_lines_flag:

            for i, env_ptr in enumerate(self.envs):
                # traj
                verts = self._traj_gen.get_traj_verts(i)
                verts[..., 2] = self._char_h
                lines = torch.cat([verts[:-1], verts[1:]], dim=-1).cpu().numpy()
                curr_cols = np.broadcast_to(traj_cols, [lines.shape[0], traj_cols.shape[-1]])
                self.gym.add_lines(self.viewer, env_ptr, lines.shape[0], lines, curr_cols)
            
            # draw lines of the bbox
            cols = np.zeros((24, 3), dtype=np.float32) # 24 lines
            cols[:12] = [1.0, 0.0, 0.0] # red
            cols[12:] = [0.0, 1.0, 0.0] # green

            # transform bps from object local space to world space
            box_bps = self._box_bps.clone()
            box_pos = self._box_states[:, 0:3]
            box_rot = self._box_states[:, 3:7]
            box_pos_exp = torch.broadcast_to(box_pos.unsqueeze(-2), (box_pos.shape[0], box_bps.shape[1], box_pos.shape[1])) # (num_envs, 3) >> (num_envs, 8, 3)
            box_rot_exp = torch.broadcast_to(box_rot.unsqueeze(-2), (box_rot.shape[0], box_bps.shape[1], box_rot.shape[1])) # (num_envs, 4) >> (num_envs, 8, 4)
            box_bps_world_space = (quat_rotate(box_rot_exp.reshape(-1, 4), box_bps.reshape(-1, 3)) + box_pos_exp.reshape(-1, 3)).reshape(self.num_envs, 8, 3) # (num_envs, 8, 3)

            verts = torch.cat([
                box_bps_world_space[:, 0, :], box_bps_world_space[:, 1, :],
                box_bps_world_space[:, 1, :], box_bps_world_space[:, 2, :],
                box_bps_world_space[:, 2, :], box_bps_world_space[:, 3, :],
                box_bps_world_space[:, 3, :], box_bps_world_space[:, 0, :],

                box_bps_world_space[:, 4, :], box_bps_world_space[:, 5, :],
                box_bps_world_space[:, 5, :], box_bps_world_space[:, 6, :],
                box_bps_world_space[:, 6, :], box_bps_world_space[:, 7, :],
                box_bps_world_space[:, 7, :], box_bps_world_space[:, 4, :],

                box_bps_world_space[:, 0, :], box_bps_world_space[:, 4, :],
                box_bps_world_space[:, 1, :], box_bps_world_space[:, 5, :],
                box_bps_world_space[:, 2, :], box_bps_world_space[:, 6, :],
                box_bps_world_space[:, 3, :], box_bps_world_space[:, 7, :],
            ], dim=-1).cpu()

            # transform bps from object local space to world space
            tar_box_pos = self._tar_pos[:, 0:3] # (num_envs, 3)
            tar_box_pos_exp = torch.broadcast_to(tar_box_pos.unsqueeze(-2), (tar_box_pos.shape[0], box_bps.shape[1], tar_box_pos.shape[1])) # (num_envs, 3) >> (num_envs, 8, 3)
            tar_box_bps_world_space = (box_bps.reshape(-1, 3) + tar_box_pos_exp.reshape(-1, 3)).reshape(self.num_envs, 8, 3) # (num_envs, 8, 3)

            verts_tar_box = torch.cat([
                tar_box_bps_world_space[:, 0, :], tar_box_bps_world_space[:, 1, :],
                tar_box_bps_world_space[:, 1, :], tar_box_bps_world_space[:, 2, :],
                tar_box_bps_world_space[:, 2, :], tar_box_bps_world_space[:, 3, :],
                tar_box_bps_world_space[:, 3, :], tar_box_bps_world_space[:, 0, :],

                tar_box_bps_world_space[:, 4, :], tar_box_bps_world_space[:, 5, :],
                tar_box_bps_world_space[:, 5, :], tar_box_bps_world_space[:, 6, :],
                tar_box_bps_world_space[:, 6, :], tar_box_bps_world_space[:, 7, :],
                tar_box_bps_world_space[:, 7, :], tar_box_bps_world_space[:, 4, :],

                tar_box_bps_world_space[:, 0, :], tar_box_bps_world_space[:, 4, :],
                tar_box_bps_world_space[:, 1, :], tar_box_bps_world_space[:, 5, :],
                tar_box_bps_world_space[:, 2, :], tar_box_bps_world_space[:, 6, :],
                tar_box_bps_world_space[:, 3, :], tar_box_bps_world_space[:, 7, :],
            ], dim=-1).cpu() # (num_envs, 12*6)

            bbox_verts = torch.cat([verts, verts_tar_box], dim=-1).numpy()

            for i, env_ptr in enumerate(self.envs):
                curr_verts = bbox_verts[i]
                curr_verts = curr_verts.reshape([24, 6])
                self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, cols)

        return
    
    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)

        if self._is_eval:
            self._success_buf[env_ids] = 0
            self._precision_buf[env_ids] = 0 # not Inf
        
        env_ids_int32 = self._box_actor_ids
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
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

        # random sample a skill
        sk_id = torch.multinomial(self._skill_disc_prob, num_samples=1, replacement=True)
        sk_name = self._skill[sk_id]
        curr_motion_lib = self._motion_lib[sk_name]

        if (self._amp_obs_demo_buf is None):
            self._build_amp_obs_demo_buf(num_samples)
        else:
            assert(self._amp_obs_demo_buf.shape[0] == num_samples)
        
        motion_ids = curr_motion_lib.sample_motions(num_samples)
        
        # since negative times are added to these values in build_amp_obs_demo,
        # we shift them into the range [0 + truncate_time, end of clip]
        truncate_time = self.dt * (self._num_amp_obs_steps - 1)
        motion_times0 = curr_motion_lib.sample_time(motion_ids, truncate_time=truncate_time)
        motion_times0 += truncate_time

        amp_obs_demo = self.build_amp_obs_demo(motion_ids, motion_times0, curr_motion_lib)
        self._amp_obs_demo_buf[:] = amp_obs_demo.view(self._amp_obs_demo_buf.shape)
        amp_obs_demo_flat = self._amp_obs_demo_buf.view(-1, self.get_num_amp_obs())

        return amp_obs_demo_flat

    def build_amp_obs_demo(self, motion_ids, motion_times0, motion_lib):
        dt = self.dt

        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps])
        motion_times = motion_times0.unsqueeze(-1)
        time_steps = -dt * torch.arange(0, self._num_amp_obs_steps, device=self.device)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = motion_lib.get_motion_state(motion_ids, motion_times)
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

            with open(os.path.join(os.getcwd(), motion_file), 'r') as f:
                motion_config = yaml.load(f, Loader=yaml.SafeLoader)

            self._skill_categories = list(motion_config['motions'].keys()) # all skill names stored in the yaml file
            self._motion_lib = {}
            for skill in self._skill_categories:
                self._motion_lib[skill] = MotionLib(motion_file=motion_file,
                                                    skill=skill,
                                                    dof_body_ids=self._dof_body_ids,
                                                    dof_offsets=self._dof_offsets,
                                                    key_body_ids=self._key_body_ids.cpu().numpy(), 
                                                    device=self.device)
        else:
            raise NotImplementedError

        return
    
    def _reset_envs(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = {}
        self._reset_ref_motion_ids = {}
        self._reset_ref_motion_times = {}
        if (len(env_ids) > 0):
            self._reset_actors(env_ids)
            self._reset_boxes(env_ids)
            self._reset_task(env_ids)
            self._reset_env_tensors(env_ids)
            self._refresh_sim_tensors()
            self._compute_observations(env_ids)
            self._init_amp_obs(env_ids)

        return

    def _reset_boxes(self, env_ids):

        ###### reset boxes

        # 1. for skill is carryWith, the initial location of the box is from the reference box motion
        for sk_name in ["pickUp", "carryWith", "putDown"]:
            if self._reset_ref_env_ids.get(sk_name) is not None:
                if (len(self._reset_ref_env_ids[sk_name]) > 0):

                    curr_env_ids = self._reset_ref_env_ids[sk_name]

                    root_pos, root_rot = self._motion_lib[sk_name].get_obj_motion_state(
                        motion_ids=self._reset_ref_motion_ids[sk_name], 
                        motion_times=self._reset_ref_motion_times[sk_name]
                    )

                    on_ground_mask = (self._box_size[curr_env_ids, 2] / 2 > root_pos[:, 2])
                    root_pos[on_ground_mask, 2] = self._box_size[curr_env_ids[on_ground_mask], 2] / 2

                    self._box_states[curr_env_ids, 0:3] = root_pos
                    self._box_states[curr_env_ids, 3:7] = root_rot
                    self._box_states[curr_env_ids, 7:10] = 0.0
                    self._box_states[curr_env_ids, 10:13] = 0.0

        ###### reset boxes when skill is loco
        for sk_name in ["loco"]:
            if self._reset_ref_env_ids.get(sk_name) is not None:
                if (len(self._reset_ref_env_ids[sk_name]) > 0):

                    # boxes
                    curr_env_ids = self._reset_ref_env_ids[sk_name]
                    humanoid_init_facings = torch.zeros((len(curr_env_ids), 3), device=self.device)
                    humanoid_init_facings[:, 0] = 1.0
                    heading_rot = torch_utils.calc_heading_quat(self._humanoid_root_states[curr_env_ids, 3:7])
                    humanoid_curr_facings = quat_rotate(heading_rot, humanoid_init_facings)

                    self._box_states[curr_env_ids, 0:2] = self._humanoid_root_states[curr_env_ids, 0:2] \
                        + humanoid_curr_facings[:, 0:2] * (self._box_size[curr_env_ids, 0] / 2 + 0.1).unsqueeze(-1)
                    self._box_states[curr_env_ids, 2] = self._humanoid_root_states[curr_env_ids, 2] + self._box_size[curr_env_ids, 2] / 2 + 0.2

                    self._box_states[curr_env_ids, 3:7] = heading_rot
                    self._box_states[curr_env_ids, 7:10] = 0.0
                    self._box_states[curr_env_ids, 10:13] = 0.0

        return

    def _reset_actors(self, env_ids):
        if (self._state_init == HumanoidCompTrajCarry.StateInit.Default):
            self._reset_default(env_ids)
        elif (self._state_init == HumanoidCompTrajCarry.StateInit.Start
              or self._state_init == HumanoidCompTrajCarry.StateInit.Random):
            self._reset_ref_state_init(env_ids)
        elif (self._state_init == HumanoidCompTrajCarry.StateInit.Hybrid):
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
        sk_ids = torch.multinomial(self._skill_init_prob, num_samples=env_ids.shape[0], replacement=True)

        for uid, sk_name in enumerate(self._skill):
            curr_motion_lib = self._motion_lib[sk_name]
            curr_env_ids = env_ids[(sk_ids == uid).nonzero().squeeze(-1)] # be careful!!!

            if len(curr_env_ids) > 0:

                num_envs = curr_env_ids.shape[0]
                motion_ids = curr_motion_lib.sample_motions(num_envs)

                if (self._state_init == HumanoidCompTrajCarry.StateInit.Random
                    or self._state_init == HumanoidCompTrajCarry.StateInit.Hybrid):
                    motion_times = curr_motion_lib.sample_time_rsi(motion_ids) # avoid times with serious self-penetration
                elif (self._state_init == HumanoidCompTrajCarry.StateInit.Start):
                    motion_times = torch.zeros(num_envs, device=self.device)
                else:
                    assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

                root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
                    = curr_motion_lib.get_motion_state(motion_ids, motion_times)

                self._set_env_state(env_ids=curr_env_ids, 
                                    root_pos=root_pos, 
                                    root_rot=root_rot, 
                                    dof_pos=dof_pos, 
                                    root_vel=root_vel, 
                                    root_ang_vel=root_ang_vel, 
                                    dof_vel=dof_vel)

                self._reset_ref_env_ids[sk_name] = curr_env_ids
                self._reset_ref_motion_ids[sk_name] = motion_ids
                self._reset_ref_motion_times[sk_name] = motion_times

                # update buffer for kinematic humanoid state
                body_pos, body_rot, body_vel, body_ang_vel \
                    = curr_motion_lib.get_motion_state_max(motion_ids, motion_times)
                self._kinematic_humanoid_rigid_body_states[curr_env_ids] = torch.cat((body_pos, body_rot, body_vel, body_ang_vel), dim=-1)

                self._every_env_init_dof_pos[curr_env_ids] = dof_pos # for "enableTrackInitState"

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

        for i, sk_name in enumerate(self._skill):
            if self._reset_ref_env_ids.get(sk_name) is not None:
                if (len(self._reset_ref_env_ids[sk_name]) > 0):
                    self._init_amp_obs_ref(self._reset_ref_env_ids[sk_name], self._reset_ref_motion_ids[sk_name],
                                           self._reset_ref_motion_times[sk_name], sk_name)

        return

    def _init_amp_obs_default(self, env_ids):
        curr_amp_obs = self._curr_amp_obs_buf[env_ids].unsqueeze(-2)
        self._hist_amp_obs_buf[env_ids] = curr_amp_obs
        return

    def _init_amp_obs_ref(self, env_ids, motion_ids, motion_times, skill_name):
        dt = self.dt
        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps - 1])
        motion_times = motion_times.unsqueeze(-1)
        time_steps = -dt * (torch.arange(0, self._num_amp_obs_steps - 1, device=self.device) + 1)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib[skill_name].get_motion_state(motion_ids, motion_times)
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
                                                           self._enable_early_termination, self._termination_heights,
                                                           self._enable_fall_termination, self._box_states[:, 2], self._box_size,)
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
def compute_carry_observations(root_states, box_states, box_bps, tar_pos, enableBboxObs):
    # type: (Tensor, Tensor, Tensor, Tensor, bool) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot) # (num_envs, 4)

    box_pos = box_states[:, 0:3]
    box_rot = box_states[:, 3:7]
    box_vel = box_states[:, 7:10]
    box_ang_vel = box_states[:, 10:13]
    
    local_box_pos = box_pos - root_pos
    local_box_pos = quat_rotate(heading_rot, local_box_pos)

    local_box_rot = quat_mul(heading_rot, box_rot)
    local_box_rot_obs = torch_utils.quat_to_tan_norm(local_box_rot)

    local_box_vel = quat_rotate(heading_rot, box_vel)
    local_box_ang_vel = quat_rotate(heading_rot, box_ang_vel)

    # compute observations for bounding points of the box
    ## transform from object local space to world space
    box_pos_exp = torch.broadcast_to(box_pos.unsqueeze(-2), (box_pos.shape[0], box_bps.shape[1], box_pos.shape[1])) # (num_envs, 3) >> (num_envs, 8, 3)
    box_rot_exp = torch.broadcast_to(box_rot.unsqueeze(-2), (box_rot.shape[0], box_bps.shape[1], box_rot.shape[1])) # (num_envs, 4) >> (num_envs, 8, 4)
    box_bps_world_space = quat_rotate(box_rot_exp.reshape(-1, 4), box_bps.reshape(-1, 3)) + box_pos_exp.reshape(-1, 3) # (num_envs*8, 3)

    ## transform from world space to humanoid local space
    heading_rot_exp = torch.broadcast_to(heading_rot.unsqueeze(-2), (heading_rot.shape[0], box_bps.shape[1], heading_rot.shape[1]))
    root_pos_exp = torch.broadcast_to(root_pos.unsqueeze(-2), (root_pos.shape[0], box_bps.shape[1], root_pos.shape[1]))
    box_bps_local_space = quat_rotate(heading_rot_exp.reshape(-1, 4), box_bps_world_space - root_pos_exp.reshape(-1, 3)) # (num_envs*8, 3)

    # task obs
    local_tar_pos = quat_rotate(heading_rot, tar_pos - root_pos) # 3d xyz

    if enableBboxObs:
        obs = torch.cat([local_box_vel, local_box_ang_vel, local_box_pos, local_box_rot_obs, box_bps_local_space.reshape(root_pos.shape[0], -1), local_tar_pos], dim=-1)
    else:
        obs = torch.cat([local_box_vel, local_box_ang_vel, local_box_pos, local_box_rot_obs, local_tar_pos], dim=-1)

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
def compute_handheld_reward(humanoid_rigid_body_pos, box_pos, hands_ids, only_height):
    # type: (Tensor, Tensor, Tensor, bool) -> Tensor
    if only_height:
        hands2box_pos_err = torch.sum((humanoid_rigid_body_pos[:, hands_ids, 2] - box_pos[:, 2].unsqueeze(-1)) ** 2, dim=-1) # height
    else:
        hands2box_pos_err = torch.sum((humanoid_rigid_body_pos[:, hands_ids].mean(dim=1) - box_pos) ** 2, dim=-1) # xyz
    hands2box = torch.exp(-5.0 * hands2box_pos_err)

    root_pos = humanoid_rigid_body_pos[:, 0, :]
    box2human = torch.sum((box_pos[..., 0:2] - root_pos[..., 0:2]) ** 2, dim=-1) # 2d
    hands2box[box2human > 0.7 ** 2] = 0 # disable this reward when the box is not close to the humanoid

    return hands2box

@torch.jit.script
def compute_traj_reward(root_pos, tar_pos, box_pos, humanoid_rigid_body_pos, hands_ids):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    pos_err_scale = 2.0

    pos_diff = tar_pos[..., 0:2] - root_pos[..., 0:2]
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)

    pos_reward = torch.exp(-pos_err_scale * pos_err)

    box2human = torch.sum((box_pos[..., 0:2] - root_pos[..., 0:2]) ** 2, dim=-1) # 2d
    pos_reward[box2human > 0.7 ** 2] = 0 # disable this reward when the box is not close to the humanoid

    box_heights = box_pos[..., 2]
    hands_heights = humanoid_rigid_body_pos[:, hands_ids, 2].mean(dim=-1)
    box2hand = hands_heights - box_heights
    pos_reward[box2hand > 0.3] = 0.0
    pos_reward[box2hand < -1.0] = 0.0 # avoid to kick up

    reward = pos_reward

    return reward

@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                           tar_pos, max_episode_length, fail_dist,
                           enable_early_termination, termination_heights,
                           enable_box_fall_termination, box_heights, box_sizes,):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, bool, Tensor, bool, Tensor, Tensor) -> Tuple[Tensor, Tensor]
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

        if enable_box_fall_termination:
            box_has_fallen = box_heights < (box_sizes[..., 2] / 2.0) + 0.2
            box_has_flied = box_heights > 3.0
            box_terminated = torch.logical_or(box_has_fallen, box_has_flied)
            terminated = torch.where(box_terminated, torch.ones_like(reset_buf), terminated)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated
