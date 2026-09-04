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
import pickle
import trimesh

from isaacgym import gymapi
from isaacgym import gymtorch

from env.tasks.humanoid import Humanoid, dof_to_obs
from utils import gym_util
from utils.motion_lib import MotionLib
from isaacgym.torch_utils import *
from isaacgym.terrain_utils import *

from utils import torch_utils
from utils import traj_generator

from tqdm import tqdm
from scipy import ndimage

class HumanoidAdaptTrajGround2Terrain(Humanoid):
    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        
        # prepare some necessary variables
        self.device = "cpu"
        self.device_type = device_type
        if device_type == "cuda" or device_type == "GPU":
            self.device = "cuda" + ":" + str(device_id)
        self.num_envs = cfg["env"]["numEnvs"]

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

        ##### debug
        self._flag_small_terrain = cfg["env"]["flagSmallTerrain"]
        self.show_sensors = cfg["env"]["flagShowSensors"]
        if self.num_envs > 1:
            self.show_sensors = False # only support single env for now

        self._is_eval = cfg["args"].eval
        if self._is_eval:
            self._flag_small_terrain = True

        self._is_test = cfg["args"].test
        if self._is_test:
            self._flag_small_terrain = True

        ##### height map sensor
        self.sensor_extent = cfg["env"].get("sensor_extent", 2)
        self.sensor_res = cfg["env"].get("sensor_res", 32)

        self.square_height_points = self.init_square_height_points()
        self.terrain_obs_type = cfg['env'].get("terrain_obs_type", "square")
        self.terrain_obs = cfg['env'].get("terrain_obs", False)
        self.terrain_obs_root = cfg['env'].get("terrain_obs_root", "pelvis")
        if self.terrain_obs_type == "square":
            self.height_points = self.square_height_points
        else:
            raise NotImplementedError

        self.center_height_points = self.init_center_height_points()

        # manage multi task obs
        self._num_tasks = 3
        task_obs_size_traj = 2 * self._num_traj_samples
        task_obs_size_height_obs = self.num_height_points
        self._each_subtask_obs_size = [
            task_obs_size_height_obs, # new height map
            task_obs_size_traj, # new traj
            task_obs_size_traj, # old traj
        ]
        self._multiple_task_names = ["new_extra", "new_traj", "old_traj"]
        self._enable_task_mask_obs = False

        # configs for amp
        state_init = cfg["env"]["stateInit"]
        self._state_init = HumanoidAdaptTrajGround2Terrain.StateInit[state_init]
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
        
        if (not self.headless) and self.show_sensors:
            self._build_sensor_state_tensors()

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
            "major_task_name": "traj",
            "has_extra": True,
        }

        return info
    
    def init_square_height_points(self):
        # 4mx4m square
        y =  torch.tensor(np.linspace(-self.sensor_extent, self.sensor_extent, self.sensor_res), device=self.device, requires_grad=False)
        x = torch.tensor(np.linspace(-self.sensor_extent, self.sensor_extent,
                                     self.sensor_res),
                         device=self.device,
                         requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs,
                             self.num_height_points,
                             3,
                             device=self.device,
                             requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def init_center_height_points(self):
        # center_height_points
        y =  torch.tensor(np.linspace(-0.2, 0.2, 3),device=self.device,requires_grad=False)
        x =  torch.tensor(np.linspace(-0.1, 0.1, 3),device=self.device,requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_center_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs,
                             self.num_center_height_points,
                             3,
                             device=self.device,
                             requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points
    
    def get_center_heights(self, root_states, env_ids=None):
        base_quat = root_states[:, 3:7]
        if self.cfg["env"]["terrain"]["terrainType"] == 'plane':
            return torch.zeros(self.num_envs,
                               self.num_center_height_points,
                               device=self.device,
                               requires_grad=False)
        elif self.cfg["env"]["terrain"]["terrainType"] == 'none':
            raise NameError("Can't measure height with terrain type 'none'")
        
        heading_rot = torch_utils.calc_heading_quat(base_quat)

        if env_ids is None:
            points = quat_apply(
                heading_rot.repeat(1, self.num_center_height_points,).reshape(-1, 4),
                self.center_height_points) + (root_states[:, :3]).unsqueeze(1)
        else:
            points = quat_apply(
                heading_rot.repeat(1, self.num_center_height_points,).reshape(-1, 4),
                self.center_height_points[env_ids]) + (
                    root_states[:, :3]).unsqueeze(1)

        heights = self.terrain.sample_height_points(points.clone(), env_ids=env_ids)
        num_envs = self.num_envs if env_ids is None else len(env_ids)

        return heights.view(num_envs, -1)
    
    def get_heights(self, root_states, env_ids=None):

        base_quat = root_states[:, 3:7]
        if self.cfg["env"]["terrain"]["terrainType"] == 'plane':
            return torch.zeros(self.num_envs,
                               self.num_height_points,
                               device=self.device,
                               requires_grad=False)
        elif self.cfg["env"]["terrain"]["terrainType"] == 'none':
            raise NameError("Can't measure height with terrain type 'none'")

        heading_rot = torch_utils.calc_heading_quat(base_quat)

        if env_ids is None:
            points = quat_apply(
                heading_rot.repeat(1, self.num_height_points).reshape(-1, 4),
                self.height_points) + (root_states[:, :3]).unsqueeze(1)
        else:
            points = quat_apply(
                heading_rot.repeat(1, self.num_height_points).reshape(-1, 4),
                self.height_points[env_ids]) + (
                    root_states[:, :3]).unsqueeze(1)

        heights = self.terrain.sample_height_points(
            points.clone(),
            env_ids=env_ids,
        )
        num_envs = self.num_envs if env_ids is None else len(env_ids)

        return heights.view(num_envs, -1)

    def _create_ground_plane(self):
        self.create_training_ground()
        return
    
    def create_training_ground(self):
        if self._flag_small_terrain:
            self.cfg["env"]["terrain"]['mapLength'] = 8
            self.cfg["env"]["terrain"]['mapWidth'] = 8

        self.terrain = Terrain(self.cfg["env"]["terrain"],
                               num_robots=self.num_envs,
                               device=self.device)

        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]
        tm_params.transform.p.x = 0
        tm_params.transform.p.y = 0
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg["env"]["terrain"]["staticFriction"]
        tm_params.dynamic_friction = self.cfg["env"]["terrain"]["dynamicFriction"]
        tm_params.restitution = self.cfg["env"]["terrain"]["restitution"]
        self.gym.add_triangle_mesh(self.sim,
                                   self.terrain.vertices.flatten(order='C'),
                                   self.terrain.triangles.flatten(order='C'),
                                   tm_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
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
            obs_size += sum(self._each_subtask_obs_size[:-1]) # exclude redundant one
        return obs_size

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._prev_root_pos[:] = self._humanoid_root_states[..., 0:3]
        return

    def _update_marker(self):
        traj_samples = self._fetch_traj_samples() # (num_envs, 10, 3)

        dummy_root_rot = torch.zeros((self.num_envs, traj_samples.shape[1], 4), dtype=torch.float32, device=self.device)
        dummy_root_rot[..., -1] = 1.0

        dummy_root_states = torch.cat([traj_samples, dummy_root_rot], dim=-1).reshape(-1, 3 + 4)

        env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids_exp = torch.broadcast_to(env_ids.unsqueeze(-1), (self.num_envs, traj_samples.shape[1])).reshape(-1)

        center_height = self.get_center_heights(dummy_root_states, env_ids_exp).mean(dim=-1) # 当前人体root 2d pos的高度
        center_height = center_height.reshape(self.num_envs, -1)

        traj_samples[..., 2] = center_height + self._char_h
        
        self._traj_marker_pos[:] = traj_samples
        # self._traj_marker_pos[..., 2] = self._humanoid_root_states[..., 2].unsqueeze(-1)

        actor_ids = torch.cat([self._traj_marker_actor_ids], dim=-1)

        if (not self.headless) and self.show_sensors:

            base_quat = self._humanoid_root_states[:, 3:7]
            heading_rot = torch_utils.calc_heading_quat(base_quat)

            points = quat_apply(
                heading_rot.repeat(1, self.num_height_points).reshape(-1, 4),
                self.height_points) + (self._humanoid_root_states[:, :3]).unsqueeze(1)
            
            env_ids = torch.arange(0, self.num_envs, device=self.device, dtype=torch.long)
            measured_heights = self.get_heights(root_states=self._humanoid_root_states, env_ids=env_ids)

            self._sensor_pos[:] = points
            self._sensor_pos[..., -1] = measured_heights

            actor_ids = torch.cat([actor_ids, self._sensor_actor_ids], dim=-1)

        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(actor_ids), len(actor_ids))
        return
    
    def _create_envs(self, num_envs, spacing, num_per_row):
        if (not self.headless):
            self._traj_marker_handles = [[] for _ in range(self.num_envs)]
            self._load_marker_asset()
        
        if (not self.headless) and self.show_sensors:
            self._sensor_handles = [[] for _ in range(self.num_envs)]

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
        # super()._build_env(env_id, env_ptr, humanoid_asset)

        col_group = env_id
        col_filter = self._get_humanoid_collision_filter()
        segmentation_id = 0

        start_pose = gymapi.Transform()
        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        if (asset_file == "mjcf/amp_humanoid.xml"):
            self._char_h = 0.89 # perfect number
        elif (asset_file == "mjcf/phys_humanoid.xml") or (asset_file == "mjcf/phys_humanoid_v2.xml"):
            self._char_h = 0.92 # perfect number
        elif (asset_file == "mjcf/phys_humanoid_v3.xml") or (asset_file == "mjcf/phys_humanoid_v3_box_foot.xml"):
            self._char_h = 0.94
        else:
            print("Unsupported character config file: {s}".format(asset_file))
            assert(False)

        pos = torch.tensor(get_axis_params(self._char_h,
                                           self.up_axis_idx)).to(self.device)
        pos[:2] += torch_rand_float(
            -1., 1., (2, 1), device=self.device).squeeze(
                1)  # ZL: segfault if we do not randomize the position

        start_pose.p = gymapi.Vec3(*pos)
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        humanoid_handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "humanoid", col_group, col_filter, segmentation_id)

        self.gym.enable_actor_dof_force_sensors(env_ptr, humanoid_handle)

        for j in range(self.num_bodies):
            self.gym.set_rigid_body_color(env_ptr, humanoid_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.54, 0.85, 0.2))

        if (self._pd_control):
            dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
            dof_prop["driveMode"] = gymapi.DOF_MODE_POS
            self.gym.set_actor_dof_properties(env_ptr, humanoid_handle, dof_prop)

        self.humanoid_handles.append(humanoid_handle)
        
        if (not self.headless):
            self._build_marker(env_id, env_ptr)

        if (not self.headless) and self.show_sensors:
            self._build_sensor(env_id, env_ptr)

        return
    
    def _build_sensor(self, env_id, env_ptr):
        col_group = self.num_envs + 100
        col_filter = 1
        segmentation_id = 0
        default_pose = gymapi.Transform()

        for i in range(self.num_height_points):
            marker_handle = self.gym.create_actor(env_ptr, self._marker_asset,
                                                  default_pose, "marker",
                                                  col_group, col_filter, segmentation_id)
            self.gym.set_actor_scale(env_ptr, marker_handle, 0.2)

            if i >= (self.num_height_points - self.sensor_res):
                color = gymapi.Vec3(0.0, 0.0, 1.0)
            else:
                color = gymapi.Vec3(0.0, 1.0, 0.0)
            self.gym.set_rigid_body_color(env_ptr, marker_handle, 0,
                                          gymapi.MESH_VISUAL,
                                          color)
            self._sensor_handles[env_id].append(marker_handle)

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
    
    def _build_sensor_state_tensors(self):
        num_actors = self._root_states.shape[0] // self.num_envs

        self._sensor_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 11:(11 + self.num_height_points), :]
        
        self._sensor_pos = self._sensor_states[..., :3]
        
        self._sensor_actor_ids = self._humanoid_actor_ids.unsqueeze(-1) + to_torch(self._sensor_handles, dtype=torch.int32, device=self.device)
        self._sensor_actor_ids = self._sensor_actor_ids.flatten()
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

        if self.terrain_obs:

            if self.terrain_obs_root == "pelvis":
                measured_heights = self.get_heights(root_states=root_states, env_ids=env_ids)
            else:
                raise NotImplementedError

            if self.cfg['env'].get("localHeightObs", False):
                heights = measured_heights - root_states[..., 2].unsqueeze(-1)
                heights = torch.clip(heights, -3, 3.)

            obs = torch.cat([heights, obs], dim=1)

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
            verts[..., 2] = self._humanoid_root_states[i, 2]
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
        elif (asset_file == "mjcf/phys_humanoid.xml") or (asset_file == "mjcf/phys_humanoid_v2.xml") or (asset_file == "mjcf/phys_humanoid_v3.xml") or (asset_file == "mjcf/phys_humanoid_v3_box_foot.xml"):
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
        if (self._state_init == HumanoidAdaptTrajGround2Terrain.StateInit.Default):
            self._reset_default(env_ids)
        elif (self._state_init == HumanoidAdaptTrajGround2Terrain.StateInit.Start
              or self._state_init == HumanoidAdaptTrajGround2Terrain.StateInit.Random):
            self._reset_ref_state_init(env_ids)
        elif (self._state_init == HumanoidAdaptTrajGround2Terrain.StateInit.Hybrid):
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
        
        if (self._state_init == HumanoidAdaptTrajGround2Terrain.StateInit.Random
            or self._state_init == HumanoidAdaptTrajGround2Terrain.StateInit.Hybrid):
            motion_times = self._motion_lib.sample_time(motion_ids)
        elif (self._state_init == HumanoidAdaptTrajGround2Terrain.StateInit.Start):
            motion_times = torch.zeros(num_envs, device=self.device)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
        
        if self._is_eval:
            root_pos = self._initial_humanoid_root_states[env_ids, 0:3]
            root_rot = self._initial_humanoid_root_states[env_ids, 3:7]
            root_vel = self._initial_humanoid_root_states[env_ids, 7:10]
            root_ang_vel = self._initial_humanoid_root_states[env_ids, 10:13]
            dof_pos = self._initial_dof_pos[env_ids]
            dof_vel = self._initial_dof_vel[env_ids]
        
        new_root_xy = self.terrain.sample_valid_locations(self.num_envs, env_ids)

        root_pos[:, 0:2] = new_root_xy

        root_states = torch.cat([root_pos, root_rot], dim=1)

        center_height = self.get_center_heights(root_states, env_ids=env_ids).mean(dim=-1) # 当前人体root 2d pos的高度

        root_pos[:, 2] = center_height + self._char_h + 0.05

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

        # body_height = rigid_body_pos[..., 2]
        # fall_height = body_height < termination_heights
        # fall_height[:, contact_body_ids] = False
        # fall_height = torch.any(fall_height, dim=-1)

        # has_fallen = torch.logical_and(fall_contact, fall_height)

        has_fallen = fall_contact

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

class Terrain:
    def __init__(self, cfg, num_robots, device) -> None:

        self.type = cfg["terrainType"]
        self.device = device
        if self.type in ["none", 'plane']:
            return
        self.horizontal_scale = 0.1 # resolution 0.1
        self.vertical_scale = 0.005
        self.border_size = 50 # 单位是m
        self.env_length = cfg["mapLength"] # 每个小env 每小块地形的长和宽 单位是m
        self.env_width = cfg["mapWidth"]
        self.proportions = [
            np.sum(cfg["terrainProportions"][:i + 1])
            for i in range(len(cfg["terrainProportions"]))
        ]

        self.env_rows = cfg["numLevels"]
        self.env_cols = cfg["numTerrains"]
        self.num_maps = self.env_rows * self.env_cols # 一共100块地形
        self.env_origins = np.zeros((self.env_rows, self.env_cols, 3))

        self.width_per_env_pixels = int(self.env_width / self.horizontal_scale) # 每个小env的宽度的单位是0.1m，宽8m，所以有80个pixels
        self.length_per_env_pixels = int(self.env_length /
                                         self.horizontal_scale)

        self.border = int(self.border_size / self.horizontal_scale)
        self.tot_cols = int(
            self.env_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(
            self.env_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        self.walkable_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        if cfg["curriculum"]:
            self.curiculum(num_robots,
                           num_terrains=self.env_cols,
                           num_levels=self.env_rows)
        else:
            self.randomized_terrain()
        self.heightsamples = torch.from_numpy(self.height_field_raw).to(self.device) # ZL: raw height field, first dimension is x, second is y
        self.walkable_field = torch.from_numpy(self.walkable_field_raw).to(self.device)
        self.vertices, self.triangles = convert_heightfield_to_trimesh(self.height_field_raw, self.horizontal_scale, self.vertical_scale, cfg["slopeTreshold"])
        self.sample_extent_x = int((self.tot_rows - self.border * 2) * self.horizontal_scale)
        self.sample_extent_y = int((self.tot_cols - self.border * 2) * self.horizontal_scale)

        coord_x, coord_y = torch.where(self.walkable_field == 0)
        coord_x_scale = coord_x * self.horizontal_scale # 转换为真实尺度
        coord_y_scale = coord_y * self.horizontal_scale
        walkable_subset = torch.logical_and(
                torch.logical_and(coord_y_scale < coord_y_scale.max() - self.border * self.horizontal_scale, coord_x_scale < coord_x_scale.max() - self.border * self.horizontal_scale),
                torch.logical_and(coord_y_scale > coord_y_scale.min() + self.border * self.horizontal_scale, coord_x_scale > coord_x_scale.min() +  self.border * self.horizontal_scale)
            )
        # import ipdb; ipdb.set_trace()
        # joblib.dump(self.walkable_field_raw, "walkable_field.pkl")

        self.coord_x_scale = coord_x_scale[walkable_subset]
        self.coord_y_scale = coord_y_scale[walkable_subset]
        self.num_samples = self.coord_x_scale.shape[0]


    def sample_valid_locations(self, max_num_envs, env_ids):
        
        num_envs = env_ids.shape[0]
        idxes = np.random.randint(0, self.num_samples, size=num_envs)
        valid_locs = torch.stack([self.coord_x_scale[idxes], self.coord_y_scale[idxes]], dim = -1)

        return valid_locs

    def world_points_to_map(self, points):
        points = (points / self.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.heightsamples.shape[0] - 2)
        py = torch.clip(py, 0, self.heightsamples.shape[1] - 2)
        return px, py


    def sample_height_points(self, points, env_ids = None):
        B, N, C = points.shape
        px, py = self.world_points_to_map(points)
        heightsamples = self.heightsamples.clone()
        if env_ids is None:
            env_ids = torch.arange(B).to(points).long()

        heights1 = heightsamples[px, py]
        heights2 = heightsamples[px + 1, py + 1] # 为啥要各加1?
        heights = torch.min(heights1, heights2)

        return heights * self.vertical_scale

    def randomized_terrain(self):
        for k in range(self.num_maps): # 5x20 共100块小env(地形块)
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.env_rows, self.env_cols))

            # Heightfield coordinate system from now on
            start_x = self.border + i * self.length_per_env_pixels
            end_x = self.border + (i + 1) * self.length_per_env_pixels
            start_y = self.border + j * self.width_per_env_pixels
            end_y = self.border + (j + 1) * self.width_per_env_pixels

            terrain = SubTerrain("terrain",
                                 width=self.width_per_env_pixels,
                                 length=self.width_per_env_pixels,
                                 vertical_scale=self.vertical_scale,
                                 horizontal_scale=self.horizontal_scale)
            choice = np.random.uniform(0, 1)
            difficulty = np.random.uniform(0.1, 1)
            slope = difficulty * 0.7
            discrete_obstacles_height = 0.025 + difficulty * 0.15
            stepping_stones_size = 2 - 1.8 * difficulty
            step_height = 0.05 + 0.175 * difficulty
            if choice < self.proportions[0]:
                if choice < 0.05:
                    slope *= -1
                pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            elif choice < self.proportions[1]:
                if choice < 0.15:
                    slope *= -1
                pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
                random_uniform_terrain(terrain,
                                       min_height=-0.1,
                                       max_height=0.1,
                                       step=0.025,
                                       downsampled_scale=0.2)
            elif choice < self.proportions[3]:
                if choice < self.proportions[2]:
                    step_height *= -1
                pyramid_stairs_terrain(terrain,
                                       step_width=0.31,
                                       step_height=step_height,
                                       platform_size=3.)
            elif choice < self.proportions[4]:
                discrete_obstacles_terrain(terrain,
                                           discrete_obstacles_height,
                                           1.,
                                           2.,
                                           40,
                                           platform_size=3.)
            elif choice < self.proportions[5]:
                stepping_stones_terrain(terrain,
                                        stone_size=stepping_stones_size,
                                        stone_distance=0.1,
                                        max_height=0.,
                                        platform_size=3.)
            elif choice < self.proportions[6]:
                # plain walking terrain
                pass
            elif choice < self.proportions[7]:
                # plain walking terrain
                pass

            self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

            env_origin_x = (i + 0.5) * self.env_length
            env_origin_y = (j + 0.5) * self.env_width
            x1 = int((self.env_length / 2. - 1) / self.horizontal_scale)
            x2 = int((self.env_length / 2. + 1) / self.horizontal_scale)
            y1 = int((self.env_width / 2. - 1) / self.horizontal_scale)
            y2 = int((self.env_width / 2. + 1) / self.horizontal_scale)
            env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * self.vertical_scale # 乘上vertical_scale后才是真实尺度
            self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
        self.walkable_field_raw = ndimage.binary_dilation(self.walkable_field_raw, iterations=3).astype(int)

    def curiculum(self, num_robots, num_terrains, num_levels):
        num_robots_per_map = int(num_robots / num_terrains)
        left_over = num_robots % num_terrains
        idx = 0
        for j in tqdm(range(num_terrains)):
            for i in range(num_levels):
                terrain = SubTerrain("terrain",
                                     width=self.width_per_env_pixels,
                                     length=self.width_per_env_pixels,
                                     vertical_scale=self.vertical_scale,
                                     horizontal_scale=self.horizontal_scale)
                difficulty = i / num_levels
                choice = j / num_terrains

                slope = difficulty * 0.7
                step_height = 0.05 + 0.175 * difficulty
                discrete_obstacles_height = 0.025 + difficulty * 0.15
                stepping_stones_size = 2 - 1.8 * difficulty

                start_x = self.border + i * self.length_per_env_pixels
                end_x = self.border + (i + 1) * self.length_per_env_pixels
                start_y = self.border + j * self.width_per_env_pixels
                end_y = self.border + (j + 1) * self.width_per_env_pixels

                if choice < self.proportions[0]:
                    if choice < 0.05:
                        slope *= -1
                    pyramid_sloped_terrain(terrain,
                                           slope=slope,
                                           platform_size=3.)
                elif choice < self.proportions[1]:
                    if choice < 0.15:
                        slope *= -1
                    pyramid_sloped_terrain(terrain,
                                           slope=slope,
                                           platform_size=3.)
                    random_uniform_terrain(terrain,
                                           min_height=-0.1,
                                           max_height=0.1,
                                           step=0.025,
                                           downsampled_scale=0.2)
                elif choice < self.proportions[3]:
                    if choice < self.proportions[2]:
                        step_height *= -1
                    pyramid_stairs_terrain(terrain,
                                           step_width=0.31,
                                           step_height=step_height,
                                           platform_size=3.)
                elif choice < self.proportions[4]:
                    discrete_obstacles_terrain(terrain,
                                               discrete_obstacles_height,
                                               1.,
                                               2.,
                                               40,
                                               platform_size=3.)
                elif choice < self.proportions[5]:
                    stepping_stones_terrain(terrain,
                                            stone_size=stepping_stones_size,
                                            stone_distance=0.1,
                                            max_height=0.,
                                            platform_size=3.)
                elif choice < self.proportions[6]:
                    # plain walking terrain
                    pass
                elif choice < self.proportions[7]:
                    # plain walking terrain
                    pass

                # Heightfield coordinate system
                self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

                robots_in_map = num_robots_per_map
                if j < left_over:
                    robots_in_map += 1

                env_origin_x = (i + 0.5) * self.env_length
                env_origin_y = (j + 0.5) * self.env_width
                x1 = int((self.env_length / 2. - 1) / self.horizontal_scale)
                x2 = int((self.env_length / 2. + 1) / self.horizontal_scale)
                y1 = int((self.env_width / 2. - 1) / self.horizontal_scale)
                y2 = int((self.env_width / 2. + 1) / self.horizontal_scale)
                env_origin_z = np.max(
                    terrain.height_field_raw[x1:x2,
                                             y1:y2]) * self.vertical_scale
                self.env_origins[i, j] = [
                    env_origin_x, env_origin_y, env_origin_z
                ]

        self.walkable_field_raw = ndimage.binary_dilation(self.walkable_field_raw, iterations=3).astype(int)
