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
import yaml
import trimesh
import pickle

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

class HumanoidAdaptCarryGround2Terrain(Humanoid):
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
        self._only_vel_reward = cfg["env"]["onlyVelReward"]
        self._only_height_handheld_reward = cfg["env"]["onlyHeightHandHeldReward"]

        self._box_vel_penalty = cfg["env"]["box_vel_penalty"]
        self._box_vel_pen_coeff = cfg["env"]["box_vel_pen_coeff"]
        self._box_vel_pen_thre = cfg["env"]["box_vel_pen_threshold"]

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
        self._build_random_density = box_cfg["build"]["randomDensity"]
        self._build_test_sizes = box_cfg["build"]["testSizes"]

        self._reset_random_rot = box_cfg["reset"]["randomRot"]
        self._reset_random_height = box_cfg["reset"]["randomHeight"]
        self._reset_random_height_prob = box_cfg["reset"]["randomHeightProb"]
        self._reset_maxTopSurfaceHeight = box_cfg["reset"]["maxTopSurfaceHeight"]
        self._reset_minBottomSurfaceHeight = box_cfg["reset"]["minBottomSurfaceHeight"]

        self._enable_bbox_obs = box_cfg["obs"]["enableBboxObs"]

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
        task_obs_size_carry = 3 + \
              3 + 6 + 3 + 3 + \
              3 * 8
        task_obs_size_height_obs = self.num_height_points
        self._each_subtask_obs_size = [
            task_obs_size_height_obs, # new height map
            task_obs_size_carry, # new carry
            task_obs_size_carry, # old carry
        ]
        self._multiple_task_names = ["new_extra", "new_carry", "old_carry"]
        self._enable_task_mask_obs = False

        # configs for amp
        state_init = cfg["env"]["stateInit"]
        self._state_init = HumanoidAdaptCarryGround2Terrain.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        assert(self._num_amp_obs_steps >= 2)

        self._power_reward = cfg["env"]["power_reward"]
        self._power_coefficient = cfg["env"]["power_coefficient"]
        self._power_dof_ids = [
            0, 1, 2, # abdomen
            3, 4, 5, # neck
            6, 7, 8, # right_shoulder
            9, # right_elbow
            10, 11, 12, # left_shoulder
            13, # left_elbow
            14, 15, 16, # right_hip
            17, 18, 19, # right_knee
            20, 21, 22, # right_ankle
            23, 24, 25, # left_hip
            26, 27, 28, # left_knee
            29, 30, 31, # left_ankle
        ]

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = {} # to enable multi-skill reference init, use dict instead of list
        self._reset_ref_motion_ids = {}
        self._reset_ref_motion_times = {}

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

        self._power_dof_ids = to_torch(self._power_dof_ids, device=self.device, dtype=torch.long)

        if (not self.headless):
            self._build_marker_state_tensors()
        
        if (not self.headless) and self.show_sensors:
            self._build_sensor_state_tensors()
        
        # tensors for platforms
        if self._reset_random_height:
            self._build_platforms_state_tensors()

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
            "major_task_name": "carry",
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
        self._prev_box_pos[:] = self._box_states[..., 0:3]
        return

    def _update_marker(self):
        
        self._marker_pos[:, :] = self._tar_pos[:, :]

        actor_ids = torch.cat([self._marker_actor_ids, self._box_actor_ids], dim=0)
        if self._reset_random_height:
            # env has two platforms
            actor_ids = torch.cat([actor_ids, self._platform_actor_ids, self._tar_platform_actor_ids], dim=0)

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
            self._marker_handles = []
            self._load_marker_asset()
        
        if (not self.headless) and self.show_sensors:
            self._sensor_handles = [[] for _ in range(self.num_envs)]
        
        if self._reset_random_height:
            self._platform_handles = []
            self._tar_platform_handles = []
            self._load_platform_asset()

        self._box_handles = []
        self._box_masses = []
        self._load_box_asset()

        super()._create_envs(num_envs, spacing, num_per_row)

        self._box_masses = to_torch(self._box_masses, device=self.device, dtype=torch.float32)
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
    
    def _load_platform_asset(self):
        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 1.0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._platform_height = 0.02
        self._platform_asset = self.gym.create_box(self.sim, 0.4, 0.4, self._platform_height, asset_options)

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

        # randomize mass
        self._box_density = torch.zeros((self.num_envs), dtype=torch.float32, device=self.device)
        if self._build_random_density:
            dist = torch.distributions.uniform.Uniform(torch.tensor([300.0], device=self.device), torch.tensor([800.0], device=self.device))
            self._box_density = dist.sample((self.num_envs,))
        else:
            self._box_density[:] = 100.0

        self._box_size = torch.tensor(self._build_base_size, device=self.device).reshape(1, 3) * self._box_scale # (num_envs, 3)

        # create asset
        self._box_assets = []
        for i in range(self.num_envs):
            asset_options = gymapi.AssetOptions()
            asset_options.angular_damping = 0.01
            asset_options.linear_damping = 0.01
            asset_options.max_angular_velocity = 100.0
            asset_options.density = self._box_density[i]
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            self._box_assets.append(self.gym.create_box(self.sim, self._box_size[i, 0], self._box_size[i, 1], self._box_size[i, 2], asset_options))
        
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

        self._build_box(env_id, env_ptr)
        
        if self._reset_random_height:
            self._build_platforms(env_id, env_ptr)
        
        if (not self.headless):
            self._build_marker(env_id, env_ptr)

        if (not self.headless) and self.show_sensors:
            self._build_sensor(env_id, env_ptr)

        return
    
    def _build_box(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        default_pose = gymapi.Transform()
        
        pos = torch.tensor(get_axis_params(10.0,
                                           self.up_axis_idx)).to(self.device)
        pos[:2] += torch_rand_float(
            -1., 1., (2, 1), device=self.device).squeeze(
                1)  # ZL: segfault if we do not randomize the position

        default_pose.p = gymapi.Vec3(*pos)
        default_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        box_handle = self.gym.create_actor(env_ptr, self._box_assets[env_id], default_pose, "box", col_group, col_filter, segmentation_id)
        self._box_handles.append(box_handle)

        mass = self.gym.get_actor_rigid_body_properties(env_ptr, box_handle)[0].mass
        self._box_masses.append(mass)

        return
    
    def _build_platforms(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        default_pose = gymapi.Transform()
        
        pos = torch.tensor(get_axis_params(-5.0,
                                           self.up_axis_idx)).to(self.device)
        pos[:2] += torch_rand_float(
            -1., 1., (2, 1), device=self.device).squeeze(
                1)  # ZL: segfault if we do not randomize the position

        default_pose.p = gymapi.Vec3(*pos)
        default_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        platform_handle = self.gym.create_actor(env_ptr, self._platform_asset, default_pose, "platform", col_group, col_filter, segmentation_id)
        self.gym.set_rigid_body_color(env_ptr, platform_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.5, 0.235, 0.6))

        default_pose = gymapi.Transform()
        
        pos = torch.tensor(get_axis_params(-10.0,
                                           self.up_axis_idx)).to(self.device)
        pos[:2] += torch_rand_float(
            -1., 1., (2, 1), device=self.device).squeeze(
                1)  # ZL: segfault if we do not randomize the position

        default_pose.p = gymapi.Vec3(*pos)
        default_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        
        tar_platform_handle = self.gym.create_actor(env_ptr, self._platform_asset, default_pose, "tar_platform", col_group, col_filter, segmentation_id)
        self.gym.set_rigid_body_color(env_ptr, tar_platform_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.0, 0.0, 0.8))

        self._platform_handles.append(platform_handle)
        self._tar_platform_handles.append(tar_platform_handle)

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
    
    def _build_marker(self, env_id, env_ptr):
        col_group = self.num_envs + 10
        col_filter = 0
        segmentation_id = 0
        default_pose = gymapi.Transform()
        
        marker_handle = self.gym.create_actor(env_ptr, self._marker_asset, default_pose, "marker", col_group, col_filter, segmentation_id)
        self.gym.set_rigid_body_color(env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.0, 0.0))
        self.gym.set_actor_scale(env_ptr, marker_handle, 0.3)
        self._marker_handles.append(marker_handle)

        return

    def _build_marker_state_tensors(self):
        num_actors = self._root_states.shape[0] // self.num_envs

        idx = self._marker_handles[0]
        self._marker_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., idx, :]
        self._marker_pos = self._marker_states[..., :3]
        
        self._marker_actor_ids = self._humanoid_actor_ids + idx

        return
    
    def _build_sensor_state_tensors(self):
        num_actors = self._root_states.shape[0] // self.num_envs

        s_idx = self._sensor_handles[0][0]
        e_idx = self._sensor_handles[0][-1] + 1

        self._sensor_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., s_idx:e_idx, :]
        
        self._sensor_pos = self._sensor_states[..., :3]
        
        self._sensor_actor_ids = self._humanoid_actor_ids.unsqueeze(-1) + to_torch(self._sensor_handles, dtype=torch.int32, device=self.device)
        self._sensor_actor_ids = self._sensor_actor_ids.flatten()
        
        return
    
    def _build_platforms_state_tensors(self):
        num_actors = self._root_states.shape[0] // self.num_envs

        idx = self._platform_handles[0]
        self._platform_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., idx, :]
        self._platform_pos = self._platform_states[..., :3]
        self._platform_default_pos = self._platform_pos.clone()
        self._platform_actor_ids = self._humanoid_actor_ids + idx

        idx = self._tar_platform_handles[0]
        self._tar_platform_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., idx, :]
        self._tar_platform_pos = self._tar_platform_states[..., :3]
        self._tar_platform_default_pos = self._tar_platform_pos.clone()
        self._tar_platform_actor_ids = self._humanoid_actor_ids + idx

        return
    
    def _build_box_tensors(self):
        num_actors = self.get_num_actors_per_env()

        idx = self._box_handles[0]
        self._box_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., idx, :]
        
        self._box_actor_ids = to_torch(num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + idx

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

        self._box_surface_bps_ids = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 4, 5],
            [2, 3, 6, 7],
            [0, 3, 4, 7],
            [1, 2, 5, 6],
        ]
        self._box_surface_bps_ids = to_torch(self._box_surface_bps_ids, dtype=torch.long, device=self.device) # (6, 4)

        return

    def _reset_task(self, env_ids):

        # for skill is putDown, the target location of the box is from the reference box motion
        for sk_name in ["putDown"]:
            if self._reset_ref_env_ids.get(sk_name) is not None:
                if (len(self._reset_ref_env_ids[sk_name]) > 0):

                    raise NotImplementedError

        # for skill is loco, pickUp, carryWith, and reset default, we random generate an target location of the box
        random_env_ids = []
        if len(self._reset_default_env_ids) > 0:
            random_env_ids.append(self._reset_default_env_ids)
        for sk_name in ["loco","loco_terrain", "pickUp", "carryWith"]:
            if self._reset_ref_env_ids.get(sk_name) is not None:
                random_env_ids.append(self._reset_ref_env_ids[sk_name])

        if len(random_env_ids) > 0:
            ids = torch.cat(random_env_ids, dim=0)

            root_pos_xy = torch.randn(len(ids), 2, device=self.device)
            root_pos_xy /= torch.linalg.norm(root_pos_xy, dim=-1, keepdim=True)
            root_pos_xy *= torch.rand(len(ids), 1, device=self.device) * 9.0 + 1.0 # randomize # 边界是50m，不会走出去整张地图的
            root_pos_xy += self._box_states[ids, :2] # get absolute pos

            new_target_pos = torch.zeros_like(self._box_states[ids, :3])
            new_target_pos[..., 0:2] = root_pos_xy

            new_target_pos[:, 2] = self._box_size[ids, 2] / 2 # place the box on the ground

            min_dist = 1.0

            # check if the new pos is too close to character or box
            target_overlap = torch.logical_or(
                torch.sum((new_target_pos[..., :2] - self._humanoid_root_states[ids, :2]) ** 2, dim=-1) < min_dist,
                torch.sum((new_target_pos[..., :2] - self._box_states[ids, :2]) ** 2, dim=-1) < min_dist
            )
            while(torch.sum(target_overlap) > 0):

                num_samples = torch.sum(target_overlap)

                root_pos_xy = torch.randn(num_samples, 2, device=self.device)
                root_pos_xy /= torch.linalg.norm(root_pos_xy, dim=-1, keepdim=True)
                root_pos_xy *= torch.rand(num_samples, 1, device=self.device) * 9.0 + 1.0 # randomize # 边界是50m，不会走出去整张地图的
                root_pos_xy += self._box_states[ids[target_overlap], :2] # get absolute pos

                new_target_pos[target_overlap, 0:2] = root_pos_xy

                new_target_pos[:, 2] = self._box_size[ids, 2] / 2 # place the box on the ground
                target_overlap = torch.logical_or(
                    torch.sum((new_target_pos[..., :2] - self._humanoid_root_states[ids, :2]) ** 2, dim=-1) < min_dist,
                    torch.sum((new_target_pos[..., :2] - self._box_states[ids, :2]) ** 2, dim=-1) < min_dist
                )

            if self._reset_random_height:
                num_envs = ids.shape[0]
                probs = to_torch(np.array([self._reset_random_height_prob] * num_envs), device=self.device)
                mask = torch.bernoulli(probs) == 1.0
                
                if mask.sum() > 0:
                    new_target_pos[mask, 2] += torch.rand(mask.sum(), device=self.device) * 1.0 + self._reset_minBottomSurfaceHeight
                    new_target_pos[mask, 2] = self._regulate_height(new_target_pos[mask, 2], self._box_size[ids[mask]])

            new_target_rot = torch.zeros_like(self._box_states[ids, 3:7])
            new_target_rot[..., -1] = 1.0

            root_states = torch.cat([new_target_pos, new_target_rot], dim=-1)
            center_height = self.get_center_heights(root_states, env_ids=ids).mean(dim=-1) # 当前人体root 2d pos的高度
            new_target_pos[:, 2] += center_height

            self._tar_pos[ids] = new_target_pos

            # we need to reset this here
            if self._reset_random_height:
                self._tar_platform_pos[ids, 0:2] = new_target_pos[:, 0:2] # xy
                self._tar_platform_pos[ids, -1] = new_target_pos[:, -1] - self._box_size[ids, 2] / 2 - self._platform_height / 2

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

    def _compute_task_obs(self, env_ids=None):
        if (env_ids is None):
            root_states = self._humanoid_root_states
            box_states = self._box_states
            box_bps = self._box_bps
            tar_pos = self._tar_pos
            box_mass = self._box_masses
        else:
            root_states = self._humanoid_root_states[env_ids]
            box_states = self._box_states[env_ids]
            box_bps = self._box_bps[env_ids]
            tar_pos = self._tar_pos[env_ids]
            box_mass = self._box_masses[env_ids]

        obs = compute_location_observations(root_states, box_states, box_bps, tar_pos,
                                            self._enable_bbox_obs)
        
        if self._build_random_density:
            obs = torch.cat([obs, box_mass.unsqueeze(-1)], dim=-1)

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
        rigid_body_pos = self._rigid_body_pos
        box_pos = self._box_states[..., 0:3]
        box_rot = self._box_states[..., 3:7]
        hands_ids = self._key_body_ids[[0, 1]]

        walk_r = compute_walk_reward(root_pos, self._prev_root_pos, box_pos, self.dt, 1.5,
                                     self._only_vel_reward, self.cfg["env"]["debug"]["vel"])
        carry_r = compute_carry_reward(box_pos, self._prev_box_pos, self._tar_pos, self.dt, 1.5, self._box_size, 
                                       self._only_vel_reward,
                                       self._box_vel_penalty, self._box_vel_pen_coeff, self._box_vel_pen_thre,
                                       self.cfg["env"]["debug"]["vel"])
        handheld_r = compute_handheld_reward(rigid_body_pos, box_pos, hands_ids, self._tar_pos, self._only_height_handheld_reward)
        putdown_r = compute_putdown_reward(box_pos, self._tar_pos)
        carry_box_reward = 2.0 * walk_r + 2.0 * carry_r + handheld_r + putdown_r

        power = torch.abs(torch.multiply(self.dof_force_tensor[:, self._power_dof_ids], self._dof_vel[:, self._power_dof_ids])).sum(dim = -1)
        power_reward = -self._power_coefficient * power

        if self._power_reward:
            self.rew_buf[:] = carry_box_reward + power_reward
        else:
            self.rew_buf[:] = carry_box_reward

        return
    
    def render(self, sync_frame_time=False):
        super().render(sync_frame_time)

        if self.viewer:
            self._draw_task()
        return

    def _draw_task(self):
        self._update_marker()

        cols = np.array([
            [0.0, 1.0, 0.0], # green
            [1.0, 0.0, 0.0], # red
        ], dtype=np.float32)

        self.gym.clear_lines(self.viewer)

        starts = self._box_states[..., 0:3] # line from box to marker
        ends = self._tar_pos[..., 0:3]

        starts_l2 = self._humanoid_root_states[..., 0:3] # line from humanoid to box
        ends_l2 = self._box_states[..., 0:3]

        verts = torch.cat([starts, ends, starts_l2, ends_l2], dim=-1).cpu().numpy()

        for i, env_ptr in enumerate(self.envs):
            curr_verts = verts[i]
            curr_verts = curr_verts.reshape([2, 6])
            self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, cols)

        # draw lines of the bbox
        cols = np.zeros((24, 3), dtype=np.float32) # 24 lines
        cols[:12] = [1.0, 0.0, 0.0] # red
        cols[12:] = [0.0, 1.0, 0.0] # greed

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
        
        radius = 0.5

        num_verts = 30
        axis = torch.tensor([[0.0, 0.0, 1.0]], device=self.device).reshape(1, 3).expand([num_verts, -1])
        ang = torch.linspace(0, 2 * np.pi, num_verts, device=self.device)
        quat = quat_from_angle_axis(ang, axis) # (num_verts, 4)

        axis = torch.tensor([[1.0, 0.0, 0.0]], device=self.device).reshape(1, 3).expand([num_verts, -1])
        axis = quat_rotate(quat, axis)
        pos  = axis * radius

        cols = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)

        for i, env_ptr in enumerate(self.envs):
            verts = pos.clone()
            verts += self._tar_pos[i, 0:3].unsqueeze(0)
            lines = torch.cat([verts[:-1], verts[1:]], dim=-1).cpu().numpy()
            curr_cols = np.broadcast_to(cols, [lines.shape[0], cols.shape[-1]])
            self.gym.add_lines(self.viewer, env_ptr, lines.shape[0], lines, curr_cols)
        
        num_verts = 30
        axis = torch.tensor([[0.0, 0.0, 1.0]], device=self.device).reshape(1, 3).expand([num_verts, -1])
        ang = torch.linspace(0, 2 * np.pi, num_verts, device=self.device)
        quat = quat_from_angle_axis(ang, axis) # (num_verts, 4)

        axis = torch.tensor([[1.0, 0.0, 0.0]], device=self.device).reshape(1, 3).expand([num_verts, -1])
        axis = quat_rotate(quat, axis)
        pos  = axis * radius

        cols = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)

        for i, env_ptr in enumerate(self.envs):
            verts = pos.clone()
            verts += self._box_states[i, 0:3].unsqueeze(0)
            lines = torch.cat([verts[:-1], verts[1:]], dim=-1).cpu().numpy()
            curr_cols = np.broadcast_to(cols, [lines.shape[0], cols.shape[-1]])
            self.gym.add_lines(self.viewer, env_ptr, lines.shape[0], lines, curr_cols)

        return
    
    def save_imgs(self):
        if self.cfg["args"].record:
            # render frame if we're saving video
            if self.save_video and self.viewer is not None:
                num = str(self.save_img_count)
                num = '0' * (6 - len(num)) + num
                new_save_dir = os.path.join(self.save_video_dir, "imgs")
                os.makedirs(new_save_dir, exist_ok=True)
                self.gym.write_viewer_image_to_file(self.viewer, f"{new_save_dir}/frame_{num}.png")

                base_quat = self._humanoid_root_states[:, 3:7]
                heading_rot = torch_utils.calc_heading_quat(base_quat)

                points = quat_apply(
                    heading_rot.repeat(1, self.num_height_points).reshape(-1, 4),
                    self.height_points) + (self._humanoid_root_states[:, :3]).unsqueeze(1)
                
                env_ids = torch.arange(0, self.num_envs, device=self.device, dtype=torch.long)
                measured_heights = self.get_heights(root_states=self._humanoid_root_states, env_ids=env_ids)

                sensor_pos = points
                sensor_pos[..., -1] = measured_heights

                parameters = {
                    "hu_rigid_body_pos_rot": self._fetch_humanoid_rigid_body_pos_rot_states().cpu(),
                    "box_states": self._box_states.clone().cpu(),
                    "box_sizes": self._box_size.clone().cpu(),
                    "box_tar_pos": self._tar_pos.clone().cpu(),
                    "platform_states": self._platform_states.clone().cpu(),
                    "tar_platform_states": self._tar_platform_states.clone().cpu(),
                    "platform_size": torch.tensor([0.4, 0.4, 0.02]),
                    "sensor_pos": sensor_pos.clone().cpu(),
                    "dt": self.dt,
                    "progress": self.progress_buf.clone().cpu()
                }

                new_save_dir = os.path.join(self.save_video_dir, "parameters")
                os.makedirs(new_save_dir, exist_ok=True)
                with open(f"{new_save_dir}/frame_{num}.pkl", 'wb') as f:
                    pickle.dump(parameters, f)

                # save terrain
                mesh_export_dir = os.path.join(self.save_video_dir, "terrain_mesh")
                os.makedirs(mesh_export_dir, exist_ok=True)
                output_path = os.path.join(mesh_export_dir, "mesh.obj")
                if not os.path.exists(output_path):
                    mesh = trimesh.Trimesh(vertices=self.terrain.vertices, faces=self.terrain.triangles)
                    mesh.export(os.path.join(mesh_export_dir, "mesh.obj"))

                self.save_img_count += 1

        else:
            super().save_imgs()

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
        box_root_pos = self._box_states[..., 0:3]

        pos_diff = self._tar_pos - box_root_pos
        pos_err = torch.norm(pos_diff, p=2, dim=-1)
        dist_mask = pos_err <= self._success_threshold
        self._success_buf[dist_mask] += 1

        self._precision_buf[dist_mask] = torch.where(pos_err[dist_mask] < self._precision_buf[dist_mask], pos_err[dist_mask], self._precision_buf[dist_mask])

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
        self._reset_human_root_pos_offset = {}
        if (len(env_ids) > 0):
            self._reset_actors(env_ids)
            self._reset_boxes(env_ids)
            self._reset_task(env_ids)
            self._reset_env_tensors(env_ids)
            self._refresh_sim_tensors()
            self._compute_observations(env_ids)
        
            self._init_amp_obs(env_ids)

        return
    
    def _regulate_height(self, h, box_size):
        top_surface_z = h + box_size[:, 2] / 2
        top_surface_z = torch.clamp_max(top_surface_z, self._reset_maxTopSurfaceHeight)
        return top_surface_z - box_size[:, 2] / 2
    
    def _reset_boxes(self, env_ids):

        # for skill is pickUp, carryWith, putDown, the initial location of the box is from the reference box motion
        for sk_name in ["pickUp", "carryWith", "putDown"]:
            if self._reset_ref_env_ids.get(sk_name) is not None:
                if (len(self._reset_ref_env_ids[sk_name]) > 0):

                    assert sk_name == "carryWith"

                    curr_env_ids = self._reset_ref_env_ids[sk_name]

                    root_pos, root_rot = self._motion_lib[sk_name].get_obj_motion_state(
                        motion_ids=self._reset_ref_motion_ids[sk_name], 
                        motion_times=self._reset_ref_motion_times[sk_name]
                    )

                    on_ground_mask = (self._box_size[curr_env_ids, 2] / 2 > root_pos[:, 2])
                    root_pos[on_ground_mask, 2] = self._box_size[curr_env_ids[on_ground_mask], 2] / 2

                    root_pos += self._reset_human_root_pos_offset[sk_name] # 对XYZ均做平移

                    self._box_states[curr_env_ids, 0:3] = root_pos
                    self._box_states[curr_env_ids, 3:7] = root_rot
                    self._box_states[curr_env_ids, 7:10] = 0.0
                    self._box_states[curr_env_ids, 10:13] = 0.0

                    # reset platform, we needn't platforms right now.
                    if self._reset_random_height:
                        self._platform_pos[curr_env_ids] = self._platform_default_pos[curr_env_ids]

        # for skill is loco and reset default, we random generate an inital location of the box
        random_env_ids = []
        if len(self._reset_default_env_ids) > 0:
            random_env_ids.append(self._reset_default_env_ids)
        for sk_name in ["loco", "loco_terrain"]:
            if self._reset_ref_env_ids.get(sk_name) is not None:
                random_env_ids.append(self._reset_ref_env_ids[sk_name])

        if len(random_env_ids) > 0:
            ids = torch.cat(random_env_ids, dim=0)

            root_pos_xy = torch.randn(len(ids), 2, device=self.device)
            root_pos_xy /= torch.linalg.norm(root_pos_xy, dim=-1, keepdim=True)
            root_pos_xy *= torch.rand(len(ids), 1, device=self.device) * 9.0 + 1.0 # randomize # 边界是50m，不会走出去整张地图的
            root_pos_xy += self._humanoid_root_states[ids, :2] # get absolute pos, humanoid_root_state will be updated after set_env_state

            root_pos_z = self._box_size[ids, 2] / 2 # place the box on the ground
            if self._reset_random_height:

                num_envs = ids.shape[0]
                probs = to_torch(np.array([self._reset_random_height_prob] * num_envs), device=self.device)
                mask = torch.bernoulli(probs) == 1.0
                
                if mask.sum() > 0:
                    root_pos_z[mask] += torch.rand(mask.sum(), device=self.device) * 1.0 + self._reset_minBottomSurfaceHeight
                    root_pos_z[mask] = self._regulate_height(root_pos_z[mask], self._box_size[ids[mask]])

            axis = torch.tensor([[0.0, 0.0, 1.0]], device=self.device).reshape(1, 3).expand([ids.shape[0], -1])
            if self._reset_random_rot:
                coeff = 1.0
            else:
                coeff = 0.0
            ang = torch.rand((len(ids),), device=self.device) * 2 * np.pi * coeff
            root_rot = quat_from_angle_axis(ang, axis)
            root_pos = torch.cat([root_pos_xy, root_pos_z.unsqueeze(-1)], dim=-1)

            root_states = torch.cat([root_pos, root_rot], dim=-1)
            center_height = self.get_center_heights(root_states, env_ids=ids).mean(dim=-1) # 当前人体root 2d pos的高度
            root_pos[:, 2] += center_height

            self._box_states[ids, 0:3] = root_pos
            self._box_states[ids, 3:7] = root_rot
            self._box_states[ids, 7:10] = 0.0
            self._box_states[ids, 10:13] = 0.0

            # we need to reset this here
            if self._reset_random_height:
                self._platform_pos[ids, 0:2] = root_pos[:, 0:2] # xy
                self._platform_pos[ids, -1] = root_pos[:, -1] - self._box_size[ids, 2] / 2 - self._platform_height / 2

                self._box_states[ids, 2] += 0.05 # add 0.05 to enable right collision detection

        return
    
    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)

        if self._is_eval:
            self._success_buf[env_ids] = 0
            self._precision_buf[env_ids] = float('Inf')

        env_ids_int32 = self._box_actor_ids[env_ids].view(-1)
        if self._reset_random_height:
            # env has two platforms
            env_ids_int32 = torch.cat([env_ids_int32, self._platform_actor_ids, self._tar_platform_actor_ids], dim=0)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        return

    def _reset_actors(self, env_ids):
        if (self._state_init == HumanoidAdaptCarryGround2Terrain.StateInit.Default):
            self._reset_default(env_ids)
        elif (self._state_init == HumanoidAdaptCarryGround2Terrain.StateInit.Start
              or self._state_init == HumanoidAdaptCarryGround2Terrain.StateInit.Random):
            self._reset_ref_state_init(env_ids)
        elif (self._state_init == HumanoidAdaptCarryGround2Terrain.StateInit.Hybrid):
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

                if (self._state_init == HumanoidAdaptCarryGround2Terrain.StateInit.Random
                    or self._state_init == HumanoidAdaptCarryGround2Terrain.StateInit.Hybrid):
                    motion_times = curr_motion_lib.sample_time_rsi(motion_ids) # avoid times with serious self-penetration
                elif (self._state_init == HumanoidAdaptCarryGround2Terrain.StateInit.Start):
                    motion_times = torch.zeros(num_envs, device=self.device)
                else:
                    assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

                root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
                    = curr_motion_lib.get_motion_state(motion_ids, motion_times)
                
                if self._is_eval:
                    root_pos = self._initial_humanoid_root_states[curr_env_ids, 0:3]
                    root_rot = self._initial_humanoid_root_states[curr_env_ids, 3:7]
                    root_vel = self._initial_humanoid_root_states[curr_env_ids, 7:10]
                    root_ang_vel = self._initial_humanoid_root_states[curr_env_ids, 10:13]
                    dof_pos = self._initial_dof_pos[curr_env_ids]
                    dof_vel = self._initial_dof_vel[curr_env_ids]
                
                old_root_xyz = root_pos.clone()

                new_root_xy = self.terrain.sample_valid_locations(len(curr_env_ids), curr_env_ids)

                root_pos[:, 0:2] = new_root_xy

                root_states = torch.cat([root_pos, root_rot], dim=1)

                center_height = self.get_center_heights(root_states, env_ids=curr_env_ids).mean(dim=-1) # 当前人体root 2d pos的高度

                root_pos[:, 2] = center_height + self._char_h + 0.05

                self._reset_human_root_pos_offset[sk_name] = root_pos - old_root_xyz

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

        env_ids = torch.arange(0, self.num_envs, dtype=torch.long, device=self.device)
        center_heights = self.get_center_heights(self._humanoid_root_states[..., 0:7], env_ids=env_ids).mean(dim=-1) # 当前人体root 2d pos的高度

        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                   self._contact_forces, self._contact_body_ids,
                                                   self._rigid_body_pos, self.max_episode_length,
                                                   self._enable_early_termination, self._termination_heights, center_heights,
                                                   self._box_states[..., 0:3])
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
def compute_location_observations(root_states, box_states, box_bps, tar_pos, enableBboxObs):
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
def compute_handheld_reward(humanoid_rigid_body_pos, box_pos, hands_ids, tar_pos, only_height):
    # type: (Tensor, Tensor, Tensor, Tensor, bool) -> Tensor
    if only_height:
        hands2box_pos_err = torch.sum((humanoid_rigid_body_pos[:, hands_ids, 2] - box_pos[:, 2].unsqueeze(-1)) ** 2, dim=-1) # height
    else:
        hands2box_pos_err = torch.sum((humanoid_rigid_body_pos[:, hands_ids].mean(dim=1) - box_pos) ** 2, dim=-1) # xyz
    hands2box = torch.exp(-5.0 * hands2box_pos_err)

    # box2tar = torch.sum((box_pos[..., 0:2] - tar_pos[..., 0:2]) ** 2, dim=-1) # 2d
    # hands2box[box2tar < 0.7 ** 2] = 1.0 # assume this reward is max when the box is close enough to its target location

    root_pos = humanoid_rigid_body_pos[:, 0, :]
    box2human = torch.sum((box_pos[..., 0:2] - root_pos[..., 0:2]) ** 2, dim=-1) # 2d
    hands2box[box2human > 0.7 ** 2] = 0 # disable this reward when the box is not close to the humanoid

    return 0.2 * hands2box

@torch.jit.script
def compute_walk_reward(root_pos, prev_root_pos, box_pos, dt, tar_vel, only_vel_reward, debug_vel):
    # type: (Tensor, Tensor, Tensor, float, float, bool, bool) -> Tensor

    # this reward encourages the character to walk towards the box and stay close to it

    pos_diff = box_pos[..., 0:2] - root_pos[..., 0:2]
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    pos_reward = torch.exp(-0.5 * pos_err)

    min_speed = tar_vel

    tar_dir = box_pos[..., 0:2] - root_pos[..., 0:2]
    tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)

    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)
    if debug_vel:
        print("humanoid_vel_tar_dir: ", tar_dir_speed[0].item())
    tar_vel_err = min_speed - tar_dir_speed
    # tar_vel_err = torch.clamp_min(tar_vel_err, 0.0) # constrain vel around the peak value
    vel_reward = torch.exp(-5.0 * (tar_vel_err * tar_vel_err))
    speed_mask = tar_dir_speed <= 0
    vel_reward[speed_mask] = 0

    dist_mask = pos_err < 0.5 ** 2
    pos_reward[dist_mask] = 1.0
    vel_reward[dist_mask] = 1.0

    if only_vel_reward:
        reward = 0.2 * vel_reward
    else:
        reward = 0.1 * pos_reward + 0.1 * vel_reward
    return reward

@torch.jit.script
def compute_carry_reward(box_pos, prev_box_pos, tar_box_pos, dt, tar_vel, box_size, only_vel_reward, box_vel_penalty, box_vel_pen_coeff, box_vel_penalty_thre, debug_vel):
    # type: (Tensor, Tensor, Tensor, float, float, Tensor, bool, bool, float, float, bool) -> Tensor
    
    # this reward encourages the character to carry the box to a target position

    pos_diff = tar_box_pos - box_pos # xyz
    pos_err_xy = torch.sum(pos_diff[..., 0:2] ** 2, dim=-1)
    pos_err_xyz = torch.sum(pos_diff * pos_diff, dim=-1)
    pos_reward_far = torch.exp(-0.5 * pos_err_xy)
    pos_reward_near = torch.exp(-10.0 * pos_err_xyz)

    min_speed = tar_vel

    tar_dir = tar_box_pos[..., 0:2] - box_pos[..., 0:2]
    tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)

    delta_root_pos = box_pos - prev_box_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)
    if debug_vel:
        print("box_vel_tar_dir: ", tar_dir_speed[0].item())
        print("box_vel_root: ", torch.norm(root_vel[0], p=2).item())
    tar_vel_err = min_speed - tar_dir_speed
    # tar_vel_err = torch.clamp_min(tar_vel_err, 0.0) # constrain vel around the peak value
    vel_reward = torch.exp(-5.0 * (tar_vel_err * tar_vel_err))
    speed_mask = tar_dir_speed <= 0
    vel_reward[speed_mask] = 0

    # 这一项在地形上面不太好，因为会有向下凹的地形
    # height_mask = box_pos[..., 2] <= (box_size[..., 2] / 2 + 0.2) # avoid learning to kick the box
    # pos_reward_far[height_mask] = 0.0
    # vel_reward[height_mask] = 0.0

    dist_mask = pos_err_xy < 0.5 ** 2
    pos_reward_far[dist_mask] = 1.0
    vel_reward[dist_mask] = 1.0

    if only_vel_reward:
        reward = 0.2 * vel_reward + 0.2 * pos_reward_near
    else:
        reward = 0.1 * pos_reward_far + 0.1 * vel_reward + 0.2 * pos_reward_near

    if box_vel_penalty:
        min_speed_penalty = box_vel_penalty_thre
        root_vel_norm = torch.norm(root_vel, p=2, dim=-1)
        root_vel_norm = torch.clamp_min(root_vel_norm, min_speed_penalty)
        root_vel_err = min_speed_penalty - root_vel_norm
        root_vel_penalty = -1 * box_vel_pen_coeff * (1 - torch.exp(-2.0 * (root_vel_err * root_vel_err)))
        reward += root_vel_penalty

    return reward

@torch.jit.script
def compute_putdown_reward(box_pos, tar_pos):
    # type: (Tensor, Tensor) -> Tensor

    reward = (torch.abs((box_pos[:, -1] - tar_pos[:, -1])) <= 0.001) * 1.0 # binary reward, 0.0 or 1.0
    
    pos_err_xy = torch.sum((tar_pos[..., :2] - box_pos[..., :2]) ** 2, dim=-1)
    reward[(pos_err_xy > 0.1 ** 2)] = 0.0
    
    return 0.2 * reward

@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_heights, center_heights, box_pos):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):

        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)
        
        human2box = torch.norm(rigid_body_pos[:, 0, 0:2] - box_pos[..., 0:2], p=2, dim=-1) >= 0.7
        fall_contact = torch.logical_and(fall_contact, human2box)

        body_height = rigid_body_pos[..., 2]
        height_thresholds = torch.zeros_like(body_height)
        height_thresholds[:] = termination_heights
        height_thresholds += center_heights[:, None]
        fall_height = body_height < height_thresholds
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_fallen = torch.logical_or(fall_contact, fall_height)

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= (progress_buf > 1)
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)
    
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
        self.border_size = 50 # 乘scale后单位才是m
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
