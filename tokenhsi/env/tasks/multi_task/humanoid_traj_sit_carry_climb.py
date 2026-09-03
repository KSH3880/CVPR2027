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
import json

from isaacgym import gymapi
from isaacgym import gymtorch

from env.tasks.humanoid import Humanoid, dof_to_obs
from utils import gym_util
from utils.motion_lib import MotionLib
from isaacgym.torch_utils import *

from utils import torch_utils
from utils import traj_generator

class ObjectLib():
    def __init__(self, mode, dataset_root, dataset_categories, num_envs, device):
        self.device = device
        self.mode = mode

        # load basic info
        dataset_categories_count = []
        obj_urdfs = []
        obj_bbox_centers = []
        obj_bbox_lengths = []
        obj_facings = []
        obj_tar_sit_pos = []
        obj_on_ground_trans = []
        for cat in dataset_categories:
            obj_list = os.listdir(os.path.join(dataset_root, mode, cat))
            dataset_categories_count.append(len(obj_list))
            for obj_name in obj_list:
                curr_dir = os.path.join(dataset_root, mode, cat, obj_name)
                obj_urdfs.append(os.path.join(curr_dir, "asset.urdf"))

                with open(os.path.join(os.getcwd(), curr_dir, "config.json"), "r") as f:
                    object_cfg = json.load(f)
                    assert not np.sum(np.abs(object_cfg["center"])) > 0.0 
                    obj_bbox_centers.append(object_cfg["center"])
                    obj_bbox_lengths.append(object_cfg["bbox"])
                    obj_facings.append(object_cfg["facing"])
                    obj_tar_sit_pos.append(object_cfg["tarSitPos"])
                    obj_on_ground_trans.append(-1 * (obj_bbox_centers[-1][2] - obj_bbox_lengths[-1][2] / 2))

        # randomly sample a fixed object for each simulation env, due to the limitation of IsaacGym
        base_prob = 1.0 / len(dataset_categories)
        averaged_probs = []
        for i in range(len(dataset_categories)):
            averaged_probs += [base_prob * (1.0 / dataset_categories_count[i])] * dataset_categories_count[i]
        weights = torch.tensor(averaged_probs, device=self.device)
        self._every_env_object_ids = torch.multinomial(weights, num_samples=num_envs, replacement=True).squeeze(-1)
        if num_envs == 1:
            self._every_env_object_ids = self._every_env_object_ids.unsqueeze(0)
        self._every_env_object_bbox_centers = to_torch(obj_bbox_centers, dtype=torch.float, device=self.device)[self._every_env_object_ids]
        self._every_env_object_bbox_lengths = to_torch(obj_bbox_lengths, dtype=torch.float, device=self.device)[self._every_env_object_ids]
        self._every_env_object_facings = to_torch(obj_facings, dtype=torch.float, device=self.device)[self._every_env_object_ids]
        self._every_env_object_tar_sit_pos = to_torch(obj_tar_sit_pos, dtype=torch.float, device=self.device)[self._every_env_object_ids]
        self._every_env_object_on_ground_trans = to_torch(obj_on_ground_trans, dtype=torch.float, device=self.device)[self._every_env_object_ids]

        self._every_env_object_circle_length = self._every_env_object_bbox_lengths[..., 0:2].norm(dim=-1)
        self._every_env_object_valid_radius = self._every_env_object_circle_length / 2 + 0.3

        self._obj_urdfs = obj_urdfs

        self._build_object_bps()

        return
    
    def _build_object_bps(self):

        bps_0 = torch.cat([     self._every_env_object_bbox_lengths[:, 0].unsqueeze(-1) / 2,      self._every_env_object_bbox_lengths[:, 1].unsqueeze(-1) / 2, -1 * self._every_env_object_bbox_lengths[:, 2].unsqueeze(-1) / 2], dim=-1)
        bps_1 = torch.cat([-1 * self._every_env_object_bbox_lengths[:, 0].unsqueeze(-1) / 2,      self._every_env_object_bbox_lengths[:, 1].unsqueeze(-1) / 2, -1 * self._every_env_object_bbox_lengths[:, 2].unsqueeze(-1) / 2], dim=-1)
        bps_2 = torch.cat([-1 * self._every_env_object_bbox_lengths[:, 0].unsqueeze(-1) / 2, -1 * self._every_env_object_bbox_lengths[:, 1].unsqueeze(-1) / 2, -1 * self._every_env_object_bbox_lengths[:, 2].unsqueeze(-1) / 2], dim=-1)
        bps_3 = torch.cat([     self._every_env_object_bbox_lengths[:, 0].unsqueeze(-1) / 2, -1 * self._every_env_object_bbox_lengths[:, 1].unsqueeze(-1) / 2, -1 * self._every_env_object_bbox_lengths[:, 2].unsqueeze(-1) / 2], dim=-1)
        bps_4 = torch.cat([     self._every_env_object_bbox_lengths[:, 0].unsqueeze(-1) / 2,      self._every_env_object_bbox_lengths[:, 1].unsqueeze(-1) / 2,      self._every_env_object_bbox_lengths[:, 2].unsqueeze(-1) / 2], dim=-1)
        bps_5 = torch.cat([-1 * self._every_env_object_bbox_lengths[:, 0].unsqueeze(-1) / 2,      self._every_env_object_bbox_lengths[:, 1].unsqueeze(-1) / 2,      self._every_env_object_bbox_lengths[:, 2].unsqueeze(-1) / 2], dim=-1)
        bps_6 = torch.cat([-1 * self._every_env_object_bbox_lengths[:, 0].unsqueeze(-1) / 2, -1 * self._every_env_object_bbox_lengths[:, 1].unsqueeze(-1) / 2,      self._every_env_object_bbox_lengths[:, 2].unsqueeze(-1) / 2], dim=-1)
        bps_7 = torch.cat([     self._every_env_object_bbox_lengths[:, 0].unsqueeze(-1) / 2, -1 * self._every_env_object_bbox_lengths[:, 1].unsqueeze(-1) / 2,      self._every_env_object_bbox_lengths[:, 2].unsqueeze(-1) / 2], dim=-1)
        
        self._every_env_object_bps = torch.cat([
            bps_0.unsqueeze(1),
            bps_1.unsqueeze(1),
            bps_2.unsqueeze(1),
            bps_3.unsqueeze(1),
            bps_4.unsqueeze(1),
            bps_5.unsqueeze(1),
            bps_6.unsqueeze(1),
            bps_7.unsqueeze(1)]
        , dim=1)
            
        self._every_env_object_bps += self._every_env_object_bbox_centers.unsqueeze(1) # (num_envs, 8, 3)

        return

class BoxLib():
    def __init__(self, mode, box_cfg, num_envs, device):
        self.device = device
        self.mode = mode

        self._build_base_size = box_cfg["build"]["baseSize"]
        self._build_random_size = box_cfg["build"]["randomSize"]
        self._build_random_mode_equal_proportion = box_cfg["build"]["randomModeEqualProportion"]
        self._build_x_scale_range = box_cfg["build"]["scaleRangeX"]
        self._build_y_scale_range = box_cfg["build"]["scaleRangeY"]
        self._build_z_scale_range = box_cfg["build"]["scaleRangeZ"]
        self._build_scale_sample_interval = box_cfg["build"]["scaleSampleInterval"]
        self._build_test_sizes = box_cfg["build"]["testSizes"]

        # rescale
        self._box_scale = torch.ones((num_envs, 3), dtype=torch.float32, device=self.device)
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
                
                if mode == "test":
                    test_sizes = torch.tensor(self._build_test_sizes, device=self.device)
                    scale_pool = torch.zeros((test_sizes.shape[0], 3), device=self.device)
                    num_scales = test_sizes.shape[0]

                    for axis in range(3):
                        scale_pool[:, axis] = test_sizes[:, axis] / self._build_base_size[axis]

            if num_envs >= num_scales:
                self._box_scale[:num_scales] = scale_pool[:num_scales] # copy

                sampled_scale_id = torch.multinomial(torch.ones(num_scales) * (1.0 / num_scales), num_samples=(num_envs - num_scales), replacement=True)
                self._box_scale[num_scales:] = scale_pool[sampled_scale_id]

                shuffled_id = torch.randperm(num_envs)
                self._box_scale = self._box_scale[shuffled_id]

            else:
                sampled_scale_id = torch.multinomial(torch.ones(num_scales) * (1.0 / num_scales), num_samples=num_envs, replacement=True)
                self._box_scale = scale_pool[sampled_scale_id]

        self._box_size = torch.tensor(self._build_base_size, device=self.device).reshape(1, 3) * self._box_scale # (num_envs, 3)
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

class HumanoidTrajSitCarryClimb(Humanoid):
    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    class TaskUID(Enum):
        traj = 0
        sit = 1
        carry = 2
        climb = 3

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):

        # configs for task
        self._enable_task_obs = cfg["env"]["enableTaskObs"]
        self._enable_task_mask_obs = cfg["env"]["enableTaskMaskObs"]
        self._enable_apply_mask_on_task_obs = cfg["env"]["enableApplyMaskOnTaskObs"]

        self._mode = cfg["env"]["mode"] # determine which set of objects to use (train or test)
        assert self._mode in ["train", "test"]

        if cfg["args"].eval:
            self._mode = "test"

        self.max_episode_length_short = cfg["env"]["episodeLengthShort"]

        self._num_tasks = 0
        self._each_subtask_obs_size = []
        self._multiple_task_names = []

        self.register_task_traj_pre_init(cfg)
        self.register_task_sit_pre_init(cfg)
        self.register_task_carry_pre_init(cfg)
        self.register_task_climb_pre_init(cfg)

        # task-specific conditional disc
        self._enable_task_specific_disc = cfg["env"]["enableTaskSpecificDisc"]

        # configs for amp
        state_init = cfg["env"]["stateInit"]
        self._state_init = HumanoidTrajSitCarryClimb.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        assert(self._num_amp_obs_steps >= 2)

        self._power_reward = cfg["env"]["power_reward"]
        self._power_coefficient = cfg["env"]["power_coefficient"]

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
        self._skill_disc_prob = torch.tensor(cfg["env"]["skillDiscProb"], device=self.device, dtype=torch.float) # probs for amp obs demo fetch
        assert self._skill_disc_prob.sum() == 1.0
        self._common_skill = cfg["env"].get("commonSkill", "")

        motion_file = cfg['env']['motion_file']
        self._load_motion(motion_file)

        self._amp_obs_buf = torch.zeros((self.num_envs, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float)
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]
        
        self._amp_obs_demo_buf = None

        # tensors for task
        self._prev_root_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self._prev_root_rot = torch.zeros([self.num_envs, 4], device=self.device, dtype=torch.float)

        self._task_init_prob = torch.tensor(cfg["env"]["taskInitProb"], device=self.device, dtype=torch.float) # probs for task init
        self._task_indicator = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._task_mask = torch.zeros([self.num_envs, self._num_tasks], device=self.device, dtype=torch.bool)

        # Interaction Early Termination (IET)
        self._enable_IET = cfg["env"]["enableIET"]
        self._success_threshold = cfg["env"]["successThreshold"]
        if self._enable_IET:
            self._max_IET_steps = cfg["env"]["maxIETSteps"]
            self._IET_step_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            self._IET_triggered_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        ###### evaluation!!!
        self._is_eval = cfg["args"].eval
        self._eval_task = cfg["args"].eval_task
        if self._is_eval:

            # specify one task to evaluate
            task_names = cfg["env"]["task"]
            assert self._eval_task in task_names, "The specified task for evaluating doesn't exist in the supported tasks!!!"
            for i, n in enumerate(task_names):
                if n != self._eval_task:
                    self._task_init_prob[i] = 0.0
                else:
                    self._task_init_prob[i] = 1.0

            self._enable_IET = False # as default, we disable this feature

            self._success_buf = torch.zeros((self.num_envs), device=self.device, dtype=torch.long)
            self._precision_buf = torch.zeros((self.num_envs), device=self.device, dtype=torch.float)

        self.register_task_traj_post_init(cfg)
        self.register_task_sit_post_init(cfg)
        self.register_task_carry_post_init(cfg)
        self.register_task_climb_post_init(cfg)
        self.post_process_disc_dataset_collection(cfg)

        # tensors for enableTrackInitState
        self._every_env_init_dof_pos = torch.zeros((self.num_envs, self.num_dof), device=self.device, dtype=torch.float)

        # tensors for fixing obs bug
        self._kinematic_humanoid_rigid_body_states = torch.zeros((self.num_envs, self.num_bodies, 13), device=self.device, dtype=torch.float)

        return
    
    def register_task_traj_pre_init(self, cfg):
        k = "traj"
        assert HumanoidTrajSitCarryClimb.TaskUID[k].value >= 0

        self._num_traj_samples = cfg["env"][k]["numTrajSamples"]
        self._traj_sample_timestep = cfg["env"][k]["trajSampleTimestep"]
        self._speed_min = cfg["env"][k]["speedMin"]
        self._speed_max = cfg["env"][k]["speedMax"]
        self._accel_max = cfg["env"][k]["accelMax"]
        self._sharp_turn_prob = cfg["env"][k]["sharpTurnProb"]
        self._sharp_turn_angle = cfg["env"][k]["sharpTurnAngle"]
        self._fail_dist = cfg["env"][k]["failDist"]

        self._num_tasks += 1
        self._each_subtask_obs_size.append(2 * self._num_traj_samples)
        self._multiple_task_names.append(k)

        return
    
    def register_task_traj_post_init(self, cfg):
        k = "traj"

        self._traj_skill = cfg["env"][k]["skill"]
        self._traj_skill_init_prob = torch.tensor(cfg["env"][k]["skillInitProb"], device=self.device, dtype=torch.float)

        if self._is_eval:
            self._traj_skill = cfg["env"][k]["eval"]["skill"]
            self._traj_skill_init_prob = torch.tensor(cfg["env"][k]["eval"]["skillInitProb"], device=self.device, dtype=torch.float)

        self._build_traj_generator()

        if (not self.headless):
            num_actors = self._root_states.shape[0] // self.num_envs

            s_idx = self._traj_marker_handles[0][0]
            e_idx = self._traj_marker_handles[0][-1] + 1

            self._traj_marker_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., s_idx:e_idx, :]
            self._traj_marker_pos = self._traj_marker_states[..., :3]

            self._traj_marker_actor_ids = self._humanoid_actor_ids.unsqueeze(-1) + to_torch(self._traj_marker_handles, dtype=torch.int32, device=self.device)
            self._traj_marker_actor_ids = self._traj_marker_actor_ids.flatten()

        return
    
    def register_task_sit_pre_init(self, cfg):
        k = "sit"
        assert HumanoidTrajSitCarryClimb.TaskUID[k].value >= 0

        self._sit_rwd_vel_penalty = cfg["env"][k]["sit_vel_penalty"]
        self._sit_rwd_vel_pen_coeff = cfg["env"][k]["sit_vel_pen_coeff"]
        self._sit_rwd_vel_pen_thre = cfg["env"][k]["sit_vel_pen_threshold"]
        self._sit_rwd_ang_vel_pen_coeff = cfg["env"][k]["sit_ang_vel_pen_coeff"]
        self._sit_rwd_ang_vel_pen_thre = cfg["env"][k]["sit_ang_vel_pen_threshold"]

        self._num_tasks += 1
        self._each_subtask_obs_size.append(3 + 3 + 6 + 2 + 8 * 3) # target sit position, object pos + 6d rot, object 2d facing dir, object bps
        self._multiple_task_names.append(k)

        # Interaction Early Termination (IET)
        self._sit_enable_IET = cfg["env"][k]["enableIET"]

        return
    
    def register_task_sit_post_init(self, cfg):
        k = "sit"

        self._sit_skill = cfg["env"][k]["skill"]
        self._sit_skill_init_prob = torch.tensor(cfg["env"][k]["skillInitProb"], device=self.device, dtype=torch.float)

        self._sit_tar_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float) # target location of the object, 3d xyz

        if self._is_eval:
            self._sit_skill = cfg["env"][k]["eval"]["skill"]
            self._sit_skill_init_prob = torch.tensor(cfg["env"][k]["eval"]["skillInitProb"], device=self.device, dtype=torch.float)

        if (not self.headless):
            num_actors = self._root_states.shape[0] // self.num_envs

            idx = self._sit_marker_handles[0]
            self._sit_marker_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., idx, :]
            self._sit_marker_pos = self._sit_marker_states[..., :3]
            
            self._sit_marker_actor_ids = self._humanoid_actor_ids + idx

        # tensors for object
        num_actors = self.get_num_actors_per_env()

        idx = self._sit_object_handles[0]
        self._sit_object_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., idx, :]
        
        self._sit_object_actor_ids = self._humanoid_actor_ids + idx

        self._initial_sit_object_states = self._sit_object_states.clone()
        self._initial_sit_object_states[:, 7:13] = 0

        return
    
    def register_task_carry_pre_init(self, cfg):
        k = "carry"
        assert HumanoidTrajSitCarryClimb.TaskUID[k].value >= 0

        self._carry_rwd_only_vel_reward = cfg["env"][k]["onlyVelReward"]
        self._carry_rwd_only_height_handheld_reward = cfg["env"][k]["onlyHeightHandHeldReward"]

        self._carry_rwd_box_vel_penalty = cfg["env"][k]["box_vel_penalty"]
        self._carry_rwd_box_vel_pen_coeff = cfg["env"][k]["box_vel_pen_coeff"]
        self._carry_rwd_box_vel_pen_thre = cfg["env"][k]["box_vel_pen_threshold"]

        self._carry_reset_random_rot = cfg["env"][k]["box"]["reset"]["randomRot"]
        self._carry_reset_random_height = cfg["env"][k]["box"]["reset"]["randomHeight"]
        self._carry_reset_random_height_prob = cfg["env"][k]["box"]["reset"]["randomHeightProb"]
        self._carry_reset_maxTopSurfaceHeight = cfg["env"][k]["box"]["reset"]["maxTopSurfaceHeight"]
        self._carry_reset_minBottomSurfaceHeight = cfg["env"][k]["box"]["reset"]["minBottomSurfaceHeight"]

        self._num_tasks += 1
        self._each_subtask_obs_size.append(3 + 3 + 6 + 3 + 3 + 8 * 3) # target box location, box state: pos (3) + rot (6) + lin vel (3) + ang vel (6), bps
        self._multiple_task_names.append(k)

        return
    
    def register_task_carry_post_init(self, cfg):
        k = "carry"

        self._carry_skill = cfg["env"][k]["skill"]
        self._carry_skill_init_prob = torch.tensor(cfg["env"][k]["skillInitProb"], device=self.device, dtype=torch.float)

        self._box_tar_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float) # target location of the box, 3d xyz

        if self._is_eval:
            self._carry_skill = cfg["env"][k]["eval"]["skill"]
            self._carry_skill_init_prob = torch.tensor(cfg["env"][k]["eval"]["skillInitProb"], device=self.device, dtype=torch.float)

        spacing = cfg["env"]["envSpacing"]
        if spacing <= 0.5:
            self._box_tar_pos_dist = torch.distributions.uniform.Uniform(
                torch.tensor([-4.5, -4.5, 0.5], device=self.device),
                torch.tensor([4.5, 4.5, 1.0], device=self.device))
        else:
            self._box_tar_pos_dist = torch.distributions.uniform.Uniform(
                torch.tensor([-(spacing - 0.5), -(spacing - 0.5), 0.5], device=self.device),
                torch.tensor([(spacing - 0.5), (spacing - 0.5), 1.0], device=self.device))

        self._prev_box_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)

        num_actors = self.get_num_actors_per_env()
        idx = self._box_handles[0]
        self._box_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., idx, :]
        
        self._box_actor_ids = self._humanoid_actor_ids + idx

        self._initial_box_states = self._box_states.clone()
        self._initial_box_states[:, 7:13] = 0

        # tensors for platforms
        if self._carry_reset_random_height:
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
    
    def register_task_climb_pre_init(self, cfg):
        k = "climb"
        assert HumanoidTrajSitCarryClimb.TaskUID[k].value >= 0

        self._num_tasks += 1
        self._each_subtask_obs_size.append(3 + 8 * 3) # target root position, object bps
        self._multiple_task_names.append(k)

        self._climb_rwd_vel_penalty = cfg["env"][k]["climb_vel_penalty"]
        self._climb_rwd_vel_pen_coeff = cfg["env"][k]["climb_vel_pen_coeff"]
        self._climb_rwd_vel_pen_thre = cfg["env"][k]["climb_vel_pen_threshold"]

        # Interaction Early Termination (IET)
        self._climb_enable_IET = cfg["env"][k]["enableIET"]

        return
    
    def register_task_climb_post_init(self, cfg):
        k = "climb"

        self._climb_skill = cfg["env"][k]["skill"]
        self._climb_skill_init_prob = torch.tensor(cfg["env"][k]["skillInitProb"], device=self.device, dtype=torch.float)

        self._climb_tar_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float) # target location of the object, 3d xyz

        if self._is_eval:
            self._climb_skill = cfg["env"][k]["eval"]["skill"]
            self._climb_skill_init_prob = torch.tensor(cfg["env"][k]["eval"]["skillInitProb"], device=self.device, dtype=torch.float)

        if (not self.headless):
            num_actors = self._root_states.shape[0] // self.num_envs

            idx = self._climb_marker_handles[0]
            self._climb_marker_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., idx, :]
            self._climb_marker_pos = self._climb_marker_states[..., :3]
            
            self._climb_marker_actor_ids = self._humanoid_actor_ids + idx

        # tensors for object
        num_actors = self.get_num_actors_per_env()

        idx = self._climb_object_handles[0]
        self._climb_object_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., idx, :]
        
        self._climb_object_actor_ids = self._humanoid_actor_ids + idx

        self._initial_climb_object_states = self._climb_object_states.clone()
        self._initial_climb_object_states[:, 7:13] = 0

        return
    
    def post_process_disc_dataset_collection(self, cfg):
        skill_idx = {sk_name: i for i, sk_name in enumerate(self._skill)}

        task_names = cfg["env"]["task"]
        for i, n in enumerate(task_names):
            print("checking whether task {} is enabled".format(n))
            prob = self._task_init_prob[i]
            if not prob > 0:
                print("task {} is not enabled".format(n))

                its_skill_set = cfg["env"][n]["skill"]
                for k in its_skill_set:
                    if k != self._common_skill:
                        self._skill_disc_prob[skill_idx[k]] = 0.0 # clear its skill init prob for training disc
        
        print("disc skill: ", self._skill)
        print("disc prob: ", self._skill_disc_prob)

        scale_coeff = 1.0 / sum(self._skill_disc_prob)
        self._skill_disc_prob *= scale_coeff

        assert torch.abs(self._skill_disc_prob.sum() - 1.0 ) < 1e-5

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
            if (self._enable_task_mask_obs):
                obs_size += self._num_tasks
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
        }
        return info

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._prev_root_pos[:] = self._humanoid_root_states[..., 0:3]
        self._prev_root_rot[:] = self._humanoid_root_states[..., 3:7]
        self._prev_box_pos[:] = self._box_states[..., 0:3]
        return

    def _update_marker(self):
        traj_samples = self._fetch_traj_samples()
        self._traj_marker_pos[:] = traj_samples
        self._traj_marker_pos[..., 2] = self._char_h

        traj_env_mask = self._task_indicator == HumanoidTrajSitCarryClimb.TaskUID["traj"].value
        self._traj_marker_pos[~traj_env_mask, :, 2] = -10.0

        self._sit_marker_pos[:] = self._sit_tar_pos # 3d xyz
        self._climb_marker_pos[:] = self._climb_tar_pos # 3d xyz

        actor_ids = torch.cat([self._traj_marker_actor_ids, self._sit_marker_actor_ids, self._sit_object_actor_ids, self._box_actor_ids, self._climb_marker_actor_ids, self._climb_object_actor_ids,], dim=0)
        if self._carry_reset_random_height:
            # env has two platforms
            actor_ids = torch.cat([actor_ids, self._platform_actor_ids, self._tar_platform_actor_ids], dim=0)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(actor_ids), len(actor_ids))
        return
    
    def _create_envs(self, num_envs, spacing, num_per_row):
        if (not self.headless):
            self._traj_marker_handles = [[] for _ in range(self.num_envs)]
            self._sit_marker_handles = []
            self._climb_marker_handles = []
            self._load_marker_asset()
        
        # load objects used in the sit task
        self._sit_obj_lib = ObjectLib(
            mode=self._mode,
            dataset_root=os.path.join(os.path.dirname(self.cfg['env']['motion_file']), self.cfg["env"]["sit"]["objDatasetDir"]),
            dataset_categories=self.cfg["env"]["sit"]["objCategories"],
            num_envs=self.num_envs,
            device=self.device,
        )

        # load physical assets
        self._sit_object_handles = []
        self._sit_object_assets = self._load_object_asset(self._sit_obj_lib._obj_urdfs)

        # load boxes used in the carry task
        self._box_lib = BoxLib(self._mode, self.cfg["env"]["carry"]["box"], self.num_envs, self.device)

        # load physical assets
        self._box_handles = []
        self._box_assets = self._load_box_asset(self._box_lib._box_size)

        if self._carry_reset_random_height:
            self._platform_handles = []
            self._tar_platform_handles = []
            self._load_platform_asset()

        # load objects used in the climb task
        self._climb_obj_lib = ObjectLib(
            mode=self._mode,
            dataset_root=os.path.join(os.path.dirname(self.cfg['env']['motion_file']), self.cfg["env"]["climb"]["objDatasetDir"]),
            dataset_categories=self.cfg["env"]["climb"]["objCategories"],
            num_envs=self.num_envs,
            device=self.device,
        )

        # load physical assets
        self._climb_object_handles = []
        self._climb_object_assets = self._load_object_asset(self._climb_obj_lib._obj_urdfs)

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
    
    def _load_box_asset(self, box_sizes):
        box_assets = []
        for i in range(self.num_envs):
            asset_options = gymapi.AssetOptions()
            asset_options.angular_damping = 0.01
            asset_options.linear_damping = 0.01
            asset_options.max_angular_velocity = 100.0
            asset_options.density = 100.0
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            box_assets.append(self.gym.create_box(self.sim, box_sizes[i, 0], box_sizes[i, 1], box_sizes[i, 2], asset_options))
        return box_assets
    
    def _load_object_asset(self, object_urdfs):
        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.fix_base_link = True # fix it
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        # Load materials from meshes
        asset_options.use_mesh_materials = True
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX

        # use default convex decomposition params
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params = gymapi.VhacdParams()
        asset_options.vhacd_params.resolution = 100000
        asset_options.vhacd_params.max_convex_hulls = 128
        asset_options.vhacd_params.max_num_vertices_per_ch = 64

        asset_options.replace_cylinder_with_capsule = False # support cylinder

        asset_root = "./"
        object_assets = []
        for urdf in object_urdfs:
            object_assets.append(self.gym.load_asset(self.sim, asset_root, urdf, asset_options))

        return object_assets

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)

        self._build_sit_object(env_id, env_ptr)

        self._build_box(env_id, env_ptr)
        if self._carry_reset_random_height:
            self._build_platforms(env_id, env_ptr)
        
        self._build_climb_object(env_id, env_ptr)

        if (not self.headless):
            self._build_marker(env_id, env_ptr)

        return
    
    def _build_traj_generator(self):
        num_envs = self.num_envs
        episode_dur = self.max_episode_length_short * self.dt # 300 * 0.033333 = 10s
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

        # traj markers
        for i in range(self._num_traj_samples):

            marker_handle = self.gym.create_actor(env_ptr, self._marker_asset, default_pose, "marker", col_group, col_filter, segmentation_id)
            self.gym.set_actor_scale(env_ptr, marker_handle, 0.5)
            self.gym.set_rigid_body_color(env_ptr, marker_handle, 0,
                                          gymapi.MESH_VISUAL,
                                          gymapi.Vec3(1.0, 0.0, 0.0))
            self._traj_marker_handles[env_id].append(marker_handle)
        
        # sit markers
        marker_handle = self.gym.create_actor(env_ptr, self._marker_asset, default_pose, "marker", col_group, col_filter, segmentation_id)
        self.gym.set_rigid_body_color(env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.0, 0.0))
        self.gym.set_actor_scale(env_ptr, marker_handle, 0.5)
        self._sit_marker_handles.append(marker_handle)

        # climb markers
        marker_handle = self.gym.create_actor(env_ptr, self._marker_asset, default_pose, "marker", col_group, col_filter, segmentation_id)
        self.gym.set_rigid_body_color(env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.0, 0.0))
        self.gym.set_actor_scale(env_ptr, marker_handle, 0.5)
        self._climb_marker_handles.append(marker_handle)

        return
    
    def _build_sit_object(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        default_pose = gymapi.Transform()
        default_pose.p.x = 1
        default_pose.p.y = 0
        default_pose.p.z = self._sit_obj_lib._every_env_object_on_ground_trans[env_id] # ensure no penetration between object and ground plane
        
        object_handle = self.gym.create_actor(env_ptr, self._sit_object_assets[self._sit_obj_lib._every_env_object_ids[env_id]], default_pose, "object", col_group, col_filter, segmentation_id)
        self._sit_object_handles.append(object_handle)

        return
    
    def _build_climb_object(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        default_pose = gymapi.Transform()
        default_pose.p.x = 10
        default_pose.p.y = 0
        default_pose.p.z = self._climb_obj_lib._every_env_object_on_ground_trans[env_id] # ensure no penetration between object and ground plane
        
        object_handle = self.gym.create_actor(env_ptr, self._climb_object_assets[self._climb_obj_lib._every_env_object_ids[env_id]], default_pose, "object", col_group, col_filter, segmentation_id)
        self._climb_object_handles.append(object_handle)

        return
    
    def _build_box(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        default_pose = gymapi.Transform()
        default_pose.p.x = 5
        default_pose.p.y = 0
        default_pose.p.z = self._box_lib._box_size[env_id, 2] / 2 # ensure no penetration between box and ground plane
    
        box_handle = self.gym.create_actor(env_ptr, self._box_assets[env_id], default_pose, "box", col_group, col_filter, segmentation_id)
        self._box_handles.append(box_handle)
        return
    
    def _build_platforms(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0
        default_pose = gymapi.Transform()

        default_pose.p.z = -10 # place under the ground
        platform_handle = self.gym.create_actor(env_ptr, self._platform_asset, default_pose, "platform", col_group, col_filter, segmentation_id)
        self.gym.set_rigid_body_color(env_ptr, platform_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.5, 0.235, 0.6))

        default_pose.p.z = -10 - self._platform_height
        tar_platform_handle = self.gym.create_actor(env_ptr, self._platform_asset, default_pose, "tar_platform", col_group, col_filter, segmentation_id)
        self.gym.set_rigid_body_color(env_ptr, tar_platform_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.0, 0.0, 0.8))

        self._platform_handles.append(platform_handle)
        self._tar_platform_handles.append(tar_platform_handle)

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

            sit_object_states = self._sit_object_states
            sit_object_bps = self._sit_obj_lib._every_env_object_bps
            sit_object_facings = self._sit_obj_lib._every_env_object_facings
            sit_tar_pos = self._sit_tar_pos

            box_states = self._box_states
            box_bps = self._box_lib._box_bps
            box_tar_pos = self._box_tar_pos

            climb_object_states = self._climb_object_states
            climb_object_bps = self._climb_obj_lib._every_env_object_bps
            climb_tar_pos = self._climb_tar_pos

            task_mask = self._task_mask
            task_indicator = self._task_indicator
        else:
            root_states = self._humanoid_root_states[env_ids]

            sit_object_states = self._sit_object_states[env_ids]
            sit_object_bps = self._sit_obj_lib._every_env_object_bps[env_ids]
            sit_object_facings = self._sit_obj_lib._every_env_object_facings[env_ids]
            sit_tar_pos = self._sit_tar_pos[env_ids]

            box_states = self._box_states[env_ids]
            box_bps = self._box_lib._box_bps[env_ids]
            box_tar_pos = self._box_tar_pos[env_ids]

            climb_object_states = self._climb_object_states[env_ids]
            climb_object_bps = self._climb_obj_lib._every_env_object_bps[env_ids]
            climb_tar_pos = self._climb_tar_pos[env_ids]
            
            task_mask = self._task_mask[env_ids]
            task_indicator = self._task_indicator[env_ids]

        traj_samples = self._fetch_traj_samples(env_ids)
        obs = compute_location_observations(root_states, traj_samples,
                                            sit_tar_pos, sit_object_states, sit_object_bps, sit_object_facings,
                                            box_states, box_bps, box_tar_pos,
                                            climb_object_states, climb_object_bps, climb_tar_pos,
                                            task_mask, self.get_multi_task_info()["each_subtask_obs_mask"], self._enable_apply_mask_on_task_obs)

        if (self._enable_task_mask_obs):
            obs = torch.cat([obs, task_mask.float()], dim=-1)

        return obs

    def _compute_reward(self, actions):
        root_pos = self._humanoid_root_states[..., 0:3]
        root_rot = self._humanoid_root_states[..., 3:7]
        sit_object_pos = self._sit_object_states[..., 0:3]
        sit_object_rot = self._sit_object_states[..., 3:7]
        climb_object_pos = self._climb_object_states[..., 0:3]
        climb_object_rot = self._climb_object_states[..., 3:7]

        rigid_body_pos = self._rigid_body_pos
        box_pos = self._box_states[..., 0:3]
        box_rot = self._box_states[..., 3:7]
        hands_ids = self._key_body_ids[[0, 1]]
        feet_ids = self._key_body_ids[[2, 3]]

        time = self.progress_buf * self.dt
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        traj_tar_pos = self._traj_gen.calc_pos(env_ids, time)

        reward = self.rew_buf.clone()

        traj_env_mask = self._task_indicator == HumanoidTrajSitCarryClimb.TaskUID["traj"].value
        if traj_env_mask.sum() > 0:
            reward[traj_env_mask] = compute_traj_reward(root_pos[traj_env_mask], traj_tar_pos[traj_env_mask])
        
        sit_env_mask = self._task_indicator == HumanoidTrajSitCarryClimb.TaskUID["sit"].value
        if sit_env_mask.sum() > 0:
            reward[sit_env_mask] = compute_sit_reward(
                root_pos[sit_env_mask], self._prev_root_pos[sit_env_mask],
                root_rot[sit_env_mask], self._prev_root_rot[sit_env_mask],
                sit_object_pos[sit_env_mask], self._sit_tar_pos[sit_env_mask], 1.5, self.dt,
                self._sit_rwd_vel_penalty, self._sit_rwd_vel_pen_coeff, self._sit_rwd_vel_pen_thre, self._sit_rwd_ang_vel_pen_coeff, self._sit_rwd_ang_vel_pen_thre)

        carry_env_mask = self._task_indicator == HumanoidTrajSitCarryClimb.TaskUID["carry"].value
        if carry_env_mask.sum() > 0:
            walk_r = compute_walk_reward(root_pos[carry_env_mask], self._prev_root_pos[carry_env_mask], box_pos[carry_env_mask], self.dt, 1.5, self._carry_rwd_only_vel_reward)
            carry_r = compute_carry_reward(box_pos[carry_env_mask], self._prev_box_pos[carry_env_mask], self._box_tar_pos[carry_env_mask], self.dt, 1.5, self._box_lib._box_size[carry_env_mask], 
                                        self._carry_rwd_only_vel_reward,
                                        self._carry_rwd_box_vel_penalty, self._carry_rwd_box_vel_pen_coeff, self._carry_rwd_box_vel_pen_thre,)
            handheld_r = compute_handheld_reward(rigid_body_pos[carry_env_mask], box_pos[carry_env_mask], hands_ids, self._box_tar_pos[carry_env_mask], self._carry_rwd_only_height_handheld_reward)
            putdown_r = compute_putdown_reward(box_pos[carry_env_mask], self._box_tar_pos[carry_env_mask])
            reward[carry_env_mask] = walk_r + carry_r + handheld_r + putdown_r

        climb_env_mask = self._task_indicator == HumanoidTrajSitCarryClimb.TaskUID["climb"].value
        if climb_env_mask.sum() > 0:
            reward[climb_env_mask] = compute_climb_reward(root_pos[climb_env_mask], self._prev_root_pos[climb_env_mask], climb_object_pos[climb_env_mask], self.dt, self._climb_tar_pos[climb_env_mask],
                                                         rigid_body_pos[climb_env_mask], feet_ids, self._char_h, self._climb_obj_lib._every_env_object_valid_radius[climb_env_mask],
                                                         self._climb_rwd_vel_penalty, self._climb_rwd_vel_pen_coeff, self._climb_rwd_vel_pen_thre)

        power = torch.abs(torch.multiply(self.dof_force_tensor, self._dof_vel)).sum(dim = -1)
        power_reward = -self._power_coefficient * power

        if self._power_reward:
            self.rew_buf[:] = reward + power_reward
        else:
            self.rew_buf[:] = reward

        if self._enable_IET:
            IET_tar_pos = torch.zeros_like(self._sit_tar_pos)
            IET_tar_pos[..., 2] = 100.0

            if self._sit_enable_IET:
                IET_tar_pos[sit_env_mask] = self._sit_tar_pos[sit_env_mask]
            
            if self._climb_enable_IET:
                IET_tar_pos[climb_env_mask] = self._climb_tar_pos[climb_env_mask]

            mask = compute_finish_state(root_pos, IET_tar_pos, self._success_threshold)
            self._IET_step_buf[mask] += 1

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

        # draw lines of the bbox
        cols = np.zeros((12, 3), dtype=np.float32) # 12 lines
        cols[:, :] = [0.0, 1.0, 0.0] # green

        red_cols = np.zeros((12, 3), dtype=np.float32) # 12 lines
        red_cols[:, :] = [1.0, 0.0, 0.0] # red

        # transform bps from object local space to world space
        tar_box_pos = self._box_tar_pos[:, 0:3] # (num_envs, 3)
        tar_box_pos_exp = torch.broadcast_to(tar_box_pos.unsqueeze(-2), (tar_box_pos.shape[0], self._box_lib._box_bps.shape[1], tar_box_pos.shape[1])) # (num_envs, 3) >> (num_envs, 8, 3)
        tar_box_bps_world_space = (self._box_lib._box_bps.reshape(-1, 3) + tar_box_pos_exp.reshape(-1, 3)).reshape(self.num_envs, 8, 3) # (num_envs, 8, 3)

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


        # transform bps from object local space to world space
        curr_box_pos = self._box_states[:, 0:3] # (num_envs, 3)
        curr_box_rot = self._box_states[:, 3:7]
        curr_box_pos_exp = torch.broadcast_to(curr_box_pos.unsqueeze(-2), (curr_box_pos.shape[0], self._box_lib._box_bps.shape[1], curr_box_pos.shape[1])) # (num_envs, 3) >> (num_envs, 8, 3)
        curr_box_rot_exp = torch.broadcast_to(curr_box_rot.unsqueeze(-2), (curr_box_rot.shape[0], self._box_lib._box_bps.shape[1], curr_box_rot.shape[1])) # (num_envs, 4) >> (num_envs, 8, 4)
        curr_box_bps_world_space = (quat_rotate(curr_box_rot_exp.reshape(-1, 4), self._box_lib._box_bps.reshape(-1, 3)) + curr_box_pos_exp.reshape(-1, 3)).reshape(self.num_envs, 8, 3) # (num_envs, 8, 3)

        verts_curr_box = torch.cat([
            curr_box_bps_world_space[:, 0, :], curr_box_bps_world_space[:, 1, :],
            curr_box_bps_world_space[:, 1, :], curr_box_bps_world_space[:, 2, :],
            curr_box_bps_world_space[:, 2, :], curr_box_bps_world_space[:, 3, :],
            curr_box_bps_world_space[:, 3, :], curr_box_bps_world_space[:, 0, :],

            curr_box_bps_world_space[:, 4, :], curr_box_bps_world_space[:, 5, :],
            curr_box_bps_world_space[:, 5, :], curr_box_bps_world_space[:, 6, :],
            curr_box_bps_world_space[:, 6, :], curr_box_bps_world_space[:, 7, :],
            curr_box_bps_world_space[:, 7, :], curr_box_bps_world_space[:, 4, :],

            curr_box_bps_world_space[:, 0, :], curr_box_bps_world_space[:, 4, :],
            curr_box_bps_world_space[:, 1, :], curr_box_bps_world_space[:, 5, :],
            curr_box_bps_world_space[:, 2, :], curr_box_bps_world_space[:, 6, :],
            curr_box_bps_world_space[:, 3, :], curr_box_bps_world_space[:, 7, :],
        ], dim=-1).cpu() # (num_envs, 12*6)


        # transform bps from object local space to world space
        curr_box_pos = self._sit_object_states[:, 0:3] # (num_envs, 3)
        curr_box_rot = self._sit_object_states[:, 3:7]
        curr_box_pos_exp = torch.broadcast_to(curr_box_pos.unsqueeze(-2), (curr_box_pos.shape[0], self._box_lib._box_bps.shape[1], curr_box_pos.shape[1])) # (num_envs, 3) >> (num_envs, 8, 3)
        curr_box_rot_exp = torch.broadcast_to(curr_box_rot.unsqueeze(-2), (curr_box_rot.shape[0], self._box_lib._box_bps.shape[1], curr_box_rot.shape[1])) # (num_envs, 4) >> (num_envs, 8, 4)
        curr_box_bps_world_space = (quat_rotate(curr_box_rot_exp.reshape(-1, 4), self._sit_obj_lib._every_env_object_bps.reshape(-1, 3)) + curr_box_pos_exp.reshape(-1, 3)).reshape(self.num_envs, 8, 3) # (num_envs, 8, 3)

        verts_curr_sit_obj = torch.cat([
            curr_box_bps_world_space[:, 0, :], curr_box_bps_world_space[:, 1, :],
            curr_box_bps_world_space[:, 1, :], curr_box_bps_world_space[:, 2, :],
            curr_box_bps_world_space[:, 2, :], curr_box_bps_world_space[:, 3, :],
            curr_box_bps_world_space[:, 3, :], curr_box_bps_world_space[:, 0, :],

            curr_box_bps_world_space[:, 4, :], curr_box_bps_world_space[:, 5, :],
            curr_box_bps_world_space[:, 5, :], curr_box_bps_world_space[:, 6, :],
            curr_box_bps_world_space[:, 6, :], curr_box_bps_world_space[:, 7, :],
            curr_box_bps_world_space[:, 7, :], curr_box_bps_world_space[:, 4, :],

            curr_box_bps_world_space[:, 0, :], curr_box_bps_world_space[:, 4, :],
            curr_box_bps_world_space[:, 1, :], curr_box_bps_world_space[:, 5, :],
            curr_box_bps_world_space[:, 2, :], curr_box_bps_world_space[:, 6, :],
            curr_box_bps_world_space[:, 3, :], curr_box_bps_world_space[:, 7, :],
        ], dim=-1).cpu() # (num_envs, 12*6)

        # transform bps from object local space to world space
        curr_box_pos = self._climb_object_states[:, 0:3] # (num_envs, 3)
        curr_box_rot = self._climb_object_states[:, 3:7]
        curr_box_pos_exp = torch.broadcast_to(curr_box_pos.unsqueeze(-2), (curr_box_pos.shape[0], self._box_lib._box_bps.shape[1], curr_box_pos.shape[1])) # (num_envs, 3) >> (num_envs, 8, 3)
        curr_box_rot_exp = torch.broadcast_to(curr_box_rot.unsqueeze(-2), (curr_box_rot.shape[0], self._box_lib._box_bps.shape[1], curr_box_rot.shape[1])) # (num_envs, 4) >> (num_envs, 8, 4)
        curr_box_bps_world_space = (quat_rotate(curr_box_rot_exp.reshape(-1, 4), self._climb_obj_lib._every_env_object_bps.reshape(-1, 3)) + curr_box_pos_exp.reshape(-1, 3)).reshape(self.num_envs, 8, 3) # (num_envs, 8, 3)

        verts_curr_climb_obj = torch.cat([
            curr_box_bps_world_space[:, 0, :], curr_box_bps_world_space[:, 1, :],
            curr_box_bps_world_space[:, 1, :], curr_box_bps_world_space[:, 2, :],
            curr_box_bps_world_space[:, 2, :], curr_box_bps_world_space[:, 3, :],
            curr_box_bps_world_space[:, 3, :], curr_box_bps_world_space[:, 0, :],

            curr_box_bps_world_space[:, 4, :], curr_box_bps_world_space[:, 5, :],
            curr_box_bps_world_space[:, 5, :], curr_box_bps_world_space[:, 6, :],
            curr_box_bps_world_space[:, 6, :], curr_box_bps_world_space[:, 7, :],
            curr_box_bps_world_space[:, 7, :], curr_box_bps_world_space[:, 4, :],

            curr_box_bps_world_space[:, 0, :], curr_box_bps_world_space[:, 4, :],
            curr_box_bps_world_space[:, 1, :], curr_box_bps_world_space[:, 5, :],
            curr_box_bps_world_space[:, 2, :], curr_box_bps_world_space[:, 6, :],
            curr_box_bps_world_space[:, 3, :], curr_box_bps_world_space[:, 7, :],
        ], dim=-1).cpu() # (num_envs, 12*6)

        for i, env_ptr in enumerate(self.envs):

            # traj
            verts = self._traj_gen.get_traj_verts(i)
            
            verts[..., 2] = self._char_h + (-10 * int(~(self._task_indicator[i] == HumanoidTrajSitCarryClimb.TaskUID["traj"].value)))
            lines = torch.cat([verts[:-1], verts[1:]], dim=-1).cpu().numpy()
            curr_cols = np.broadcast_to(traj_cols, [lines.shape[0], traj_cols.shape[-1]])
            self.gym.add_lines(self.viewer, env_ptr, lines.shape[0], lines, curr_cols)

            curr_verts = verts_tar_box[i].numpy()
            curr_verts = curr_verts.reshape([12, 6])
            self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, cols)

            curr_verts = verts_curr_box[i].numpy()
            curr_verts = curr_verts.reshape([12, 6])
            self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, red_cols)

            curr_verts = verts_curr_sit_obj[i].numpy()
            curr_verts = curr_verts.reshape([12, 6])
            self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, red_cols)

            curr_verts = verts_curr_climb_obj[i].numpy()
            curr_verts = curr_verts.reshape([12, 6])
            self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, red_cols)

        return

    def post_physics_step(self):
        super().post_physics_step()
        
        self._update_hist_amp_obs()
        self._compute_amp_observations()

        amp_obs_flat = self._amp_obs_buf.view(-1, self.get_num_amp_obs())
        self.extras["amp_obs"] = amp_obs_flat
        self.extras["policy_obs"] = self.obs_buf.clone()

        if self._is_eval:
            if self._eval_task in ["sit", "climb"]:
                self._compute_metrics_evaluation_v1()
            elif self._eval_task == "traj":
                self._compute_metrics_evaluation_v2()
            elif self._eval_task == "carry":
                self._compute_metrics_evaluation_v3()
            else:
                raise NotImplementedError
            self.extras["success"] = self._success_buf
            self.extras["precision"] = self._precision_buf
        
        for task_name in self._multiple_task_names:
            curr_task_uid = HumanoidTrajSitCarryClimb.TaskUID[task_name].value
            curr_env_mask = self._task_indicator == curr_task_uid

            self.extras[task_name] = curr_env_mask

        return
    
    def _compute_metrics_evaluation_v1(self):
        root_pos = self._humanoid_root_states[..., 0:3]
        root_rot = self._humanoid_root_states[..., 3:7]

        tar_pos = torch.zeros_like(root_pos)
        
        sit_env_mask = self._task_indicator == HumanoidTrajSitCarryClimb.TaskUID["sit"].value
        tar_pos[sit_env_mask] = self._sit_tar_pos[sit_env_mask]

        climb_env_mask = self._task_indicator == HumanoidTrajSitCarryClimb.TaskUID["climb"].value
        tar_pos[climb_env_mask] = self._climb_tar_pos[climb_env_mask]

        pos_diff = tar_pos - root_pos
        pos_err = torch.norm(pos_diff, p=2, dim=-1)
        dist_mask = pos_err <= self._success_threshold
        self._success_buf[dist_mask] += 1

        self._precision_buf[dist_mask] = torch.where(pos_err[dist_mask] < self._precision_buf[dist_mask], pos_err[dist_mask], self._precision_buf[dist_mask])

        return
    
    def _compute_metrics_evaluation_v2(self):
        root_pos = self._humanoid_root_states[..., 0:3]
        root_rot = self._humanoid_root_states[..., 3:7]

        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        timesteps = torch.ones_like(env_ids, device=self.device, dtype=torch.float)

        coeff = 0.98

        timesteps[:] = self.max_episode_length_short * coeff * self.dt # float('Inf')
        traj_final_tar_pos = self._traj_gen.calc_pos(env_ids, timesteps)

        timesteps[:] = self.progress_buf * self.dt
        traj_curr_tar_pos = self._traj_gen.calc_pos(env_ids, timesteps)

        pos_diff = traj_final_tar_pos[..., 0:2] - root_pos[..., 0:2]
        pos_err = torch.norm(pos_diff, p=2, dim=-1)
        dist_mask = pos_err <= (self._success_threshold + 0.1)
        time_mask = self.progress_buf >= self.max_episode_length_short * coeff

        success_mask = torch.logical_and(dist_mask, time_mask)
        self._success_buf[success_mask] += 1

        pos_diff = traj_curr_tar_pos[..., 0:2] - root_pos[..., 0:2]
        pos_err = torch.norm(pos_diff, p=2, dim=-1)
        self._precision_buf += pos_err

        return
    
    def _compute_metrics_evaluation_v3(self):
        box_root_pos = self._box_states[..., 0:3]

        pos_diff = self._box_tar_pos - box_root_pos
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

        # assign a task label
        task_onehot = torch.zeros(num_samples * self._num_amp_obs_steps, self._num_tasks, device=self.device, dtype=torch.float32)
        if sk_name != self._common_skill:
            if sk_name in self._traj_skill:
                task_onehot[..., HumanoidTrajSitCarryClimb.TaskUID["traj"].value] = 1.0
            elif sk_name in self._sit_skill:
                task_onehot[..., HumanoidTrajSitCarryClimb.TaskUID["sit"].value] = 1.0
            elif sk_name in self._carry_skill:
                task_onehot[..., HumanoidTrajSitCarryClimb.TaskUID["carry"].value] = 1.0
            elif sk_name in self._climb_skill:
                task_onehot[..., HumanoidTrajSitCarryClimb.TaskUID["climb"].value] = 1.0
            else:
                raise NotImplementedError
        else:
            prob = self._task_init_prob.clone()
            nonzero = prob.nonzero().squeeze(-1)
            prob[nonzero] = 1.0 / nonzero.shape[0]
            random_task = torch.multinomial(prob, num_samples=1, replacement=True)
            task_onehot[..., random_task] = 1.0

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

        if self._enable_task_specific_disc:
            amp_obs_demo = torch.cat([amp_obs_demo, task_onehot], dim=-1)

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

        if self._enable_task_specific_disc:
            self._num_amp_obs_per_step += self._num_tasks

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
    
    def _reset_task_traj(self, env_ids):
        root_pos = self._humanoid_root_states[env_ids, 0:3]
        self._traj_gen.reset(env_ids, root_pos)
        return
    
    def _reset_task_sit(self, env_ids):

        not_task_env_ids = env_ids[self._task_indicator[env_ids] != HumanoidTrajSitCarryClimb.TaskUID["sit"].value]
        if len(not_task_env_ids) > 0:
            self._sit_object_states[not_task_env_ids, :] = 0.0
            self._sit_object_states[not_task_env_ids, 2] = -3.0 # place under the ground
            self._sit_object_states[not_task_env_ids, 6] = 1.0 # quat

        is_task_env_ids = env_ids[self._task_indicator[env_ids] == HumanoidTrajSitCarryClimb.TaskUID["sit"].value]
        if len(is_task_env_ids) > 0:
            local_reset_ref_env_ids = self._reset_ref_env_ids["sit"]
            local_reset_ref_motion_ids = self._reset_ref_motion_ids["sit"]
            local_reset_ref_motion_times = self._reset_ref_motion_times["sit"]

            # for skill is sit, the initial location of the object is from the reference object motion
            for sk_name in ["sit"]:
                if local_reset_ref_env_ids.get(sk_name) is not None:
                    if (len(local_reset_ref_env_ids[sk_name]) > 0):
                        object_root_pos, object_root_rot = self._motion_lib[sk_name].get_obj_motion_state_single_frame(local_reset_ref_motion_ids[sk_name])
                        self._sit_object_states[local_reset_ref_env_ids[sk_name], 0:2] = object_root_pos[..., 0:2]
                        self._sit_object_states[local_reset_ref_env_ids[sk_name], 2] = self._sit_obj_lib._every_env_object_on_ground_trans[local_reset_ref_env_ids[sk_name]]
                        self._sit_object_states[local_reset_ref_env_ids[sk_name], 3:7] = object_root_rot
                        self._sit_object_states[local_reset_ref_env_ids[sk_name], 7:13] = 0.0 # clear vels

            # for skill is loco, we random generate an inital location of the object
            random_env_ids = []
            for sk_name in ["loco_sit"]:
                if local_reset_ref_env_ids.get(sk_name) is not None:
                    random_env_ids.append(local_reset_ref_env_ids[sk_name])

            if len(random_env_ids) > 0:
                ids = torch.cat(random_env_ids, dim=0)

                root_pos_xy = torch.randn(len(ids), 2, device=self.device)
                root_pos_xy /= torch.linalg.norm(root_pos_xy, dim=-1, keepdim=True)
                root_pos_xy *= torch.rand(len(ids), 1, device=self.device) * 4.0 + 1.0 # randomize
                root_pos_xy += self._humanoid_root_states[ids, :2] # get absolute pos, humanoid_root_state will be updated after set_env_state

                root_pos_z = torch.zeros((len(ids)), device=self.device, dtype=torch.float32)
                root_pos_z[:] = self._sit_obj_lib._every_env_object_on_ground_trans[ids] # place the object on the ground

                axis = torch.tensor([[0.0, 0.0, 1.0]], device=self.device).reshape(1, 3).expand([ids.shape[0], -1])
                ang = torch.rand((len(ids),), device=self.device) * 2 * np.pi
                root_rot = quat_from_angle_axis(ang, axis)
                root_pos = torch.cat([root_pos_xy, root_pos_z.unsqueeze(-1)], dim=-1)

                self._sit_object_states[ids, 0:3] = root_pos
                self._sit_object_states[ids, 3:7] = root_rot
                self._sit_object_states[ids, 7:10] = 0.0
                self._sit_object_states[ids, 10:13] = 0.0

        target_sit_locations = self._sit_obj_lib._every_env_object_tar_sit_pos[env_ids]

        # transform from object local space to world space
        translation_to_world = self._sit_object_states[env_ids, 0:3]
        rotation_to_world = self._sit_object_states[env_ids, 3:7]
        target_sit_locations = quat_rotate(rotation_to_world, target_sit_locations) + translation_to_world

        self._sit_tar_pos[env_ids] = target_sit_locations

        return
    
    def _reset_task_carry(self, env_ids):

        not_task_env_ids = env_ids[self._task_indicator[env_ids] != HumanoidTrajSitCarryClimb.TaskUID["carry"].value]
        if len(not_task_env_ids) > 0:
            self._box_states[not_task_env_ids, :] = 0.0
            self._box_states[not_task_env_ids, 2] = 5.0 # place above the ground
            self._box_states[not_task_env_ids, 6] = 1.0 # quat

            self._platform_pos[not_task_env_ids, :] = 0.0
            self._platform_pos[not_task_env_ids, -1] = 5.0 - self._box_lib._box_size[not_task_env_ids, 2] / 2 - self._platform_height / 2 - 0.05
            self._tar_platform_pos[not_task_env_ids, :] = 0.0
            self._tar_platform_pos[not_task_env_ids, -1] = 5.0 - self._box_lib._box_size[not_task_env_ids, 2] / 2 - self._platform_height / 2 - 0.05 - 1.0

            self._box_tar_pos[not_task_env_ids, 0:3] = self._box_states[not_task_env_ids, 0:3]

        is_task_env_ids = env_ids[self._task_indicator[env_ids] == HumanoidTrajSitCarryClimb.TaskUID["carry"].value]
        if len(is_task_env_ids) > 0:
            local_reset_ref_env_ids = self._reset_ref_env_ids["carry"]
            local_reset_ref_motion_ids = self._reset_ref_motion_ids["carry"]
            local_reset_ref_motion_times = self._reset_ref_motion_times["carry"]

            # for skill is pickUp, carryWith, putDown, the initial location of the box is from the reference box motion
            for sk_name in ["pickUp", "carryWith", "putDown"]:
                if local_reset_ref_env_ids.get(sk_name) is not None:
                    if (len(local_reset_ref_env_ids[sk_name]) > 0):

                        curr_env_ids = local_reset_ref_env_ids[sk_name]

                        root_pos, root_rot = self._motion_lib[sk_name].get_obj_motion_state(
                            motion_ids=local_reset_ref_motion_ids[sk_name], 
                            motion_times=local_reset_ref_motion_times[sk_name]
                        )

                        on_ground_mask = (self._box_lib._box_size[curr_env_ids, 2] / 2 > root_pos[:, 2])
                        root_pos[on_ground_mask, 2] = self._box_lib._box_size[curr_env_ids[on_ground_mask], 2] / 2

                        self._box_states[curr_env_ids, 0:3] = root_pos
                        self._box_states[curr_env_ids, 3:7] = root_rot
                        self._box_states[curr_env_ids, 7:10] = 0.0
                        self._box_states[curr_env_ids, 10:13] = 0.0

                        # reset platform, we needn't platforms right now.
                        if self._carry_reset_random_height:
                            self._platform_pos[curr_env_ids] = self._platform_default_pos[curr_env_ids]

            # for skill is loco , we random generate an inital location of the box
            random_env_ids = []
            for sk_name in ["loco_carry"]:
                if local_reset_ref_env_ids.get(sk_name) is not None:
                    random_env_ids.append(local_reset_ref_env_ids[sk_name])

            if len(random_env_ids) > 0:
                ids = torch.cat(random_env_ids, dim=0)

                root_pos_xy = torch.randn(len(ids), 2, device=self.device)
                root_pos_xy /= torch.linalg.norm(root_pos_xy, dim=-1, keepdim=True)
                root_pos_xy *= torch.rand(len(ids), 1, device=self.device) * 9.0 + 1.0 # randomize
                root_pos_xy += self._humanoid_root_states[ids, :2] # get absolute pos, humanoid_root_state will be updated after set_env_state

                root_pos_z = self._box_lib._box_size[ids, 2] / 2 # place the box on the ground
                if self._carry_reset_random_height:

                    num_envs = ids.shape[0]
                    probs = to_torch(np.array([self._carry_reset_random_height_prob] * num_envs), device=self.device)
                    mask = torch.bernoulli(probs) == 1.0
                    
                    if mask.sum() > 0:
                        root_pos_z[mask] += torch.rand(mask.sum(), device=self.device) * 1.0 + self._carry_reset_minBottomSurfaceHeight
                        root_pos_z[mask] = self._regulate_height(root_pos_z[mask], self._box_lib._box_size[ids[mask]])

                axis = torch.tensor([[0.0, 0.0, 1.0]], device=self.device).reshape(1, 3).expand([ids.shape[0], -1])
                if self._carry_reset_random_rot:
                    coeff = 1.0
                else:
                    coeff = 0.0
                ang = torch.rand((len(ids),), device=self.device) * 2 * np.pi * coeff
                root_rot = quat_from_angle_axis(ang, axis)
                root_pos = torch.cat([root_pos_xy, root_pos_z.unsqueeze(-1)], dim=-1)

                self._box_states[ids, 0:3] = root_pos
                self._box_states[ids, 3:7] = root_rot
                self._box_states[ids, 7:10] = 0.0
                self._box_states[ids, 10:13] = 0.0

                # we need to reset this here
                if self._carry_reset_random_height:
                    self._platform_pos[ids, 0:2] = root_pos[:, 0:2] # xy
                    self._platform_pos[ids, -1] = root_pos[:, -1] - self._box_lib._box_size[ids, 2] / 2 - self._platform_height / 2

                    self._box_states[ids, 2] += 0.05 # add 0.05 to enable right collision detection

            # for skill is putDown, the target location of the box is from the reference box motion
            for sk_name in ["putDown"]:
                if local_reset_ref_env_ids.get(sk_name) is not None:
                    if (len(local_reset_ref_env_ids[sk_name]) > 0):

                        curr_env_ids = local_reset_ref_env_ids[sk_name]

                        root_pos, root_rot = self._motion_lib[sk_name].get_obj_motion_state(
                            motion_ids=local_reset_ref_motion_ids[sk_name], 
                            motion_times=self._motion_lib[sk_name].get_motion_length(local_reset_ref_motion_ids[sk_name]) # use last frame
                        )

                        root_pos[:, 2] = self._box_lib._box_size[curr_env_ids, 2] / 2 # make tar pos 100% on the ground
                        self._box_tar_pos[curr_env_ids] = root_pos

                        # reset tar platform
                        if self._carry_reset_random_height:
                            self._tar_platform_pos[curr_env_ids] = self._tar_platform_default_pos[curr_env_ids]

            # for skill is loco, pickUp, carryWith, we random generate an target location of the box
            random_env_ids = []

            for sk_name in ["loco_carry", "pickUp", "carryWith"]:
                if local_reset_ref_env_ids.get(sk_name) is not None:
                    random_env_ids.append(local_reset_ref_env_ids[sk_name])

            if len(random_env_ids) > 0:
                ids = torch.cat(random_env_ids, dim=0)

                new_target_pos = self._box_tar_pos_dist.sample((len(ids),))
                new_target_pos[:, 2] = self._box_lib._box_size[ids, 2] / 2 # place the box on the ground

                min_dist = 1.0

                # check if the new pos is too close to character or box
                target_overlap = torch.logical_or(
                    torch.sum((new_target_pos[..., :2] - self._humanoid_root_states[ids, :2]) ** 2, dim=-1) < min_dist,
                    torch.sum((new_target_pos[..., :2] - self._box_states[ids, :2]) ** 2, dim=-1) < min_dist
                )
                while(torch.sum(target_overlap) > 0):
                    new_target_pos[target_overlap] = self._box_tar_pos_dist.sample((torch.sum(target_overlap),))
                    new_target_pos[:, 2] = self._box_lib._box_size[ids, 2] / 2 # place the box on the ground
                    target_overlap = torch.logical_or(
                        torch.sum((new_target_pos[..., :2] - self._humanoid_root_states[ids, :2]) ** 2, dim=-1) < min_dist,
                        torch.sum((new_target_pos[..., :2] - self._box_states[ids, :2]) ** 2, dim=-1) < min_dist
                    )

                if self._carry_reset_random_height:
                    num_envs = ids.shape[0]
                    probs = to_torch(np.array([self._carry_reset_random_height_prob] * num_envs), device=self.device)
                    mask = torch.bernoulli(probs) == 1.0
                    
                    if mask.sum() > 0:
                        new_target_pos[mask, 2] += torch.rand(mask.sum(), device=self.device) * 1.0 + self._carry_reset_minBottomSurfaceHeight
                        new_target_pos[mask, 2] = self._regulate_height(new_target_pos[mask, 2], self._box_lib._box_size[ids[mask]])

                self._box_tar_pos[ids] = new_target_pos

                # we need to reset this here
                if self._carry_reset_random_height:
                    self._tar_platform_pos[ids, 0:2] = new_target_pos[:, 0:2] # xy
                    self._tar_platform_pos[ids, -1] = new_target_pos[:, -1] - self._box_lib._box_size[ids, 2] / 2 - self._platform_height / 2

        return
    
    def _regulate_height(self, h, box_size):
        top_surface_z = h + box_size[:, 2] / 2
        top_surface_z = torch.clamp_max(top_surface_z, self._carry_reset_maxTopSurfaceHeight)
        return top_surface_z - box_size[:, 2] / 2
    
    def _reset_task_climb(self, env_ids):

        not_task_env_ids = env_ids[self._task_indicator[env_ids] != HumanoidTrajSitCarryClimb.TaskUID["climb"].value]
        if len(not_task_env_ids) > 0:
            self._climb_object_states[not_task_env_ids, :] = 0.0
            self._climb_object_states[not_task_env_ids, 2] = -6.0 # place under the ground
            self._climb_object_states[not_task_env_ids, 6] = 1.0 # quat

        is_task_env_ids = env_ids[self._task_indicator[env_ids] == HumanoidTrajSitCarryClimb.TaskUID["climb"].value]
        if len(is_task_env_ids) > 0:
            local_reset_ref_env_ids = self._reset_ref_env_ids["climb"]
            local_reset_ref_motion_ids = self._reset_ref_motion_ids["climb"]
            local_reset_ref_motion_times = self._reset_ref_motion_times["climb"]

            # for skill is climb, the initial location of the object is from the reference object motion
            for sk_name in ["climb"]:
                if local_reset_ref_env_ids.get(sk_name) is not None:
                    if (len(local_reset_ref_env_ids[sk_name]) > 0):
                        object_root_pos, object_root_rot = self._motion_lib[sk_name].get_obj_motion_state_single_frame(local_reset_ref_motion_ids[sk_name])
                        self._climb_object_states[local_reset_ref_env_ids[sk_name], 0:2] = object_root_pos[..., 0:2]
                        self._climb_object_states[local_reset_ref_env_ids[sk_name], 2] = self._climb_obj_lib._every_env_object_on_ground_trans[local_reset_ref_env_ids[sk_name]]
                        self._climb_object_states[local_reset_ref_env_ids[sk_name], 3:7] = object_root_rot
                        self._climb_object_states[local_reset_ref_env_ids[sk_name], 7:13] = 0.0

            # for skill is loco and reset default, we random generate an inital location of the object
            random_env_ids = []
            for sk_name in ["loco_climb"]:
                if local_reset_ref_env_ids.get(sk_name) is not None:
                    random_env_ids.append(local_reset_ref_env_ids[sk_name])

            if len(random_env_ids) > 0:
                ids = torch.cat(random_env_ids, dim=0)

                root_pos_xy = torch.randn(len(ids), 2, device=self.device)
                root_pos_xy /= torch.linalg.norm(root_pos_xy, dim=-1, keepdim=True)
                root_pos_xy *= torch.rand(len(ids), 1, device=self.device) * 9.0 + 1.0 # randomize
                root_pos_xy += self._humanoid_root_states[ids, :2] # get absolute pos, humanoid_root_state will be updated after set_env_state

                root_pos_z = torch.zeros((len(ids)), device=self.device, dtype=torch.float32)
                root_pos_z[:] = self._climb_obj_lib._every_env_object_on_ground_trans[ids] # place the object on the ground

                axis = torch.tensor([[0.0, 0.0, 1.0]], device=self.device).reshape(1, 3).expand([ids.shape[0], -1])
                ang = torch.rand((len(ids),), device=self.device) * 2 * np.pi
                root_rot = quat_from_angle_axis(ang, axis)
                root_pos = torch.cat([root_pos_xy, root_pos_z.unsqueeze(-1)], dim=-1)

                self._climb_object_states[ids, 0:3] = root_pos
                self._climb_object_states[ids, 3:7] = root_rot
                self._climb_object_states[ids, 7:10] = 0.0
                self._climb_object_states[ids, 10:13] = 0.0

        self._climb_tar_pos[env_ids] = self._climb_object_states[env_ids, 0:3]
        self._climb_tar_pos[env_ids, -1] += self._climb_obj_lib._every_env_object_tar_sit_pos[env_ids, -1] + self._char_h

        return

    def _reset_task_indicator(self, env_ids):
        n = len(env_ids)

        self._task_indicator[env_ids] = torch.multinomial(self._task_init_prob, num_samples=n, replacement=True)
        
        for task_name in self._multiple_task_names:
            curr_task_uid = HumanoidTrajSitCarryClimb.TaskUID[task_name].value
            curr_env_ids = env_ids[self._task_indicator[env_ids] == curr_task_uid]

            if len(curr_env_ids) > 0:
                self._task_mask[curr_env_ids, :] = False
                self._task_mask[curr_env_ids, curr_task_uid] = True

                self._reset_ref_env_ids[task_name] = {}
                self._reset_ref_motion_ids[task_name] = {}
                self._reset_ref_motion_times[task_name] = {}

                if task_name == "traj":
                    skill = self._traj_skill
                    skill_init_prob = self._traj_skill_init_prob
                elif task_name == "sit":
                    skill = self._sit_skill
                    skill_init_prob = self._sit_skill_init_prob
                elif task_name == "carry":
                    skill = self._carry_skill
                    skill_init_prob = self._carry_skill_init_prob
                elif task_name == "climb":
                    skill = self._climb_skill
                    skill_init_prob = self._climb_skill_init_prob
                else:
                    raise NotImplementedError

                sk_ids = torch.multinomial(skill_init_prob, num_samples=curr_env_ids.shape[0], replacement=True)
                for uid, sk_name in enumerate(skill):
                    skilled_curr_env_ids = curr_env_ids[(sk_ids == uid).nonzero().squeeze(-1)] # be careful!!!
                    if len(skilled_curr_env_ids) > 0:
                        self._reset_ref_env_ids[task_name][sk_name] = skilled_curr_env_ids

                        num_envs = skilled_curr_env_ids.shape[0]
                        motion_ids = self._motion_lib[sk_name].sample_motions(num_envs)
                        motion_times = self._motion_lib[sk_name].sample_time_rsi(motion_ids) # avoid times with serious self-penetration
                        
                        self._reset_ref_motion_ids[task_name][sk_name] = motion_ids
                        self._reset_ref_motion_times[task_name][sk_name] = motion_times

        assert self._task_mask[env_ids].sum() == len(env_ids)

        return
    
    def _reset_envs(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = {}
        self._reset_ref_motion_ids = {}
        self._reset_ref_motion_times = {}
        if (len(env_ids) > 0):
            self._reset_task_indicator(env_ids)
            self._reset_ref_state_init(env_ids)
            self._reset_task_traj(env_ids)
            self._reset_task_sit(env_ids)
            self._reset_task_carry(env_ids)
            self._reset_task_climb(env_ids)
            self._reset_env_tensors(env_ids)
            self._refresh_sim_tensors()
            self._compute_observations(env_ids)
            self._init_amp_obs(env_ids)

        return
    
    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)
        
        if self._enable_IET:
            self._IET_step_buf[env_ids] = 0
            self._IET_triggered_buf[env_ids] = 0
        
        if self._is_eval:
            self._success_buf[env_ids] = 0
            self._precision_buf[env_ids] = float('Inf')

            traj_env_mask = self._task_indicator[env_ids] == HumanoidTrajSitCarryClimb.TaskUID["traj"].value
            self._precision_buf[env_ids[traj_env_mask]] = 0 # not Inf

        env_ids_int32 = torch.cat([self._sit_object_actor_ids[env_ids], self._box_actor_ids[env_ids], self._climb_object_actor_ids[env_ids]], dim=0)
        if self._carry_reset_random_height:
            # env has two platforms
            env_ids_int32 = torch.cat([env_ids_int32, self._platform_actor_ids[env_ids], self._tar_platform_actor_ids[env_ids]], dim=0)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        return

    def _reset_actors(self, env_ids):
        if (self._state_init == HumanoidTrajSitCarryClimb.StateInit.Default):
            self._reset_default(env_ids)
        elif (self._state_init == HumanoidTrajSitCarryClimb.StateInit.Start
              or self._state_init == HumanoidTrajSitCarryClimb.StateInit.Random):
            self._reset_ref_state_init(env_ids)
        elif (self._state_init == HumanoidTrajSitCarryClimb.StateInit.Hybrid):
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

        for sk_name in self._skill:
            curr_motion_lib = self._motion_lib[sk_name]

            curr_env_ids = torch.tensor([], dtype=torch.int32, device=self.device)
            for k, v in self._reset_ref_env_ids.items():
                for kk, vv in v.items():
                    if kk == sk_name:
                        curr_env_ids = torch.hstack([curr_env_ids, vv])
            
            curr_motion_ids = torch.tensor([], dtype=torch.int32, device=self.device)
            for k, v in self._reset_ref_motion_ids.items():
                for kk, vv in v.items():
                    if kk == sk_name:
                        curr_motion_ids = torch.hstack([curr_motion_ids, vv])
            
            curr_motion_times = torch.tensor([], dtype=torch.float32, device=self.device)
            for k, v in self._reset_ref_motion_times.items():
                for kk, vv in v.items():
                    if kk == sk_name:
                        curr_motion_times = torch.hstack([curr_motion_times, vv])

            if len(curr_env_ids) > 0:
                
                root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
                    = curr_motion_lib.get_motion_state(curr_motion_ids, curr_motion_times)

                self._set_env_state(env_ids=curr_env_ids, 
                                    root_pos=root_pos, 
                                    root_rot=root_rot, 
                                    dof_pos=dof_pos, 
                                    root_vel=root_vel, 
                                    root_ang_vel=root_ang_vel, 
                                    dof_vel=dof_vel)

                # update buffer for kinematic humanoid state
                body_pos, body_rot, body_vel, body_ang_vel \
                    = curr_motion_lib.get_motion_state_max(curr_motion_ids, curr_motion_times)
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

            curr_env_ids = torch.tensor([], dtype=torch.int32, device=self.device)
            for k, v in self._reset_ref_env_ids.items():
                for kk, vv in v.items():
                    if kk == sk_name:
                        curr_env_ids = torch.hstack([curr_env_ids, vv])
            
            curr_motion_ids = torch.tensor([], dtype=torch.int32, device=self.device)
            for k, v in self._reset_ref_motion_ids.items():
                for kk, vv in v.items():
                    if kk == sk_name:
                        curr_motion_ids = torch.hstack([curr_motion_ids, vv])
            
            curr_motion_times = torch.tensor([], dtype=torch.float32, device=self.device)
            for k, v in self._reset_ref_motion_times.items():
                for kk, vv in v.items():
                    if kk == sk_name:
                        curr_motion_times = torch.hstack([curr_motion_times, vv])

            if (len(curr_env_ids) > 0):
                self._init_amp_obs_ref(curr_env_ids, curr_motion_ids, curr_motion_times, sk_name)
        
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
        

        if self._enable_task_specific_disc and self._enable_task_mask_obs:
            motion_labels = self._task_mask[env_ids]
            motion_labels = torch.broadcast_to(motion_labels.unsqueeze(-2), [motion_labels.shape[0], self._num_amp_obs_steps - 1, motion_labels.shape[1]])

            amp_obs_demo = torch.cat([amp_obs_demo, motion_labels.reshape(-1, motion_labels.shape[-1]).float()], dim=-1)

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
            amp_obs = build_amp_observations(self._rigid_body_pos[:, 0, :],
                                                               self._rigid_body_rot[:, 0, :],
                                                               self._rigid_body_vel[:, 0, :],
                                                               self._rigid_body_ang_vel[:, 0, :],
                                                               self._dof_pos, self._dof_vel, key_body_pos,
                                                               self._local_root_obs, self._root_height_obs, 
                                                               self._dof_obs_size, self._dof_offsets)
            
            if self._enable_task_specific_disc:
                self._curr_amp_obs_buf[:] = torch.cat([amp_obs, self._task_mask.float()], dim=-1)
            else:
                self._curr_amp_obs_buf[:] = amp_obs

        else:
            kinematic_rigid_body_pos = self._kinematic_humanoid_rigid_body_states[:, :, 0:3]
            key_body_pos = kinematic_rigid_body_pos[:, self._key_body_ids, :]
            amp_obs = build_amp_observations(self._kinematic_humanoid_rigid_body_states[env_ids, 0, 0:3],
                                                                   self._kinematic_humanoid_rigid_body_states[env_ids, 0, 3:7],
                                                                   self._kinematic_humanoid_rigid_body_states[env_ids, 0, 7:10],
                                                                   self._kinematic_humanoid_rigid_body_states[env_ids, 0, 10:13],
                                                                   self._dof_pos[env_ids], self._dof_vel[env_ids], key_body_pos[env_ids],
                                                                   self._local_root_obs, self._root_height_obs, 
                                                                   self._dof_obs_size, self._dof_offsets)
            
            if self._enable_task_specific_disc:
                self._curr_amp_obs_buf[env_ids] = torch.cat([amp_obs, self._task_mask[env_ids].float()], dim=-1)
            else:
                self._curr_amp_obs_buf[env_ids] = amp_obs

        return
    
    def _compute_reset(self):
        time = self.progress_buf * self.dt
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        tar_pos = self._traj_gen.calc_pos(env_ids, time)

        traj_env_mask = self._task_indicator == HumanoidTrajSitCarryClimb.TaskUID["traj"].value
        carry_env_mask = self._task_indicator == HumanoidTrajSitCarryClimb.TaskUID["carry"].value
        
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                           self._contact_forces, self._contact_body_ids,
                                                           self._rigid_body_pos, tar_pos,
                                                           self.max_episode_length, self._fail_dist, traj_env_mask, 
                                                           carry_env_mask, self.max_episode_length_short,
                                                           self._enable_IET, self._max_IET_steps, self._IET_step_buf,
                                                           self._enable_early_termination, self._termination_heights)
        return


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_finish_state(root_pos, tar_pos, success_threshold):
    # type: (Tensor, Tensor, float) -> Tensor
    pos_diff = tar_pos - root_pos
    pos_err = torch.norm(pos_diff, p=2, dim=-1)
    dist_mask = pos_err <= success_threshold
    return dist_mask

@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                           tar_pos, max_episode_length, fail_dist, traj_env_mask, 
                           carry_env_mask, max_episode_length_short,
                           enable_IET, max_IET_steps, IET_step_buf,
                           enable_early_termination, termination_heights):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, Tensor, Tensor, float, bool, int, Tensor, bool, Tensor) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)
    IET_triggered = torch.zeros_like(reset_buf)

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

        tar_fail[~traj_env_mask] = False

        has_failed = torch.logical_or(has_fallen, tar_fail)

        terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)

    max_episode_length_tensor = torch.zeros_like(reset_buf)
    max_episode_length_tensor[:] = max_episode_length - 1

    if (~carry_env_mask).sum() > 0:
        max_episode_length_tensor[~carry_env_mask] = max_episode_length_short - 1

    if enable_IET:
        IET_triggered = torch.where(IET_step_buf >= max_IET_steps - 1, torch.ones_like(reset_buf), IET_triggered)
        reset = torch.logical_or(IET_triggered, terminated)
        reset = torch.where(progress_buf >= max_episode_length_tensor, torch.ones_like(reset_buf), reset)
    else:
        reset = torch.where(progress_buf >= max_episode_length_tensor, torch.ones_like(reset_buf), terminated)

    return reset, terminated

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
def compute_location_observations(root_states, traj_samples, sit_tar_pos, sit_object_states, sit_object_bps, sit_object_facings, box_states, box_bps, box_tar_pos, climb_object_states, climb_object_bps, climb_tar_pos, task_mask, each_subtask_obs_mask, enable_apply_mask):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    B = root_pos.shape[0]
    NM = traj_samples.shape[1]

    heading_rot_exp = torch.broadcast_to(heading_rot.unsqueeze(1), (B, NM, 4))
    root_pos_exp = torch.broadcast_to(root_pos.unsqueeze(1), (B, NM, 3))

    # traj
    N = traj_samples.shape[1]
    local_traj_sampls = quat_rotate(heading_rot_exp[:, :N].reshape(-1, 4), traj_samples.reshape(-1, 3) - root_pos_exp[:, :N].reshape(-1, 3))[..., 0:2]

    # sit
    local_sit_tar_pos = quat_rotate(heading_rot, sit_tar_pos - root_pos) # 3d xyz

    sit_obj_root_pos = sit_object_states[:, 0:3]
    sit_obj_root_rot = sit_object_states[:, 3:7]

    local_sit_obj_root_pos = quat_rotate(heading_rot, sit_obj_root_pos - root_pos)
    local_sit_obj_root_rot = quat_mul(heading_rot, sit_obj_root_rot)
    local_sit_obj_root_rot = torch_utils.quat_to_tan_norm(local_sit_obj_root_rot)

    N = sit_object_bps.shape[1]
    sit_obj_root_pos_exp = torch.broadcast_to(sit_obj_root_pos.unsqueeze(1), (B, N, 3)).reshape(-1, 3) # [4096, 3] >> [4096, 8, 3] >> [4096*8, 3]
    sit_obj_root_rot_exp = torch.broadcast_to(sit_obj_root_rot.unsqueeze(1), (B, N, 4)).reshape(-1, 4) # [4096, 4] >> [4096, 8, 4] >> [4096*8, 4]

    sit_obj_bps_world_space = quat_rotate(sit_obj_root_rot_exp, sit_object_bps.reshape(-1, 3)) + sit_obj_root_pos_exp
    sit_obj_bps_local_space = quat_rotate(heading_rot_exp[:, :N].reshape(-1, 4), sit_obj_bps_world_space - root_pos_exp[:, :N].reshape(-1, 3)).reshape(-1, N * 3)

    sit_face_vec_world_space = quat_rotate(sit_obj_root_rot, sit_object_facings)
    sit_face_vec_local_space = quat_rotate(heading_rot, sit_face_vec_world_space)[..., 0:2]

    # carry
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
    box_bps_local_space = quat_rotate(heading_rot_exp[:, :N].reshape(-1, 4), box_bps_world_space - root_pos_exp[:, :N].reshape(-1, 3)) # (num_envs*8, 3)

    # task obs
    local_box_tar_pos = quat_rotate(heading_rot, box_tar_pos - root_pos) # 3d xyz

    # climb
    local_climb_tar_pos = quat_rotate(heading_rot, climb_tar_pos - root_pos) # 3d xyz

    climb_obj_root_pos = climb_object_states[:, 0:3]
    climb_obj_root_rot = climb_object_states[:, 3:7]

    N = climb_object_bps.shape[1]
    climb_obj_root_pos_exp = torch.broadcast_to(climb_obj_root_pos.unsqueeze(1), (B, N, 3)).reshape(-1, 3) # [4096, 3] >> [4096, 8, 3] >> [4096*8, 3]
    climb_obj_root_rot_exp = torch.broadcast_to(climb_obj_root_rot.unsqueeze(1), (B, N, 4)).reshape(-1, 4) # [4096, 4] >> [4096, 8, 4] >> [4096*8, 4]

    climb_obj_bps_world_space = quat_rotate(climb_obj_root_rot_exp, climb_object_bps.reshape(-1, 3)) + climb_obj_root_pos_exp
    climb_obj_bps_local_space = quat_rotate(heading_rot_exp[:, :N].reshape(-1, 4), climb_obj_bps_world_space - root_pos_exp[:, :N].reshape(-1, 3)).reshape(-1, N * 3)

    obs = torch.cat([
        local_traj_sampls.reshape(root_pos.shape[0], -1),
        local_sit_tar_pos, sit_obj_bps_local_space, sit_face_vec_local_space, local_sit_obj_root_pos, local_sit_obj_root_rot, 
        local_box_vel, local_box_ang_vel, local_box_pos, local_box_rot_obs, box_bps_local_space.reshape(root_pos.shape[0], -1), local_box_tar_pos,
        local_climb_tar_pos, climb_obj_bps_local_space,
        ], dim=-1)

    if enable_apply_mask:
        mask = task_mask[:, None, :].float() @ torch.broadcast_to(each_subtask_obs_mask[None, :, :].float(), (root_states.shape[0], each_subtask_obs_mask.shape[0], each_subtask_obs_mask.shape[1]))
        obs *= mask.squeeze(1)
    
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
def compute_sit_reward(root_pos, prev_root_pos, root_rot, prev_root_rot, 
                       object_root_pos, tar_pos, tar_speed, dt,
                       sit_vel_penalty, sit_vel_pen_coeff, sit_vel_penalty_thre, sit_ang_vel_pen_coeff, sit_ang_vel_penalty_thre,):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, bool, float, float, float, float,) -> Tensor

    d_obj2human_xy = torch.sum((object_root_pos[..., 0:2] - root_pos[..., 0:2]) ** 2, dim=-1)
    reward_far_pos = torch.exp(-0.5 * d_obj2human_xy)

    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir = object_root_pos[..., 0:2] - root_pos[..., 0:2] # d* is a horizontal unit vector pointing from the root to the object's location
    tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)
    tar_vel_err = tar_speed - tar_dir_speed
    reward_far_vel = torch.exp(-2.0 * tar_vel_err * tar_vel_err)

    reward_far_final = 0.0 * reward_far_pos + 1.0 * reward_far_vel
    dist_mask = (d_obj2human_xy <= 0.5 ** 2)
    reward_far_final[dist_mask] = 1.0

    # when humanoid is close to the object
    reward_near = torch.exp(-10.0 * torch.sum((tar_pos - root_pos) ** 2, dim=-1))

    reward = 0.7 * reward_near + 0.3 * reward_far_final

    if sit_vel_penalty:
        min_speed_penalty = sit_vel_penalty_thre
        root_vel_norm = torch.norm(root_vel, p=2, dim=-1)
        root_vel_norm = torch.clamp_min(root_vel_norm, min_speed_penalty)
        root_vel_err = min_speed_penalty - root_vel_norm
        root_vel_penalty = -1 * sit_vel_pen_coeff * (1 - torch.exp(-2.0 * (root_vel_err * root_vel_err)))
        dist_mask = (d_obj2human_xy <= 1.5 ** 2)
        root_vel_penalty[~dist_mask] = 0.0
        reward += root_vel_penalty

        root_z_ang_vel = torch.abs((get_euler_xyz(root_rot)[2] - get_euler_xyz(prev_root_rot)[2]) / dt)
        root_z_ang_vel = torch.clamp_min(root_z_ang_vel, sit_ang_vel_penalty_thre)
        root_z_ang_vel_err = sit_ang_vel_penalty_thre - root_z_ang_vel
        root_z_ang_vel_penalty = -1 * sit_ang_vel_pen_coeff * (1 - torch.exp(-0.5 * (root_z_ang_vel_err ** 2)))
        root_z_ang_vel_penalty[~dist_mask] = 0.0
        reward += root_z_ang_vel_penalty

    return reward

@torch.jit.script
def compute_climb_reward(root_pos, prev_root_pos, object_pos, dt, tar_pos, rigid_body_pos, feet_ids, char_h,
                         valid_radius,
                         climb_vel_penalty, climb_vel_pen_coeff, climb_vel_penalty_thre):
    # type: (Tensor, Tensor, Tensor, float, Tensor, Tensor, Tensor, float, Tensor, bool, float, float) -> Tensor

    pos_diff = object_pos[..., 0:2] - root_pos[..., 0:2]
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    pos_reward = torch.exp(-0.5 * pos_err)

    min_speed = 1.5

    tar_dir = object_pos[..., 0:2] - root_pos[..., 0:2]
    tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)

    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)
    tar_vel_err = min_speed - tar_dir_speed
    # tar_vel_err = torch.clamp_min(tar_vel_err, 0.0) # constrain vel around the peak value
    vel_reward = torch.exp(-2.0 * (tar_vel_err * tar_vel_err))
    speed_mask = tar_dir_speed <= 0
    vel_reward[speed_mask] = 0

    dist_mask = (pos_err <= valid_radius ** 2)
    pos_reward[dist_mask] = 1.0
    vel_reward[dist_mask] = 1.0

    pos_reward_near = torch.exp(-10 * torch.sum((tar_pos - root_pos) ** 2, dim=-1))

    feet_height_err = (rigid_body_pos[:, feet_ids, -1].mean(dim=1) - (tar_pos[..., 2] - char_h)) ** 2 # height
    feet_height_reward = torch.exp(-50.0 * feet_height_err)
    feet_height_reward[~dist_mask] = 0.0

    reward = 0.0 * pos_reward + 0.2 * vel_reward + 0.5 * pos_reward_near + 0.3 * feet_height_reward

    if climb_vel_penalty:
        thre_tensor = torch.ones_like(valid_radius)
        thre_tensor[pos_err <= 1.5 ** 2] = 1.5
        thre_tensor[pos_err <= valid_radius ** 2] = climb_vel_penalty_thre
        min_speed_penalty = thre_tensor
        root_vel_norm = torch.norm(root_vel, p=2, dim=-1) # torch.abs(root_vel[..., -1]) # only consider Z
        root_vel_norm = torch.clamp_min(root_vel_norm, min_speed_penalty)
        root_vel_err = min_speed_penalty - root_vel_norm
        root_vel_penalty = -1 * climb_vel_pen_coeff * (1 - torch.exp(-2.0 * (root_vel_err * root_vel_err)))
        dist_mask = (pos_err <= 1.5 ** 2)
        root_vel_penalty[~dist_mask] = 0.0
        reward += root_vel_penalty

    return reward

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
def compute_walk_reward(root_pos, prev_root_pos, box_pos, dt, tar_vel, only_vel_reward):
    # type: (Tensor, Tensor, Tensor, float, float, bool) -> Tensor

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
def compute_carry_reward(box_pos, prev_box_pos, tar_box_pos, dt, tar_vel, box_size, only_vel_reward, box_vel_penalty, box_vel_pen_coeff, box_vel_penalty_thre):
    # type: (Tensor, Tensor, Tensor, float, float, Tensor, bool, bool, float, float) -> Tensor
    
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
    tar_vel_err = min_speed - tar_dir_speed
    # tar_vel_err = torch.clamp_min(tar_vel_err, 0.0) # constrain vel around the peak value
    vel_reward = torch.exp(-5.0 * (tar_vel_err * tar_vel_err))
    speed_mask = tar_dir_speed <= 0
    vel_reward[speed_mask] = 0

    height_mask = box_pos[..., 2] <= (box_size[..., 2] / 2 + 0.2) # avoid learning to kick the box
    pos_reward_far[height_mask] = 0.0
    vel_reward[height_mask] = 0.0

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
