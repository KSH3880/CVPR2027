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

import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from utils import torch_utils

from env.tasks.base_task import BaseTask

class Humanoid(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine

        self.num_agents = self.cfg["env"].get("numAgents", 1)
        self._pd_control = self.cfg["env"]["pdControl"]
        self.power_scale = self.cfg["env"]["powerScale"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self._local_root_obs = self.cfg["env"]["localRootObs"]
        self._local_root_obs_policy = self.cfg["env"]["localRootObsPolicy"]
        self._root_height_obs = self.cfg["env"].get("rootHeightObs", True)
        self._root_height_obs_policy = self.cfg["env"].get("rootHeightObsPolicy", True)
        self._enable_early_termination = self.cfg["env"]["enableEarlyTermination"]
        
        key_bodies = self.cfg["env"]["keyBodies"]
        self._setup_character_props(key_bodies)

        self.cfg["env"]["numObservations"] = self.get_obs_size()
        self.cfg["env"]["numActions"] = self.get_action_size()

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless
         
        super().__init__(cfg=self.cfg)
        
        self.dt = self.control_freq_inv * sim_params.dt
        
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        sensors_per_env = 2
        # two force sensors per humanoid, so the count scales with the agents
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(
            self.num_envs * self.num_agents, sensors_per_env * 6)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(
            self.num_envs * self.num_agents, self.num_dof)
        
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self._root_states = gymtorch.wrap_tensor(actor_root_state)
        num_actors = self.get_num_actors_per_env()
        A = self.num_agents

        # Humanoids occupy the first A actor slots of every env, so a humanoid
        # tensor is the leading slice of each env's actor block.
        #
        # That slice must stay a window into simulator memory: values written on
        # reset have to reach the sim, and refreshed values have to be readable.
        # Flattening it to (num_envs * A, ...) destroys that -- the offset of flat
        # index i = e*A + a is (i // A)*num_actors*D + (i % A)*D, which is not
        # linear in i, so no stride can express it and torch silently hands back a
        # copy. Keeping the env axis is what makes the slice representable.
        #
        # With A == 1 the agent axis is squeezed away and every shape below is
        # exactly what the single-agent code has always seen, which is what keeps
        # the A=1 regression a valid check. The agent axis only appears at A >= 2,
        # and the only task that runs there is multi_task.
        def _agents(t):
            return t.squeeze(1) if A == 1 else t

        self._humanoid_root_states = _agents(self._root_states.view(
            self.num_envs, num_actors, actor_root_state.shape[-1])[:, :A])
        self._initial_humanoid_root_states = self._humanoid_root_states.clone()
        self._initial_humanoid_root_states[..., 7:13] = 0

        # Two shapes of the same table. Subclasses derive object ids from the
        # flat one (`_humanoid_actor_ids + idx`), while reset indexes by env and
        # needs the agent axis to pick every agent of the chosen envs.
        actor_ids = (
            num_actors * torch.arange(self.num_envs, device=self.device, dtype=torch.int32)[:, None]
            + torch.arange(A, device=self.device, dtype=torch.int32)[None, :])
        self._humanoid_actor_ids_per_env = actor_ids
        self._humanoid_actor_ids = actor_ids.reshape(-1)

        # create some wrapper tensors for different slices
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        dofs_per_env = self._dof_state.shape[0] // self.num_envs
        _dof = _agents(self._dof_state.view(self.num_envs, dofs_per_env, 2)[:, :A * self.num_dof]
                       .view(self.num_envs, A, self.num_dof, 2))
        self._dof_pos = _dof[..., 0]
        self._dof_vel = _dof[..., 1]
        
        self._initial_dof_pos = torch.zeros_like(self._dof_pos, device=self.device, dtype=torch.float)
        self._initial_dof_vel = torch.zeros_like(self._dof_vel, device=self.device, dtype=torch.float)
        
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        rigid_body_state_reshaped = _agents(self._rigid_body_state.view(
            self.num_envs, bodies_per_env, 13)[:, :A * self.num_bodies].view(
            self.num_envs, A, self.num_bodies, 13))

        self._initial_humanoid_rigid_body_states = rigid_body_state_reshaped[..., :self.num_bodies, :].clone()
        self._initial_humanoid_rigid_body_states[..., 7:13] = 0

        self._rigid_body_pos = rigid_body_state_reshaped[..., :self.num_bodies, 0:3]
        self._rigid_body_rot = rigid_body_state_reshaped[..., :self.num_bodies, 3:7]
        self._rigid_body_vel = rigid_body_state_reshaped[..., :self.num_bodies, 7:10]
        self._rigid_body_ang_vel = rigid_body_state_reshaped[..., :self.num_bodies, 10:13]

        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._contact_forces = _agents(contact_force_tensor.view(
            self.num_envs, bodies_per_env, 3)[:, :A * self.num_bodies].view(
            self.num_envs, A, self.num_bodies, 3))
        
        self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        
        self._build_termination_heights()
        
        contact_bodies = self.cfg["env"]["contactBodies"]
        self._key_body_ids = self._build_key_body_ids_tensor(key_bodies)
        self._contact_body_ids = self._build_contact_body_ids_tensor(contact_bodies)
        
        if self.viewer != None:
            self._init_camera()
            self._camera_follow_flag = True
            
        return

    def agent_rows(self, env_ids):
        """env 인덱스 -> (env, agent) row 인덱스. A==1 이면 그대로."""
        A = self.num_agents
        if A == 1:
            return env_ids
        return (env_ids.long().unsqueeze(-1) * A
                + torch.arange(A, device=self.device, dtype=torch.long)).flatten()

    def all_rows(self):
        return torch.arange(self.num_envs * self.num_agents, device=self.device, dtype=torch.long)

    def agent_axis(self, t):
        """agent 축을 되살린 창. A==1 이면 크기 1 축을 끼운다 -- 여전히 view 다."""
        return t.unsqueeze(1) if self.num_agents == 1 else t

    def humanoid_rows(self, t):
        """(E, A, ...) 를 (E*A, ...) 로 편다. 읽기 전용 -- 쓰기는 창을 통해야 한다."""
        return t.flatten(0, 1) if self.num_agents > 1 else t

    def get_obs_size(self):
        return self._num_obs

    def get_action_size(self):
        return self._num_actions

    def get_num_actors_per_env(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        return num_actors

    def create_sim(self):
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'z')
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        return

    def reset(self, env_ids=None):
        if (env_ids is None):
            env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
        elif self.num_agents > 1:
            # 학습 루프는 row 당 done 을 받으므로 여기 오는 것도 row 인덱스다 (list 로 올 수도 있다).
            # 리셋은 env 단위이고 한 env 의 에이전트는 같이 끝나므로 접어서 중복을 없앤다.
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            env_ids = torch.unique(torch.div(env_ids, self.num_agents, rounding_mode="floor"))
        self._reset_envs(env_ids)
        return

    def set_char_color(self, col, env_ids):
        for env_id in env_ids:
            env_ptr = self.envs[env_id]
            handle = self.humanoid_handles[env_id]

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL,
                                              gymapi.Vec3(col[0], col[1], col[2]))

        return

    def _reset_envs(self, env_ids):
        if (len(env_ids) > 0):
            self._reset_actors(env_ids)
            self._reset_env_tensors(env_ids)
            self._refresh_sim_tensors()
            self._compute_observations(env_ids)
        return

    def _reset_env_tensors(self, env_ids):
        env_ids_int32 = self._humanoid_actor_ids_per_env[env_ids].flatten()
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        return

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)
        return

    def _setup_character_props(self, key_bodies):
        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        num_key_bodies = len(key_bodies)

        if (asset_file == "mjcf/amp_humanoid.xml"):
            self._dof_body_ids = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
            self._dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
            self._dof_obs_size = 72
            self._num_actions = 28
            self._num_actions_joint = self._num_actions
            self._num_obs = 1 + 15 * (3 + 6 + 3 + 3) - 3
        elif (asset_file == "mjcf/phys_humanoid.xml") or (asset_file == "mjcf/phys_humanoid_v2.xml") or (asset_file == "mjcf/phys_humanoid_v3.xml") or (asset_file == "mjcf/phys_humanoid_v3_box_foot.xml"):
            self._dof_body_ids = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
            self._dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 20, 23, 26, 29, 32]
            self._dof_obs_size = 72
            self._num_actions = 28 + 2 * 2
            self._num_actions_joint = self._num_actions
            self._num_obs = 1 + 15 * (3 + 6 + 3 + 3) - 3

        else:
            print("Unsupported character config file: {s}".format(asset_file))
            assert(False)

        return

    def _build_termination_heights(self):
        head_term_height = 0.3

        termination_height = self.cfg["env"]["terminationHeight"]
        self._termination_heights = np.array([termination_height] * self.num_bodies)

        head_id = self.gym.find_actor_rigid_body_handle(self.envs[0], self.humanoid_handles[0], "head")
        self._termination_heights[head_id] = max(head_term_height, self._termination_heights[head_id])
        self._termination_heights = to_torch(self._termination_heights, device=self.device)
        return

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = self.cfg["env"]["asset"]["assetRoot"]
        asset_file = self.cfg["env"]["asset"]["assetFileName"]

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        #asset_options.fix_base_link = True
        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]
        
        # create force sensors at the feet
        right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "right_foot")
        left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "left_foot")
        sensor_pose = gymapi.Transform()

        self.gym.create_asset_force_sensor(humanoid_asset, right_foot_idx, sensor_pose)
        self.gym.create_asset_force_sensor(humanoid_asset, left_foot_idx, sensor_pose)

        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = to_torch(motor_efforts, device=self.device)

        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_shapes = self.gym.get_asset_rigid_shape_count(humanoid_asset)
        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)

        self.humanoid_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []
        
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self._build_env(i, env_ptr, humanoid_asset)
            self.envs.append(env_ptr)

        dof_prop = self.gym.get_actor_dof_properties(self.envs[0], self.humanoid_handles[0])
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        if (self._pd_control):
            self._build_pd_action_offset_scale()

        return
    
    def _build_env(self, env_id, env_ptr, humanoid_asset):
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
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # One actor per agent, spread around a small circle so they do not spawn
        # inside one another. Resets place them properly; this only has to be
        # non-overlapping at creation time.
        colors = [gymapi.Vec3(0.54, 0.85, 0.2), gymapi.Vec3(0.28, 0.60, 0.93),
                  gymapi.Vec3(0.93, 0.65, 0.25), gymapi.Vec3(0.80, 0.40, 0.85)]
        for _a in range(self.num_agents):
            if self.num_agents == 1:
                off_x = off_y = 0.0
            else:
                ang = 2.0 * np.pi * _a / self.num_agents
                off_x, off_y = 1.5 * np.cos(ang), 1.5 * np.sin(ang)
            start_pose.p = gymapi.Vec3(off_x, off_y, self._char_h)

            humanoid_handle = self.gym.create_actor(
                env_ptr, humanoid_asset, start_pose, "humanoid", col_group, col_filter, segmentation_id)

            self.gym.enable_actor_dof_force_sensors(env_ptr, humanoid_handle)

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(env_ptr, humanoid_handle, j, gymapi.MESH_VISUAL,
                                              colors[_a % len(colors)])

            if (self._pd_control):
                dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
                dof_prop["driveMode"] = gymapi.DOF_MODE_POS
                self.gym.set_actor_dof_properties(env_ptr, humanoid_handle, dof_prop)

            self.humanoid_handles.append(humanoid_handle)

        return

    def _build_pd_action_offset_scale(self):
        num_joints = len(self._dof_offsets) - 1
        
        lim_low = self.dof_limits_lower.cpu().numpy()
        lim_high = self.dof_limits_upper.cpu().numpy()

        for j in range(num_joints):
            dof_offset = self._dof_offsets[j]
            dof_size = self._dof_offsets[j + 1] - self._dof_offsets[j]

            if (dof_size == 3):
                curr_low = lim_low[dof_offset:(dof_offset + dof_size)]
                curr_high = lim_high[dof_offset:(dof_offset + dof_size)]
                curr_low = np.max(np.abs(curr_low))
                curr_high = np.max(np.abs(curr_high))
                curr_scale = max([curr_low, curr_high])
                curr_scale = 1.2 * curr_scale
                curr_scale = min([curr_scale, np.pi])

                lim_low[dof_offset:(dof_offset + dof_size)] = -curr_scale
                lim_high[dof_offset:(dof_offset + dof_size)] = curr_scale
                
                #lim_low[dof_offset:(dof_offset + dof_size)] = -np.pi
                #lim_high[dof_offset:(dof_offset + dof_size)] = np.pi


            elif (dof_size == 1):
                curr_low = lim_low[dof_offset]
                curr_high = lim_high[dof_offset]
                curr_mid = 0.5 * (curr_high + curr_low)
                
                # extend the action range to be a bit beyond the joint limits so that the motors
                # don't lose their strength as they approach the joint limits
                curr_scale = 0.7 * (curr_high - curr_low)
                curr_low = curr_mid - curr_scale
                curr_high = curr_mid + curr_scale

                lim_low[dof_offset] = curr_low
                lim_high[dof_offset] =  curr_high

        self._pd_action_offset = 0.5 * (lim_high + lim_low)
        self._pd_action_scale = 0.5 * (lim_high - lim_low)
        self._pd_action_offset = to_torch(self._pd_action_offset, device=self.device)
        self._pd_action_scale = to_torch(self._pd_action_scale, device=self.device)

        return

    def _get_humanoid_collision_filter(self):
        """
        Setting the collision filter to 0 will enable collisions between all shapes in the actor. 
        Setting the collision filter to anything > 0 will disable all self collisions.
        """
        if self.cfg["env"]["enableSelfCollisionDetection"]:
            return 0
        else:
            return 1

    def _compute_reward(self, actions):
        self.rew_buf[:] = compute_humanoid_reward(self.obs_buf)
        return

    def _compute_reset(self):
        # Falling is a per-agent event -- it is read off that agent's own contact
        # forces and body heights -- but the reset it triggers is per-env. Run the
        # unchanged termination rule over agent rows, then fold it down.
        A = self.num_agents
        progress_rows = self.progress_buf if A == 1 else self.progress_buf.repeat_interleave(A)
        # read-only, so folding the agent axis into rows may copy without harm
        contact = self._contact_forces.flatten(0, 1) if A > 1 else self._contact_forces
        body_pos = self._rigid_body_pos.flatten(0, 1) if A > 1 else self._rigid_body_pos
        _, terminated = compute_humanoid_reset(torch.zeros_like(progress_rows), progress_rows,
                                               contact, self._contact_body_ids,
                                               body_pos, self.max_episode_length,
                                               self._enable_early_termination, self._termination_heights)
        self._terminate_buf[:] = terminated.view(self.num_envs, A).amax(dim=1)
        self.reset_buf[:] = torch.where(self.progress_buf >= self.max_episode_length - 1,
                                        torch.ones_like(self.reset_buf), self._terminate_buf)
        return

    def _refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        return

    def _compute_observations(self, env_ids=None):
        obs = self._compute_humanoid_obs(env_ids)

        if (env_ids is None):
            self.obs_buf[:] = obs
        else:
            self.obs_buf[env_ids] = obs

        return

    def _compute_humanoid_obs(self, env_ids=None):
        # 관측은 (env, agent) 마다 한 줄이다. 시뮬 텐서는 agent 축을 갖고 있으므로 펴고,
        # _kinematic_... 은 이미 row 당 한 줄이라 row 로 인덱싱한다.
        if (env_ids is None):
            body_pos = self.humanoid_rows(self._rigid_body_pos)
            body_rot = self.humanoid_rows(self._rigid_body_rot)
            body_vel = self.humanoid_rows(self._rigid_body_vel)
            body_ang_vel = self.humanoid_rows(self._rigid_body_ang_vel)
        else:
            rows = self.agent_rows(env_ids)
            body_pos = self._kinematic_humanoid_rigid_body_states[rows, :, 0:3]
            body_rot = self._kinematic_humanoid_rigid_body_states[rows, :, 3:7]
            body_vel = self._kinematic_humanoid_rigid_body_states[rows, :, 7:10]
            body_ang_vel = self._kinematic_humanoid_rigid_body_states[rows, :, 10:13]

        obs = compute_humanoid_observations_max(body_pos, body_rot, body_vel, body_ang_vel, self._local_root_obs_policy,
                                                self._root_height_obs_policy)
        return obs

    def _reset_actors(self, env_ids):
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        return

    def flush_metrics_atexit(self):
        if hasattr(self, "report_metrics"):
            self.report_metrics()


    def _grab_frame(self):
        """MA_VIDEO 녹화. 카메라를 env 0 의 에이전트 중앙에 따라붙인다.

        뷰어(_init_camera 계열)와 별개다 -- 뷰어는 X11+OpenGL 을 요구하는데 이 서버는
        /dev/dri 권한이 없어 못 쓴다. 카메라 센서는 GPU 그래픽스 파이프라인으로 직접
        렌더하므로 headless 에서 유일하게 동작한다.
        """
        import numpy as _np
        from isaacgym import gymapi as _g
        if not hasattr(self, "_vid"):
            cp = _g.CameraProperties()
            cp.width, cp.height = 1280, 720
            self._vid_cam = self.gym.create_camera_sensor(self.envs[0], cp)
            self._vid = []
            self._vid_every = int(os.environ.get("MA_VIDEO_EVERY", 2))
            self._vid_max = int(os.environ.get("MA_VIDEO_FRAMES", 900))
            self._vid_n = 0
        self._vid_n += 1
        if self._vid_n % self._vid_every:
            return
        h = self.humanoid_rows(self._humanoid_root_states)
        A = self.num_agents
        pos = h[:A, 0:3]                             # env 0 의 에이전트들
        c = pos.mean(dim=0).tolist()                 # 중앙
        # 두 사람이 다 프레임에 들어오도록 거리를 벌어진 정도에 맞춘다.
        spread = float(torch.cdist(pos[:, :2], pos[:, :2]).max()) if A > 1 else 0.0
        d = float(os.environ.get("MA_VIDEO_DIST", 0)) or max(2.2, 0.9 * spread + 1.6)
        # **set_camera_location 은 env 로컬 좌표를 받는다.** root_states 는 전역이라
        # 그대로 넣으면 env 원점만큼 어긋나 카메라가 엉뚱한 곳을 본다.
        o = self.gym.get_env_origin(self.envs[0])
        cx, cy, cz = c[0] - o.x, c[1] - o.y, c[2] - o.z
        self.gym.set_camera_location(self._vid_cam, self.envs[0],
                                     _g.Vec3(cx - d * 0.7, cy - d * 0.7, cz + d * 0.55),
                                     _g.Vec3(cx, cy, cz - 0.3))
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        img = self.gym.get_camera_image(self.sim, self.envs[0], self._vid_cam, _g.IMAGE_COLOR)
        self._vid.append(_np.array(img).reshape(720, 1280, 4)[:, :, :3].copy())
        if len(self._vid) >= self._vid_max:
            import cv2
            out = os.environ["MA_VIDEO"]
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            w = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
            for f in self._vid:
                w.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            w.release()
            print(f"[ma] video -> {out}  ({len(self._vid)} frames)", flush=True)
            raise SystemExit(0)
        return

    def dump_agent_layout(self):
        """모든 텐서를 env 축인지 row 축인지로 분류해 찍는다. MA_DUMP=1 일 때만.

        A>=2 로 갈 때 손대야 하는 곳은 "크기가 틀린 텐서" 다. 크래시를 하나씩 잡으면
        남은 개수를 모른 채 진행하게 되므로, 한 번에 전부 나열해서 목록으로 만든다.
        같은 표가 P0-2 (per-agent 물체) 가 고칠 자리 목록이기도 하다 -- 그때는 지금
        env 로 분류된 물체 텐서들이 row 로 옮겨간다.
        """
        E, A = self.num_envs, self.num_agents
        R = E * A
        rows = {"env": [], "row": [], "(E,A,..)": [], "기타": []}

        def classify(name, t):
            n = t.shape[0]
            if t.dim() >= 2 and n == E and t.shape[1] == A and A > 1:
                rows["(E,A,..)"].append((name, tuple(t.shape)))
            elif n == E:
                rows["env"].append((name, tuple(t.shape)))
            elif n == R:
                rows["row"].append((name, tuple(t.shape)))
            else:
                rows["기타"].append((name, tuple(t.shape)))

        seen = set()
        holders = [(self, "")] + [(v, f"{k}.") for k, v in vars(self).items()
                                  if k.endswith("_lib") and hasattr(v, "__dict__")]
        for holder, prefix in holders:
            for k, v in vars(holder).items():
                if torch.is_tensor(v) and v.dim() >= 1 and id(v) not in seen:
                    seen.add(id(v)); classify(prefix + k, v)

        print(f"\n===== agent layout dump: E={E} A={A} rows={R} =====")
        for kind in ("env", "(E,A,..)", "row", "기타"):
            if not rows[kind]:
                continue
            print(f"\n--- {kind} ({len(rows[kind])}) ---")
            for name, shape in sorted(rows[kind]):
                print(f"  {name:52s} {shape}")
        print("\n===== end dump =====\n", flush=True)

    def pre_physics_step(self, actions):
        d = os.environ.get("MA_OBSDUMP")
        if d and not getattr(self, "_obsdumped", False):
            self._obsdumped = True
            import numpy as _np
            _np.save(d, self.obs_buf.detach().cpu().numpy())
            print(f"[ma] obs -> {d}  {tuple(self.obs_buf.shape)}", flush=True)
            raise SystemExit(0)
        if os.environ.get("MA_DUMP") and not getattr(self, "_dumped", False):
            self._dumped = True
            self.dump_agent_layout()
        # MA_VIDEO=<mp4> : 카메라 센서로 headless 녹화. env 0 의 두 에이전트를 따라간다.
        #   MA_VIDEO_FRAMES  받을 프레임 수 (기본 900 = 30 fps 로 30 초)
        #   MA_VIDEO_EVERY   몇 스텝마다 한 프레임 (기본 2)
        # X11/VNC 가 없는 서버용. 카메라는 X 를 안 거치고 GPU 로 직접 렌더한다.
        if os.environ.get("MA_VIDEO"):
            self._grab_frame()

        # MA_POSDUMP=1 : 리셋 직후 env 0~2 의 실제 좌표. 박스가 뭔가에 파묻혀 튀는지 본다.
        if os.environ.get("MA_POSDUMP") and not getattr(self, "_posdumped", False):
            self._posdumped = True
            A = self.num_agents
            h = self._humanoid_root_states.view(self.num_envs, A, -1)
            for e in range(min(3, self.num_envs)):
                for a in range(A):
                    p = [f"hum ({h[e,a,0]:+7.2f},{h[e,a,1]:+7.2f},{h[e,a,2]:+5.2f})"]
                    for nm in ("_box_states", "_platform_states", "_tar_platform_states"):
                        t = getattr(self, nm, None)
                        if t is not None:
                            v = t.view(self.num_envs, A, -1)[e, a]
                            p.append(f"{nm[1:6]} ({v[0]:+7.2f},{v[1]:+7.2f},{v[2]:+6.2f})")
                    sz = getattr(getattr(self, "_box_lib", None), "_box_size", None)
                    if sz is not None:
                        p.append(f"boxsz {sz[e*A+a].tolist()}")
                    print(f"[pos] env{e} agent{a}  " + "  ".join(p), flush=True)
            raise SystemExit(0)
        self.actions = actions.to(self.device).clone()
        # MA_FREEZE=k : k 번 에이전트만 움직이고 나머지는 액션 0. 게으른 에이전트 진단용.
        if not hasattr(self, "_freeze_mask"):
            _fz = os.environ.get("MA_FREEZE")
            self._freeze_mask = None if (_fz is None or self.num_agents == 1) else \
                (torch.arange(self._rows, device=self.device) % self.num_agents) != int(_fz)
        if self._freeze_mask is not None:
            self.actions[self._freeze_mask] = 0.0
        if (self._pd_control):
            pd_tar = self._action_to_pd_targets(self.actions)

            # tracking initial state of each env (use for debug)
            if self.cfg["env"]["enableTrackInitState"]:
                pd_tar = self._every_env_init_dof_pos.clone()

            pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
            self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)
        else:
            forces = self.actions * self.motor_efforts.unsqueeze(0) * self.power_scale
            force_tensor = gymtorch.unwrap_tensor(forces)
            self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

        return

    def post_physics_step(self):
        self.progress_buf += 1

        self._refresh_sim_tensors()
        self._compute_observations()
        self._compute_reward(self.actions)
        self._compute_reset()
        
        # rl_games sees one row per (env, agent); every agent of an env shares
        # that env's episode outcome.
        self.extras["terminate"] = (self._terminate_buf if self.num_agents == 1
                                    else self._terminate_buf.repeat_interleave(self.num_agents))

        # debug viz
        if self.viewer and self.debug_viz:
            self._update_debug_viz()

        return

    def render(self, sync_frame_time=False):
        if self.viewer:
            self._update_camera()

        super().render(sync_frame_time)
        return

    def _build_key_body_ids_tensor(self, key_body_names):
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []

        for body_name in key_body_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _build_contact_body_ids_tensor(self, contact_body_names):
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []

        for body_name in contact_body_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _action_to_pd_targets(self, action):
        pd_tar = self._pd_action_offset + self._pd_action_scale * action
        return pd_tar

    def _init_camera(self):
        # A>=2 에서 _humanoid_root_states 는 (E, A, 13) 이라 [0, 0:3] 이 xyz 가 아니라
        # **앞 세 에이전트**를 고른다. 기본은 env0/agent0이고, 영상에서는
        # MS_CAM_AGENT로 같은 env의 다른 agent를 선택할 수 있다.
        self._camera_follow_agent = int(os.environ.get("MS_CAM_AGENT", 0))
        if not 0 <= self._camera_follow_agent < self.num_agents:
            raise ValueError(
                f"MS_CAM_AGENT must be in [0, {self.num_agents - 1}], "
                f"got {self._camera_follow_agent}"
            )
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._cam_prev_char_pos = self.humanoid_rows(self._humanoid_root_states)[
            self._camera_follow_agent, 0:3
        ].cpu().numpy()
        
        cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0], 
                              self._cam_prev_char_pos[1] - 3.0, 
                              1.0)
        cam_target = gymapi.Vec3(self._cam_prev_char_pos[0],
                                 self._cam_prev_char_pos[1],
                                 1.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        return

    def _update_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self.humanoid_rows(self._humanoid_root_states)[
            self._camera_follow_agent, 0:3
        ].cpu().numpy()
        
        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
        cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], 1.0)
        new_cam_pos = gymapi.Vec3(char_root_pos[0] + cam_delta[0], 
                                  char_root_pos[1] + cam_delta[1], 
                                  cam_pos[2])

        if self._camera_follow_flag:
            self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

        self._cam_prev_char_pos[:] = char_root_pos
        return

    def _update_debug_viz(self):
        self.gym.clear_lines(self.viewer)
        return
    
    def _fetch_humanoid_rigid_body_pos_rot_states(self, env_ids=None):
        if env_ids is None:
            env_ids = to_torch(np.arange(self.num_envs), dtype=torch.long, device=self.device)

        rigid_body_pos_rot_states = torch.cat((self._rigid_body_pos[env_ids], self._rigid_body_rot[env_ids]), dim=-1)
        kinematic_buffer = self._kinematic_humanoid_rigid_body_states[env_ids, :, 0:7]
        
        # For the two variables that have just been reset, 
        # the values ​​will not be updated, so read the correct values ​​from the buffer
        mask = (self.progress_buf[env_ids] == 0)
        rigid_body_pos_rot_states[mask] = kinematic_buffer[mask]
        return rigid_body_pos_rot_states

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def dof_to_obs(pose, dof_obs_size, dof_offsets):
    # type: (Tensor, int, List[int]) -> Tensor
    joint_obs_size = 6
    num_joints = len(dof_offsets) - 1

    dof_obs_shape = pose.shape[:-1] + (dof_obs_size,)
    dof_obs = torch.zeros(dof_obs_shape, device=pose.device)
    dof_obs_offset = 0

    for j in range(num_joints):
        dof_offset = dof_offsets[j]
        dof_size = dof_offsets[j + 1] - dof_offsets[j]
        joint_pose = pose[:, dof_offset:(dof_offset + dof_size)]

        # assume this is a spherical joint
        if (dof_size == 3):
            joint_pose_q = torch_utils.exp_map_to_quat(joint_pose)
        elif (dof_size == 1):
            axis = torch.tensor([0.0, 1.0, 0.0], dtype=joint_pose.dtype, device=pose.device)
            joint_pose_q = quat_from_angle_axis(joint_pose[..., 0], axis)
        else:
            joint_pose_q = None
            assert(False), "Unsupported joint type"

        joint_dof_obs = torch_utils.quat_to_tan_norm(joint_pose_q)
        dof_obs[:, (j * joint_obs_size):((j + 1) * joint_obs_size)] = joint_dof_obs

    assert((num_joints * joint_obs_size) == dof_obs_size)

    return dof_obs

@torch.jit.script
def compute_humanoid_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos,
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
def compute_humanoid_observations_max(body_pos, body_rot, body_vel, body_ang_vel, local_root_obs, root_height_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, bool) -> Tensor
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    
    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    
    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
    flat_local_body_pos = quat_rotate(flat_heading_rot, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
    local_body_pos = local_body_pos[..., 3:] # remove root pos

    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])
    flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot)
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1])
    
    if (not local_root_obs):
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot)
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = quat_rotate(flat_heading_rot, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])
    
    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = quat_rotate(flat_heading_rot, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])
    
    obs = torch.cat((root_h_obs, local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel), dim=-1)
    return obs


@torch.jit.script
def compute_humanoid_reward(obs_buf):
    # type: (Tensor) -> Tensor
    reward = torch.ones_like(obs_buf[:, 0])
    return reward

@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_heights):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor) -> Tuple[Tensor, Tensor]
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
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)
    
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated
