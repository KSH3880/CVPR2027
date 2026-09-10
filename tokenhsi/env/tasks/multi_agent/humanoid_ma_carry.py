# Multi-agent box carrying.
#
# M humanoids share one scene with O >= M boxes and M target locations.  Every humanoid
# owns exactly one distinct box; the remaining O-M boxes are unassigned distractors.
# legacy_multirow builds one (env, agent) row in that agent's own heading frame, with
# entities rotated so the ego comes first:
#
#   row = [ humanoid_0..humanoid_{M-1} | object_0..object_{O-1} | goal_0..goal_{M-1} ]
#           M * 230                      O * 39                   M * 6
#
# clean_scene instead stores one H/O/T node set per env: 223-D Humans, 30-D Objects,
# and 1-D Targets, followed by one 7-D pose per token for GTA.
# Both layouts are entity-blocked so tokenizers are shared across same-type slots.
#
# Actor layout per env: [humanoids, boxes, (source/target platform pairs), (markers)]

import os
import yaml
from enum import Enum
import numpy as np
import torch

from isaacgym import gymapi
from isaacgym import gymtorch

from env.tasks.multi_agent.humanoid_ma import HumanoidMA
from env.tasks.multi_agent.scene_features import build_gta_pose_records
from env.tasks.multi_agent.relation_task import CarryRelationMixin
from utils.relation_task_spec import STATE_MODE, LEGACY_MODE, validate_relation_config
from env.tasks.humanoid import dof_to_obs
from utils.motion_lib import MotionLib
from isaacgym.torch_utils import *

from utils import torch_utils


class HumanoidMACarry(CarryRelationMixin, HumanoidMA):
    _PLATFORM_COLLISION_FILTER = 1 << 30
    REWARD_TERM_NAMES = ("walk", "carry", "handheld", "putdown", "power", "collision", "total")

    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._relation_cfg = cfg['env'].get('relationReward', {})
        validate_relation_config(self._relation_cfg)
        self._state_relation = self._relation_cfg.get('mode', LEGACY_MODE) == STATE_MODE
        if self._state_relation:
            if cfg['env'].get('policyObsMode', 'legacy_multirow') != 'clean_scene':
                raise ValueError('state_relation_v0 requires policyObsMode=clean_scene')
            self.REWARD_TERM_NAMES = ('holding_delta', 'at_delta', 'holding_velocity',
                'at_velocity', 'success_bonus', 'power', 'collision', 'box_speed', 'total')
        num_agents = int(cfg["env"].get("numAgents", 1))
        configured_objects = int(cfg["env"].get("numObjects", 0))
        self.num_objects = num_agents if configured_objects <= 0 else configured_objects
        assert self.num_objects >= num_agents, \
            "numObjects ({}) must be >= numAgents ({})".format(self.num_objects, num_agents)

        self._enable_task_obs = cfg["env"]["enableTaskObs"]
        self._only_vel_reward = cfg["env"]["onlyVelReward"]
        self._only_height_handheld_reward = cfg["env"]["onlyHeightHandHeldReward"]

        self._box_vel_penalty = cfg["env"]["box_vel_penalty"]
        self._box_vel_pen_coeff = cfg["env"]["box_vel_pen_coeff"]
        self._box_vel_pen_thre = cfg["env"]["box_vel_pen_threshold"]

        self._agent_collision_penalty = cfg["env"].get("agentCollisionPenalty", True)
        self._agent_collision_coeff = cfg["env"].get("agentCollisionCoeff", 0.5)
        self._agent_collision_dist = cfg["env"].get("agentCollisionDist", 0.7)

        self._mode = cfg["env"]["mode"]
        assert self._mode in ["train", "test"]
        if cfg["args"].eval:
            self._mode = "test"

        box_cfg = cfg["env"]["box"]
        self._build_base_size = box_cfg["build"]["baseSize"]
        self._build_random_size = box_cfg["build"]["randomSize"]
        self._build_random_mode_equal_proportion = box_cfg["build"]["randomModeEqualProportion"]
        self._build_x_scale_range = box_cfg["build"]["scaleRangeX"]
        self._build_y_scale_range = box_cfg["build"]["scaleRangeY"]
        self._build_z_scale_range = box_cfg["build"]["scaleRangeZ"]
        self._build_scale_sample_interval = box_cfg["build"]["scaleSampleInterval"]
        self._build_test_sizes = box_cfg["build"]["testSizes"]

        assert box_cfg["build"].get("randomDensity", False) is False, \
            "randomDensity is not supported by the multi-agent carry task yet"
        assert box_cfg["obs"]["enableBboxObs"] is True, \
            "the multi-agent object token always carries the bbox points"

        self._reset_random_rot = box_cfg["reset"]["randomRot"]
        self._reset_random_height = box_cfg["reset"]["randomHeight"]
        self._reset_random_height_prob = box_cfg["reset"]["randomHeightProb"]
        self._reset_min_platform_height = box_cfg["reset"].get("minPlatformHeight", 0.0)
        self._reset_max_top_surface_height = box_cfg["reset"]["maxTopSurfaceHeight"]
        self._randomize_box_assignment = bool(box_cfg["reset"].get("randomAssignment", True))
        self._random_arena_box_spawn = bool(
            getattr(cfg["args"], "test", False)
            and box_cfg["reset"].get("testRandomArenaSpawn", True))
        configured_radius = float(box_cfg["reset"].get("arenaSpawnRadius", 0.0))
        self._box_spawn_radius = configured_radius if configured_radius > 0.0 \
            else max(0.5, float(cfg["env"]["envSpacing"]) - 0.5)
        self._box_min_agent_dist = float(box_cfg["reset"].get("minAgentDistance", 1.0))
        self._box_min_box_dist = float(box_cfg["reset"].get("minBoxDistance", 0.7))

        state_init = cfg["env"]["stateInit"]
        self._state_init = HumanoidMACarry.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        assert (self._num_amp_obs_steps >= 2)

        self._reset_default_slots = None
        self._reset_ref_slots = {}
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
        self._skill_init_prob = torch.tensor(cfg["env"]["skillInitProb"], device=self.device, dtype=torch.float)
        self._skill_disc_prob = torch.tensor(cfg["env"]["skillDiscProb"], device=self.device, dtype=torch.float)

        motion_file = cfg['env']['motion_file']
        self._load_motion(motion_file)

        N, M = self.num_envs, self.num_agents

        self._amp_obs_buf = torch.zeros((N, M, self._num_amp_obs_steps, self._num_amp_obs_per_step),
                                        device=self.device, dtype=torch.float)
        self._curr_amp_obs_buf = self._amp_obs_buf[:, :, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, :, 1:]
        self._amp_obs_demo_buf = None

        self._prev_root_pos = torch.zeros([N, M, 3], device=self.device, dtype=torch.float)
        self._prev_box_pos = torch.zeros([N, M, 3], device=self.device, dtype=torch.float)
        self._tar_pos = torch.zeros([N, M, 3], device=self.device, dtype=torch.float)
        self._reward_term_sums = torch.zeros(len(self.REWARD_TERM_NAMES), device=self.device)
        self._reward_term_count = 0

        # each agent samples its target inside its own disc so that, with several agents,
        # the scene stays inside the env and the sub-tasks do not fully overlap
        spacing = cfg["env"]["envSpacing"]
        reach = min(4.5, max(0.5, spacing - 0.5))
        if M > 1:
            reach = max(1.5, reach - float(self._agent_spawn_offsets.norm(dim=-1).max()))
        self._tar_reach = reach
        self._tar_height_range = [0.5, 1.0]

        if self._enable_markers:
            self._build_marker_state_tensors()

        self._build_box_tensors()
        if self._reset_random_height:
            self._build_platform_state_tensors()
        self._build_assignment_tensors()

        self._every_env_init_dof_pos = torch.zeros((N, M, self.num_dof), device=self.device, dtype=torch.float)
        self._kinematic_humanoid_rigid_body_states = torch.zeros((N, M, self.num_bodies, 13),
                                                                 device=self.device, dtype=torch.float)

        self._is_eval = cfg["args"].eval
        if self._is_eval:
            self._success_buf = torch.zeros((N * M), device=self.device, dtype=torch.long)
            self._precision_buf = torch.zeros((N * M), device=self.device, dtype=torch.float)
            self._success_threshold = cfg["env"]["eval"]["successThreshold"]

            self._skill = cfg["env"]["eval"]["skill"]
            self._skill_init_prob = torch.tensor(cfg["env"]["eval"]["skillInitProb"],
                                                 device=self.device, dtype=torch.float)

        if self._state_relation:
            self._init_relation_runtime()
        return

    # ------------------------------------------------------------------ sizes

    def get_relation_suffix_size(self):
        return 9 * self.num_agents if self._state_relation else 0

    def get_object_obs_size(self):
        # lin vel (3) + ang vel (3) + pos (3) + rot tan-norm (6) + bbox points (8 * 3)
        return 3 + 3 + 3 + 6 + 24

    def get_goal_obs_size(self):
        # target in ego frame (3) + object-to-target in ego frame (3)
        return 3 + 3

    def get_clean_object_obs_size(self):
        # Object-local linear/angular velocity and local bbox corners.
        return 3 + 3 + 24

    def get_clean_goal_obs_size(self):
        # Carry targets have no intrinsic state; position/frame lives in the GTA pose.
        return 1

    def get_scene_kinematic_size(self):
        # Env-local position plus local-to-env quaternion used only by GTA.
        return 3 + 4

    def get_scene_arena_scale(self):
        """Retained for configuration plumbing and historical Geo ablations."""
        return self._arena_scale

    def get_geometry_obs_size(self):
        # relative position + tan/normal orientation + relative lin/ang velocity
        return 3 + 6 + 3 + 3

    def get_scene_entity_sizes(self):
        return [self.get_clean_humanoid_obs_size(),
                self.get_clean_object_obs_size(),
                self.get_clean_goal_obs_size()]

    def get_scene_normalized_entity_sizes(self):
        """Normalize intrinsic H/O nodes; constant Target and GTA poses pass through."""
        return [self.get_clean_humanoid_self_obs_size(), self.get_clean_object_obs_size(), 0]

    def get_scene_num_tokens(self):
        return 2 * self.num_agents + self.num_objects

    def get_task_obs_size(self):
        if not self._enable_task_obs:
            return 0
        return self.num_objects * self.get_object_obs_size() \
            + self.num_agents * self.get_goal_obs_size()

    def get_obs_size(self):
        if self.is_scene_policy():
            node_width = (self.num_agents * self.get_clean_humanoid_obs_size()
                          + self.num_objects * self.get_clean_object_obs_size()
                          + self.num_agents * self.get_clean_goal_obs_size())
            kinematic_width = self.get_scene_num_tokens() * self.get_scene_kinematic_size()
            return node_width + kinematic_width + self.get_relation_suffix_size()

        obs_size = self.num_agents * self.get_humanoid_obs_size()
        if self._enable_task_obs:
            obs_size += self.get_task_obs_size()
        return obs_size

    # ------------------------------------------------------------------ assets / envs

    def _create_envs(self, num_envs, spacing, num_per_row):
        # markers are also needed off-screen when we record training videos
        self._enable_markers = (not self.headless) or self._video_enabled

        if self._enable_markers:
            self._marker_handles = []
            self._load_marker_asset()

        if self._reset_random_height:
            self._platform_handles = []
            self._tar_platform_handles = []
            self._load_platform_asset()

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

    def _load_platform_asset(self):
        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 0.0
        # GPU PhysX does not move fixed collision shapes with root-state updates.
        asset_options.density = 1.0e10
        asset_options.fix_base_link = False
        asset_options.disable_gravity = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._platform_height = 0.02
        self._platform_inactive_height = 20.0
        self._platform_asset = self.gym.create_box(self.sim, 0.4, 0.4, self._platform_height, asset_options)
        return

    def _load_box_asset(self):
        N, O = self.num_envs, self.num_objects
        num_boxes = N * O

        self._box_scale = torch.ones((num_boxes, 3), dtype=torch.float32, device=self.device)
        if self._build_random_size:
            assert int((self._build_x_scale_range[1] - self._build_x_scale_range[0]) % self._build_scale_sample_interval) == 0
            assert int((self._build_y_scale_range[1] - self._build_y_scale_range[0]) % self._build_scale_sample_interval) == 0
            assert int((self._build_z_scale_range[1] - self._build_z_scale_range[0]) % self._build_scale_sample_interval) == 0

            x_scale_linespace = torch.arange(self._build_x_scale_range[0],
                                             self._build_x_scale_range[1] + self._build_scale_sample_interval,
                                             self._build_scale_sample_interval)

            if self._build_random_mode_equal_proportion:
                num_scales = len(x_scale_linespace)
                scale_pool = torch.zeros((num_scales, 3), device=self.device)
                for idx, curr_x in enumerate(x_scale_linespace):
                    scale_pool[idx] = torch.tensor([curr_x, curr_x, curr_x])

                if self._mode == "test":
                    test_sizes = torch.tensor(self._build_test_sizes, device=self.device)
                    scale_pool = torch.zeros((test_sizes.shape[0], 3), device=self.device)
                    num_scales = test_sizes.shape[0]
                    for axis in range(3):
                        scale_pool[:, axis] = test_sizes[:, axis] / self._build_base_size[axis]
            else:
                y_scale_linespace = torch.arange(self._build_y_scale_range[0],
                                                 self._build_y_scale_range[1] + self._build_scale_sample_interval,
                                                 self._build_scale_sample_interval)
                z_scale_linespace = torch.arange(self._build_z_scale_range[0],
                                                 self._build_z_scale_range[1] + self._build_scale_sample_interval,
                                                 self._build_scale_sample_interval)
                num_scales = len(x_scale_linespace) * len(y_scale_linespace) * len(z_scale_linespace)
                scale_pool = torch.zeros((num_scales, 3), device=self.device)
                idx = 0
                for curr_x in x_scale_linespace:
                    for curr_y in y_scale_linespace:
                        for curr_z in z_scale_linespace:
                            scale_pool[idx] = torch.tensor([curr_x, curr_y, curr_z])
                            idx += 1

            if num_boxes >= num_scales:
                self._box_scale[:num_scales] = scale_pool[:num_scales]
                sampled = torch.multinomial(torch.ones(num_scales) * (1.0 / num_scales),
                                            num_samples=(num_boxes - num_scales), replacement=True)
                self._box_scale[num_scales:] = scale_pool[sampled]
                self._box_scale = self._box_scale[torch.randperm(num_boxes)]
            else:
                sampled = torch.multinomial(torch.ones(num_scales) * (1.0 / num_scales),
                                            num_samples=num_boxes, replacement=True)
                self._box_scale = scale_pool[sampled]

        self._box_density = torch.full((num_boxes,), 100.0, dtype=torch.float32, device=self.device)
        self._box_size = torch.tensor(self._build_base_size, device=self.device).reshape(1, 3) * self._box_scale

        self._box_assets = []
        for i in range(num_boxes):
            asset_options = gymapi.AssetOptions()
            asset_options.angular_damping = 0.01
            asset_options.linear_damping = 0.01
            asset_options.max_angular_velocity = 100.0
            asset_options.density = self._box_density[i]
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            self._box_assets.append(self.gym.create_box(self.sim,
                                                        self._box_size[i, 0],
                                                        self._box_size[i, 1],
                                                        self._box_size[i, 2],
                                                        asset_options))

        # (N, O, ...) from here on
        self._box_scale = self._box_scale.view(N, O, 3)
        self._box_size = self._box_size.view(N, O, 3)
        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)

        for humanoid_handle in self.humanoid_handles_all[env_id]:
            self._filter_platform_foot_collisions(env_ptr, humanoid_handle)

        for object_id in range(self.num_objects):
            self._build_box(env_id, object_id, env_ptr)

        if self._reset_random_height:
            for agent_id in range(self.num_agents):
                self._build_platforms(env_id, agent_id, env_ptr)

        if self._enable_markers:
            for a in range(self.num_agents):
                self._build_marker(env_id, a, env_ptr)
        return

    def _filter_platform_foot_collisions(self, env_ptr, humanoid_handle):
        body_dict = self.gym.get_actor_rigid_body_dict(env_ptr, humanoid_handle)
        shape_ranges = self.gym.get_actor_rigid_body_shape_indices(env_ptr, humanoid_handle)
        shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, humanoid_handle)

        for body_name in ("left_foot", "right_foot"):
            shape_range = shape_ranges[body_dict[body_name]]
            for shape_id in range(shape_range.start, shape_range.start + shape_range.count):
                shape_props[shape_id].filter |= self._PLATFORM_COLLISION_FILTER

        self.gym.set_actor_rigid_shape_properties(env_ptr, humanoid_handle, shape_props)
        return

    def _build_box(self, env_id, object_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        box_idx = env_id * self.num_objects + object_id
        # This pose only exists until the first reset.  Spread extra objects around the
        # arena so that actor creation itself never depends on an owner assignment.
        angle = 2.0 * np.pi * float(object_id) / float(max(1, self.num_objects))
        radius = min(self._box_spawn_radius, 1.0 + 0.5 * float(object_id))

        default_pose = gymapi.Transform()
        default_pose.p.x = radius * np.cos(angle)
        default_pose.p.y = radius * np.sin(angle)
        default_pose.p.z = float(self._box_size.view(-1, 3)[box_idx, 2]) / 2

        box_handle = self.gym.create_actor(env_ptr, self._box_assets[box_idx], default_pose,
                                           "box_{}".format(object_id), col_group, col_filter, segmentation_id)
        self._box_handles.append(box_handle)
        return

    def _build_platforms(self, env_id, agent_id, env_ptr):
        col_group = env_id
        col_filter = self._PLATFORM_COLLISION_FILTER
        segmentation_id = 0

        default_pose = gymapi.Transform()
        default_pose.p.x = 0.5 * agent_id
        default_pose.p.z = self._platform_inactive_height
        platform_handle = self.gym.create_actor(
            env_ptr, self._platform_asset, default_pose,
            "platform_{}".format(agent_id), col_group, col_filter, segmentation_id)
        self.gym.set_rigid_body_color(
            env_ptr, platform_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.5, 0.235, 0.6))

        default_pose.p.z = self._platform_inactive_height + 0.5
        tar_platform_handle = self.gym.create_actor(
            env_ptr, self._platform_asset, default_pose,
            "tar_platform_{}".format(agent_id), col_group, col_filter, segmentation_id)
        self.gym.set_rigid_body_color(
            env_ptr, tar_platform_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.0, 0.0, 0.8))

        self._platform_handles.append(platform_handle)
        self._tar_platform_handles.append(tar_platform_handle)
        return

    def _build_marker(self, env_id, agent_id, env_ptr):
        col_group = self.num_envs + 1
        col_filter = 0
        segmentation_id = 0
        default_pose = gymapi.Transform()

        marker_handle = self.gym.create_actor(env_ptr, self._marker_asset, default_pose,
                                              "marker_{}".format(agent_id), col_group, col_filter, segmentation_id)
        self.gym.set_rigid_body_color(env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.0, 0.0))
        self.gym.set_actor_scale(env_ptr, marker_handle, 0.3)
        self._marker_handles.append(marker_handle)
        return

    def _build_box_tensors(self):
        N, M, O = self.num_envs, self.num_agents, self.num_objects
        num_actors = self.get_num_actors_per_env()

        self._box_states = self._root_states.view(N, num_actors, self._root_states.shape[-1])[:, M:M + O, :]
        self._box_actor_ids = \
            (num_actors * torch.arange(N, device=self.device, dtype=torch.int32)).unsqueeze(-1) \
            + torch.arange(O, device=self.device, dtype=torch.int32).unsqueeze(0) + M  # (N, O)

        self._initial_box_states = self._box_states.clone()
        self._initial_box_states[..., 7:13] = 0

        self._build_box_bps()
        return

    def _build_box_bps(self):
        half = self._box_size / 2.0                      # (N, O, 3)
        signs = torch.tensor([
            [1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, -1],
            [1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1],
        ], device=self.device, dtype=torch.float)        # (8, 3)
        self._box_bps = half.unsqueeze(-2) * signs.view(1, 1, 8, 3)  # (N, O, 8, 3)
        return

    def _build_platform_state_tensors(self):
        N, M, O = self.num_envs, self.num_agents, self.num_objects
        num_actors = self.get_num_actors_per_env()
        actor_base = num_actors * torch.arange(N, device=self.device, dtype=torch.int32).unsqueeze(-1)
        platform_offsets = M + O + 2 * torch.arange(M, device=self.device, dtype=torch.int32)
        tar_platform_offsets = platform_offsets + 1
        root_states = self._root_states.view(N, num_actors, self._root_states.shape[-1])

        platform_start = M + O
        self._platform_states = root_states[:, platform_start:platform_start + 2 * M:2, :]
        self._platform_pos = self._platform_states[..., :3]
        self._platform_default_pos = self._platform_pos.clone()
        self._platform_actor_ids = actor_base + platform_offsets.unsqueeze(0)

        self._tar_platform_states = root_states[:, platform_start + 1:platform_start + 2 * M:2, :]
        self._tar_platform_pos = self._tar_platform_states[..., :3]
        self._tar_platform_default_pos = self._tar_platform_pos.clone()
        self._tar_platform_actor_ids = actor_base + tar_platform_offsets.unsqueeze(0)
        return

    def _build_assignment_tensors(self):
        """Initialise logical owner slots independently from physical box indices."""
        N, M, O = self.num_envs, self.num_agents, self.num_objects
        identity = torch.arange(M, device=self.device, dtype=torch.long).unsqueeze(0).expand(N, -1)
        self._agent_box_assignment = identity.clone()               # (N, M), physical ids
        self._logical_box_order = torch.arange(O, device=self.device, dtype=torch.long) \
            .unsqueeze(0).expand(N, -1).clone()                      # assigned first, extras last
        return

    def _sample_box_assignments(self, env_ids):
        """Assign M distinct physical boxes and put unassigned boxes after them."""
        K, M, O = env_ids.shape[0], self.num_agents, self.num_objects
        if self._randomize_box_assignment:
            assignment = torch.rand(K, O, device=self.device).argsort(dim=-1)[:, :M]
        else:
            assignment = torch.arange(M, device=self.device, dtype=torch.long).unsqueeze(0).expand(K, -1)

        all_ids = torch.arange(O, device=self.device, dtype=torch.long).unsqueeze(0).expand(K, -1)
        assigned_mask = torch.zeros(K, O, device=self.device, dtype=torch.bool)
        assigned_mask.scatter_(1, assignment, True)
        unassigned = all_ids.masked_select(~assigned_mask).view(K, O - M)

        self._agent_box_assignment[env_ids] = assignment
        self._logical_box_order[env_ids] = torch.cat([assignment, unassigned], dim=-1)
        self._update_box_assignment_colors(env_ids)
        return

    def _update_box_assignment_colors(self, env_ids):
        """Match assigned boxes to owner colors; render unassigned boxes in grey."""
        rendered_envs = []
        if self.viewer is not None:
            rendered_envs = env_ids.detach().cpu().tolist()
        elif self._video_enabled and torch.any(env_ids == 0):
            # Headless training only records env 0, so recoloring hundreds of invisible
            # environments would add needless host API work to every reset.
            rendered_envs = [0]

        unassigned_color = gymapi.Vec3(0.45, 0.45, 0.45)
        for env_id in rendered_envs:
            env_ptr = self.envs[env_id]
            assignment = self._agent_box_assignment[env_id].detach().cpu().tolist()
            owner_by_box = {physical_box: agent_id
                            for agent_id, physical_box in enumerate(assignment)}

            for physical_box in range(self.num_objects):
                box_handle = self._box_handles[env_id * self.num_objects + physical_box]
                agent_id = owner_by_box.get(physical_box)
                color = unassigned_color if agent_id is None else self._agent_color(agent_id)
                self.gym.set_rigid_body_color(
                    env_ptr, box_handle, 0, gymapi.MESH_VISUAL, color)
        return

    @staticmethod
    def _gather_entity_slots(values, indices):
        """values (B, K, ...) + indices (B, S) -> (B, S, ...)."""
        batch = torch.arange(values.shape[0], device=values.device).unsqueeze(-1)
        return values[batch, indices]

    def _assigned_box_values(self, values, env_ids=None):
        if env_ids is None:
            return self._gather_entity_slots(values, self._agent_box_assignment)
        return self._gather_entity_slots(values[env_ids], self._agent_box_assignment[env_ids])

    def _regulate_height(self, height, box_size):
        top_surface_z = height + box_size[:, 2] / 2
        top_surface_z = torch.clamp_max(top_surface_z, self._reset_max_top_surface_height)
        return top_surface_z - box_size[:, 2] / 2

    def _logical_box_values(self, values, env_ids=None):
        if env_ids is None:
            return self._gather_entity_slots(values, self._logical_box_order)
        return self._gather_entity_slots(values[env_ids], self._logical_box_order[env_ids])

    def _build_marker_state_tensors(self):
        N, M, O = self.num_envs, self.num_agents, self.num_objects
        num_actors = self._root_states.shape[0] // N
        marker_start = M + O + (2 * M if self._reset_random_height else 0)

        self._marker_states = self._root_states.view(N, num_actors, self._root_states.shape[-1]) \
            [:, marker_start:marker_start + M, :]
        self._marker_pos = self._marker_states[..., :3]
        self._marker_actor_ids = \
            (num_actors * torch.arange(N, device=self.device, dtype=torch.int32)).unsqueeze(-1) \
            + torch.arange(M, device=self.device, dtype=torch.int32).unsqueeze(0) + marker_start
        return

    # ------------------------------------------------------------------ observations

    def _compute_observations(self, env_ids=None):
        if self.is_scene_policy():
            obs = self._compute_clean_scene_obs(env_ids)
            if env_ids is None:
                self.obs_buf[:] = obs
            else:
                self.obs_buf[env_ids] = obs
            return

        humanoid_obs = self._compute_humanoid_obs(env_ids)          # (B, M, M * 230)

        if (self._enable_task_obs):
            task_obs = self._compute_task_obs(env_ids)              # (B, M, O * 39 + M * 6)
            obs = torch.cat([humanoid_obs, task_obs], dim=-1)
        else:
            obs = humanoid_obs

        B = obs.shape[0]
        obs = obs.reshape(B * self.num_agents, -1)

        if (env_ids is None):
            self.obs_buf[:] = obs
        else:
            self.obs_buf[self._flat_slot_ids(env_ids)] = obs
        return

    def _compute_clean_scene_obs(self, env_ids=None):
        """Build one clean, compact scene observation per environment.

        Layout: [M*H223 | O*O30 | M*T1 | (2M+O)*pose7].  Pose records are
        env-local position plus local-to-env orientation and bypass node RMS.
        """
        if env_ids is None:
            body_pos = self._rigid_body_pos
            body_rot = self._rigid_body_rot
            body_vel = self._rigid_body_vel
            body_ang_vel = self._rigid_body_ang_vel
            box_states = self._logical_box_values(self._box_states)
            box_bps = self._logical_box_values(self._box_bps)
            tar_pos = self._tar_pos
            origins = self._env_origins
        else:
            kin = self._kinematic_humanoid_rigid_body_states[env_ids]
            body_pos = kin[..., 0:3]
            body_rot = kin[..., 3:7]
            body_vel = kin[..., 7:10]
            body_ang_vel = kin[..., 10:13]
            box_states = self._logical_box_values(self._box_states, env_ids)
            box_bps = self._logical_box_values(self._box_bps, env_ids)
            tar_pos = self._tar_pos[env_ids]
            origins = self._env_origins[env_ids]

        B, M, O = body_pos.shape[0], self.num_agents, self.num_objects
        human_nodes = self._compute_clean_humanoid_nodes(
            body_pos, body_rot, body_vel, body_ang_vel, origins)

        box_rot = box_states[..., 3:7]
        box_rot_inv = box_rot.clone()
        box_rot_inv[..., 0:3] *= -1.0
        object_local = torch.cat([
            quat_rotate(box_rot_inv.reshape(-1, 4), box_states[..., 7:10].reshape(-1, 3)).view(B, O, 3),
            quat_rotate(box_rot_inv.reshape(-1, 4), box_states[..., 10:13].reshape(-1, 3)).view(B, O, 3),
            box_bps.reshape(B, O, 24),
        ], dim=-1)
        object_nodes = object_local
        goal_nodes = torch.ones(B, M, 1, device=self.device, dtype=body_pos.dtype)
        assert human_nodes.shape == (B, M, self.get_clean_humanoid_obs_size())
        assert object_nodes.shape == (B, O, self.get_clean_object_obs_size())
        assert goal_nodes.shape == (B, M, self.get_clean_goal_obs_size())

        # Use the same body state source as the human nodes. During reference-state
        # resets the simulator tensor has not necessarily been refreshed yet.
        root_pos = body_pos[..., 0, :]
        root_rot = body_rot[..., 0, :]
        human_heading = torch_utils.calc_heading_quat(root_rot.reshape(-1, 4)).view(B, M, 4)

        nodes = torch.cat([
            human_nodes.reshape(B, -1),
            object_nodes.reshape(B, -1),
            goal_nodes.reshape(B, -1),
        ], dim=-1)
        entity_poses = build_gta_pose_records(
            root_pos, human_heading,
            box_states[..., 0:3], box_rot,
            tar_pos, origins)
        parts = [nodes, entity_poses.reshape(B, -1)]
        if self._state_relation:
            parts.append(self.relation_runtime.suffix(env_ids))
        return torch.cat(parts, dim=-1)

    def _compute_task_obs(self, env_ids=None):
        if (env_ids is None):
            root_states = self._humanoid_root_states
            tar_pos = self._tar_pos
            logical_box_states = self._logical_box_values(self._box_states)
            logical_box_bps = self._logical_box_values(self._box_bps)
        else:
            root_states = self._humanoid_root_states[env_ids]
            tar_pos = self._tar_pos[env_ids]
            logical_box_states = self._logical_box_values(self._box_states, env_ids)
            logical_box_bps = self._logical_box_values(self._box_bps, env_ids)

        B = root_states.shape[0]
        M, O = self.num_agents, self.num_objects
        perm = self._ego_perm
        n_obj = self.get_object_obs_size()
        n_goal = self.get_goal_obs_size()

        own_root_pos = root_states[..., 0:3].reshape(-1, 3)                       # (B*M, 3)
        own_heading = torch_utils.calc_heading_quat_inv(root_states[..., 3:7].reshape(-1, 4))
        assigned_states = logical_box_states[:, :M]
        assigned_bps = logical_box_bps[:, :M]

        if self._obs_frame == "owner":
            # Assigned object/goal pairs stay in their owner's frame exactly as in the
            # old M=O checkpoint.  Physical assignment is hidden by logical reordering.
            assigned_object_obs, goal_obs = compute_object_goal_observations(
                own_root_pos, own_heading,
                assigned_states.reshape(-1, assigned_states.shape[-1]),
                assigned_bps.reshape(-1, assigned_bps.shape[-2], 3),
                tar_pos.reshape(-1, 3))

            assigned_block = assigned_object_obs.view(B, M, n_obj)[:, perm]

            # An unassigned object has no owner frame.  Express it in each observer's
            # heading frame: this keeps its local geometry useful without inventing an
            # ownership edge.  It remains a REL_NONE distractor in the transformer.
            if O > M:
                E = O - M
                extra_states = logical_box_states[:, M:]
                extra_bps = logical_box_bps[:, M:]
                obs_pos = own_root_pos.view(B, M, 1, 3).expand(B, M, E, 3).reshape(-1, 3)
                obs_rot = own_heading.view(B, M, 1, 4).expand(B, M, E, 4).reshape(-1, 4)
                row_extra_states = extra_states.unsqueeze(1).expand(B, M, E, 13).reshape(-1, 13)
                row_extra_bps = extra_bps.unsqueeze(1).expand(B, M, E, 8, 3).reshape(-1, 8, 3)
                extra_object_obs, _ = compute_object_goal_observations(
                    obs_pos, obs_rot, row_extra_states, row_extra_bps, row_extra_states[:, 0:3])
                object_tokens = torch.cat(
                    [assigned_block, extra_object_obs.view(B, M, E, n_obj)], dim=2)
            else:
                object_tokens = assigned_block

            object_block = object_tokens.reshape(B, M, O * n_obj)
            goal_block = goal_obs.view(B, M, n_goal)[:, perm].reshape(B, M, M * n_goal)
            return torch.cat([object_block, goal_block], dim=-1)

        # Both remaining modes construct one row per observer. Assigned object slots and
        # goals follow the ego-first permutation; unassigned objects stay at the tail.
        extra_ids = torch.arange(M, O, device=self.device, dtype=torch.long).unsqueeze(0).expand(M, -1)
        object_perm = torch.cat([perm, extra_ids], dim=-1)                       # (M, O)
        row_box_states = logical_box_states[:, object_perm]                     # (B, M, O, 13)
        row_box_bps = logical_box_bps[:, object_perm]                           # (B, M, O, 8, 3)
        row_assigned_states = assigned_states[:, perm]                          # (B, M, M, 13)
        row_tar = tar_pos[:, perm]                                               # (B, M, M, 3)

        if self._obs_frame == "global":
            origins = self._env_origins if env_ids is None else self._env_origins[env_ids]
            object_frame_pos = origins.view(B, 1, 1, 3).expand(B, M, O, 3)
            goal_frame_pos = origins.view(B, 1, 1, 3).expand(B, M, M, 3)
            object_frame_rot = torch.zeros(B, M, O, 4, device=self.device)
            goal_frame_rot = torch.zeros(B, M, M, 4, device=self.device)
            object_frame_rot[..., 3] = 1.0
            goal_frame_rot[..., 3] = 1.0
        else:
            object_frame_pos = own_root_pos.view(B, M, 1, 3).expand(B, M, O, 3)
            goal_frame_pos = own_root_pos.view(B, M, 1, 3).expand(B, M, M, 3)
            object_frame_rot = own_heading.view(B, M, 1, 4).expand(B, M, O, 4)
            goal_frame_rot = own_heading.view(B, M, 1, 4).expand(B, M, M, 4)

        object_obs, _ = compute_object_goal_observations(
            object_frame_pos.reshape(-1, 3), object_frame_rot.reshape(-1, 4),
            row_box_states.reshape(-1, 13), row_box_bps.reshape(-1, 8, 3),
            row_box_states[..., 0:3].reshape(-1, 3))

        _, goal_obs = compute_object_goal_observations(
            goal_frame_pos.reshape(-1, 3), goal_frame_rot.reshape(-1, 4),
            row_assigned_states.reshape(-1, 13),
            assigned_bps[:, perm].reshape(-1, 8, 3), row_tar.reshape(-1, 3))

        return torch.cat([object_obs.view(B, M, O * n_obj),
                          goal_obs.view(B, M, M * n_goal)], dim=-1)

    # ------------------------------------------------------------------ reward / reset

    def _compute_reward(self, actions):
        if self._state_relation:
            return self._compute_relation_reward(compute_agent_collision_penalty)
        N, M, nb = self.num_envs, self.num_agents, self.num_bodies
        B = N * M

        root_pos = self._humanoid_root_states[..., 0:3].reshape(B, 3)
        rigid_body_pos = self._rigid_body_pos.reshape(B, nb, 3)
        assigned_box_states = self._assigned_box_values(self._box_states)
        assigned_box_size = self._assigned_box_values(self._box_size)
        box_pos = assigned_box_states[..., 0:3].reshape(B, 3)
        tar_pos = self._tar_pos.reshape(B, 3)
        box_size = assigned_box_size.reshape(B, 3)
        hands_ids = self._key_body_ids[[0, 1]]

        walk_r = compute_walk_reward(root_pos, self._prev_root_pos.reshape(B, 3), box_pos, self.dt, 1.5,
                                     self._only_vel_reward)
        carry_r = compute_carry_reward(box_pos, self._prev_box_pos.reshape(B, 3), tar_pos, self.dt, 1.5, box_size,
                                       self._only_vel_reward,
                                       self._box_vel_penalty, self._box_vel_pen_coeff, self._box_vel_pen_thre)
        handheld_r = compute_handheld_reward(rigid_body_pos, box_pos, hands_ids, self._only_height_handheld_reward)
        putdown_r = compute_putdown_reward(box_pos, tar_pos)

        reward = walk_r + carry_r + handheld_r + putdown_r
        power_r = torch.zeros_like(reward)
        collision_r = torch.zeros_like(reward)

        if self._power_reward:
            power = torch.abs(torch.multiply(self.dof_force_tensor, self._dof_vel)).sum(dim=-1)  # (N, M)
            power_r = -self._power_coefficient * power.reshape(B)
            reward = reward + power_r

        if self._agent_collision_penalty and M > 1:
            collision_r = -self._agent_collision_coeff * compute_agent_collision_penalty(
                self._humanoid_root_states[..., 0:3], self._agent_collision_dist).reshape(B)
            reward = reward + collision_r

        self.rew_buf[:] = reward
        reward_terms = torch.stack(
            [walk_r, carry_r, handheld_r, putdown_r, power_r, collision_r, reward], dim=-1)
        self.extras["reward_terms"] = reward_terms
        self._reward_term_sums += reward_terms.sum(dim=0)
        self._reward_term_count += B
        return

    def consume_reward_term_means(self):
        if self._reward_term_count == 0:
            return None
        means = self._reward_term_sums / self._reward_term_count
        self._reward_term_sums.zero_()
        self._reward_term_count = 0
        return means

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._prev_root_pos[:] = self._humanoid_root_states[..., 0:3]
        self._prev_box_pos[:] = self._assigned_box_values(self._box_states)[..., 0:3]
        return

    def post_physics_step(self):
        self.progress_buf += 1

        self._refresh_sim_tensors()
        if self._state_relation:
            # Commit history once, then expose exactly that state to PPO/replay.
            self._compute_reward(self.actions)
            self._compute_observations()
        else:
            self._compute_observations()
            self._compute_reward(self.actions)
        self._compute_reset()

        self.extras["terminate"] = self._terminate_buf

        self._update_hist_amp_obs()
        self._compute_amp_observations()

        self.extras["amp_obs"] = self._amp_obs_buf.reshape(self.num_envs * self.num_agents, self.get_num_amp_obs())
        self.extras["policy_obs"] = self.obs_buf.clone()

        if self._is_eval:
            self._compute_metrics_evaluation()
            self.extras["success"] = self._success_buf
            self.extras["precision"] = self._precision_buf

        if self._recording:
            self._capture_video_frame()

        if self.viewer and self.debug_viz:
            self._update_debug_viz()

        return

    def _compute_metrics_evaluation(self):
        if self._state_relation:
            self._success_buf.copy_(self.relation_runtime.done.flatten().long())
            errors = (self._assigned_box_values(self._box_states)[..., :3] - self._tar_pos).norm(dim=-1)
            self._precision_buf.copy_(errors.flatten())
            return
        B = self.num_envs * self.num_agents
        assigned_box_pos = self._assigned_box_values(self._box_states)[..., 0:3]
        pos_err = torch.norm(self._tar_pos.reshape(B, 3) - assigned_box_pos.reshape(B, 3), p=2, dim=-1)
        dist_mask = pos_err <= self._success_threshold
        self._success_buf[dist_mask] += 1
        self._precision_buf[dist_mask] = torch.where(pos_err[dist_mask] < self._precision_buf[dist_mask],
                                                     pos_err[dist_mask], self._precision_buf[dist_mask])
        return

    def _video_focus_points(self):
        # Only the characters. Including the boxes pushes the camera back and, worse, tends
        # to park a box between the camera and the humanoid it belongs to.
        if self._video_focus == "agent0":
            return self._humanoid_root_states[0, 0:1, 0:3]
        return self._humanoid_root_states[0, :, 0:3]

    # ------------------------------------------------------------------ resets

    def _reset_envs(self, env_ids):
        if self._state_relation:
            pending = env_ids
            for attempt in range(16):
                if len(pending) == 0:
                    return
                self._reset_default_slots = None
                self._reset_ref_slots = {}
                self._reset_ref_motion_ids = {}
                self._reset_ref_motion_times = {}
                self._reset_actors(pending)
                self._sample_box_assignments(pending)
                self._reset_boxes(pending)
                self._reset_task(pending)
                self._reset_env_tensors(pending)
                self._refresh_sim_tensors()
                self._reset_relation_history(pending)
                self._compute_observations(pending)
                self._init_amp_obs(pending)
                pending = pending[self.relation_runtime.done[pending].all(-1)]
            if len(pending):
                raise RuntimeError('All-subgoal-success reset persisted after 16 resamples')
            return
        self._reset_default_slots = None
        self._reset_ref_slots = {}
        self._reset_ref_motion_ids = {}
        self._reset_ref_motion_times = {}

        if (len(env_ids) > 0):
            self._reset_actors(env_ids)
            self._sample_box_assignments(env_ids)
            self._reset_boxes(env_ids)
            self._reset_task(env_ids)
            self._reset_env_tensors(env_ids)
            self._refresh_sim_tensors()
            self._compute_observations(env_ids)
            self._init_amp_obs(env_ids)

        return

    def _reset_actors(self, env_ids):
        slot_env, slot_agent = self._expand_slots(env_ids)

        if (self._state_init == HumanoidMACarry.StateInit.Default):
            self._reset_default(slot_env, slot_agent)
            self._reset_default_slots = (slot_env, slot_agent)
        elif (self._state_init == HumanoidMACarry.StateInit.Start
              or self._state_init == HumanoidMACarry.StateInit.Random):
            self._reset_ref_state_init(slot_env, slot_agent)
        elif (self._state_init == HumanoidMACarry.StateInit.Hybrid):
            probs = to_torch(np.array([self._hybrid_init_prob] * slot_env.shape[0]), device=self.device)
            ref_mask = torch.bernoulli(probs) == 1.0

            if ref_mask.sum() > 0:
                self._reset_ref_state_init(slot_env[ref_mask], slot_agent[ref_mask])
            rest = torch.logical_not(ref_mask)
            if rest.sum() > 0:
                self._reset_default(slot_env[rest], slot_agent[rest])
                self._reset_default_slots = (slot_env[rest], slot_agent[rest])
        else:
            assert (False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))
        return

    def _reset_default(self, slot_env, slot_agent):
        super()._reset_default(slot_env, slot_agent)
        self._kinematic_humanoid_rigid_body_states[slot_env, slot_agent] = \
            self._initial_humanoid_rigid_body_states[slot_env, slot_agent]
        self._every_env_init_dof_pos[slot_env, slot_agent] = self._initial_dof_pos[slot_env, slot_agent]
        return

    def _reset_ref_state_init(self, slot_env, slot_agent):
        num_slots = slot_env.shape[0]
        sk_ids = torch.multinomial(self._skill_init_prob, num_samples=num_slots, replacement=True)

        for uid, sk_name in enumerate(self._skill):
            curr_motion_lib = self._motion_lib[sk_name]
            sel = (sk_ids == uid).nonzero().squeeze(-1)
            if len(sel) == 0:
                continue

            curr_env = slot_env[sel]
            curr_agent = slot_agent[sel]
            offset = self._agent_spawn_offsets[curr_agent] + self._env_origins[curr_env]

            motion_ids = curr_motion_lib.sample_motions(len(sel))

            if (self._state_init == HumanoidMACarry.StateInit.Random
                    or self._state_init == HumanoidMACarry.StateInit.Hybrid):
                motion_times = curr_motion_lib.sample_time_rsi(motion_ids)
            elif (self._state_init == HumanoidMACarry.StateInit.Start):
                motion_times = torch.zeros(len(sel), device=self.device)
            else:
                assert (False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
                = curr_motion_lib.get_motion_state(motion_ids, motion_times)

            self._set_env_state(curr_env, curr_agent,
                                root_pos=root_pos + offset,
                                root_rot=root_rot,
                                dof_pos=dof_pos,
                                root_vel=root_vel,
                                root_ang_vel=root_ang_vel,
                                dof_vel=dof_vel)

            self._reset_ref_slots[sk_name] = (curr_env, curr_agent)
            self._reset_ref_motion_ids[sk_name] = motion_ids
            self._reset_ref_motion_times[sk_name] = motion_times

            body_pos, body_rot, body_vel, body_ang_vel \
                = curr_motion_lib.get_motion_state_max(motion_ids, motion_times)
            body_pos = body_pos + offset.unsqueeze(-2)
            self._kinematic_humanoid_rigid_body_states[curr_env, curr_agent] = \
                torch.cat((body_pos, body_rot, body_vel, body_ang_vel), dim=-1)

            self._every_env_init_dof_pos[curr_env, curr_agent] = dof_pos

        return

    def _set_env_state(self, slot_env, slot_agent, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self._humanoid_root_states[slot_env, slot_agent, 0:3] = root_pos
        self._humanoid_root_states[slot_env, slot_agent, 3:7] = root_rot
        self._humanoid_root_states[slot_env, slot_agent, 7:10] = root_vel
        self._humanoid_root_states[slot_env, slot_agent, 10:13] = root_ang_vel

        self._dof_pos[slot_env, slot_agent] = dof_pos
        self._dof_vel[slot_env, slot_agent] = dof_vel
        return

    def _collect_random_slots(self, skills):
        """Slots whose box / target must be sampled instead of taken from a motion clip."""
        envs, agents = [], []
        if self._reset_default_slots is not None:
            envs.append(self._reset_default_slots[0])
            agents.append(self._reset_default_slots[1])
        for sk_name in skills:
            if self._reset_ref_slots.get(sk_name) is not None:
                envs.append(self._reset_ref_slots[sk_name][0])
                agents.append(self._reset_ref_slots[sk_name][1])
        if len(envs) == 0:
            return None
        return torch.cat(envs, dim=0), torch.cat(agents, dim=0)

    def _sample_arena_xy(self, centers):
        """Uniform samples by area inside the configured arena disc."""
        count = centers.shape[0]
        angle = torch.rand(count, device=self.device) * (2.0 * np.pi)
        radius = torch.sqrt(torch.rand(count, device=self.device)) * self._box_spawn_radius
        offset = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1) * radius.unsqueeze(-1)
        return centers + offset

    def _reset_all_boxes_random_arena(self, env_ids):
        """Inference reset: every physical box is independent of every agent owner."""
        K, O = env_ids.shape[0], self.num_objects
        centers = self._env_origins[env_ids, 0:2]
        agent_xy = self._humanoid_root_states[env_ids, :, 0:2]
        positions = torch.zeros(K, O, 2, device=self.device)

        for object_id in range(O):
            candidate = self._sample_arena_xy(centers)
            for _ in range(64):
                near_agent = torch.sum((candidate.unsqueeze(1) - agent_xy) ** 2, dim=-1).min(dim=-1).values \
                    < self._box_min_agent_dist ** 2
                if object_id > 0:
                    near_box = torch.sum((candidate.unsqueeze(1) - positions[:, :object_id]) ** 2, dim=-1) \
                        .min(dim=-1).values < self._box_min_box_dist ** 2
                    invalid = torch.logical_or(near_agent, near_box)
                else:
                    invalid = near_agent
                if not torch.any(invalid):
                    break
                candidate[invalid] = self._sample_arena_xy(centers[invalid])
            positions[:, object_id] = candidate

        self._box_states[env_ids, :, 0:2] = positions
        self._box_states[env_ids, :, 2] = self._box_size[env_ids, :, 2] / 2.0

        if self._reset_random_height:
            M = self.num_agents
            batch = torch.arange(K, device=self.device).unsqueeze(-1)
            world_env = env_ids.unsqueeze(-1).expand(-1, M)
            assigned = self._agent_box_assignment[env_ids]
            assigned_size = self._box_size[world_env, assigned]
            assigned_z = assigned_size[..., 2] / 2.0
            height_mask = torch.rand(K, M, device=self.device) < self._reset_random_height_prob
            height_offset = torch.zeros_like(assigned_z)
            height_offset[height_mask] = torch.rand(height_mask.sum(), device=self.device)
            height_mask &= height_offset >= self._reset_min_platform_height
            if torch.any(height_mask):
                assigned_z[height_mask] += height_offset[height_mask]
                assigned_z[height_mask] = self._regulate_height(
                    assigned_z[height_mask], assigned_size[height_mask])

            assigned_xy = positions[batch, assigned]
            self._platform_pos[env_ids] = self._platform_default_pos[env_ids]
            if torch.any(height_mask):
                agent_ids = torch.arange(M, device=self.device).unsqueeze(0).expand(K, -1)
                active_env = world_env[height_mask]
                active_agent = agent_ids[height_mask]
                self._platform_pos[active_env, active_agent, 0:2] = assigned_xy[height_mask]
                self._platform_pos[active_env, active_agent, 2] = \
                    assigned_z[height_mask] - assigned_size[..., 2][height_mask] / 2.0 \
                    - self._platform_height / 2.0
            self._box_states[world_env, assigned, 2] = assigned_z + 0.05

        axis = torch.zeros(K * O, 3, device=self.device)
        axis[:, 2] = 1.0
        coeff = 1.0 if self._reset_random_rot else 0.0
        angle = torch.rand(K * O, device=self.device) * (2.0 * np.pi) * coeff
        self._box_states[env_ids, :, 3:7] = quat_from_angle_axis(angle, axis).view(K, O, 4)
        self._box_states[env_ids, :, 7:13] = 0.0
        return

    def _reset_unassigned_boxes_random_arena(self, env_ids):
        """Give training-time distractors a valid pose without changing assigned resets."""
        K, M, O = env_ids.shape[0], self.num_agents, self.num_objects
        if O == M:
            return

        assignments = self._agent_box_assignment[env_ids]
        unassigned = torch.ones(K, O, device=self.device, dtype=torch.bool)
        unassigned.scatter_(1, assignments, False)

        centers = self._env_origins[env_ids, 0:2]
        agent_xy = self._humanoid_root_states[env_ids, :, 0:2]
        assigned_xy = self._assigned_box_values(self._box_states, env_ids)[..., 0:2]
        occupied = torch.zeros(K, M + O, 2, device=self.device)
        occupied_valid = torch.zeros(K, M + O, device=self.device, dtype=torch.bool)
        occupied[:, :M] = assigned_xy
        occupied_valid[:, :M] = True

        for object_id in range(O):
            active = unassigned[:, object_id]
            if not torch.any(active):
                continue
            candidate = self._sample_arena_xy(centers)
            for _ in range(64):
                near_agent = torch.sum((candidate.unsqueeze(1) - agent_xy) ** 2, dim=-1).min(dim=-1).values \
                    < self._box_min_agent_dist ** 2
                dist2 = torch.sum((candidate.unsqueeze(1) - occupied) ** 2, dim=-1)
                near_box = torch.logical_and(dist2 < self._box_min_box_dist ** 2, occupied_valid).any(dim=-1)
                invalid = torch.logical_and(active, torch.logical_or(near_agent, near_box))
                if not torch.any(invalid):
                    break
                candidate[invalid] = self._sample_arena_xy(centers[invalid])

            active_env = torch.arange(K, device=self.device)[active]
            physical_ids = torch.full((active_env.shape[0],), object_id,
                                      device=self.device, dtype=torch.long)
            world_env = env_ids[active]
            self._box_states[world_env, physical_ids, 0:2] = candidate[active]
            self._box_states[world_env, physical_ids, 2] = \
                self._box_size[world_env, physical_ids, 2] / 2.0

            axis = torch.zeros(active_env.shape[0], 3, device=self.device)
            axis[:, 2] = 1.0
            coeff = 1.0 if self._reset_random_rot else 0.0
            angle = torch.rand(active_env.shape[0], device=self.device) * (2.0 * np.pi) * coeff
            self._box_states[world_env, physical_ids, 3:7] = quat_from_angle_axis(angle, axis)
            self._box_states[world_env, physical_ids, 7:13] = 0.0

            occupied[active, M + object_id] = candidate[active]
            occupied_valid[active, M + object_id] = True
        return

    def _reset_boxes(self, env_ids):
        if self._random_arena_box_spawn:
            self._reset_all_boxes_random_arena(env_ids)
            return

        if self._reset_random_height:
            self._platform_pos[env_ids] = self._platform_default_pos[env_ids]

        # boxes that come from a reference motion
        for sk_name in ["pickUp", "carryWith", "putDown"]:
            if self._reset_ref_slots.get(sk_name) is None:
                continue

            curr_env, curr_agent = self._reset_ref_slots[sk_name]
            offset = self._agent_spawn_offsets[curr_agent] + self._env_origins[curr_env]

            root_pos, root_rot = self._motion_lib[sk_name].get_obj_motion_state(
                motion_ids=self._reset_ref_motion_ids[sk_name],
                motion_times=self._reset_ref_motion_times[sk_name])
            root_pos = root_pos + offset

            curr_box = self._agent_box_assignment[curr_env, curr_agent]
            box_size = self._box_size[curr_env, curr_box]
            on_ground = (box_size[:, 2] / 2 > root_pos[:, 2])
            root_pos[on_ground, 2] = box_size[on_ground, 2] / 2

            self._box_states[curr_env, curr_box, 0:3] = root_pos
            self._box_states[curr_env, curr_box, 3:7] = root_rot
            self._box_states[curr_env, curr_box, 7:13] = 0.0

            if self._reset_random_height:
                self._platform_pos[curr_env, curr_agent] = \
                    self._platform_default_pos[curr_env, curr_agent]

        # boxes that are randomly placed around their owner
        random_slots = self._collect_random_slots(["loco"])
        if random_slots is not None:
            curr_env, curr_agent = random_slots
            K = curr_env.shape[0]
            curr_box = self._agent_box_assignment[curr_env, curr_agent]

            root_pos_xy = torch.randn(K, 2, device=self.device)
            root_pos_xy /= torch.linalg.norm(root_pos_xy, dim=-1, keepdim=True)
            root_pos_xy *= torch.rand(K, 1, device=self.device) * (self._tar_reach - 1.0) + 1.0
            root_pos_xy += self._humanoid_root_states[curr_env, curr_agent, :2]

            root_pos_z = self._box_size[curr_env, curr_box, 2] / 2
            if self._reset_random_height:
                height_mask = torch.rand(K, device=self.device) < self._reset_random_height_prob
                height_offset = torch.zeros_like(root_pos_z)
                height_offset[height_mask] = torch.rand(height_mask.sum(), device=self.device)
                height_mask &= height_offset >= self._reset_min_platform_height
                if height_mask.sum() > 0:
                    root_pos_z[height_mask] += height_offset[height_mask]
                    root_pos_z[height_mask] = self._regulate_height(
                        root_pos_z[height_mask], self._box_size[curr_env[height_mask], curr_box[height_mask]])

            axis = torch.tensor([[0.0, 0.0, 1.0]], device=self.device).reshape(1, 3).expand([K, -1])
            coeff = 1.0 if self._reset_random_rot else 0.0
            ang = torch.rand((K,), device=self.device) * 2 * np.pi * coeff
            root_rot = quat_from_angle_axis(ang, axis)
            root_pos = torch.cat([root_pos_xy, root_pos_z.unsqueeze(-1)], dim=-1)

            self._box_states[curr_env, curr_box, 0:3] = root_pos
            self._box_states[curr_env, curr_box, 3:7] = root_rot
            self._box_states[curr_env, curr_box, 7:13] = 0.0

            if self._reset_random_height:
                if height_mask.sum() > 0:
                    active_env = curr_env[height_mask]
                    active_agent = curr_agent[height_mask]
                    active_box = curr_box[height_mask]
                    self._platform_pos[active_env, active_agent, 0:2] = root_pos[height_mask, 0:2]
                    self._platform_pos[active_env, active_agent, 2] = \
                        root_pos[height_mask, 2] - self._box_size[active_env, active_box, 2] / 2 \
                        - self._platform_height / 2
                self._box_states[curr_env, curr_box, 2] += 0.05

        self._reset_unassigned_boxes_random_arena(env_ids)

        return

    def _reset_task(self, env_ids):
        if self._reset_random_height:
            self._tar_platform_pos[env_ids] = self._tar_platform_default_pos[env_ids]

        # target taken from the last frame of a putDown clip
        for sk_name in ["putDown"]:
            if self._reset_ref_slots.get(sk_name) is None:
                continue

            curr_env, curr_agent = self._reset_ref_slots[sk_name]
            curr_box = self._agent_box_assignment[curr_env, curr_agent]
            offset = self._agent_spawn_offsets[curr_agent] + self._env_origins[curr_env]

            root_pos, root_rot = self._motion_lib[sk_name].get_obj_motion_state(
                motion_ids=self._reset_ref_motion_ids[sk_name],
                motion_times=self._motion_lib[sk_name].get_motion_length(self._reset_ref_motion_ids[sk_name]))
            root_pos = root_pos + offset
            root_pos[:, 2] = self._box_size[curr_env, curr_box, 2] / 2

            self._tar_pos[curr_env, curr_agent] = root_pos
            if self._reset_random_height:
                self._tar_platform_pos[curr_env, curr_agent] = \
                    self._tar_platform_default_pos[curr_env, curr_agent]

        # randomly sampled targets, inside the owner's own disc
        random_slots = self._collect_random_slots(["loco", "pickUp", "carryWith"])
        if random_slots is not None:
            curr_env, curr_agent = random_slots
            K = curr_env.shape[0]
            curr_box = self._agent_box_assignment[curr_env, curr_agent]
            center = self._agent_spawn_offsets[curr_agent][:, :2] + self._env_origins[curr_env, :2]
            min_dist = 1.0

            def sample(n):
                xy = (torch.rand(n, 2, device=self.device) * 2.0 - 1.0) * self._tar_reach
                return xy

            new_xy = sample(K) + center
            new_z = self._box_size[curr_env, curr_box, 2] / 2
            new_target_pos = torch.cat([new_xy, new_z.unsqueeze(-1)], dim=-1)

            overlap = torch.logical_or(
                torch.sum((new_target_pos[..., :2] - self._humanoid_root_states[curr_env, curr_agent, :2]) ** 2, dim=-1) < min_dist,
                torch.sum((new_target_pos[..., :2] - self._box_states[curr_env, curr_box, :2]) ** 2, dim=-1) < min_dist)

            tries = 0
            while torch.sum(overlap) > 0 and tries < 32:
                n = int(torch.sum(overlap))
                new_target_pos[overlap, :2] = sample(n) + center[overlap]
                overlap = torch.logical_or(
                    torch.sum((new_target_pos[..., :2] - self._humanoid_root_states[curr_env, curr_agent, :2]) ** 2, dim=-1) < min_dist,
                    torch.sum((new_target_pos[..., :2] - self._box_states[curr_env, curr_box, :2]) ** 2, dim=-1) < min_dist)
                tries += 1

            if self._reset_random_height:
                height_mask = torch.rand(K, device=self.device) < self._reset_random_height_prob
                height_offset = torch.zeros_like(new_target_pos[:, 2])
                height_offset[height_mask] = torch.rand(height_mask.sum(), device=self.device)
                height_mask &= height_offset >= self._reset_min_platform_height
                if height_mask.sum() > 0:
                    new_target_pos[height_mask, 2] += height_offset[height_mask]
                    new_target_pos[height_mask, 2] = self._regulate_height(
                        new_target_pos[height_mask, 2],
                        self._box_size[curr_env[height_mask], curr_box[height_mask]])

            self._tar_pos[curr_env, curr_agent] = new_target_pos
            if self._reset_random_height:
                if height_mask.sum() > 0:
                    active_env = curr_env[height_mask]
                    active_agent = curr_agent[height_mask]
                    active_box = curr_box[height_mask]
                    self._tar_platform_pos[active_env, active_agent, 0:2] = \
                        new_target_pos[height_mask, 0:2]
                    self._tar_platform_pos[active_env, active_agent, 2] = \
                        new_target_pos[height_mask, 2] - self._box_size[active_env, active_box, 2] / 2 \
                        - self._platform_height / 2

        return

    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)

        if self._is_eval:
            flat = self._flat_slot_ids(env_ids)
            self._success_buf[flat] = 0
            self._precision_buf[flat] = float('Inf')

        ids = [self._box_actor_ids[env_ids].contiguous().view(-1)]
        if self._enable_markers:
            self._marker_pos[env_ids] = self._tar_pos[env_ids]
            ids.append(self._marker_actor_ids[env_ids].contiguous().view(-1))
        if self._reset_random_height:
            self._platform_states[env_ids, :, 3:6] = 0.0
            self._platform_states[env_ids, :, 6] = 1.0
            self._platform_states[env_ids, :, 7:13] = 0.0
            self._tar_platform_states[env_ids, :, 3:6] = 0.0
            self._tar_platform_states[env_ids, :, 6] = 1.0
            self._tar_platform_states[env_ids, :, 7:13] = 0.0
            ids.extend([
                self._platform_actor_ids[env_ids].contiguous().view(-1),
                self._tar_platform_actor_ids[env_ids].contiguous().view(-1),
            ])
        ids = torch.cat(ids)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(ids), len(ids))
        return

    # ------------------------------------------------------------------ AMP

    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)

        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        num_key_bodies = len(key_bodies)

        if (asset_file == "mjcf/amp_humanoid.xml"):
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + 28 + 3 * num_key_bodies
        elif (asset_file in ("mjcf/phys_humanoid.xml", "mjcf/phys_humanoid_v2.xml", "mjcf/phys_humanoid_v3.xml")):
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + 28 + 2 * 2 + 3 * num_key_bodies
        else:
            print("Unsupported character config file: {s}".format(asset_file))
            assert (False)
        return

    def get_num_amp_obs(self):
        return self._num_amp_obs_steps * self._num_amp_obs_per_step

    def _load_motion(self, motion_file):
        assert (self._dof_offsets[-1] == self.num_dof)

        ext = os.path.splitext(motion_file)[1]
        if (ext == ".yaml"):
            with open(os.path.join(os.getcwd(), motion_file), 'r') as f:
                motion_config = yaml.load(f, Loader=yaml.SafeLoader)

            self._skill_categories = list(motion_config['motions'].keys())
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

    def fetch_amp_obs_demo(self, num_samples):
        sk_ids = torch.multinomial(self._skill_disc_prob, num_samples=num_samples, replacement=True)

        if self._amp_obs_demo_buf is None:
            self._build_amp_obs_demo_buf(num_samples)
        else:
            assert (self._amp_obs_demo_buf.shape[0] == num_samples)

        motion_ids = []
        motion_times0 = []
        for uid, sk_name in enumerate(self._skill):
            curr_num = (sk_ids == uid).sum().item()
            if curr_num == 0:
                continue
            curr_motion_ids = self._motion_lib[sk_name].sample_motions(curr_num)
            truncate_time = self.dt * (self._num_amp_obs_steps - 1)
            curr_motion_times0 = self._motion_lib[sk_name].sample_time(
                curr_motion_ids, truncate_time=truncate_time)
            curr_motion_times0 += truncate_time
            curr_demo = self.build_amp_obs_demo(curr_motion_ids, curr_motion_times0, self._motion_lib[sk_name])
            motion_ids.append(curr_demo)

        amp_obs_demo = torch.cat(motion_ids, dim=0)
        self._amp_obs_demo_buf[:] = amp_obs_demo.view(self._amp_obs_demo_buf.shape)
        return self._amp_obs_demo_buf.view(-1, self.get_num_amp_obs())

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
        self._amp_obs_demo_buf = torch.zeros((num_samples, self._num_amp_obs_steps, self._num_amp_obs_per_step),
                                             device=self.device, dtype=torch.float32)
        return

    def _init_amp_obs(self, env_ids):
        self._compute_amp_observations(env_ids)

        if self._reset_default_slots is not None:
            self._init_amp_obs_default(*self._reset_default_slots)

        for sk_name in self._skill:
            if self._reset_ref_slots.get(sk_name) is not None:
                self._init_amp_obs_ref(self._reset_ref_slots[sk_name],
                                       self._reset_ref_motion_ids[sk_name],
                                       self._reset_ref_motion_times[sk_name], sk_name)
        return

    def _init_amp_obs_default(self, slot_env, slot_agent):
        curr = self._curr_amp_obs_buf[slot_env, slot_agent].unsqueeze(-2)
        self._hist_amp_obs_buf[slot_env, slot_agent] = curr
        return

    def _init_amp_obs_ref(self, slots, motion_ids, motion_times, skill_name):
        slot_env, slot_agent = slots
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
        self._hist_amp_obs_buf[slot_env, slot_agent] = \
            amp_obs_demo.view(slot_env.shape[0], self._num_amp_obs_steps - 1, self._num_amp_obs_per_step)
        return

    def _update_hist_amp_obs(self, env_ids=None):
        for i in reversed(range(self._amp_obs_buf.shape[2] - 1)):
            self._amp_obs_buf[:, :, i + 1] = self._amp_obs_buf[:, :, i]
        return

    def _compute_amp_observations(self, env_ids=None):
        N, M, nb, nd = self.num_envs, self.num_agents, self.num_bodies, self.num_dof

        if (env_ids is None):
            B = N * M
            body_pos = self._rigid_body_pos.reshape(B, nb, 3)
            body_rot = self._rigid_body_rot.reshape(B, nb, 4)
            body_vel = self._rigid_body_vel.reshape(B, nb, 3)
            body_ang_vel = self._rigid_body_ang_vel.reshape(B, nb, 3)
            dof_pos = self._dof_pos.reshape(B, nd)
            dof_vel = self._dof_vel.reshape(B, nd)

            obs = build_amp_observations(body_pos[:, 0, :], body_rot[:, 0, :], body_vel[:, 0, :],
                                         body_ang_vel[:, 0, :], dof_pos, dof_vel,
                                         body_pos[:, self._key_body_ids, :],
                                         self._local_root_obs, self._root_height_obs,
                                         self._dof_obs_size, self._dof_offsets)
            self._curr_amp_obs_buf[:] = obs.view(N, M, self._num_amp_obs_per_step)
        else:
            K = env_ids.shape[0]
            B = K * M
            kin = self._kinematic_humanoid_rigid_body_states[env_ids].reshape(B, nb, 13)
            dof_pos = self._dof_pos[env_ids].reshape(B, nd)
            dof_vel = self._dof_vel[env_ids].reshape(B, nd)

            obs = build_amp_observations(kin[:, 0, 0:3], kin[:, 0, 3:7], kin[:, 0, 7:10], kin[:, 0, 10:13],
                                         dof_pos, dof_vel, kin[:, self._key_body_ids, 0:3],
                                         self._local_root_obs, self._root_height_obs,
                                         self._dof_obs_size, self._dof_offsets)
            self._curr_amp_obs_buf[env_ids] = obs.view(K, M, self._num_amp_obs_per_step)
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
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1],
                                           local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
                                               heading_rot_expand.shape[2])
    local_end_pos = quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0],
                                            local_key_body_pos.shape[1] * local_key_body_pos.shape[2])

    dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets)

    obs = torch.cat((root_h_obs, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel,
                     flat_local_key_pos), dim=-1)
    return obs


@torch.jit.script
def compute_object_goal_observations(ego_root_pos, ego_heading_rot, box_states, box_bps, tar_pos):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    """Object and goal tokens of one box, in an arbitrary observer's heading frame."""
    box_pos = box_states[:, 0:3]
    box_rot = box_states[:, 3:7]
    box_vel = box_states[:, 7:10]
    box_ang_vel = box_states[:, 10:13]

    local_box_pos = quat_rotate(ego_heading_rot, box_pos - ego_root_pos)
    local_box_rot = quat_mul(ego_heading_rot, box_rot)
    local_box_rot_obs = torch_utils.quat_to_tan_norm(local_box_rot)
    local_box_vel = quat_rotate(ego_heading_rot, box_vel)
    local_box_ang_vel = quat_rotate(ego_heading_rot, box_ang_vel)

    num_bps = box_bps.shape[1]
    box_pos_exp = torch.broadcast_to(box_pos.unsqueeze(-2), (box_pos.shape[0], num_bps, 3))
    box_rot_exp = torch.broadcast_to(box_rot.unsqueeze(-2), (box_rot.shape[0], num_bps, 4))
    box_bps_world = quat_rotate(box_rot_exp.reshape(-1, 4), box_bps.reshape(-1, 3)) + box_pos_exp.reshape(-1, 3)

    heading_exp = torch.broadcast_to(ego_heading_rot.unsqueeze(-2), (ego_heading_rot.shape[0], num_bps, 4))
    ego_pos_exp = torch.broadcast_to(ego_root_pos.unsqueeze(-2), (ego_root_pos.shape[0], num_bps, 3))
    box_bps_local = quat_rotate(heading_exp.reshape(-1, 4), box_bps_world - ego_pos_exp.reshape(-1, 3))
    box_bps_local = box_bps_local.reshape(box_pos.shape[0], num_bps * 3)

    object_obs = torch.cat([local_box_vel, local_box_ang_vel, local_box_pos, local_box_rot_obs, box_bps_local], dim=-1)

    # goal is given twice: where the target is w.r.t. the observer, and where it is
    # w.r.t. the box the observer has to move (the actually actionable quantity).
    # The second one reuses local_box_pos instead of a second quat_rotate: the rotation
    # is linear, so R(tar - box) == R(tar - ego) - R(box - ego), and the two-rotate form
    # trips a fuser bug in the legacy TorchScript executor that IsaacGym switches on.
    local_tar_pos = quat_rotate(ego_heading_rot, tar_pos - ego_root_pos)
    local_box2tar = local_tar_pos - local_box_pos
    goal_obs = torch.cat([local_tar_pos, local_box2tar], dim=-1)

    return object_obs, goal_obs


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
def compute_carry_reward(box_pos, prev_box_pos, tar_box_pos, dt, tar_vel, box_size, only_vel_reward,
                         box_vel_penalty, box_vel_pen_coeff, box_vel_penalty_thre):
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


@torch.jit.script
def compute_agent_collision_penalty(root_pos, min_dist):
    # type: (Tensor, float) -> Tensor
    """Soft penalty on humanoids standing on top of each other. root_pos: (N, M, 3)."""
    diff = root_pos.unsqueeze(2) - root_pos.unsqueeze(1)                       # (N, M, M, 3)
    dist = torch.norm(diff[..., 0:2], p=2, dim=-1)                             # (N, M, M)
    M = root_pos.shape[1]
    eye = torch.eye(M, device=root_pos.device, dtype=torch.bool).unsqueeze(0)
    dist = torch.where(eye, torch.full_like(dist, 1e6), dist)
    violation = torch.clamp_min(min_dist - dist, 0.0) / min_dist
    return violation.max(dim=-1)[0]                                            # (N, M)
