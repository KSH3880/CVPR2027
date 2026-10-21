import math
import os
import yaml
from enum import Enum
import numpy as np
import torch
from isaacgym import gymapi
from isaacgym import gymtorch
from env.tasks.multi_agent.humanoid_ma import HumanoidMA
from env.tasks.multi_agent.scene_features import (
    build_gta_pose_records,
    scenario_neutral_targets,
)
from utils.relation_task_spec import validate_relation_config
from env.tasks.humanoid import dof_to_obs
from utils.motion_lib import MotionLib
from isaacgym.torch_utils import quat_rotate, quat_mul, to_torch, quat_from_angle_axis
from utils import torch_utils
from env.tasks.multi_agent.collision_reward import (
    agent_collision_config,
    compute_agent_cpa_collision_penalty,
)
from env.tasks.multi_agent.edge_context_task import EdgeContextTaskMixin
from utils.edge_context_spec import at_goal_marker_slots
from utils.edge_ontop_spec import batched
from utils.edge_stage1_spec import (
    compile_stage1_graph,
    ground_standalone_interaction_targets,
    max_stage1_stack_height,
)
from utils.edge_scenario_spec import (
    PRESETS as SCENARIO_PRESETS,
    classify_templates,
    agent_object_indices,
    agent_goal_indices,
    scenario_templates,
)
from env.tasks.multi_agent.edge_ontop_task import SampledOnTopTaskMixin


class HumanoidMACarry(SampledOnTopTaskMixin, EdgeContextTaskMixin, HumanoidMA):
    _PLATFORM_COLLISION_FILTER = 1 << 30
    REWARD_TERM_NAMES = (
        "walk",
        "carry",
        "handheld",
        "putdown",
        "power",
        "collision",
        "total",
    )

    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    def __init__(
        self, cfg, sim_params, physics_engine, device_type, device_id, headless
    ):
        self._shared_support = cfg["env"].get("sharedSupport") is not None
        self._stage2 = cfg["env"].get("stage2") is not None or self._shared_support
        self._relation_cfg = cfg["env"]["relationReward"]
        validate_relation_config(self._relation_cfg)
        from utils.unified_training import validate_unified_env

        validate_unified_env(cfg["env"])
        self._amp_task_conditioning = True
        self._edge_stage1 = self._edge_interaction = self._edge_ontop = (
            self._edge_context
        ) = True
        self._state_relation = True
        self._size_aware_rsi = True
        self._policy_task_roles = True
        self._scenario_no_climb = self._scenario_with_climb = True
        self._semantic_only_stage1 = True
        default_preset = (
            "holding_at" if cfg["args"].test or cfg["args"].eval else "random_scenario"
        )
        from utils.task_role_spec import task_preset

        self._task_graph_preset = task_preset(
            getattr(cfg["args"], "task_graph", "") or default_preset
        )
        self._task_role_swap = bool(getattr(cfg["args"], "task_role_swap", False))
        allowed_presets = SCENARIO_PRESETS
        if self._stage2:
            from utils.stage2_spec import PRESETS
            allowed_presets = PRESETS
        if self._shared_support:
            from utils.shared_support import PRESETS
            allowed_presets = PRESETS
        if cfg["env"].get("pushExtension") is not None:
            allowed_presets = allowed_presets + ("push_at",)
        if self._task_graph_preset not in allowed_presets or getattr(
            cfg["args"], "task_camera", "stack"
        ) not in ("stack", "agent"):
            raise ValueError("Unknown Stage1 graph preset or camera mode")
        if not (cfg["args"].test or cfg["args"].eval):
            expected_shape = (4, 8) if self._stage2 else (2, 4)
            if self._shared_support:
                expected_shape = (4, cfg["env"]["numObjects"])
            if self._task_graph_preset != "random_scenario" or (
                cfg["env"]["numAgents"],
                cfg["env"]["numObjects"],
            ) != expected_shape:
                raise ValueError(
                    "Training requires random_scenario and the configured experiment entity counts"
                )
        self._relation_graph_spec = cfg["env"]["relationGraph"]
        self._template_rsi = cfg["env"]["templateRsi"]
        num_agents = int(cfg["env"].get("numAgents", 1))
        configured_objects = int(cfg["env"].get("numObjects", 0))
        self.num_objects = num_agents if configured_objects <= 0 else configured_objects
        self.num_physical_objects = 12 if self._stage2 else self.num_objects
        if self._shared_support:
            from utils.shared_support import capacities
            self.num_physical_objects = capacities(cfg["env"])[1]
        assert (
            self.num_objects >= num_agents
        ), "numObjects ({}) must be >= numAgents ({})".format(
            self.num_objects, num_agents
        )
        self._generic_scenario_eval = (not self._stage2) and cfg["args"].eval and (
            num_agents,
            self.num_objects,
        ) != (2, 4)
        if self._generic_scenario_eval:
            radius = float(cfg["env"].get("agentSpawnRadius", 2.5))
            if num_agents > 2:
                radius = max(radius, 2.5 / np.sin(np.pi / num_agents))
            cfg["env"]["agentSpawnRadius"] = radius
            cfg["env"]["envSpacing"] = max(
                float(cfg["env"]["envSpacing"]),
                radius + 3.0,
                0.5 + 4.5 * np.sqrt(self.num_objects / 4.0),
            )
        compiler = compile_stage1_graph
        compiler(self._relation_graph_spec, num_agents, self.num_objects)
        self._context_suffix_width = 5 * num_agents
        self.REWARD_TERM_NAMES = (
            "edge_state",
            "edge_progress",
            "edge_success",
            "power",
            "collision",
            "box_speed",
            "total",
        )
        self._enable_task_obs = cfg["env"]["enableTaskObs"]
        self._only_vel_reward = cfg["env"]["onlyVelReward"]
        self._only_height_handheld_reward = cfg["env"]["onlyHeightHandHeldReward"]
        self._box_vel_penalty = cfg["env"]["box_vel_penalty"]
        self._box_vel_pen_coeff = cfg["env"]["box_vel_pen_coeff"]
        self._box_vel_pen_thre = cfg["env"]["box_vel_pen_threshold"]
        self._agent_collision_config = agent_collision_config(cfg["env"])
        self._agent_collision_penalty = self._agent_collision_config["enabled"]
        self._agent_collision_coeff = self._agent_collision_config["coefficient"]
        self._agent_collision_dist = self._agent_collision_config["distance"]
        self._mode = cfg["env"]["mode"]
        assert self._mode in ["train", "test"]
        if cfg["args"].eval:
            self._mode = "test"
        box_cfg = cfg["env"]["box"]
        self._size_aware_rsi = True
        self._build_base_size = box_cfg["build"]["baseSize"]
        self._build_random_size = box_cfg["build"]["randomSize"]
        self._build_random_mode_equal_proportion = box_cfg["build"][
            "randomModeEqualProportion"
        ]
        self._build_x_scale_range = box_cfg["build"]["scaleRangeX"]
        self._build_y_scale_range = box_cfg["build"]["scaleRangeY"]
        self._build_z_scale_range = box_cfg["build"]["scaleRangeZ"]
        self._build_scale_sample_interval = box_cfg["build"]["scaleSampleInterval"]
        self._build_test_sizes = box_cfg["build"]["testSizes"]
        assert (
            box_cfg["build"].get("randomDensity", False) is False
        ), "randomDensity is not supported by the multi-agent carry task yet"
        assert (
            box_cfg["obs"]["enableBboxObs"] is True
        ), "the multi-agent object token always carries the bbox points"
        self._reset_random_rot = box_cfg["reset"]["randomRot"]
        self._reset_random_height = True
        self._stage1_platforms = True
        from utils.stage1_v2_fixed import uses_fixed_platforms

        self._fixed_platforms = uses_fixed_platforms(cfg["env"])
        self._platform_warmup_done = False
        self._reset_random_height_prob = box_cfg["reset"]["randomHeightProb"]
        self._ground_standalone_sit_climb = True
        self._reset_min_platform_height = box_cfg["reset"].get("minPlatformHeight", 0.0)
        self._reset_max_top_surface_height = box_cfg["reset"]["maxTopSurfaceHeight"]
        self._randomize_box_assignment = False
        self._random_arena_box_spawn = False
        configured_radius = float(box_cfg["reset"].get("arenaSpawnRadius", 0.0))
        self._box_spawn_radius = (
            configured_radius
            if configured_radius > 0.0
            else max(0.5, float(cfg["env"]["envSpacing"]) - 0.5)
        )
        self._box_min_agent_dist = float(box_cfg["reset"].get("minAgentDistance", 1.0))
        self._box_min_box_dist = float(box_cfg["reset"].get("minBoxDistance", 0.7))
        state_init = cfg["env"]["stateInit"]
        self._state_init = HumanoidMACarry.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        assert self._num_amp_obs_steps >= 2
        self._reset_default_slots = None
        self._reset_ref_slots = {}
        self._reset_ref_motion_ids = {}
        self._reset_ref_motion_times = {}
        self._power_reward = cfg["env"]["power_reward"]
        self._power_coefficient = cfg["env"]["power_coefficient"]
        super().__init__(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=physics_engine,
            device_type=device_type,
            device_id=device_id,
            headless=headless,
        )
        self._skill = cfg["env"]["skill"]
        self._skill_init_prob = torch.tensor(
            cfg["env"]["skillInitProb"], device=self.device, dtype=torch.float
        )
        self._skill_disc_prob = torch.tensor(
            cfg["env"]["skillDiscProb"], device=self.device, dtype=torch.float
        )
        self._independent_template_rsi = None
        if self._template_rsi is not None:
            expected = {
                "HOLDING": {"loco", "pickUp"},
                "SIT": {"loco", "sit"},
                "CLIMB": {"loco", "climb"},
                "HOLDING_AT": {"loco", "pickUp", "carryWith", "putDown"},
                "HOLDING_ON_TOP": {"loco", "pickUp", "carryWith", "putDown"},
            }
            if cfg["env"].get("pushExtension") is not None:
                expected["PUSH_AT"] = {"loco", "push"}
            templates = scenario_templates(self._relation_graph_spec)
            for rows in (self._template_rsi,):
                if rows is None:
                    continue
                if tuple(rows) != templates:
                    raise ValueError(
                        "templateRsi rows must follow the scenario templates"
                    )
                for name, values in rows.items():
                    if (
                        len(values) != len(self._skill)
                        or any(
                            (
                                isinstance(value, bool)
                                or not isinstance(value, (int, float))
                                or (not math.isfinite(value))
                                or (value < 0)
                                for value in values
                            )
                        )
                        or abs(sum(values) - 1.0) > 1e-08
                        or any(
                            (
                                value and skill not in expected[name]
                                for skill, value in zip(self._skill, values)
                            )
                        )
                    ):
                        raise ValueError("Invalid templateRsi row: " + name)
            self._template_rsi_weights = torch.tensor(
                [self._template_rsi[k] for k in templates],
                device=self.device,
                dtype=torch.float,
            )
        motion_file = cfg["env"]["motion_file"]
        self._load_motion(motion_file)
        N, M = (self.num_envs, self.num_agents)
        self._amp_obs_buf = torch.zeros(
            (N, M, self._num_amp_obs_steps, self._num_amp_obs_per_step),
            device=self.device,
            dtype=torch.float,
        )
        self._curr_amp_obs_buf = self._amp_obs_buf[:, :, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, :, 1:]
        self._amp_obs_demo_buf = None
        self._prev_root_pos = torch.zeros(
            [N, M, 3], device=self.device, dtype=torch.float
        )
        self._prev_box_pos = torch.zeros(
            [N, M, 3], device=self.device, dtype=torch.float
        )
        self._tar_pos = torch.zeros([N, M, 3], device=self.device, dtype=torch.float)
        self._reward_term_sums = torch.zeros(
            len(self.REWARD_TERM_NAMES), device=self.device
        )
        self._reward_term_count = 0
        spacing = cfg["env"]["envSpacing"]
        reach = min(4.5, max(0.5, spacing - 0.5))
        if M > 1:
            reach = max(
                1.5, reach - float(self._agent_spawn_offsets.norm(dim=-1).max())
            )
        self._tar_reach = reach
        self._tar_height_range = [0.5, 1.0]
        if self._enable_markers:
            self._build_marker_state_tensors()
        self._build_box_tensors()
        self._build_platform_state_tensors()
        self._build_assignment_tensors()
        self._every_env_init_dof_pos = torch.zeros(
            (N, M, self.num_dof), device=self.device, dtype=torch.float
        )
        self._kinematic_humanoid_rigid_body_states = torch.zeros(
            (N, M, self.num_bodies, 13), device=self.device, dtype=torch.float
        )
        self._is_eval = cfg["args"].eval
        if self._is_eval:
            self._success_buf = torch.zeros(N * M, device=self.device, dtype=torch.long)
            self._precision_buf = torch.zeros(
                N * M, device=self.device, dtype=torch.float
            )
            self._success_threshold = cfg["env"]["eval"]["successThreshold"]
            self._skill = cfg["env"]["eval"]["skill"]
            self._skill_init_prob = torch.tensor(
                cfg["env"]["eval"]["skillInitProb"],
                device=self.device,
                dtype=torch.float,
            )
        from utils.task_role_spec import task_size_probabilities as task_probabilities

        self._size_template_probabilities = (self._push_size_probabilities
            if hasattr(self, "_push_size_probabilities") else task_probabilities(
            self._box_size[:, :M], cfg["env"]["box"]["build"]["taskSizeRanges"]))
        self._init_relation_runtime()
        if not self._is_eval:
            from env.tasks.multi_agent.size_rsi_cache import SizeRsiCache

            cache_class = getattr(self, "_solo_cache_class", SizeRsiCache)
            self._size_rsi_cache = cache_class(self)
            self._size_rsi_requested = torch.zeros(
                len(scenario_templates(self._relation_graph_spec)), len(self._skill), device=self.device
            )
            self._size_rsi_executed = torch.zeros_like(self._size_rsi_requested)
            self._size_rsi_fallbacks = torch.zeros(len(scenario_templates(self._relation_graph_spec)), device=self.device)
            self._size_rsi_success_counts = torch.zeros(2, 5, 2, device=self.device)
            self._size_rsi_pending_post = torch.zeros(
                N, dtype=torch.bool, device=self.device
            )
        return

    def get_relation_suffix_size(self):
        return self._context_suffix_width

    def get_object_obs_size(self):
        return 3 + 3 + 3 + 6 + 24

    def get_goal_obs_size(self):
        return 3 + 3

    def get_clean_object_obs_size(self):
        return 3 + 3 + 24

    def get_clean_goal_obs_size(self):
        return 1

    def get_scene_kinematic_size(self):
        return 3 + 4

    def get_scene_arena_scale(self):
        """Scene spacing used by runner configuration."""
        return self._arena_scale

    def get_geometry_obs_size(self):
        return 3 + 6 + 3 + 3

    def get_scene_entity_sizes(self):
        return [
            self.get_clean_humanoid_obs_size(),
            self.get_clean_object_obs_size(),
            self.get_clean_goal_obs_size(),
        ]

    def get_scene_normalized_entity_sizes(self):
        """Normalize intrinsic H/O nodes; constant Target and GTA poses pass through."""
        return [
            self.get_clean_humanoid_self_obs_size(),
            self.get_clean_object_obs_size(),
            0,
        ]

    def get_scene_num_tokens(self):
        return 2 * self.num_agents + self.num_objects

    def get_task_obs_size(self):
        if not self._enable_task_obs:
            return 0
        return (
            self.num_objects * self.get_object_obs_size()
            + self.num_agents * self.get_goal_obs_size()
        )

    def get_obs_size(self):
        node_width = (
            self.num_agents * self.get_clean_humanoid_obs_size()
            + self.num_objects * self.get_clean_object_obs_size()
            + self.num_agents * self.get_clean_goal_obs_size()
        )
        kinematic_width = self.get_scene_num_tokens() * self.get_scene_kinematic_size()
        return node_width + kinematic_width + self.get_relation_suffix_size()

    def _create_envs(self, num_envs, spacing, num_per_row):
        self._enable_markers = not self.headless or self._video_enabled
        if self._enable_markers:
            self._marker_handles = []
            self._load_marker_asset()
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
        self._marker_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )
        return

    def _load_platform_asset(self):
        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0 if self._fixed_platforms else 0.0
        asset_options.density = 1.0 if self._fixed_platforms else 10000000000.0
        asset_options.fix_base_link = self._fixed_platforms
        asset_options.disable_gravity = not self._fixed_platforms
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        self._platform_height = 0.02
        self._platform_inactive_height = 20.0
        self._platform_asset = self.gym.create_box(
            self.sim, 0.4, 0.4, self._platform_height, asset_options
        )
        return

    def _load_box_asset(self):
        N, O = (self.num_envs, self.num_objects)
        num_boxes = N * O
        self._box_density = torch.full(
            (num_boxes,), 100.0, dtype=torch.float32, device=self.device
        )
        from utils.size_rsi import sample_sizes

        size_args = (
            (self.cfg["env"]["box"]["build"]["taskSizeRanges"],)
            if self._relation_cfg.get("stage1_variant") == "stage1_v2"
            else ()
        )
        self._box_size = sample_sizes(
            N,
            self.device,
            self._task_graph_preset,
            *size_args,
            num_agents=self.num_agents,
            num_objects=O
        ).reshape(num_boxes, 3)
        self._box_scale = self._box_size / torch.tensor(
            self._build_base_size, device=self.device
        )
        stack_height = max_stage1_stack_height(self._box_size.reshape(N, O, 3))
        if self._relation_cfg.get("stage1_variant") == "stage1_v2":
            from utils.stage1_v2 import max_bound_stack_height

            stack_height = max_bound_stack_height(self._box_size.reshape(N, O, 3))
        if (stack_height > self._reset_max_top_surface_height + 1e-06).any():
            raise ValueError(
                "Actual box assets exceed the configured stack height limit"
            )
        self._box_assets = []
        for i in range(num_boxes):
            asset_options = gymapi.AssetOptions()
            asset_options.angular_damping = 0.01
            asset_options.linear_damping = 0.01
            asset_options.max_angular_velocity = 100.0
            asset_options.density = self._box_density[i]
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            self._box_assets.append(
                self.gym.create_box(
                    self.sim,
                    self._box_size[i, 0],
                    self._box_size[i, 1],
                    self._box_size[i, 2],
                    asset_options,
                )
            )
        self._box_scale = self._box_scale.view(N, O, 3)
        self._box_size = self._box_size.view(N, O, 3)
        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        for humanoid_handle in self.humanoid_handles_all[env_id]:
            self._filter_platform_foot_collisions(env_ptr, humanoid_handle)
        for object_id in range(self.num_physical_objects):
            self._build_box(env_id, object_id, env_ptr)
        for agent_id in range(self.num_agents):
            self._build_platforms(env_id, agent_id, env_ptr)
        if self._enable_markers:
            for a in range(self.num_agents):
                self._build_marker(env_id, a, env_ptr)
        return

    def _filter_platform_foot_collisions(self, env_ptr, humanoid_handle):
        body_dict = self.gym.get_actor_rigid_body_dict(env_ptr, humanoid_handle)
        shape_ranges = self.gym.get_actor_rigid_body_shape_indices(
            env_ptr, humanoid_handle
        )
        shape_props = self.gym.get_actor_rigid_shape_properties(
            env_ptr, humanoid_handle
        )
        for body_name in ("left_foot", "right_foot"):
            shape_range = shape_ranges[body_dict[body_name]]
            for shape_id in range(
                shape_range.start, shape_range.start + shape_range.count
            ):
                shape_props[shape_id].filter |= self._PLATFORM_COLLISION_FILTER
        self.gym.set_actor_rigid_shape_properties(env_ptr, humanoid_handle, shape_props)
        return

    def _build_box(self, env_id, object_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0
        box_idx = env_id * self.num_physical_objects + object_id
        angle = 2.0 * np.pi * float(object_id) / float(max(1, self.num_objects))
        radius = min(self._box_spawn_radius, 1.0 + 0.5 * float(object_id))
        default_pose = gymapi.Transform()
        default_pose.p.x = radius * np.cos(angle)
        default_pose.p.y = radius * np.sin(angle)
        default_pose.p.z = float(self._box_size.view(-1, 3)[box_idx, 2]) / 2
        box_handle = self.gym.create_actor(
            env_ptr,
            self._box_assets[box_idx],
            default_pose,
            "box_{}".format(object_id),
            col_group,
            col_filter,
            segmentation_id,
        )
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
            env_ptr,
            self._platform_asset,
            default_pose,
            "platform_{}".format(agent_id),
            col_group,
            col_filter,
            segmentation_id,
        )
        self.gym.set_rigid_body_color(
            env_ptr,
            platform_handle,
            0,
            gymapi.MESH_VISUAL,
            gymapi.Vec3(0.5, 0.235, 0.6),
        )
        default_pose.p.z = self._platform_inactive_height + 0.5
        tar_platform_handle = self.gym.create_actor(
            env_ptr,
            self._platform_asset,
            default_pose,
            "tar_platform_{}".format(agent_id),
            col_group,
            col_filter,
            segmentation_id,
        )
        self.gym.set_rigid_body_color(
            env_ptr,
            tar_platform_handle,
            0,
            gymapi.MESH_VISUAL,
            gymapi.Vec3(0.0, 0.0, 0.8),
        )
        self._platform_handles.append(platform_handle)
        self._tar_platform_handles.append(tar_platform_handle)
        return

    def _build_marker(self, env_id, agent_id, env_ptr):
        col_group = self.num_envs + 1
        col_filter = 0
        segmentation_id = 0
        default_pose = gymapi.Transform()
        marker_handle = self.gym.create_actor(
            env_ptr,
            self._marker_asset,
            default_pose,
            "marker_{}".format(agent_id),
            col_group,
            col_filter,
            segmentation_id,
        )
        self.gym.set_rigid_body_color(
            env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, self._agent_color(agent_id)
        )
        self.gym.set_actor_scale(env_ptr, marker_handle, 0.3)
        self._marker_handles.append(marker_handle)
        return

    def _build_box_tensors(self):
        N, M, O = (self.num_envs, self.num_agents, self.num_physical_objects)
        num_actors = self.get_num_actors_per_env()
        self._box_states = self._root_states.view(
            N, num_actors, self._root_states.shape[-1]
        )[:, M : M + O, :]
        self._box_actor_ids = (
            (
                num_actors * torch.arange(N, device=self.device, dtype=torch.int32)
            ).unsqueeze(-1)
            + torch.arange(O, device=self.device, dtype=torch.int32).unsqueeze(0)
            + M
        )
        self._initial_box_states = self._box_states.clone()
        self._initial_box_states[..., 7:13] = 0
        self._build_box_bps()
        return

    def _build_box_bps(self):
        half = self._box_size / 2.0
        signs = torch.tensor(
            [
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, 1],
                [-1, 1, 1],
                [-1, -1, 1],
                [1, -1, 1],
            ],
            device=self.device,
            dtype=torch.float,
        )
        self._box_bps = half.unsqueeze(-2) * signs.view(1, 1, 8, 3)
        return

    def _build_platform_state_tensors(self):
        N, M, O = (self.num_envs, self.num_agents, self.num_physical_objects)
        num_actors = self.get_num_actors_per_env()
        actor_base = num_actors * torch.arange(
            N, device=self.device, dtype=torch.int32
        ).unsqueeze(-1)
        platform_offsets = (
            M + O + 2 * torch.arange(M, device=self.device, dtype=torch.int32)
        )
        tar_platform_offsets = platform_offsets + 1
        root_states = self._root_states.view(N, num_actors, self._root_states.shape[-1])
        platform_start = M + O
        self._platform_states = root_states[
            :, platform_start : platform_start + 2 * M : 2, :
        ]
        self._platform_pos = self._platform_states[..., :3]
        self._platform_default_pos = self._platform_pos.clone()
        self._platform_actor_ids = actor_base + platform_offsets.unsqueeze(0)
        self._tar_platform_states = root_states[
            :, platform_start + 1 : platform_start + 2 * M : 2, :
        ]
        self._tar_platform_pos = self._tar_platform_states[..., :3]
        self._tar_platform_default_pos = self._tar_platform_pos.clone()
        self._tar_platform_actor_ids = actor_base + tar_platform_offsets.unsqueeze(0)
        return

    def _build_assignment_tensors(self):
        """Initialise logical owner slots independently from physical box indices."""
        N, M, O = (self.num_envs, self.num_agents, self.num_objects)
        identity = (
            torch.arange(M, device=self.device, dtype=torch.long)
            .unsqueeze(0)
            .expand(N, -1)
        )
        self._agent_box_assignment = identity.clone()
        self._logical_box_order = (
            torch.arange(O, device=self.device, dtype=torch.long)
            .unsqueeze(0)
            .expand(N, -1)
            .clone()
        )
        return

    def _sample_box_assignments(self, env_ids):
        """Assign M distinct physical boxes and put unassigned boxes after them."""
        K, M, O = (env_ids.shape[0], self.num_agents, self.num_objects)
        assignment = (
            torch.arange(M, device=self.device, dtype=torch.long)
            .unsqueeze(0)
            .expand(K, -1)
        )
        all_ids = (
            torch.arange(O, device=self.device, dtype=torch.long)
            .unsqueeze(0)
            .expand(K, -1)
        )
        assigned_mask = torch.zeros(K, O, device=self.device, dtype=torch.bool)
        assigned_mask.scatter_(1, assignment, True)
        unassigned = all_ids.masked_select(~assigned_mask).view(K, O - M)
        self._agent_box_assignment[env_ids] = assignment
        self._logical_box_order[env_ids] = torch.cat([assignment, unassigned], dim=-1)
        self._update_box_assignment_colors(env_ids)
        return

    def _update_box_assignment_colors(self, env_ids):
        """Color each task object by its agents; shared objects use yellow."""
        rendered_envs = []
        if self.viewer is not None:
            rendered_envs = env_ids.detach().cpu().tolist()
        elif self._video_enabled and torch.any(env_ids == 0):
            rendered_envs = [0]
        unassigned_color = gymapi.Vec3(0.45, 0.45, 0.45)
        shared_color = gymapi.Vec3(1.0, 0.85, 0.15)
        for env_id in rendered_envs:
            env_ptr = self.envs[env_id]
            slot_env = torch.full(
                (self.num_agents,), env_id, device=self.device, dtype=torch.long
            )
            slot_agent = torch.arange(self.num_agents, device=self.device)
            logical = agent_object_indices(
                self.relation_runtime.graph, slot_env, slot_agent
            )
            assignment = (
                self._logical_box_order[env_id, logical].detach().cpu().tolist()
            )
            owners_by_box = {}
            for agent_id, physical_box in enumerate(assignment):
                owners_by_box.setdefault(physical_box, []).append(agent_id)
            for physical_box in range(self.num_physical_objects):
                box_handle = self._box_handles[env_id * self.num_physical_objects + physical_box]
                owners = owners_by_box.get(physical_box, [])
                color = (
                    shared_color
                    if len(owners) > 1
                    else self._agent_color(owners[0]) if owners else unassigned_color
                )
                self.gym.set_rigid_body_color(
                    env_ptr, box_handle, 0, gymapi.MESH_VISUAL, color
                )
        return

    @staticmethod
    def _gather_entity_slots(values, indices):
        """values (B, K, ...) + indices (B, S) -> (B, S, ...)."""
        batch = torch.arange(values.shape[0], device=values.device).unsqueeze(-1)
        return values[batch, indices]

    def _assigned_box_values(self, values, env_ids=None):
        if env_ids is None:
            return self._gather_entity_slots(values, self._agent_box_assignment)
        return self._gather_entity_slots(
            values[env_ids], self._agent_box_assignment[env_ids]
        )

    def _regulate_height(self, height, box_size):
        top_surface_z = height + box_size[:, 2] / 2
        top_surface_z = torch.clamp_max(
            top_surface_z, self._reset_max_top_surface_height
        )
        return top_surface_z - box_size[:, 2] / 2

    def _ground_stage1_standalone_targets(self, env_ids):
        ground_standalone_interaction_targets(
            self.relation_runtime.graph,
            env_ids,
            self._agent_box_assignment,
            self._box_states,
            self._box_size,
            self._platform_pos,
            self._platform_default_pos,
        )

    def _logical_box_values(self, values, env_ids=None):
        if env_ids is None:
            return self._gather_entity_slots(values, self._logical_box_order)
        return self._gather_entity_slots(
            values[env_ids], self._logical_box_order[env_ids]
        )

    def _build_marker_state_tensors(self):
        N, M, O = (self.num_envs, self.num_agents, self.num_physical_objects)
        num_actors = self._root_states.shape[0] // N
        marker_start = M + O + 2 * M
        self._marker_states = self._root_states.view(
            N, num_actors, self._root_states.shape[-1]
        )[:, marker_start : marker_start + M, :]
        self._marker_pos = self._marker_states[..., :3]
        self._marker_actor_ids = (
            (
                num_actors * torch.arange(N, device=self.device, dtype=torch.int32)
            ).unsqueeze(-1)
            + torch.arange(M, device=self.device, dtype=torch.int32).unsqueeze(0)
            + marker_start
        )
        return

    def _compute_observations(self, env_ids=None):
        obs = self._compute_clean_scene_obs(env_ids)
        if env_ids is None:
            self.obs_buf[:] = obs
        else:
            self.obs_buf[env_ids] = obs
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
        B, M, O = (body_pos.shape[0], self.num_agents, self.num_objects)
        human_nodes = self._compute_clean_humanoid_nodes(
            body_pos, body_rot, body_vel, body_ang_vel, origins
        )
        box_rot = box_states[..., 3:7]
        box_rot_inv = box_rot.clone()
        box_rot_inv[..., 0:3] *= -1.0
        object_local = torch.cat(
            [
                quat_rotate(
                    box_rot_inv.reshape(-1, 4), box_states[..., 7:10].reshape(-1, 3)
                ).view(B, O, 3),
                quat_rotate(
                    box_rot_inv.reshape(-1, 4), box_states[..., 10:13].reshape(-1, 3)
                ).view(B, O, 3),
                box_bps.reshape(B, O, 24),
            ],
            dim=-1,
        )
        object_nodes = object_local
        goal_nodes = torch.ones(B, M, 1, device=self.device, dtype=body_pos.dtype)
        assert human_nodes.shape == (B, M, self.get_clean_humanoid_obs_size())
        assert object_nodes.shape == (B, O, self.get_clean_object_obs_size())
        assert goal_nodes.shape == (B, M, self.get_clean_goal_obs_size())
        root_pos = body_pos[..., 0, :]
        root_rot = body_rot[..., 0, :]
        human_heading = torch_utils.calc_heading_quat(root_rot.reshape(-1, 4)).view(
            B, M, 4
        )
        nodes = torch.cat(
            [
                human_nodes.reshape(B, -1),
                object_nodes.reshape(B, -1),
                goal_nodes.reshape(B, -1),
            ],
            dim=-1,
        )
        entity_poses = build_gta_pose_records(
            root_pos,
            human_heading,
            box_states[..., 0:3],
            box_rot,
            tar_pos,
            origins,
            goal_rotation=self.cfg["env"].get("goalRotation", "human_heading"),
        )
        parts = [nodes, entity_poses.reshape(B, -1)]
        suffix = self.relation_runtime.suffix(env_ids)
        parts.append(suffix)
        return torch.cat(parts, dim=-1)

    def _compute_agent_collision_violation(self, root_pos, min_dist):
        return compute_agent_cpa_collision_penalty(
            root_pos,
            self._humanoid_root_states[..., 7:10],
            min_dist,
            self._agent_collision_config["ttc_discount"],
            self.dt,
        )

    def _compute_reward(self, actions):
        return self._compute_relation_reward(self._compute_agent_collision_violation)

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
        self._prev_box_pos[:] = self._reward_box_values(self._box_states)[..., 0:3]
        return

    def post_physics_step(self):
        self.progress_buf += 1
        self._refresh_sim_tensors()
        self._compute_reward(self.actions)
        self._compute_observations()
        self._compute_reset()
        self._finish_relation_diagnostics()
        self.extras["terminate"] = self._terminate_buf
        self._update_hist_amp_obs()
        self._compute_amp_observations()
        self.extras["amp_obs"] = self._amp_obs_buf.reshape(
            self.num_envs * self.num_agents, self.get_num_amp_obs()
        )
        self.extras["policy_obs"] = self.obs_buf.clone()
        if self._is_eval:
            self._compute_metrics_evaluation()
            self.extras["success"] = self._success_buf
            self.extras["precision"] = self._precision_buf
        if self._recording:
            self._capture_video_frame()
        if self.viewer and self.debug_viz:
            self._update_debug_viz()
        self._draw_ontop_context()
        return

    def _compute_metrics_evaluation(self):
        self._success_buf.copy_(self.relation_runtime.done.flatten().long())
        _, diag = self._evaluate_relations()
        graph = self.relation_runtime.graph
        from env.tasks.multi_agent.edge_context_reward import owner_sum

        mask = batched(graph.required_goal & graph.edge_valid, self.num_envs).float()
        errors = owner_sum(diag["distance"] * mask, graph) / owner_sum(
            mask.expand(self.num_envs, -1), graph
        ).clamp_min(1)
        self._precision_buf.copy_(errors.flatten())
        return

    def _video_focus_points(self):
        if self._video_focus == "stack":
            return torch.cat(
                [self._humanoid_root_states[0, :, :3], self._box_states[0, :, :3]], 0
            )
        if self._video_focus == "agent0":
            return self._humanoid_root_states[0, 0:1, 0:3]
        return self._humanoid_root_states[0, :, 0:3]

    def _reset_envs(self, env_ids):
        if self._fixed_platforms and (not self._platform_warmup_done) and len(env_ids):
            if len(env_ids) != self.num_envs:
                raise ValueError("Fixed platforms require a full first reset")
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self._refresh_sim_tensors()
            self._platform_warmup_done = True
            print(
                "[stage1_v2_fixed] Initial platform physics step complete", flush=True
            )
        return self._reset_ontop_context_envs(env_ids)

    def _reset_actors(self, env_ids):
        slot_env, slot_agent = self._expand_slots(env_ids)
        if self._state_init == HumanoidMACarry.StateInit.Default:
            self._reset_default(slot_env, slot_agent)
            self._reset_default_slots = (slot_env, slot_agent)
        elif (
            self._state_init == HumanoidMACarry.StateInit.Start
            or self._state_init == HumanoidMACarry.StateInit.Random
        ):
            self._reset_ref_state_init(slot_env, slot_agent)
        elif self._state_init == HumanoidMACarry.StateInit.Hybrid:
            probs = to_torch(
                np.array([self._hybrid_init_prob] * slot_env.shape[0]),
                device=self.device,
            )
            ref_mask = torch.bernoulli(probs) == 1.0
            if ref_mask.sum() > 0:
                self._reset_ref_state_init(slot_env[ref_mask], slot_agent[ref_mask])
            rest = torch.logical_not(ref_mask)
            if rest.sum() > 0:
                self._reset_default(slot_env[rest], slot_agent[rest])
                self._reset_default_slots = (slot_env[rest], slot_agent[rest])
        else:
            assert False, "Unsupported state initialization strategy: {:s}".format(
                str(self._state_init)
            )
        return

    def _reset_default(self, slot_env, slot_agent):
        super()._reset_default(slot_env, slot_agent)
        self._kinematic_humanoid_rigid_body_states[slot_env, slot_agent] = (
            self._initial_humanoid_rigid_body_states[slot_env, slot_agent]
        )
        self._every_env_init_dof_pos[slot_env, slot_agent] = self._initial_dof_pos[
            slot_env, slot_agent
        ]
        return

    def _scenario_physical_object(self, slot_env, slot_agent):
        logical = agent_object_indices(
            self.relation_runtime.graph, slot_env, slot_agent
        )
        return self._logical_box_order[slot_env, logical]

    def _scenario_goal(self, slot_env, slot_agent):
        return agent_goal_indices(self.relation_runtime.graph, slot_env, slot_agent)

    def _reset_ref_state_init(self, slot_env, slot_agent):
        num_slots = slot_env.shape[0]
        if self._template_rsi is not None and (not self._is_eval):
            graph = self.relation_runtime.graph
            template = classify_templates(graph, slot_env, slot_agent, True)
            weights = self._template_rsi_weights[template]
            sk_ids = torch.multinomial(weights, 1).flatten()
            logical = agent_object_indices(graph, slot_env, slot_agent)
            advanced = torch.tensor(
                [name != "loco" for name in self._skill], device=self.device
            )[sk_ids]
            for _ in range(16):
                conflict = (
                    (slot_env[:, None] == slot_env[None, :])
                    & (logical[:, None] == logical[None, :])
                    & advanced[:, None]
                    & advanced[None, :]
                    & ~torch.eye(num_slots, dtype=torch.bool, device=self.device)
                )
                retry = conflict.any(-1)
                if not retry.any():
                    break
                sk_ids[retry] = torch.multinomial(weights[retry], 1).flatten()
                advanced = torch.tensor(
                    [name != "loco" for name in self._skill], device=self.device
                )[sk_ids]
            if retry.any():
                raise RuntimeError(
                    "No valid template RSI combination after 16 attempts"
                )
            from utils.size_rsi import resolve_skills

            requested = sk_ids.clone()
            sk_ids, fallback = resolve_skills(
                sk_ids,
                weights,
                self._size_rsi_cache.availability(slot_env, slot_agent, template),
                self._skill.index("loco"),
            )
            self._size_rsi_requested.index_put_(
                (template, requested),
                torch.ones_like(template, dtype=torch.float),
                accumulate=True,
            )
            self._size_rsi_fallbacks.index_add_(0, template, fallback.float())
        else:
            sk_ids = torch.multinomial(
                self._skill_init_prob, num_samples=num_slots, replacement=True
            )
        for uid, sk_name in enumerate(self._skill):
            curr_motion_lib = self._motion_lib[sk_name]
            sel = (sk_ids == uid).nonzero().squeeze(-1)
            if len(sel) == 0:
                continue
            curr_env = slot_env[sel]
            curr_agent = slot_agent[sel]
            offset = self._agent_spawn_offsets[curr_agent] + self._env_origins[curr_env]
            if not self._is_eval:
                motion_ids, motion_times = self._size_rsi_cache.sample(
                    sk_name, curr_env, curr_agent, template[sel]
                )
            else:
                motion_ids = curr_motion_lib.sample_motions(len(sel))
                if (
                    self._state_init == HumanoidMACarry.StateInit.Random
                    or self._state_init == HumanoidMACarry.StateInit.Hybrid
                ):
                    motion_times = curr_motion_lib.sample_time_rsi(motion_ids)
                elif self._state_init == HumanoidMACarry.StateInit.Start:
                    motion_times = torch.zeros(len(sel), device=self.device)
                else:
                    assert (
                        False
                    ), "Unsupported state initialization strategy: {:s}".format(
                        str(self._state_init)
                    )
            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = (
                curr_motion_lib.get_motion_state(motion_ids, motion_times)
            )
            self._set_env_state(
                curr_env,
                curr_agent,
                root_pos=root_pos + offset,
                root_rot=root_rot,
                dof_pos=dof_pos,
                root_vel=root_vel,
                root_ang_vel=root_ang_vel,
                dof_vel=dof_vel,
            )
            self._reset_ref_slots[sk_name] = (curr_env, curr_agent)
            self._reset_ref_motion_ids[sk_name] = motion_ids
            self._reset_ref_motion_times[sk_name] = motion_times
            body_pos, body_rot, body_vel, body_ang_vel = (
                curr_motion_lib.get_motion_state_max(motion_ids, motion_times)
            )
            body_pos = body_pos + offset.unsqueeze(-2)
            self._kinematic_humanoid_rigid_body_states[curr_env, curr_agent] = (
                torch.cat((body_pos, body_rot, body_vel, body_ang_vel), dim=-1)
            )
            self._every_env_init_dof_pos[curr_env, curr_agent] = dof_pos
        return

    def _set_env_state(
        self,
        slot_env,
        slot_agent,
        root_pos,
        root_rot,
        dof_pos,
        root_vel,
        root_ang_vel,
        dof_vel,
    ):
        self._humanoid_root_states[slot_env, slot_agent, 0:3] = root_pos
        self._humanoid_root_states[slot_env, slot_agent, 3:7] = root_rot
        self._humanoid_root_states[slot_env, slot_agent, 7:10] = root_vel
        self._humanoid_root_states[slot_env, slot_agent, 10:13] = root_ang_vel
        self._dof_pos[slot_env, slot_agent] = dof_pos
        self._dof_vel[slot_env, slot_agent] = dof_vel
        return

    def _collect_random_slots(self, skills):
        """Slots whose box / target must be sampled instead of taken from a motion clip."""
        envs, agents = ([], [])
        if self._reset_default_slots is not None:
            envs.append(self._reset_default_slots[0])
            agents.append(self._reset_default_slots[1])
        for sk_name in skills:
            if self._reset_ref_slots.get(sk_name) is not None:
                envs.append(self._reset_ref_slots[sk_name][0])
                agents.append(self._reset_ref_slots[sk_name][1])
        if len(envs) == 0:
            return None
        return (torch.cat(envs, dim=0), torch.cat(agents, dim=0))

    def _reset_boxes(self, env_ids):
        self._reset_all_boxes_random_arena(env_ids)
        self._platform_pos[env_ids] = self._platform_default_pos[env_ids]
        reference_object_skills = ["pickUp", "carryWith", "putDown"]
        reference_object_skills += ["sit"]
        reference_object_skills += ["climb"]
        for sk_name in reference_object_skills:
            if self._reset_ref_slots.get(sk_name) is None:
                continue
            curr_env, curr_agent = self._reset_ref_slots[sk_name]
            offset = self._agent_spawn_offsets[curr_agent] + self._env_origins[curr_env]
            if sk_name in ("sit", "climb"):
                root_pos, root_rot = self._motion_lib[
                    sk_name
                ].get_obj_motion_state_single_frame(self._reset_ref_motion_ids[sk_name])
            else:
                root_pos, root_rot = self._motion_lib[sk_name].get_obj_motion_state(
                    motion_ids=self._reset_ref_motion_ids[sk_name],
                    motion_times=self._reset_ref_motion_times[sk_name],
                )
            root_pos = root_pos + offset
            curr_box = self._scenario_physical_object(curr_env, curr_agent)
            box_size = self._box_size[curr_env, curr_box]
            if sk_name in ("sit", "climb"):
                root_pos[:, 2] = box_size[:, 2] / 2
            else:
                on_ground = box_size[:, 2] / 2 > root_pos[:, 2]
                root_pos[on_ground, 2] = box_size[on_ground, 2] / 2
            self._box_states[curr_env, curr_box, 0:3] = root_pos
            self._box_states[curr_env, curr_box, 3:7] = root_rot
            self._box_states[curr_env, curr_box, 7:13] = 0.0
            self._platform_pos[curr_env, curr_agent] = self._platform_default_pos[
                curr_env, curr_agent
            ]
        owner_range = self.cfg["env"]["box"]["reset"].get("ownerLocoDistanceRange")
        random_slots = (
            None if owner_range is None else self._collect_random_slots(["loco"])
        )
        if random_slots is not None:
            curr_env, curr_agent = random_slots
            K = curr_env.shape[0]
            curr_box = self._scenario_physical_object(curr_env, curr_agent)
            from utils.stage1_sampling import uses_tokenhsi_sampling, sample_source_xy

            if uses_tokenhsi_sampling(self.cfg["env"]):
                templates = classify_templates(
                    self.relation_runtime.graph, curr_env, curr_agent, True
                )
                root_pos_xy = sample_source_xy(
                    self._humanoid_root_states[curr_env, curr_agent, :2], templates
                )
            else:
                root_pos_xy = torch.randn(K, 2, device=self.device)
                root_pos_xy /= torch.linalg.norm(root_pos_xy, dim=-1, keepdim=True)
                lo, hi = (
                    owner_range if owner_range is not None else (1.0, self._tar_reach)
                )
                root_pos_xy *= torch.rand(K, 1, device=self.device) * (hi - lo) + lo
                root_pos_xy += self._humanoid_root_states[curr_env, curr_agent, :2]
            root_pos_z = self._box_size[curr_env, curr_box, 2] / 2
            axis = (
                torch.tensor([[0.0, 0.0, 1.0]], device=self.device)
                .reshape(1, 3)
                .expand([K, -1])
            )
            coeff = 1.0 if self._reset_random_rot else 0.0
            ang = torch.rand((K,), device=self.device) * 2 * np.pi * coeff
            root_rot = quat_from_angle_axis(ang, axis)
            root_pos = torch.cat([root_pos_xy, root_pos_z.unsqueeze(-1)], dim=-1)
            self._box_states[curr_env, curr_box, 0:3] = root_pos
            self._box_states[curr_env, curr_box, 3:7] = root_rot
            self._box_states[curr_env, curr_box, 7:13] = 0.0
        return

    def _reset_task(self, env_ids):
        self._tar_pos[env_ids] = scenario_neutral_targets(
            self._env_origins, env_ids, self.num_agents
        )
        self._tar_platform_pos[env_ids] = self._tar_platform_default_pos[env_ids]
        for sk_name in ["putDown"]:
            if self._reset_ref_slots.get(sk_name) is None:
                continue
            curr_env, curr_agent = self._reset_ref_slots[sk_name]
            templates = classify_templates(
                self.relation_runtime.graph, curr_env, curr_agent, True
            )
            is_at = templates == scenario_templates(self._relation_graph_spec).index(
                "HOLDING_AT"
            )
            curr_env, curr_agent = (curr_env[is_at], curr_agent[is_at])
            if not len(curr_env):
                continue
            motion_ids = self._reset_ref_motion_ids[sk_name][is_at]
            curr_box = self._scenario_physical_object(curr_env, curr_agent)
            goal_index = self._scenario_goal(curr_env, curr_agent)
            offset = self._agent_spawn_offsets[curr_agent] + self._env_origins[curr_env]
            root_pos, root_rot = self._motion_lib[sk_name].get_obj_motion_state(
                motion_ids=motion_ids,
                motion_times=self._motion_lib[sk_name].get_motion_length(motion_ids),
            )
            root_pos = root_pos + offset
            root_pos[:, 2] = self._box_size[curr_env, curr_box, 2] / 2
            self._tar_pos[curr_env, goal_index] = root_pos
            self._tar_platform_pos[curr_env, curr_agent] = (
                self._tar_platform_default_pos[curr_env, curr_agent]
            )
        random_slots = self._collect_random_slots(["loco", "pickUp", "carryWith"])
        if random_slots is not None:
            curr_env, curr_agent = random_slots
            template = classify_templates(
                self.relation_runtime.graph, curr_env, curr_agent, True
            )
            keep = template == 3
            curr_env, curr_agent = (curr_env[keep], curr_agent[keep])
            if not len(curr_env):
                return
            K = curr_env.shape[0]
            curr_box = self._scenario_physical_object(curr_env, curr_agent)
            goal_index = self._scenario_goal(curr_env, curr_agent)
            center = (
                self._agent_spawn_offsets[curr_agent][:, :2]
                + self._env_origins[curr_env, :2]
            )
            min_dist = 1.0

            def sample(n):
                xy = (
                    torch.rand(n, 2, device=self.device) * 2.0 - 1.0
                ) * self._tar_reach
                return xy

            from utils.stage1_sampling import uses_tokenhsi_sampling, sample_goal_xy

            native_sampling = uses_tokenhsi_sampling(self.cfg["env"])
            new_xy = (
                sample_goal_xy(
                    self._humanoid_root_states[curr_env, curr_agent, :2],
                    self._box_states[curr_env, curr_box, :2],
                    self._env_origins[curr_env, :2],
                )
                if native_sampling
                else sample(K) + center
            )
            new_z = self._box_size[curr_env, curr_box, 2] / 2
            new_target_pos = torch.cat([new_xy, new_z.unsqueeze(-1)], dim=-1)
            overlap = torch.logical_or(
                torch.sum(
                    (
                        new_target_pos[..., :2]
                        - self._humanoid_root_states[curr_env, curr_agent, :2]
                    )
                    ** 2,
                    dim=-1,
                )
                < min_dist,
                torch.sum(
                    (new_target_pos[..., :2] - self._box_states[curr_env, curr_box, :2])
                    ** 2,
                    dim=-1,
                )
                < min_dist,
            )
            tries = 0
            while torch.sum(overlap) > 0 and tries < 32:
                n = int(torch.sum(overlap))
                new_target_pos[overlap, :2] = sample(n) + center[overlap]
                overlap = torch.logical_or(
                    torch.sum(
                        (
                            new_target_pos[..., :2]
                            - self._humanoid_root_states[curr_env, curr_agent, :2]
                        )
                        ** 2,
                        dim=-1,
                    )
                    < min_dist,
                    torch.sum(
                        (
                            new_target_pos[..., :2]
                            - self._box_states[curr_env, curr_box, :2]
                        )
                        ** 2,
                        dim=-1,
                    )
                    < min_dist,
                )
                tries += 1
            self._tar_pos[curr_env, goal_index] = new_target_pos
        return

    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)
        if self._is_eval:
            flat = self._flat_slot_ids(env_ids)
            self._success_buf[flat] = 0
            self._precision_buf[flat] = float("Inf")
        ids = [self._box_actor_ids[env_ids].contiguous().view(-1)]
        if self._enable_markers:
            self._marker_pos[env_ids] = self._tar_pos[env_ids]
            from utils.edge_ontop_spec import select_graph

            graph = select_graph(self.relation_runtime.graph, env_ids)
            at_edges, goal_slots, active = at_goal_marker_slots(graph)
            self._marker_pos[env_ids, :, 2] = torch.where(
                active, self._tar_pos[env_ids, :, 2], 20.0
            )
            shown = (
                torch.ones_like(env_ids, dtype=torch.bool)
                if self.viewer is not None
                else env_ids == 0
            )
            rows, edges = (at_edges & shown[:, None]).nonzero(as_tuple=True)
            assignments = (
                torch.stack(
                    (
                        env_ids[rows],
                        goal_slots[rows, edges],
                        graph.edge_owner[rows, edges],
                    ),
                    dim=-1,
                )
                .cpu()
                .tolist()
            )
            for env_id, goal_id, owner in assignments:
                marker = self._marker_handles[env_id * self.num_agents + goal_id]
                self.gym.set_rigid_body_color(
                    self.envs[env_id],
                    marker,
                    0,
                    gymapi.MESH_VISUAL,
                    self._agent_color(owner),
                )
            ids.append(self._marker_actor_ids[env_ids].contiguous().view(-1))
        self._platform_states[env_ids, :, 3:6] = 0.0
        self._platform_states[env_ids, :, 6] = 1.0
        self._platform_states[env_ids, :, 7:13] = 0.0
        self._tar_platform_states[env_ids, :, 3:6] = 0.0
        self._tar_platform_states[env_ids, :, 6] = 1.0
        self._tar_platform_states[env_ids, :, 7:13] = 0.0
        ids.extend(
            [
                self._platform_actor_ids[env_ids].contiguous().view(-1),
                self._tar_platform_actor_ids[env_ids].contiguous().view(-1),
            ]
        )
        ids = torch.cat(ids)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(ids),
            len(ids),
        )
        return

    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)
        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        num_key_bodies = len(key_bodies)
        if asset_file == "mjcf/amp_humanoid.xml":
            self._num_amp_obs_per_step = (
                13 + self._dof_obs_size + 28 + 3 * num_key_bodies
            )
        elif asset_file in (
            "mjcf/phys_humanoid.xml",
            "mjcf/phys_humanoid_v2.xml",
            "mjcf/phys_humanoid_v3.xml",
        ):
            self._num_amp_obs_per_step = (
                13 + self._dof_obs_size + 28 + 2 * 2 + 3 * num_key_bodies
            )
        else:
            print("Unsupported character config file: {s}".format(asset_file))
            assert False
        if self.cfg["env"].get("ampTaskConditioning", False):
            self._num_amp_obs_per_step += 3
        return

    def get_num_amp_obs(self):
        return self._num_amp_obs_steps * self._num_amp_obs_per_step

    def _load_motion(self, motion_file):
        assert self._dof_offsets[-1] == self.num_dof
        ext = os.path.splitext(motion_file)[1]
        if ext == ".yaml":
            with open(os.path.join(os.getcwd(), motion_file), "r") as f:
                motion_config = yaml.load(f, Loader=yaml.SafeLoader)
            self._skill_categories = list(motion_config["motions"].keys())
            self._motion_lib = {}
            for skill in self._skill_categories:
                self._motion_lib[skill] = MotionLib(
                    motion_file=motion_file,
                    skill=skill,
                    dof_body_ids=self._dof_body_ids,
                    dof_offsets=self._dof_offsets,
                    key_body_ids=self._key_body_ids.cpu().numpy(),
                    device=self.device,
                )
        else:
            raise NotImplementedError
        return

    def _amp_family(self, env_ids=None, slot_agent=None):
        from utils.unified_training import family_from_templates

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if slot_agent is None:
            slot_agent = torch.arange(self.num_agents, device=self.device).repeat(
                len(env_ids)
            )
            env_ids = env_ids.repeat_interleave(self.num_agents)
        return family_from_templates(
            classify_templates(
                self.relation_runtime.graph, env_ids, slot_agent, with_climb=True
            )
        )

    def _fetch_conditioned_amp_demo(self, num_samples):
        from utils.unified_training import EXPERT_PROBS, append_family

        family_probs = (0.65, 0.1, 0.25)
        family = torch.multinomial(
            torch.tensor(family_probs, device=self.device),
            num_samples,
            replacement=True,
        )
        weights = torch.tensor(EXPERT_PROBS, device=self.device)[family]
        skills = torch.multinomial(weights, 1).squeeze(-1)
        result = torch.empty(num_samples, self.get_num_amp_obs(), device=self.device)
        for uid, name in enumerate(self.cfg["env"]["skill"]):
            indices = (skills == uid).nonzero(as_tuple=False).flatten()
            if not len(indices):
                continue
            lib = self._motion_lib[name]
            motions = lib.sample_motions(len(indices))
            truncate = self.dt * (self._num_amp_obs_steps - 1)
            times = lib.sample_time(motions, truncate_time=truncate) + truncate
            motion_obs = self.build_amp_obs_demo(motions, times, lib).reshape(
                len(indices), self._num_amp_obs_steps, -1
            )
            labels = family[indices, None].expand(-1, self._num_amp_obs_steps)
            result[indices] = append_family(motion_obs, labels).flatten(1)
        return result

    def fetch_amp_obs_demo(self, num_samples):
        return self._fetch_conditioned_amp_demo(num_samples)

    def build_amp_obs_demo(self, motion_ids, motion_times0, motion_lib):
        dt = self.dt
        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps])
        motion_times = motion_times0.unsqueeze(-1)
        time_steps = -dt * torch.arange(0, self._num_amp_obs_steps, device=self.device)
        motion_times = motion_times + time_steps
        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = (
            motion_lib.get_motion_state(motion_ids, motion_times)
        )
        amp_obs_demo = build_amp_observations(
            root_pos,
            root_rot,
            root_vel,
            root_ang_vel,
            dof_pos,
            dof_vel,
            key_pos,
            self._local_root_obs,
            self._root_height_obs,
            self._dof_obs_size,
            self._dof_offsets,
        )
        return amp_obs_demo

    def _build_amp_obs_demo_buf(self, num_samples):
        self._amp_obs_demo_buf = torch.zeros(
            (num_samples, self._num_amp_obs_steps, self._num_amp_obs_per_step),
            device=self.device,
            dtype=torch.float32,
        )
        return

    def _init_amp_obs(self, env_ids):
        self._compute_amp_observations(env_ids)
        if self._reset_default_slots is not None:
            self._init_amp_obs_default(*self._reset_default_slots)
        for sk_name in self._skill:
            if self._reset_ref_slots.get(sk_name) is not None:
                self._init_amp_obs_ref(
                    self._reset_ref_slots[sk_name],
                    self._reset_ref_motion_ids[sk_name],
                    self._reset_ref_motion_times[sk_name],
                    sk_name,
                )
        return

    def _init_amp_obs_default(self, slot_env, slot_agent):
        curr = self._curr_amp_obs_buf[slot_env, slot_agent].unsqueeze(-2)
        self._hist_amp_obs_buf[slot_env, slot_agent] = curr
        return

    def _init_amp_obs_ref(self, slots, motion_ids, motion_times, skill_name):
        slot_env, slot_agent = slots
        dt = self.dt
        motion_ids = torch.tile(
            motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps - 1]
        )
        motion_times = motion_times.unsqueeze(-1)
        time_steps = -dt * (
            torch.arange(0, self._num_amp_obs_steps - 1, device=self.device) + 1
        )
        motion_times = motion_times + time_steps
        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = (
            self._motion_lib[skill_name].get_motion_state(motion_ids, motion_times)
        )
        amp_obs_demo = build_amp_observations(
            root_pos,
            root_rot,
            root_vel,
            root_ang_vel,
            dof_pos,
            dof_vel,
            key_pos,
            self._local_root_obs,
            self._root_height_obs,
            self._dof_obs_size,
            self._dof_offsets,
        )
        from utils.unified_training import append_family

        labels = self._amp_family(slot_env, slot_agent).repeat_interleave(
            self._num_amp_obs_steps - 1
        )
        amp_obs_demo = append_family(amp_obs_demo, labels)
        self._hist_amp_obs_buf[slot_env, slot_agent] = amp_obs_demo.view(
            slot_env.shape[0], self._num_amp_obs_steps - 1, self._num_amp_obs_per_step
        )
        return

    def _update_hist_amp_obs(self, env_ids=None):
        for i in reversed(range(self._amp_obs_buf.shape[2] - 1)):
            self._amp_obs_buf[:, :, i + 1] = self._amp_obs_buf[:, :, i]
        return

    def _compute_amp_observations(self, env_ids=None):
        N, M, nb, nd = (self.num_envs, self.num_agents, self.num_bodies, self.num_dof)
        if env_ids is None:
            B = N * M
            body_pos = self._rigid_body_pos.reshape(B, nb, 3)
            body_rot = self._rigid_body_rot.reshape(B, nb, 4)
            body_vel = self._rigid_body_vel.reshape(B, nb, 3)
            body_ang_vel = self._rigid_body_ang_vel.reshape(B, nb, 3)
            dof_pos = self._dof_pos.reshape(B, nd)
            dof_vel = self._dof_vel.reshape(B, nd)
            obs = build_amp_observations(
                body_pos[:, 0, :],
                body_rot[:, 0, :],
                body_vel[:, 0, :],
                body_ang_vel[:, 0, :],
                dof_pos,
                dof_vel,
                body_pos[:, self._key_body_ids, :],
                self._local_root_obs,
                self._root_height_obs,
                self._dof_obs_size,
                self._dof_offsets,
            )
            from utils.unified_training import append_family

            obs = append_family(obs, self._amp_family())
            self._curr_amp_obs_buf[:] = obs.view(N, M, self._num_amp_obs_per_step)
        else:
            K = env_ids.shape[0]
            B = K * M
            kin = self._kinematic_humanoid_rigid_body_states[env_ids].reshape(B, nb, 13)
            dof_pos = self._dof_pos[env_ids].reshape(B, nd)
            dof_vel = self._dof_vel[env_ids].reshape(B, nd)
            obs = build_amp_observations(
                kin[:, 0, 0:3],
                kin[:, 0, 3:7],
                kin[:, 0, 7:10],
                kin[:, 0, 10:13],
                dof_pos,
                dof_vel,
                kin[:, self._key_body_ids, 0:3],
                self._local_root_obs,
                self._root_height_obs,
                self._dof_obs_size,
                self._dof_offsets,
            )
            from utils.unified_training import append_family

            obs = append_family(obs, self._amp_family(env_ids))
            self._curr_amp_obs_buf[env_ids] = obs.view(K, M, self._num_amp_obs_per_step)
        return

    def _sample_arena_xy(self, centers):
        """Uniform samples by area inside the configured arena disc."""
        count = centers.shape[0]
        angle = torch.rand(count, device=self.device) * (2.0 * np.pi)
        radius = (
            torch.sqrt(torch.rand(count, device=self.device)) * self._box_spawn_radius
        )
        offset = torch.stack(
            [torch.cos(angle), torch.sin(angle)], dim=-1
        ) * radius.unsqueeze(-1)
        return centers + offset

    def _reset_all_boxes_random_arena(self, env_ids):
        """Inference reset: every physical box is independent of every agent owner."""
        K, O = (env_ids.shape[0], self.num_physical_objects)
        centers = self._env_origins[env_ids, 0:2]
        agent_xy = self._humanoid_root_states[env_ids, :, 0:2]
        positions = torch.zeros(K, O, 2, device=self.device)
        for object_id in range(O):
            candidate = self._sample_arena_xy(centers)
            for _ in range(64):
                near_agent = (
                    torch.sum((candidate.unsqueeze(1) - agent_xy) ** 2, dim=-1)
                    .min(dim=-1)
                    .values
                    < self._box_min_agent_dist**2
                )
                if object_id > 0:
                    near_box = (
                        torch.sum(
                            (candidate.unsqueeze(1) - positions[:, :object_id]) ** 2,
                            dim=-1,
                        )
                        .min(dim=-1)
                        .values
                        < self._box_min_box_dist**2
                    )
                    invalid = torch.logical_or(near_agent, near_box)
                else:
                    invalid = near_agent
                if not torch.any(invalid):
                    break
                candidate[invalid] = self._sample_arena_xy(centers[invalid])
            positions[:, object_id] = candidate
        self._box_states[env_ids, :, 0:2] = positions
        self._box_states[env_ids, :, 2] = self._box_size[env_ids, :, 2] / 2.0
        axis = torch.zeros(K * O, 3, device=self.device)
        axis[:, 2] = 1.0
        coeff = 1.0 if self._reset_random_rot else 0.0
        angle = torch.rand(K * O, device=self.device) * (2.0 * np.pi) * coeff
        self._box_states[env_ids, :, 3:7] = quat_from_angle_axis(angle, axis).view(
            K, O, 4
        )
        self._box_states[env_ids, :, 7:13] = 0.0
        self._ground_stage1_standalone_targets(env_ids)
        return


@torch.jit.script
def build_amp_observations(
    root_pos,
    root_rot,
    root_vel,
    root_ang_vel,
    dof_pos,
    dof_vel,
    key_body_pos,
    local_root_obs,
    root_height_obs,
    dof_obs_size,
    dof_offsets,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, int, List[int]) -> Tensor
    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    if local_root_obs:
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = torch_utils.quat_to_tan_norm(root_rot_obs)
    if not root_height_obs:
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h
    local_root_vel = quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = quat_rotate(heading_rot, root_ang_vel)
    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(
        local_key_body_pos.shape[0] * local_key_body_pos.shape[1],
        local_key_body_pos.shape[2],
    )
    flat_heading_rot = heading_rot_expand.view(
        heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
        heading_rot_expand.shape[2],
    )
    local_end_pos = quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(
        local_key_body_pos.shape[0],
        local_key_body_pos.shape[1] * local_key_body_pos.shape[2],
    )
    dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets)
    obs = torch.cat(
        (
            root_h_obs,
            root_rot_obs,
            local_root_vel,
            local_root_ang_vel,
            dof_obs,
            dof_vel,
            flat_local_key_pos,
        ),
        dim=-1,
    )
    return obs
