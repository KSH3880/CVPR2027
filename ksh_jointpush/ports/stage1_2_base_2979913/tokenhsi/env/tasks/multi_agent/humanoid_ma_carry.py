import math
from enum import Enum
import numpy as np
import torch
from isaacgym import gymapi
from env.tasks.multi_agent.humanoid_ma import HumanoidMA
from env.tasks.multi_agent.scene_features import object_task_owners
from utils.relation_task_spec import validate_relation_config
from env.tasks.multi_agent.collision_reward import (
    agent_collision_config,
    compute_agent_cpa_collision_penalty,
)
from env.tasks.multi_agent.edge_context_task import EdgeContextTaskMixin
from utils.edge_ontop_spec import batched
from utils.edge_stage1_spec import compile_stage1_graph, ground_standalone_interaction_targets
from utils.edge_scenario_spec import PRESETS as SCENARIO_PRESETS, scenario_templates
from env.tasks.multi_agent.edge_ontop_task import SampledOnTopTaskMixin


from env.tasks.multi_agent.scene_observation import SceneObservationMixin
from env.tasks.multi_agent.reference_motion import ReferenceMotionMixin
from env.tasks.multi_agent.scene_reset import SceneResetMixin


class HumanoidMACarry(SceneObservationMixin, ReferenceMotionMixin, SceneResetMixin,
                      SampledOnTopTaskMixin, EdgeContextTaskMixin, HumanoidMA):
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
        self._coupled = cfg["env"].get("stage2JointPush") is not None
        self._stage2_push = cfg["env"].get("stage2Policy") is not None
        self._before = cfg["env"].get("before") is not None
        self._relation_cfg = cfg["env"]["relationReward"]
        validate_relation_config(self._relation_cfg)
        from utils.unified_training import validate_unified_env

        validate_unified_env(cfg["env"])
        default_preset = (
            "holding_at" if cfg["args"].test or cfg["args"].eval else "random_scenario"
        )
        from utils.task_role_spec import task_preset

        self._task_graph_preset = task_preset(
            getattr(cfg["args"], "task_graph", "") or default_preset
        )
        self._task_role_swap = bool(getattr(cfg["args"], "task_role_swap", False))
        allowed_presets = SCENARIO_PRESETS
        if cfg["env"].get("jointPush") is not None:
            allowed_presets = (*allowed_presets, "push")
        if self._before:
            from utils.before_spec import PRESETS
            allowed_presets = (*PRESETS, "push") if cfg["env"].get("jointPush") is not None else PRESETS
        if self._coupled:
            from utils.coupled_push_spec import PRESETS
            allowed_presets = PRESETS
        if self._task_graph_preset not in allowed_presets or getattr(
            cfg["args"], "task_camera", "stack"
        ) not in ("stack", "agent"):
            raise ValueError("Unknown Stage1 graph preset or camera mode")
        if not (cfg["args"].test or cfg["args"].eval):
            expected_shape = (2, 4)
            if self._before or self._coupled:
                expected_shape = (cfg["env"]["numAgents"], cfg["env"]["numObjects"])
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
        self.num_physical_objects = self.num_objects
        if self._before:
            self.num_physical_objects = (max(5*num_agents, self.num_objects+num_agents)
                if cfg["env"].get("jointPush") is not None else max(4*num_agents, self.num_objects))
        if self._coupled:
            self.num_physical_objects = cfg["env"]["stage2JointPush"]["physical_boxes"]
        assert (
            self.num_objects >= num_agents
        ), "numObjects ({}) must be >= numAgents ({})".format(
            self.num_objects, num_agents
        )
        self._generic_scenario_eval = (not self._before and not self._coupled) and cfg["args"].eval and (
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
        compile_stage1_graph(self._relation_graph_spec, num_agents, self.num_objects)
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
        self._ground_only = self._before and box_cfg['reset']['platformMode'] == 'ground_only'
        self._reset_random_height = not self._ground_only
        from utils.stage1_config import uses_fixed_platforms

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
            if cfg["env"].get("jointPush") is not None:
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

        self._size_template_probabilities = (self._joint_probabilities if cfg["env"].get("jointPush") is not None and not self._before and not self._coupled
            else task_probabilities(self._box_size[:, :M], cfg["env"]["box"]["build"]["taskSizeRanges"]))
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
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 1.0
        asset_options.fix_base_link = True
        asset_options.disable_gravity = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        self._platform_height = 0.02
        self._platform_inactive_height = -20.0 if self._ground_only else 20.0
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

        self._box_size = sample_sizes(
            N,
            self.device,
            self._task_graph_preset,
            self.cfg["env"]["box"]["build"]["taskSizeRanges"],
            num_agents=self.num_agents,
            num_objects=O
        ).reshape(num_boxes, 3)
        if (getattr(self.cfg["args"], "eval_protocol", "") == "stage1_single"
                and self.cfg["args"].eval_single_task == "AT"):
            from utils.evaluation_protocol import single_at_sizes
            self._box_size[:] = single_at_sizes(N, self.device)
        self._box_scale = self._box_size / torch.tensor(
            self._build_base_size, device=self.device
        )
        from utils.stage1_geometry import max_bound_stack_height
        if getattr(self.cfg["args"], "eval_protocol", "") == "stage1_single":
            heights = self._box_size.reshape(N, O, 3)[..., 2]
            stack_height = heights.sum(-1) if O == 2 else heights.amax(-1)
        else:
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
        # The marker asset has a 10 cm visual radius. PUSH scenes span tens of
        # metres, so show a 20 cm target rather than the ordinary 3 cm dot.
        marker_scale = 2.0 if self.cfg["env"].get("jointPush") is not None else 0.3
        self.gym.set_actor_scale(env_ptr, marker_handle, marker_scale)
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
            owners_by_box = {}
            for logical, owners in object_task_owners(self.relation_runtime.graph, env_id).items():
                physical_box = int(self._logical_box_order[env_id, logical])
                owners_by_box.setdefault(physical_box, set()).update(owners)
            for physical_box in range(self.num_physical_objects):
                box_handle = self._box_handles[env_id * self.num_physical_objects + physical_box]
                owners = owners_by_box.get(physical_box, [])
                color = (
                    shared_color
                    if len(owners) > 1
                    else self._agent_color(next(iter(owners))) if owners else unassigned_color
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


    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)
        self._num_amp_obs_per_step = 13 + self._dof_obs_size + 32 + 3*len(key_bodies) + getattr(self, "_amp_num_families", 3)


