# Multi-agent humanoid base class.
#
# Generalizes env/tasks/humanoid.py from "one humanoid per env" to "M humanoids per env".
# The original class is left untouched; this one re-implements __init__ because the base
# hardcodes single-humanoid tensor shapes inline (e.g. dof_force_tensor.view(num_envs, num_dof)),
# which raises before any subclass hook could run.
#
# Actor layout inside every env (order matters, it defines the root-state indexing):
#     [humanoid_0 .. humanoid_{M-1}, <task actors...>]
# Humanoids are created first so their rigid bodies occupy [0, M * num_bodies).
#
# Per-agent tensors are kept as (num_envs, M, ...) *views* of the sim tensors, so in-place
# writes still propagate to Isaac Gym. A slot is addressed by a pair of index tensors
# (env_ids, agent_ids); resets always happen at env granularity, so the two are built with
# _expand_slots().
#
# Observations follow the ego-centric, ego-first convention: one row per (env, agent),
# laid out env-major / agent-minor to match rl_games' num_agents contract.

import math
import os

import numpy as np
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from utils import torch_utils

from env.tasks.base_task import BaseTask
from env.tasks.humanoid import Humanoid


class HumanoidMA(Humanoid):

    _AGENT_COLOR_PALETTE = (
        (0.54, 0.85, 0.20),
        (0.28, 0.55, 0.90),
        (0.90, 0.60, 0.20),
        (0.80, 0.35, 0.75),
    )

    @classmethod
    def _agent_color(cls, agent_id):
        color = cls._AGENT_COLOR_PALETTE[agent_id % len(cls._AGENT_COLOR_PALETTE)]
        return gymapi.Vec3(*color)

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine

        self.num_agents = cfg["env"].get("numAgents", 1)
        assert self.num_agents >= 1

        # How each entity token is expressed:
        #
        # "global": one omniscient, centralized frame -- every entity is described relative
        #           to the env centre, with global orientations, scaled by the env size.
        #           There is no owner/self notion anywhere, cross-agent geometry is exact,
        #           and because every row then holds the same tokens up to a permutation,
        #           the transformer's equivariance makes row i's token 0 equal to row 0's
        #           token i: one scene encoding really does yield all M actions.
        #           Cost: the SE(2) (heading) invariance that AMP humanoid control relies on
        #           is no longer free -- the policy must learn to map a world-frame scene to
        #           body-frame PD targets.
        # NOTE: "owner" has the same one-encoding-for-all-agents property as "global" (its
        #       token contents also do not depend on who is observing), while keeping heading
        #       invariance, which is why it is the default. Only "ego" gives that property up.
        # "owner":  every entity is described in the frame of the agent that owns it. Keeps
        #           heading invariance and puts every token in exactly the distribution a
        #           single-agent policy would see, at the cost of cross-agent geometry being
        #           carried only by the arena-pose features.
        # "ego":    every entity re-expressed in the *observing* agent's frame. M times the
        #           entity transforms; teammates' objects land in a frame the single-agent
        #           tokenizer never saw.
        self._obs_frame = cfg["env"].get("obsFrame", "owner")
        assert self._obs_frame in ("global", "owner", "ego")

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

        # spawn offsets are needed by _build_env, which runs inside BaseTask.__init__
        self._build_agent_spawn_offsets(device_type, device_id)

        video_cfg = cfg["env"].get("video", {})
        self._video_enabled = bool(video_cfg.get("enable", False))
        self._video_every = int(video_cfg.get("everyNEpochs", 50))
        self._video_num_frames = int(video_cfg.get("numFrames", 300))
        self._video_w = int(video_cfg.get("width", 1024))
        self._video_h = int(video_cfg.get("height", 768))
        self._video_cam_dist = float(video_cfg.get("camDistance", 3.0))
        self._video_cam_height = float(video_cfg.get("camHeight", 1.5))
        self._video_fov = float(video_cfg.get("fov", 45.0))
        # "all" keeps every humanoid of env 0 in frame; "agent0" follows one character
        self._video_focus = video_cfg.get("focus", "all")
        self._recording = False
        self._video_frames = []
        self._video_path = None
        self._video_cam = None

        # off-screen camera sensors work headless, but only if a graphics device is kept
        BaseTask.__init__(self, cfg=self.cfg, enable_camera_sensors=self._video_enabled)

        self.dt = self.control_freq_inv * sim_params.dt

        # ---- sim tensors -------------------------------------------------------------
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        N = self.num_envs
        M = self.num_agents
        nb = self.num_bodies
        nd = self.num_dof

        sensors_per_env = 2 * M
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(N, sensors_per_env * 6)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(N, M, nd)

        self._root_states = gymtorch.wrap_tensor(actor_root_state)
        num_actors = self.get_num_actors_per_env()

        # humanoids are actors [0, M) of every env
        self._humanoid_root_states = self._root_states.view(N, num_actors, actor_root_state.shape[-1])[:, :M, :]
        self._initial_humanoid_root_states = self._humanoid_root_states.clone()
        self._initial_humanoid_root_states[..., 7:13] = 0

        self._humanoid_actor_ids = \
            (num_actors * torch.arange(N, device=self.device, dtype=torch.int32)).unsqueeze(-1) \
            + torch.arange(M, device=self.device, dtype=torch.int32).unsqueeze(0)  # (N, M)

        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        dofs_per_env = self._dof_state.shape[0] // N
        assert dofs_per_env == M * nd, \
            "only the humanoids may own DOFs (got {} dofs/env, expected {})".format(dofs_per_env, M * nd)
        dof_state_reshaped = self._dof_state.view(N, M, nd, 2)
        self._dof_pos = dof_state_reshaped[..., 0]
        self._dof_vel = dof_state_reshaped[..., 1]

        self._initial_dof_pos = torch.zeros_like(self._dof_pos, device=self.device, dtype=torch.float)
        self._initial_dof_vel = torch.zeros_like(self._dof_vel, device=self.device, dtype=torch.float)

        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        bodies_per_env = self._rigid_body_state.shape[0] // N
        rigid_body_state_reshaped = self._rigid_body_state.view(N, bodies_per_env, 13)[:, :M * nb, :].view(N, M, nb, 13)

        self._initial_humanoid_rigid_body_states = rigid_body_state_reshaped.clone()
        self._initial_humanoid_rigid_body_states[..., 7:13] = 0

        self._rigid_body_pos = rigid_body_state_reshaped[..., 0:3]
        self._rigid_body_rot = rigid_body_state_reshaped[..., 3:7]
        self._rigid_body_vel = rigid_body_state_reshaped[..., 7:10]
        self._rigid_body_ang_vel = rigid_body_state_reshaped[..., 10:13]

        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._contact_forces = contact_force_tensor.view(N, bodies_per_env, 3)[:, :M * nb, :].view(N, M, nb, 3)

        # ---- per-slot buffers --------------------------------------------------------
        # obs / rew / terminate are per (env, agent); reset / progress stay per env.
        self.obs_buf = torch.zeros((N * M, self.get_obs_size()), device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(N * M, device=self.device, dtype=torch.float)
        self._terminate_buf = torch.ones(N * M, device=self.device, dtype=torch.long)

        self._build_termination_heights()

        contact_bodies = self.cfg["env"]["contactBodies"]
        self._key_body_ids = self._build_key_body_ids_tensor(key_bodies)
        self._contact_body_ids = self._build_contact_body_ids_tensor(contact_bodies)

        # perm[i, k] = index of the k-th entity as seen by ego i (ego-first rotation)
        ar = torch.arange(M, device=self.device)
        self._ego_perm = (ar.unsqueeze(-1) + ar.unsqueeze(0)) % M  # (M, M)

        if self.viewer is not None:
            self._init_camera()
            # Start multi-agent viewer sessions in free-camera mode.  Press F to
            # toggle following agent 0 when desired.
            self._camera_follow_flag = self.cfg["env"].get("cameraFollow", False)

        return

    # ------------------------------------------------------------------ sizes

    def get_humanoid_obs_size(self):
        # base _num_obs drops the (always-zero) root position; the relative form keeps it
        # so that every humanoid token - ego or not - has an identical layout, and appends
        # the arena-local pose (xy + heading cos/sin) so that agents can locate each other
        # even when each of them is described in its own frame.
        return self._num_obs + 3 + 4

    def _global_frame(self, num_rows, env_ids=None):
        """Observer frame for "global" mode: the env centre, with no rotation."""
        origins = self._env_origins if env_ids is None else self._env_origins[env_ids]
        pos = origins.unsqueeze(1).expand(-1, self.num_agents, -1).reshape(-1, 3)

        rot = torch.zeros(pos.shape[0], 4, device=self.device, dtype=torch.float)
        rot[:, 3] = 1.0  # identity quaternion -> keep world orientation
        return pos, rot

    def _arena_pose_feats(self, root_pos, root_rot, env_ids=None):
        """(K, 4): root xy relative to the env origin, plus heading cos/sin."""
        origins = self._env_origins if env_ids is None else self._env_origins[env_ids]
        origins = origins.unsqueeze(1).expand(-1, self.num_agents, -1).reshape(-1, 3)

        xy = (root_pos[:, 0:2] - origins[:, 0:2]) / self._arena_scale
        heading = torch_utils.calc_heading(root_rot)
        return torch.cat([xy, torch.cos(heading).unsqueeze(-1), torch.sin(heading).unsqueeze(-1)], dim=-1)

    def get_obs_size(self):
        return self.num_agents * self.get_humanoid_obs_size()

    def get_action_size(self):
        return self._num_actions

    def get_number_of_agents(self):
        return self.num_agents

    # ------------------------------------------------------------------ env building

    def _build_agent_spawn_offsets(self, device_type, device_id):
        device = "cpu"
        if device_type in ("cuda", "GPU"):
            device = "cuda:" + str(device_id)

        M = self.num_agents
        if M == 1:
            offsets = torch.zeros(1, 3)
        else:
            radius = self.cfg["env"].get("agentSpawnRadius", 2.5)
            ang = torch.arange(M, dtype=torch.float) * (2.0 * np.pi / M)
            offsets = torch.stack([radius * torch.cos(ang), radius * torch.sin(ang), torch.zeros(M)], dim=-1)

        self._agent_spawn_offsets = offsets.to(device)
        return

    def _create_envs(self, num_envs, spacing, num_per_row):
        self.humanoid_handles_all = []
        super()._create_envs(num_envs, spacing, num_per_row)

        # root states are global; the env origin turns them into arena-local coordinates,
        # which is the shared reference the humanoid tokens need in "owner" frame mode
        origins = [self.gym.get_env_origin(env_ptr) for env_ptr in self.envs]
        self._env_origins = to_torch([[o.x, o.y, o.z] for o in origins],
                                     device=self.device, dtype=torch.float)  # (N, 3)
        self._arena_scale = float(spacing)
        self._setup_video_camera()
        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        col_group = env_id
        col_filter = self._get_humanoid_collision_filter()
        segmentation_id = 0

        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        if (asset_file == "mjcf/amp_humanoid.xml"):
            self._char_h = 0.89
        elif (asset_file == "mjcf/phys_humanoid.xml") or (asset_file == "mjcf/phys_humanoid_v2.xml"):
            self._char_h = 0.92
        elif (asset_file == "mjcf/phys_humanoid_v3.xml") or (asset_file == "mjcf/phys_humanoid_v3_box_foot.xml"):
            self._char_h = 0.94
        else:
            print("Unsupported character config file: {s}".format(asset_file))
            assert (False)

        handles = []
        for a in range(self.num_agents):
            offset = self._agent_spawn_offsets[a]

            start_pose = gymapi.Transform()
            start_pose.p = gymapi.Vec3(float(offset[0]), float(offset[1]), self._char_h)
            start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

            handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose,
                                           "humanoid_{}".format(a), col_group, col_filter, segmentation_id)
            self.gym.enable_actor_dof_force_sensors(env_ptr, handle)

            col = self._agent_color(a)
            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL, col)

            if (self._pd_control):
                dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
                dof_prop["driveMode"] = gymapi.DOF_MODE_POS
                self.gym.set_actor_dof_properties(env_ptr, handle, dof_prop)

            handles.append(handle)

        # humanoid_handles keeps one handle per env so the base helpers
        # (_build_key_body_ids_tensor, set_char_color, ...) keep working
        self.humanoid_handles.append(handles[0])
        self.humanoid_handles_all.append(handles)
        return

    # ------------------------------------------------------------------ video

    def _setup_video_camera(self):
        """One off-screen camera on env 0, used to record training progress."""
        if not self._video_enabled:
            return

        props = gymapi.CameraProperties()
        props.width = self._video_w
        props.height = self._video_h
        props.horizontal_fov = self._video_fov  # the default is very wide; characters end up tiny
        props.enable_tensors = False
        self._video_cam = self.gym.create_camera_sensor(self.envs[0], props)
        return

    def _video_focus_points(self):
        """Everything the recording should keep in frame, for env 0. (K, 3)"""
        if self._video_focus == "agent0":
            return self._humanoid_root_states[0, 0:1, 0:3]
        return self._humanoid_root_states[0, :, 0:3]

    def request_video(self, path):
        """Start recording the next _video_num_frames steps of env 0 into `path`."""
        if not self._video_enabled or self._recording:
            return
        self._video_path = path
        self._video_frames = []
        self._recording = True
        return

    def _capture_video_frame(self):
        if self._video_cam is None:
            return

        # Frame the crew of env 0. Distance and height grow with how far apart they are,
        # so M=1 gets a close shot and M>1 pulls back just enough to keep everyone in.
        pts = self._video_focus_points()
        center = pts.mean(dim=0)
        radius = float((pts - center).norm(dim=-1).max())

        dist = self._video_cam_dist + 1.2 * radius
        height = self._video_cam_height + 0.5 * radius

        # set_camera_location takes ENV-LOCAL coordinates when an env is passed, while the
        # root states are global -- subtract the env origin or the camera lands in the void.
        center = (center - self._env_origins[0]).cpu().numpy()
        cam_pos = gymapi.Vec3(float(center[0]) - dist,
                              float(center[1]) - dist,
                              float(center[2]) + height)
        cam_tgt = gymapi.Vec3(float(center[0]), float(center[1]), float(center[2]) + 0.4)
        self.gym.set_camera_location(self._video_cam, self.envs[0], cam_pos, cam_tgt)

        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        img = self.gym.get_camera_image(self.sim, self.envs[0], self._video_cam, gymapi.IMAGE_COLOR)
        self._video_frames.append(img.reshape(self._video_h, self._video_w, 4)[:, :, :3].copy())

        if len(self._video_frames) >= self._video_num_frames:
            self._write_video()
        return

    def _write_video(self):
        import cv2

        self._recording = False
        frames, path = self._video_frames, self._video_path
        self._video_frames = []
        if len(frames) == 0 or path is None:
            return

        os.makedirs(os.path.dirname(path), exist_ok=True)
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 max(1, round(1.0 / self.dt)), (w, h))
        for f in frames:
            writer.write(f[:, :, ::-1])  # RGB -> BGR
        writer.release()
        print("[MA] wrote {} ({} frames)".format(path, len(frames)), flush=True)
        return

    # ------------------------------------------------------------------ slot helpers

    def _expand_slots(self, env_ids):
        """(K,) env ids -> ((K*M,) env ids, (K*M,) agent ids), env-major / agent-minor."""
        M = self.num_agents
        slot_env = env_ids.repeat_interleave(M)
        slot_agent = torch.arange(M, device=self.device, dtype=torch.long).repeat(env_ids.shape[0])
        return slot_env, slot_agent

    def _flat_slot_ids(self, env_ids):
        """(K,) env ids -> (K*M,) flat row indices into obs_buf / rew_buf."""
        M = self.num_agents
        return (env_ids.unsqueeze(-1) * M + torch.arange(M, device=self.device, dtype=torch.long)).view(-1)

    # ------------------------------------------------------------------ observations

    def _compute_observations(self, env_ids=None):
        obs = self._compute_humanoid_obs(env_ids)

        if (env_ids is None):
            self.obs_buf[:] = obs.view(self.num_envs * self.num_agents, -1)
        else:
            self.obs_buf[self._flat_slot_ids(env_ids)] = obs.view(env_ids.shape[0] * self.num_agents, -1)
        return

    def _compute_humanoid_obs(self, env_ids=None):
        """Returns (B, M, M * humanoid_obs_size): for every ego, every humanoid, ego-first."""
        if (env_ids is None):
            body_pos = self._rigid_body_pos
            body_rot = self._rigid_body_rot
            body_vel = self._rigid_body_vel
            body_ang_vel = self._rigid_body_ang_vel
        else:
            kin = self._kinematic_humanoid_rigid_body_states[env_ids]
            body_pos = kin[..., 0:3]
            body_rot = kin[..., 3:7]
            body_vel = kin[..., 7:10]
            body_ang_vel = kin[..., 10:13]

        return self._build_humanoid_tokens(body_pos, body_rot, body_vel, body_ang_vel, env_ids)

    def _build_humanoid_tokens(self, body_pos, body_rot, body_vel, body_ang_vel, env_ids=None):
        B = body_pos.shape[0]
        M = self.num_agents
        nb = self.num_bodies

        own_root_pos = body_pos[:, :, 0, :].reshape(-1, 3)                      # (B*M, 3)
        own_root_rot = body_rot[:, :, 0, :].reshape(-1, 4)                      # (B*M, 4)
        own_heading = torch_utils.calc_heading_quat_inv(own_root_rot)           # (B*M, 4)

        arena = self._arena_pose_feats(own_root_pos, own_root_rot, env_ids)     # (B*M, 4)

        if self._obs_frame in ("global", "owner"):
            # one transform per humanoid; a row is then a pure gather.
            #   global -> everybody described in the env-centred world frame
            #   owner  -> everybody described in its own heading frame
            if self._obs_frame == "global":
                frame_pos, frame_rot = self._global_frame(B * M, env_ids)
            else:
                frame_pos, frame_rot = own_root_pos, own_heading

            obs = compute_humanoid_observations_rel_max(
                body_pos.reshape(-1, nb, 3), body_rot.reshape(-1, nb, 4),
                body_vel.reshape(-1, nb, 3), body_ang_vel.reshape(-1, nb, 3),
                frame_pos, frame_rot,
                self._local_root_obs_policy, self._root_height_obs_policy)

            if self._obs_frame == "global":
                # positions live in [-1, 1]-ish arena units; the entity-wise normaliser
                # would rescale them anyway, this just keeps the raw obs readable
                obs[:, 1:1 + 3 * nb] = obs[:, 1:1 + 3 * nb] / self._arena_scale

            obs = torch.cat([obs, arena], dim=-1).view(B, M, -1)                # (B, M, size)
            obs = obs[:, self._ego_perm]                                        # (B, M, M, size)
            return obs.reshape(B, M, M * self.get_humanoid_obs_size())

        # "ego": M transforms per humanoid, one per observer
        perm = self._ego_perm
        sub_pos = body_pos[:, perm].reshape(-1, nb, 3)                          # (B*M*M, nb, 3)
        sub_rot = body_rot[:, perm].reshape(-1, nb, 4)
        sub_vel = body_vel[:, perm].reshape(-1, nb, 3)
        sub_ang_vel = body_ang_vel[:, perm].reshape(-1, nb, 3)

        ego_pos = own_root_pos.view(B, M, 3).unsqueeze(2).expand(B, M, M, 3).reshape(-1, 3)
        ego_rot = own_heading.view(B, M, 4).unsqueeze(2).expand(B, M, M, 4).reshape(-1, 4)

        obs = compute_humanoid_observations_rel_max(
            sub_pos, sub_rot, sub_vel, sub_ang_vel, ego_pos, ego_rot,
            self._local_root_obs_policy, self._root_height_obs_policy)

        sub_arena = arena.view(B, M, 4)[:, perm].reshape(-1, 4)
        obs = torch.cat([obs, sub_arena], dim=-1)
        return obs.view(B, M, M * self.get_humanoid_obs_size())

    # ------------------------------------------------------------------ stepping

    def pre_physics_step(self, actions):
        N, M, nd = self.num_envs, self.num_agents, self.num_dof
        self.actions = actions.to(self.device).clone().view(N, M, nd)

        if (self._pd_control):
            pd_tar = self._action_to_pd_targets(self.actions)

            if self.cfg["env"]["enableTrackInitState"]:
                pd_tar = self._every_env_init_dof_pos.clone()

            pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar.contiguous().view(N, M * nd))
            self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)
        else:
            forces = self.actions * self.motor_efforts.unsqueeze(0) * self.power_scale
            force_tensor = gymtorch.unwrap_tensor(forces.contiguous().view(N, M * nd))
            self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

        return

    def _compute_reset(self):
        N, M, nb = self.num_envs, self.num_agents, self.num_bodies

        reset_slot, terminate_slot = compute_humanoid_reset(
            self._terminate_buf,
            self.progress_buf.repeat_interleave(M),
            self._contact_forces.reshape(N * M, nb, 3),
            self._contact_body_ids,
            self._rigid_body_pos.reshape(N * M, nb, 3),
            self.max_episode_length,
            self._enable_early_termination,
            self._termination_heights)

        # an agent is "terminated" only if it fell itself; the episode however is shared,
        # so a reset triggered by anyone resets the whole env (the others are truncated,
        # which the value bootstrap handles through the per-agent terminate flag).
        self._terminate_buf[:] = terminate_slot
        self.reset_buf[:] = reset_slot.view(N, M).max(dim=1)[0]
        return

    # ------------------------------------------------------------------ resets

    def _reset_actors(self, env_ids):
        slot_env, slot_agent = self._expand_slots(env_ids)
        self._reset_default(slot_env, slot_agent)
        return

    def _reset_default(self, slot_env, slot_agent):
        self._humanoid_root_states[slot_env, slot_agent] = \
            self._initial_humanoid_root_states[slot_env, slot_agent]
        self._dof_pos[slot_env, slot_agent] = self._initial_dof_pos[slot_env, slot_agent]
        self._dof_vel[slot_env, slot_agent] = self._initial_dof_vel[slot_env, slot_agent]
        return

    def _reset_env_tensors(self, env_ids):
        actor_ids = self._humanoid_actor_ids[env_ids].contiguous().view(-1)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(actor_ids), len(actor_ids))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(actor_ids), len(actor_ids))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[self._flat_slot_ids(env_ids)] = 0
        return

    def set_char_color(self, col, env_ids):
        for env_id in env_ids:
            env_ptr = self.envs[env_id]
            for handle in self.humanoid_handles_all[env_id]:
                for j in range(self.num_bodies):
                    self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL,
                                                  gymapi.Vec3(col[0], col[1], col[2]))
        return

    def _update_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self._humanoid_root_states[0, 0, 0:3].cpu().numpy()

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

    def _init_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._cam_prev_char_pos = self._humanoid_root_states[0, 0, 0:3].cpu().numpy()

        cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0], self._cam_prev_char_pos[1] - 3.0, 1.0)
        cam_target = gymapi.Vec3(self._cam_prev_char_pos[0], self._cam_prev_char_pos[1], 1.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        return


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_humanoid_observations_rel_max(body_pos, body_rot, body_vel, body_ang_vel,
                                          obs_root_pos, obs_heading_rot,
                                          local_root_obs, root_height_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool) -> Tensor
    """Body-level observation of one humanoid expressed in an arbitrary observer frame.

    Identical to compute_humanoid_observations_max when the observer is the subject itself,
    except that the subject's root position is kept (it is exactly zero in that case) so
    that ego and non-ego humanoid tokens share one layout and one tokenizer.
    """
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 2:3]
    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h

    num_bodies = body_pos.shape[1]

    heading_rot_expand = obs_heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, num_bodies, 1))
    flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
                                                  heading_rot_expand.shape[2])

    obs_root_pos_expand = obs_root_pos.unsqueeze(-2)
    local_body_pos = body_pos - obs_root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1],
                                                 local_body_pos.shape[2])
    flat_local_body_pos = quat_rotate(flat_heading_rot, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0],
                                                 local_body_pos.shape[1] * local_body_pos.shape[2])

    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])
    flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot)
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0],
                                                         body_rot.shape[1] * flat_local_body_rot_obs.shape[1])

    if (not local_root_obs):
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot)
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = quat_rotate(flat_heading_rot, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])

    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = quat_rotate(flat_heading_rot, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0],
                                                         body_ang_vel.shape[1] * body_ang_vel.shape[2])

    obs = torch.cat((root_h_obs, local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel), dim=-1)
    return obs


@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_heights):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        fall_contact = torch.any(torch.abs(masked_contact_buf).sum(dim=-1) > 0.1, dim=-1)

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
