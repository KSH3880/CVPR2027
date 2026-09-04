"""Two-agent carry extension for learning release and workspace clearance.

Agent 0 remains an exact legacy-carry rehearsal row.  Agent 1 learns release
and clearance, while ``REL_SUPPORT_MODE`` selects either a thin floating shelf
or a solid box below its target surface.  Across the two independent runs all
other geometry and random seeds are paired, leaving the support's below-surface
collision geometry as the controlled difference.

Observation dimensions are unchanged.  Before support contact the exact
HumanoidMASteerCarry reward is used.  Once a physically viable support contact
is latched, the objective becomes hand withdrawal, support retention and then
body clearance.  The phase is monotonic; settling motion never switches the
policy back to carry.
"""

import math
import os

import numpy as np
import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_conjugate, quat_rotate

from env.tasks.adapt_interaction_skills.humanoid_ma_steer_carry import (
    HumanoidMASteerCarry,
)
from tokenhsi.utils import steer_path as sp


def _f(name, default):
    return float(os.environ.get(name, default))


def _i(name, default):
    return int(os.environ.get(name, default))


class HumanoidMADualSupportRelease(HumanoidMASteerCarry):
    """Shared-policy shelf/solid-support release-and-clear curriculum."""

    LEGACY = -1
    CARRY = 0
    RELEASE = 1
    CLEAR = 2
    SUCCESS = 3

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        if int(cfg["env"].get("numAgents", 1)) != 2:
            raise ValueError("HumanoidMADualSupportRelease requires env.numAgents=2")

        # Geometry is sampled once per parallel env because Isaac Gym asset
        # dimensions cannot be changed at reset.  Poses are sampled every reset.
        self._rel_seed = _i("REL_SEED", int(cfg.get("seed", 0)) + 19019)
        self._rel_support_mode = os.environ.get("REL_SUPPORT_MODE", "shelf")
        if self._rel_support_mode not in ("shelf", "box"):
            raise ValueError(
                "REL_SUPPORT_MODE must be shelf or box; got "
                f"{self._rel_support_mode}"
            )
        self._rel_release_scale = _f("REL_RELEASE_SCALE", 0.25)
        self._rel_top_lo = _f("REL_TOP_LO", 0.25)
        self._rel_top_hi = _f("REL_TOP_HI", 0.90)
        self._rel_support_xy_lo = _f("REL_SUPPORT_XY_LO", 0.35)
        self._rel_support_xy_hi = _f("REL_SUPPORT_XY_HI", 0.65)
        self._rel_support_margin = _f("REL_SUPPORT_MARGIN", 0.08)
        self._rel_target_dist_lo = _f("REL_TARGET_DIST_LO", 2.0)
        self._rel_target_dist_hi = _f("REL_TARGET_DIST_HI", 5.0)
        self._rel_target_jitter = math.radians(_f("REL_TARGET_JITTER_DEG", 50.0))
        self._rel_clear_lo = _f("REL_CLEAR_LO", 0.8)
        self._rel_clear_hi = _f("REL_CLEAR_HI", 1.2)

        self._rel_contact_tol = _f("REL_CONTACT_TOL", 0.035)
        self._rel_com_margin = _f("REL_COM_MARGIN", 0.02)
        self._rel_entry_lin = _f("REL_ENTRY_LIN", 0.25)
        self._rel_entry_ang = _f("REL_ENTRY_ANG", 1.0)
        self._rel_entry_steps = _i("REL_ENTRY_STEPS", 5)
        self._rel_hand_clear = _f("REL_HAND_CLEAR", 0.15)
        self._rel_hand_steps = _i("REL_HAND_STEPS", 5)
        self._rel_stable_lin = _f("REL_STABLE_LIN", 0.08)
        self._rel_stable_ang = _f("REL_STABLE_ANG", 0.20)
        self._rel_stable_steps = _i("REL_STABLE_STEPS", 20)
        self._rel_drop_z = _f("REL_DROP_Z", 0.08)

        if not (0 < self._rel_top_lo <= self._rel_top_hi):
            raise ValueError("REL_TOP_LO/HI must satisfy 0 < LO <= HI")
        if not (0 < self._rel_target_dist_lo <= self._rel_target_dist_hi):
            raise ValueError("REL_TARGET_DIST_LO/HI must satisfy 0 < LO <= HI")
        if not (0 < self._rel_clear_lo <= self._rel_clear_hi):
            raise ValueError("REL_CLEAR_LO/HI must satisfy 0 < LO <= HI")

        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)

    # ------------------------------------------------------------------ assets

    def _load_box_asset(self, box_sizes):
        """Pair carried-box sizes within each env; retain diversity across envs."""
        paired = box_sizes.view(-1, 2, 3)
        paired[:, 1, :] = paired[:, 0, :]
        self._box_lib._build_box_bps()
        return super()._load_box_asset(box_sizes)

    @staticmethod
    def _fixed_asset_options():
        opts = gymapi.AssetOptions()
        opts.angular_damping = 0.01
        opts.linear_damping = 0.01
        opts.max_angular_velocity = 100.0
        opts.density = 100.0
        opts.fix_base_link = True
        opts.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        return opts

    def _load_platform_asset(self):
        """Build a paired thin-shelf/solid-box target asset for every env."""
        opts = self._fixed_asset_options()
        self._platform_height = 0.02
        # Source platforms remain the native thin shelf for both rows.
        self._platform_asset = self.gym.create_box(self.sim, 0.4, 0.4, 0.02, opts)

        carry = self._box_lib._box_size.detach().cpu().view(self.num_envs, 2, 3)[:, 0]
        rng = np.random.default_rng(self._rel_seed)
        raw_top = rng.uniform(self._rel_top_lo, self._rel_top_hi, self.num_envs)
        # Match the native carry cap: carried-box top surface cannot exceed 1.2 m.
        top = np.minimum(raw_top, np.maximum(self._rel_top_lo, 1.2 - carry[:, 2].numpy()))

        lo_x = np.maximum(self._rel_support_xy_lo,
                          carry[:, 0].numpy() + self._rel_support_margin)
        lo_y = np.maximum(self._rel_support_xy_lo,
                          carry[:, 1].numpy() + self._rel_support_margin)
        hi_x = np.maximum(lo_x, self._rel_support_xy_hi)
        hi_y = np.maximum(lo_y, self._rel_support_xy_hi)
        sx = rng.uniform(lo_x, hi_x)
        sy = rng.uniform(lo_y, hi_y)

        pair_size = np.stack([sx, sy, top], axis=-1)
        pair_size = np.repeat(pair_size[:, None, :], 2, axis=1).reshape(-1, 3)
        self._rel_support_size = torch.tensor(
            pair_size, dtype=torch.float32, device=self.device
        )
        self._rel_support_top = self._rel_support_size[:, 2].clone()

        self._rel_target_assets = []
        for row, size in enumerate(pair_size):
            if row % self.num_agents == 0:
                # Agent 0 is the exact legacy-carry rehearsal row, including
                # the parent's native 0.4 m thin target platform.
                self._rel_target_assets.append(self._platform_asset)
                continue
            height = 0.02 if self._rel_support_mode == "shelf" else float(size[2])
            self._rel_target_assets.append(self.gym.create_box(
                self.sim, float(size[0]), float(size[1]), height, opts
            ))
        return

    def _build_platforms(self, env_id, env_ptr):
        """Preserve parent actor ordering while selecting row-specific targets."""
        col_group = env_id
        col_filter = 0
        segmentation_id = 0
        plats, tar_plats = [], []

        for a in range(self.num_agents):
            pose = gymapi.Transform()
            pose.p.z = -10.0
            handle = self.gym.create_actor(
                env_ptr, self._platform_asset, pose, f"platform{a}",
                col_group, col_filter, segmentation_id,
            )
            self.gym.set_rigid_body_color(
                env_ptr, handle, 0, gymapi.MESH_VISUAL,
                gymapi.Vec3(0.5, 0.235, 0.6),
            )
            plats.append(handle)

        for a in range(self.num_agents):
            row = env_id * self.num_agents + a
            pose = gymapi.Transform()
            pose.p.z = -10.0
            handle = self.gym.create_actor(
                env_ptr, self._rel_target_assets[row], pose, f"tar_platform{a}",
                col_group, col_filter, segmentation_id,
            )
            color = gymapi.Vec3(0.0, 0.0, 0.8) if a == 0 else gymapi.Vec3(0.08, 0.18, 0.55)
            self.gym.set_rigid_body_color(
                env_ptr, handle, 0, gymapi.MESH_VISUAL, color
            )
            tar_plats.append(handle)

        self._platform_handles.append(plats)
        self._tar_platform_handles.append(tar_plats)
        return

    # --------------------------------------------------------------- lifecycle

    def _create_envs(self, num_envs, spacing, num_per_row):
        super()._create_envs(num_envs, spacing, num_per_row)
        rows = num_envs * self.num_agents
        self._rel_phase = torch.zeros(rows, dtype=torch.long, device=self.device)
        self._rel_entry_count = torch.zeros(rows, dtype=torch.long, device=self.device)
        self._rel_hand_count = torch.zeros(rows, dtype=torch.long, device=self.device)
        self._rel_stable_count = torch.zeros(rows, dtype=torch.long, device=self.device)
        self._rel_success = torch.zeros(rows, dtype=torch.bool, device=self.device)
        self._rel_latched_box = torch.zeros((rows, 13), device=self.device)
        self._rel_support_xy = torch.zeros((rows, 2), device=self.device)
        self._rel_clear_goal = torch.zeros((rows, 2), device=self.device)
        self._rel_clear_req = torch.zeros(rows, device=self.device)
        print(
            f"[release] support={self._rel_support_mode}; "
            "agent0=legacy carry, agent1=release/clear"
        )
        return

    def _post_object_reset(self, env_ids):
        super()._post_object_reset(env_ids)
        if len(env_ids) == 0:
            return

        all_rows = self.agent_rows(env_ids)
        rows = all_rows[(all_rows % self.num_agents) == 1]
        a = rows % self.num_agents
        boxes = self.humanoid_rows(self._box_states)
        roots = self.humanoid_rows(self._humanoid_root_states)

        # Random outward target from each row's current box.  MA_SEP separates
        # the workspaces; the jitter prevents a fixed global location/direction.
        root_xy = roots[rows, 0:2]
        root_pair = self.agent_axis(self._humanoid_root_states)[env_ids, :, 0:2]
        midpoint = root_pair.mean(dim=1)
        direction = root_xy - midpoint
        fallback = torch.stack([
            torch.zeros_like(a, dtype=torch.float),
            torch.where(a == 0, -torch.ones_like(a, dtype=torch.float),
                        torch.ones_like(a, dtype=torch.float)),
        ], dim=-1)
        norm = direction.norm(dim=-1, keepdim=True)
        direction = torch.where(norm > 1e-4, direction / norm.clamp(min=1e-4), fallback)
        jitter = (torch.rand(len(rows), device=self.device) * 2.0 - 1.0) * self._rel_target_jitter
        cs, sn = torch.cos(jitter), torch.sin(jitter)
        dx = direction[:, 0] * cs - direction[:, 1] * sn
        dy = direction[:, 0] * sn + direction[:, 1] * cs
        direction = torch.stack([dx, dy], dim=-1)
        distance = self._rel_target_dist_lo + torch.rand(len(rows), device=self.device) * (
            self._rel_target_dist_hi - self._rel_target_dist_lo
        )
        target_xy = boxes[rows, 0:2] + direction * distance[:, None]

        self._rel_support_xy[rows] = target_xy
        self._box_tar_pos[rows, 0:2] = target_xy
        self._box_tar_pos[rows, 2] = (
            self._rel_support_top[rows] + 0.5 * self._box_lib._box_size[rows, 2]
        )

        target_states = self.humanoid_rows(self._tar_platform_states)
        target_states[rows, 0:2] = target_xy
        if self._rel_support_mode == "shelf":
            target_states[rows, 2] = (
                self._rel_support_top[rows] - 0.5 * self._platform_height
            )
        else:
            target_states[rows, 2] = 0.5 * self._rel_support_top[rows]
        target_states[rows, 3:7] = 0.0
        target_states[rows, 6] = 1.0
        target_states[rows, 7:13] = 0.0

        # Agent 0 stays on the exact parent carry path.  Only agent 1 can enter
        # release/clear, so its legacy partner continuously rehearses the old
        # behavior in the same PPO batch.
        self._rel_phase[all_rows] = self.LEGACY
        self._rel_phase[rows] = self.CARRY
        self._rel_entry_count[all_rows] = 0
        self._rel_hand_count[all_rows] = 0
        self._rel_stable_count[all_rows] = 0
        self._rel_success[all_rows] = False
        self._rel_latched_box[all_rows] = 0.0
        self._rel_clear_goal[all_rows] = 0.0
        self._rel_clear_req[rows] = self._rel_clear_lo + torch.rand(
            len(rows), device=self.device
        ) * (self._rel_clear_hi - self._rel_clear_lo)

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(self._tar_platform_actor_ids[rows]),
            len(rows),
        )
        self._reset_steer(rows)
        self._reset_progress_ref(env_ids)
        return

    # ------------------------------------------------------------- measurements

    def _support_features(self):
        box = self.humanoid_rows(self._box_states)
        bps = self._box_lib._box_bps
        quat = box[:, None, 3:7].expand(-1, bps.shape[1], -1)
        corners = quat_rotate(quat.reshape(-1, 4), bps.reshape(-1, 3))
        corners = corners.view(box.shape[0], bps.shape[1], 3) + box[:, None, 0:3]
        min_z = corners[:, :, 2].amin(dim=1)

        half = 0.5 * self._rel_support_size[:, 0:2]
        edge_margin = (half - (box[:, 0:2] - self._rel_support_xy).abs()).amin(dim=-1)
        inside = edge_margin >= self._rel_com_margin
        height_err = (min_z - self._rel_support_top).abs()
        contact = height_err <= self._rel_contact_tol
        return box, min_z, edge_margin, inside, height_err, contact

    def _hand_surface_distance(self):
        box = self.humanoid_rows(self._box_states)
        rb = self.humanoid_rows(self._rigid_body_pos)
        hands = rb[:, self._key_body_ids[[0, 1]], :]
        rel = hands - box[:, None, 0:3]
        inv = quat_conjugate(box[:, 3:7])[:, None, :].expand(-1, 2, -1)
        local = quat_rotate(inv.reshape(-1, 4), rel.reshape(-1, 3)).view(-1, 2, 3)
        half = 0.5 * self._box_lib._box_size[:, None, :]
        outside = torch.clamp(local.abs() - half, min=0.0)
        return outside.norm(dim=-1).amin(dim=-1)

    def _set_clear_paths(self, rows):
        if len(rows) == 0:
            return
        roots = self.humanoid_rows(self._humanoid_root_states)[rows, 0:2]
        away = roots - self._rel_support_xy[rows]
        norm = away.norm(dim=-1, keepdim=True)
        a = rows % self.num_agents
        fallback = torch.stack([
            torch.zeros_like(a, dtype=torch.float),
            torch.where(a == 0, -torch.ones_like(a, dtype=torch.float),
                        torch.ones_like(a, dtype=torch.float)),
        ], dim=-1)
        away = torch.where(norm > 1e-4, away / norm.clamp(min=1e-4), fallback)
        clear = self._rel_support_xy[rows] + away * self._rel_clear_req[rows, None]
        waypoint = roots + away * 0.15
        self._rel_clear_goal[rows] = clear

        self._steer_tick += 1
        seed = self.steer_seed + self._steer_tick + int(rows[0])
        path, s_waypoint, s_end = sp.gen_full_v2(
            roots, waypoint, clear, seed, 0.0, 0.35, 120.0,
            p_two=0.0, skew=0.8, spread=(0.85, 1.8), lat_frac=0.25,
            with_end=True,
        )
        self._gt_path[rows] = path
        self._s_end[rows] = s_end
        self._arc_root[rows] = 0.0
        self._arc_box[rows] = s_waypoint
        self._mscale[rows] = 1.0
        self._prev_arc[rows] = 0.0
        return

    # ---------------------------------------------------------------- rewards

    def _compute_reward(self, actions):
        # Exact legacy reward until the support-contact latch fires.
        super()._compute_reward(actions)
        post = self._rel_phase >= self.RELEASE
        if not bool(post.any()):
            return

        box, min_z, edge_margin, inside, height_err, contact = self._support_features()
        hand_dist = self._hand_surface_distance()
        root = self.humanoid_rows(self._humanoid_root_states)[:, 0:3]

        support_xy_score = torch.sigmoid(20.0 * edge_margin)
        contact_score = torch.exp(-80.0 * height_err.square())
        support_score = support_xy_score * contact_score
        stable_score = torch.exp(
            -10.0 * box[:, 7:10].square().sum(dim=-1)
            -0.5 * box[:, 10:13].square().sum(dim=-1)
        )
        hand_score = torch.clamp(hand_dist / self._rel_hand_clear, 0.0, 1.0)
        root_dist = (root[:, 0:2] - self._rel_support_xy).norm(dim=-1)
        clear_score = torch.clamp(
            (root_dist - 0.40) / (self._rel_clear_lo - 0.40), 0.0, 1.0
        )

        release_reward = 0.45 * support_score + 0.20 * stable_score + 0.35 * hand_score
        clear_reward = (
            0.40 * support_score + 0.20 * stable_score
            + 0.15 * hand_score + 0.25 * clear_score
        )
        replacement = torch.where(
            self._rel_phase == self.RELEASE, release_reward, clear_reward
        )
        replacement = torch.where(
            self._rel_phase == self.SUCCESS, replacement + 0.5, replacement
        )

        dropped = post & (
            (min_z < self._rel_support_top - self._rel_drop_z)
            | (edge_margin < -0.15)
        )
        replacement = torch.where(dropped, -torch.ones_like(replacement), replacement)
        self.rew_buf[post] = replacement[post]
        return

    # --------------------------------------------------------------- controller

    def _update_release_controller(self):
        box, min_z, edge_margin, inside, height_err, contact = self._support_features()
        hand_dist = self._hand_surface_distance()

        carry = self._rel_phase == self.CARRY
        placeable = (
            inside & contact
            & (box[:, 7:10].norm(dim=-1) <= self._rel_entry_lin)
            & (box[:, 10:13].norm(dim=-1) <= self._rel_entry_ang)
        )
        self._rel_entry_count = torch.where(
            carry & placeable,
            self._rel_entry_count + 1,
            torch.where(carry, torch.zeros_like(self._rel_entry_count), self._rel_entry_count),
        )
        enter_release = carry & (self._rel_entry_count >= self._rel_entry_steps)
        if bool(enter_release.any()):
            rows = torch.nonzero(enter_release, as_tuple=False).squeeze(-1)
            self._rel_phase[rows] = self.RELEASE
            self._rel_latched_box[rows] = box[rows]
            # Reuse the existing variable-speed steer observation as the role/
            # phase cue, keeping the ms18 observation and model shapes intact.
            self._mscale[rows] = self._rel_release_scale
            env_ids = torch.unique(torch.div(rows, self.num_agents, rounding_mode="floor"))
            self._compute_observations(env_ids)

        release = self._rel_phase == self.RELEASE
        hands_clear = hand_dist >= self._rel_hand_clear
        self._rel_hand_count = torch.where(
            release & hands_clear,
            self._rel_hand_count + 1,
            torch.where(release, torch.zeros_like(self._rel_hand_count), self._rel_hand_count),
        )
        enter_clear = release & (self._rel_hand_count >= self._rel_hand_steps)
        if bool(enter_clear.any()):
            rows = torch.nonzero(enter_clear, as_tuple=False).squeeze(-1)
            self._rel_phase[rows] = self.CLEAR
            self._set_clear_paths(rows)
            env_ids = torch.unique(torch.div(rows, self.num_agents, rounding_mode="floor"))
            self._compute_observations(env_ids)

        clear = self._rel_phase == self.CLEAR
        root = self.humanoid_rows(self._humanoid_root_states)[:, 0:3]
        root_dist = (root[:, 0:2] - self._rel_support_xy).norm(dim=-1)
        final_ok = (
            clear & inside & contact & hands_clear
            & (root_dist >= self._rel_clear_lo)
            & (box[:, 7:10].norm(dim=-1) <= self._rel_stable_lin)
            & (box[:, 10:13].norm(dim=-1) <= self._rel_stable_ang)
        )
        self._rel_stable_count = torch.where(
            final_ok,
            self._rel_stable_count + 1,
            torch.where(clear, torch.zeros_like(self._rel_stable_count), self._rel_stable_count),
        )
        success = clear & (self._rel_stable_count >= self._rel_stable_steps)
        if bool(success.any()):
            rows = torch.nonzero(success, as_tuple=False).squeeze(-1)
            self._rel_phase[rows] = self.SUCCESS
            self._rel_success[rows] = True

        self.extras["release_phase"] = self._rel_phase.float()
        self.extras["release_success"] = self._rel_success.float()
        self.extras["release_hand_distance"] = hand_dist
        self.extras["release_support_margin"] = edge_margin
        return

    def post_physics_step(self):
        super().post_physics_step()
        self._update_release_controller()
        return
