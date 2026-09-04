"""No-training two-agent box-stacking probe for the ms18 policy.

The actor network and observation layout are exactly HumanoidMASteerCarry's.
A centralized task controller slows agent 1 until agent 0 has released a stable
base box, then restores agent 1's nominal speed and retargets its existing carry
goal and steering path to the top of the measured base box.  No teammate token
is required by the policy.

This is deliberately an inference probe, not a new learned stacking reward.
"""

import math
import os

import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_rotate

from env.tasks.adapt_interaction_skills.humanoid_ma_steer_carry import (
    HumanoidMASteerCarry,
)


def _env_float(name, default):
    return float(os.environ.get(name, default))


class HumanoidMAStackCarry(HumanoidMASteerCarry):
    """Centralized sequential stacking controller around a frozen ms18 actor."""

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        if int(cfg["env"].get("numAgents", 1)) != 2:
            raise ValueError("HumanoidMAStackCarry requires env.numAgents=2")

        self._stack_goal_x = _env_float("STACK_GOAL_X", 4.0)
        self._stack_goal_y = _env_float("STACK_GOAL_Y", 0.0)
        self._stack_mode = os.environ.get("STACK_MODE", "stack")
        if self._stack_mode not in ("stack", "retreat", "static"):
            raise ValueError(f"STACK_MODE must be stack, retreat, or static, got {self._stack_mode}")
        self._stack_stage_y = _env_float("STACK_STAGE_Y", 3.0)
        self._stack_clear_dist = _env_float("STACK_CLEAR_DIST", 1.5)
        self._stack_support_size = _env_float("STACK_SUPPORT_SIZE", 0.4)
        self._stack_top_size = _env_float("STACK_TOP_SIZE", 0.22)
        if self._stack_top_size <= 0.0:
            raise ValueError(f"STACK_TOP_SIZE must be positive, got {self._stack_top_size}")
        self._stack_xy_tol = _env_float("STACK_XY_TOL", 0.10)
        self._stack_z_tol = _env_float("STACK_Z_TOL", 0.06)
        self._stack_lin_tol = _env_float("STACK_LIN_TOL", 0.10)
        self._stack_ang_tol = _env_float("STACK_ANG_TOL", 0.30)
        self._stack_tilt_deg = _env_float("STACK_TILT_DEG", 10.0)
        self._stack_hand_clear = _env_float("STACK_HAND_CLEAR", 0.25)
        self._stack_stable_steps = int(_env_float("STACK_STABLE_STEPS", 20))
        self._stack_top_stable_steps = int(_env_float("STACK_TOP_STABLE_STEPS", 20))
        self._stack_follow_alpha = _env_float("STACK_FOLLOW_ALPHA", 0.10)
        self._stack_base_scale = _env_float("STACK_BASE_SCALE", 1.0)
        self._stack_wait_scale = _env_float("STACK_WAIT_SCALE", 0.25)
        self._stack_go_scale = _env_float("STACK_GO_SCALE", 1.0)
        self._stack_require_release = bool(int(os.environ.get("STACK_REQUIRE_RELEASE", "1")))
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)

    def _load_box_asset(self, box_sizes):
        """Force only agent 1's carried box to a deterministic small cube.

        BoxLib normally samples each agent independently from the nine eval
        sizes, so a one-environment viewer can randomly give agent 1 a 0.57 m
        cube.  Update both the physical asset size and BoxLib's BPS observation
        before actors are created.  Agent 0's unused box remains unchanged.
        """
        if self._stack_mode == "static":
            box_sizes[1::2, :] = self._stack_top_size
            self._box_lib._build_box_bps()
        return super()._load_box_asset(box_sizes)

    def _load_platform_asset(self):
        """Use a fixed box-shaped support instead of the original 2 cm shelf."""
        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 100.0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        self._platform_height = self._stack_support_size
        self._platform_asset = self.gym.create_box(
            self.sim,
            self._stack_support_size,
            self._stack_support_size,
            self._stack_support_size,
            asset_options,
        )

    def _create_envs(self, num_envs, spacing, num_per_row):
        super()._create_envs(num_envs, spacing, num_per_row)
        self._stack_base_goal = torch.zeros((num_envs, 3), device=self.device)
        self._stack_base_ready = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._stack_base_count = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._stack_top_count = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._stack_success = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._stack_latched_base = torch.zeros((num_envs, 3), device=self.device)
        self._stack_clear_goal = torch.zeros((num_envs, 2), device=self.device)
        self._stack_clear_done = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        return

    def _sync_stack_actors(self, env_ids):
        """Push boxes and the hidden helper platforms after probe-specific reset."""
        rows = self.agent_rows(env_ids)
        actor_ids = [self._box_actor_ids[rows]]
        if self._carry_reset_random_height:
            actor_ids.extend([
                self._platform_actor_ids[rows],
                self._tar_platform_actor_ids[rows],
            ])
        actor_ids = torch.cat(actor_ids)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(actor_ids),
            len(actor_ids),
        )

    def _post_object_reset(self, env_ids):
        # Let the parent initialize all policy-side buffers, then replace only the
        # physical layout and goals needed by this inference probe.
        super()._post_object_reset(env_ids)
        if len(env_ids) == 0:
            return

        A = self.num_agents
        boxes = self.agent_axis(self._box_states)
        sizes = self._box_lib._box_size.view(self.num_envs, A, 3)

        # The pretrained random-height task creates thin helper platforms.  They
        # must not support the top box here: the agent-0 box is the real support.
        if self._carry_reset_random_height:
            for states in (self._platform_states, self._tar_platform_states):
                shaped = self.agent_axis(states)
                shaped[env_ids, :, 0:2] = 0.0
                shaped[env_ids, :, 2] = -10.0
                shaped[env_ids, :, 7:13] = 0.0

        # Start both interaction boxes on the floor.  view_stack_local.sh forces
        # loco initialization, so moving the boxes does not break a held pose.
        boxes[env_ids, :, 2] = sizes[env_ids, :, 2] * 0.5
        boxes[env_ids, :, 7:13] = 0.0

        # A fixed team goal makes the first no-training probe reproducible.  The
        # top goal is replaced with the measured base pose once it is stable.
        initial_h = self.agent_axis(self._initial_humanoid_root_states)
        origin = initial_h[env_ids, :, 0:2].mean(dim=1)
        goal_xy = origin + torch.tensor(
            [self._stack_goal_x, self._stack_goal_y], device=self.device
        )
        self._stack_base_goal[env_ids, 0:2] = goal_xy
        self._stack_base_goal[env_ids, 2] = sizes[env_ids, 0, 2] * 0.5

        r0 = env_ids * A
        r1 = r0 + 1
        self._box_tar_pos[r0] = self._stack_base_goal[env_ids]
        if self._stack_mode == "static":
            # A fixed, box-shaped tar_platform is the support.  Agent 0 and its
            # own dynamic box are sent to the opposite side so only agent 1's
            # high-placement capability is measured.
            self._box_tar_pos[r0, 0:2] = origin + torch.tensor(
                [-self._stack_goal_x, -self._stack_stage_y], device=self.device
            )
            self._box_tar_pos[r0, 2] = sizes[env_ids, 0, 2] * 0.5
            self._box_tar_pos[r1, 0:2] = goal_xy
            self._box_tar_pos[r1, 2] = self._stack_support_size + sizes[env_ids, 1, 2] * 0.5
            support = self.agent_axis(self._tar_platform_states)
            support[env_ids, 1, 0:2] = goal_xy
            support[env_ids, 1, 2] = self._stack_support_size * 0.5
            support[env_ids, 1, 3:7] = 0.0
            support[env_ids, 1, 6] = 1.0
            support[env_ids, 1, 7:13] = 0.0
        elif self._stack_mode == "retreat":
            # Keep agent 1 moving, but give it an independent ground placement
            # goal outside agent 0's workspace.  This isolates whether agent 0
            # can obey a post-place body-only steering command.
            self._box_tar_pos[r1, 0:2] = goal_xy + torch.tensor(
                [0.0, self._stack_stage_y], device=self.device
            )
            self._box_tar_pos[r1, 2] = sizes[env_ids, 1, 2] * 0.5
        else:
            self._box_tar_pos[r1, 0:2] = goal_xy
            self._box_tar_pos[r1, 2] = sizes[env_ids, 0, 2] + sizes[env_ids, 1, 2] * 0.5

        self._stack_base_ready[env_ids] = False
        self._stack_base_count[env_ids] = 0
        self._stack_top_count[env_ids] = 0
        self._stack_success[env_ids] = False
        self._stack_latched_base[env_ids] = 0.0
        self._stack_clear_goal[env_ids] = 0.0
        self._stack_clear_done[env_ids] = False

        self._sync_stack_actors(env_ids)
        self._reset_steer(self.agent_rows(env_ids))
        # Both agents move from the first step.  Agent 0 receives nominal speed,
        # while agent 1 sees the shortest steering window used by ms18 training
        # (0.25 * 1.6 m -> about 0.375 m/s) until the support box is ready.
        if self._stack_mode == "static":
            self._mscale[r0] = self._stack_wait_scale
            self._mscale[r1] = self._stack_go_scale
            self._stack_base_ready[env_ids] = True
        else:
            self._mscale[r0] = self._stack_base_scale
            self._mscale[r1] = self._stack_wait_scale
        self._reset_progress_ref(env_ids)
        if self._stack_mode == "static":
            msg = (
                f"[stack] reset mode=static envs={len(env_ids)}; fixed "
                f"{self._stack_support_size:.2f}m box support, agent1 carries "
                f"{self._stack_top_size:.2f}m cube at speed scale={self._stack_go_scale:.2f}"
            )
        else:
            msg = (
                f"[stack] reset mode={self._stack_mode} envs={len(env_ids)}; both agents active, "
                f"agent1 speed scale={self._stack_wait_scale:.2f} until base is stable for "
                f"{self._stack_stable_steps} steps"
            )
        print(msg, flush=True)
        return

    def _hands_clear(self, agent):
        hands = self._rigid_body_pos[:, agent, self._key_body_ids[[0, 1]], :]
        box = self.agent_axis(self._box_states)[:, agent, 0:3]
        return torch.norm(hands - box[:, None, :], dim=-1).amin(dim=-1) > self._stack_hand_clear

    def _upright(self, quat):
        up = torch.zeros((len(quat), 3), device=self.device)
        up[:, 2] = 1.0
        world_up = quat_rotate(quat, up)
        return world_up[:, 2] >= math.cos(math.radians(self._stack_tilt_deg))

    def _update_stack_controller(self):
        boxes = self.agent_axis(self._box_states)
        sizes = self._box_lib._box_size.view(self.num_envs, self.num_agents, 3)
        b0, b1 = boxes[:, 0], boxes[:, 1]

        if self._stack_mode == "static":
            r1 = torch.arange(self.num_envs, device=self.device) * self.num_agents + 1
            target = self._box_tar_pos[r1]
            top_cond = (
                (torch.norm(b1[:, 0:2] - target[:, 0:2], dim=-1) <= self._stack_xy_tol)
                & ((b1[:, 2] - target[:, 2]).abs() <= self._stack_z_tol)
                & (torch.norm(b1[:, 7:10], dim=-1) <= self._stack_lin_tol)
                & (torch.norm(b1[:, 10:13], dim=-1) <= self._stack_ang_tol)
                & self._upright(b1[:, 3:7])
                & self._hands_clear(1)
            )
            self._stack_top_count = torch.where(
                top_cond,
                self._stack_top_count + 1,
                torch.zeros_like(self._stack_top_count),
            )
            new_success = (~self._stack_success) & (
                self._stack_top_count >= self._stack_top_stable_steps
            )
            if bool(new_success.any()):
                env_ids = torch.nonzero(new_success, as_tuple=False).squeeze(-1)
                self._stack_success[env_ids] = True
                print(
                    f"[stack-static] SUCCESS envs={env_ids.tolist()}; "
                    "agent1 placed on fixed box support",
                    flush=True,
                )
            return

        base_cond = (
            (torch.norm(b0[:, 0:2] - self._stack_base_goal[:, 0:2], dim=-1) <= self._stack_xy_tol)
            & ((b0[:, 2] - self._stack_base_goal[:, 2]).abs() <= self._stack_z_tol)
            & (torch.norm(b0[:, 7:10], dim=-1) <= self._stack_lin_tol)
            & (torch.norm(b0[:, 10:13], dim=-1) <= self._stack_ang_tol)
            & self._upright(b0[:, 3:7])
        )
        if self._stack_require_release:
            base_cond &= self._hands_clear(0)

        pending = ~self._stack_base_ready
        self._stack_base_count = torch.where(
            pending & base_cond,
            self._stack_base_count + 1,
            torch.where(pending, torch.zeros_like(self._stack_base_count), self._stack_base_count),
        )
        just_ready = pending & (self._stack_base_count >= self._stack_stable_steps)

        if bool(just_ready.any()):
            env_ids = torch.nonzero(just_ready, as_tuple=False).squeeze(-1)
            self._stack_base_ready[env_ids] = True
            self._stack_latched_base[env_ids] = b0[env_ids, 0:3]
            if self._stack_mode == "retreat":
                r0 = env_ids * self.num_agents
                saved_box_target = self._box_tar_pos[r0].clone()
                clear_xy = b0[env_ids, 0:2] + torch.tensor(
                    [self._stack_clear_dist, 0.0], device=self.device
                )
                self._stack_clear_goal[env_ids] = clear_xy
                # _reset_steer always draws root -> own box -> final target.  Use
                # the clear point only while drawing the path, then restore the
                # box's placement goal.  The resulting command asks the body to
                # continue forward while the carry target says leave the box put.
                self._box_tar_pos[r0, 0:2] = clear_xy
                self._reset_steer(r0)
                self._mscale[r0] = self._stack_go_scale
                self._box_tar_pos[r0] = saved_box_target
                print(
                    f"[stack-retreat] base stable envs={env_ids.tolist()}; "
                    f"agent0 commanded {self._stack_clear_dist:.2f}m clear",
                    flush=True,
                )
                self._compute_observations(env_ids)
            else:
                r1 = env_ids * self.num_agents + 1
                self._box_tar_pos[r1, 0:2] = b0[env_ids, 0:2]
                self._box_tar_pos[r1, 2] = (
                    b0[env_ids, 2]
                    + 0.5 * (sizes[env_ids, 0, 2] + sizes[env_ids, 1, 2])
                )
                self._reset_steer(r1)
                self._mscale[r1] = self._stack_go_scale
                print(
                    f"[stack] base ready envs={env_ids.tolist()}; agent1 speed "
                    f"scale={self._stack_go_scale:.2f} and retargeted",
                    flush=True,
                )

        if self._stack_mode == "retreat":
            active = self._stack_base_ready & ~self._stack_clear_done
            if bool(active.any()):
                root0 = self.agent_axis(self._humanoid_root_states)[:, 0, 0:3]
                body_clear = torch.norm(
                    root0[:, 0:2] - self._stack_latched_base[:, 0:2], dim=-1
                ) >= 1.0
                base_shift = torch.norm(
                    b0[:, 0:3] - self._stack_latched_base, dim=-1
                )
                passed = active & body_clear & (base_shift <= self._stack_xy_tol) & self._hands_clear(0)
                if bool(passed.any()):
                    env_ids = torch.nonzero(passed, as_tuple=False).squeeze(-1)
                    self._stack_clear_done[env_ids] = True
                    print(
                        f"[stack-retreat] PASS envs={env_ids.tolist()}; "
                        f"agent0 cleared while base stayed put",
                        flush=True,
                    )
            self.extras["stack_clear_done"] = self._stack_clear_done.float()
            return

        # Once agent 1 is close, small support-box motion is reflected in the
        # ordinary carry target.  The path is deliberately not reset every step.
        active = self._stack_base_ready & ~self._stack_success
        if bool(active.any()):
            env_ids = torch.nonzero(active, as_tuple=False).squeeze(-1)
            r1 = env_ids * self.num_agents + 1
            desired = b0[env_ids, 0:3].clone()
            desired[:, 2] += 0.5 * (
                sizes[env_ids, 0, 2] + sizes[env_ids, 1, 2]
            )
            self._box_tar_pos[r1] = torch.lerp(
                self._box_tar_pos[r1], desired, self._stack_follow_alpha
            )

            target = self._box_tar_pos[r1]
            top_cond = (
                (torch.norm(b1[env_ids, 0:2] - target[:, 0:2], dim=-1) <= self._stack_xy_tol)
                & ((b1[env_ids, 2] - target[:, 2]).abs() <= self._stack_z_tol)
                & (torch.norm(b1[env_ids, 7:10], dim=-1) <= self._stack_lin_tol)
                & (torch.norm(b1[env_ids, 10:13], dim=-1) <= self._stack_ang_tol)
                & self._upright(b1[env_ids, 3:7])
                & self._hands_clear(1)[env_ids]
            )
            self._stack_top_count[env_ids] = torch.where(
                top_cond,
                self._stack_top_count[env_ids] + 1,
                torch.zeros_like(self._stack_top_count[env_ids]),
            )
            success_ids = env_ids[self._stack_top_count[env_ids] >= self._stack_top_stable_steps]
            new_success = success_ids[~self._stack_success[success_ids]]
            if len(new_success) > 0:
                self._stack_success[new_success] = True
                print(f"[stack] SUCCESS envs={new_success.tolist()}", flush=True)

            # Observations were computed earlier in the parent post step.  Refresh
            # active envs so the next policy action receives the new target now.
            self._compute_observations(env_ids)

    def post_physics_step(self):
        super().post_physics_step()
        self._update_stack_controller()
        self.extras["stack_base_ready"] = self._stack_base_ready.float()
        self.extras["stack_success"] = self._stack_success.float()
        return
