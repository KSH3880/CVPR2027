"""Carry-only crossings with episode-fixed straight routes and stop commands."""

import os
import torch

from carry_planner.env_adapter import HumanoidMACarryPlannerTrain
from carry_speed.control import fixed_route, rate_limit_speed, scene_geometry


class HumanoidMACarrySpeedTrain(HumanoidMACarryPlannerTrain):
    def __init__(self, cfg, *args, **kwargs):
        self._speed_free_probability = float(os.environ.get("CARRY_SPEED_FREE_PROB", "0.2"))
        if not 0 <= self._speed_free_probability <= 1:
            raise ValueError("CARRY_SPEED_FREE_PROB must be in [0,1]")
        if os.environ.get("MS_SCEN", "cross") != "cross":
            raise ValueError("fixed-route speed experiments require MS_SCEN=cross")
        if float(os.environ.get("CARRY_PLANNER_CONVERGE_PROB", "0")) != 0:
            raise ValueError("speed crossings require separated goals (converge_prob=0)")
        super().__init__(cfg, *args, **kwargs)
        # The inherited reset function passes these buffers to TorchScript even
        # with IET disabled. Keep its interface valid without enabling IET.
        if not hasattr(self, "_max_IET_steps"):
            self._max_IET_steps = cfg["env"]["maxIETSteps"]
            self._IET_step_buf = torch.zeros(self._rows, device=self.device, dtype=torch.long)
            self._IET_triggered_buf = torch.zeros_like(self._IET_step_buf)

    def _ensure_speed_buffers(self):
        if not hasattr(self, "speed_priority"):
            self.speed_priority = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.speed_free = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.speed_center = torch.zeros(self.num_envs, 2, device=self.device)
            self.speed_paths = torch.zeros(self.num_envs, 2, 33, 2, device=self.device)
            self.speed_success_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.speed_scored = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.speed_completed = 0
            self.speed_successes = 0
            self.speed_success_total = torch.zeros(self.num_envs, device=self.device)

    def apply_layout(self, env_ids):
        super().apply_layout(env_ids)
        self._ensure_speed_buffers()
        if len(env_ids) == 0:
            return
        roots = self.agent_axis(self._humanoid_root_states)
        boxes = self.agent_axis(self._box_states)
        center = self.agent_axis(self._initial_humanoid_root_states)[env_ids, :, :2].mean(1)
        distance = 2.5 + 1.5 * torch.rand(len(env_ids), device=self.device)
        jitter = 0.2 * (torch.rand(len(env_ids), 2, device=self.device) - 0.5)
        free = torch.rand(len(env_ids), device=self.device) < self._speed_free_probability
        start, goal = scene_geometry(roots[env_ids, :, :2], center, distance, jitter, free)
        shift = start - roots[env_ids, :, :2]
        roots[env_ids, :, :2] = start
        boxes[env_ids, :, :2] += shift
        self._box_tar_pos[self.agent_rows(env_ids), :2] = goal.reshape(-1, 2)
        if self._carry_reset_random_height:
            self.agent_axis(self._platform_states)[env_ids, :, :2] += shift
            self.agent_axis(self._tar_platform_states)[env_ids, :, :2] = goal
        self.speed_priority[env_ids] = torch.randint(0, 2, (len(env_ids),), device=self.device)
        self.speed_free[env_ids] = free
        self.speed_center[env_ids] = center
        self.speed_success_count[env_ids] = 0
        self.speed_scored[env_ids] = False

    def _plan_envs(self, env_ids):
        """Install once per reset. Subsequent decisions modify only _mscale."""
        if len(env_ids) == 0:
            return
        self._ensure_speed_buffers()
        state = self._coord_state(env_ids)
        path = fixed_route(state.root_xy, state.goal_xy)
        self.speed_paths[env_ids] = path
        self._carry_reset_cursor_on_install = True
        super()._install_plan(env_ids, path, torch.full_like(path[..., 0], 1.5))
        self._coord_has_valid[env_ids] = True
        self._coord_last_replan[env_ids] = self.progress_buf[env_ids]
        self._coord_phase[env_ids] = state.phase

    def set_requested_speeds(self, speed):
        if speed.shape != (self.num_envs, 2) or not torch.isfinite(speed).all():
            raise ValueError("requested speeds must be finite [num_envs,2]")
        if bool(((speed < 0) | (speed > 1.5)).any()):
            raise ValueError("requested speeds must be in [0,1.5]")
        self._mscale[:] = speed.reshape(-1, 1) / 1.5

    def _set_speed_anchor(self, rows):
        root = self.humanoid_rows(self._humanoid_root_states)[rows]
        self._coord_cmd_speed[rows] = root[:, 7:9].norm(dim=-1).clamp(0, 1.5)
        self._coord_cmd_tick[rows] = -1

    def _update_speed_command(self, rows):
        if len(rows) == 0:
            return
        tick = self.progress_rows()[rows]
        update = tick != self._coord_cmd_tick[rows]
        chosen = rows[update]
        if len(chosen) == 0:
            return
        desired = 1.5 * self._mscale[chosen, 0]
        if self._coord_speed_limit:
            desired = rate_limit_speed(self._coord_cmd_speed[chosen], desired, self.dt,
                                       self._coord_accel_up, self._coord_accel_down)
        self._coord_cmd_speed[chosen] = desired.clamp(0, 1.5)
        self._coord_cmd_tick[chosen] = tick[update]

    def _compute_reset(self):
        super()._compute_reset()
        if not hasattr(self, "speed_scored"):
            return
        state = self._coord_state(torch.arange(self.num_envs, device=self.device))
        arrived = ((state.box_xyz[..., :2] - state.goal_xy).norm(dim=-1) < 0.3).all(-1)
        grounded = (state.held < 0.5).all(-1)
        still = (state.root_vel_xy.norm(dim=-1) < 0.3).all(-1)
        ready = arrived & grounded & still
        self.speed_success_count[:] = torch.where(ready, self.speed_success_count + 1, 0)
        success = self.speed_success_count >= 10
        self.reset_buf[:] |= success.long()
        completed = self.reset_buf.bool() & ~self.speed_scored
        self.speed_completed += int(completed.sum())
        self.speed_successes += int((completed & success).sum())
        self.speed_success_total[:] += (completed & success).float()
        self.speed_scored[:] |= completed
