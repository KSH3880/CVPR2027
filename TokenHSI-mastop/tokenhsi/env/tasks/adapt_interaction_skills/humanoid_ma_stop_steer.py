"""Humanoid-only stop/restart and two-agent oracle-yield training.

The task preserves the stage-1 354-D policy contract:
self 223 + task blocks [20, 38, 42, 27] + one-hot 4.
Only the trajectory block is populated; no chair, box, platform, or climb actor
is created. ``MSTOP_SCEN=mixed`` samples free, solo-stop, near-miss, and
conflict episodes inside one Stage-2 run. The oracle changes only the trajectory
command; the policy still has to physically stop, hold, and resume.
"""

import atexit
import os

import numpy as np
import torch
from isaacgym import gymapi, gymtorch

from env.tasks.humanoid import Humanoid
from env.tasks.basic_interaction_skills.humanoid_traj import (
    HumanoidTraj,
    build_amp_observations,
    compute_location_observations,
)


TASK_DIMS = [20, 38, 42, 27]
TASK_NAMES = ["traj", "sit", "carry", "climb"]
SCEN_FREE, SCEN_SOLO, SCEN_NEAR, SCEN_CONFLICT = range(4)
SCEN_NAMES = ["free", "solo", "near", "conflict"]


def _f(name, default):
    return float(os.environ.get(name, default))


def _mix_probs(value):
    """Parse free,solo,near,conflict probabilities and normalize them."""
    parts = [float(item.strip()) for item in value.split(",")]
    if len(parts) != len(SCEN_NAMES) or any(item < 0 for item in parts):
        raise ValueError(
            "MSTOP_MIX must be four non-negative values: "
            "free,solo,near,conflict")
    total = sum(parts)
    if total <= 0:
        raise ValueError("MSTOP_MIX must have a positive sum")
    return [item / total for item in parts]


class HumanoidMAStopSteer(HumanoidTraj):
    """Run solo M=0 and two-agent oracle stop tests without task objects."""

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.stop_scen = os.environ.get("MSTOP_SCEN", "solo")
        self._enable_task_mask_obs = True
        if self.stop_scen not in ("solo", "cross", "mixed"):
            raise ValueError("MSTOP_SCEN must be solo, cross, or mixed")
        self.stop_path_len = _f("MSTOP_L", 12.0)
        self.stop_nominal_speed = _f("MSTOP_SPEED", 1.5)
        self.stop_sample_dt = _f("MSTOP_SAMPLE_DT", 0.5)
        self.stop_at_s = _f("MSTOP_SOLO_S", 3.0)
        self.stop_hold_s = _f("MSTOP_HOLD", 1.5)
        self.stop_solo_random = bool(int(os.environ.get("MSTOP_SOLO_RANDOM", "0")))
        self.stop_s_min = _f("MSTOP_SOLO_S_MIN", 1.5)
        self.stop_s_max = _f("MSTOP_SOLO_S_MAX", 5.0)
        self.stop_hold_min = _f("MSTOP_HOLD_MIN", 0.5)
        self.stop_hold_max = _f("MSTOP_HOLD_MAX", 2.0)
        self.stop_oracle_horizon = _f("MSTOP_HORIZON", 2.5)
        self.stop_oracle_samples = int(_f("MSTOP_ORACLE_SAMPLES", 31))
        self.stop_safe_dist = _f("MSTOP_SAFE_D", 0.9)
        self.stop_clearance = _f("MSTOP_CLEAR", 1.0)
        self.stop_collision_dist = _f("MSTOP_COLLISION_D", 0.12)
        self.stop_mix_probs = _mix_probs(
            os.environ.get("MSTOP_MIX", "0.35,0.20,0.20,0.25"))
        self.stop_parallel_gap = _f("MSTOP_PARALLEL_GAP", 2.0)
        self.stop_near_dt_min = _f("MSTOP_NEAR_DT_MIN", 1.25)
        self.stop_near_dt_max = _f("MSTOP_NEAR_DT_MAX", 2.50)
        self.stop_conflict_dt_max = _f("MSTOP_CONFLICT_DT_MAX", 0.35)
        self.stop_angle_min = np.deg2rad(_f("MSTOP_ANGLE_MIN", 60.0))
        self.stop_angle_max = np.deg2rad(_f("MSTOP_ANGLE_MAX", 120.0))
        self.stop_speed_ema_alpha = _f("MSTOP_SPEED_EMA", 0.20)
        self.stop_resume_speed = _f("MSTOP_RESUME_SPEED", 0.80)
        self.stop_path_reward_w = _f("MSTOP_PATH_W", 0.35)
        self.stop_velocity_reward_w = _f("MSTOP_VEL_W", 0.65)
        self.stop_creep_penalty = _f("MSTOP_CREEP_C", 0.20)
        self.stop_creep_speed = _f("MSTOP_CREEP_V", 0.10)
        # DDP ranks use independent but reproducible episode streams.
        rank = int(os.environ.get("RANK", "0"))
        self.stop_seed = int(_f("MSTOP_SEED", 0)) + 1_000_003 * rank
        self._stop_reset_count = 0
        self.stop_metrics_path = os.environ.get("MSTOP_METRICS")
        if self.stop_metrics_path and int(os.environ.get("WORLD_SIZE", "1")) > 1:
            root, ext = os.path.splitext(self.stop_metrics_path)
            rank = int(os.environ.get("RANK", "0"))
            self.stop_metrics_path = f"{root}.rank{rank}{ext or '.npy'}"

        if not 0.0 <= self.stop_speed_ema_alpha <= 1.0:
            raise ValueError("MSTOP_SPEED_EMA must be in [0, 1]")
        if self.stop_near_dt_min > self.stop_near_dt_max:
            raise ValueError("MSTOP_NEAR_DT_MIN must be <= MSTOP_NEAR_DT_MAX")
        if self.stop_angle_min > self.stop_angle_max:
            raise ValueError("MSTOP_ANGLE_MIN must be <= MSTOP_ANGLE_MAX")

        super().__init__(
            cfg=cfg, sim_params=sim_params, physics_engine=physics_engine,
            device_type=device_type, device_id=device_id, headless=headless)

        expected = 1 if self.stop_scen == "solo" else 2
        if self.num_agents != expected:
            raise ValueError(
                f"MSTOP_SCEN={self.stop_scen} requires numAgents={expected}")
        if self._num_traj_samples != 10:
            raise ValueError("stage-1 checkpoint requires numTrajSamples=10")

        # Replace A=1-sized buffers allocated by HumanoidTraj.
        rows = self._rows
        self._amp_obs_buf = torch.zeros(
            (rows, self._num_amp_obs_steps, self._num_amp_obs_per_step),
            device=self.device)
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]
        self._amp_obs_demo_buf = None
        self._prev_root_pos = torch.zeros((rows, 3), device=self.device)
        self._every_env_init_dof_pos = torch.zeros(
            (rows, self.num_dof), device=self.device)
        self._kinematic_humanoid_rigid_body_states = torch.zeros(
            (rows, self.num_bodies, 13), device=self.device)
        if self._is_eval:
            self._success_buf = torch.zeros(
                rows, device=self.device, dtype=torch.long)
            self._precision_buf = torch.zeros(rows, device=self.device)

        envs = self.num_envs
        self._met_min_body = torch.full((envs,), 1e6, device=self.device)
        self._met_collision_steps = torch.zeros(
            envs, device=self.device, dtype=torch.long)
        self._met_stop_steps = torch.zeros(
            (envs, self.num_agents), device=self.device, dtype=torch.long)
        self._met_stop_speed_sum = torch.zeros((envs, self.num_agents), device=self.device)
        self._met_both_stop_steps = torch.zeros(
            envs, device=self.device, dtype=torch.long)
        self._met_resumed = torch.zeros(
            envs, device=self.device, dtype=torch.long)
        self._met_go_speed_sum = torch.zeros(
            (envs, self.num_agents), device=self.device)
        self._met_go_steps = torch.zeros(
            (envs, self.num_agents), device=self.device, dtype=torch.long)
        self._met_resume_latency = torch.full(
            (envs,), -1.0, device=self.device)
        self._resume_pending = torch.zeros(
            envs, device=self.device, dtype=torch.bool)
        self._resume_timer = torch.zeros(
            envs, device=self.device, dtype=torch.long)
        self._metric_rows = []
        atexit.register(self.report_metrics)
        print(
            f"[mastop] object-free scen={self.stop_scen} A={self.num_agents} "
            f"obs={self.get_obs_size()} speed={self.stop_nominal_speed:.2f} "
            f"mix={','.join(f'{p:.2f}' for p in self.stop_mix_probs)}",
            flush=True)

    def _setup_character_props(self, key_bodies):
        HumanoidTraj._setup_character_props(self, key_bodies)
        self._num_amp_obs_per_step += len(TASK_DIMS)

    def _init_camera(self):
        """Use a fixed overview camera for two-agent VNC visualization."""
        if os.environ.get("MS_CAM") != "top":
            return super()._init_camera()
        self.gym.refresh_actor_root_state_tensor(self.sim)
        origin = self.agent_axis(
            self._initial_humanoid_root_states)[0, :, 0:2].mean(dim=0).cpu().numpy()
        height = _f("MS_CAM_H", 17.0)
        back = _f("MS_CAM_B", 9.0)
        self._cam_prev_char_pos = np.array(
            [origin[0], origin[1], 1.0], dtype=np.float32)
        self.gym.viewer_camera_look_at(
            self.viewer, None,
            gymapi.Vec3(float(origin[0]), float(origin[1] - back), height),
            gymapi.Vec3(float(origin[0]), float(origin[1]), 0.0))

    def _update_camera(self):
        if os.environ.get("MS_CAM") == "top":
            return
        return super()._update_camera()

    def _with_traj_amp_label(self, amp_obs):
        label = torch.zeros((amp_obs.shape[0], len(TASK_DIMS)), device=amp_obs.device)
        label[:, 0] = 1.0
        return torch.cat([amp_obs, label], dim=-1)

    def build_amp_obs_demo(self, motion_ids, motion_times0):
        amp_obs = HumanoidTraj.build_amp_obs_demo(self, motion_ids, motion_times0)
        return self._with_traj_amp_label(amp_obs)

    # Exact stage-1 actor/network interface.
    def get_task_obs_size(self):
        return sum(TASK_DIMS) + len(TASK_DIMS)

    def get_multi_task_info(self):
        count = len(TASK_DIMS)
        mask = torch.zeros(
            count, sum(TASK_DIMS), dtype=torch.bool, device=self.device)
        index = torch.cumsum(torch.tensor([0] + TASK_DIMS), dim=0).to(self.device)
        for i in range(count):
            mask[i, index[i]:index[i + 1]] = True
        return {
            "onehot_size": count,
            "tota_subtask_obs_size": sum(TASK_DIMS),
            "each_subtask_obs_size": TASK_DIMS,
            "each_subtask_obs_mask": mask,
            "each_subtask_obs_indx": index,
            "enable_task_mask_obs": True,
            "each_subtask_name": TASK_NAMES,
        }

    # Markers are fixed visual-only actors, not task/collision objects.
    def _build_marker(self, env_id, env_ptr):
        colors = [gymapi.Vec3(0.2, 0.9, 0.2), gymapi.Vec3(0.2, 0.5, 1.0)]
        for agent in range(self.num_agents):
            for _ in range(self._num_traj_samples):
                handle = self.gym.create_actor(
                    env_ptr, self._marker_asset, gymapi.Transform(),
                    "traj_marker", self.num_envs + 10, 1, 0)
                self.gym.set_actor_scale(env_ptr, handle, 0.35)
                self.gym.set_rigid_body_color(
                    env_ptr, handle, 0, gymapi.MESH_VISUAL,
                    colors[agent % len(colors)])
                self._traj_marker_handles[env_id].append(handle)

    def _build_marker_state_tensors(self):
        actors = self.get_num_actors_per_env()
        handles = torch.tensor(
            self._traj_marker_handles, device=self.device, dtype=torch.long)
        roots = self._root_states.view(self.num_envs, actors, 13)
        env = torch.arange(self.num_envs, device=self.device)[:, None]
        self._traj_marker_states = roots[env, handles]
        self._traj_marker_pos = self._traj_marker_states[..., :3].view(
            self.num_envs, self.num_agents, self._num_traj_samples, 3)
        self._traj_marker_actor_ids = (
            torch.arange(
                self.num_envs, device=self.device, dtype=torch.int32)[:, None]
            * actors + handles.to(torch.int32)).flatten()

    # HumanoidTraj calls this hook during construction.
    def _build_traj_generator(self):
        rows = self._rows
        self._path_start = torch.zeros((rows, 2), device=self.device)
        self._path_dir = torch.zeros((rows, 2), device=self.device)
        self._path_dir[:, 0] = 1.0
        self._arc = torch.zeros(rows, device=self.device)
        self._stop_mask = torch.zeros(rows, device=self.device, dtype=torch.bool)
        self._solo_triggered = torch.zeros(
            rows, device=self.device, dtype=torch.bool)
        self._solo_stop_at = torch.full((rows,), self.stop_at_s, device=self.device)
        self._solo_hold_steps = torch.full((rows,), max(1, int(round(self.stop_hold_s / self.dt))), device=self.device, dtype=torch.long)
        self._solo_stop_until = torch.zeros(
            rows, device=self.device, dtype=torch.long)
        self._solo_active = torch.zeros(
            rows, device=self.device, dtype=torch.bool)
        self._scenario = torch.full(
            (self.num_envs,), SCEN_SOLO, device=self.device, dtype=torch.long)
        self._solo_agent = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self._yield_active = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool)
        self._yield_agent = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self._pred_min_dist = torch.full(
            (self.num_envs,), 1e6, device=self.device)
        self._pred_min_time = torch.zeros(self.num_envs, device=self.device)
        self._resume_event = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool)
        self._speed_ema = torch.full(
            (rows,), self.stop_nominal_speed, device=self.device)

    def progress_rows(self):
        if self.num_agents == 1:
            return self.progress_buf
        return self.progress_buf.repeat_interleave(self.num_agents)

    def _layout_paths(self, env_ids):
        rows = self.agent_rows(env_ids)
        env, agent = torch.div(
            rows, self.num_agents, rounding_mode="floor"), rows % self.num_agents
        half = self.stop_path_len / 2.0
        initial = self.agent_axis(self._initial_humanoid_root_states)
        origin = initial[:, :, 0:2].mean(dim=1)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.stop_seed + self._stop_reset_count)
        count = len(env_ids)
        sample = torch.rand((count, 6), generator=generator).to(self.device)
        coin = torch.randint(
            0, 2, (count, 2), generator=generator).to(self.device)

        if self.stop_scen == "solo":
            scenario = torch.full(
                (count,), SCEN_SOLO, device=self.device, dtype=torch.long)
        elif self.stop_scen == "cross":
            scenario = torch.full(
                (count,), SCEN_CONFLICT, device=self.device,
                dtype=torch.long)
        else:
            probs = torch.tensor(self.stop_mix_probs, dtype=torch.float32)
            scenario = torch.multinomial(
                probs, count, replacement=True,
                generator=generator).to(self.device)
        self._scenario[env_ids] = scenario

        base_yaw = (sample[:, 0] * 2.0 - 1.0) * np.pi
        angle = self.stop_angle_min + sample[:, 1] * (
            self.stop_angle_max - self.stop_angle_min)
        if self.stop_scen == "cross":
            base_yaw[:] = 0.0
            angle[:] = np.pi / 2.0
        elif self.stop_scen == "solo":
            base_yaw[:] = 0.0

        d0 = torch.stack(
            [torch.cos(base_yaw), torch.sin(base_yaw)], dim=-1)
        if self.num_agents == 1:
            directions = d0[:, None, :]
            starts = origin[env_ids, None, :] - directions * half
            initial_arc = torch.zeros((count, 1), device=self.device)
            self._solo_agent[env_ids] = 0
            self._yield_agent[env_ids] = 0
        else:
            d1 = torch.stack([
                torch.cos(base_yaw + angle),
                torch.sin(base_yaw + angle)], dim=-1)
            parallel = (
                (scenario == SCEN_FREE) | (scenario == SCEN_SOLO))
            d1[parallel] = d0[parallel]
            directions = torch.stack([d0, d1], dim=1)
            starts = origin[env_ids, None, :] - directions * half

            normal = torch.stack([-d0[:, 1], d0[:, 0]], dim=-1)
            starts[parallel, 0] -= (
                normal[parallel] * (self.stop_parallel_gap / 2.0))
            starts[parallel, 1] += (
                normal[parallel] * (self.stop_parallel_gap / 2.0))

            self._solo_agent[env_ids] = coin[:, 0]
            lead = coin[:, 1]
            initial_arc = torch.zeros((count, 2), device=self.device)
            near_dt = self.stop_near_dt_min + sample[:, 2] * (
                self.stop_near_dt_max - self.stop_near_dt_min)
            conflict_dt = sample[:, 3] * self.stop_conflict_dt_max
            timing_dt = torch.where(
                scenario == SCEN_NEAR, near_dt,
                torch.where(
                    scenario == SCEN_CONFLICT, conflict_dt,
                    torch.zeros_like(near_dt)))
            cross = (
                (scenario == SCEN_NEAR) |
                (scenario == SCEN_CONFLICT))
            local_env = torch.arange(count, device=self.device)[cross]
            initial_arc[local_env, lead[cross]] = (
                timing_dt[cross] * self.stop_nominal_speed)
            # The later arrival yields. For a synchronous arrival the random
            # lead coin is the per-episode tie break, avoiding both-stop.
            self._yield_agent[env_ids] = 1 - lead
            solo_envs = env_ids[scenario == SCEN_SOLO]
            self._yield_agent[solo_envs] = self._solo_agent[solo_envs]

        start = starts.reshape(-1, 2)
        direction = directions.reshape(-1, 2)
        row_arc = initial_arc.reshape(-1)
        yaw = torch.atan2(direction[:, 1], direction[:, 0])
        self._path_start[rows] = start
        self._path_dir[rows] = direction
        self._arc[rows] = row_arc

        humanoid = self.agent_axis(self._humanoid_root_states)
        humanoid[env, agent, 0:2] = start + direction * row_arc[:, None]
        humanoid[env, agent, 3:7] = 0.0
        humanoid[env, agent, 5] = torch.sin(yaw / 2.0)
        humanoid[env, agent, 6] = torch.cos(yaw / 2.0)
        speed = torch.linalg.vector_norm(
            humanoid[env, agent, 7:9], dim=-1).clamp(0.5, 1.5)
        humanoid[env, agent, 7:9] = direction * speed[:, None]
        humanoid[env, agent, 9:13] = 0.0

        self._stop_mask[rows] = False
        self._solo_triggered[rows] = False
        self._solo_stop_until[rows] = 0
        self._solo_active[rows] = False
        self._yield_active[env_ids] = False
        self._resume_event[env_ids] = False
        self._pred_min_dist[env_ids] = 1e6
        self._pred_min_time[env_ids] = 0.0
        self._speed_ema[rows] = self.stop_nominal_speed

        random_solo = self.stop_solo_random or self.stop_scen == "mixed"
        if random_solo:
            stop_sample = sample[:, 4].repeat_interleave(self.num_agents)
            hold_sample = sample[:, 5].repeat_interleave(self.num_agents)
            self._solo_stop_at[rows] = self.stop_s_min + stop_sample * (
                self.stop_s_max - self.stop_s_min)
            hold_s = self.stop_hold_min + hold_sample * (
                self.stop_hold_max - self.stop_hold_min)
            self._solo_hold_steps[rows] = torch.clamp(
                (hold_s / self.dt).round().long(), min=1)
        else:
            self._solo_stop_at[rows] = self.stop_at_s
            self._solo_hold_steps[rows] = max(
                1, int(round(self.stop_hold_s / self.dt)))
        self._stop_reset_count += 1

    def _project_arc(self):
        root = self.humanoid_rows(self._humanoid_root_states)[:, 0:2]
        projected = ((root - self._path_start) * self._path_dir).sum(dim=-1)
        projected = projected.clamp(0.0, self.stop_path_len)
        self._arc[:] = torch.maximum(self._arc - 0.10, projected)

    def _update_solo_stop(self):
        steps = self.progress_rows()
        row = torch.arange(self._rows, device=self.device)
        row_env = torch.div(
            row, self.num_agents, rounding_mode="floor")
        row_agent = row % self.num_agents
        eligible = (
            (self._scenario[row_env] == SCEN_SOLO) &
            (row_agent == self._solo_agent[row_env]))
        trigger = (
            eligible & (~self._solo_triggered) &
            (self._arc >= self._solo_stop_at))
        if trigger.any():
            self._solo_triggered[trigger] = True
            self._solo_stop_until[trigger] = (
                steps[trigger] + self._solo_hold_steps[trigger])
        active = (
            eligible & self._solo_triggered &
            (steps < self._solo_stop_until))
        released = self._solo_active & (~active)
        self._solo_active[:] = active
        self._stop_mask |= active
        if released.any():
            self._resume_event[torch.unique(row_env[released])] = True

    def _update_cross_stop(self):
        envs = self.num_envs
        arc = self._arc.view(envs, 2)
        start = self._path_start.view(envs, 2, 2)
        direction = self._path_dir.view(envs, 2, 2)
        candidate = (
            (self._scenario == SCEN_NEAR) |
            (self._scenario == SCEN_CONFLICT))
        times = torch.linspace(
            0.0, self.stop_oracle_horizon, self.stop_oracle_samples,
            device=self.device)
        root = self.humanoid_rows(self._humanoid_root_states)
        measured_speed = (root[:, 7:9] * self._path_dir).sum(dim=-1)
        measured_speed = measured_speed.clamp(
            0.0, 2.0 * self.stop_nominal_speed)
        self._speed_ema.lerp_(measured_speed, self.stop_speed_ema_alpha)
        forecast_speed = self._speed_ema.view(envs, 2)
        future_s = (
            arc[:, :, None] +
            forecast_speed[:, :, None] * times[None, None, :]
        ).clamp(max=self.stop_path_len)
        future = (
            start[:, :, None, :] +
            direction[:, :, None, :] * future_s[..., None])
        distances = torch.linalg.vector_norm(
            future[:, 0] - future[:, 1], dim=-1)
        pred_dist, index = distances.min(dim=-1)
        pred_time = times[index]
        self._pred_min_dist[:] = torch.where(
            candidate, pred_dist, torch.full_like(pred_dist, 1e6))
        self._pred_min_time[:] = torch.where(
            candidate, pred_time, torch.zeros_like(pred_time))

        center = self.stop_path_len / 2.0
        before_clear = (
            (arc[:, 0] < center + self.stop_clearance) &
            (arc[:, 1] < center + self.stop_clearance))
        risk = (
            candidate &
            (pred_dist < self.stop_safe_dist) &
            (pred_time <= self.stop_oracle_horizon) &
            before_clear)
        self._yield_active |= risk
        self._yield_active &= candidate

        env = torch.arange(envs, device=self.device)
        nonyield = 1 - self._yield_agent
        passed = arc[env, nonyield] > center + self.stop_clearance
        released = candidate & self._yield_active & passed
        self._yield_active[released] = False
        self._resume_event |= released

        active = env[self._yield_active]
        if len(active) > 0:
            self._stop_mask[active * 2 + self._yield_agent[active]] = True

    def _update_command_state(self):
        self._project_arc()
        self._resume_event[:] = False
        self._stop_mask[:] = False
        self._update_solo_stop()
        if self.num_agents == 2:
            self._update_cross_stop()

    def _fetch_traj_samples(self, env_ids=None):
        rows = self.all_rows() if env_ids is None else self.agent_rows(env_ids)
        offsets = (
            torch.arange(self._num_traj_samples, device=self.device).float() *
            self.stop_sample_dt * self.stop_nominal_speed)
        offsets = offsets[None].expand(len(rows), -1).clone()
        offsets[self._stop_mask[rows]] = 0.0
        distance = (self._arc[rows, None] + offsets).clamp(
            max=self.stop_path_len)
        xy = (
            self._path_start[rows, None, :] +
            self._path_dir[rows, None, :] * distance[..., None])
        root_z = self.humanoid_rows(self._humanoid_root_states)[rows, 2]
        z = root_z[:, None, None].expand(
            -1, self._num_traj_samples, 1)
        return torch.cat([xy, z], dim=-1)

    def _compute_task_obs(self, env_ids=None):
        rows = self.all_rows() if env_ids is None else self.agent_rows(env_ids)
        root = self.humanoid_rows(self._humanoid_root_states)[rows]
        traj = compute_location_observations(
            root, self._fetch_traj_samples(env_ids))
        blocks = torch.zeros(
            (len(rows), sum(TASK_DIMS)), device=self.device)
        blocks[:, :TASK_DIMS[0]] = traj
        onehot = torch.zeros(
            (len(rows), len(TASK_DIMS)), device=self.device)
        onehot[:, 0] = 1.0
        return torch.cat([blocks, onehot], dim=-1)

    def _compute_observations(self, env_ids=None):
        self._update_command_state()
        humanoid_obs = Humanoid._compute_humanoid_obs(self, None)
        rows = self.all_rows() if env_ids is None else self.agent_rows(env_ids)
        obs = torch.cat(
            [humanoid_obs[rows], self._compute_task_obs(env_ids)], dim=-1)
        if env_ids is None:
            self.obs_buf[:] = obs
        else:
            self.obs_buf[rows] = obs

    def pre_physics_step(self, actions):
        Humanoid.pre_physics_step(self, actions)
        self._prev_root_pos[:] = self.humanoid_rows(
            self._humanoid_root_states)[:, 0:3]

    def _compute_reward(self, actions):
        root = self.humanoid_rows(self._humanoid_root_states)
        relative = root[:, 0:2] - self._path_start
        along = (relative * self._path_dir).sum(dim=-1)
        lateral = torch.linalg.vector_norm(
            relative - along[:, None] * self._path_dir, dim=-1)
        velocity = root[:, 7:9]
        forward_vel = (velocity * self._path_dir).sum(dim=-1)
        lateral_vel = torch.linalg.vector_norm(
            velocity - forward_vel[:, None] * self._path_dir, dim=-1)
        moving = torch.exp(
            -2.0 * (forward_vel - self.stop_nominal_speed).square()
            -2.0 * lateral_vel.square())
        stopped = torch.exp(
            -5.0 * torch.linalg.vector_norm(velocity, dim=-1).square())
        velocity_reward = torch.where(self._stop_mask, stopped, moving)
        path_reward = torch.exp(-2.0 * lateral.square())
        creep = torch.relu(
            torch.linalg.vector_norm(velocity, dim=-1) -
            self.stop_creep_speed)
        creep_penalty = (
            -self.stop_creep_penalty * creep.square() * self._stop_mask.float())
        power = torch.abs(
            self.dof_force_tensor * self.humanoid_rows(self._dof_vel)
        ).sum(dim=-1)
        power_reward = (
            -self._power_coefficient * power if self._power_reward
            else torch.zeros_like(power))
        self.rew_buf[:] = (
            self.stop_path_reward_w * path_reward +
            self.stop_velocity_reward_w * velocity_reward +
            creep_penalty + power_reward)

    def _compute_reset(self):
        Humanoid._compute_reset(self)

    def _reset_envs(self, env_ids):
        if len(env_ids) == 0:
            return
        live = env_ids[self.progress_buf[env_ids] > 0]
        if len(live) > 0:
            self._finish_metrics(live)
        self._reset_default_env_ids = torch.empty(
            0, device=self.device, dtype=torch.long)
        self._reset_ref_env_ids = torch.empty(
            0, device=self.device, dtype=torch.long)
        self._reset_actors(env_ids)
        self._layout_paths(env_ids)
        Humanoid._reset_env_tensors(self, env_ids)
        self._refresh_sim_tensors()
        self._compute_observations(env_ids)
        self._init_amp_obs(env_ids)
        self._reset_metrics(env_ids)

    def _reset_actors(self, env_ids):
        if self._state_init == HumanoidTraj.StateInit.Default:
            self._reset_default(env_ids)
        else:
            self._reset_ref_state_init(env_ids)

    def _reset_default(self, env_ids):
        self._humanoid_root_states[env_ids] = (
            self._initial_humanoid_root_states[env_ids])
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        rows = self.agent_rows(env_ids)
        self._reset_default_env_ids = rows
        self._every_env_init_dof_pos[rows] = self.humanoid_rows(
            self._initial_dof_pos)[rows]

    def _reset_ref_state_init(self, env_ids):
        rows = self.agent_rows(env_ids)
        motion_ids = self._motion_lib.sample_motions(len(rows))
        if self._state_init in (
                HumanoidTraj.StateInit.Random, HumanoidTraj.StateInit.Hybrid):
            motion_times = self._motion_lib.sample_time(motion_ids)
        else:
            motion_times = torch.zeros(len(rows), device=self.device)
        states = self._motion_lib.get_motion_state(motion_ids, motion_times)
        root_pos, root_rot, dof_pos = states[0], states[1], states[2]
        root_vel, root_ang_vel, dof_vel = states[3], states[4], states[5]
        env, agent = torch.div(
            rows, self.num_agents, rounding_mode="floor"), rows % self.num_agents
        humanoid = self.agent_axis(self._humanoid_root_states)
        humanoid[env, agent, 0:3] = root_pos
        humanoid[env, agent, 3:7] = root_rot
        humanoid[env, agent, 7:10] = root_vel
        humanoid[env, agent, 10:13] = root_ang_vel
        self.agent_axis(self._dof_pos)[env, agent] = dof_pos
        self.agent_axis(self._dof_vel)[env, agent] = dof_vel
        self._every_env_init_dof_pos[rows] = dof_pos
        self._reset_ref_env_ids = rows
        self._reset_ref_motion_ids = motion_ids
        self._reset_ref_motion_times = motion_times

    def _compute_amp_observations(self, env_ids=None):
        body_pos = self.humanoid_rows(self._rigid_body_pos)
        body_rot = self.humanoid_rows(self._rigid_body_rot)
        body_vel = self.humanoid_rows(self._rigid_body_vel)
        body_ang = self.humanoid_rows(self._rigid_body_ang_vel)
        dof_pos = self.humanoid_rows(self._dof_pos)
        dof_vel = self.humanoid_rows(self._dof_vel)
        rows = self.all_rows() if env_ids is None else self.agent_rows(env_ids)
        key_pos = body_pos[rows][:, self._key_body_ids, :]
        amp_obs = build_amp_observations(
            body_pos[rows, 0], body_rot[rows, 0],
            body_vel[rows, 0], body_ang[rows, 0],
            dof_pos[rows], dof_vel[rows], key_pos,
            self._local_root_obs, self._root_height_obs,
            self._dof_obs_size, self._dof_offsets)
        self._curr_amp_obs_buf[rows] = self._with_traj_amp_label(amp_obs)

    def _init_amp_obs(self, env_ids):
        rows = self.agent_rows(env_ids)
        self._compute_amp_observations(env_ids)
        if len(self._reset_default_env_ids) > 0:
            current = self._curr_amp_obs_buf[
                self._reset_default_env_ids].unsqueeze(1)
            self._hist_amp_obs_buf[self._reset_default_env_ids] = current
        if len(self._reset_ref_env_ids) > 0:
            count = self._num_amp_obs_steps - 1
            motion_ids = torch.tile(
                self._reset_ref_motion_ids.unsqueeze(-1),
                [1, count]).reshape(-1)
            times = (
                self._reset_ref_motion_times.unsqueeze(-1)
                - self.dt * (
                    torch.arange(count, device=self.device) + 1)
            ).reshape(-1)
            states = self._motion_lib.get_motion_state(motion_ids, times)
            history = build_amp_observations(
                states[0], states[1], states[3], states[4],
                states[2], states[5], states[6],
                self._local_root_obs, self._root_height_obs,
                self._dof_obs_size, self._dof_offsets)
            history = self._with_traj_amp_label(history)
            self._hist_amp_obs_buf[self._reset_ref_env_ids] = history.view(
                len(self._reset_ref_env_ids), count, -1)
        assert len(rows) == len(env_ids) * self.num_agents

    def post_physics_step(self):
        Humanoid.post_physics_step(self)
        self._update_hist_amp_obs()
        self._compute_amp_observations()
        self.extras["amp_obs"] = self._amp_obs_buf.view(
            -1, self.get_num_amp_obs())
        self.extras["policy_obs"] = self.obs_buf.clone()
        self.extras["mastop_stop"] = self._stop_mask.float()
        self.extras["mastop_scenario"] = self._scenario.repeat_interleave(
            self.num_agents).float()
        self.extras["mastop_yield_active"] = self._yield_active.repeat_interleave(
            self.num_agents).float()
        self.extras["mastop_pred_min_dist"] = (
            self._pred_min_dist if self.num_agents == 1
            else self._pred_min_dist.repeat_interleave(self.num_agents))
        if self._is_eval:
            self._compute_metrics_evaluation()
            self.extras["success"] = self._success_buf
            self.extras["precision"] = self._precision_buf
        self._step_metrics()

    def _compute_metrics_evaluation(self):
        root = self.humanoid_rows(self._humanoid_root_states)
        final = self._path_start + self._path_dir * self.stop_path_len
        final_error = torch.linalg.vector_norm(
            final - root[:, 0:2], dim=-1)
        self._success_buf[final_error <= self._success_threshold] = 1

        relative = root[:, 0:2] - self._path_start
        along = (relative * self._path_dir).sum(dim=-1)
        lateral = torch.linalg.vector_norm(
            relative - along[:, None] * self._path_dir, dim=-1)
        self._precision_buf += lateral

    def _agent_body_min_dist(self):
        if self.num_agents < 2:
            return torch.full(
                (self.num_envs,), 1e6, device=self.device)
        return torch.cdist(
            self._rigid_body_pos[:, 0],
            self._rigid_body_pos[:, 1]).amin(dim=(1, 2))

    def _step_metrics(self):
        distance = self._agent_body_min_dist()
        self._met_min_body[:] = torch.minimum(
            self._met_min_body, distance)
        self._met_collision_steps += (
            distance < self.stop_collision_dist).long()
        stop = self._stop_mask.view(self.num_envs, self.num_agents)
        self._met_stop_steps += stop.long()
        speed = torch.linalg.vector_norm(self.humanoid_rows(self._humanoid_root_states)[:, 7:9], dim=-1).view(self.num_envs, self.num_agents)
        self._met_stop_speed_sum += speed * stop.float()
        go = ~stop
        self._met_go_speed_sum += speed * go.float()
        self._met_go_steps += go.long()
        self._met_both_stop_steps += stop.all(dim=1).long()
        self._met_resumed += self._resume_event.long()

        new_resume = self._resume_event & (~self._resume_pending)
        self._resume_pending[new_resume] = True
        self._resume_timer[new_resume] = 0
        env = torch.arange(self.num_envs, device=self.device)
        resume_speed = speed[env, self._yield_agent]
        recovered = self._resume_pending & (
            resume_speed >=
            self.stop_resume_speed * self.stop_nominal_speed)
        self._met_resume_latency[recovered] = (
            self._resume_timer[recovered].float() * self.dt)
        self._resume_pending[recovered] = False
        self._resume_timer[self._resume_pending] += 1

    def _reset_metrics(self, env_ids):
        self._met_min_body[env_ids] = 1e6
        self._met_collision_steps[env_ids] = 0
        self._met_stop_steps[env_ids] = 0
        self._met_stop_speed_sum[env_ids] = 0.0
        self._met_both_stop_steps[env_ids] = 0
        self._met_resumed[env_ids] = 0
        self._met_go_speed_sum[env_ids] = 0.0
        self._met_go_steps[env_ids] = 0
        self._met_resume_latency[env_ids] = -1.0
        self._resume_pending[env_ids] = False
        self._resume_timer[env_ids] = 0
        if self._is_eval:
            rows = self.agent_rows(env_ids)
            self._success_buf[rows] = 0
            self._precision_buf[rows] = 0.0

    def _finish_metrics(self, env_ids):

        for env in env_ids.tolist():
            avg_stop_speed = self._met_stop_speed_sum[env] / self._met_stop_steps[env].clamp(min=1)
            avg_go_speed = self._met_go_speed_sum[env] / self._met_go_steps[env].clamp(min=1)
            self._metric_rows.append([
                float(self._scenario[env]), float(self._yield_agent[env]),
                float(self._met_min_body[env]),
                float(self._met_collision_steps[env]),
                *[float(v) for v in self._met_stop_steps[env]],
                float(self._met_both_stop_steps[env]),
                float(self._met_resumed[env]),
                float(self.progress_buf[env]),
                *[float(v) for v in avg_stop_speed],
                *[float(v) for v in avg_go_speed],
                float(self._met_resume_latency[env]),
                float(self._pred_min_dist[env]),
                float(self._pred_min_time[env])])
        if self.stop_metrics_path:
            self.report_metrics()

    def report_metrics(self):
        output = self.stop_metrics_path
        if not output or not self._metric_rows:
            return
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        metrics = np.asarray(self._metric_rows, dtype=np.float32)
        np.save(output, metrics)
        print(f"[mastop] metrics -> {output} {metrics.shape}", flush=True)

    def _update_marker(self):
        samples = self._fetch_traj_samples().view(
            self.num_envs, self.num_agents,
            self._num_traj_samples, 3)
        self._traj_marker_pos[:] = samples
        self._traj_marker_pos[..., 2] = 0.05
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(self._traj_marker_actor_ids),
            len(self._traj_marker_actor_ids))

    def _draw_task(self):
        self._update_marker()
        self.gym.clear_lines(self.viewer)
        colors = [
            np.array([[0.2, 0.9, 0.2]], np.float32),
            np.array([[0.2, 0.5, 1.0]], np.float32)]
        for env, env_ptr in enumerate(self.envs):
            for agent in range(self.num_agents):
                row = env * self.num_agents + agent
                start = self._path_start[row]
                end = start + self._path_dir[row] * self.stop_path_len
                line = np.array([[
                    float(start[0]), float(start[1]), 0.03,
                    float(end[0]), float(end[1]), 0.03]], np.float32)
                self.gym.add_lines(
                    self.viewer, env_ptr, 1, line,
                    colors[agent % len(colors)])
