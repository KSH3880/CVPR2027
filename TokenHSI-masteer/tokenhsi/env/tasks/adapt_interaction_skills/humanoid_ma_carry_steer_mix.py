"""Equal mix of native two-agent carry and box-free steering episodes."""

import os

import numpy as np
import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate

from env.tasks.adapt_interaction_skills.humanoid_ma_carry import CARRY_HI, CARRY_LO, TEAMMATE_DIM
from env.tasks.adapt_interaction_skills.humanoid_ma_steer_carry import HumanoidMASteerCarry
from tokenhsi.utils import steer_path as sp
from utils import torch_utils


class HumanoidMACarrySteerMix(HumanoidMASteerCarry):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        if int(cfg["env"].get("numAgents", 1)) != 2:
            raise ValueError("HumanoidMACarrySteerMix requires two agents")
        self._solo_dist = float(os.environ.get("MS_STEER_ONLY_DIST", "3.0"))
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)

    def _create_envs(self, num_envs, spacing, num_per_row):
        super()._create_envs(num_envs, spacing, num_per_row)
        order = np.random.default_rng(self.steer_seed + 81).permutation(num_envs)
        solo = np.zeros(num_envs, dtype=np.bool_)
        solo[order[:num_envs // 2]] = True
        self._solo_env = torch.as_tensor(solo, device=self.device)
        self._solo_goal = torch.zeros(self._rows, 3, device=self.device)
        self._solo_prev_yaw = torch.zeros(self._rows, device=self.device)
        self._solo_stop_count = torch.zeros(num_envs, dtype=torch.long, device=self.device)

    def _post_object_reset(self, env_ids):
        super()._post_object_reset(env_ids)
        solo_envs = env_ids[self._solo_env[env_ids]]
        if len(solo_envs) == 0:
            return
        rows = self.agent_rows(solo_envs)
        self._solo_stop_count[solo_envs] = 0
        roots = self.humanoid_rows(self._humanoid_root_states)[rows]
        xy = roots[:, :2]
        other_xy = xy.reshape(-1, 2, 2).flip(1).reshape(-1, 2)
        direction = torch.nn.functional.normalize(xy - other_xy, dim=-1)
        goal = xy + self._solo_dist * direction
        waypoint = xy + 0.15 * direction
        self._steer_tick += 1
        path, _, n_end = sp.gen_full_v2(
            xy, waypoint, goal, self.steer_seed + self._steer_tick + int(rows[0]),
            0.0, 0.1, 120.0, p_two=0.0, skew=0.8,
            spread=(0.85, 1.8), lat_frac=0.25, with_end=True,
        )
        self._gt_path[rows] = path
        self._s_end[rows] = (n_end.float() - 1.0) * sp.DS
        cell_arc = torch.arange(sp.V, device=self.device, dtype=path.dtype) * sp.DS
        remaining = self._s_end[rows, None] - cell_arc[None, :]
        taper = (remaining / 0.4).clamp(0.0, 1.0)
        self._mscale[rows] *= taper
        self._arc_root[rows] = 0.0
        self._prev_arc[rows] = 0.0
        self._solo_goal[rows, :2] = goal
        self._solo_goal[rows, 2] = roots[:, 2]
        forward = torch.zeros_like(roots[:, :3])
        forward[:, 0] = 1.0
        facing = torch.nn.functional.normalize(
            quat_rotate(roots[:, 3:7], forward)[:, :2], dim=-1
        )
        self._solo_prev_yaw[rows] = torch.acos(
            (facing * direction).sum(dim=-1).clamp(-1.0, 1.0)
        )

        boxes = self.humanoid_rows(self._box_states)
        boxes[rows, :2] = xy + 100.0
        boxes[rows, 7:13] = 0.0
        actor_ids = self._box_actor_ids[rows]
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(actor_ids), len(actor_ids),
        )

    def _compute_task_obs(self, env_ids=None):
        obs = super()._compute_task_obs(env_ids)
        rows = self.all_rows() if env_ids is None else self.agent_rows(env_ids)
        solo = self._solo_env[torch.div(rows, self.num_agents, rounding_mode="floor")]
        if not bool(solo.any()):
            return obs
        obs = obs.clone()
        solo_rows = rows[solo]
        roots = self.humanoid_rows(self._humanoid_root_states)[solo_rows]
        goal = self._solo_goal[solo_rows].clone()
        goal[:, 2] = roots[:, 2]
        local_goal = quat_rotate(
            torch_utils.calc_heading_quat_inv(roots[:, 3:7]),
            goal - roots[:, :3],
        )
        start = TEAMMATE_DIM * (self.num_agents - 1) + self.steer_dim()
        carry_dim = CARRY_HI - CARRY_LO
        for offset in (0, carry_dim):
            obs[solo, start + offset:start + offset + carry_dim - 3] = 0.0
            obs[solo, start + offset + carry_dim - 3:start + offset + carry_dim] = local_goal
        return obs

    def _compute_reward(self, actions):
        super()._compute_reward(actions)
        rows = self.all_rows()[self._solo_env.repeat_interleave(self.num_agents)]
        if len(rows) == 0:
            return
        roots = self.humanoid_rows(self._humanoid_root_states)[rows]
        arc = self._arc_root[rows]
        remaining = (self._s_end[rows] - arc).clamp(min=0.0)
        speed = (arc - self._prev_arc[rows]) / self.dt
        command = self._m_at(arc, rows) / 1.6
        moving = remaining > 0.4
        speed_reward = torch.exp(-4.0 * (command - speed).square())
        speed_reward = torch.where(speed > 0.0, speed_reward, 0.0)
        path_penalty = self.steer_pos_c * (
            torch.exp(-0.5 * self._lat_root[rows].square()) - 1.0
        )
        index = (arc / sp.DS).long().clamp(0, sp.V - 2)
        tangent = torch.nn.functional.normalize(
            self._gt_path[rows, index + 1] - self._gt_path[rows, index], dim=-1
        )
        forward = torch.zeros_like(roots[:, :3])
        forward[:, 0] = 1.0
        facing = torch.nn.functional.normalize(
            quat_rotate(roots[:, 3:7], forward)[:, :2], dim=-1
        )
        alignment = (facing * tangent).sum(dim=-1).clamp(-1.0, 1.0)
        yaw = torch.acos(alignment)
        yaw_progress = self._solo_prev_yaw[rows] - yaw
        self._solo_prev_yaw[rows] = yaw
        cosine_gain = alignment.clamp(0.0, 1.0)
        heading_reward = (2.0 * cosine_gain + 1.5) * yaw_progress
        stop = torch.exp(-4.0 * roots[:, 7:9].square().sum(dim=-1))
        reward = torch.where(
            moving,
            2.0 * (0.2 * self.steer_vel_w * speed_reward + path_penalty)
            + heading_reward,
            stop,
        )
        if self._power_reward:
            power = torch.abs(
                self.dof_force_tensor[rows] * self.humanoid_rows(self._dof_vel)[rows]
            ).sum(dim=-1)
            reward -= self._power_coefficient * power
        self.rew_buf[rows] = reward

    def _compute_reset(self):
        super()._compute_reset()
        env_ids = torch.where(self._solo_env)[0]
        if len(env_ids) == 0:
            return
        rows = self.agent_rows(env_ids)
        roots = self.humanoid_rows(self._humanoid_root_states)[rows]
        close = (
            (roots[:, :2] - self._solo_goal[rows, :2]).norm(dim=-1) < 0.18
        )
        still = roots[:, 7:9].norm(dim=-1) < 0.2
        ready = (close & still).view(-1, self.num_agents).all(dim=1)
        self._solo_stop_count[env_ids] = torch.where(
            ready, self._solo_stop_count[env_ids] + 1,
            torch.zeros_like(self._solo_stop_count[env_ids]),
        )
        self.reset_buf[env_ids] |= (self._solo_stop_count[env_ids] >= 10).long()

    def _metric_extra_cols(self, rows):
        cols = super()._metric_extra_cols(rows)
        env = torch.div(rows, self.num_agents, rounding_mode="floor")
        cols.append(self._solo_env[env].float())
