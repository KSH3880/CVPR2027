"""Carry-free multi-agent trajectory steering.

This task deliberately keeps three concerns separate:

* ``HumanoidTrajSitCarryClimb`` owns the validated A>=2 simulator tensors,
  resets, AMP observations and native trajectory generator.
* This file adds only a root-only teammate token and exposes the native traj
  observation twice for the adapt builder (new/old tokenizer contract).
* The optional network-side ``MS_HIER_TARGET=traj`` head edits that trajectory
  command before the frozen stage1 locomotion actor consumes it.

No box state enters the policy observation or reward.  The parent still builds
its inactive task assets because removing those actors is a separate simulator
refactor; all episodes are forced to the traj task and inactive objects are
placed below the scene by the existing reset code.

Observation (A=2, default K=10):
    self 223 | teammate 12 | new_traj 20 | old_traj 20 = 275
"""

import os

import torch

from env.tasks.multi_task.humanoid_traj_sit_carry_climb import HumanoidTrajSitCarryClimb
from utils import torch_utils
from isaacgym.torch_utils import quat_mul, quat_rotate


TEAMMATE_ROOT_DIM = 12


class HumanoidMASteer(HumanoidTrajSitCarryClimb):
    """A>=2 trajectory following with teammate-aware collision avoidance."""

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._ma_steer_token = os.environ.get("MA_TOKEN", "live")
        assert self._ma_steer_token in ("live", "zero"), self._ma_steer_token
        # Existing shared viewers pass --eval_task carry unconditionally.  The
        # task identity is not a viewer concern, so make this class self-contained
        # and force its own evaluation task before the parent reads it.
        if cfg["args"].eval:
            cfg["args"].eval_task = "traj"
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine,
                         device_type=device_type, device_id=device_id, headless=headless)

        # The shared parent conservatively forces A>=2 to carry.  This subclass
        # is the audited traj-only exception: traj generator/reset/reward are
        # already row-aware, and no env-owned task object participates.
        traj_id = HumanoidTrajSitCarryClimb.TaskUID["traj"].value
        self._task_init_prob[:] = 0.0
        self._task_init_prob[traj_id] = 1.0
        print(f"[ma-steer] A={self.num_agents}: traj 전용, carry 비활성", flush=True)

        self._ep_track = torch.zeros(self._rows, device=self.device)
        self._ep_track_n = torch.zeros(self._rows, device=self.device)

    def _traj_dim(self):
        return 2 * self._num_traj_samples

    def get_task_obs_size(self):
        return TEAMMATE_ROOT_DIM * (self.num_agents - 1) + 2 * self._traj_dim()

    def get_multi_task_info(self):
        # Parent observation construction still needs its original four-task
        # mask.  Network construction needs the adapt three-token contract.
        if getattr(self, "_use_base_info", False):
            return HumanoidTrajSitCarryClimb.get_multi_task_info(self)

        teammate_dim = TEAMMATE_ROOT_DIM * (self.num_agents - 1)
        traj_dim = self._traj_dim()
        each = [teammate_dim, traj_dim, traj_dim]
        names = ["new_extra", "new_traj", "old_traj"]
        index = torch.cumsum(torch.tensor([0] + each), dim=0).to(self.device)
        mask = torch.zeros(len(each), sum(each), dtype=torch.bool, device=self.device)
        for i in range(len(each)):
            mask[i, index[i]:index[i + 1]] = True
        return {
            "onehot_size": len(each),
            "tota_subtask_obs_size": sum(each),
            "each_subtask_obs_size": each,
            "each_subtask_obs_mask": mask,
            "each_subtask_obs_indx": index,
            "enable_task_mask_obs": False,
            "each_subtask_name": names,
            "major_task_name": "traj",
            "has_extra": True,
        }

    def _teammate_root_obs(self, rows):
        """Relative root pose/velocity only; boxes do not exist semantically."""
        A = self.num_agents
        if A == 1:
            return torch.zeros((len(rows), 0), device=self.device)

        h = self.humanoid_rows(self._humanoid_root_states)
        e = torch.div(rows, A, rounding_mode="floor")
        a = rows % A
        my_pos = h[rows, 0:3]
        heading_inv = torch_utils.calc_heading_quat_inv(h[rows, 3:7])
        out = []
        for k in range(1, A):
            other = e * A + (a + k) % A
            rel_pos = quat_rotate(heading_inv, h[other, 0:3] - my_pos).clamp(-10.0, 10.0)
            rel_vel = quat_rotate(heading_inv, h[other, 7:10])
            rel_heading = torch_utils.quat_to_tan_norm(
                quat_mul(heading_inv, h[other, 3:7]))
            out.append(torch.cat((rel_pos, rel_vel, rel_heading), dim=-1))
        obs = torch.cat(out, dim=-1)
        return torch.zeros_like(obs) if self._ma_steer_token == "zero" else obs

    def _compute_task_obs(self, env_ids=None):
        # Reuse the native local trajectory observation instead of maintaining
        # a second path/window implementation in this task.
        self._use_base_info = True
        try:
            parent = super()._compute_task_obs(env_ids)
        finally:
            self._use_base_info = False
        traj = parent[:, :self._traj_dim()]
        rows = self.all_rows() if env_ids is None else self.agent_rows(env_ids)
        return torch.cat((self._teammate_root_obs(rows), traj, traj), dim=-1)

    def update_metrics(self):
        """Carry-free subset of the parent's MA metrics plus traj error."""
        rows = self.all_rows()
        d = self.agent_min_dist()
        close = d < self._metric_tau
        self._ep_collide += close.long()
        self._ep_dmin = torch.minimum(self._ep_dmin, d)
        for i, threshold in enumerate(self._DBANDS):
            self._ep_band[:, i] += (d < threshold).float()
        self._ep_enters += (close & ~self._was_close).long()
        self._was_close = close

        root = self.humanoid_rows(self._humanoid_root_states)
        step_dist = torch.norm(root[:, 0:3] - self._prev_root_pos, p=2, dim=-1)
        self._ep_path += step_dist
        self._ep_still += (step_dist / self.dt < self._metric_still_v).long()

        time = self.progress_rows() * self.dt
        target = self._traj_gen.calc_pos(rows, time)
        self._ep_track += torch.norm(root[:, 0:2] - target[:, 0:2], dim=-1)
        self._ep_track_n += 1.0
        return

    def _metric_extra_cols(self, rows):
        return [self._ep_track[rows], self._ep_track_n[rows]]

    def _metric_reset_extra(self, rows):
        self._ep_track[rows] = 0.0
        self._ep_track_n[rows] = 0.0
        return

    def _apply_makespan_reward(self):
        # There is no terminal delivery event in continuous trajectory
        # tracking, so the carry-specific makespan bonus is undefined.
        return

    def record_traj(self):
        """Optional carry-free trace: progress, root xy and current target xy."""
        out = os.environ.get("MA_TRAJ")
        if not out:
            return
        n = min(int(os.environ.get("MA_TRAJ_ENVS", 24)), self.num_envs)
        A = self.num_agents
        if not hasattr(self, "_traj_rec"):
            self._traj_rec = []
        root = self.agent_axis(self._humanoid_root_states)[:n, :, 0:2]
        rows = torch.arange(n * A, device=self.device)
        time = self.progress_rows()[rows] * self.dt
        target = self._traj_gen.calc_pos(rows, time)[:, 0:2].view(n, A, 2)
        progress = self.progress_buf[:n, None, None].float().expand(n, A, 1)
        self._traj_rec.append(torch.cat((progress, root, target), dim=-1).cpu())
        if len(self._traj_rec) >= int(os.environ.get("MA_TRAJ_STEPS", 400)):
            import numpy as np
            data = torch.stack(self._traj_rec, dim=0).numpy()
            np.save(out, data)  # (T, env, agent, progress+root_xy+target_xy)
            print(f"[ma-steer] traj -> {out} {data.shape}", flush=True)
            raise SystemExit(0)
        return