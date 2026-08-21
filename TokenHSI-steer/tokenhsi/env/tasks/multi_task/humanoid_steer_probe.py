"""S0 probe: no training, no new inputs.

Reproduces the rule-based result by overwriting the carry target with a
lookahead point on a ground-truth path each step, while scoring against the
*real* placement target. Everything is configured through STEER_* environment
variables so no shared config file has to change.

    STEER_LOOKAHEAD=1.2 STEER_HANDOFF=1.5 STEER_CURVATURE=0.15 \
    python ./tokenhsi/run.py --task HumanoidSteerProbe ... --test --eval --eval_task carry
"""

import atexit
import os
import numpy as np
import torch

from tokenhsi.env.tasks.multi_task.humanoid_traj_sit_carry_climb import HumanoidTrajSitCarryClimb
from tokenhsi.utils import steer_path as sp


def _f(name, default):
    return float(os.environ.get(name, default))


class HumanoidSteerProbe(HumanoidTrajSitCarryClimb):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.steer_lookahead = _f("STEER_LOOKAHEAD", 1.2)
        self.steer_handoff = _f("STEER_HANDOFF", 1.5)
        self.steer_curvature = _f("STEER_CURVATURE", 0.15)
        self.steer_radius = _f("STEER_RADIUS", 4.0)
        self.steer_sigma = _f("STEER_SIGMA", 0.0)
        self.steer_seed = int(_f("STEER_SEED", 0))
        self.steer_enabled = bool(int(os.environ.get("STEER_ENABLED", "1")))
        self.steer_tag = os.environ.get("STEER_TAG", "probe")
        self.steer_max_log = int(_f("STEER_MAX_LOG", 4000))

        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine,
                         device_type=device_type, device_id=device_id, headless=headless)

        n = self.num_envs
        self._final_tar_pos = torch.zeros(n, 3, device=self.device)
        self._gt_path = torch.zeros(n, sp.V, 2, device=self.device)
        self._episode_id = torch.zeros(n, dtype=torch.long, device=self.device)
        self._steer_step = 0

        t = self.steer_max_log
        self._log = {
            "root": np.zeros((t, n, 2), np.float32),
            "box": np.zeros((t, n, 3), np.float32),
            "s": np.zeros((t, n), np.float32),
            "lat": np.zeros((t, n), np.float32),
            "progress": np.zeros((t, n), np.int16),
            "episode": np.zeros((t, n), np.int32),
            "held": np.zeros((t, n), np.uint8),
            "active": np.zeros((t, n), np.uint8),
        }
        atexit.register(self._dump)

    # ---- ground truth path ------------------------------------------------

    def _reset_task_carry(self, env_ids):
        super()._reset_task_carry(env_ids)
        if len(env_ids) == 0:
            return
        self._final_tar_pos[env_ids] = self._box_tar_pos[env_ids].clone()
        box = self._box_states[env_ids, 0:2]
        tar = self._box_tar_pos[env_ids, 0:2]
        self._gt_path[env_ids] = sp.gen_gt(
            box, tar, self.steer_curvature,
            seed=self.steer_seed + int(self._steer_step) + int(env_ids[0]))
        self._episode_id[env_ids] += 1

    # ---- steering ---------------------------------------------------------

    def _held(self):
        hands = self._rigid_body_pos[:, self._key_body_ids[[0, 1]], :].mean(dim=1)
        return (hands - self._box_states[:, 0:3]).norm(dim=-1) < 0.35

    def _steer_state(self):
        """Project the BOX onto the transport path: the path it must travel."""
        box_xy = self._box_states[:, 0:2]
        path = self._gt_path
        if self.steer_sigma > 0:
            s0, _, _ = sp.project(box_xy, path)
            path = sp.synth_step(path, s0, self.steer_radius, self.steer_sigma,
                                 self.steer_seed * 100003 + self._steer_step)
        s, lat, _ = sp.project(box_xy, path)
        point = sp.window(path, s, 1, self.steer_lookahead)[:, 0]
        return s, lat, point

    def _compute_observations(self, env_ids=None):
        if self.steer_enabled and env_ids is None:
            s, lat, point = self._steer_state()
            remaining = (self._box_states[:, 0:2] - self._final_tar_pos[:, 0:2]).norm(dim=-1)
            # steering only governs transport: before pickup and inside the
            # handoff radius the policy gets its native final target
            active = self._held() & (remaining > self.steer_handoff)
            tar = self._final_tar_pos.clone()
            tar[active, 0:2] = point[active]
            self._box_tar_pos[:] = tar
            self._last_s, self._last_lat, self._last_active = s, lat, active
        super()._compute_observations(env_ids)

    # ---- metrics against the real target ----------------------------------

    def _compute_metrics_evaluation_v3(self):
        saved = self._box_tar_pos
        self._box_tar_pos = self._final_tar_pos
        try:
            super()._compute_metrics_evaluation_v3()
        finally:
            self._box_tar_pos = saved

    def post_physics_step(self):
        super().post_physics_step()
        i = self._steer_step
        if i < self.steer_max_log:
            hands = self._rigid_body_pos[:, self._key_body_ids[[0, 1]], :].mean(dim=1)
            self._log["root"][i] = self._humanoid_root_states[:, 0:2].cpu().numpy()
            self._log["box"][i] = self._box_states[:, 0:3].cpu().numpy()
            self._log["progress"][i] = self.progress_buf.cpu().numpy()
            self._log["episode"][i] = self._episode_id.cpu().numpy()
            self._log["held"][i] = ((hands - self._box_states[:, 0:3]).norm(dim=-1)
                                    < 0.35).cpu().numpy()
            if hasattr(self, "_last_s"):
                self._log["s"][i] = self._last_s.cpu().numpy()
                self._log["lat"][i] = self._last_lat.cpu().numpy()
                self._log["active"][i] = self._last_active.cpu().numpy()
        self._steer_step += 1

    def _dump(self):
        out = os.environ.get("STEER_OUT", "/home/hwanhee/CVPR2027/runs/steer")
        os.makedirs(out, exist_ok=True)
        n = min(self._steer_step, self.steer_max_log)
        if n == 0:
            return
        path = os.path.join(out, f"{self.steer_tag}.npz")
        np.savez_compressed(
            path,
            gt=self._gt_path.cpu().numpy(),
            final_tar=self._final_tar_pos.cpu().numpy(),
            lookahead=self.steer_lookahead,
            handoff=self.steer_handoff,
            curvature=self.steer_curvature,
            radius=self.steer_radius,
            sigma=self.steer_sigma,
            enabled=self.steer_enabled,
            **{k: v[:n] for k, v in self._log.items()},
        )
        print(f"[steer] wrote {path}  steps={n} envs={self.num_envs}", flush=True)
