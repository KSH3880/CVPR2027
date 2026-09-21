"""Viewer-only task for continuous, box-free steering with an ms18-family policy.

This task is deliberately not used by training or evaluation. It keeps the
checkpoint ABI of HumanoidMASteerCarry, forces A=1 through the launcher, hides
carry objects via eval_task=traj, and replaces the unrelated TokenHSI traj with
one long steering command path per episode.
"""

import os
import torch

from env.tasks.adapt_interaction_skills.humanoid_ma_steer_carry import (
    HumanoidMASteerCarry,
)
from tokenhsi.utils import steer_path as sp


class HumanoidMAOnlySteering(HumanoidMASteerCarry):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        if int(cfg["env"].get("numAgents", 1)) != 1:
            raise ValueError("HumanoidMAOnlySteering requires numAgents=1")
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)

        # eval_task=traj normally checks a separate TokenHSI trajectory and ends
        # after episodeLengthShort. Neither applies to this viewer-only path.
        self._fail_dist = 1.0e9
        self.max_episode_length_short = self.max_episode_length
        print(
            f"[only-steering] continuous path for {self.max_episode_length} frames; "
            "carry objects disabled",
            flush=True,
        )

    def _reset_steer(self, rows):
        # Retain the parent's speed-profile sampling and metric reset, then
        # replace its root->box->target route with a standalone long path.
        super()._reset_steer(rows)
        if len(rows) == 0:
            return

        h = self.humanoid_rows(self._humanoid_root_states)[rows]
        root = h[:, 0:2]
        heading = torch.atan2(
            2.0 * (h[:, 6] * h[:, 5] + h[:, 3] * h[:, 4]),
            1.0 - 2.0 * (h[:, 4] ** 2 + h[:, 5] ** 2),
        )
        direction = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
        target_dist = float(os.environ.get("ONLY_STEERING_TARGET_DIST", "18.0"))
        curvature = float(os.environ.get("ONLY_STEERING_CURVATURE", "0.08"))
        if float(os.environ.get("MS_LAT_MAX", "2.2")) == 0.0:
            curvature = 0.0

        target = root + target_dist * direction
        seed = self.steer_seed + self._steer_tick + int(rows[0]) + 100003
        self._gt_path[rows] = sp.gen_gt(root, target, curvature, seed)

        # gen_gt extends the final tangent to the full 32 m buffer. Treat that
        # extension as valid so the steering window never collapses before the
        # 600-frame viewer episode resets.
        self._s_end[rows] = (sp.V - 1) * sp.DS
        self._arc_root[rows] = 0.0
        self._arc_box[rows] = 0.0
        self._prev_arc[rows] = 0.0
