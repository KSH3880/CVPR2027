"""Frozen ms18 follows explicit routes while box assignments change."""

from contextlib import contextmanager
import os
import numpy as np
import torch

from carry_planner.env_adapter import HumanoidMACarryPlannerTrain
from carry_box_swap.assignment import assigned_inputs, assignment_rows


class HumanoidMACarryBoxSwapMS18(HumanoidMACarryPlannerTrain):
    def __init__(self, cfg, *args, **kwargs):
        self.goals_follow_box = bool(int(os.environ.get("CARRY_BOX_SWAP_GOALS_FOLLOW_BOX", "1")))
        super().__init__(cfg, *args, **kwargs)
        self._enable_IET = False
        # This probe scores complete first episodes, not staggered PPO rollouts.
        self._carry_timeout_deadline.fill_(self.max_episode_length)
        self._ensure_assignment()

    def _ensure_assignment(self):
        if not hasattr(self, "box_assignment"):
            self.box_assignment = torch.arange(2, device=self.device)[None].expand(self.num_envs, -1).clone()

    @contextmanager
    def _assigned(self):
        self._ensure_assignment()
        if getattr(self, "_swap_inputs_active", False):
            yield
            return
        with assigned_inputs(self, self.goals_follow_box):
            self._swap_inputs_active = True
            try:
                yield
            finally:
                self._swap_inputs_active = False

    def _reset_envs(self, env_ids):
        self._ensure_assignment()
        self.box_assignment[env_ids] = torch.tensor([0, 1], device=self.device)
        super()._reset_envs(env_ids)

    def set_box_assignment(self, env_ids, assignment):
        assignment_rows(assignment)
        if len(env_ids) != len(assignment):
            raise ValueError("assignment batch must match env_ids")
        self._ensure_assignment()
        self.box_assignment[env_ids] = assignment
        # Never execute a cached route belonging to the previous assignment.
        self._coord_has_valid[env_ids] = False
        self._plan_envs(env_ids)

    def _coord_state(self, env_ids):
        with self._assigned():
            return super()._coord_state(env_ids)

    def _compute_task_obs(self, env_ids=None):
        with self._assigned():
            return super()._compute_task_obs(env_ids)

    def _compute_reward(self, actions):
        with self._assigned():
            return super()._compute_reward(actions)

    def _plan_envs(self, env_ids):
        if len(env_ids) == 0:
            return
        with self._assigned():
            state = super()._coord_state(env_ids)
            path, speed = self._analytic_plan(state)
            t = torch.linspace(0, 1, 33, device=self.device)
            direct = state.root_xy[:, :, None] + t[None, None, :, None] * (state.goal_xy-state.root_xy)[:, :, None]
            path = torch.where((state.held >= .5)[..., None, None], direct, path)
            self._carry_reset_cursor_on_install = True
            super()._install_plan(env_ids, path, speed)
            if not hasattr(self, "swap_paths"):
                self.swap_paths = torch.zeros(self.num_envs, 2, 33, 2, device=self.device)
            self.swap_paths[env_ids] = path
            self._coord_has_valid[env_ids] = True
            self._coord_last_replan[env_ids] = self.progress_buf[env_ids]
            self._coord_phase[env_ids] = state.phase
            self._coord_replans[env_ids] += 1

    def _maybe_replan(self, env_ids):
        if not self._coord_ready or len(env_ids) == 0:
            return
        state = self._coord_state(env_ids)
        changed = (state.phase != self._coord_phase[env_ids]).any(-1)
        self._plan_envs(env_ids[changed])

    def _draw_task(self):
        # The ms18 ribbon endpoint must use the assigned box's destination too.
        with self._assigned():
            super()._draw_task()
        root = self.humanoid_rows(self._humanoid_root_states).reshape(self.num_envs, 2, -1)
        boxes = self.humanoid_rows(self._box_states)[assignment_rows(self.box_assignment)].reshape(self.num_envs, 2, -1)
        lines = torch.cat((root[..., :3], boxes[..., :3]), -1).cpu().numpy()
        colors = np.asarray([[0., 1., 1.], [1., .5, 0.]], dtype=np.float32)
        for env_id, env in enumerate(self.envs):
            self.gym.add_lines(self.viewer, env, 2, lines[env_id], colors)
