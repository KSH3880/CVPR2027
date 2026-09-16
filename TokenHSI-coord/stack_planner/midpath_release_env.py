"""Planner-free visual diagnostic for release at an intermediate path point."""

from __future__ import annotations

import os
import numpy as np
import torch

from env.tasks.adapt_interaction_skills.humanoid_ma_steer_carry import (
    HumanoidMASteerCarry,
)
from stack_planner.reset_transaction import CarryOnlySingleCommitReset


class HumanoidMAMidpathReleaseView(
    CarryOnlySingleCommitReset, HumanoidMASteerCarry
):
    """Plain carry task whose steering route continues past the carry goal."""

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._midpath_extension = float(os.environ.get("STACK_MIDPATH_EXTENSION", "2.0"))
        if self._midpath_extension <= 0.0:
            raise ValueError("STACK_MIDPATH_EXTENSION must be positive")
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        if not hasattr(self, "_midpath_endpoint"):
            self._midpath_endpoint = torch.zeros(
                self.num_envs, 2, device=self.device
            )
            self._midpath_installed = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        self._midpath_goal = torch.zeros(self.num_envs, 3, device=self.device)
        print("[midpath-release] planner=disabled task=plain-steer-carry "
              "extension={:.2f}m; carry goal is an intermediate path point".format(
                  self._midpath_extension), flush=True)

    def _update_marker(self):
        # Generic multi-task rendering submits disabled sit/climb marker ids;
        # this carry-only diagnostic has no valid actors for those ids.
        return

    def _draw_task(self):
        """Draw exactly the A1 path consumed by the frozen steering policy."""
        if self.viewer is None:
            return
        self.gym.clear_lines(self.viewer)
        if not hasattr(self, "_midpath_endpoint"):
            return
        path = self._gt_path.detach().cpu().numpy()
        s_end = self._s_end.detach().cpu().numpy()
        goals = self._midpath_goal.detach().cpu().numpy()
        endpoints = self._midpath_endpoint.detach().cpu().numpy()
        ds = float(self._gt_path_ds) if hasattr(self, "_gt_path_ds") else 0.1

        cyan = np.array([[0.05, 1.0, 1.0]], dtype=np.float32)
        yellow = np.array([[1.0, 0.9, 0.1]], dtype=np.float32)
        red = np.array([[1.0, 0.15, 0.1]], dtype=np.float32)
        for env_id, env_ptr in enumerate(self.envs):
            row = env_id * self.num_agents
            end = min(int(s_end[row] / ds) + 1, path.shape[1])
            points = path[row, :max(end, 2)]
            start, finish = points[:-1], points[1:]
            delta = finish - start
            normal = np.stack((-delta[:, 1], delta[:, 0]), axis=-1)
            normal /= np.maximum(
                np.linalg.norm(normal, axis=-1, keepdims=True), 1e-6
            )
            strips = []
            # Isaac Gym lines have no thickness. Parallel lines form a bright
            # 36-cm ribbon that remains visible over the checkerboard floor.
            for offset in np.linspace(-0.18, 0.18, 19):
                a = start + normal * offset
                b = finish + normal * offset
                strips.append(np.concatenate((
                    np.column_stack((a, np.full(len(a), 0.14))),
                    np.column_stack((b, np.full(len(b), 0.14))),
                ), axis=1))
            lines = np.concatenate(strips, axis=0).astype(np.float32)
            self.gym.add_lines(
                self.viewer, env_ptr, len(lines), lines,
                np.repeat(cyan, len(lines), axis=0),
            )

            def cross(xy, color, z):
                radius = 0.28
                marker = np.array([
                    [xy[0] - radius, xy[1], z, xy[0] + radius, xy[1], z],
                    [xy[0], xy[1] - radius, z, xy[0], xy[1] + radius, z],
                ], dtype=np.float32)
                self.gym.add_lines(
                    self.viewer, env_ptr, 2, marker,
                    np.repeat(color, 2, axis=0),
                )

            cross(goals[env_id, :2], yellow, 0.08)
            cross(endpoints[env_id], red, 0.09)

    def _post_object_reset(self, env_ids):
        super()._post_object_reset(env_ids)
        if hasattr(self, "_midpath_installed"):
            self._midpath_installed[env_ids] = False

    def install_midpath_paths(self, env_ids=None):
        """Install A1 root→box→carry-goal→endpoint without changing its goal."""
        if env_ids is None:
            env_ids = torch.arange(
                self.num_envs, device=self.device, dtype=torch.long
            )
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if len(env_ids) == 0:
            return
        rows = self.agent_rows(env_ids).reshape(-1, self.num_agents)
        r0 = rows[:, 0]
        boxes = self.humanoid_rows(self._box_states)[r0, :2]
        roots = self.humanoid_rows(self._humanoid_root_states)[r0, :2]
        # This is the ordinary carry target consumed by reward/observation.
        # It remains unchanged after route construction.
        goal = self._box_tar_pos[r0].clone()
        direction = goal[:, :2] - boxes
        norm = direction.norm(dim=-1, keepdim=True)
        direction = torch.where(norm > 1e-4, direction, goal[:, :2] - roots)
        norm = direction.norm(dim=-1, keepdim=True)
        fallback = torch.zeros_like(direction)
        fallback[:, 0] = 1.0
        direction = torch.where(norm > 1e-4, direction / norm.clamp(min=1e-4), fallback)
        endpoint = goal.clone()
        endpoint[:, :2] += self._midpath_extension * direction
        # _reset_steer has no endpoint argument, so expose the extended point
        # only while it builds the route. Restore the real carry target before
        # the next observation/reward is computed. Because endpoint extends
        # the box->goal ray, the real goal lies on that route's second leg.
        saved_goal = self._box_tar_pos[r0].clone()
        try:
            self._box_tar_pos[r0] = endpoint
            self._reset_steer(r0)
        finally:
            self._box_tar_pos[r0] = saved_goal
        if not hasattr(self, "_midpath_endpoint"):
            self._midpath_endpoint = torch.zeros(
                self.num_envs, 2, device=self.device
            )
            self._midpath_installed = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        self._midpath_endpoint[env_ids] = endpoint[:, :2]
        self._midpath_goal[env_ids] = goal
        self._midpath_installed[env_ids] = True
        if bool((env_ids == 0).any()):
            i = int(torch.nonzero(env_ids == 0, as_tuple=False)[0])
            end_index = torch.clamp(
                (self._s_end[r0[i]] / 0.1).round().long(),
                max=self._gt_path.shape[1] - 1,
            )
            installed_end = self._gt_path[r0[i], end_index]
            install_error = torch.norm(installed_end - endpoint[i, :2])
            print("[midpath-release] env0 carry_goal=({:+.2f},{:+.2f}) "
                  "path_end=({:+.2f},{:+.2f}) installed_error={:.3f}m".format(
                      float(goal[i, 0]), float(goal[i, 1]),
                      float(endpoint[i, 0]), float(endpoint[i, 1]),
                      float(install_error)), flush=True)

__all__ = ["HumanoidMAMidpathReleaseView"]
