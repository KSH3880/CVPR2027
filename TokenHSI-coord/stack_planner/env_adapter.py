"""Runtime-only bridge from learned stack trajectories to frozen ms18 Carry.

The planner always observes the real simulator state.  Only this compatibility
boundary converts the learned A1 retreat endpoint into the virtual Carry box
needed by the currently coupled low-level policy.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from isaacgym import gymtorch

from coordinator.schema import MAX_SPEED, MIN_SPEED, PATH_DS, PATH_VERTICES, CoordinatorState
from coordinator.sequential_bridge import carry_rows, plan_validity, resample_plan
from env.tasks.adapt_interaction_skills.humanoid_ma_sequential_stack_carry import (
    HumanoidMASequentialStackCarry,
)
from stack_planner.reward import StackPhysicalState
from tokenhsi.utils import steer_path as sp


class HumanoidMAStackPlannerTrain(HumanoidMASequentialStackCarry):
    """Sequential stack task with an externally installed stack-planner path."""

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        if sp.DS != PATH_DS or sp.V != PATH_VERTICES or self.steer_dim() != 12:
            raise ValueError("stack planner requires ms18's 0.1m/320-point/12-D steer ABI")
        if not self.steer_clip or self.steer_zero:
            raise ValueError("stack planner requires MS_CLIP=1 and MS_ZERO=0")
        self._planner_virtual_retreat_pos = self._a1_retreat_pos.clone()
        self._planner_applied = torch.zeros(self._rows, dtype=torch.bool, device=self.device)

    @staticmethod
    def _yaw(states):
        return torch.atan2(
            2 * (states[..., 6] * states[..., 5] + states[..., 3] * states[..., 4]),
            1 - 2 * (states[..., 4].square() + states[..., 5].square()),
        )

    def planner_state(self, env_ids=None):
        """The unchanged coordinator input contract, using only real box state."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        rows = self.agent_rows(env_ids)
        root = self.humanoid_rows(self._humanoid_root_states)[rows].reshape(-1, 2, 13)
        box = self.humanoid_rows(self._box_states)[rows].reshape(-1, 2, 13)
        size = self._box_lib._box_size[rows].reshape(-1, 2, 3)
        goal = self._box_tar_pos[rows, :2].reshape(-1, 2, 2)
        bottom = box[..., 2] - size[..., 2] / 2
        lifted = bottom > self._initial_box_bottom_z[rows].reshape(-1, 2) + 0.08
        near = (root[..., :2] - box[..., :2]).norm(dim=-1) <= 0.7
        rigid = self.humanoid_rows(self._rigid_body_pos)[rows]
        hands_on = self._handheld_score(
            root.reshape(-1, 13)[:, :3], box.reshape(-1, 13)[:, :3], rigid
        ).reshape(-1, 2) > 0.35
        held = lifted & hands_on
        at_goal = (box[..., :2] - goal).norm(dim=-1) <= 0.15
        phase = torch.where(held, torch.full_like(bottom, 2.0), near.float())
        phase = torch.where(at_goal & ~held, torch.full_like(phase, 3.0), phase)
        return CoordinatorState(
            root[..., :2], self._yaw(root), root[..., 7:9],
            box[..., :3], self._yaw(box), box[..., 7:9],
            size[..., :2], goal, held.float(), phase,
        )

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        if hasattr(self, "_planner_virtual_retreat_pos") and len(env_ids):
            self._planner_virtual_retreat_pos[env_ids] = self._a1_retreat_pos[env_ids]
            self._planner_applied[self.agent_rows(env_ids)] = False

    def _reset_env_tensors(self, env_ids):
        """Commit only actors that exist in this carry-only training task.

        The generic multi-task reset also submits disabled sit/climb actor IDs.
        This workspace intentionally lacks some of those URDF assets, so that
        broad commit can become an illegal CUDA actor access on episode reset.
        """
        humanoids = self._humanoid_actor_ids_per_env[env_ids].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(humanoids), len(humanoids),
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(humanoids), len(humanoids),
        )
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0

        rows = self.agent_rows(env_ids)
        objects = self._box_actor_ids[rows]
        if self._carry_reset_random_height:
            objects = torch.cat((objects, self._platform_actor_ids[rows],
                                 self._tar_platform_actor_ids[rows]))
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(objects), len(objects),
        )
        if getattr(self, "_enable_IET", False):
            self._IET_step_buf[rows] = 0
            self._IET_triggered_buf[rows] = 0
        if getattr(self, "_is_eval", False):
            self._success_buf[rows] = 0
            self._precision_buf[rows] = float("inf")

    def _virtual_retreat_carry_obs(self, rows, env_ids):
        if not hasattr(self, "_planner_virtual_retreat_pos"):
            return super()._virtual_retreat_carry_obs(rows, env_ids)
        original = self._a1_retreat_pos[env_ids].clone()
        try:
            self._a1_retreat_pos[env_ids] = self._planner_virtual_retreat_pos[env_ids]
            return super()._virtual_retreat_carry_obs(rows, env_ids)
        finally:
            self._a1_retreat_pos[env_ids] = original

    @staticmethod
    def _retreat_valid(path, speed, root, active):
        finite = (torch.isfinite(path).flatten(start_dim=-2).all(dim=-1)
                  & torch.isfinite(speed).all(dim=-1))
        anchors = (path[..., 0, :] - root[:, None]).norm(dim=-1) < 0.01
        delta = path[..., 1:, :] - path[..., :-1, :]
        lengths = delta.norm(dim=-1)
        buffer_ok = lengths.sum(dim=-1) < (PATH_VERTICES - 2) * PATH_DS
        product = lengths[..., :-1] * lengths[..., 1:]
        cosine = (delta[..., :-1, :] * delta[..., 1:, :]).sum(-1) / product.clamp(min=1e-8)
        turns = torch.rad2deg(torch.acos(cosine.clamp(-1, 1)))
        turns = torch.where(product > 1e-10, turns, torch.zeros_like(turns))
        speed_ok = ((speed >= MIN_SPEED) & (speed <= MAX_SPEED)).all(dim=-1)
        valid = finite & anchors & buffer_ok & speed_ok & (turns.amax(dim=-1) <= 46)
        return (valid | ~active[:, None]).all(dim=-1)

    @torch.no_grad()
    def install_external_plan(self, output):
        """Install candidate zero; return per-env (valid, conservative-safe)."""
        state = self.planner_state()
        phase = self._stack_phase
        active = carry_rows(phase, self._carry_rehearsal)
        retreat_env = (phase == self.A1_RETREAT) & ~self._carry_rehearsal
        path = output["path_world"][:, 0].clone()
        speed = output["speed"][:, 0].clone()
        if retreat_env.any():
            path[retreat_env, 0] = output["retreat_path_world"][retreat_env, 0, 0]
            speed[retreat_env, 0] = output["retreat_speed"][retreat_env, 0, 0]
        ordinary_valid = plan_validity(
            output["path_world"], output["speed"], state, active
        )[:, 0]
        retreat_valid = self._retreat_valid(
            output["retreat_path_world"], output["retreat_speed"],
            state.root_xy, active,
        )[:, 0]
        valid = torch.where(retreat_env, retreat_valid, ordinary_valid)

        # Inactive roots are stationary for the safety projection.
        projected = torch.where(
            active[..., None, None], path,
            state.root_xy[..., None, :].expand_as(path),
        )
        separation = (projected[:, 0] - projected[:, 1]).norm(dim=-1)
        safe = separation.amin(dim=-1) >= 0.8

        install = valid[:, None] & active
        if retreat_env.any():
            endpoint = output["retreat_path_world"][:, 0, 0, -1]
            self._planner_virtual_retreat_pos[retreat_env, :2] = endpoint[retreat_env]
        if install.any():
            dense, dense_speed, end = resample_plan(path, speed)
            flat_install = install.reshape(-1)
            rows = self.all_rows()[flat_install]
            flat_path = dense.reshape(-1, sp.V, 2)[flat_install]
            self._gt_path[rows] = flat_path
            self._mscale[rows] = dense_speed.reshape(-1, sp.V)[flat_install] / MAX_SPEED
            self._s_end[rows] = end.reshape(-1)[flat_install]
            self._arc_root[rows] = 0
            self._prev_arc[rows] = 0
            planner_box = state.box_xyz[..., :2].clone()
            planner_box[retreat_env, 0] = self._planner_virtual_retreat_pos[retreat_env, :2]
            self._arc_box[rows] = sp.project(
                planner_box.reshape(-1, 2)[flat_install], flat_path
            )[0]
            self._planner_applied[rows] = True
        return valid, safe

    def planner_physical_state(self):
        rows = self.all_rows().view(self.num_envs, 2)
        r0, r1 = rows[:, 0], rows[:, 1]
        roots = self.humanoid_rows(self._humanoid_root_states)
        boxes = self.humanoid_rows(self._box_states)
        bottom_target = torch.where(
            self._top_committed[:, None], self._committed_bottom_pos,
            self._bottom_nominal_pos,
        )
        bottom = boxes[r0]
        top = boxes[r1]
        # Define stacking from current physical support geometry, even before
        # the task has exposed A2's command.  This avoids rewarding A2 merely
        # for remaining at its stage/wait target.
        top_target = bottom[:, :3].clone()
        top_target[:, 2] += (
            0.5 * (self._box_lib._box_size[r0, 2]
                   + self._box_lib._box_size[r1, 2])
            + self.stack_top_clearance
        )

        def tilt(states):
            q = states[:, 3:7]
            upright = 1.0 - 2.0 * (q[:, 0].square() + q[:, 1].square())
            return torch.acos(upright.clamp(-1.0, 1.0))

        bottom_clear = self._hand_box_clearance(r0)
        top_clear = self._hand_box_clearance(r1)
        clearance = (roots[r0, :2] - bottom[:, :2]).norm(dim=-1)
        return StackPhysicalState(
            bottom[:, :3] - bottom_target,
            bottom[:, 7:10], bottom[:, 10:13], tilt(bottom),
            (bottom_clear >= self.stack_hand_clear_done).float(),
            (roots[r0, :2] - bottom[:, :2]).norm(dim=-1), clearance,
            top[:, :3] - top_target,
            top[:, 7:10], top[:, 10:13], tilt(top),
            (top_clear >= self.stack_hand_clear_done).float(),
            (roots[r1, :2] - top[:, :2]).norm(dim=-1),
            self._seq_top_reached.float(),
        )

    def planner_collision_cost(self):
        state = self.planner_state()
        root, box = state.root_xy, state.box_xyz[..., :2]
        radius = 0.5 * state.box_size_xy.norm(dim=-1)
        hh = F.relu(1.0 - (root[:, 0] - root[:, 1]).norm(dim=-1)).square()
        bb = F.relu(radius.sum(-1) + 0.15 - (box[:, 0] - box[:, 1]).norm(dim=-1)).square()
        hb01 = F.relu(0.35 + radius[:, 1] - (root[:, 0] - box[:, 1]).norm(dim=-1)).square()
        hb10 = F.relu(0.35 + radius[:, 0] - (root[:, 1] - box[:, 0]).norm(dim=-1)).square()
        return hh + bb + 0.5 * (hb01 + hb10)

    def planner_fall(self):
        bottom_lost = self.extras.get(
            "stack_bottom_displaced",
            torch.zeros(self._rows, dtype=torch.bool, device=self.device),
        ).reshape(self.num_envs, 2).any(dim=1)
        return self._terminate_buf.bool() & ~bottom_lost


__all__ = ["HumanoidMAStackPlannerTrain"]
