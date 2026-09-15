"""Runtime-only bridge from learned stack trajectories to frozen ms18 Carry.

The planner always observes the real simulator state.  Only this compatibility
boundary converts the learned A1 retreat endpoint into the virtual Carry box
needed by the currently coupled low-level policy.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from isaacgym import gymtorch

from coordinator.schema import MAX_SPEED, MIN_SPEED, PATH_DS, PATH_VERTICES, CoordinatorState
from coordinator.sequential_bridge import carry_rows
from env.tasks.adapt_interaction_skills.humanoid_ma_sequential_stack_carry import (
    HumanoidMASequentialStackCarry,
)
from stack_planner.reward import StackPhysicalState
from stack_planner.constraints import (
    free_path_validity, retreat_endpoint_change_cost, held_box_body_cost,
)
from stack_planner.execution import execution_view, retreat_box_geometry
from tokenhsi.utils import steer_path as sp


def _resample_unified_plan(path, speed):
    """Interpolate a selected variable-length plan onto the frozen steer ABI."""
    if path.shape != (*speed.shape, 2) or path.ndim != 4 or path.shape[1] != 2:
        raise ValueError("expected selected path [B,2,P,2] and speed [B,2,P]")
    points = path.shape[-2]
    if points < 2:
        raise ValueError("a plan needs at least two points")
    lengths = (path[..., 1:, :] - path[..., :-1, :]).norm(dim=-1)
    cumulative = torch.cat((
        torch.zeros_like(lengths[..., :1]), lengths.cumsum(dim=-1),
    ), dim=-1)
    end = cumulative[..., -1]
    query = torch.minimum(
        torch.arange(PATH_VERTICES, device=path.device) * PATH_DS,
        end[..., None],
    ).to(path.dtype)
    flat = cumulative.reshape(-1, points).contiguous()
    flat_query = query.reshape(-1, PATH_VERTICES).contiguous()
    lower = (torch.searchsorted(flat, flat_query, right=True) - 1).clamp(
        0, points - 2
    )
    ds = flat.gather(1, lower + 1) - flat.gather(1, lower)
    fraction = (
        (flat_query - flat.gather(1, lower)) / ds.clamp(min=1e-8)
    ).clamp(0, 1)
    values = torch.cat((path, speed[..., None]), dim=-1).reshape(-1, points, 3)
    row = torch.arange(values.shape[0], device=path.device)[:, None]
    dense = values[row, lower] + fraction[..., None] * (
        values[row, lower + 1] - values[row, lower]
    )
    dense = dense.reshape(path.shape[0], 2, PATH_VERTICES, 3)
    return dense[..., :2], dense[..., 2].clamp(MIN_SPEED, MAX_SPEED), end


class HumanoidMAStackPlannerTrain(HumanoidMASequentialStackCarry):
    """Sequential stack task with an externally installed stack-planner path."""

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._planner_fresh_start = bool(int(os.environ.get(
            "STACK_PLANNER_FRESH_START", "1"
        )))
        if self._planner_fresh_start:
            carry_cfg = cfg["env"]["carry"]
            skills = list(carry_cfg["skill"])
            if "loco_carry" not in skills:
                raise ValueError("fresh stack-planner reset requires loco_carry")
            fresh_prob = [float(name == "loco_carry") for name in skills]
            carry_cfg["skillInitProb"] = fresh_prob
            if "eval" in carry_cfg:
                carry_cfg["eval"]["skillInitProb"] = list(fresh_prob)
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        if sp.DS != PATH_DS or sp.V != PATH_VERTICES or self.steer_dim() != 12:
            raise ValueError("stack planner requires ms18's 0.1m/320-point/12-D steer ABI")
        if not self.steer_clip or self.steer_zero:
            raise ValueError("stack planner requires MS_CLIP=1 and MS_ZERO=0")
        self._planner_command_accel = float(os.environ.get(
            "STACK_PLANNER_COMMAND_ACCEL", "0.75"
        ))
        if self._planner_command_accel <= 0.0:
            raise ValueError("STACK_PLANNER_COMMAND_ACCEL must be positive")
        self._planner_virtual_retreat_pos = self._a1_retreat_pos.clone()
        # Readiness gates A2 while awaiting the first valid planner retreat;
        # it never blocks subsequent replans or fixes the retreat endpoint.
        self._planner_retreat_ready = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._planner_endpoint_history_valid = torch.zeros_like(self._planner_retreat_ready)
        self._planner_endpoint_change_cost = torch.zeros(self.num_envs, device=self.device)
        # Every active planner tick is a new decision, including retreat.
        self._planner_policy_decision = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._planner_retreat_box_path_penalty = torch.zeros(
            self.num_envs, device=self.device
        )
        self._planner_retreat_box_min_clearance = torch.zeros(
            self.num_envs, device=self.device
        )
        self._planner_applied = torch.zeros(self._rows, dtype=torch.bool, device=self.device)
        root = self.humanoid_rows(self._humanoid_root_states)
        self._planner_command_speed = root[:, 7:9].norm(dim=-1).clamp(
            MIN_SPEED, MAX_SPEED
        )
        self._planner_speed_tick = torch.full(
            (self._rows,), -1, dtype=torch.long, device=self.device
        )

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
        settled = (
            box[..., 7:10].norm(dim=-1) < self.stack_vel_tol
        ) & (
            box[..., 10:13].norm(dim=-1) < self.stack_ang_vel_tol
        )
        phase = torch.where(held, torch.full_like(bottom, 2.0), near.float())
        phase = torch.where(
            at_goal & settled & ~held, torch.full_like(phase, 3.0), phase
        )
        return CoordinatorState(
            root[..., :2], self._yaw(root), root[..., 7:9],
            box[..., :3], self._yaw(box), box[..., 7:9],
            size[..., :2], goal, held.float(), phase,
        )

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        if hasattr(self, "_planner_virtual_retreat_pos") and len(env_ids):
            self._planner_virtual_retreat_pos[env_ids] = self._a1_retreat_pos[env_ids]
            self._planner_retreat_ready[env_ids] = False
            self._planner_endpoint_history_valid[env_ids] = False
            self._planner_endpoint_change_cost[env_ids] = 0.0
            self._planner_policy_decision[env_ids] = True
            self._planner_retreat_box_path_penalty[env_ids] = 0.0
            self._planner_retreat_box_min_clearance[env_ids] = 0.0
            rows = self.agent_rows(env_ids)
            self._planner_applied[rows] = False
            root = self.humanoid_rows(self._humanoid_root_states)[rows]
            self._planner_command_speed[rows] = root[:, 7:9].norm(dim=-1).clamp(
                MIN_SPEED, MAX_SPEED
            )
            self._planner_speed_tick[rows] = -1
            # The parent built reset observations before the planner command
            # state above was reset. Rebuild only these environments so the
            # first post-reset action cannot see a stale speed command.
            self._compute_observations(env_ids)

    def _reset_env_tensors(self, env_ids):
        """Commit only actors that exist in this carry-only training task.

        The generic multi-task reset also submits disabled sit/climb actor IDs.
        This workspace intentionally lacks some of those URDF assets, so that
        broad commit can become an illegal CUDA actor access on episode reset.
        """
        humanoids = self._humanoid_actor_ids_per_env[env_ids].flatten()
        rows = self.agent_rows(env_ids)
        objects = self._box_actor_ids[rows]
        if self._carry_reset_random_height:
            objects = torch.cat((objects, self._platform_actor_ids[rows],
                                 self._tar_platform_actor_ids[rows]))
        # The authoritative Isaac Gym GPU root-state setter must contain the
        # complete actor set. A humanoid-only call followed by an object-only
        # call can leave just the latter pending, so the next episode inherits
        # old agent roots (or vice versa).
        root_actors = torch.cat((humanoids, objects))
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(root_actors), len(root_actors),
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(humanoids), len(humanoids),
        )
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        if getattr(self, "_enable_IET", False):
            self._IET_step_buf[rows] = 0
            self._IET_triggered_buf[rows] = 0
        if getattr(self, "_is_eval", False):
            self._success_buf[rows] = 0
            self._precision_buf[rows] = float("inf")

    def _post_object_reset(self, env_ids):
        super()._post_object_reset(env_ids)
        if len(env_ids) == 0:
            return
        # Sequential-stack post processing may move/hide a target platform
        # with another indexed root-state call. On the GPU pipeline the final
        # call must be the authoritative superset; otherwise that platform-only
        # call can leave humanoids and boxes at the preceding episode pose.
        humanoids = self._humanoid_actor_ids_per_env[env_ids].flatten()
        rows = self.agent_rows(env_ids)
        objects = self._box_actor_ids[rows]
        if self._carry_reset_random_height:
            objects = torch.cat((objects, self._platform_actor_ids[rows],
                                 self._tar_platform_actor_ids[rows]))
        root_actors = torch.cat((humanoids, objects))
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(root_actors), len(root_actors),
        )

    def _virtual_retreat_carry_obs(self, rows, env_ids):
        if not hasattr(self, "_planner_virtual_retreat_pos"):
            return super()._virtual_retreat_carry_obs(rows, env_ids)
        original = self._a1_retreat_pos[env_ids].clone()
        try:
            self._a1_retreat_pos[env_ids] = self._planner_virtual_retreat_pos[env_ids]
            obs = super()._virtual_retreat_carry_obs(rows, env_ids).clone()
            # The inherited bridge uses real box Z for its center, BPS and
            # goal. Correct all three in observation space only. Its inverse
            # heading transform is yaw-only, so a world Z shift is local Z.
            ground = float(os.environ.get("STACK_GROUND_Z", "0.0"))
            height = ground + 0.5 * self._box_lib._box_size[rows, 2]
            real_z = self.humanoid_rows(self._box_states)[rows, 2]
            dz = height - real_z
            # Carry layout: velocity(3), angular velocity(3), center(3),
            # rotation(6), BPS(3*N), placement goal(3).
            obs[:, 8] += dz
            bps = obs[:, 15:-3].reshape(len(rows), -1, 3)
            bps[..., 2] += dz[:, None]
            obs[:, -1] += dz
            self._planner_virtual_retreat_pos[env_ids, 2] = height
            return obs
        finally:
            self._a1_retreat_pos[env_ids] = original

    def _update_retreat_goal(self, env_ids):
        """Planner-only phase entry: no inherited outward retreat target."""
        if not len(env_ids):
            return
        rows = self.agent_rows(env_ids).reshape(-1, 2)[:, 0]
        root = self.humanoid_rows(self._humanoid_root_states)[rows, :3].clone()
        self._a1_retreat_start[env_ids] = root[:, :2]
        self._a1_retreat_dir[env_ids] = 0.0
        self._a1_retreat_pos[env_ids] = root
        if hasattr(self, "_planner_virtual_retreat_pos"):
            self._planner_retreat_ready[env_ids] = False
            self._planner_endpoint_history_valid[env_ids] = False

    def _activate_retreat_steer(self, env_ids):
        """Hold at phase-entry root until a learned retreat is accepted."""
        if not len(env_ids):
            return
        rows = self.agent_rows(env_ids).reshape(-1, 2)[:, 0]
        self._gt_path[rows] = self._a1_retreat_start[env_ids, None, :]
        self._s_end[rows] = 0.0
        self._arc_root[rows] = 0.0
        self._arc_box[rows] = 0.0
        self._prev_arc[rows] = 0.0
        self._mscale[rows] = 0.0
        self._prev_retreat_dist[env_ids] = 0.0

    def _planner_a2_ready_ids(self, env_ids):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if not hasattr(self, "_planner_retreat_ready"):
            return env_ids
        waiting = ((self._stack_phase[env_ids] == self.A1_RETREAT)
                   & ~self._carry_rehearsal[env_ids]
                   & ~self._planner_retreat_ready[env_ids])
        return env_ids[~waiting]

    def _activate_a2_top_goal(self, env_ids):
        # Parent completion sees zero distance to a hold target. Prevent its
        # side effects as well as the phase transition until a plan is accepted.
        super()._activate_a2_top_goal(self._planner_a2_ready_ids(env_ids))

    def _set_phase(self, env_ids, phase):
        if phase == self.A2_RESUME:
            env_ids = self._planner_a2_ready_ids(env_ids)
        super()._set_phase(env_ids, phase)

    @torch.no_grad()
    def install_external_plan(self, output):
        """Install candidate zero; return per-env (valid, conservative-safe)."""
        state = self.planner_state()
        phase = self._stack_phase
        active = carry_rows(phase, self._carry_rehearsal)
        retreat_env = (phase == self.A1_RETREAT) & ~self._carry_rehearsal
        # Retain the accepted virtual endpoint after A2 starts. Reset or a new
        # retreat entry clears readiness, not simply leaving phase 2.
        retreat_rows = torch.zeros_like(active)
        retreat_rows[:, 0] = retreat_env
        model_path = output["path_world"][:, 0]
        self._planner_endpoint_change_cost = retreat_endpoint_change_cost(
            model_path[:, 0, -1], self._planner_virtual_retreat_pos[:, :2],
            retreat_env & self._planner_endpoint_history_valid,
            float(os.environ.get("STACK_PLANNER_RETREAT_ENDPOINT_TOLERANCE", "0.10")),
        )
        self._planner_endpoint_history_valid.copy_(retreat_env)
        # Always use the raw world endpoint, without validity/readiness gates.
        self._planner_virtual_retreat_pos[:, :2] = model_path[:, 0, -1]
        a1_rows = self.all_rows().reshape(self.num_envs, 2)[:, 0]
        self._planner_virtual_retreat_pos[:, 2] = (
            float(os.environ.get("STACK_GROUND_Z", "0.0"))
            + 0.5 * self._box_lib._box_size[a1_rows, 2]
        )
        path = execution_view(
            model_path, state.box_xyz[..., :2], state.goal_xy, retreat_rows
        )
        speed = torch.full(
            path.shape[:-1], MAX_SPEED, device=path.device, dtype=path.dtype,
        )
        model_speed = torch.full_like(output["path_world"][..., 0], MAX_SPEED)
        valid = free_path_validity(
            output["path_world"], model_speed, state.root_xy, active
        )[:, 0]
        self._planner_policy_decision = active.any(dim=-1)

        # Inactive roots are stationary for the safety projection.
        projected = torch.where(
            active[..., None, None], path,
            state.root_xy[..., None, :].expand_as(path),
        )
        separation = (projected[:, 0] - projected[:, 1]).norm(dim=-1)
        safe = separation.amin(dim=-1) >= 0.8

        install = valid[:, None] & active
        endpoint = path[:, 0, -1]
        commit_retreat = retreat_env & install[:, 0]
        self._planner_retreat_box_path_penalty.zero_()
        self._planner_retreat_box_min_clearance.zero_()
        if commit_retreat.any():
            geometry = retreat_box_geometry(path, state, commit_retreat)
            self._planner_retreat_box_path_penalty[commit_retreat] = geometry[
                "penalty"
            ]
            self._planner_retreat_box_min_clearance[commit_retreat] = geometry[
                "minimum_clearance"
            ]
            # The inherited phase-2 completion test must judge the exact same
            # learned endpoint that the frozen executor sees, not the legacy
            # manually generated retreat goal/direction.
            # Keep the phase-entry progress origin even as targets replan.
            root = self._a1_retreat_start[commit_retreat]
            delta = endpoint[commit_retreat] - root
            distance = delta.norm(dim=-1, keepdim=True)
            old_direction = self._a1_retreat_dir[commit_retreat]
            direction = torch.where(
                distance > 1e-4, delta / distance.clamp(min=1e-4), old_direction
            )
            self._a1_retreat_dir[commit_retreat] = direction
            self._a1_retreat_pos[commit_retreat, :2] = endpoint[commit_retreat]
            self._planner_retreat_ready[commit_retreat] = True
        # Invalid candidates leave the execution path untouched. The virtual
        # box still follows the latest raw planner endpoint, with no fallback.
        if install.any():
            dense, dense_speed, end = _resample_unified_plan(path, speed)
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
        self._planner_plan_installed = install.any(dim=-1)
        return valid, safe

    def _planner_update_speed(self, rows):
        """Rate-limit command changes across independently decoded replans."""
        if not hasattr(self, "_planner_command_speed") or not len(rows):
            return
        tick = self.progress_rows()[rows]
        selected = self._planner_applied[rows] & (
            tick != self._planner_speed_tick[rows]
        )
        rows, tick = rows[selected], tick[selected]
        if not len(rows):
            return
        j = (self._arc_root[rows] / sp.DS).long().clamp(0, sp.V - 1)
        desired = MAX_SPEED * self._mscale[rows, j]
        max_delta = self._planner_command_accel * self.dt
        delta = (desired - self._planner_command_speed[rows]).clamp(
            -max_delta, max_delta
        )
        self._planner_command_speed[rows] = (
            self._planner_command_speed[rows] + delta
        ).clamp(MIN_SPEED, MAX_SPEED)
        self._planner_speed_tick[rows] = tick

    def _m_at(self, arc, rows=None):
        nominal = super()._m_at(arc, rows)
        if not hasattr(self, "_planner_command_speed"):
            return nominal
        rows = (
            torch.arange(len(arc), device=self.device)
            if rows is None else rows
        )
        speed_to_window = self.steer_m_nom / MAX_SPEED
        planned = speed_to_window * self._planner_command_speed[rows]
        command = torch.where(self._planner_applied[rows], planned, nominal)
        env_ids = torch.div(rows, self.num_agents, rounding_mode="floor")
        waiting = ((rows % self.num_agents == 0)
                   & (self._stack_phase[env_ids] == self.A1_RETREAT)
                   & ~self._carry_rehearsal[env_ids]
                   & ~self._planner_retreat_ready[env_ids])
        return torch.where(waiting, torch.zeros_like(command), command)

    def _compute_task_obs(self, env_ids=None):
        if hasattr(self, "_planner_command_speed"):
            rows = self.all_rows() if env_ids is None else self.agent_rows(env_ids)
            self._planner_update_speed(rows)
        return super()._compute_task_obs(env_ids)

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
        # Root-distance proxies miss a carried box striking the other torso,
        # legs or arms. Add a 3D oriented-box cost using real rigid-body state.
        rigid = self.humanoid_rows(self._rigid_body_pos).reshape(self.num_envs, 2, -1, 3)
        keep = torch.ones(rigid.shape[2], dtype=torch.bool, device=self.device)
        keep[self._key_body_ids[[0, 1]]] = False
        sizes = self._box_lib._box_size.reshape(self.num_envs, 2, 3)
        body01 = held_box_body_cost(rigid[:, 0, keep], state.box_xyz[:, 1],
                                   state.box_heading[:, 1], sizes[:, 1], state.held[:, 1])
        body10 = held_box_body_cost(rigid[:, 1, keep], state.box_xyz[:, 0],
                                   state.box_heading[:, 0], sizes[:, 0], state.held[:, 0])
        return hh + bb + 0.5 * (hb01 + hb10) + 10.0 * (body01 + body10)

    def planner_fall(self):
        bottom_lost = self.extras.get(
            "stack_bottom_displaced",
            torch.zeros(self._rows, dtype=torch.bool, device=self.device),
        ).reshape(self.num_envs, 2).any(dim=1)
        return self._terminate_buf.bool() & ~bottom_lost


__all__ = ["HumanoidMAStackPlannerTrain"]
