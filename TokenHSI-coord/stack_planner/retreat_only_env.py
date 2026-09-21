"""Minimal post-placement A1-retreat task for planner validation.

This task deliberately removes carry, stacking and A2 coordination from the
learning question.  Every episode starts with Box1 resting on its target, A1
standing beside it after release, and A2 holding position well outside the
workspace.  The planner owns only A1's complete retreat path.
"""

from __future__ import annotations

import os

from isaacgym import gymtorch
import torch
import torch.nn.functional as F

from .env_adapter import HumanoidMAStackPlannerTrain
from .schema import STACK_PATH_POINTS
from .view_env import HumanoidMAStackPlannerView


class _RetreatOnlyMixin:
    """Environment overrides shared by headless training and visual view."""

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id,
                 headless):
        self._planner_retreat_only_clearance = float(os.environ.get(
            "STACK_RETREAT_ONLY_CLEARANCE", "1.5",
        ))
        self._planner_retreat_only_a2_distance = float(os.environ.get(
            "STACK_RETREAT_ONLY_A2_DISTANCE", "3.0",
        ))
        if self._planner_retreat_only_clearance <= 0.0:
            raise ValueError("STACK_RETREAT_ONLY_CLEARANCE must be positive")
        if self._planner_retreat_only_a2_distance <= 2.0:
            raise ValueError("STACK_RETREAT_ONLY_A2_DISTANCE must exceed 2m")

        # The construction reset happens before stack curriculum buffers
        # exist.  Start both rows from a valid carry reference; the scene is
        # converted to the post-release diagnostic immediately afterwards.
        carry_cfg = cfg["env"]["carry"]
        skills = list(carry_cfg["skill"])
        if "carryWith" not in skills:
            raise ValueError("retreat-only reset requires carryWith")
        carry_prob = [float(name == "carryWith") for name in skills]
        carry_cfg["skillInitProb"] = carry_prob
        if "eval" in carry_cfg:
            carry_cfg["eval"]["skillInitProb"] = list(carry_prob)
        os.environ["STACK_PLANNER_FRESH_START"] = "0"

        super().__init__(
            cfg, sim_params, physics_engine, device_type, device_id, headless,
        )
        self._planner_retreat_only_ready = True
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._force_retreat_only_scene(env_ids)
        self._compute_observations()
        print(
            "[stack-retreat-only] post-place reset active; "
            "A1_path_only=True A2_stationary=True route_visit=False "
            "clearance={:.2f}m".format(self._planner_retreat_only_clearance),
            flush=True,
        )

    def _reset_task_indicator(self, env_ids):
        super()._reset_task_indicator(env_ids)
        if not hasattr(self, "_curriculum_task") or len(env_ids) == 0:
            return
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._curriculum_task[env_ids] = self.TASK_RETREAT
        if self._carry_rehearsal is not None:
            self._carry_rehearsal[env_ids] = False

    def _post_object_reset(self, env_ids):
        super()._post_object_reset(env_ids)
        if (getattr(self, "_planner_retreat_only_ready", False)
                and len(env_ids)):
            self._force_retreat_only_scene(env_ids)

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        if (getattr(self, "_planner_retreat_only_ready", False)
                and len(env_ids)):
            # The generic planner reset clears this latch after object reset;
            # here Box1 is already the placed support from frame zero.
            self._planner_bottom_stable_seen[env_ids] = True

    @torch.no_grad()
    def _force_retreat_only_scene(self, env_ids):
        """Create one valid, stationary post-release scene and commit it."""
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if len(env_ids) == 0:
            return
        rows = self.agent_rows(env_ids).view(-1, 2)
        r0, r1 = rows[:, 0], rows[:, 1]
        roots = self.humanoid_rows(self._humanoid_root_states)
        boxes = self.humanoid_rows(self._box_states)
        sizes = self._box_lib._box_size

        bottom = self._box_tar_pos[r0].clone()
        bottom[:, 2] = (
            float(os.environ.get("STACK_GROUND_Z", "0.0"))
            + 0.5 * sizes[r0, 2]
        )
        outward = roots[r0, :2] - boxes[r0, :2]
        fallback = torch.zeros_like(outward)
        fallback[:, 0] = -1.0
        norm = outward.norm(dim=-1, keepdim=True)
        outward = torch.where(
            norm > 1e-3, outward / norm.clamp(min=1e-4), fallback,
        )

        boxes[r0, :3] = bottom
        boxes[r0, 7:13] = 0.0
        roots[r0, :2] = bottom[:, :2] + 0.45 * outward
        roots[r0, 7:13] = 0.0

        # Preserve A2's reference hand/box relation but translate both far
        # outside A1's workspace. Its carry and steer goals remain its current
        # box position, so it only supplies a realistic stationary scene token.
        a2_target_xy = bottom[:, :2] - (
            self._planner_retreat_only_a2_distance * outward
        )
        shift = a2_target_xy - boxes[r1, :2]
        roots[r1, :2] += shift
        boxes[r1, :2] += shift
        roots[r1, 7:13] = 0.0
        boxes[r1, 7:13] = 0.0
        a2_hold = boxes[r1, :3].clone()

        self._bottom_nominal_pos[env_ids] = bottom
        self._committed_bottom_pos[env_ids] = bottom
        self._box_tar_pos[r0] = bottom
        self._a2_stage_pos[env_ids] = a2_hold
        self._box_tar_pos[r1] = a2_hold
        self._committed_top_pos[env_ids] = a2_hold
        self._top_committed[env_ids] = True
        self._bottom_stable_count[env_ids] = self.stack_stable_steps
        self._planner_bottom_stable_seen[env_ids] = True
        self._top_stable_count[env_ids] = 0
        self._stack_phase[env_ids] = self.A1_RETREAT
        self._stack_phase_age[env_ids] = 0
        self._seq_top_reached[env_ids] = False
        self._carry_rehearsal[env_ids] = False
        self._ever_held[r0] = True
        self._held[r0] = False
        rr = rows.reshape(-1)
        self._initial_box_bottom_z[rr] = (
            boxes[rr, 2] - 0.5 * sizes[rr, 2]
        )

        self._update_retreat_goal(env_ids)
        self._activate_retreat_steer(env_ids)
        self._reset_steer_to(r1, a2_hold)

        actor_ids = torch.cat((
            self._humanoid_actor_ids_per_env[env_ids].reshape(-1),
            self._box_actor_ids[rr],
        ))
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(actor_ids), len(actor_ids),
        )
        humanoid_ids = self._humanoid_actor_ids_per_env[env_ids].reshape(-1)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(humanoid_ids), len(humanoid_ids),
        )
        self._refresh_sim_tensors()

    def planner_active_rows(self):
        active = torch.zeros(
            self.num_envs, 2, dtype=torch.bool, device=self.device,
        )
        active[:, 0] = self._stack_phase == self.A1_RETREAT
        return active

    def planner_execution_rows(self):
        return self.planner_active_rows()

    def planner_route_visit_enabled(self):
        return False

    def planner_remove_retreat_potential(self):
        # In the isolated task, direction-free clearance progress is the task
        # objective. It saturates at the success radius and does not prescribe
        # a hand-authored retreat direction.
        return False

    def _planner_execution_view(self, model_path, state, retreat_rows,
                                model_speed):
        # No carry prefix exists in this diagnostic. The model's complete A1
        # path starts at the live root and is installed directly.
        path = model_path.clone()
        speed = model_speed[:, 0].clone()
        path[:, 1] = state.root_xy[:, 1, None, :]
        speed[:, 1] = 0.0
        return path, speed

    def planner_physical_state(self):
        state = super().planner_physical_state()
        planner_state = self.planner_state()
        clearance = (
            planner_state.root_xy[:, 0] - planner_state.box_xyz[:, 0, :2]
        ).norm(dim=-1)
        state.bottom_root_distance = clearance
        state.clearance = clearance
        state.stack_success = (
            clearance >= self._planner_retreat_only_clearance
        ).to(state.stack_success)
        return state

    def planner_collision_terms(self):
        """Measure the isolated failure: A1 entering its placed Box1."""
        state = self.planner_state()
        distance = (
            state.root_xy[:, 0] - state.box_xyz[:, 0, :2]
        ).norm(dim=-1)
        radius = 0.5 * state.box_size_xy[:, 0].norm(dim=-1)
        own_box = F.relu(0.35 + radius - distance).square()
        zero = torch.zeros_like(own_box)
        return {
            "agent_agent": zero,
            "agent_box": own_box,
            "held_box_body": zero,
            "total": own_box,
        }

    def _a1_fallen(self):
        contact = self.humanoid_rows(self._contact_forces).clone()
        contact[:, self._contact_body_ids, :] = 0
        fall_contact = torch.any(torch.abs(contact) > 0.1, dim=-1).any(dim=-1)
        body_height = self.humanoid_rows(self._rigid_body_pos)[..., 2]
        fall_height = body_height < self._termination_heights
        fall_height[:, self._contact_body_ids] = False
        fall_height = fall_height.any(dim=-1)
        fallen = (
            fall_contact & fall_height & (self.progress_rows() > 1)
        ).view(self.num_envs, 2)
        return fallen[:, 0]

    def _compute_reset(self):
        super()._compute_reset()
        if not getattr(self, "_planner_retreat_only_ready", False):
            return
        # A2 is outside the task and must not end A1's rollout. Keep only the
        # ordinary horizon and A1's own physical fall; bottom displacement is
        # added by the inherited sequential post-step check afterwards.
        a1_fall = self._a1_fallen()
        timeout = self.progress_buf >= self.max_episode_length - 1
        self._terminate_buf.copy_(a1_fall)
        self.reset_buf.copy_(timeout | a1_fall)

    def planner_fall(self):
        return self._a1_fallen()

    def _update_stack_coordinator(self):
        """Keep A2 closed and end only on A1's measured clearance."""
        if not hasattr(self, "_stack_phase"):
            return
        self._stack_phase_age += 1
        rows = self.all_rows().view(self.num_envs, 2)
        roots = self.humanoid_rows(self._humanoid_root_states)
        boxes = self.humanoid_rows(self._box_states)
        clearance = (roots[rows[:, 0], :2] - boxes[rows[:, 0], :2]).norm(
            dim=-1,
        )
        success = (
            (self._stack_phase == self.A1_RETREAT)
            & (clearance >= self._planner_retreat_only_clearance)
        )
        ids = torch.nonzero(success, as_tuple=False).squeeze(-1)
        if len(ids):
            self._set_phase(ids, self.DONE)
            self.reset_buf[ids] = 1


class HumanoidMAStackPlannerRetreatTrain(
    _RetreatOnlyMixin, HumanoidMAStackPlannerTrain,
):
    pass


class HumanoidMAStackPlannerRetreatView(
    _RetreatOnlyMixin, HumanoidMAStackPlannerView,
):
    pass


__all__ = [
    "HumanoidMAStackPlannerRetreatTrain",
    "HumanoidMAStackPlannerRetreatView",
]
