"""C1 state-only coordinator in front of the frozen ms18 executor.

The inherited action policy contract is untouched.  This wrapper only replaces
the provider of ``_gt_path``, ``_mscale`` and ``_s_end`` with a selected learned
joint future and rate-limits the speed encoded by the six-point steer window.
"""

from __future__ import annotations

import atexit
import os
from pathlib import Path

import numpy as np
import torch

from coordinator.checkpoint import load_checkpoint
from coordinator.c2_checkpoint import C2_CANDIDATES, load_c2_checkpoint
from coordinator.simple_checkpoint import load_simple_checkpoint
from coordinator.geometry import resample_path_and_speed
from coordinator.planner import apply_fixed_priority, select_candidate
from coordinator.schema import (
    AGENTS,
    CANDIDATES,
    MAX_SPEED,
    MIN_SPEED,
    PATH_DS,
    PATH_POINTS,
    PATH_VERTICES,
    CoordinatorState,
)
from env.tasks.adapt_interaction_skills.humanoid_ma_steer_carry import HumanoidMASteerCarry
from tokenhsi.utils import steer_path as sp


class HumanoidMACoordCarry(HumanoidMASteerCarry):
    """A=2 Carry task driven by a joint high-level coordinator."""

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._coord_ready = False
        # Picking up a low box necessarily puts the hands close to the ground.
        # Match the sequential carry executor: hand contact is valid manipulation,
        # while torso/head/pelvis/knee contacts still trigger fall termination.
        self._coord_allow_hand_contact = bool(int(os.environ.get(
            "COORD_ALLOW_HAND_CONTACT", "1")))
        if self._coord_allow_hand_contact:
            contact_bodies = list(cfg["env"].get("contactBodies", []))
            for body_name in ("right_hand", "left_hand"):
                if body_name not in contact_bodies:
                    contact_bodies.append(body_name)
            cfg["env"]["contactBodies"] = contact_bodies
        # Viewer-only diagnostic layout.  Ordinary training/evaluation never
        # enables this: it intentionally changes reset geometry so that two
        # straight, nominal-speed root->box->goal timelines meet together.
        self._view_timed_cross = bool(int(
            os.environ.get("MS_VIEW_TIMED_CROSS", "0")
        ))
        self._view_timed_cross_prob = float(
            os.environ.get("MS_VIEW_TIMED_CROSS_PROB", "1.0")
        )
        self._view_timed_cross_tol = float(
            os.environ.get("MS_VIEW_TIMED_CROSS_TOL", "0.25")
        )
        self._view_timed_cross_pre = float(
            os.environ.get("MS_VIEW_TIMED_CROSS_PRE", "2.0")
        )
        self._view_timed_cross_post = float(
            os.environ.get("MS_VIEW_TIMED_CROSS_POST", "4.0")
        )
        self._view_timed_cross_max_shift = float(
            os.environ.get("MS_VIEW_TIMED_CROSS_MAX_SHIFT", "8.0")
        )
        self._view_timed_cross_reset = 0
        if self._view_timed_cross:
            if not bool(int(os.environ.get("COORD_VIEWER", "0"))):
                raise ValueError(
                    "MS_VIEW_TIMED_CROSS is viewer-only; use scripts/coord/view_local.sh"
                )
            if os.environ.get("MS_SCEN", "free") != "cross":
                raise ValueError("MS_VIEW_TIMED_CROSS requires MS_SCEN=cross")
            if not 0.0 <= self._view_timed_cross_prob <= 1.0:
                raise ValueError("MS_VIEW_TIMED_CROSS_PROB must be in [0, 1]")
            if min(
                self._view_timed_cross_tol,
                self._view_timed_cross_pre,
                self._view_timed_cross_post,
                self._view_timed_cross_max_shift,
            ) < 0.0:
                raise ValueError("timed-cross distances/tolerance must be non-negative")
        self._coord_provider = os.environ.get("COORD_PROVIDER", "learned").lower()
        self._coord_model_kind = os.environ.get("COORD_MODEL", "c1").lower()
        self._coord_random_priority_explicit = (
            "COORD_C2_RANDOM_PRIORITY" in os.environ
        )
        self._coord_learned_priority_explicit = (
            "COORD_C2_LEARNED_PRIORITY" in os.environ
        )
        self._coord_random_priority = bool(int(
            os.environ.get("COORD_C2_RANDOM_PRIORITY", "0")
        ))
        self._coord_learned_priority = bool(int(
            os.environ.get("COORD_C2_LEARNED_PRIORITY", "0")
        ))
        if self._coord_random_priority and self._coord_learned_priority:
            raise ValueError("random_priority and learned_priority are exclusive")
        if self._coord_model_kind not in ("c1", "c2", "simple"):
            raise ValueError("COORD_MODEL must be c1, c2, or simple")
        if self._coord_random_priority and self._coord_model_kind != "c2":
            raise ValueError("COORD_C2_RANDOM_PRIORITY requires COORD_MODEL=c2")
        self._coord_candidates_k = C2_CANDIDATES if self._coord_model_kind == "c2" else CANDIDATES
        if self._coord_provider not in ("learned", "analytic", "external"):
            raise ValueError("COORD_PROVIDER must be learned, analytic, or external")
        self._coord_replan_steps = int(os.environ.get("COORD_REPLAN_STEPS", "6"))
        if self._coord_replan_steps <= 0:
            raise ValueError("COORD_REPLAN_STEPS must be positive")
        self._coord_accel_up = float(os.environ.get("COORD_ACCEL_UP", "0.75"))
        self._coord_accel_down = float(os.environ.get("COORD_ACCEL_DOWN", "1.0"))
        self._coord_speed_limit = bool(int(os.environ.get("COORD_SPEED_LIMIT", "1")))
        self._coord_preserve_pickup_approach = bool(int(os.environ.get(
            "COORD_PRESERVE_PICKUP_APPROACH", "1")))
        self._coord_grasp_stable_steps = int(os.environ.get(
            "COORD_GRASP_STABLE_STEPS", "12"))
        if self._coord_grasp_stable_steps < 1:
            raise ValueError("COORD_GRASP_STABLE_STEPS must be positive")
        self._coord_draw_candidates = bool(int(os.environ.get("COORD_DRAW_CANDIDATES", "1")))
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        if self.num_agents != AGENTS:
            raise ValueError(f"HumanoidMACoordCarry requires exactly {AGENTS} agents")
        if sp.DS != PATH_DS or sp.V != PATH_VERTICES:
            raise ValueError(
                f"ms18 path contract mismatch: expected DS/V={PATH_DS}/{PATH_VERTICES}, "
                f"got {sp.DS}/{sp.V}"
            )

        self._coordinator = None
        self._coord_checkpoint = None
        if self._coord_provider == "learned":
            raw = os.environ.get("COORD_CKPT")
            if not raw:
                raise ValueError("COORD_CKPT is required for COORD_PROVIDER=learned")
            checkpoint = self._resolve_checkpoint(raw)
            if self._coord_model_kind == "simple":
                loader = load_simple_checkpoint
            elif self._coord_model_kind == "c2":
                loader = load_c2_checkpoint
            else:
                loader = load_checkpoint
            self._coordinator, self._coord_checkpoint = loader(checkpoint, self.device)
            self._coordinator.eval()
            checkpoint_priority = self._coord_checkpoint.get(
                "extras", {}
            ).get("random_priority")
            if checkpoint_priority is not None:
                checkpoint_priority = bool(checkpoint_priority)
                if (
                    self._coord_random_priority_explicit
                    and self._coord_random_priority != checkpoint_priority
                ):
                    raise ValueError(
                        "COORD_C2_RANDOM_PRIORITY disagrees with checkpoint"
                    )
                self._coord_random_priority = checkpoint_priority
            checkpoint_learned = bool(
                getattr(self._coordinator.config, "learned_priority", False)
            )
            if (
                self._coord_learned_priority_explicit
                and self._coord_learned_priority != checkpoint_learned
            ):
                raise ValueError(
                    "COORD_C2_LEARNED_PRIORITY disagrees with checkpoint"
                )
            self._coord_learned_priority = checkpoint_learned
            if self._coord_random_priority and self._coord_learned_priority:
                raise ValueError(
                    "checkpoint enables both random and learned priority"
                )
            print(
                f"[coord] loaded {checkpoint} model={self._coord_model_kind} K={self._coord_candidates_k} "
                f"replan={self._coord_replan_steps} action steps "
                f"allow_hand_contact={self._coord_allow_hand_contact} "
                f"preserve_pickup_approach={self._coord_preserve_pickup_approach}",
                flush=True,
            )
        else:
            print(f"[coord] provider={self._coord_provider} replan={self._coord_replan_steps}", flush=True)

        self._coord_phase = torch.zeros(self.num_envs, AGENTS, device=self.device)
        self._coord_held_streak = torch.zeros(
            self.num_envs, AGENTS, device=self.device, dtype=torch.long
        )
        self._coord_held_tick = torch.full(
            (self.num_envs,), -1, device=self.device, dtype=torch.long
        )
        self._coord_last_replan = torch.full(
            (self.num_envs,), -self._coord_replan_steps, device=self.device, dtype=torch.long
        )
        self._coord_has_valid = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._coord_replans = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._coord_invalid = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._coord_unsafe = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._coord_fallback = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # Replan-level failed predicates: finite / anchors / buffer / speed / curve.
        self._coord_invalid_reasons = torch.zeros(5, device=self.device, dtype=torch.long)
        # Curve failures split by how many agents already hold their box.
        # This distinguishes tiny pre-grasp approach zigzags from carry-path
        # failures without changing planning or reward behavior.
        self._coord_curve_invalid_by_held = torch.zeros(
            3, device=self.device, dtype=torch.long
        )
        self._coord_selected = torch.full(
            (self.num_envs,), -1, device=self.device, dtype=torch.long
        )
        self._coord_priority_agent = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self._coord_priority_locked = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self._coord_choice_counts = torch.zeros(
            self._coord_candidates_k, device=self.device, dtype=torch.long
        )
        # diversity, selected min HH, BB margin, HB margin, makespan, collision
        self._coord_diag_sum = torch.zeros(6, device=self.device)
        self._coord_diag_count = 0
        self._coord_candidates = torch.zeros(
            self.num_envs, self._coord_candidates_k, AGENTS, PATH_POINTS, 2, device=self.device
        )
        self._coord_cmd_speed = torch.full(
            (self._rows,), MIN_SPEED, device=self.device
        )
        self._coord_cmd_tick = torch.full(
            (self._rows,), -1, device=self.device, dtype=torch.long
        )
        self._coord_limited_lookup = False
        # Directly measure the rate-limited command actually sent to ms18.
        # Parent speed bins re-read the raw path profile after the limiter scope.
        self._coord_ep_collision_phase = torch.zeros(
            self._rows, 2, device=self.device
        )
        self._coord_ep_sent_vbin_n = torch.zeros(
            self._rows, 4, device=self.device
        )
        self._coord_ep_sent_low_phase = torch.zeros(
            self._rows, 2, device=self.device
        )
        # Per-step pair modes, duplicated onto both agent rows so they follow
        # the existing episode dump/reset lifecycle.  Columns are
        # [neither slow, exactly one slow, both slow].
        self._coord_ep_pair_sent_mode = torch.zeros(
            self._rows, 3, device=self.device
        )
        self._coord_ep_pair_collision_mode = torch.zeros(
            self._rows, 3, device=self.device
        )
        self._coord_ready = True
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._sample_coord_priority(env_ids)
        state = self._coord_state(env_ids)
        self._coord_phase[:] = state.phase
        self._set_speed_anchor(self.agent_rows(env_ids))
        self._plan_envs(env_ids)
        atexit.register(self._report_coord_runtime)

    def apply_layout(self, env_ids):
        """Apply the normal layout, then optionally synchronize Cross timing.

        This hook is deliberately guarded by both ``COORD_VIEWER`` and
        ``MS_VIEW_TIMED_CROSS``.  It cannot alter existing training or batch
        evaluation.  The whole humanoid/box/platform package is translated;
        moving only the box would break reference-motion grasp geometry.
        """
        super().apply_layout(env_ids)
        if self._view_timed_cross and len(env_ids) > 0:
            self._apply_view_timed_cross(env_ids)

    def _view_cross_arrival(self, env_ids, crossing):
        """Estimate nominal ms18 arrival at a common crossing in seconds."""
        h = self.agent_axis(self._humanoid_root_states)[env_ids]
        b = self.agent_axis(self._box_states)[env_ids]
        rows = self.agent_rows(env_ids)
        half_height = self._box_lib._box_size[rows, 2].reshape(-1, AGENTS) * 0.5
        held = b[..., 2] > half_height + 0.2
        root_near = (h[..., :2] - b[..., :2]).norm(dim=-1) <= 0.7

        # Frozen-ms18 system-ID values at the 1.5 m/s command.  They are
        # overrides rather than a checkpoint dependency because this feature
        # only constructs a diagnostic viewer reset.
        approach_speed = float(os.environ.get(
            "MS_VIEW_CROSS_APPROACH_SPEED", "1.401"
        ))
        carry_speed = float(os.environ.get(
            "MS_VIEW_CROSS_CARRY_SPEED", "1.310"
        ))
        dwell_far = float(os.environ.get("MS_VIEW_CROSS_DWELL", "0.900"))
        dwell_near = float(os.environ.get(
            "MS_VIEW_CROSS_DWELL_NEAR", "0.533"
        ))
        if approach_speed <= 0.0 or carry_speed <= 0.0:
            raise ValueError("timed-cross measured speeds must be positive")

        root_box = (h[..., :2] - b[..., :2]).norm(dim=-1)
        box_cross = (b[..., :2] - crossing[:, None]).norm(dim=-1)
        root_cross = (h[..., :2] - crossing[:, None]).norm(dim=-1)
        dwell = torch.where(
            root_near,
            torch.full_like(root_box, dwell_near),
            torch.full_like(root_box, dwell_far),
        )
        approach_time = root_box / approach_speed + dwell
        arrival = approach_time + box_cross / carry_speed
        return torch.where(held, root_cross / carry_speed, arrival), carry_speed

    def _shift_view_cross_package(self, env_ids, agent, shift):
        """Move a complete agent package away from the shared crossing."""
        if not bool((shift > 0.0).any()):
            return
        h = self.agent_axis(self._humanoid_root_states)
        b = self.agent_axis(self._box_states)
        initial = self.agent_axis(self._initial_humanoid_root_states)
        crossing = initial[env_ids, :, :2].mean(dim=1)
        away = b[env_ids, agent, :2] - crossing
        fallback = torch.zeros_like(away)
        fallback[:, agent] = -1.0
        norm = away.norm(dim=-1, keepdim=True)
        away = torch.where(norm > 1e-5, away / norm.clamp(min=1e-5), fallback)
        delta = away * shift[:, None]
        h[env_ids, agent, :2] += delta
        b[env_ids, agent, :2] += delta
        if self._carry_reset_random_height:
            self.agent_axis(self._platform_states)[env_ids, agent, :2] += delta

    def _apply_view_timed_cross(self, env_ids):
        """Make nominal straight/full-speed crossing arrivals nearly equal."""
        # A private CPU generator avoids perturbing simulator/policy RNG.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            int(os.environ.get("MS_SEED", "0"))
            + 104729 * self._view_timed_cross_reset
        )
        self._view_timed_cross_reset += 1
        chosen_cpu = torch.rand(len(env_ids), generator=generator)
        chosen = chosen_cpu < self._view_timed_cross_prob
        if not bool(chosen.any()):
            return
        ids = env_ids[chosen.to(env_ids.device)]

        initial = self.agent_axis(self._initial_humanoid_root_states)
        crossing = initial[ids, :, :2].mean(dim=1)
        boxes = self.agent_axis(self._box_states)

        # Keep every box on the approach side and at least PRE metres from the
        # common crossing.  The original root-box relative pose is preserved.
        for agent in range(AGENTS):
            distance = (boxes[ids, agent, :2] - crossing).norm(dim=-1)
            self._shift_view_cross_package(
                ids, agent, (self._view_timed_cross_pre - distance).clamp(min=0.0)
            )

        total_shift = torch.zeros(len(ids), device=ids.device)
        # Align by sliding whichever agent is estimated to arrive first away
        # from the crossing.  Repeating also corrects held-state geometry where
        # root and box are close but not exactly coincident.
        for _ in range(3):
            arrival, carry_speed = self._view_cross_arrival(ids, crossing)
            earlier = arrival.argmin(dim=1)
            correction = arrival.diff(dim=1).abs().squeeze(1) * carry_speed
            correction = torch.minimum(
                correction,
                (self._view_timed_cross_max_shift - total_shift).clamp(min=0.0),
            )
            for agent in range(AGENTS):
                self._shift_view_cross_package(
                    ids, agent, correction * (earlier == agent).to(correction.dtype)
                )
            total_shift += correction

        # Make both carry legs pass through the exact same point, then continue
        # POST metres beyond it.  This is still a root->box->goal state input;
        # no trajectory or scenario feature is supplied to the coordinator.
        rows = self.agent_rows(ids)
        targets = torch.empty(len(ids), AGENTS, 2, device=ids.device)
        for agent in range(AGENTS):
            toward = crossing - boxes[ids, agent, :2]
            toward = toward / toward.norm(dim=-1, keepdim=True).clamp(min=1e-5)
            targets[:, agent] = crossing + toward * self._view_timed_cross_post
        self._box_tar_pos[rows, :2] = targets.reshape(-1, 2)
        if self._carry_reset_random_height:
            self.agent_axis(self._tar_platform_states)[ids, :, :2] = targets

        arrival, _ = self._view_cross_arrival(ids, crossing)
        gap = arrival.diff(dim=1).abs().squeeze(1)
        status = "OK" if bool((gap <= self._view_timed_cross_tol).all()) else "WARN"
        print(
            f"[coord-view timed-cross] {status} envs={len(ids)}/{len(env_ids)} "
            f"arrival_gap median={gap.median().item():.3f}s "
            f"max={gap.max().item():.3f}s tol={self._view_timed_cross_tol:.3f}s "
            f"extra_shift max={total_shift.max().item():.2f}m",
            flush=True,
        )

    def _sample_coord_priority(self, env_ids):
        if self._coord_random_priority and len(env_ids) > 0:
            self._coord_priority_agent[env_ids] = torch.randint(
                0, AGENTS, (len(env_ids),), device=self.device
            )
            self._coord_priority_locked[env_ids] = True
        elif self._coord_learned_priority and len(env_ids) > 0:
            self._coord_priority_agent[env_ids] = 0
            self._coord_priority_locked[env_ids] = False
        elif len(env_ids) > 0:
            self._coord_priority_agent[env_ids] = 0
            self._coord_priority_locked[env_ids] = True

    @staticmethod
    def _resolve_checkpoint(raw: str) -> Path:
        path = Path(raw).expanduser()
        candidates = [path]
        if not path.is_absolute():
            candidates.extend((Path.cwd() / path, Path.cwd().parent / path))
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(f"coordinator checkpoint not found: {raw}")

    @staticmethod
    def _yaw(root_state: torch.Tensor) -> torch.Tensor:
        return torch.atan2(
            2.0 * (root_state[..., 6] * root_state[..., 5]
                   + root_state[..., 3] * root_state[..., 4]),
            1.0 - 2.0 * (root_state[..., 4].square() + root_state[..., 5].square()),
        )

    def _coord_state(self, env_ids) -> CoordinatorState:
        rows = self.agent_rows(env_ids)
        n = len(env_ids)
        humanoid = self.humanoid_rows(self._humanoid_root_states)[rows].reshape(n, AGENTS, -1)
        box = self.humanoid_rows(self._box_states)[rows].reshape(n, AGENTS, -1)
        goal = self._box_tar_pos[rows, :2].reshape(n, AGENTS, 2)
        size = self._box_lib._box_size[rows, :2].reshape(n, AGENTS, 2)
        raw_held = box[..., 2] > self._box_lib._box_size[rows, 2].reshape(n, AGENTS) / 2.0 + 0.2
        if hasattr(self, "_coord_held_streak"):
            held = self._coord_held_streak[env_ids] >= self._coord_grasp_stable_steps
        else:
            held = raw_held
        root_near = (humanoid[..., :2] - box[..., :2]).norm(dim=-1) <= 0.7
        goal_near = (box[..., :2] - goal).norm(dim=-1) <= 0.15
        phase = torch.zeros_like(held, dtype=humanoid.dtype)
        phase = torch.where(root_near, torch.ones_like(phase), phase)
        phase = torch.where(held, torch.full_like(phase, 2.0), phase)
        phase = torch.where(goal_near & ~held, torch.full_like(phase, 3.0), phase)
        return CoordinatorState(
            root_xy=humanoid[..., :2],
            heading=self._yaw(humanoid),
            root_vel_xy=humanoid[..., 7:9],
            box_xyz=box[..., :3],
            box_heading=self._yaw(box),
            box_vel_xy=box[..., 7:9],
            box_size_xy=size,
            goal_xy=goal,
            held=held.float(),
            phase=phase,
        )

    @staticmethod
    def _analytic_plan(state: CoordinatorState):
        t = torch.linspace(0.0, 1.0, 17, device=state.device, dtype=state.root_xy.dtype)
        first = state.root_xy[:, :, None] + t[None, None, :, None] * (
            state.box_xyz[..., :2] - state.root_xy
        )[:, :, None]
        second = state.box_xyz[..., :2][:, :, None] + t[None, None, :, None] * (
            state.goal_xy - state.box_xyz[..., :2]
        )[:, :, None]
        path = torch.cat((first, second[:, :, 1:]), dim=2)
        speed = torch.full(path.shape[:-1], MAX_SPEED, device=state.device, dtype=state.root_xy.dtype)
        return path, speed

    def _preserve_executor_pickup_approach(self, output, state):
        """Keep the executor's learned root-to-box pickup geometry intact.

        The 340-D masteer actor learned pickup from the native straight approach
        ending at the box anchor. C13 may bend those first 16 segments to avoid
        another agent, which changes the final box-relative arrival direction
        and prevents the hands from establishing the learned grasp. C13 still
        owns approach timing and the complete box-to-goal carry path.
        """
        if not self._coord_preserve_pickup_approach:
            return output
        path = output["path_world"]
        unit = torch.linspace(0.0, 1.0, 17, device=path.device,
                              dtype=path.dtype)
        straight = (state.root_xy[:, :, None]
                    + unit[None, None, :, None]
                    * (state.box_xyz[..., :2] - state.root_xy)[:, :, None])
        straight = straight[:, None].expand(-1, path.shape[1], -1, -1, -1)
        constrained = dict(output)
        constrained_path = torch.cat((straight, path[..., 17:, :]), dim=-2)
        constrained["path_world"] = constrained_path
        constrained["trajectory"] = torch.cat(
            (constrained_path, output["speed"].unsqueeze(-1)), dim=-1)
        ds = (constrained_path[..., 1:, :] - constrained_path[..., :-1, :]
              ).norm(dim=-1).clamp(min=1e-5)
        speed = output["speed"]
        constrained["acceleration"] = (
            speed[..., 1:].square() - speed[..., :-1].square()) / (2.0 * ds)
        if "waypoint_residual" in constrained:
            residual = constrained["waypoint_residual"].clone()
            residual[..., :15, :] = 0.0
            constrained["waypoint_residual"] = residual
        return constrained

    @staticmethod
    def _plan_validity_checks(
        state: CoordinatorState, path: torch.Tensor, speed: torch.Tensor
    ):
        finite = torch.isfinite(path).flatten(start_dim=1).all(dim=-1)
        finite &= torch.isfinite(speed).flatten(start_dim=1).all(dim=-1)
        anchors = (path[:, :, 0] - state.root_xy).norm(dim=-1).amax(dim=1) <= 0.01
        anchors &= (path[:, :, 16] - state.box_xyz[..., :2]).norm(dim=-1).amax(dim=1) <= 0.01
        anchors &= (path[:, :, 32] - state.goal_xy).norm(dim=-1).amax(dim=1) <= 0.01
        segment = path[:, :, 1:] - path[:, :, :-1]
        length = segment.norm(dim=-1).sum(dim=-1)
        buffer_ok = (length < (PATH_VERTICES - 2) * PATH_DS).all(dim=1)
        speed_ok = ((speed >= MIN_SPEED - 1e-5) & (speed <= MAX_SPEED + 1e-5))
        speed_ok = speed_ok.flatten(start_dim=1).all(dim=-1)
        v0, v1 = segment[:, :, :-1], segment[:, :, 1:]
        cosine = (v0 * v1).sum(dim=-1) / (
            v0.norm(dim=-1) * v1.norm(dim=-1)
        ).clamp(min=1e-7)
        turn = torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0)))
        turn[:, :, 14:17] = 0.0
        turn[:, :, :16] = torch.where(
            state.held[:, :, None] >= 0.5,
            torch.zeros_like(turn[:, :, :16]),
            turn[:, :, :16],
        )
        curve_ok = turn.flatten(start_dim=1).amax(dim=-1) <= 46.0
        return torch.stack((finite, anchors, buffer_ok, speed_ok, curve_ok), dim=-1)

    @staticmethod
    def _validate_plan(state: CoordinatorState, path: torch.Tensor, speed: torch.Tensor):
        return HumanoidMACoordCarry._plan_validity_checks(
            state, path, speed
        ).all(dim=-1)

    def _install_plan(self, env_ids, path, speed):
        dense, dense_speed, end_s = resample_path_and_speed(path, speed)
        rows = self.agent_rows(env_ids)
        flat_path = dense.reshape(-1, PATH_VERTICES, 2)
        flat_end = end_s.reshape(-1)
        flat_box = self.humanoid_rows(self._box_states)[rows, :2]
        s_box, _, _ = sp.project(flat_box, flat_path)
        self._gt_path[rows] = flat_path
        self._mscale[rows] = dense_speed.reshape(-1, PATH_VERTICES) / MAX_SPEED
        self._s_end[rows] = flat_end
        self._arc_root[rows] = 0.0
        self._arc_box[rows] = s_box
        self._prev_arc[rows] = 0.0
        self._coord_cmd_tick[rows] = -1

    @torch.no_grad()
    def _plan_envs(self, env_ids):
        if len(env_ids) == 0:
            return
        state = self._coord_state(env_ids)
        if self._coord_provider in ("analytic", "external"):
            path, speed = self._analytic_plan(state)
            candidate_valid = torch.ones(len(env_ids), device=self.device, dtype=torch.bool)
            candidate_safe = candidate_valid
            selected = torch.full((len(env_ids),), -1, device=self.device, dtype=torch.long)
        else:
            with torch.inference_mode():
                output = self._coordinator(state)
                if self._coord_learned_priority:
                    unlocked = ~self._coord_priority_locked[env_ids]
                    if unlocked.any():
                        proposed = output["priority_logits"].argmax(dim=-1)
                        chosen = env_ids[unlocked]
                        self._coord_priority_agent[chosen] = proposed[unlocked]
                        self._coord_priority_locked[chosen] = True
                    output = apply_fixed_priority(
                        output, state, self._coord_priority_agent[env_ids]
                    )
                elif self._coord_random_priority:
                    output = apply_fixed_priority(
                        output, state, self._coord_priority_agent[env_ids]
                    )
                output = self._preserve_executor_pickup_approach(output, state)
                choice = select_candidate(output, state)
            self._record_choice_diagnostics(output, choice)
            self._coord_candidates[env_ids] = output["path_world"]
            path, speed = choice.path_world, choice.speed
            candidate_valid, candidate_safe = choice.valid, choice.safe
            selected = choice.index
        validity_checks = self._plan_validity_checks(state, path, speed)
        bridge_valid = validity_checks.all(dim=-1)
        valid = candidate_valid & bridge_valid
        self._coord_invalid_reasons += (~validity_checks).sum(dim=0)
        curve_invalid = ~validity_checks[:, 4]
        held_count = (state.held >= 0.5).sum(dim=-1).long()
        for count in range(3):
            self._coord_curve_invalid_by_held[count] += (
                curve_invalid & (held_count == count)
            ).sum()
        self._coord_replans[env_ids] += 1
        self._coord_invalid[env_ids] += (~valid).long()
        self._coord_unsafe[env_ids] += (valid & ~candidate_safe).long()
        self._coord_selected[env_ids] = selected

        if valid.any():
            chosen = env_ids[valid]
            self._install_plan(chosen, path[valid], speed[valid])
            self._coord_has_valid[chosen] = True

        need_fallback = (~valid) & (~self._coord_has_valid[env_ids])
        if need_fallback.any():
            fallback_state = state.index(need_fallback)
            fallback_path, fallback_speed = self._analytic_plan(fallback_state)
            chosen = env_ids[need_fallback]
            self._install_plan(chosen, fallback_path, fallback_speed)
            self._coord_has_valid[chosen] = True
            self._coord_fallback[chosen] += 1
        # An invalid replan after a valid one keeps the previous complete plan.
        self._coord_last_replan[env_ids] = self.progress_buf[env_ids]
        self._coord_phase[env_ids] = state.phase

    def _set_speed_anchor(self, rows):
        humanoid = self.humanoid_rows(self._humanoid_root_states)[rows]
        actual = humanoid[:, 7:9].norm(dim=-1).clamp(MIN_SPEED, MAX_SPEED)
        self._coord_cmd_speed[rows] = actual
        self._coord_cmd_tick[rows] = -1

    def _update_speed_command(self, rows):
        if len(rows) == 0:
            return
        tick = self.progress_rows()[rows]
        update = tick != self._coord_cmd_tick[rows]
        if not update.any():
            return
        chosen = rows[update]
        desired = HumanoidMASteerCarry._m_at(
            self, self._arc_root[chosen], chosen
        ) / 1.6
        if self._coord_speed_limit:
            current = self._coord_cmd_speed[chosen]
            delta = desired - current
            delta = torch.minimum(delta, torch.full_like(delta, self._coord_accel_up * self.dt))
            delta = torch.maximum(delta, torch.full_like(delta, -self._coord_accel_down * self.dt))
            desired = current + delta
        self._coord_cmd_speed[chosen] = desired.clamp(MIN_SPEED, MAX_SPEED)
        self._coord_cmd_tick[chosen] = tick[update]

    def _m_at(self, arc, rows=None):
        if getattr(self, "_coord_limited_lookup", False):
            index = torch.arange(arc.shape[0], device=self.device) if rows is None else rows
            return 1.6 * self._coord_cmd_speed[index]
        return super()._m_at(arc, rows)

    def _scen_selfcheck(self, rows):
        """Skip the inherited GT-path geometry assertion.

        The parent invokes this while constructing its temporary scenario path,
        immediately before this subclass replaces that path with a coordinator
        output.  Straight/cross assertions about the discarded GT provider are
        therefore neither meaningful nor valid for learned curved candidates;
        coordinator plans are checked by ``_validate_plan`` instead.
        """
        if not getattr(self, "_coord_selfcheck_noted", False):
            self._coord_selfcheck_noted = True
            print("[coord] inherited GT scenario self-check skipped; validating selected C1 plan", flush=True)
        return

    def _reset_steer(self, rows):
        super()._reset_steer(rows)
        if not getattr(self, "_coord_ready", False) or len(rows) == 0:
            return
        env_ids = torch.div(rows, AGENTS, rounding_mode="floor").unique()
        self._coord_held_streak[env_ids] = 0
        self._coord_held_tick[env_ids] = -1
        self._sample_coord_priority(env_ids)
        self._coord_has_valid[env_ids] = False
        self._coord_last_replan[env_ids] = -self._coord_replan_steps
        self._set_speed_anchor(self.agent_rows(env_ids))
        self._plan_envs(env_ids)

    def _maybe_replan(self, env_ids):
        if not self._coord_ready or self._coord_provider == "external" or len(env_ids) == 0:
            return
        env_ids = env_ids.unique()
        # A box briefly crossing the lift threshold is still in the pickup
        # transient.  Replanning as carry on that single frame changes the
        # steer window while the hands are closing and commonly drops it.
        tick = self.progress_buf[env_ids]
        update = tick != self._coord_held_tick[env_ids]
        if update.any():
            ids = env_ids[update]
            rows = self.agent_rows(ids)
            box = self.humanoid_rows(self._box_states)[rows].reshape(len(ids), AGENTS, -1)
            half = self._box_lib._box_size[rows, 2].reshape(len(ids), AGENTS) * 0.5
            raw_held = box[..., 2] > half + 0.2
            previous = self._coord_held_streak[ids]
            self._coord_held_streak[ids] = torch.where(
                raw_held, previous + 1, torch.zeros_like(previous)
            )
            self._coord_held_tick[ids] = tick[update]
        state = self._coord_state(env_ids)
        phase_changed = (state.phase != self._coord_phase[env_ids]).any(dim=1)
        due = (self.progress_buf[env_ids] - self._coord_last_replan[env_ids]) >= self._coord_replan_steps
        self._plan_envs(env_ids[due | phase_changed])

    def coord_state(self, env_ids=None) -> CoordinatorState:
        """Public refreshed-state interface used by the high-level trainer."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        return self._coord_state(env_ids)

    def coord_priority_agent(self, env_ids=None) -> torch.Tensor:
        """Return the episode-persistent physical agent with fixed priority."""
        if env_ids is None:
            env_ids = torch.arange(
                self.num_envs, device=self.device, dtype=torch.long
            )
        return self._coord_priority_agent[env_ids]

    def coord_priority_locked(self, env_ids=None) -> torch.Tensor:
        """Whether the episode-persistent priority choice has been made."""
        if env_ids is None:
            env_ids = torch.arange(
                self.num_envs, device=self.device, dtype=torch.long
            )
        return self._coord_priority_locked[env_ids]

    def set_coord_priority_agent(
        self, priority_agent: torch.Tensor, choose: torch.Tensor
    ) -> None:
        """Commit C11 categorical choices only for newly reset episodes."""
        if priority_agent.shape != (self.num_envs,):
            raise ValueError("priority_agent must be [num_envs]")
        if choose.shape != (self.num_envs,) or choose.dtype != torch.bool:
            raise ValueError("choose must be bool [num_envs]")
        choose = choose & ~self._coord_priority_locked
        if choose.any():
            self._coord_priority_agent[choose] = priority_agent[choose]
            self._coord_priority_locked[choose] = True

    def update_metrics(self):
        # Capture motion before the parent advances _prev_arc.
        if hasattr(self, "_coord_ep_sent_vbin_n"):
            v_real = ((self._arc_root - self._prev_arc) / self.dt).clamp(min=0.0)
            moving = v_real > self._metric_still_v
            sent_speed = self._coord_cmd_speed.clone()
            sent_bin = torch.bucketize(sent_speed, self._VBINS)
            distance = self.agent_min_dist()
            close = distance < self._metric_tau
            box = self.humanoid_rows(self._box_states)
            gate = box[:, 2] > self._box_lib._box_size[:, 2] / 2.0 + 0.2
            carrying_env = gate.reshape(self.num_envs, AGENTS).any(dim=1)
            carrying = carrying_env.repeat_interleave(AGENTS)
        super().update_metrics()
        if not hasattr(self, "_coord_ep_sent_vbin_n"):
            return
        rows = torch.arange(self._rows, device=self.device)
        self._coord_ep_sent_vbin_n[rows, sent_bin] += moving.float()
        phase_index = carrying.long()
        self._coord_ep_collision_phase[rows, phase_index] += close.float()
        sent_low = moving & (sent_speed <= float(self._VBINS[-1]))
        self._coord_ep_sent_low_phase[rows, phase_index] += sent_low.float()
        moving_pair = moving.reshape(self.num_envs, AGENTS)
        low_pair = sent_low.reshape(self.num_envs, AGENTS)
        active_pair = moving_pair.any(dim=1)
        low_count = low_pair.sum(dim=1)
        pair_mode = torch.stack(
            (
                active_pair & (low_count == 0),
                active_pair & (low_count == 1),
                active_pair & (low_count == AGENTS),
            ),
            dim=-1,
        )
        close_pair = close.reshape(self.num_envs, AGENTS).any(dim=1)
        pair_mode_rows = pair_mode.repeat_interleave(AGENTS, dim=0)
        collision_mode_rows = (
            pair_mode & close_pair[:, None]
        ).repeat_interleave(AGENTS, dim=0)
        self._coord_ep_pair_sent_mode += pair_mode_rows.float()
        self._coord_ep_pair_collision_mode += collision_mode_rows.float()

    def _metric_extra_cols(self, rows):
        cols = super()._metric_extra_cols(rows)
        if not hasattr(self, "_coord_ep_sent_vbin_n"):
            return cols
        cols += [self._coord_ep_collision_phase[rows, i] for i in range(2)]
        cols += [self._coord_ep_sent_vbin_n[rows, i] for i in range(4)]
        cols += [self._coord_ep_sent_low_phase[rows, i] for i in range(2)]
        cols += [self._coord_ep_pair_sent_mode[rows, i] for i in range(3)]
        cols += [self._coord_ep_pair_collision_mode[rows, i] for i in range(3)]
        return cols

    def _metric_reset_extra(self, rows):
        super()._metric_reset_extra(rows)
        if hasattr(self, "_coord_ep_sent_vbin_n"):
            self._coord_ep_collision_phase[rows] = 0.0
            self._coord_ep_sent_vbin_n[rows] = 0.0
            self._coord_ep_sent_low_phase[rows] = 0.0
            self._coord_ep_pair_sent_mode[rows] = 0.0
            self._coord_ep_pair_collision_mode[rows] = 0.0

    @torch.no_grad()
    def _record_choice_diagnostics(self, output, choice):
        residual = output.get(
            "waypoint_residual", output["control_residual"]
        ).flatten(start_dim=2)
        if self._coord_candidates_k == 1:
            diversity = torch.zeros(residual.shape[0], device=self.device)
        else:
            pairwise = torch.cdist(residual, residual)
            upper = torch.triu(
                torch.ones(
                    self._coord_candidates_k, self._coord_candidates_k,
                    device=self.device, dtype=torch.bool,
                ),
                diagonal=1,
            )
            diversity = pairwise[:, upper].mean(dim=1)
        row = torch.arange(choice.index.shape[0], device=self.device)
        metrics = choice.diagnostics
        selected = [
            metrics[name][row, choice.index]
            for name in ("min_hh", "min_bb_margin", "min_hb_margin", "makespan", "collision")
        ]
        values = torch.stack((diversity, *selected), dim=-1)
        self._coord_diag_sum += values.sum(dim=0)
        self._coord_diag_count += int(values.shape[0])
        self._coord_choice_counts += torch.bincount(
            choice.index, minlength=self._coord_candidates_k
        )[:self._coord_candidates_k]

    @torch.no_grad()
    def install_external_coord(self, output, env_ids=None):
        """Install sampled K futures from the trainer without owning its model."""
        if self._coord_provider != "external":
            raise RuntimeError("install_external_coord requires COORD_PROVIDER=external")
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        state = self._coord_state(env_ids)
        if self._coord_random_priority or self._coord_learned_priority:
            output = apply_fixed_priority(
                output, state, self._coord_priority_agent[env_ids]
            )
        choice = select_candidate(output, state)
        self._record_choice_diagnostics(output, choice)
        self._coord_candidates[env_ids] = output["path_world"]
        valid = choice.valid & self._validate_plan(state, choice.path_world, choice.speed)
        self._coord_replans[env_ids] += 1
        self._coord_invalid[env_ids] += (~valid).long()
        self._coord_unsafe[env_ids] += (valid & ~choice.safe).long()
        self._coord_selected[env_ids] = choice.index
        if valid.any():
            chosen = env_ids[valid]
            self._install_plan(chosen, choice.path_world[valid], choice.speed[valid])
            self._coord_has_valid[chosen] = True
        need_fallback = (~valid) & (~self._coord_has_valid[env_ids])
        if need_fallback.any():
            fallback_state = state.index(need_fallback)
            path, speed = self._analytic_plan(fallback_state)
            chosen = env_ids[need_fallback]
            self._install_plan(chosen, path, speed)
            self._coord_has_valid[chosen] = True
            self._coord_fallback[chosen] += 1
        self._coord_last_replan[env_ids] = self.progress_buf[env_ids]
        self._coord_phase[env_ids] = state.phase
        return choice.index, valid, choice.safe

    def _compute_task_obs(self, env_ids=None):
        active = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        self._maybe_replan(active)
        rows = self.all_rows() if env_ids is None else self.agent_rows(env_ids)
        self._update_speed_command(rows)
        self._coord_limited_lookup = True
        try:
            return super()._compute_task_obs(env_ids)
        finally:
            self._coord_limited_lookup = False

    def _draw_task(self):
        # The inherited ribbon reads planned dense speed from _mscale, while
        # its waist-high K-point window calls _m_at().  Make that active window
        # use the same rate-limited command that is actually sent to ms18.
        self._coord_limited_lookup = True
        try:
            super()._draw_task()
        finally:
            self._coord_limited_lookup = False
        if (
            not self._coord_draw_candidates
            or self.viewer is None
            or self._coord_provider != "learned"
        ):
            return
        # Selected plan is already the bright inherited ribbon. Draw all learned
        # alternatives as thin muted lines so the K modes can be inspected.
        candidates = self._coord_candidates[0].detach().cpu().numpy()
        colors = np.asarray(
            [[0.35, 0.35, 0.35], [0.25, 0.45, 0.70], [0.55, 0.35, 0.65], [0.35, 0.60, 0.45]],
            dtype=np.float32,
        )
        for k in range(self._coord_candidates_k):
            for agent in range(AGENTS):
                points = candidates[k, agent]
                vertices = np.concatenate(
                    (
                        points[:-1], np.full((PATH_POINTS - 1, 1), 0.04, dtype=np.float32),
                        points[1:], np.full((PATH_POINTS - 1, 1), 0.04, dtype=np.float32),
                    ),
                    axis=1,
                ).astype(np.float32)
                color = np.repeat(colors[k:k + 1], PATH_POINTS - 1, axis=0)
                self.gym.add_lines(self.viewer, self.envs[0], PATH_POINTS - 1, vertices, color)

    def _report_coord_runtime(self):
        replans = int(self._coord_replans.sum().item())
        invalid = int(self._coord_invalid.sum().item())
        unsafe = int(self._coord_unsafe.sum().item())
        fallback = int(self._coord_fallback.sum().item())
        print(
            f"COORD_SUMMARY provider={self._coord_provider} replans={replans} "
            f"invalid={invalid} invalid_rate={invalid / max(replans, 1):.6f} "
            f"unsafe_selected={unsafe} analytic_fallback={fallback}",
            flush=True,
        )
        reasons = self._coord_invalid_reasons.tolist()
        print(
            "COORD_INVALID_REASONS "
            f"finite={int(reasons[0])} anchors={int(reasons[1])} "
            f"buffer={int(reasons[2])} speed={int(reasons[3])} "
            f"curve={int(reasons[4])}",
            flush=True,
        )
        curve_by_held = self._coord_curve_invalid_by_held.tolist()
        print(
            "COORD_CURVE_INVALID_BY_HELD "
            f"held0={int(curve_by_held[0])} held1={int(curve_by_held[1])} "
            f"held2={int(curve_by_held[2])}",
            flush=True,
        )
        if self._coord_diag_count:
            mean = (self._coord_diag_sum / self._coord_diag_count).tolist()
            counts = ",".join(str(int(value)) for value in self._coord_choice_counts.tolist())
            print(
                f"COORD_CANDIDATES choices={counts} diversity={mean[0]:.4f} "
                f"min_hh={mean[1]:.4f} bb_margin={mean[2]:.4f} "
                f"hb_margin={mean[3]:.4f} makespan={mean[4]:.4f} "
                f"pred_collision={mean[5]:.6f}",
                flush=True,
            )
