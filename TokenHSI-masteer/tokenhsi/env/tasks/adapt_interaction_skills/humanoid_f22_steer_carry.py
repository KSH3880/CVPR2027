"""S1+: Carry with a steering trajectory window as a separate input channel.

Reuses the adapt scaffold (pretrained carry tokens + one trainable extra token,
loaded from --hrl_checkpoint) but swaps the terrain height map for K waypoints
of the commanded path, expressed in the root's yaw frame at arc-length spacing.

The final target (`_tar_pos`, the last 3 dims of the carry block) is never
touched: steering and placement stay separate channels.

Config via STEER_* environment variables, so no shared cfg has to change:
  STEER_K=6 STEER_SPACING=0.4 STEER_CURVATURE=0.15 STEER_RADIUS=4.0
  STEER_ZERO=1   emit a zero window (obs-level off switch)
"""

import atexit
import math
import os
from isaacgym import gymtorch   # STEER_REPLAY writes states straight into the sim
import numpy as np
import torch
import torch.nn.functional as F

from tokenhsi.env.tasks.humanoid import Humanoid
from tokenhsi.env.tasks.adapt_interaction_skills.humanoid_adapt_carry_ground2terrain import (
    HumanoidAdaptCarryGround2Terrain,
)
from tokenhsi.utils import steer_path as sp


def _f(name, default):
    return float(os.environ.get(name, default))


class _FlatTerrain:
    """Stands in for the trimesh terrain on flat ground.

    Only two methods are reached once terrain_obs is off: valid spawn locations
    and height samples (zero on a plane).

    Spawn locations must SPREAD THE CHARACTERS OUT, which is not obvious because
    nothing in the task cares where in the world an episode happens -- the box
    and target are sampled relative to the character, so returning the origin
    for every env is behaviourally identical. It is not identical to PhysX.
    envSpacing is 0, so every env is built at the same world position, and
    cross-env contacts are already filtered by collision group; but the GPU
    broadphase still tracks every pair whose bounds overlap. Stack N characters
    on one spot and that is N^2 aggregate pairs, which is what actually caps
    num_envs: at 2048 PhysX warns it is dropping interactions, at 4096 it needs
    a 4x buffer, at 8192 it dies. The real terrain scatters spawns across its
    mesh and never trips this -- the official 2048-env terrain run logs zero
    warnings. This does the same on bare ground.
    """

    # metres between env cells. Bigger means fewer overlapping bounds and a
    # smaller broadphase pair count; STEER_CELL exists to sweep it.
    CELL = float(os.environ.get("STEER_CELL", 60.0))

    def __init__(self, device, num_envs):
        self.device = device
        cols = max(int(math.ceil(math.sqrt(num_envs))), 1)
        idx = torch.arange(num_envs, device=device)
        grid = torch.stack([(idx % cols).float(), idx.div(cols, rounding_mode="floor").float()], dim=-1)
        self._origins = (grid - cols / 2.0) * self.CELL

    def sample_valid_locations(self, n, env_ids=None):
        if env_ids is None:
            return self._origins[:n]
        return self._origins[env_ids]

    def sample_height_points(self, points, env_ids=None):
        return torch.zeros(points.shape[:-1], device=self.device)


class HumanoidF22SteerCarry(HumanoidAdaptCarryGround2Terrain):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.steer_k = int(_f("STEER_K", 6))
        self.steer_spacing = _f("STEER_SPACING", 0.4)
        self.steer_curvature = _f("STEER_CURVATURE", 0.15)
        self.steer_radius = _f("STEER_RADIUS", 4.0)
        self.steer_sigma = _f("STEER_SIGMA", 0.0)
        self.steer_seed = int(_f("STEER_SEED", 0))
        self.steer_zero = bool(int(os.environ.get("STEER_ZERO", "0")))
        self.steer_handoff = _f("STEER_HANDOFF", 1.5)
        self.steer_reaim = bool(int(os.environ.get("STEER_REAIM", "0")))
        self.steer_terminate_on_success = bool(int(os.environ.get("STEER_TERM", "0")))
        # rule-based reference: overwrite the carry target with a lookahead point,
        # exactly as the no-training probe did, but inside THIS env
        self.steer_probe = bool(int(os.environ.get("STEER_PROBE", "0")))
        self.steer_lookahead = _f("STEER_LOOKAHEAD", 1.2)
        self.steer_aim = os.environ.get("STEER_AIM", "lookahead")  # or "tangent"
        # cross-track term, paid for out of walk_r: during transport the root is
        # always within 0.5 m of the box it carries, so walk_r sits pinned at its
        # maximum and contributes no gradient. That 0.2 is free to respend.
        self.steer_xtrack = _f("STEER_XTRACK", 0.0)
        # Drop the held boolean from the xtrack term the way the walk/carry terms
        # already did: root on leg 1, box on leg 2, each paying half. F4 made this
        # worth trying -- xtrack won the reward batch (0.139 vs base 0.208) but is
        # the only place a phase switch survives, and removing phase switches is
        # what took approach deviation from 0.214 to 0.155 in the first place.
        self.steer_xtrack_pf = bool(int(os.environ.get("STEER_XTRACK_PF", "0")))
        # aim-point pos term: the shipped pos form with the aim point as the goal,
        # so pos and vel pull toward the same place. "lat" swaps the aim point for
        # the perpendicular distance, which is the same quantity with the lookahead
        # floor removed -- d^2 = L^2 + lat^2, so aim compresses lat by that floor
        # and reaches only exp(-0.5*L^2) at its best. Both pin on the native
        # box/target distance, never on their own goal.
        self.steer_pos = os.environ.get("STEER_POS", "off")  # off | aim | lat | latpen
        self.steer_pos_c = _f("STEER_POS_C", 0.1)
        # Radius inside which the STEERING term stops paying, separate from the
        # 0.5 m band the shipped terms pin on. The box sits on a 0.4 x 0.4 plate
        # when it is raised, and the plate is at chest height; the native policy
        # picks its own approach and squares up to it, but a commanded path forces
        # a direction and can walk the torso into the plate. Widening this hands
        # the last stretch back to the policy without touching the native reward,
        # so _verify_phasefree still matches. 0.5 = the shipped band, i.e. no-op.
        self.steer_pin = _f("STEER_PIN_STEER", 0.5)
        # The reward has been phase-free since the ratchets went in -- _arc_root
        # and _arc_box both advance every step and neither is gated on a
        # boolean. The OBSERVATION never caught up: _track() still picks one
        # subject with `held`, and at the grasp the anchor teleports 0.37 m (the
        # root-to-box distance while carrying). Measured, arc length jumps
        # 0.393 m in one frame against 0.0085 m of normal drift -- the whole
        # window slides a full slot, and deviation spikes 0.09 -> 0.16 for the
        # 1.3 s it takes to settle. Three ways out, all off by default:
        #   STEER_DUAL=1        feed BOTH windows, one per ratchet. No boolean
        #                       anywhere, obs mirrors the reward exactly, and
        #                       the transport leg is visible during approach.
        #                       Costs 2x the extra-token width.
        #   STEER_GRIP_RAMP=N   spread the same switch over N frames.
        #   STEER_HYST=1        hysteresis on the grasp test (8.7 % of episodes
        #                       flip it more than twice).
        #   STEER_SLIM=1        the compact form of DUAL. The second window is
        #                       almost the first one shifted 0.37 m, so instead
        #                       of paying 2x the width it carries the two
        #                       numbers the reward actually measures on the box:
        #                       how far ahead it sits in arc length, and how far
        #                       off the path it is. 2K+2 instead of 4K, and the
        #                       policy finally SEES the lat_box it is graded on.
        # Blend the path tangent into the velocity direction. 0 = chord only,
        # which is what everything up to F15 used. See _aim_dir.
        self.steer_tang = _f("STEER_TANG", 0.0)
        # Shorten the lookahead where the path bends: L = L0 / (1 + g*kappa).
        # Same target as STEER_TANG (corner cutting) by a different route --
        # sagitta goes as L^2, so halving L on a curve quarters the cut. Kept as
        # a fallback because a short FIXED L already lost (0.2901 vs 0.1994):
        # short everywhere makes the direction swing when the body is off path.
        self.steer_l_kappa = _f("STEER_L_KAPPA", 0.0)
        self.steer_slim = bool(int(os.environ.get("STEER_SLIM", "0")))
        self.steer_dual = bool(int(os.environ.get("STEER_DUAL", "0")))
        self.steer_grip_ramp = _f("STEER_GRIP_RAMP", 0.0)
        self.steer_hyst = bool(int(os.environ.get("STEER_HYST", "0")))
        # arc-length progress: rewards covering the path itself rather than
        # heading toward a point on it. Corner-cutting inflates it, so it is
        # only meaningful paired with the cross-track term.
        self.steer_progress = bool(int(os.environ.get("STEER_PROGRESS", "0")))
        # transport-only path (box -> target), the S0-S6 definition. Kept so the
        # earlier numbers stay reproducible; the default is now the whole task.
        self.steer_legacy = bool(int(os.environ.get("STEER_LEGACY", "0")))
        # steer the approach leg too. Off reproduces "path starts at the box".
        self.steer_approach = bool(int(os.environ.get("STEER_APPROACH", "1")))
        # Phase-free reward: instead of adding a term chosen by a held boolean,
        # rebuild walk_r and carry_r with the aim direction swapped in and every
        # other rule -- weights, pinning bands, gates -- left exactly as shipped.
        # walk_r keeps steering the root along the approach leg and carry_r keeps
        # steering the box along the transport leg, which is what the path already
        # encodes: it is generated as character -> box -> target.
        self.steer_phasefree = bool(int(os.environ.get("STEER_PHASEFREE", "0")))
        # --- time-indexed steering (STEER_TIME=1) ---------------------------
        # The arc-length window is anchored to the body's own projection, so
        # "stand still" and "you are on track" are the same observation and the
        # reward has 1.5 m/s hardcoded. Indexing the command by TIME instead
        # makes speed a property of the trajectory: point spacing is speed,
        # repeated points are a stop. Nothing here is on unless STEER_TIME=1.
        self.steer_time = bool(int(os.environ.get("STEER_TIME", "0")))
        self.steer_stride = int(_f("STEER_STRIDE", 8))   # 8 steps = 0.4 m at 1.5
        self.steer_vkappa = _f("STEER_VKAPPA", 0.0)      # v = v_nom/(1+g*kappa)
        self.steer_vnom = _f("STEER_VNOM", 1.5)
        self.steer_stops = int(_f("STEER_STOPS", 0))     # stops per episode
        self.steer_stoplen = int(_f("STEER_STOPLEN", 45))  # 1.5 s at 30 Hz
        # tau advances on its own clock, and the clock STOPS while the body is
        # closing on the box but has not grasped yet. Grasping takes one to two
        # seconds of standing still, and tau at 1.5 m/s does not wait: measured
        # offline on the best existing policy, |xy - tau_t| is p90 0.81 m before
        # the grasp and p90 5.07 m after it. The character tops out at 1.59 m/s
        # against a 1.5 m/s command, so 0.09 m/s of headroom needs 33 s to undo a
        # 3 m gap inside a 20 s episode -- the lag is permanent once opened.
        #
        # This is deliberately NOT a stop baked into tau at generation time. A
        # planner can say where to go and when, but not the millisecond a hand
        # closes; baking it in puts the pause at a fixed step, which is the wrong
        # moment whenever the policy arrives early or late. Navigation timing
        # stays commanded, manipulation timing stays reactive.
        self.steer_grabpin = _f("STEER_GRABPIN", 0.5)
        # STEER_REPLAN=<N>: rebuild the path and tau from the body's ACTUAL state
        # every N control steps, which is what a world model does in deployment
        # (METHOD_V2 4.2: actual state -> plan -> act -> actual state -> replan).
        #
        # Training on one global trajectory for a whole 20 s episode solves a
        # different problem than deployment solves: lag accumulates with nothing
        # to undo it, and the grasp pin exists only to patch the one place where
        # that hurts most. Re-anchoring every N steps bounds the lag by
        # construction and makes the pin redundant.
        #
        # It also turns N into the axis the multi-agent layer actually needs:
        # trk_err as a function of N is "how long can the policy hold on its own",
        # i.e. how often the planner has to speak.
        #
        # NOTE for scoring: with replanning the stored path changes mid-episode,
        # so app_xt (distance to *the* curve) stops being well defined. Read the
        # online trk_err instead -- it is logged per frame against the tau in
        # force at that frame.
        # STEER_RESYNC=<N>: every N steps, snap tau's clock to the index whose
        # point is nearest the body. The path does not change -- only the clock.
        #
        # This is the cheap half of replanning. A world model replanning from the
        # actual state produces, for a single agent with nothing in the way, very
        # nearly the old plan starting from where the body now is; the shape
        # barely moves, the anchor does. Re-anchoring the clock reproduces that
        # without the discontinuity that regenerating the whole path introduces
        # (STEER_REPLAN redraws the bow direction too, so the command swings
        # every N steps, which no planner would do).
        #
        # It also bounds the lag by construction: e resets to roughly 0 every N
        # steps, so exp(-0.5 e^2) never reaches the flat region where its
        # gradient dies. That is the same job the grasp pin does, done generally.
        # STEER_MRAND=<n>: split the path into n arc segments and give each one a
        # random speed multiplier. The window still shows six equally spaced
        # points, but the length it spans (M) is the command: M = 2.4 m is the
        # 1.5 m/s the policy already knows, and half that means half the speed.
        # Target speed reads straight off it, v_tar = M / 1.6 s, so the 1.5
        # constant leaves the reward.
        #
        # The multiplier is deliberately NOT a function of curvature. Tying it to
        # the path shape lets the policy satisfy the reward by reading geometry
        # it can already see -- "there is a bend, so slow down" -- and never learn
        # to read the spacing at all. Deployment needs the opposite: a world
        # model asking for half speed on a straight because another agent is
        # crossing. Randomising it is what forces the spacing to be the signal.
        self.steer_mrand = int(_f("STEER_MRAND", 0))
        self.steer_m_nom = _f("STEER_M_NOM", 2.4)
        self.steer_m_lo = _f("STEER_M_LO", 0.25)
        self.steer_resync = int(_f("STEER_RESYNC", 0))
        self.steer_replan = int(_f("STEER_REPLAN", 0))
        self.steer_shape_seed = None
        # loaded lazily on the first step: num_envs and the sim tensors do not
        # exist yet at this point in __init__
        self._replay_path = os.environ.get("STEER_REPLAY", "")
        self._replay = None
        self.steer_back = _f("STEER_RATCHET_BACK", 0.5)
        # Textbook pure pursuit scales the lookahead with speed. Ours is fixed,
        # so a character slowed by the box it carries still aims as far ahead as
        # one walking free. STEER_L_GAIN > 0 switches to L = gain * speed.
        self.steer_l_gain = _f("STEER_L_GAIN", 0.0)
        self.steer_l_min = _f("STEER_L_MIN", 0.4)
        self.steer_l_max = _f("STEER_L_MAX", 3.0)
        # Separate lookaheads for the two legs: the approach is empty-handed and
        # nimble, the transport is loaded. No reason the same value is best.
        self.steer_l_carry = _f("STEER_L_CARRY", 0.0)   # 0 = same as approach
        # Stanley's other half: align the facing direction with the path, not
        # just the velocity. A humanoid can drift sideways with correct velocity.
        self.steer_heading = _f("STEER_HEADING", 0.0)
        # End the episode once the body is hopelessly far from the commanded
        # path, the way TokenHSI's own traj task does. Stops wasting the rest of
        # a rollout that can no longer produce a useful gradient.
        self.steer_faildist = _f("STEER_FAILDIST", 0.0)
        # v2 paths bow once per leg by a distance drawn in metres, capped by turn
        # rate. v1 (default) is the three-mode fraction-of-length version every
        # result so far was produced with; kept so those stay reproducible.
        # v2 is the base from 2026-08-14 on. v1's three sine modes wiggled so
        # often that a straight walk scored almost as well as a steered one
        # (control deviated 0.168 m, steered 0.117 m), which left no room to
        # measure anything. v1 is kept only to reproduce the F0-F3 numbers.
        self.steer_traj = os.environ.get("STEER_TRAJ", "v2")
        self.steer_lat_min = _f("STEER_LAT_MIN", 0.3)
        self.steer_lat_max = _f("STEER_LAT_MAX", 1.5)
        self.steer_turn_max = _f("STEER_TURN_MAX", 35.0)
        # probability a leg weaves as an S instead of bowing once. Two humps
        # change side once, which a straight walk still cannot fake; three
        # change side twice and that is the v1 shape that hid the effect.
        self.steer_hump2 = _f("STEER_HUMP2", 0.0)
        # How far the crest of a leg's bow may wander off centre. 0 reproduces
        # every path generated so far: sin(pi t) peaks at t=0.5 always, so two
        # legs with the same amplitude have the same shape and the only variety
        # in the whole distribution is size and side.
        self.steer_skew = _f("STEER_SKEW", 0.0)
        # Which direction the velocity terms measure against. "aim" is what every
        # F-series run used -- the native direction was replaced by the aim point.
        # "native" puts it back, leaving path following entirely to the lat term,
        # so the two stop competing: the velocity term paces toward the goal and
        # the lat term keeps the body on the path that leads there.
        self.steer_vel = os.environ.get("STEER_VEL", "aim")   # aim | native
        # How the bow's turning is distributed along a leg: >1 concentrates it at
        # the crest, <1 spreads it toward the ends. The crest height is unchanged
        # either way -- the shape is renormalised -- so this is independent of
        # STEER_LAT_MAX, which is the point of having it.
        self.steer_spread_min = _f("STEER_SPREAD_MIN", 1.0)
        self.steer_spread_max = _f("STEER_SPREAD_MAX", 1.0)
        # Cap the drawn bow at this fraction of the leg's own length. 0 = off.
        self.steer_lat_frac = _f("STEER_LAT_FRAC", 0.0)
        # Override the cfg's onlyVelReward from the environment so both arms can
        # run from one config file. True (the shipped carry / backbone setting)
        # drops the position term and doubles the velocity one in BOTH legs.
        self.steer_onlyvel = os.environ.get("STEER_ONLYVEL", "")
        self._steer_tick = 0

        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine,
                         device_type=device_type, device_id=device_id, headless=headless)

        # Overwrite the single flag the parent read from cfg, so the shipped
        # reward and the phase-free rebuild always agree -- overriding only one
        # of them would make STEER_PF_VERIFY report a mismatch that is really
        # just the two halves reading different settings.
        if self.steer_onlyvel != "":
            self._only_vel_reward = bool(int(self.steer_onlyvel))

        self._gt_path = torch.zeros(self.num_envs, sp.V, 2, device=self.device)
        if self.steer_mrand > 0:
            self._mscale = torch.ones(self.num_envs, sp.V, device=self.device)
        if self.steer_time:
            self._tau_H = int(self.max_episode_length) + self.steer_stride * self.steer_k + 4
            self._tau = torch.zeros(self.num_envs, self._tau_H, 2, device=self.device)
            self._tau_clock = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._s_box = torch.zeros(self.num_envs, device=self.device)
        self._success_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._final_tar = torch.zeros(self.num_envs, 3, device=self.device)

        # path feasibility instrumentation: nothing here constrains generation,
        # it only records what was asked for so the envelope can be measured
        # against what the character actually managed to follow.
        self._diag_n = int(_f("STEER_DIAG", 0))
        if self._diag_n:
            self._diag = {k: [] for k in
                          ("len_m", "s_box_m", "corner_deg", "turn_1.5m_deg",
                           "turn_0.5m_deg", "max_curv_1/m", "leg1_m", "leg2_m")}
            atexit.register(self._dump_diag)

        # optional path logging for offline scoring / plots
        self._log_n = int(_f("STEER_LOG", 0))
        if self._log_n:
            n = self.num_envs
            self._log = {"root": np.zeros((self._log_n, n, 2), np.float32),
                         "box": np.zeros((self._log_n, n, 3), np.float32),
                         "episode": np.zeros((self._log_n, n), np.int32),
                         "held": np.zeros((self._log_n, n), np.uint8),
                         "arc": np.zeros((self._log_n, n), np.float32)}
            if self.steer_mrand > 0:
                # commanded speed per frame. success and app_xt cannot show
                # whether a speed command was obeyed -- a policy can track the
                # path perfectly at the wrong speed -- and a slow command also
                # costs episode time, so a delivery failure has two possible
                # causes. Logging the command separates them after the fact
                # instead of forcing a rerun.
                self._log["cmdv"] = np.zeros((self._log_n, n), np.float32)
                self._log["actv"] = np.zeros((self._log_n, n), np.float32)
            if self.steer_time:
                # tau[t] and the error against it. app_xt (perpendicular
                # distance) reads 0 for a body standing on the path three
                # seconds late, so it cannot score a timed command at all.
                self._log["tau"] = np.zeros((self._log_n, n, 2), np.float32)
                self._log["trk"] = np.zeros((self._log_n, n), np.float32)
            # STEER_DUMP_STATE=<k>: full pose of the first k envs, so a run can be
            # replayed later with no policy and no physics. Rendering while the
            # policy runs mixes two different stalls -- the character genuinely
            # standing still, and the viewer being starved by whatever else holds
            # the card -- and the video cannot tell them apart. Replay removes the
            # second one: the frames are already decided, only drawing is left.
            self._dump_state = int(os.environ.get("STEER_DUMP_STATE", "0"))
            if self._dump_state > 0:
                k = min(self._dump_state, n)
                self._dump_k = k
                nd = self._dof_pos.shape[-1]
                self._log["st_root"] = np.zeros((self._log_n, k, 13), np.float32)
                self._log["st_box"] = np.zeros((self._log_n, k, 13), np.float32)
                self._log["st_dof"] = np.zeros((self._log_n, k, nd), np.float32)
                self._log["st_dofv"] = np.zeros((self._log_n, k, nd), np.float32)
            self._log_i = 0
            self._episode_id = torch.zeros(n, dtype=torch.long, device=self.device)
            # The dumped path has to be the one that was in force during the
            # logged window. _gt_path keeps being regenerated after the buffer
            # fills, so dumping it directly pairs a trajectory with a path from
            # a later episode -- silently, and the score looks like bad tracking.
            #
            # Two slots, indexed by episode parity, because the last episode in
            # the window is often a stub: --eval resets every env together at
            # each repeat boundary, so keeping only the newest path can leave
            # nothing but a one-step episode to score against.
            # z0: the box height AT RESET. Reading it back out of the per-step
            # box log does not work -- the episode counter is bumped before the
            # box tensor is refreshed, so the boundary frame still holds the
            # previous episode's carried box (z~1.0), and keying off the pickup
            # transition instead mixes in re-grabs of dropped boxes (96 % on the
            # ground). Both readings were wrong. Recorded here it is unambiguous,
            # which is what lets success be split by "was the box on a platform".
            self._snap = [{"gt": torch.zeros(n, sp.V, 2, device=self.device),
                           "s_box": torch.zeros(n, device=self.device),
                           "final_tar": torch.zeros(n, 3, device=self.device),
                           "z0": torch.zeros(n, device=self.device),
                           "ep": torch.full((n,), -1, dtype=torch.long, device=self.device)}
                          for _ in range(2)]
            atexit.register(self._dump_log)
        self._prev_arc = torch.zeros(self.num_envs, device=self.device)
        # monotone arc-length trackers, one per subject. Scalars that only move
        # forward, so no boolean ever flips and nothing jumps at pickup.
        self._arc_root = torch.zeros(self.num_envs, device=self.device)
        self._arc_box = torch.zeros(self.num_envs, device=self.device)
        # frames since the grasp, for STEER_GRIP_RAMP
        self._lat_box = torch.zeros(self.num_envs, device=self.device)
        self._grip_t = torch.zeros(self.num_envs, device=self.device)
        # latched grasp state for STEER_HYST, refreshed once per step
        self._held_state = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._held_tick = -1

    def init_square_height_points(self):
        """The extra token carries the steering window, not a height map.

        Sizing runs through num_height_points, so claiming the window's width
        here makes every downstream size (obs buffer, tokenizer input) correct
        without touching the shared adapt scaffold.
        """
        k = int(_f("STEER_K", 6))
        if self.steer_time:
            self.num_height_points = 2 * k + 2              # tau[t] + K future
        elif bool(int(os.environ.get("STEER_DUAL", "0"))):
            self.num_height_points = 4 * k
        elif bool(int(os.environ.get("STEER_SLIM", "0"))):
            self.num_height_points = 2 * k + 2
        else:
            self.num_height_points = 2 * k
        return torch.zeros(1, self.num_height_points, 3, device=self.device
                           if hasattr(self, "device") else "cpu")

    def _create_ground_plane(self):
        """Flat ground: the terrain scaffold builds a trimesh even when the
        config asks for a plane, and none of it is used once terrain_obs is off."""
        self.terrain = _FlatTerrain(self.device, self.num_envs)
        Humanoid._create_ground_plane(self)

    def get_center_heights(self, root_states, env_ids=None):
        n = root_states.shape[0]
        return torch.zeros(n, self.center_height_points.shape[1], device=self.device)

    def _reset_task(self, env_ids):
        super()._reset_task(env_ids)
        if len(env_ids) == 0:
            return
        root = self._humanoid_root_states[env_ids, 0:2]
        box = self._box_states[env_ids, 0:2]
        tar = self._tar_pos[env_ids, 0:2]
        self._final_tar[env_ids] = self._tar_pos[env_ids].clone()
        if self._log_n:
            self._episode_id[env_ids] += 1
        self._prev_arc[env_ids] = 0.0
        self._arc_root[env_ids] = 0.0
        seed = self.steer_seed + self._steer_tick + int(env_ids[0])
        if self.steer_legacy:
            self._gt_path[env_ids] = sp.gen_gt(box, tar, self.steer_curvature, seed)
            self._s_box[env_ids] = 0.0
        else:
            if self.steer_traj in ("v2", "v3"):
                if self.steer_traj == "v3":
                    # v3 fixes the four shape knobs the user settled on. It is a
                    # separate name rather than new defaults so that re-evaluating
                    # a v2 checkpoint keeps drawing v2 paths -- changing a default
                    # under finished runs is what silently scored five v1 policies
                    # on v2 paths once already.
                    lat_lo, lat_hi = 0.0, _f("STEER_LAT_MAX", 2.2)
                    skew = _f("STEER_SKEW", 0.8)
                    hump2 = _f("STEER_HUMP2", 0.1)
                    turn = _f("STEER_TURN_MAX", 120.0)
                    spread = (_f("STEER_SPREAD_MIN", 0.85), _f("STEER_SPREAD_MAX", 1.8))
                    frac = _f("STEER_LAT_FRAC", 0.25)
                else:
                    lat_lo, lat_hi = self.steer_lat_min, self.steer_lat_max
                    skew, hump2 = self.steer_skew, self.steer_hump2
                    turn = self.steer_turn_max
                    spread = (self.steer_spread_min, self.steer_spread_max)
                    frac = self.steer_lat_frac
                path, s_box = sp.gen_full_v2(root, box, tar, seed, lat_lo, lat_hi,
                                             turn, p_two=hump2, skew=skew, spread=spread,
                                             lat_frac=frac)
            else:
                path, s_box = sp.gen_full(root, box, tar, self.steer_curvature, seed,
                                          modes=int(_f("STEER_MODES", 3)))
            self._gt_path[env_ids] = path
            self._s_box[env_ids] = s_box
            self._arc_box[env_ids] = s_box      # the box starts where the legs meet
        if self.steer_time:
            self._tau_clock[env_ids] = 0
            stops = None
            if self.steer_stops > 0:
                # Sample inside the REAL path, not the buffer. The buffer holds
                # 320 vertices (32 m) but a path ends at the target after 114 on
                # median, so a uniform draw over the buffer puts two thirds of
                # the stops in the padding past the target, where the episode
                # never reaches them: STEER_STOPS=1 would silently mean "a stop
                # in one episode out of three".
                gcur = torch.Generator(device="cpu"); gcur.manual_seed(seed)
                # Both legs: waiting before fetching the box is as real a command
                # as waiting while carrying it. 10-90 % keeps the stop off the
                # spawn and off the target, where putdown owns the last metre.
                tar_xy = self._tar_pos[env_ids, 0:2]
                end = (self._gt_path[env_ids] - tar_xy[:, None, :]).norm(dim=-1).argmin(dim=1)
                span = end.float().clamp(min=20.0)
                u = torch.rand((len(env_ids), self.steer_stops), generator=gcur).to(self.device)
                stops = ((0.1 + 0.8 * u) * span[:, None]).long()
            tar_xy = self._tar_pos[env_ids, 0:2]
            end_v = (self._gt_path[env_ids] - tar_xy[:, None, :]).norm(dim=-1).argmin(dim=1)
            self._tau[env_ids] = sp.build_tau(
                self._gt_path[env_ids], self._tau_H, self.dt,
                v_nom=self.steer_vnom, g_kappa=self.steer_vkappa,
                stops=stops, stop_len=self.steer_stoplen, end=end_v)
        if self.steer_mrand > 0:
            gcur = torch.Generator(device="cpu"); gcur.manual_seed(seed + 17)
            n = self.steer_mrand
            picks = torch.rand((len(env_ids), n), generator=gcur).to(self.device)
            mult = self.steer_m_lo + (1.0 - self.steer_m_lo) * (picks * 4).floor() / 3.0
            edge = torch.linspace(0, sp.V, n + 1, device=self.device).long()
            for j in range(n):
                self._mscale[env_ids[:, None], torch.arange(edge[j], edge[j + 1],
                    device=self.device)[None, :]] = mult[:, j:j + 1]
        self._grip_t[env_ids] = 0.0
        self._held_state[env_ids] = False
        if self._log_n and self._log_i < self._log_n:
            for slot in (0, 1):
                ids = env_ids[(self._episode_id[env_ids] % 2) == slot]
                if len(ids) == 0:
                    continue
                sn = self._snap[slot]
                sn["gt"][ids] = self._gt_path[ids]
                sn["s_box"][ids] = self._s_box[ids]
                sn["final_tar"][ids] = self._final_tar[ids]
                sn["z0"][ids] = self._box_states[ids, 2]
                sn["ep"][ids] = self._episode_id[ids]
        if self._diag_n:
            self._record_diag(env_ids)

    def _held(self):
        hands = self._rigid_body_pos[:, self._key_body_ids[[0, 1]], :].mean(dim=1)
        d = (hands - self._box_states[:, 0:3]).norm(dim=-1)
        if not self.steer_hyst:
            return d < 0.35
        # Latch: grab at 0.35, release at 0.45. _held is called several times a
        # step (obs, draw, log), so the state advances once per tick and every
        # caller inside that tick reads the same answer.
        if self._held_tick != self._steer_tick:
            self._held_state = torch.where(d < 0.35, True,
                                           torch.where(d > 0.45, False, self._held_state))
            self._held_tick = self._steer_tick
        return self._held_state

    def _track(self):
        """What the path is steering right now, and which leg it lives on.

        Before the box is picked up the commanded path is steering the character
        itself along the approach leg; afterwards it is steering the box along
        the transport leg. The bounds matter because a full-task path turns back
        on itself at the box: without them the carried box can project onto the
        approach leg and run its arc length backwards.
        """
        held = self._held()
        box_xy = self._box_states[:, 0:2]
        root_xy = self._humanoid_root_states[:, 0:2]
        if self.steer_grip_ramp > 0:
            # Same destination, spread over N frames: 0.376 m / 15 = 0.025 m per
            # frame, which is the scale the window already drifts at.
            w = (self._grip_t / self.steer_grip_ramp).clamp(max=1.0).unsqueeze(-1)
            xy = (1.0 - w) * root_xy + w * box_xy
        else:
            xy = torch.where(held.unsqueeze(-1), box_xy, root_xy)
        big = torch.full_like(self._s_box, sp.V * sp.DS)
        s_lo = torch.where(held, self._s_box, torch.zeros_like(self._s_box))
        s_hi = torch.where(held, big, self._s_box)
        if self.steer_legacy:
            return xy, held, None, None
        return xy, held, s_lo, s_hi

    def _record_diag(self, env_ids):
        """Log what each generated path asks for. Measurement only."""
        p = self._gt_path[env_ids]
        i_tar = (p - self._final_tar[env_ids, None, 0:2]).norm(dim=-1).argmin(dim=1)
        seg = (p[:, 1:] - p[:, :-1]).norm(dim=-1)
        keep = torch.arange(seg.shape[1], device=p.device)[None, :] < i_tar[:, None]
        st = sp.path_stats(p, self._s_box[env_ids])
        n = min(len(env_ids), max(1, self._diag_n // 64))
        self._diag["len_m"] += (seg * keep).sum(dim=1)[:n].tolist()
        self._diag["leg1_m"] += self._s_box[env_ids][:n].tolist()
        self._diag["leg2_m"] += ((seg * keep).sum(dim=1) - self._s_box[env_ids])[:n].tolist()
        for k in ("s_box_m", "corner_deg", "turn_1.5m_deg", "turn_0.5m_deg",
                  "max_curv_1/m"):
            self._diag[k] += st[k][:n].tolist()

    def _dump_diag(self):
        import json
        out = {}
        for k, v in self._diag.items():
            if not v:
                continue
            a = np.asarray(v, np.float64)
            out[k] = {"n": len(a), "mean": float(a.mean()), "p50": float(np.median(a)),
                      "p90": float(np.percentile(a, 90)), "max": float(a.max())}
        path = os.environ.get("STEER_DIAG_OUT", "/home/hwanhee/CVPR2027/runs/steer/diag.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print("[steer] path diagnostics ->", path)

    def _apply_probe(self):
        track_xy, held, s_lo, s_hi = self._track()
        s, _, _ = sp.project(track_xy, self._gt_path, s_lo, s_hi)
        point = sp.window(self._gt_path, s, 1, self.steer_lookahead)[:, 0]
        goal = torch.where(held.unsqueeze(-1), self._final_tar[:, 0:2],
                           self._box_states[:, 0:2])
        remaining = (track_xy - goal).norm(dim=-1)
        active = held & (remaining > self.steer_handoff)
        tar = self._final_tar.clone()
        tar[active, 0:2] = point[active]
        self._tar_pos[:] = tar

    def _compute_metrics_evaluation(self):
        if not self.steer_probe:
            return super()._compute_metrics_evaluation()
        saved = self._tar_pos
        self._tar_pos = self._final_tar
        try:
            super()._compute_metrics_evaluation()
        finally:
            self._tar_pos = saved

    def _tau_at(self, offs, env_ids=None):
        """tau at progress + each offset, clamped inside the buffer."""
        t = self._tau_clock if env_ids is None else self._tau_clock[env_ids]
        tau = self._tau if env_ids is None else self._tau[env_ids]
        idx = (t[:, None] + offs[None, :]).clamp(0, self._tau_H - 1).long()
        ar = torch.arange(tau.shape[0], device=self.device)[:, None]
        return tau[ar, idx]

    def _replan(self, env_ids):
        """Redraw the path from where the body actually is, and restart tau."""
        if len(env_ids) == 0:
            return
        held = self._held()[env_ids]
        root = self._humanoid_root_states[env_ids, 0:2]
        box = self._box_states[env_ids, 0:2]
        tar = self._tar_pos[env_ids, 0:2]
        # after the grasp the box is the thing being steered, so the plan starts
        # there; before it, the character still has to walk to the box
        start = torch.where(held[:, None], box, root)
        seed = self.steer_seed + 7919 * (self._steer_tick // max(self.steer_replan, 1))
        path, s_box = sp.gen_full_v2(
            start, box, tar, seed,
            0.0, _f("STEER_LAT_MAX", 2.2), _f("STEER_TURN_MAX", 120.0),
            p_two=_f("STEER_HUMP2", 0.1), skew=_f("STEER_SKEW", 0.8),
            spread=(_f("STEER_SPREAD_MIN", 0.85), _f("STEER_SPREAD_MAX", 1.8)),
            lat_frac=_f("STEER_LAT_FRAC", 0.25))
        self._gt_path[env_ids] = path
        self._s_box[env_ids] = s_box
        end_v = (path - tar[:, None, :]).norm(dim=-1).argmin(dim=1)
        # a replan carries the stops too. In deployment the planner emits the
        # pause as part of the trajectory it hands over -- a stop is not a
        # feature of the interface, it is just what the points look like -- so a
        # replan that drops them would quietly train "stops never happen".
        stops = None
        if self.steer_stops > 0:
            gcur = torch.Generator(device="cpu")
            gcur.manual_seed(int(seed) + 31)
            span = end_v.float().clamp(min=20.0)
            u = torch.rand((len(env_ids), self.steer_stops), generator=gcur).to(self.device)
            stops = ((0.1 + 0.8 * u) * span[:, None]).long()
        self._tau[env_ids] = sp.build_tau(
            path, self._tau_H, self.dt, v_nom=self.steer_vnom,
            g_kappa=self.steer_vkappa, stops=stops,
            stop_len=self.steer_stoplen, end=end_v)
        self._tau_clock[env_ids] = 0

    def _m_at(self, arc, env_ids=None):
        """Commanded window length at this arc position (metres)."""
        ms = self._mscale if env_ids is None else self._mscale[env_ids]
        j = (arc / sp.DS).long().clamp(0, sp.V - 1)
        ar = torch.arange(ms.shape[0], device=self.device)
        return self.steer_m_nom * ms[ar, j]

    def _steer_obs(self, root_states, env_ids=None):
        n = root_states.shape[0]
        if self.steer_time and not self.steer_zero:
            # tau[t] first: it IS the tracking error once the body's own xy is
            # subtracted, which is the only thing that tells the policy it is
            # running late. Time indexing has no self-correcting projection.
            offs = torch.arange(0, self.steer_k + 1, device=self.device) * self.steer_stride
            offs[0] = 0
            pts = self._tau_at(offs, env_ids)
            root_xy = root_states[:, 0:2]
            heading = torch.atan2(
                2.0 * (root_states[:, 6] * root_states[:, 5] + root_states[:, 3] * root_states[:, 4]),
                1.0 - 2.0 * (root_states[:, 4] ** 2 + root_states[:, 5] ** 2))
            return sp.to_local(pts, root_xy, heading).reshape(n, -1)
        width = 2 * self.steer_k + (2 if self.steer_slim else 0)
        if self.steer_dual:
            width = 4 * self.steer_k
        if self.steer_zero:
            return torch.zeros(n, width, device=self.device)

        if self.steer_slim:
            # One window, anchored on the character's own ratchet, plus the two
            # scalars the reward grades the box on. No grasp test, nothing
            # switches subject, and the observation now names exactly what the
            # penalty measures instead of leaving the policy to infer it.
            path = self._gt_path if env_ids is None else self._gt_path[env_ids]
            a_root = self._arc_root if env_ids is None else self._arc_root[env_ids]
            a_box = self._arc_box if env_ids is None else self._arc_box[env_ids]
            lat_b = self._lat_box if env_ids is None else self._lat_box[env_ids]
            pts = sp.window(path, a_root, self.steer_k, self.steer_spacing)
            root_xy = root_states[:, 0:2]
            heading = torch.atan2(
                2.0 * (root_states[:, 6] * root_states[:, 5] + root_states[:, 3] * root_states[:, 4]),
                1.0 - 2.0 * (root_states[:, 4] ** 2 + root_states[:, 5] ** 2))
            win = sp.to_local(pts, root_xy, heading).reshape(n, -1)
            return torch.cat([win, (a_box - a_root)[:, None], lat_b[:, None]], dim=-1)

        if self.steer_mrand > 0:
            # Same six points, same root-yaw frame as always. Only the length
            # they span changes, and that length IS the speed command.
            path = self._gt_path if env_ids is None else self._gt_path[env_ids]
            a_root = self._arc_root if env_ids is None else self._arc_root[env_ids]
            M = self._m_at(a_root, env_ids)
            root_xy = root_states[:, 0:2]
            heading = torch.atan2(
                2.0 * (root_states[:, 6] * root_states[:, 5] + root_states[:, 3] * root_states[:, 4]),
                1.0 - 2.0 * (root_states[:, 4] ** 2 + root_states[:, 5] ** 2))
            off = torch.arange(1, self.steer_k + 1, device=self.device)[None, :] / self.steer_k
            q = ((a_root[:, None] + M[:, None] * off) / sp.DS).clamp(0, sp.V - 2)
            lo = q.floor().long(); frac = (q - lo.float())[..., None]
            ar = torch.arange(path.shape[0], device=self.device)[:, None]
            pts = path[ar, lo] + frac * (path[ar, lo + 1] - path[ar, lo])
            return sp.to_local(pts, root_xy, heading).reshape(n, -1)

        if self.steer_dual:
            # One window per ratchet, no grasp test anywhere. The reward already
            # runs both scalars every step; this makes the observation read the
            # same two, so nothing switches subject and nothing jumps.
            # _arc_* are one step stale here (observations are computed before
            # rewards), which is 0.01 m -- and it guarantees obs and reward are
            # anchored to the identical number rather than two projections that
            # can disagree.
            path = self._gt_path if env_ids is None else self._gt_path[env_ids]
            a_root = self._arc_root if env_ids is None else self._arc_root[env_ids]
            a_box = self._arc_box if env_ids is None else self._arc_box[env_ids]
            pts = torch.cat([sp.window(path, a_root, self.steer_k, self.steer_spacing),
                             sp.window(path, a_box, self.steer_k, self.steer_spacing)],
                            dim=1)
            root_xy = root_states[:, 0:2]
            heading = torch.atan2(
                2.0 * (root_states[:, 6] * root_states[:, 5] + root_states[:, 3] * root_states[:, 4]),
                1.0 - 2.0 * (root_states[:, 4] ** 2 + root_states[:, 5] ** 2))
            return sp.to_local(pts, root_xy, heading).reshape(n, -1)

        track_xy, held, s_lo, s_hi = self._track()
        if env_ids is not None:
            track_xy, held = track_xy[env_ids], held[env_ids]
            s_lo = None if s_lo is None else s_lo[env_ids]
            s_hi = None if s_hi is None else s_hi[env_ids]
        path = self._gt_path if env_ids is None else self._gt_path[env_ids]
        if self.steer_sigma > 0:
            s0, _, _ = sp.project(track_xy, path, s_lo, s_hi)
            path = sp.synth_step(path, s0, self.steer_radius, self.steer_sigma,
                                 self.steer_seed * 100003 + self._steer_tick)
        s, _, _ = sp.project(track_xy, path, s_lo, s_hi)
        pts = sp.window(path, s, self.steer_k, self.steer_spacing)
        if not self.steer_approach:
            # the window is only defined once the box is in hand; before that the
            # channel is silent rather than pointing somewhere arbitrary
            pts = pts * held[:, None, None].to(pts.dtype)

        root_xy = root_states[:, 0:2]
        heading = torch.atan2(
            2.0 * (root_states[:, 6] * root_states[:, 5] + root_states[:, 3] * root_states[:, 4]),
            1.0 - 2.0 * (root_states[:, 4] ** 2 + root_states[:, 5] ** 2))
        return sp.to_local(pts, root_xy, heading).reshape(n, -1)

    def _compute_task_obs(self, env_ids=None):
        if self.steer_probe and env_ids is None:
            self._apply_probe()
        obs = super()._compute_task_obs(env_ids)
        root_states = (self._humanoid_root_states if env_ids is None
                       else self._humanoid_root_states[env_ids])
        return torch.cat([self._steer_obs(root_states, env_ids), obs], dim=-1)

    # ---- phase-free reward -------------------------------------------------

    def _ratchet(self, xy, arc):
        """Project onto the path, searching only forward of where we already were.

        A whole-task path doubles back at the box, so an unrestricted nearest
        point can put a carried box back on the approach leg and run the arc
        length backwards. A monotone scalar fixes that without any boolean: the
        search window is anchored to the previous value, and a small backward
        allowance lets the character recover from being knocked off course.
        """
        lo = arc - self.steer_back
        hi = arc + 2.0
        s, lat, tang = sp.project(xy, self._gt_path, lo, hi)
        s = torch.maximum(s, lo)
        return s, lat, tang

    def _lookahead(self, vel_xy, carry):
        """How far ahead to aim, in metres of arc.

        Fixed by default. STEER_L_GAIN turns on the textbook rule L = gain*speed,
        which matters here because carrying the box slows the character down: a
        fixed lookahead then aims proportionally further ahead exactly when the
        body is least able to get there. STEER_L_CARRY sets a separate value for
        the transport leg, which is loaded and less nimble than the approach.
        """
        base = self.steer_lookahead
        if carry and self.steer_l_carry > 0:
            base = self.steer_l_carry
        if self.steer_l_kappa > 0:
            # Local curvature over 1 m of arc, centred on the body's own
            # projection, in rad/m. Shorten L where the path bends.
            arc = self._arc_box if carry else self._arc_root
            d = 5.0 * sp.DS
            a = self._aim_pt(arc, -d)
            b = self._aim_pt(arc, torch.zeros_like(arc) + d) if torch.is_tensor(arc) \
                else self._aim_pt(arc, d)
            c = self._aim_pt(arc, torch.zeros_like(arc)) if torch.is_tensor(arc) \
                else self._aim_pt(arc, 0.0)
            v1 = c - a
            v2 = b - c
            cross = (v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
            dot = (v1 * v2).sum(dim=-1)
            kappa = torch.atan2(cross.abs(), dot) / (2.0 * d)
            return (base / (1.0 + self.steer_l_kappa * kappa)).clamp(0.3, base)
        if self.steer_l_gain <= 0:
            return base
        spd = vel_xy.norm(dim=-1)
        return (self.steer_l_gain * spd).clamp(self.steer_l_min, self.steer_l_max)

    def _aim_dir(self, xy, arc, look=None):
        """Unit vector from a body toward the point L further along the path.

        Aiming at a point ON the path, from the body's own position, is what
        makes this self-correcting: the further the body has drifted sideways,
        the more the direction tilts back toward the path. Aiming along the
        tangent instead would leave a parallel offset unpunished, which is why
        the tangent variant was the only S3 arm below its control.

        STEER_TANG blends that tangent back in, because the objection above no
        longer holds: latpen now supplies the lateral restoring force, so the
        velocity direction does not have to. What the chord DOES supply, and
        should not, is corner cutting -- a chord of length L across curvature k
        sits k*L^2/8 inside the arc, which at L=1.2 and 10-20 deg/m predicts
        0.047 m against a measured 0.051 m. Deviation on those sections is
        67-72 % inside the curve, and they are 31 % of the approach.

            w = 0    chord only (what every run so far used)
            w = 1    tangent only (shape from the path, position from latpen)

        The tangent is taken AT the aim point, not under the body: it is the
        direction the path is heading where we are aiming, which is what the
        velocity term is supposed to encode.
        """
        pt = self._aim_pt(arc, look)
        chord = torch.nn.functional.normalize(pt - xy, dim=-1)
        if self.steer_tang <= 0:
            return chord
        L = self.steer_lookahead if look is None else look
        d = 5.0 * sp.DS                      # 0.5 m of arc, centred on the aim point
        fwd = self._aim_pt(arc, (L + d) if not torch.is_tensor(L) else L + d)
        back = self._aim_pt(arc, (L - d) if not torch.is_tensor(L) else L - d)
        tang = torch.nn.functional.normalize(fwd - back, dim=-1)
        w = self.steer_tang
        return torch.nn.functional.normalize((1.0 - w) * chord + w * tang, dim=-1)

    def _aim_pt(self, arc, look=None):
        """The aim point itself, L further along the path."""
        L = self.steer_lookahead if look is None else look
        if torch.is_tensor(L):
            # per-env lookahead: one point each, at its own arc offset
            q = ((arc + L) / sp.DS).clamp(0, sp.V - 2)
            lo = q.floor().long()
            frac = (q - lo.float())[:, None]
            ar = torch.arange(self._gt_path.shape[0], device=self.device)
            return self._gt_path[ar, lo] + frac * (self._gt_path[ar, lo + 1] - self._gt_path[ar, lo])
        return sp.window(self._gt_path, arc, 1, L)[:, 0]

    def _compute_reward_time(self):
        """Same shipped reward, with 1.5 replaced by the commanded speed and the
        perpendicular distance replaced by the distance to tau[t].

        Two lines differ from TokenHSI:

            vel   exp(-5*(1.5 - u.v)^2)      ->  exp(-5*(|v_cmd| - u_cmd.v)^2)
            pos   (none) / c*(exp(-.5*lat^2)-1) -> c*(exp(-.5*|xy-tau_t|^2)-1)

        `lat` is the perpendicular distance to the curve, so standing on the
        path scores a perfect 0 no matter how late the body is; that is why a
        separate velocity term with a hardcoded 1.5 was needed at all. |xy-tau_t|
        carries both the sideways miss and the lateness, so one term covers what
        two used to, and the 1.5 disappears into the spacing of tau.

        With tau built at a uniform 1.5 m/s this is arithmetically the old
        reward -- that is the stage-1 regression.
        """
        root_pos = self._humanoid_root_states[..., 0:3]
        box_pos = self._box_states[..., 0:3]
        tar = self._tar_pos
        held = self._held()
        xy = torch.where(held[:, None], box_pos[:, 0:2], root_pos[:, 0:2])
        prev = torch.where(held[:, None], self._prev_box_pos[:, 0:2],
                           self._prev_root_pos[:, 0:2])
        vel = (xy - prev) / self.dt

        offs = torch.tensor([0, 1], device=self.device)
        pair = self._tau_at(offs)
        tau_t, tau_n = pair[:, 0], pair[:, 1]
        v_cmd = (tau_n - tau_t) / self.dt
        spd = v_cmd.norm(dim=-1)
        moving = spd > 0.05

        along = (F.normalize(v_cmd, dim=-1) * vel).sum(dim=-1)
        vel_r = torch.where(moving,
                            torch.exp(-5.0 * (spd - along) ** 2),
                            torch.exp(-5.0 * vel.pow(2).sum(dim=-1)))
        vel_r = torch.where(moving & (along <= 0), torch.zeros_like(vel_r), vel_r)

        e = (xy - tau_t).pow(2).sum(dim=-1)
        self._trk_err = e.sqrt()
        steer_r = self.steer_pos_c * (torch.exp(-0.5 * e) - 1.0)

        diff = tar - box_pos
        pos_near = torch.exp(-10.0 * (diff ** 2).sum(dim=-1))

        hands = self._rigid_body_pos[:, self._key_body_ids[[0, 1]]].mean(dim=1)
        handheld = torch.exp(-5.0 * ((hands - box_pos) ** 2).sum(dim=-1))
        handheld[((box_pos[:, 0:2] - root_pos[:, 0:2]) ** 2).sum(dim=-1) > 0.7 ** 2] = 0.0

        putdown = (torch.abs(box_pos[:, -1] - tar[:, -1]) <= 0.001).float()
        putdown[(diff[:, 0:2] ** 2).sum(dim=-1) > 0.1 ** 2] = 0.0

        # 0.8 and 4c match the shipped ceiling: the old reward paid 0.2*vel on
        # each of two legs at weight 2, and one steer term per leg at the same
        # weight. One tracked body now, so the coefficients fold together.
        self.rew_buf[:] = (0.8 * vel_r + 4.0 * steer_r + 0.4 * pos_near
                           + 0.2 * handheld + 0.2 * putdown)
        if self.steer_grabpin > 0:
            near = ((box_pos[:, 0:2] - root_pos[:, 0:2]) ** 2).sum(dim=-1) < self.steer_grabpin ** 2
            wait = near & (~held)
        else:
            wait = torch.zeros_like(held)
        self._tau_clock = torch.where(wait, self._tau_clock,
                                      (self._tau_clock + 1).clamp(max=self._tau_H - 2))
        if self.steer_resync > 0:
            due = (self._tau_clock % self.steer_resync == 0) & (self._tau_clock > 0)
            if due.any():
                ids = torch.nonzero(due, as_tuple=False).flatten()
                # Search only a window around the current clock. A whole-path
                # argmin looks correct and is not: the path doubles back at the
                # box, so a body carrying the box past it matches a point on the
                # approach leg just as well, and the clock jumps seconds
                # backwards. The shipped code kept a monotone arc ratchet for
                # exactly this. Rewinding is the point here -- that is how the
                # lag resets -- so the window is wide behind and narrow ahead,
                # rather than monotone.
                back, fwd = 3 * self.steer_resync, self.steer_resync
                lo = (self._tau_clock[ids] - back).clamp(min=0)
                span = back + fwd + 1
                off = torch.arange(span, device=self.device)[None, :]
                idxw = (lo[:, None] + off).clamp(max=self._tau_H - 2)
                ar = torch.arange(len(ids), device=self.device)[:, None]
                d = (self._tau[ids][ar, idxw] - xy[ids][:, None, :]).norm(dim=-1)
                self._tau_clock[ids] = idxw[ar[:, 0], d.argmin(dim=1)]
        if self.steer_replan > 0:
            due = (self._tau_clock >= self.steer_replan)
            if due.any():
                self._replan(torch.nonzero(due, as_tuple=False).flatten())
        if self._power_reward:
            power = torch.abs(self.dof_force_tensor[:, self._power_dof_ids]
                              * self._dof_vel[:, self._power_dof_ids]).sum(dim=-1)
            self.rew_buf -= self._power_coefficient * power

    def _compute_reward_phasefree(self):
        """Rebuild the shipped carry reward with the two aim directions swapped.

        Nothing is added and nothing is removed: walk_r still rewards the root
        for closing on the box and moving at 1.5 m/s, carry_r still rewards the
        box for closing on the target, both still saturate inside 0.5 m of their
        own goal, and handheld_r / putdown_r are untouched. Only the direction
        each velocity term measures against changes, from "straight at the goal"
        to "at the next point on the commanded path". The ceiling is 1.2 with
        onlyVelReward on, 1.6 with it off.

        STEER_POS is the one thing that does add: a second pos term aimed at the
        steering goal, which lifts the ceiling to 1.6 + 4*STEER_POS_C. That makes
        its runs incomparable by raw return, so they are judged on success rate
        and tracking like every other arm.
        """
        if self.steer_time:
            self._compute_reward_time()
            return
        root_pos = self._humanoid_root_states[..., 0:3]
        box_pos = self._box_states[..., 0:3]
        tar = self._tar_pos

        self._arc_root, lat_root, _ = self._ratchet(root_pos[:, 0:2], self._arc_root)
        self._arc_box, lat_box, _ = self._ratchet(box_pos[:, 0:2], self._arc_box)
        self._lat_box = lat_box     # STEER_SLIM feeds this straight to the policy
        root_vel = (root_pos - self._prev_root_pos)[:, 0:2] / self.dt
        box_vel3 = (box_pos - self._prev_box_pos) / self.dt
        l_root = self._lookahead(root_vel, carry=False)
        l_box = self._lookahead(box_vel3[:, 0:2], carry=True)
        if self.steer_mrand > 0:
            # M at each body's own ratchet: the walk leg reads the character's,
            # the carry leg the box's, exactly as the two legs already split.
            # L = 1.2 m was half the 2.4 m window, so it stays half of whatever
            # the window now spans -- aim and speed remain one command.
            m_root = self._m_at(self._arc_root)
            m_box = self._m_at(self._arc_box)
            v_root, v_box = m_root / 1.6, m_box / 1.6
            l_root, l_box = m_root * 0.5, m_box * 0.5
        else:
            v_root = v_box = None
        if self.steer_vel == "native":
            u_root = torch.nn.functional.normalize(
                box_pos[:, 0:2] - root_pos[:, 0:2], dim=-1)
            u_box = torch.nn.functional.normalize(tar[:, 0:2] - box_pos[:, 0:2], dim=-1)
        else:
            u_root = self._aim_dir(root_pos[:, 0:2], self._arc_root, l_root)
            u_box = self._aim_dir(box_pos[:, 0:2], self._arc_box, l_box)

        def vel_term(vel_xy, direction, pin, v_tar=None):
            """Shipped form, with 1.5 replaced by the commanded speed when there
            is one. v_tar = M / 1.6 s: the window spans M metres and the policy
            has always had 1.6 s of horizon, so a shorter window is a slower
            command and nothing else in the reward has to know about it.

            At v_tar = 0 the along-track projection is meaningless (there is no
            direction to be along), so the term degenerates to punishing any
            motion at all -- which is what "stop" means."""
            if v_tar is None:
                v_tar = torch.full_like(vel_xy[:, 0], 1.5)
            along = (direction * vel_xy).sum(dim=-1)
            r = torch.exp(-5.0 * (v_tar - along) ** 2)
            r[along <= 0] = 0.0
            stopped = v_tar < 0.05
            if stopped.any():
                r = torch.where(stopped, torch.exp(-5.0 * vel_xy.pow(2).sum(dim=-1)), r)
            r[pin] = 1.0
            return r

        def steer_pos_term(xy, arc, look, lat, pin):
            """The shipped pos form measured against the steering goal.

            Same exp(-0.5*err) and same pin as the native term beside it; only
            the goal differs. "lat" is the same measurement with the lookahead
            floor removed, so it spans the full 0..1 where "aim" is capped at
            exp(-0.5*L^2) -- half the sensitivity to a metre of deviation.
            """
            if self.steer_pos == "aim":
                err = ((self._aim_pt(arc, look) - xy) ** 2).sum(dim=-1)
            else:
                err = lat ** 2
            r = torch.exp(-0.5 * err)
            if self.steer_pos == "latpen":
                # Same shape as the shipped pos term, written as a penalty:
                # 0 on the path, -1 far off. An ADDITIVE term cannot win here --
                # raising its coefficient raises the episode return, run-to-run
                # spread stays about 1 % of that return, so signal and noise grow
                # together and the ratio tops out near 2.4 no matter how large c
                # gets. Subtracting a constant removes the inflation and the
                # ratio becomes linear in c again.
                r = r - 1.0
            r[pin] = 1.0 if self.steer_pos != "latpen" else 0.0
            return r

        # walk_r: the root along the approach leg
        pos_err = ((box_pos[:, 0:2] - root_pos[:, 0:2]) ** 2).sum(dim=-1)
        pin_walk = pos_err < 0.5 ** 2
        pos_walk = torch.exp(-0.5 * pos_err)
        pos_walk[pin_walk] = 1.0
        # onlyVelReward drops the position term and doubles the velocity one, in
        # both legs. It is True in the shipped carry task and in the stage-1
        # backbone; ground2terrain is the only config that turns it off, because
        # on slopes the character cannot reach 1.5 m/s and the velocity term
        # collapses to 0.018, so the position term was there to cover for it.
        if self._only_vel_reward:
            walk_r = 0.2 * vel_term(root_vel, u_root, pin_walk, v_root)
        else:
            walk_r = 0.1 * pos_walk + 0.1 * vel_term(root_vel, u_root, pin_walk, v_root)
        if self.steer_pos != "off":
            walk_r = walk_r + self.steer_pos_c * steer_pos_term(
                root_pos[:, 0:2], self._arc_root, l_root, lat_root,
                pos_err < self.steer_pin ** 2)

        # carry_r: the box along the transport leg
        diff = tar - box_pos
        err_xy = (diff[:, 0:2] ** 2).sum(dim=-1)
        pin_carry = err_xy < 0.5 ** 2
        pos_far = torch.exp(-0.5 * err_xy)
        pos_far[pin_carry] = 1.0
        pos_near = torch.exp(-10.0 * (diff ** 2).sum(dim=-1))
        box_vel = box_vel3[:, 0:2]
        if self._only_vel_reward:
            carry_r = 0.2 * vel_term(box_vel, u_box, pin_carry, v_box) + 0.2 * pos_near
        else:
            carry_r = (0.1 * pos_far + 0.1 * vel_term(box_vel, u_box, pin_carry, v_box)
                       + 0.2 * pos_near)
        if self.steer_pos != "off":
            carry_r = carry_r + self.steer_pos_c * steer_pos_term(
                box_pos[:, 0:2], self._arc_box, l_box, lat_box,
                err_xy < self.steer_pin ** 2)
        if self._box_vel_penalty:
            # shipped as "to avoid stiff grasping": punish flinging the box
            thre = self._box_vel_pen_thre
            spd = box_vel3.norm(dim=-1).clamp(min=thre)
            carry_r = carry_r - self._box_vel_pen_coeff * (1.0 - torch.exp(-2.0 * (thre - spd) ** 2))

        hands = self._rigid_body_pos[:, self._key_body_ids[[0, 1]]].mean(dim=1)
        handheld = torch.exp(-5.0 * ((hands - box_pos) ** 2).sum(dim=-1))
        handheld[((box_pos[:, 0:2] - root_pos[:, 0:2]) ** 2).sum(dim=-1) > 0.7 ** 2] = 0.0
        handheld_r = 0.2 * handheld

        putdown = (torch.abs(box_pos[:, -1] - tar[:, -1]) <= 0.001).float()
        putdown[err_xy > 0.1 ** 2] = 0.0
        putdown_r = 0.2 * putdown

        self.rew_buf[:] = 2.0 * walk_r + 2.0 * carry_r + handheld_r + putdown_r
        if self._power_reward:
            power = torch.abs(self.dof_force_tensor[:, self._power_dof_ids]
                              * self._dof_vel[:, self._power_dof_ids]).sum(dim=-1)
            self.rew_buf -= self._power_coefficient * power
        if self.steer_xtrack > 0:
            if self.steer_xtrack_pf:
                # Same penalty, without the last held boolean in this path: the
                # root owns leg 1 and the box owns leg 2, exactly as the walk and
                # carry terms above already split. Paying both at once costs at
                # most 0.2 more, and during transport the root is inside 0.5 m of
                # the box so lat_root and lat_box agree to within that.
                pen = (torch.exp(-2.0 * lat_root ** 2) - 1.0) \
                    + (torch.exp(-2.0 * lat_box ** 2) - 1.0)
                self.rew_buf += 0.1 * self.steer_xtrack * pen
            else:
                lat = torch.where(self._held(), lat_box, lat_root)
                self.rew_buf += 0.2 * self.steer_xtrack * (torch.exp(-2.0 * lat ** 2) - 1.0)
        if self.steer_heading > 0:
            # Stanley's heading term. Velocity can point along the path while the
            # body faces elsewhere -- a humanoid side-steps -- and that costs turn
            # authority on the next corner.
            q = self._humanoid_root_states
            yaw = torch.atan2(2.0 * (q[:, 6] * q[:, 5] + q[:, 3] * q[:, 4]),
                              1.0 - 2.0 * (q[:, 4] ** 2 + q[:, 5] ** 2))
            face = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
            cos_err = (face * u_root).sum(dim=-1).clamp(-1.0, 1.0)
            self.rew_buf += 0.2 * self.steer_heading * (
                torch.exp(-2.0 * torch.acos(cos_err) ** 2) - 1.0)

    # ---- re-aimed reward ---------------------------------------------------


    def _verify_phasefree(self, actions):
        """Prove the rebuilt reward IS the shipped one when the aim is unchanged.

        Eighty lines of reimplemented reward is exactly the kind of thing that
        looks right and is quietly off by a pinning band or a coefficient. Point
        the aim at each term's native goal and the two must agree to float noise;
        any gap is a transcription bug, not a design choice.
        """
        super()._compute_reward(actions)
        ref = self.rew_buf.clone()
        aim, pos = self._aim_dir, self.steer_pos
        xt, hd = self.steer_xtrack, self.steer_heading
        # Every term that ADDS to the shipped reward has to be off for this
        # identity to hold -- what is under test is the aim swap, nothing else.
        # Leaving xtrack on made the check report a 2.0e-01 mismatch, which is
        # exactly its own maximum, so the alarm was the check's own fault.
        self.steer_pos = "off"
        self.steer_xtrack = 0.0
        self.steer_heading = 0.0
        self._aim_dir = lambda xy, arc, look=None: torch.nn.functional.normalize(
            (self._tar_pos[:, 0:2] if arc is self._arc_box else self._box_states[:, 0:2]) - xy, dim=-1)
        self._compute_reward_phasefree()
        self._aim_dir, self.steer_pos = aim, pos
        self.steer_xtrack, self.steer_heading = xt, hd
        d = (self.rew_buf - ref).abs()
        print(f"[verify] phase-free vs shipped: max {d.max():.3e}  mean {d.mean():.3e}  "
              f"{'MATCH' if d.max() < 1e-4 else 'MISMATCH'}", flush=True)

    def _hold_probe(self):
        """STEER_HOLD_PROBE=1 : "제자리에 있어라" 명령을 얼마나 지키는지 잰다.

        success(목표 도달)는 이 실험에서 지표가 아니다 -- 서 있으라고 했으니 도달하면 안 된다.
        재야 할 것은 (1) 안 움직였나 (2) 상자를 계속 들고 있나 (3) 안 넘어졌나 다.
        """
        import os as _o
        if not _o.environ.get("STEER_HOLD_PROBE"):
            return
        h = self._humanoid_root_states
        if not hasattr(self, "_hp0"):
            self._hp0 = h[:, 0:2].clone()
            self._hpb0 = self._box_states[:, 2].clone()
            self._hpn = 0
            return
        self._hpn += 1
        if self._hpn == 300:
            import numpy as _np
            disp = torch.norm(h[:, 0:2] - self._hp0, dim=-1).cpu().numpy()
            bz = self._box_states[:, 2].cpu().numpy()
            rz = h[:, 2].cpu().numpy()
            print(f"[hold] 300스텝 후  이동거리 중앙 {_np.median(disp):.2f} m  "
                  f"(<0.5m {100*(disp<0.5).mean():.0f}%  <1m {100*(disp<1.0).mean():.0f}%)", flush=True)
            print(f"[hold]   상자높이 중앙 {_np.median(bz):.2f} m (처음 {_np.median(self._hpb0.cpu().numpy()):.2f})  "
                  f"사람높이 중앙 {_np.median(rz):.2f} m  (0.5 미만이면 넘어짐 {100*(rz<0.5).mean():.0f}%)", flush=True)
            raise SystemExit(0)
        return

    def _compute_reward(self, actions):
        self._hold_probe()
        if os.environ.get("STEER_PF_VERIFY") and self._steer_tick % 60 == 0:
            self._verify_phasefree(actions)
            return
        if self.steer_phasefree and not self.steer_zero:
            self._compute_reward_phasefree()
            return
        super()._compute_reward(actions)
        if self.steer_zero or (not self.steer_reaim and self.steer_xtrack <= 0
                               and not self.steer_progress):
            return

        box_pos = self._box_states[..., 0:3]
        root_xy = self._humanoid_root_states[:, 0:2]
        tar = self._tar_pos
        track_xy, held, s_lo, s_hi = self._track()
        if not self.steer_approach:
            held = torch.ones_like(held)
            track_xy, s_lo, s_hi = box_pos[:, 0:2], None, None

        # Both phases have the same shape: a native reward term drives something
        # straight at a goal, and we swap that direction for the commanded path.
        #   approach   walk_r's velocity term    root -> box
        #   transport  carry_r's velocity term   box  -> final target
        # Both are weighted 0.1 inside their reward and both rewards are scaled
        # by 2.0, so either swap costs exactly 0.2 and the ceiling is unchanged.
        COEF = 2.0 * 0.1
        native_goal = torch.where(held.unsqueeze(-1), tar[:, 0:2], box_pos[:, 0:2])
        native_dir = torch.nn.functional.normalize(native_goal - track_xy, dim=-1)
        remaining = (native_goal - track_xy).norm(dim=-1)

        # aim at a point ON the path, not along its tangent: the tangent has no
        # restoring force, so a box running parallel 2 m off the path would score
        # full marks. This is the pure-pursuit rule the no-training probe used.
        arc, lat, tang = sp.project(track_xy, self._gt_path, s_lo, s_hi)
        if self.steer_aim == "tangent":
            path_dir = tang
        else:
            look = sp.window(self._gt_path, arc, 1, self.steer_lookahead)[:, 0]
            path_dir = torch.nn.functional.normalize(look - track_xy, dim=-1)
        # hand the direction back to the native goal for the last stretch of each
        # leg, so steering never fights the grasp or the placement
        w = ((remaining - self.steer_handoff) / 0.5).clamp(0.0, 1.0).unsqueeze(-1)
        u = torch.nn.functional.normalize(w * path_dir + (1 - w) * native_dir, dim=-1)

        if self.steer_xtrack > 0:
            # Pay for this out of a term that is already carrying no gradient.
            # While carrying, walk_r's velocity term is pinned at 1.0 (the root is
            # always within 0.5 m of the box it holds) so its 0.2 is dead weight.
            # During the approach it is carry_r's velocity term that is dead: the
            # box has not moved, so its speed toward the target is zero. Either
            # way the swap is free and the ceiling does not move.
            on_path = torch.exp(-2.0 * lat ** 2)
            pinned = ((box_pos[:, 0:2] - root_xy) ** 2).sum(dim=-1) < 0.5 ** 2
            spendable = torch.where(held, pinned, ~pinned)
            self.rew_buf += COEF * self.steer_xtrack * spendable.float() * (on_path - 1.0)

        vel_src = torch.where(held.unsqueeze(-1),
                              (box_pos - self._prev_box_pos)[:, 0:2],
                              (self._humanoid_root_states[:, 0:3] - self._prev_root_pos)[:, 0:2])
        vel = vel_src / self.dt

        if self.steer_progress:
            ds = (arc - self._prev_arc).clamp(min=0.0)
            self._prev_arc = arc
            speed = ds / self.dt
            prog = torch.exp(-5.0 * (1.5 - speed) ** 2)
            prog[speed <= 0] = 0.0
            prog[remaining < 0.5] = 1.0
            along_native = (native_dir * vel).sum(dim=-1)
            old = torch.exp(-5.0 * (1.5 - along_native) ** 2)
            old[along_native <= 0] = 0.0
            old[remaining < 0.5] = 1.0
            self.rew_buf += COEF * (prog - old)
            return

        if not self.steer_reaim:
            return

        def vel_term(direction):
            along = (direction * vel).sum(dim=-1)
            r = torch.exp(-5.0 * (1.5 - along) ** 2)
            r[along <= 0] = 0.0
            r[remaining < 0.5] = 1.0  # native pinning band, as shipped
            return r

        # swap the shipped goal-aimed velocity term for the path-aimed one:
        # same weight, same algebra, only the direction changes
        self.rew_buf += COEF * (vel_term(u) - vel_term(native_dir))

    def _compute_reset(self):
        super()._compute_reset()
        if self.steer_faildist > 0:
            # TokenHSI's traj task ends an episode once the character is further
            # than failDist from its commanded point. Past that the rollout can
            # no longer produce a useful gradient, so the samples are wasted.
            _, lat_r, _ = sp.project(self._humanoid_root_states[:, 0:2], self._gt_path,
                                     self._arc_root - self.steer_back, self._arc_root + 2.0)
            self.reset_buf[lat_r.abs() > self.steer_faildist] = 1
        if not self.steer_terminate_on_success:
            return
        err = (self._box_states[..., 0:3] - self._tar_pos).norm(dim=-1)
        self._success_steps = torch.where(err < 0.2, self._success_steps + 1,
                                          torch.zeros_like(self._success_steps))
        done = self._success_steps >= 10
        self.reset_buf[done] = 1
        self._success_steps[done] = 0

    def _draw_task(self):
        """Replace the shipped task lines with just the two paths.

        The stock overlay (box bounding box, box->target, human->box) adds about
        thirty lines per env and buries the paths, so only the markers are kept
        from it. Set STEER_DRAW_TASK=1 to get the original lines back.

        Magenta band on the ground: the whole path the world model asked for,
        cut at the placement target so it ends where the box should land.
        Cyan band at root height: the K points actually fed to the policy.
        Yellow post: the point the reward is aiming at, drawn floor to root.

        add_lines has no width, so a band is many parallel hairlines packed
        close enough to merge. Widths are in metres and tunable:
        STEER_GT_WIDTH (default 0.30), STEER_WIN_WIDTH (default 0.16).
        """
        if os.environ.get("STEER_DRAW_TASK"):
            super()._draw_task()
            if self.viewer is None or self.steer_zero:
                return
        else:
            self._update_marker()
            if self.viewer is None:
                return
            self.gym.clear_lines(self.viewer)
            if self.steer_zero:
                return
        import numpy as np

        track_xy, _, s_lo, s_hi = self._track()
        arc, _, _ = sp.project(track_xy, self._gt_path, s_lo, s_hi)
        win = sp.window(self._gt_path, arc, self.steer_k, self.steer_spacing)
        aim = sp.window(self._gt_path, arc, 1, self.steer_lookahead)[:, 0]
        root_z = self._humanoid_root_states[:, 2].cpu().numpy()

        gt_all = self._gt_path.cpu().numpy()
        # cut at the real placement target, not _tar_pos: probe mode overwrites
        # _tar_pos with a lookahead point every step
        tar_xy = self._final_tar[:, 0:2].cpu().numpy()
        win_np = win.cpu().numpy()
        aim_np = aim.cpu().numpy()

        gt_w = _f("STEER_GT_WIDTH", 0.30)
        win_w = _f("STEER_WIN_WIDTH", 0.16)

        def band(pts, z, width, spacing=0.02):
            """One path drawn as a solid band of parallel hairlines."""
            n = max(int(width / spacing), 3)
            a, b = pts[:-1], pts[1:]
            d = b - a
            nrm = np.stack([-d[:, 1], d[:, 0]], axis=-1)
            nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-6)
            out = []
            for off in np.linspace(-width / 2, width / 2, n):
                aa, bb = a + nrm * off, b + nrm * off
                out.append(np.concatenate([
                    aa, np.full((len(aa), 1), z),
                    bb, np.full((len(bb), 1), z)], axis=1))
            return np.concatenate(out).astype(np.float32)

        def col(rgb, n):
            return np.tile(np.array([rgb], np.float32), (n, 1))

        for i, env_ptr in enumerate(self.envs):
            end = int(np.linalg.norm(gt_all[i] - tar_xy[i], axis=-1).argmin()) + 1
            g = gt_all[i][:max(end, 2)][::2]
            v = band(g, 0.04, gt_w)
            self.gym.add_lines(self.viewer, env_ptr, len(v), v, col([1.00, 0.15, 0.65], len(v)))

            zw = float(root_z[i])
            v = band(win_np[i], zw, win_w)
            self.gym.add_lines(self.viewer, env_ptr, len(v), v, col([0.10, 0.95, 1.00], len(v)))

            a = aim_np[i]
            off = np.linspace(-0.04, 0.04, 9)[:, None]
            v = np.tile(np.array([[a[0], a[1], 0.02, a[0], a[1], zw]], np.float32), (9, 1))
            v[:, [0, 3]] += off
            self.gym.add_lines(self.viewer, env_ptr, len(v), v, col([1.0, 0.90, 0.10], len(v)))

    def _replay_load(self):
        z = np.load(self._replay_path)
        k = min(z["st_root"].shape[1], self.num_envs)
        self._replay = {"i": 0, "k": k, "T": z["st_root"].shape[0],
                        "root": torch.as_tensor(z["st_root"][:, :k], device=self.device),
                        "box": torch.as_tensor(z["st_box"][:, :k], device=self.device),
                        "dof": torch.as_tensor(z["st_dof"][:, :k], device=self.device),
                        "dofv": (torch.as_tensor(z["st_dofv"][:, :k], device=self.device)
                                 if "st_dofv" in z else None)}
        print(f"[replay] {self._replay_path}  {self._replay['T']} frames x {k} envs", flush=True)

    def _replay_step(self):
        """Write one recorded frame into the sim, BEFORE physics runs.

        IsaacGym's state setters are deferred -- they take effect at the next
        simulate(), not immediately -- and the render for a step happens before
        that step's physics. Writing in post_physics_step therefore showed the
        physics result of stepping FROM the recording with mismatched velocities,
        not the recording: the PD controller fought a pose it had zero velocity
        for and the character shook and sank into the box.

        Written here instead, and with the PD target pinned to the recorded pose,
        the controller holds what was recorded rather than chasing the policy.

        Why replay at all: a video captured while the policy runs holds two
        different stalls and cannot separate them -- the character genuinely
        standing still (the carry leg pins inside 0.5 m of the target, and
        putdown needs the box within 1 mm of target height, so stopping is
        correct there) and the viewer starving because something else holds the
        card. Here the frames are already decided.
        """
        r = self._replay
        i = r["i"] % r["T"]
        r["i"] += 1
        k = r["k"]
        self._humanoid_root_states[:k] = r["root"][i]
        self._box_states[:k] = r["box"][i]
        self._dof_pos[:k] = r["dof"][i]
        if r["dofv"] is not None:
            self._dof_vel[:k] = r["dofv"][i]
        else:
            self._dof_vel[:k] = 0.0
        ids = torch.cat([self._humanoid_actor_ids[:k],
                         self._box_actor_ids[:k]]).to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(ids), len(ids))
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(self._humanoid_actor_ids[:k].to(torch.int32)), k)
        if getattr(self, "_pd_control", False):
            tar = self._dof_pos.clone()
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(tar))

    def pre_physics_step(self, actions):
        if getattr(self, "_replay_path", ""):
            if self._replay is None:
                self._replay_load()
            self.actions = actions.to(self.device).clone()
            self._replay_step()
            return
        return super().pre_physics_step(actions)

    def post_physics_step(self):
        super().post_physics_step()
        if getattr(self, "_replay_path", ""):
            self.reset_buf[:] = 0
            self._terminate_buf[:] = 0
            self.progress_buf[:] = 0
            return
        if self._log_n and self._log_i < self._log_n:
            i = self._log_i
            self._log["root"][i] = self._humanoid_root_states[:, 0:2].cpu().numpy()
            self._log["box"][i] = self._box_states[:, 0:3].cpu().numpy()
            self._log["episode"][i] = self._episode_id.cpu().numpy()
            txy, hld, lo, hi = self._track()
            self._log["held"][i] = hld.cpu().numpy()
            self._log["arc"][i] = sp.project(txy, self._gt_path, lo, hi)[0].cpu().numpy()
            if self.steer_mrand > 0:
                hld = self._held()
                arc = torch.where(hld, self._arc_box, self._arc_root)
                self._log["cmdv"][i] = (self._m_at(arc) / 1.6).cpu().numpy()
                cur = torch.where(hld[:, None], self._box_states[:, 0:2],
                                  self._humanoid_root_states[:, 0:2])
                prev = torch.where(hld[:, None], self._prev_box_pos[:, 0:2],
                                   self._prev_root_pos[:, 0:2])
                self._log["actv"][i] = ((cur - prev).norm(dim=-1) / self.dt).cpu().numpy()
            if self.steer_time:
                self._log["tau"][i] = self._tau_at(
                    torch.zeros(1, dtype=torch.long, device=self.device))[:, 0].cpu().numpy()
                trk = getattr(self, "_trk_err", None)
                if trk is None:
                    trk = torch.zeros(self.num_envs, device=self.device)
                self._log["trk"][i] = trk.cpu().numpy()
            if getattr(self, "_dump_state", 0) > 0:
                k = self._dump_k
                self._log["st_root"][i] = self._humanoid_root_states[:k].cpu().numpy()
                self._log["st_box"][i] = self._box_states[:k].cpu().numpy()
                self._log["st_dof"][i] = self._dof_pos[:k].cpu().numpy()
                self._log["st_dofv"][i] = self._dof_vel[:k].cpu().numpy()
            self._log_i += 1
        if self.steer_grip_ramp > 0:
            h = self._held()
            self._grip_t = torch.where(h, self._grip_t + 1.0,
                                       torch.zeros_like(self._grip_t))
        self._steer_tick += 1

    def _dump_log(self):
        if not self._log_n or self._log_i == 0:
            return
        out = os.environ.get("STEER_OUT", "/home/hwanhee/CVPR2027/runs/steer")
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, os.environ.get("STEER_TAG", "trained") + ".npz")
        n = self._log_i
        snaps = {f"{k}{i}": self._snap[i][k].cpu().numpy()
                 for i in (0, 1) for k in ("gt", "s_box", "final_tar", "z0", "ep")}
        np.savez_compressed(path, **snaps,
                            gt=self._snap[1]["gt"].cpu().numpy(),
                            final_tar=self._snap[1]["final_tar"].cpu().numpy(),
                            s_box=self._snap[1]["s_box"].cpu().numpy(),
                            legacy=self.steer_legacy, approach=self.steer_approach,
                            lookahead=self.steer_lookahead, handoff=self.steer_handoff,
                            curvature=self.steer_curvature, radius=self.steer_radius,
                            sigma=self.steer_sigma, enabled=True,
                            **{k: v[:n] for k, v in self._log.items()})
        print(f"[steer] wrote {path}  steps={n}", flush=True)