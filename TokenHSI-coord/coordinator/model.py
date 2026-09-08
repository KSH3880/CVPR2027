"""K-query joint trajectory coordinator with smooth acceleration output."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Union

import torch
from torch import nn

from .geometry import (
    TOKEN_DIM,
    build_bezier_paths,
    build_waypoint_paths,
    conflict_window_profile,
    direct_speed_profile,
    integrate_speed,
    physical_point_speed_profile,
    point_speed_profile,
    slowdown_window_profile,
    shared_to_world,
    state_to_tokens,
)
from .schema import (
    ACCEL_KNOTS, AGENTS, CANDIDATES, MAX_SPEED, MIN_SPEED,
    PATH_POINTS, WAYPOINT_RESIDUAL_POINTS, CoordinatorState,
)
from .planner import crossing_arrival_metrics


MLP_CONFLICT_FEATURES = 7


@dataclass(frozen=True)
class CoordinatorConfig:
    token_dim: int = TOKEN_DIM
    d_model: int = 128
    nhead: int = 4
    encoder_layers: int = 3
    decoder_layers: int = 2
    feedforward: int = 256
    dropout: float = 0.0
    candidates: int = CANDIDATES
    residual_scale: float = 3.0
    # Start exactly from the strong analytic executor contract: straight pickup
    # approach, max nominal speed, and only a bounded carry-path correction.
    analytic_prior: bool = False
    # C2 removes pickup timing from the learned action so speed/path are the
    # only coordination mechanisms.  None preserves the original C1 head.
    fixed_pickup_dwell: Optional[float] = None
    # Receding-horizon C2 can continue from measured speed instead of resetting
    # every newly planned profile to 1.5 m/s.
    actual_initial_speed: bool = False
    # Use the first acceleration knot as an immediate per-agent speed command.
    # Zero remains exactly 1.5 m/s; negative values request slowdown now.
    immediate_slowdown: bool = False
    # Interpret the eight raw knots as a direct smooth path-local speed profile
    # instead of cumulatively integrating acceleration.
    direct_speed_profile: bool = False
    # Give only the speed head compact state-derived timing scalars.  This does
    # not choose a yielder; it makes the sign of the two agents' nominal
    # crossing-time difference easy to learn in the toy task.
    arrival_features: bool = False
    # Fixed two-agent toy alternative: flatten the same six state tokens and
    # predict the joint path/speed action with a small MLP.  The public action
    # and path contracts stay identical, so this isolates the backbone choice.
    mlp_backbone: bool = False
    # Append seven deterministic full-speed crossing measurements derived
    # from the same external state.  No GT trajectory or scenario label is
    # introduced; this only saves the toy MLP from approximating route
    # intersection and arc-time arithmetic internally.
    mlp_conflict_features: bool = False
    # Predict one offset for each of the 30 non-anchor path points instead of
    # four Bezier control offsets. Root, box and goal remain hard anchors.
    direct_waypoints: bool = False
    # Emit one joint (waypoint residual, point speed) vector rather than two
    # independent heads. The public result is [B,K,2,33,(x,y,v)].
    joint_point_speed: bool = False
    # Interpret the point values as local speed caps, then derive physically
    # reachable braking/recovery tails. This keeps the learned output general:
    # it selects risky points, not a hand-authored centre/width event.
    physical_speed_caps: bool = False
    # Fixed low-pass decoder passes applied independently to the 15 residual
    # points on each leg. Zero keeps legacy checkpoints bit-identical.
    waypoint_smoothing_passes: int = 0
    # Below a one-metre remaining leg, scale its residual by leg length so the
    # same offsets cannot create sharp turns immediately beside an anchor.
    waypoint_distance_scaling: bool = False
    # Jointly emit one smooth slowdown interval (centre, width, depth) per
    # agent instead of a free speed value at every waypoint.
    slowdown_window: bool = False
    # Metre cap for the complete interval. Zero allows the full path length.
    slowdown_width_max: float = 0.0
    # Predict only slowdown length/depth; deterministically end the interval at
    # the closest geometric point of the two jointly predicted routes.
    conflict_window: bool = False
    # C3b profile: reach the learned minimum speed at the conflict point, then
    # recover over the distance implied by the executor acceleration limit.
    # False preserves the original symmetric pre-conflict cosine exactly.
    conflict_min_at_crossing: bool = False
    # C6 profile: hold the learned minimum speed on a constant plateau ending
    # at the conflict point; derive braking and recovery ramps physically.
    conflict_plateau: bool = False
    # Smooth sigmoid slowdown depth. Unlike the legacy zero-centred clamp,
    # negative raw values retain gradient and can recover from zero slowdown.
    smooth_depth: bool = False
    # C11: predict which physical agent keeps straight/full-speed priority.
    # The categorical choice is sampled only on the first plan and then held
    # for the episode; it is not fed back as an observation token.
    learned_priority: bool = False

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


class JointCoordinator(nn.Module):
    """Predict unnamed learned joint futures from actual shared state only."""

    def __init__(self, config: Optional[CoordinatorConfig] = None):
        super().__init__()
        self.config = config or CoordinatorConfig()
        c = self.config
        if c.candidates <= 0:
            raise ValueError("candidates must be positive")
        if c.joint_point_speed and not (
            c.mlp_backbone and c.direct_waypoints and c.direct_speed_profile
        ):
            raise ValueError(
                "joint_point_speed requires the MLP direct-waypoint/direct-speed modes"
            )
        if c.physical_speed_caps and not c.joint_point_speed:
            raise ValueError("physical_speed_caps requires joint_point_speed")
        if c.slowdown_window and not (
            c.mlp_backbone and c.direct_waypoints and c.direct_speed_profile
        ):
            raise ValueError(
                "slowdown_window requires the MLP direct-waypoint/direct-speed modes"
            )
        if c.slowdown_window and c.joint_point_speed:
            raise ValueError("slowdown_window and joint_point_speed are exclusive")
        if c.conflict_window and not (
            c.mlp_backbone and c.direct_waypoints and c.direct_speed_profile
        ):
            raise ValueError(
                "conflict_window requires the MLP direct-waypoint/direct-speed modes"
            )
        if c.conflict_window and (c.slowdown_window or c.joint_point_speed):
            raise ValueError(
                "conflict_window, slowdown_window and joint_point_speed are exclusive"
            )
        if c.conflict_min_at_crossing and not c.conflict_window:
            raise ValueError("conflict_min_at_crossing requires conflict_window")
        if c.conflict_plateau and not c.conflict_window:
            raise ValueError("conflict_plateau requires conflict_window")
        if c.conflict_plateau and c.conflict_min_at_crossing:
            raise ValueError(
                "conflict_plateau and conflict_min_at_crossing are exclusive"
            )
        if c.smooth_depth and not (c.slowdown_window or c.conflict_window):
            raise ValueError("smooth_depth requires a slowdown window mode")
        if c.slowdown_width_max < 0.0:
            raise ValueError("slowdown_width_max must be zero or positive")
        if c.waypoint_smoothing_passes < 0:
            raise ValueError("waypoint_smoothing_passes must be non-negative")
        if c.waypoint_smoothing_passes and not c.direct_waypoints:
            raise ValueError("waypoint smoothing requires direct_waypoints")
        if c.mlp_backbone:
            if c.candidates != 1:
                raise ValueError("the flat MLP backbone is defined for K=1")
            self.mlp_backbone = nn.Sequential(
                nn.Linear(
                    AGENTS * 3 * c.token_dim
                    + (MLP_CONFLICT_FEATURES if c.mlp_conflict_features else 0),
                    c.feedforward,
                ),
                nn.Tanh(),
                nn.Linear(c.feedforward, c.d_model),
                nn.Tanh(),
            )
            self.input_proj = None
            self.entity_embedding = None
            self.agent_embedding = None
            self.encoder = None
            self.decoder = None
            self.register_parameter("candidate_queries", None)
        else:
            if c.mlp_conflict_features:
                raise ValueError("mlp_conflict_features requires mlp_backbone")
            self.mlp_backbone = None
            self.input_proj = nn.Linear(c.token_dim, c.d_model)
            self.entity_embedding = nn.Embedding(3, c.d_model)
            self.agent_embedding = nn.Embedding(AGENTS, c.d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=c.d_model,
                nhead=c.nhead,
                dim_feedforward=c.feedforward,
                dropout=c.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=c.encoder_layers,
                norm=nn.LayerNorm(c.d_model)
            )
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=c.d_model,
                nhead=c.nhead,
                dim_feedforward=c.feedforward,
                dropout=c.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer, num_layers=c.decoder_layers,
                norm=nn.LayerNorm(c.d_model)
            )
            self.candidate_queries = nn.Parameter(
                torch.empty(c.candidates, c.d_model)
            )
            nn.init.normal_(self.candidate_queries, std=0.02)
        hidden = c.d_model * 2

        def head(output: int, input_dim: Optional[int] = None) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(input_dim or c.d_model, hidden),
                nn.GELU(),
                nn.Linear(hidden, output),
            )

        path_values = AGENTS * (
            WAYPOINT_RESIDUAL_POINTS if c.direct_waypoints else 4
        ) * 2
        if c.joint_point_speed or c.slowdown_window or c.conflict_window:
            speed_values = PATH_POINTS if c.joint_point_speed else (3 if c.slowdown_window else 2)
            self.trajectory_head = head(path_values + AGENTS * speed_values)
            self.path_head = None
        else:
            self.trajectory_head = None
            self.path_head = head(path_values)
        # With timing features, use one shared per-agent speed head driven only
        # by the six sufficient timing scalars.  The Transformer still predicts
        # the path; keeping its large latent out of this tiny decision prevents
        # slot/noise shortcuts and makes the toy speed rule easy to learn.
        self.accel_head = (
            None if (c.joint_point_speed or c.slowdown_window or c.conflict_window) else (
                head(ACCEL_KNOTS, 6)
                if c.arrival_features
                else head(AGENTS * ACCEL_KNOTS)
            )
        )
        self.dwell_head = head(AGENTS)
        self.value_head = head(1)
        self.risk_head = head(3)  # learned diagnostics: HH / BB / human-other-box
        self.priority_head = head(AGENTS) if c.learned_priority else None
        if self.priority_head is not None:
            nn.init.zeros_(self.priority_head[-1].weight)
            nn.init.zeros_(self.priority_head[-1].bias)

        if c.analytic_prior:
            if c.joint_point_speed or c.slowdown_window or c.conflict_window:
                nn.init.zeros_(self.trajectory_head[-1].weight)
                nn.init.zeros_(self.trajectory_head[-1].bias)
                if c.physical_speed_caps:
                    # sigmoid(-4) starts close to the full-speed prior while
                    # retaining gradient toward both stronger and weaker caps.
                    nn.init.constant_(
                        self.trajectory_head[-1].bias[path_values:], -4.0
                    )
                if c.conflict_plateau:
                    # Start with a short, nearly inactive plateau while
                    # preserving gradients for collision-driven activation.
                    nn.init.constant_(
                        self.trajectory_head[-1].bias[path_values::2], -3.0
                    )
                    nn.init.constant_(
                        self.trajectory_head[-1].bias[path_values + 1::2], -3.0
                    )
            else:
                nn.init.zeros_(self.path_head[-1].weight)
                nn.init.zeros_(self.path_head[-1].bias)
                nn.init.zeros_(self.accel_head[-1].weight)
                nn.init.zeros_(self.accel_head[-1].bias)

        self.register_buffer("entity_ids", torch.tensor([0, 1, 2] * AGENTS), persistent=False)
        self.register_buffer("agent_ids", torch.tensor([0, 0, 0, 1, 1, 1]), persistent=False)

    def forward(self, state: Union[CoordinatorState, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not isinstance(state, CoordinatorState):
            state = CoordinatorState.from_mapping(state)
        tokens, frame = state_to_tokens(state)
        if self.config.mlp_backbone:
            flat_state = tokens.flatten(start_dim=1)
            if self.config.mlp_conflict_features:
                with torch.no_grad():
                    zero_residual = torch.zeros(
                        state.batch_size, 1, AGENTS,
                        WAYPOINT_RESIDUAL_POINTS, 2,
                        device=state.device, dtype=state.root_xy.dtype,
                    )
                    nominal_local = build_waypoint_paths(frame, zero_residual)
                    nominal_path = shared_to_world(
                        nominal_local,
                        frame["center"][:, None],
                        frame["angle"][:, None],
                    )
                    nominal_speed = torch.full(
                        nominal_path.shape[:-1], MAX_SPEED,
                        device=state.device, dtype=state.root_xy.dtype,
                    )
                    dwell = torch.full(
                        nominal_path.shape[:3],
                        float(self.config.fixed_pickup_dwell or 0.0),
                        device=state.device, dtype=state.root_xy.dtype,
                    )
                    dwell = torch.where(
                        state.held[:, None] >= 0.5,
                        torch.zeros_like(dwell), dwell,
                    )
                    crossing = crossing_arrival_metrics(
                        nominal_path, nominal_speed, dwell
                    )
                    distance = crossing["crossing_distance"][:, 0]
                    delta = crossing["crossing_arrival_delta"][:, 0]
                    t0 = crossing["crossing_arrival_t0"][:, 0]
                    t1 = crossing["crossing_arrival_t1"][:, 0]
                    i0 = crossing["crossing_index0"][:, 0].to(t0.dtype)
                    i1 = crossing["crossing_index1"][:, 0].to(t0.dtype)
                    conflict_features = torch.stack(
                        (
                            distance / 5.0,
                            delta / 5.0,
                            t0 / 10.0,
                            t1 / 10.0,
                            i0 / 33.0,
                            i1 / 33.0,
                            torch.relu(1.0 - distance),
                        ),
                        dim=-1,
                    )
                flat_state = torch.cat((flat_state, conflict_features), dim=-1)
            decoded = self.mlp_backbone(flat_state)[:, None]
        else:
            memory = self.input_proj(tokens)
            memory = memory + self.entity_embedding(self.entity_ids)[None]
            memory = memory + self.agent_embedding(self.agent_ids)[None]
            memory = self.encoder(memory)
            query = self.candidate_queries[None].expand(
                state.batch_size, -1, -1
            )
            decoded = self.decoder(query, memory)

        point_speed_raw = None
        slowdown_window_raw = None
        if self.config.joint_point_speed or self.config.slowdown_window or self.config.conflict_window:
            joint_raw = self.trajectory_head(decoded)
            path_values = AGENTS * WAYPOINT_RESIDUAL_POINTS * 2
            path_raw = joint_raw[..., :path_values]
            if self.config.joint_point_speed:
                point_speed_raw = joint_raw[..., path_values:].reshape(
                    state.batch_size, self.config.candidates, AGENTS, PATH_POINTS
                )
            elif self.config.slowdown_window:
                slowdown_window_raw = joint_raw[..., path_values:].reshape(
                    state.batch_size, self.config.candidates, AGENTS, 3
                )
            else:
                slowdown_window_raw = joint_raw[..., path_values:].reshape(
                    state.batch_size, self.config.candidates, AGENTS, 2
                )
        else:
            path_raw = self.path_head(decoded)
        waypoint_residual = None
        if self.config.direct_waypoints:
            waypoint_residual = path_raw.reshape(
                state.batch_size, self.config.candidates, AGENTS,
                WAYPOINT_RESIDUAL_POINTS, 2,
            )
            waypoint_residual = (
                self.config.residual_scale * torch.tanh(waypoint_residual)
            )
            approach = torch.where(
                state.held[:, None, :, None, None] >= 0.5,
                torch.zeros_like(waypoint_residual[..., :15, :]),
                waypoint_residual[..., :15, :],
            )
            waypoint_residual = torch.cat(
                (approach, waypoint_residual[..., 15:, :]), dim=-2
            )
            path_local = build_waypoint_paths(
                frame, waypoint_residual,
                self.config.waypoint_smoothing_passes,
                self.config.waypoint_distance_scaling,
            )
            residual = torch.zeros(
                state.batch_size, self.config.candidates, AGENTS, 4, 2,
                device=state.device, dtype=state.root_xy.dtype,
            )
        else:
            residual = path_raw.reshape(
                state.batch_size, self.config.candidates, AGENTS, 4, 2
            )
            residual = self.config.residual_scale * torch.tanh(residual)
            if self.config.analytic_prior:
                first_leg = torch.zeros_like(residual[..., :2, :])
            else:
                first_leg = torch.where(
                    state.held[:, None, :, None, None] >= 0.5,
                    torch.zeros_like(residual[..., :2, :]),
                    residual[..., :2, :],
                )
            residual = torch.cat((first_leg, residual[..., 2:, :]), dim=-2)
            path_local = build_bezier_paths(frame, residual)
        path_world = shared_to_world(
            path_local,
            frame["center"][:, None],
            frame["angle"][:, None],
        )

        speed_context = decoded
        slowdown_window = None
        slowdown_request = None
        if self.config.joint_point_speed:
            accel_raw = point_speed_raw
        elif self.config.slowdown_window:
            accel_raw = slowdown_window_raw
        elif self.config.conflict_window:
            accel_raw = slowdown_window_raw
        elif self.config.arrival_features:
            with torch.no_grad():
                nominal_speed = torch.full(
                    path_world.shape[:-1], MAX_SPEED,
                    device=path_world.device, dtype=path_world.dtype,
                )
                dwell_value = float(self.config.fixed_pickup_dwell or 0.0)
                nominal_dwell = torch.full(
                    path_world.shape[:3], dwell_value,
                    device=path_world.device, dtype=path_world.dtype,
                )
                nominal_dwell = torch.where(
                    state.held[:, None] >= 0.5,
                    torch.zeros_like(nominal_dwell), nominal_dwell,
                )
                crossing = crossing_arrival_metrics(
                    path_world, nominal_speed, nominal_dwell
                )
                nominal_t = torch.stack(
                    (
                        crossing["crossing_arrival_t0"],
                        crossing["crossing_arrival_t1"],
                    ),
                    dim=-1,
                )
                distance = crossing["crossing_distance"]
                own_index = torch.stack(
                    (
                        crossing["crossing_index0"],
                        crossing["crossing_index1"],
                    ),
                    dim=-1,
                ).to(path_world.dtype)
                other_t = nominal_t.flip(-1)
                agent_sign = torch.tensor(
                    [-1.0, 1.0], device=path_world.device,
                    dtype=path_world.dtype,
                ).reshape(1, 1, AGENTS).expand_as(nominal_t)
                timing = torch.stack(
                    (
                        (nominal_t - other_t) / 4.0,
                        distance[..., None].expand_as(nominal_t) / 4.0,
                        nominal_t / 10.0,
                        other_t / 10.0,
                        own_index / 33.0,
                        agent_sign,
                    ),
                    dim=-1,
                )
            speed_context = timing
            accel_raw = self.accel_head(speed_context)
        else:
            accel_raw = self.accel_head(speed_context).reshape(
                state.batch_size, self.config.candidates, AGENTS, ACCEL_KNOTS
            )
        if self.config.joint_point_speed:
            if self.config.physical_speed_caps:
                speed, acceleration, slowdown_request = physical_point_speed_profile(
                    path_world, point_speed_raw
                )
            else:
                speed, acceleration = point_speed_profile(path_world, point_speed_raw)
        elif self.config.slowdown_window:
            speed, acceleration, slowdown_window = slowdown_window_profile(
                path_world, slowdown_window_raw,
                self.config.slowdown_width_max,
                self.config.smooth_depth,
            )
        elif self.config.conflict_window:
            zero_residual = torch.zeros_like(waypoint_residual)
            straight_local = build_waypoint_paths(frame, zero_residual)
            straight_world = shared_to_world(
                straight_local,
                frame["center"][:, None],
                frame["angle"][:, None],
            )
            speed, acceleration, slowdown_window = conflict_window_profile(
                path_world, slowdown_window_raw, straight_world,
                self.config.smooth_depth,
                self.config.conflict_min_at_crossing,
                self.config.conflict_plateau,
            )
        elif self.config.direct_speed_profile:
            speed, acceleration = direct_speed_profile(path_world, accel_raw)
        elif self.config.immediate_slowdown:
            initial_speed = (
                MAX_SPEED
                + (MAX_SPEED - MIN_SPEED) * torch.tanh(accel_raw[..., 0])
            ).clamp(MIN_SPEED, MAX_SPEED)
        elif self.config.analytic_prior and not self.config.actual_initial_speed:
            initial_speed = torch.full(
                (state.batch_size, self.config.candidates, AGENTS),
                MAX_SPEED,
                device=state.device,
                dtype=state.root_xy.dtype,
            )
        else:
            initial_speed = state.root_vel_xy.norm(dim=-1)[:, None].expand(
                -1, self.config.candidates, -1
            )
        if not (
            self.config.direct_speed_profile
            or self.config.joint_point_speed
            or self.config.slowdown_window
            or self.config.conflict_window
        ):
            speed, acceleration = integrate_speed(path_world, accel_raw, initial_speed)
        # Pickup time is part of the future timeline, not a fake zero-speed steer command.
        if self.config.fixed_pickup_dwell is None:
            pickup_dwell = 0.5 + 2.5 * torch.sigmoid(
                self.dwell_head(decoded).reshape(
                    state.batch_size, self.config.candidates, AGENTS
                )
            )
        else:
            pickup_dwell = torch.full(
                (state.batch_size, self.config.candidates, AGENTS),
                float(self.config.fixed_pickup_dwell),
                device=state.device,
                dtype=state.root_xy.dtype,
            )
        pickup_dwell = torch.where(
            state.held[:, None] >= 0.5, torch.zeros_like(pickup_dwell), pickup_dwell
        )
        result = {
            "path_local": path_local,
            "path_world": path_world,
            "speed": speed,
            "acceleration": acceleration,
            "pickup_dwell": pickup_dwell,
            "candidate_value": self.value_head(decoded).squeeze(-1),
            "risk_logits": self.risk_head(decoded),
            "control_residual": residual,
            "accel_knots_raw": accel_raw,
            "trajectory": torch.cat((path_world, speed.unsqueeze(-1)), dim=-1),
        }
        if self.priority_head is not None:
            # C2 is K=1.  Keep the role decision outside the trajectory action:
            # PPO treats it as a separate categorical first-plan action.
            result["priority_logits"] = self.priority_head(decoded).squeeze(1)
        if waypoint_residual is not None:
            result["waypoint_residual"] = waypoint_residual
        if point_speed_raw is not None:
            result["point_speed_raw"] = point_speed_raw
        if slowdown_request is not None:
            result["slowdown_request"] = slowdown_request
        if slowdown_window_raw is not None:
            result["slowdown_window_raw"] = slowdown_window_raw
            result["slowdown_window"] = slowdown_window
        return result
