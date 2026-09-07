"""Deterministic multi-candidate teacher for V1 Free/Cross scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from tokenhsi.utils import steer_path as sp

from .geometry import sample_coarse_from_dense
from .schema import AGENTS, SPEED_VALUES, PlannerState


CURVE_SEEDS = tuple(range(8))
CANDIDATE_COUNT = 1 + len(CURVE_SEEDS)
ROOT_CLEARANCE = 1.0
TURN_LIMIT_DEG = 35.0


@dataclass
class OracleResult:
    coarse_path: torch.Tensor       # [B,2,33,2]
    speed_class: torch.Tensor       # [B,2,4]
    oracle_valid: torch.Tensor      # [B]
    min_clearance: torch.Tensor     # [B]
    makespan: torch.Tensor          # [B]

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "coarse_path": self.coarse_path,
            "speed_class": self.speed_class,
            "oracle_valid": self.oracle_valid,
            "min_clearance": self.min_clearance,
            "makespan": self.makespan,
        }


def _candidate_paths(state: PlannerState) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return dense/end/box/feasible as [B,A,C,...]."""
    root, box, goal = state.root_xy, state.box_xyz[..., :2], state.goal_xy
    b = state.batch_size
    paths, boxes, ends = [], [], []
    # Straight candidate is exactly the same generator with zero lateral bow.
    specs = [(17_000, 0.0)] + [(seed, 2.2) for seed in CURVE_SEEDS]
    for seed, lat_max in specs:
        flat_root = root.reshape(b * AGENTS, 2)
        flat_box = box.reshape(b * AGENTS, 2)
        flat_goal = goal.reshape(b * AGENTS, 2)
        path, s_box, end_s = sp.gen_full_v2(
            flat_root, flat_box, flat_goal, seed,
            0.0, lat_max, TURN_LIMIT_DEG,
            p_two=0.1, skew=0.8, spread=(0.85, 1.8), lat_frac=0.25,
            with_end=True,
        )
        paths.append(path.reshape(b, AGENTS, sp.V, 2))
        boxes.append(s_box.reshape(b, AGENTS))
        ends.append(end_s.reshape(b, AGENTS))
    dense = torch.stack(paths, dim=2)
    s_box = torch.stack(boxes, dim=2)
    end_s = torch.stack(ends, dim=2)

    flat_dense = dense.reshape(b * AGENTS * CANDIDATE_COUNT, sp.V, 2)
    flat_box_s = s_box.reshape(-1)
    stats = sp.path_stats(flat_dense, flat_box_s)
    turn_ok = stats["turn_1.5m_deg"] <= TURN_LIMIT_DEG + 1e-4
    target = goal[:, :, None].expand(-1, -1, CANDIDATE_COUNT, -1).reshape(-1, 2)
    end_idx = (end_s.reshape(-1) / sp.DS).round().long().clamp(0, sp.V - 1)
    ar = torch.arange(flat_dense.shape[0])
    endpoint_ok = (flat_dense[ar, end_idx] - target).norm(dim=-1) <= 0.01
    buffer_ok = end_s.reshape(-1) < (sp.V - 1) * sp.DS
    feasible = (turn_ok & endpoint_ok & buffer_ok).reshape(b, AGENTS, CANDIDATE_COUNT)
    return dense, s_box, end_s, feasible


def timed_positions(
    path: torch.Tensor,
    end_s: torch.Tensor,
    speed_class: torch.Tensor,
    *,
    dt: float = 0.2,
    steps: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Time-roll out dense paths. Inputs have leading N; outputs [N,T,2], arcs, makespan."""
    speeds_table = torch.tensor(SPEED_VALUES, device=path.device, dtype=path.dtype)
    speeds = speeds_table[speed_class.long()]
    quarter_len = end_s[:, None] / 4.0
    durations = quarter_len / speeds.clamp(min=1e-6)
    start_t = torch.cat((torch.zeros_like(durations[:, :1]), durations.cumsum(dim=1)[:, :-1]), dim=1)
    makespan = durations.sum(dim=1)
    if steps is None:
        steps = max(int(torch.ceil(makespan.max() / dt).item()) + 1, 2)
    t = torch.arange(steps, device=path.device, dtype=path.dtype)[None, :] * dt
    elapsed = (t[:, :, None] - start_t[:, None]).clamp(min=0.0)
    elapsed = torch.minimum(elapsed, durations[:, None])
    arc = (elapsed * speeds[:, None]).sum(dim=-1).clamp(max=end_s[:, None])
    q = (arc / sp.DS).clamp(0, sp.V - 2)
    lo = q.floor().long()
    frac = (q - lo).unsqueeze(-1)
    ar = torch.arange(path.shape[0], device=path.device)[:, None]
    pos = path[ar, lo] + frac * (path[ar, lo + 1] - path[ar, lo])
    return pos, arc, makespan


def timed_clearance(
    path0: torch.Tensor,
    path1: torch.Tensor,
    end0: torch.Tensor,
    end1: torch.Tensor,
    speed0: torch.Tensor,
    speed1: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_time = torch.maximum(
        end0 / torch.tensor(SPEED_VALUES, device=end0.device)[speed0].min(dim=1).values,
        end1 / torch.tensor(SPEED_VALUES, device=end1.device)[speed1].min(dim=1).values,
    )
    steps = max(int(torch.ceil(max_time.max() / 0.2).item()) + 1, 2)
    p0, a0, m0 = timed_positions(path0, end0, speed0, steps=steps)
    p1, a1, m1 = timed_positions(path1, end1, speed1, steps=steps)
    distance = (p0 - p1).norm(dim=-1)
    minimum, at = distance.min(dim=1)
    ar = torch.arange(path0.shape[0], device=path0.device)
    return minimum, a0[ar, at], a1[ar, at], torch.maximum(m0, m1)


def _curvature_score(path: torch.Tensor, s_box: torch.Tensor) -> torch.Tensor:
    stats = sp.path_stats(path, s_box)
    return stats["max_curv_1/m"]


@torch.no_grad()
def solve_oracle(state: PlannerState) -> OracleResult:
    """Select one joint pair, slowing A1 first and A0 only if necessary."""
    state.validate()
    if state.device.type != "cpu":
        raise ValueError("the synthetic V1 oracle is intentionally CPU-only")
    dense, s_box, end_s, feasible = _candidate_paths(state)
    b, _, c = dense.shape[:3]
    pair_count = c * c
    i0 = torch.arange(c).repeat_interleave(c)
    i1 = torch.arange(c).repeat(c)

    p0 = dense[:, 0, i0].reshape(b * pair_count, sp.V, 2)
    p1 = dense[:, 1, i1].reshape(b * pair_count, sp.V, 2)
    e0 = end_s[:, 0, i0].reshape(-1)
    e1 = end_s[:, 1, i1].reshape(-1)
    pair_feasible = (feasible[:, 0, i0] & feasible[:, 1, i1]).reshape(-1)
    speed0 = torch.full((b * pair_count, 4), 3, dtype=torch.long)
    speed1 = speed0.clone()
    clearance, conflict0, conflict1, makespan = timed_clearance(p0, p1, e0, e1, speed0, speed1)
    best_clearance = clearance.clone()
    best_makespan = makespan.clone()
    best_speed0, best_speed1 = speed0.clone(), speed1.clone()

    # Slow only the spatial quarter containing the all-fast closest encounter.
    # Every operation is batched over all 81 joint curve pairs.
    unresolved = pair_feasible & (clearance < ROOT_CLEARANCE)
    for who in (1, 0):
        conflict = conflict1 if who == 1 else conflict0
        ends = e1 if who == 1 else e0
        quarter = torch.floor(4.0 * conflict / ends.clamp(min=1e-6)).long().clamp(0, 3)
        if who == 0 and unresolved.any():
            # "A0 also yields": keep A1 at its strongest attempted slowdown
            # while testing A0, instead of silently restoring A1 to full speed.
            q1 = torch.floor(4.0 * conflict1 / e1.clamp(min=1e-6)).long().clamp(0, 3)
            idx = unresolved.nonzero(as_tuple=False).squeeze(-1)
            speed1[idx, q1[idx]] = 0
        for slow_class in (2, 1, 0):
            if not unresolved.any():
                break
            trial0, trial1 = speed0.clone(), speed1.clone()
            idx = unresolved.nonzero(as_tuple=False).squeeze(-1)
            if who == 1:
                trial1[idx, quarter[idx]] = slow_class
            else:
                trial0[idx, quarter[idx]] = slow_class
            trial_clear, _, _, trial_span = timed_clearance(p0, p1, e0, e1, trial0, trial1)
            improved = unresolved & (trial_clear > best_clearance)
            best_clearance = torch.where(improved, trial_clear, best_clearance)
            best_makespan = torch.where(improved, trial_span, best_makespan)
            best_speed0[improved] = trial0[improved]
            best_speed1[improved] = trial1[improved]
            accepted = unresolved & (trial_clear >= ROOT_CLEARANCE)
            speed0[accepted] = trial0[accepted]
            speed1[accepted] = trial1[accepted]
            clearance = torch.where(accepted, trial_clear, clearance)
            makespan = torch.where(accepted, trial_span, makespan)
            unresolved &= ~accepted

    valid = pair_feasible & (clearance >= ROOT_CLEARANCE)
    failed = ~valid
    clearance = torch.where(failed, best_clearance, clearance)
    makespan = torch.where(failed, best_makespan, makespan)
    speed0[failed] = best_speed0[failed]
    speed1[failed] = best_speed1[failed]
    length = e0 + e1
    curve = _curvature_score(p0, s_box[:, 0, i0].reshape(-1))
    curve += _curvature_score(p1, s_box[:, 1, i1].reshape(-1))
    # Lexicographic ranking with safe scale separation for this bounded domain.
    valid_score = makespan * 1e6 + length * 1e3 + curve
    valid_score = valid_score.masked_fill(~valid, float("inf"))
    valid_matrix = valid.reshape(b, pair_count)
    best_valid = valid_score.reshape(b, pair_count).argmin(dim=1)
    # If no solution clears 1m, keep the pair with largest clearance as an
    # explicit invalid label. It is serialized for diagnostics but not trained.
    invalid_clearance = clearance.masked_fill(~pair_feasible, float("-inf"))
    best_invalid = invalid_clearance.reshape(b, pair_count).argmax(dim=1)
    any_valid = valid_matrix.any(dim=1)
    best = torch.where(any_valid, best_valid, best_invalid)
    flat = torch.arange(b) * pair_count + best

    chosen_i0, chosen_i1 = i0[best], i1[best]
    chosen_dense = torch.stack(
        (dense[torch.arange(b), 0, chosen_i0], dense[torch.arange(b), 1, chosen_i1]), dim=1
    )
    chosen_sbox = torch.stack(
        (s_box[torch.arange(b), 0, chosen_i0], s_box[torch.arange(b), 1, chosen_i1]), dim=1
    )
    chosen_end = torch.stack(
        (end_s[torch.arange(b), 0, chosen_i0], end_s[torch.arange(b), 1, chosen_i1]), dim=1
    )
    coarse = sample_coarse_from_dense(
        chosen_dense.reshape(b * AGENTS, sp.V, 2),
        chosen_sbox.reshape(-1),
        chosen_end.reshape(-1),
        state.box_xyz[..., :2].reshape(-1, 2),
        state.goal_xy.reshape(-1, 2),
    ).reshape(b, AGENTS, 33, 2)
    chosen_speed = torch.stack((speed0[flat], speed1[flat]), dim=1)
    return OracleResult(
        coarse_path=coarse,
        speed_class=chosen_speed,
        oracle_valid=any_valid,
        min_clearance=clearance[flat],
        makespan=makespan[flat],
    )
