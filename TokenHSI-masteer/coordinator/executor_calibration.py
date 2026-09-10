"""Pure helpers for frozen-ms18 command-response measurement."""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np
import torch

from .schema import AGENTS, MAX_SPEED


PROFILE_NAMES = (
    "const_0375",
    "const_0750",
    "const_1125",
    "const_1500",
    "a1_step_0750",
    "a1_step_0375",
)
CONSTANT_SPEEDS = (0.375, 0.75, 1.125, 1.5)

# Frozen-ms18 timing model fitted from the two Cross system-ID runs recorded in
# ``runs/coord_measurement/ms18_executor_sysid_v1/calibration.json``.  Keep the
# table small and explicit: it is a loss-side time calibration, not another
# learned model and not an executor command remap.
MS18_TIMING_COMMAND_SPEEDS = CONSTANT_SPEEDS
MS18_TIMING_APPROACH_SPEEDS = (
    0.8124912043412527,
    0.8980326533317566,
    1.1695294419924418,
    1.401019803682963,
)
MS18_TIMING_CARRY_SPEEDS = (
    0.6581523915131887,
    0.837885320186615,
    1.1242645462354024,
    1.3104453484217324,
)
MS18_TIMING_PICKUP_DWELL_S = 0.9000000357627869
MS18_TIMING_PHASE1_REMAINING_DWELL_S = 0.5333333611488342


def requested_speeds(
    profile_ids: torch.Tensor,
    held_age_steps: torch.Tensor,
    *,
    slow_start: int = 30,
    restore_start: int = 120,
) -> torch.Tensor:
    """Return per-agent requested speed for the six fixed system-ID profiles."""
    if profile_ids.ndim != 1:
        raise ValueError("profile_ids must be [E]")
    if held_age_steps.shape != (profile_ids.shape[0], AGENTS):
        raise ValueError(f"held_age_steps must be [E,{AGENTS}]")
    if not 0 <= slow_start < restore_start:
        raise ValueError("expected 0 <= slow_start < restore_start")

    speed = torch.full(
        (profile_ids.shape[0], AGENTS), MAX_SPEED,
        device=profile_ids.device, dtype=torch.float32,
    )
    for profile, value in enumerate(CONSTANT_SPEEDS):
        speed[profile_ids == profile] = value

    in_slow_window = (
        (held_age_steps[:, 1] >= slow_start)
        & (held_age_steps[:, 1] < restore_start)
    )
    profile_075 = (profile_ids == 4) & in_slow_window
    profile_0375 = (profile_ids == 5) & in_slow_window
    speed[profile_075, 1] = 0.75
    speed[profile_0375, 1] = 0.375
    return speed


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if values.ndim != 3:
        raise ValueError("speed arrays must be [T,E,A]")
    if window <= 1:
        return values.astype(np.float64, copy=True)
    out = np.empty_like(values, dtype=np.float64)
    for env in range(values.shape[1]):
        for agent in range(values.shape[2]):
            x = values[:, env, agent].astype(np.float64)
            cumulative = np.cumsum(np.concatenate(([0.0], x)))
            for step in range(values.shape[0]):
                begin = max(0, step - window + 1)
                out[step, env, agent] = (
                    cumulative[step + 1] - cumulative[begin]
                ) / (step + 1 - begin)
    return out


def _stats(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0, "median": float("nan"), "p10": float("nan"), "p90": float("nan")}
    return {
        "n": int(values.size),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def fit_first_order_response(
    sent_speed: np.ndarray,
    actual_speed: np.ndarray,
    mask: np.ndarray,
    dt: float,
) -> Dict[str, float]:
    """Fit v_next=v+dt/tau*(alpha*v_cmd-v) after smoothing gait oscillation."""
    sent = np.asarray(sent_speed, dtype=np.float64)
    actual = np.asarray(actual_speed, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if sent.shape != actual.shape or sent.shape != valid.shape:
        raise ValueError("sent, actual and mask must share [T,E,A]")
    if sent.shape[0] < 2 or dt <= 0.0:
        return {"n": 0, "alpha": float("nan"), "tau_s": float("nan"), "one_step_mae": float("nan")}

    smooth = _rolling_mean(actual, max(1, int(round(0.5 / dt))))
    stable = np.zeros_like(valid)
    stable[1:] = np.abs(sent[1:] - sent[:-1]) < 0.02
    ratio_mask = valid & stable & (sent > 0.3) & (smooth > 0.05)
    ratio = smooth[ratio_mask] / sent[ratio_mask]
    if ratio.size < 8:
        return {"n": int(ratio.size), "alpha": float("nan"), "tau_s": float("nan"), "one_step_mae": float("nan")}
    alpha = float(np.clip(np.median(ratio), 0.1, 2.0))

    pair = valid[:-1] & valid[1:]
    # Use smoothing only for the steady-state gain.  Regressing dynamics on a
    # trailing mean would add the filter's own delay to the executor's tau.
    current = actual[:-1][pair]
    target = alpha * sent[:-1][pair]
    delta = (actual[1:] - actual[:-1])[pair]
    x = target - current
    denom = float(np.dot(x, x))
    beta = float(np.dot(x, delta) / denom) if denom > 1e-9 else float("nan")
    if not np.isfinite(beta) or beta <= 1e-4:
        tau = float("nan")
        mae = float("nan")
    else:
        beta = min(beta, 1.0)
        tau = float(dt / beta)
        mae = float(np.mean(np.abs(delta - beta * x)))
    return {"n": int(pair.sum()), "alpha": alpha, "tau_s": tau, "one_step_mae": mae}


def _step_response(
    requested: np.ndarray,
    sent: np.ndarray,
    actual: np.ndarray,
    held: np.ndarray,
    en_route: np.ndarray,
    profile_ids: np.ndarray,
    profile: int,
    dt: float,
) -> Dict[str, object]:
    """Measure agent-1 slow/restore lag for one carry-triggered step profile."""
    smooth = _rolling_mean(actual, max(1, int(round(0.3 / dt))))
    slow_lag = []
    restore_lag = []
    baseline_speed = []
    reduced_speed = []
    reduced_sent = []
    for env in np.flatnonzero(profile_ids == profile):
        req = requested[:, env, 1]
        valid = held[:, env, 1] & en_route[:, env, 1]
        down = np.flatnonzero(valid & (req < 1.49))
        if down.size == 0:
            continue
        begin = int(down[0])
        up = np.flatnonzero((np.arange(req.size) > begin) & valid & (req >= 1.49))
        end = int(up[0]) if up.size else req.size
        pre = np.arange(max(0, begin - 20), max(0, begin - 5))
        low = np.arange(min(end, begin + 40), max(min(end, begin + 41), end - 10))
        pre = pre[valid[pre]]
        low = low[valid[low]]
        if pre.size < 5 or low.size < 5:
            continue
        high_value = float(np.median(smooth[pre, env, 1]))
        low_value = float(np.median(smooth[low, env, 1]))
        if high_value - low_value < 0.08:
            continue
        midpoint = 0.5 * (high_value + low_value)
        sent_high = float(np.median(sent[pre, env, 1]))
        sent_low = float(np.median(sent[low, env, 1]))
        sent_mid = 0.5 * (sent_high + sent_low)
        search = np.arange(begin, end)
        actual_cross = search[smooth[search, env, 1] <= midpoint]
        sent_cross = search[sent[search, env, 1] <= sent_mid]
        if actual_cross.size and sent_cross.size:
            slow_lag.append((int(actual_cross[0]) - int(sent_cross[0])) * dt)
        if up.size:
            restore = int(up[0])
            search = np.arange(restore, req.size)
            actual_cross = search[smooth[search, env, 1] >= midpoint]
            sent_cross = search[sent[search, env, 1] >= sent_mid]
            if actual_cross.size and sent_cross.size:
                restore_lag.append((int(actual_cross[0]) - int(sent_cross[0])) * dt)
        baseline_speed.append(high_value)
        reduced_speed.append(low_value)
        reduced_sent.append(float(np.median(sent[low, env, 1])))
    return {
        "valid_envs": len(reduced_speed),
        "baseline_actual_mps": _stats(np.asarray(baseline_speed)),
        "reduced_sent_mps": _stats(np.asarray(reduced_sent)),
        "reduced_actual_mps": _stats(np.asarray(reduced_speed)),
        "slow_lag_s": _stats(np.asarray(slow_lag)),
        "restore_lag_s": _stats(np.asarray(restore_lag)),
    }


def summarize_measurement(
    arrays: Mapping[str, np.ndarray], metadata: Mapping[str, object]
) -> Dict[str, object]:
    """Produce compact calibration and collision-alignment statistics."""
    dt = float(metadata["dt"])
    active = np.asarray(arrays["active"], dtype=bool)
    held = np.asarray(arrays["held"], dtype=bool)
    sent = np.asarray(arrays["sent_speed"], dtype=np.float64)
    actual = np.asarray(arrays["displacement_speed"], dtype=np.float64)
    root_box = np.asarray(arrays["root_box_distance"], dtype=np.float64)
    box_goal = np.linalg.norm(
        np.asarray(arrays["box_xyz"], dtype=np.float64)[..., :2]
        - np.asarray(arrays["goal_xy"], dtype=np.float64),
        axis=-1,
    )
    agent_active = np.broadcast_to(active[..., None], held.shape)
    approach = agent_active & (~held) & (root_box > 0.8)
    en_route = box_goal > 0.5
    carry = agent_active & held & en_route

    response = {
        "approach": fit_first_order_response(sent, actual, approach, dt),
        "carry": fit_first_order_response(sent, actual, carry, dt),
    }
    profiles: Dict[str, object] = {}
    profile_ids = np.asarray(arrays["profile_id"], dtype=np.int64)
    smooth_actual = _rolling_mean(actual, max(1, int(round(0.5 / dt))))
    for profile, name in enumerate(PROFILE_NAMES):
        env_mask = profile_ids == profile
        item: Dict[str, object] = {}
        for phase_name, phase_mask in (("approach", approach), ("carry", carry)):
            mask = phase_mask & env_mask[None, :, None]
            ratio_mask = mask & (sent > 0.3) & (smooth_actual > 0.05)
            item[phase_name] = {
                "sent_speed": _stats(sent[mask]),
                "actual_speed": _stats(smooth_actual[mask]),
                "actual_over_sent": _stats(smooth_actual[ratio_mask] / sent[ratio_mask]),
            }
        profiles[name] = item
    step_response = {
        "a1_step_0750": _step_response(
            np.asarray(arrays["requested_speed"]), sent, actual, held,
            en_route, profile_ids, 4, dt,
        ),
        "a1_step_0375": _step_response(
            np.asarray(arrays["requested_speed"]), sent, actual, held,
            en_route, profile_ids, 5, dt,
        ),
    }

    pickup = np.asarray(arrays["pickup_dwell_s"], dtype=np.float64)
    pickup_stats = _stats(pickup[pickup >= 0.0])
    body_min = np.asarray(arrays["body_min_distance"], dtype=np.float64)
    body_episode_min = np.nanmin(np.where(active, body_min, np.nan), axis=0)
    collision = {
        "episodes": int(body_episode_min.size),
        "body_min_m": _stats(body_episode_min),
        "fraction_below_0p3": float(np.mean(body_episode_min < 0.3)),
        "fraction_below_0p4": float(np.mean(body_episode_min < 0.4)),
        "fraction_below_0p5": float(np.mean(body_episode_min < 0.5)),
    }

    predicted = np.asarray(arrays["predicted_crossing_time"], dtype=np.float64)
    observed = np.asarray(arrays["actual_crossing_time"], dtype=np.float64)
    observed_distance = np.asarray(arrays["actual_crossing_distance"], dtype=np.float64)
    constant = profile_ids < len(CONSTANT_SPEEDS)
    valid_cross = constant[:, None] & (observed >= 0.0) & (observed_distance < 0.75)
    arrival_error = observed - predicted
    both = valid_cross.all(axis=1)
    crossing: Dict[str, object] = {
        "valid_agent_arrivals": int(valid_cross.sum()),
        "arrival_error_s": _stats(arrival_error[valid_cross]),
        "arrival_abs_error_s": _stats(np.abs(arrival_error[valid_cross])),
        "valid_pairs": int(both.sum()),
    }
    if both.any():
        predicted_gap = np.abs(predicted[:, 1] - predicted[:, 0])
        observed_gap = np.abs(observed[:, 1] - observed[:, 0])
        crossing["gap_abs_error_s"] = _stats(np.abs(observed_gap[both] - predicted_gap[both]))
    else:
        crossing["gap_abs_error_s"] = _stats(np.asarray([], dtype=np.float64))

    proxy = np.asarray(arrays["root_distance"], dtype=np.float64)
    valid_distance = active & np.isfinite(proxy) & np.isfinite(body_min)
    if valid_distance.sum() >= 2:
        correlation = float(np.corrcoef(proxy[valid_distance], body_min[valid_distance])[0, 1])
    else:
        correlation = float("nan")
    return {
        "schema_version": "tokenhsi-ms18-executor-measure-v1",
        "metadata": dict(metadata),
        "response": response,
        "profiles": profiles,
        "step_response": step_response,
        "pickup_dwell_s": pickup_stats,
        "collision": collision,
        "crossing": crossing,
        "root_body_distance_correlation": correlation,
    }
